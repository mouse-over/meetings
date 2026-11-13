//! Post-processing of clustering output over per-turn speaker embeddings.
//!
//! Fast clustering with a threshold that works for short recordings over-splits long ones into many small
//! clusters, and sometimes splits one speaker into two big ones. Two steps fix that (validated on 5 recordings,
//! see `spikes/sherpa-diarization/RESULTS.md`):
//!
//! 1. clusters with less than `min_share` of total speech are reassigned turn by turn to the nearest big cluster
//!    (cosine similarity of the turn embedding to the cluster centroid; turns too short to embed go to the nearest
//!    big turn in time);
//! 2. the pair of clusters with the most similar centroids is merged while their cosine similarity is above
//!    `similarity` (parts of one speaker had 0.78–0.97, different speakers at most 0.53).

use crate::Turn;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct MergeParams {
    /// Clusters with less than this share of total speech are reassigned (0.01 = 1 %).
    pub min_share: f64,
    /// Merge clusters whose centroids have cosine similarity above this.
    pub similarity: f32,
    /// Optional cap on the number of clusters (keeps merging the most similar pair).
    pub max_speakers: Option<usize>,
}

impl Default for MergeParams {
    fn default() -> Self {
        Self { min_share: 0.01, similarity: 0.7, max_speakers: None }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MergeReport {
    /// Clusters returned by the clustering step.
    pub clusters_raw: usize,
    /// After reassigning small clusters.
    pub clusters_after_small: usize,
    /// After merging similar clusters.
    pub clusters_final: usize,
    /// Cosine similarities of the merged pairs, in merge order.
    pub merged_similarities: Vec<f32>,
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn distinct_speakers(turns: &[Turn]) -> Vec<usize> {
    let mut s: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
    s.sort_unstable();
    s.dedup();
    s
}

fn speech(turns: &[Turn], speaker: usize) -> f64 {
    turns.iter().filter(|t| t.speaker == speaker).map(Turn::duration).sum()
}

/// Duration-weighted mean of the (normalized) turn embeddings of `speaker`, normalized.
fn centroid(turns: &[Turn], emb: &[Option<Vec<f32>>], speaker: usize) -> Option<Vec<f32>> {
    let mut acc: Option<Vec<f32>> = None;
    for (t, e) in turns.iter().zip(emb) {
        if t.speaker != speaker {
            continue;
        }
        if let Some(e) = e {
            let w = t.duration() as f32;
            let a = acc.get_or_insert_with(|| vec![0.0; e.len()]);
            a.iter_mut().zip(e).for_each(|(a, x)| *a += w * x);
        }
    }
    acc.map(|mut a| {
        normalize(&mut a);
        a
    })
}

/// Run both merge steps. `emb[i]` is the embedding of `turns[i]` (`None` if too short to embed).
pub fn merge_clusters(turns: &[Turn], emb: &[Option<Vec<f32>>], p: &MergeParams) -> (Vec<Turn>, MergeReport) {
    assert_eq!(turns.len(), emb.len());
    let emb: Vec<Option<Vec<f32>>> = emb
        .iter()
        .map(|e| {
            e.clone().map(|mut v| {
                normalize(&mut v);
                v
            })
        })
        .collect();
    let mut out = turns.to_vec();
    let mut report = MergeReport { clusters_raw: distinct_speakers(turns).len(), ..Default::default() };

    // 1. Reassign small clusters.
    let total: f64 = turns.iter().map(Turn::duration).sum();
    let speakers = distinct_speakers(turns);
    let big: Vec<usize> = speakers.iter().copied().filter(|&s| speech(turns, s) >= p.min_share * total).collect();
    let centroids: Vec<(usize, Vec<f32>)> =
        big.iter().filter_map(|&s| centroid(turns, &emb, s).map(|c| (s, c))).collect();
    if !centroids.is_empty() && big.len() < speakers.len() {
        for (i, t) in turns.iter().enumerate() {
            if big.contains(&t.speaker) {
                continue;
            }
            out[i].speaker = match &emb[i] {
                Some(e) => centroids.iter().max_by(|a, b| dot(e, &a.1).total_cmp(&dot(e, &b.1))).expect("non-empty").0,
                None => {
                    let mid = (t.start + t.end) / 2.0;
                    turns
                        .iter()
                        .filter(|o| big.contains(&o.speaker))
                        .min_by(|a, b| {
                            let d = |o: &Turn| (o.start - mid).abs().min((o.end - mid).abs());
                            d(a).total_cmp(&d(b))
                        })
                        .expect("big clusters have turns")
                        .speaker
                }
            };
        }
    }
    report.clusters_after_small = distinct_speakers(&out).len();

    // 2. Merge similar clusters (and, with a cap, the most similar ones until the cap is met).
    loop {
        let speakers = distinct_speakers(&out);
        let cents: Vec<(usize, Vec<f32>)> =
            speakers.iter().filter_map(|&s| centroid(&out, &emb, s).map(|c| (s, c))).collect();
        let mut best: Option<(f32, usize, usize)> = None;
        for i in 0..cents.len() {
            for j in i + 1..cents.len() {
                let sim = dot(&cents[i].1, &cents[j].1);
                if best.is_none_or(|b| sim > b.0) {
                    best = Some((sim, cents[i].0, cents[j].0));
                }
            }
        }
        let over_cap = p.max_speakers.is_some_and(|m| speakers.len() > m.max(1));
        let Some((sim, a, b)) = best else { break };
        if sim <= p.similarity && !over_cap {
            break;
        }
        let (keep, drop) = if speech(&out, a) >= speech(&out, b) { (a, b) } else { (b, a) };
        out.iter_mut().filter(|t| t.speaker == drop).for_each(|t| t.speaker = keep);
        report.merged_similarities.push(sim);
    }
    report.clusters_final = distinct_speakers(&out).len();
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vector along `axis` with a small tilt towards `tilt_axis`.
    fn v(axis: usize, tilt_axis: usize, tilt: f32) -> Option<Vec<f32>> {
        let mut e = vec![0.0; 4];
        e[axis] = 1.0;
        e[tilt_axis] += tilt;
        Some(e)
    }

    #[test]
    fn small_cluster_goes_to_most_similar_big_cluster() {
        // speakers 0 and 1: 50 s each; speaker 2: 0.6 s (< 1 %), sounds like speaker 1
        let turns = vec![Turn::new(0.0, 50.0, 0), Turn::new(50.0, 100.0, 1), Turn::new(100.0, 100.6, 2)];
        let emb = vec![v(0, 0, 0.0), v(1, 1, 0.0), v(1, 2, 0.3)];
        let p = MergeParams { similarity: 0.99, ..Default::default() };
        let (out, r) = merge_clusters(&turns, &emb, &p);
        assert_eq!(out[2].speaker, 1);
        assert_eq!((r.clusters_raw, r.clusters_after_small, r.clusters_final), (3, 2, 2));
    }

    #[test]
    fn unembeddable_small_turn_goes_to_nearest_big_turn_in_time() {
        let turns = vec![Turn::new(0.0, 50.0, 0), Turn::new(60.0, 110.0, 1), Turn::new(58.0, 58.3, 2)];
        let emb = vec![v(0, 0, 0.0), v(1, 1, 0.0), None];
        let (out, _) = merge_clusters(&turns, &emb, &MergeParams::default());
        assert_eq!(out[2].speaker, 1);
    }

    #[test]
    fn similar_big_clusters_are_merged_into_larger() {
        // speaker 1 and 2 are the same person (cos ~0.95), speaker 0 is different
        let turns = vec![
            Turn::new(0.0, 40.0, 0),
            Turn::new(40.0, 70.0, 1),
            Turn::new(70.0, 80.0, 2),
            Turn::new(80.0, 120.0, 0),
        ];
        let emb = vec![v(0, 0, 0.0), v(1, 2, 0.3), v(2, 1, 3.0), v(0, 3, 0.1)];
        let (out, r) = merge_clusters(&turns, &emb, &MergeParams::default());
        assert_eq!(out.iter().map(|t| t.speaker).collect::<Vec<_>>(), [0, 1, 1, 0]);
        assert_eq!(r.clusters_final, 2);
        assert_eq!(r.merged_similarities.len(), 1);
        assert!(r.merged_similarities[0] > 0.9);
    }

    #[test]
    fn different_speakers_stay_apart() {
        // cos(a, b) = 0.5 < 0.7
        let turns = vec![Turn::new(0.0, 10.0, 0), Turn::new(10.0, 20.0, 1)];
        let emb = vec![Some(vec![1.0, 0.0]), Some(vec![0.5, 0.866])];
        let (out, r) = merge_clusters(&turns, &emb, &MergeParams::default());
        assert_eq!(out, turns);
        assert!(r.merged_similarities.is_empty());
    }

    #[test]
    fn merging_repeats_until_no_similar_pair() {
        // three parts of one speaker
        let turns = vec![Turn::new(0.0, 10.0, 0), Turn::new(10.0, 20.0, 1), Turn::new(20.0, 30.0, 2)];
        let emb = vec![v(0, 1, 0.2), v(0, 2, 0.3), v(0, 3, 0.4)];
        let (out, r) = merge_clusters(&turns, &emb, &MergeParams::default());
        assert!(out.iter().all(|t| t.speaker == out[0].speaker));
        assert_eq!(r.clusters_final, 1);
        assert_eq!(r.merged_similarities.len(), 2);
    }

    #[test]
    fn centroid_is_duration_weighted() {
        let turns = vec![Turn::new(0.0, 9.0, 0), Turn::new(9.0, 10.0, 0)];
        let emb = vec![Some(vec![1.0, 0.0]), Some(vec![0.0, 1.0])];
        let c = centroid(&turns, &emb, 0).unwrap();
        assert!(c[0] > 0.99 && c[1] < 0.12, "{c:?}");
    }

    #[test]
    fn cap_forces_merges_of_most_similar() {
        let turns = vec![Turn::new(0.0, 10.0, 0), Turn::new(10.0, 20.0, 1), Turn::new(20.0, 30.0, 2)];
        let emb = vec![Some(vec![1.0, 0.0, 0.0]), Some(vec![0.6, 0.8, 0.0]), Some(vec![0.0, 0.0, 1.0])];
        let p = MergeParams { max_speakers: Some(2), ..Default::default() };
        let (out, r) = merge_clusters(&turns, &emb, &p);
        assert_eq!(out[0].speaker, out[1].speaker);
        assert_ne!(out[0].speaker, out[2].speaker);
        assert_eq!(r.clusters_final, 2);
    }

    #[test]
    fn empty_input() {
        let (out, r) = merge_clusters(&[], &[], &MergeParams::default());
        assert!(out.is_empty());
        assert_eq!(r.clusters_final, 0);
    }
}
