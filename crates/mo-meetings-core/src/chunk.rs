//! Speech chunks for ASR, derived from diarization: at most `max_len` seconds, cut in pauses between turns.
//!
//! Whisper works in 30 s windows. Cutting at turn boundaries (instead of fixed 30 s windows) keeps words whole
//! and lets every chunk get the glossary prompt, which whisper otherwise uses only in the first window.

use crate::{Chunk, SAMPLE_RATE, Turn};

#[derive(Clone, Debug)]
pub struct ChunkRules {
    /// Maximum chunk length (s). Whisper's window is 30 s.
    pub max_len: f64,
    /// Start a new chunk when the pause between turns is longer than this (s); long silences invite hallucinations.
    pub max_gap: f64,
    /// Padding added around speech, taken only from the pause (never more than half of it) (s).
    pub pad: f64,
    /// A single turn longer than `max_len` is cut at the quietest 100 ms frame in the last `search` seconds.
    pub search: f64,
}

impl Default for ChunkRules {
    fn default() -> Self {
        Self { max_len: 30.0, max_gap: 3.0, pad: 0.2, search: 10.0 }
    }
}

/// Build chunks from (any, possibly overlapping) turns. `audio` (16 kHz) is used only to find quiet cut points
/// inside very long turns and to clamp the end.
pub fn speech_chunks(turns: &[Turn], audio: &[f32], rules: &ChunkRules) -> Vec<Chunk> {
    let total = audio.len() as f64 / SAMPLE_RATE as f64;
    // Merge turns into contiguous speech regions (union over speakers).
    let mut spans: Vec<(f64, f64)> = turns.iter().map(|t| (t.start.max(0.0), t.end.min(total))).collect();
    spans.retain(|(s, e)| e > s);
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut regions: Vec<(f64, f64)> = Vec::new();
    for (s, e) in spans {
        match regions.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => regions.push((s, e)),
        }
    }
    // Padding must not push a chunk over whisper's window.
    let limit = rules.max_len - 2.0 * rules.pad;
    // Split regions longer than the limit at quiet points.
    let mut pieces: Vec<(f64, f64)> = Vec::new();
    for (s, e) in regions {
        let mut start = s;
        while e - start > limit {
            let lo = start + (limit - rules.search).max(limit / 2.0);
            let cut = quietest(audio, lo, start + limit);
            pieces.push((start, cut));
            start = cut;
        }
        pieces.push((start, e));
    }
    // Greedily join pieces while the chunk stays within max_len and gaps are short.
    let mut chunks: Vec<(f64, f64)> = Vec::new();
    for (s, e) in pieces {
        match chunks.last_mut() {
            Some(last) if s - last.1 <= rules.max_gap && e - last.0 <= limit => last.1 = e,
            _ => chunks.push((s, e)),
        }
    }
    // Pad into the surrounding pauses.
    let n = chunks.len();
    (0..n)
        .map(|i| {
            let (s, e) = chunks[i];
            let before = if i == 0 { s } else { (s - chunks[i - 1].1) / 2.0 };
            let after = if i + 1 == n { total - e } else { (chunks[i + 1].0 - e) / 2.0 };
            Chunk { start: (s - rules.pad.min(before)).max(0.0), end: (e + rules.pad.min(after)).min(total) }
        })
        .collect()
}

/// Time (s) of the 100 ms frame with the lowest energy in `[from, to]`.
fn quietest(audio: &[f32], from: f64, to: f64) -> f64 {
    let frame = (SAMPLE_RATE / 10) as usize;
    let a = ((from * SAMPLE_RATE as f64) as usize).min(audio.len());
    let b = ((to * SAMPLE_RATE as f64) as usize).min(audio.len());
    if b <= a + frame {
        return to;
    }
    let mut best = (f32::INFINITY, b);
    let mut i = a;
    while i + frame <= b {
        let energy: f32 = audio[i..i + frame].iter().map(|x| x * x).sum();
        if energy < best.0 {
            best = (energy, i + frame / 2);
        }
        i += frame;
    }
    best.1 as f64 / SAMPLE_RATE as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(seconds: f64) -> Vec<f32> {
        vec![0.0; (seconds * SAMPLE_RATE as f64) as usize]
    }

    #[test]
    fn joins_short_turns_and_pads() {
        let audio = silence(20.0);
        let turns = [Turn::new(1.0, 3.0, 0), Turn::new(3.5, 6.0, 1), Turn::new(5.0, 7.0, 0)];
        let c = speech_chunks(&turns, &audio, &ChunkRules::default());
        assert_eq!(c, vec![Chunk { start: 0.8, end: 7.2 }]);
    }

    #[test]
    fn splits_at_max_len_between_turns() {
        let audio = silence(70.0);
        let turns: Vec<Turn> = (0..12).map(|i| Turn::new(i as f64 * 5.0, i as f64 * 5.0 + 4.0, i % 2)).collect();
        let c = speech_chunks(&turns, &audio, &ChunkRules::default());
        assert!(c.len() >= 2);
        for ch in &c {
            assert!(ch.end - ch.start <= 30.0 + 1e-9, "{ch:?}");
        }
        // Cuts fall in pauses: no turn is split between two chunks.
        for t in &turns {
            assert!(c.iter().any(|ch| ch.start <= t.start && t.end <= ch.end), "{t:?} split");
        }
    }

    #[test]
    fn long_gap_starts_new_chunk() {
        let audio = silence(30.0);
        let c = speech_chunks(&[Turn::new(0.0, 2.0, 0), Turn::new(10.0, 12.0, 0)], &audio, &ChunkRules::default());
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn long_turn_is_cut_at_quiet_point() {
        let mut audio: Vec<f32> = (0..(70 * SAMPLE_RATE)).map(|i| ((i as f32) * 0.05).sin() * 0.5).collect();
        // quiet 300 ms at 25 s
        let q = (25.0 * SAMPLE_RATE as f64) as usize;
        audio[q..q + 4800].iter_mut().for_each(|x| *x = 0.0);
        let c = speech_chunks(&[Turn::new(0.0, 65.0, 0)], &audio, &ChunkRules::default());
        assert!(c.len() >= 3, "{c:?}");
        assert!((c[0].end - 25.05).abs() < 0.06, "{c:?}");
        for ch in &c {
            assert!(ch.end - ch.start <= 30.0 + 1e-9, "{ch:?}");
        }
    }

    #[test]
    fn empty() {
        assert!(speech_chunks(&[], &silence(5.0), &ChunkRules::default()).is_empty());
    }
}
