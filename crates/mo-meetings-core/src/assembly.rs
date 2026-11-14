//! Words + diarization → speaker-labelled cues.

use crate::{Cue, Turn, Word};

/// Rules for splitting words into cues.
#[derive(Clone, Debug)]
pub struct CueRules {
    /// Split when the gap between words is longer than this (s).
    pub max_pause: f64,
    /// Split when the cue would be longer than this (s, measured from the cue start to the next word start).
    pub max_duration: f64,
    /// Split after a sentence end (`.`, `?`, `!`) once the cue is longer than this (s).
    pub sentence_min_duration: f64,
}

impl Default for CueRules {
    fn default() -> Self {
        Self { max_pause: 1.0, max_duration: 7.0, sentence_min_duration: 3.0 }
    }
}

/// Make turns non-overlapping: an overlap is cut at the later start; a turn nested inside another one splits it.
pub fn exclusive_turns(turns: &[Turn]) -> Vec<Turn> {
    let mut sorted = turns.to_vec();
    sorted.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)));
    let mut out: Vec<Turn> = Vec::with_capacity(sorted.len());
    for t in sorted {
        if let Some(last) = out.last_mut()
            && t.start < last.end
        {
            if t.end <= last.end {
                let (end, speaker) = (last.end, last.speaker);
                last.end = t.start;
                let rest = Turn::new(t.end, end, speaker);
                out.push(t);
                out.push(rest);
                continue;
            }
            last.end = t.start;
        }
        out.push(t);
    }
    out.retain(|t| t.duration() > 1e-3);
    out
}

/// Speaker of the span `[start, end]`: the turn speaker with the largest overlap, else the nearest turn.
pub fn speaker_for(turns: &[Turn], start: f64, end: f64) -> Option<usize> {
    let mut overlap: Vec<(usize, f64)> = Vec::new();
    for t in turns {
        let o = end.min(t.end) - start.max(t.start);
        if o > 0.0 {
            match overlap.iter_mut().find(|(s, _)| *s == t.speaker) {
                Some((_, v)) => *v += o,
                None => overlap.push((t.speaker, o)),
            }
        }
    }
    // First maximum wins (stable, like Counter.most_common in the reference implementation).
    let mut best: Option<(usize, f64)> = None;
    for (s, o) in overlap {
        if best.is_none_or(|(_, b)| o > b) {
            best = Some((s, o));
        }
    }
    if let Some((s, _)) = best {
        return Some(s);
    }
    let mid = (start + end) / 2.0;
    turns.iter().min_by(|a, b| distance(a, mid).total_cmp(&distance(b, mid))).map(|t| t.speaker)
}

fn distance(t: &Turn, x: f64) -> f64 {
    (t.start - x).abs().min((t.end - x).abs())
}

/// Assign a speaker to every word (by overlap with `turns`, which should be [`exclusive_turns`]).
pub fn assign_speakers(words: &mut [Word], turns: &[Turn]) {
    for w in words {
        w.speaker = speaker_for(turns, w.start, w.end);
    }
}

/// Map internal speaker ids to display numbers 1..N by first appearance in `words`.
pub fn speaker_numbers(words: &[Word]) -> Vec<(usize, usize)> {
    let mut order: Vec<(usize, usize)> = Vec::new();
    for w in words {
        if let Some(s) = w.speaker
            && !order.iter().any(|(id, _)| *id == s)
        {
            order.push((s, order.len() + 1));
        }
    }
    order
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end().ends_with(['.', '?', '!'])
}

/// Group words into cues. Words must be sorted by time and have speakers assigned.
pub fn build_cues(words: &[Word], rules: &CueRules) -> Vec<Cue> {
    let numbers = speaker_numbers(words);
    let number =
        |s: Option<usize>| s.and_then(|s| numbers.iter().find(|(id, _)| *id == s).map(|(_, n)| *n)).unwrap_or(0);
    let mut cues = Vec::new();
    let mut cur: Vec<&Word> = Vec::new();
    let flush = |cur: &mut Vec<&Word>, cues: &mut Vec<Cue>| {
        if cur.is_empty() {
            return;
        }
        let text: String = cur.iter().map(|w| w.text.as_str()).collect();
        let text = text.trim().to_string();
        if !text.is_empty() {
            cues.push(Cue { start: cur[0].start, end: cur[cur.len() - 1].end, speaker: number(cur[0].speaker), text });
        }
        cur.clear();
    };
    for w in words {
        if let (Some(first), Some(last)) = (cur.first(), cur.last()) {
            let split = w.speaker != last.speaker
                || w.start - last.end > rules.max_pause
                || w.start - first.start > rules.max_duration
                || (ends_sentence(&last.text) && last.end - first.start > rules.sentence_min_duration);
            if split {
                flush(&mut cur, &mut cues);
            }
        }
        cur.push(w);
    }
    flush(&mut cur, &mut cues);
    cues
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(text: &str, start: f64, end: f64, speaker: usize) -> Word {
        Word { speaker: Some(speaker), ..Word::new(text, start, end) }
    }

    #[test]
    fn exclusive_cuts_overlap_at_later_start() {
        let t = exclusive_turns(&[Turn::new(0.0, 5.0, 0), Turn::new(4.0, 8.0, 1)]);
        assert_eq!(t, vec![Turn::new(0.0, 4.0, 0), Turn::new(4.0, 8.0, 1)]);
    }

    #[test]
    fn exclusive_splits_nested_turn() {
        let t = exclusive_turns(&[Turn::new(0.0, 10.0, 0), Turn::new(3.0, 4.0, 1)]);
        assert_eq!(t, vec![Turn::new(0.0, 3.0, 0), Turn::new(3.0, 4.0, 1), Turn::new(4.0, 10.0, 0)]);
    }

    #[test]
    fn exclusive_drops_empty_and_sorts() {
        let t = exclusive_turns(&[Turn::new(5.0, 6.0, 1), Turn::new(0.0, 5.0, 0), Turn::new(5.0, 5.0, 2)]);
        assert_eq!(t, vec![Turn::new(0.0, 5.0, 0), Turn::new(5.0, 6.0, 1)]);
    }

    #[test]
    fn speaker_by_largest_overlap() {
        let turns = [Turn::new(0.0, 1.2, 0), Turn::new(1.2, 3.0, 1)];
        assert_eq!(speaker_for(&turns, 1.0, 1.5), Some(1));
        assert_eq!(speaker_for(&turns, 0.9, 1.3), Some(0));
    }

    #[test]
    fn overlap_sums_over_turns_of_same_speaker() {
        let turns = [Turn::new(0.0, 1.0, 0), Turn::new(1.0, 1.6, 1), Turn::new(1.6, 2.2, 0)];
        // speaker 0: 0.5 + 0.4, speaker 1: 0.6
        assert_eq!(speaker_for(&turns, 0.5, 2.0), Some(0));
    }

    #[test]
    fn speaker_falls_back_to_nearest_turn() {
        let turns = [Turn::new(0.0, 1.0, 0), Turn::new(5.0, 6.0, 1)];
        assert_eq!(speaker_for(&turns, 3.5, 3.9), Some(1));
        assert_eq!(speaker_for(&turns, 1.5, 2.0), Some(0));
        assert_eq!(speaker_for(&[], 1.0, 2.0), None);
    }

    #[test]
    fn numbers_by_first_appearance() {
        let words = [w(" a", 0.0, 0.1, 7), w(" b", 0.2, 0.3, 3), w(" c", 0.4, 0.5, 7)];
        assert_eq!(speaker_numbers(&words), vec![(7, 1), (3, 2)]);
    }

    #[test]
    fn cue_splits_on_speaker_change() {
        let words = [w(" Ahoj", 0.0, 0.3, 5), w(" jak", 0.4, 0.6, 5), w(" Dobře", 0.7, 1.0, 2)];
        let cues = build_cues(&words, &CueRules::default());
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0], Cue { start: 0.0, end: 0.6, speaker: 1, text: "Ahoj jak".into() });
        assert_eq!(cues[1], Cue { start: 0.7, end: 1.0, speaker: 2, text: "Dobře".into() });
    }

    #[test]
    fn cue_splits_on_pause_over_one_second() {
        let words = [w(" a", 0.0, 0.5, 0), w(" b", 1.5, 1.8, 0), w(" c", 2.81, 3.0, 0)];
        let cues = build_cues(&words, &CueRules::default());
        assert_eq!(cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["a b", "c"]);
    }

    #[test]
    fn cue_splits_after_seven_seconds() {
        let words: Vec<Word> = (0..20).map(|i| w(" slovo", i as f64 * 0.5, i as f64 * 0.5 + 0.4, 0)).collect();
        let cues = build_cues(&words, &CueRules::default());
        assert!(cues.len() >= 2);
        for c in &cues {
            assert!(c.end - c.start <= 7.5, "{c:?}");
        }
        assert_eq!(cues[0].text.split(' ').count(), 15); // starts 0.0 .. 7.0
    }

    #[test]
    fn cue_splits_on_sentence_end_after_three_seconds() {
        let words = [
            w(" Krátká.", 0.0, 1.0, 0),
            w(" Pokračuje", 1.1, 2.0, 0),
            w(" věta", 2.1, 3.2, 0),
            w(" zde.", 3.3, 3.5, 0),
            w(" Nová", 3.6, 4.0, 0),
        ];
        let cues = build_cues(&words, &CueRules::default());
        // "Krátká." ends a sentence but the cue is only 1 s long, so no split there.
        assert_eq!(cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["Krátká. Pokračuje věta zde.", "Nová"]);
    }

    #[test]
    fn assign_then_cue() {
        let turns = exclusive_turns(&[Turn::new(0.0, 2.0, 4), Turn::new(1.9, 4.0, 9)]);
        let mut words = vec![Word::new(" Dobrý", 0.1, 0.5), Word::new(" den.", 0.6, 1.0), Word::new(" Ahoj", 2.2, 2.6)];
        assign_speakers(&mut words, &turns);
        let cues = build_cues(&words, &CueRules::default());
        assert_eq!(cues.len(), 2);
        assert_eq!((cues[0].speaker, cues[1].speaker), (1, 2));
        assert_eq!(cues[0].text, "Dobrý den.");
    }
}
