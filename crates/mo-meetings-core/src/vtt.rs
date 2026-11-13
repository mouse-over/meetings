//! Teams-shaped WebVTT writer: `WEBVTT`, numeric cue ids, `<v Speaker N>text</v>`, HTML entities.

use crate::Cue;
use std::fmt::Write;

/// Format seconds as `HH:MM:SS.mmm` (rounded to milliseconds).
pub fn timestamp(seconds: f64) -> String {
    let ms = (seconds.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}.{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

/// Escape text for a VTT cue payload (`&`, `<`, `>`), like Python's `html.escape(quote=False)`.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// Render cues as a WebVTT document.
pub fn render(cues: &[Cue]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for (i, cue) in cues.iter().enumerate() {
        let _ = write!(
            out,
            "{}\n{} --> {}\n<v Speaker {}>{}</v>\n\n",
            i + 1,
            timestamp(cue.start),
            timestamp(cue.end),
            cue.speaker,
            escape(&cue.text)
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps() {
        assert_eq!(timestamp(0.0), "00:00:00.000");
        assert_eq!(timestamp(3725.0456), "01:02:05.046");
        assert_eq!(timestamp(59.9996), "00:01:00.000");
        assert_eq!(timestamp(-1.0), "00:00:00.000");
    }

    #[test]
    fn escapes_html_but_not_quotes() {
        assert_eq!(escape("a < b && c > d"), "a &lt; b &amp;&amp; c &gt; d");
        assert_eq!(escape("\"it's\""), "\"it's\"");
        assert_eq!(escape("Žluťoučký kůň"), "Žluťoučký kůň");
    }

    #[test]
    fn renders_teams_shape() {
        let cues = vec![
            Cue { start: 1.0, end: 2.5, speaker: 1, text: "Ahoj <všichni>.".into() },
            Cue { start: 3.0, end: 4.0, speaker: 2, text: "R&D".into() },
        ];
        assert_eq!(
            render(&cues),
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.500\n<v Speaker 1>Ahoj &lt;všichni&gt;.</v>\n\n\
             2\n00:00:03.000 --> 00:00:04.000\n<v Speaker 2>R&amp;D</v>\n\n"
        );
    }

    #[test]
    fn renders_empty() {
        assert_eq!(render(&[]), "WEBVTT\n\n");
    }
}
