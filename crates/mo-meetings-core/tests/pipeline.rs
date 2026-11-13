//! End-to-end run of the bundled pipeline. Needs downloaded models (`mo-meetings models download`) and a recording:
//!
//! ```sh
//! MO_MEETINGS_TEST_AUDIO=/path/to/meeting.mp4 cargo test --release -p mo-meetings-core --features metal -- --ignored
//! ```

use mo_meetings_core::pipeline::{self, Config};
use mo_meetings_core::vtt;
use std::path::PathBuf;

#[test]
#[ignore = "needs models and MO_MEETINGS_TEST_AUDIO"]
fn transcribes_a_recording() {
    let input = PathBuf::from(std::env::var("MO_MEETINGS_TEST_AUDIO").expect("set MO_MEETINGS_TEST_AUDIO"));
    let cfg = Config::default();
    let tr = pipeline::run(&input, &cfg, &mut |_| {}).expect("pipeline");
    assert!(!tr.words.is_empty(), "no words recognized");
    assert!(tr.meta.num_speakers >= 1);
    assert!(tr.words.windows(2).all(|w| w[0].start <= w[1].start), "words sorted");
    assert!(tr.words.iter().all(|w| w.speaker.is_some()), "every word has a speaker");
    for c in &tr.chunks {
        assert!(c.end - c.start <= 30.0 + 1e-6, "chunk longer than whisper's window: {c:?}");
    }
    let doc = vtt::render(&tr.cues);
    assert!(doc.starts_with("WEBVTT\n\n1\n"));
    assert!(doc.contains("<v Speaker 1>"));
}
