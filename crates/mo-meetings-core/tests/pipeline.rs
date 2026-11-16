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
    assert!(tr.words.iter().all(|w| w.prob.is_some_and(|p| (0.0..=1.0).contains(&p))), "every word has a probability");
    assert_eq!(tr.chunk_languages.len(), tr.chunks.len());
    assert!(tr.chunk_languages.iter().all(|l| l.as_deref() == cfg.asr.language.as_deref()));
    for c in &tr.chunks {
        assert!(c.end - c.start <= 30.0 + 1e-6, "chunk longer than whisper's window: {c:?}");
    }
    let doc = vtt::render(&tr.cues);
    assert!(doc.starts_with("WEBVTT\n\n1\n"));
    assert!(doc.contains("<v Speaker 1>"));
}

/// The steps one by one with one loaded model: auto-detected language per chunk, a re-run of some chunks
/// with other options, and cancellation.
#[test]
#[ignore = "needs models and MO_MEETINGS_TEST_AUDIO"]
fn runs_the_steps_one_by_one() {
    use mo_meetings_core::asr::AsrOptions;
    use mo_meetings_core::asr::whisper::{WhisperAsr, WhisperParams};
    use mo_meetings_core::diarize::sherpa::{SherpaDiarizer, SherpaParams};
    use mo_meetings_core::{Error, models};
    use std::sync::atomic::AtomicBool;

    let input = PathBuf::from(std::env::var("MO_MEETINGS_TEST_AUDIO").expect("set MO_MEETINGS_TEST_AUDIO"));
    let cfg = Config::default();
    let (audio, _) = mo_meetings_core::audio::load(&input).expect("decode");
    let sp = SherpaParams::new(
        models::require("pyannote-segmentation-3.0").unwrap(),
        models::require("titanet-large").unwrap(),
    );
    let mut diarizer = SherpaDiarizer::new(sp).unwrap();
    let mut asr =
        WhisperAsr::new(&models::require(&cfg.asr_model).unwrap(), cfg.asr_model.clone(), WhisperParams::default())
            .unwrap();

    let d = pipeline::diarize(&audio, &mut diarizer, None).unwrap();
    let chunks = pipeline::speech_chunks(&d.turns, &audio, &cfg.chunk_rules);
    assert!(chunks.len() >= 2, "{chunks:?}");
    let no = AtomicBool::new(false);
    let auto = AsrOptions { language: None, prompt: None };
    let mut streamed = 0;
    let all = pipeline::transcribe_chunks(
        &audio,
        &chunks,
        &mut asr,
        &auto,
        &mut |_| {
            streamed += 1;
            Ok(())
        },
        &no,
    )
    .unwrap();
    assert_eq!(streamed, chunks.len());
    for r in &all {
        eprintln!(
            "chunk {} {:.1}-{:.1} s: {:?}, {} words",
            r.index,
            r.chunk.start,
            r.chunk.end,
            r.language,
            r.words.len()
        );
        assert!(r.language.as_deref().is_some_and(|l| l.len() == 2), "detected language: {:?}", r.language);
    }

    // The same model, one chunk again with a forced language.
    let en = AsrOptions { language: Some("en".into()), prompt: None };
    let again = pipeline::transcribe_chunks(&audio, &chunks[1..2], &mut asr, &en, &mut |_| Ok(()), &no).unwrap();
    assert_eq!(again[0].language.as_deref(), Some("en"));
    assert_eq!(again[0].chunk, chunks[1]);

    let words: Vec<_> = all.into_iter().flat_map(|r| r.words).collect();
    let a = pipeline::assemble(words, &d.turns, &cfg.cue_rules);
    assert!(!a.cues.is_empty() && a.words.iter().all(|w| w.speaker.is_some()));
    let unsure = a.words.iter().filter(|w| w.prob.is_some_and(|p| p < 0.5)).count();
    eprintln!("{} words, {unsure} with probability < 0.5", a.words.len());

    let r = pipeline::transcribe_chunks(&audio, &chunks, &mut asr, &auto, &mut |_| Ok(()), &AtomicBool::new(true));
    assert!(matches!(r, Err(Error::Cancelled)));
}
