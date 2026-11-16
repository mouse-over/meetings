use super::*;
use mo_meetings_core::SAMPLE_RATE;
use mo_meetings_core::asr::{Asr, AsrOptions, Recognized};
use mo_meetings_core::diarize::{DiarizeOptions, Diarizer};
use std::sync::atomic::Ordering;

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mo-meetings-project-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_recording(path: &Path, seed: u8) {
    std::fs::write(path, (0..100_000u32).map(|i| (i as u8).wrapping_mul(seed)).collect::<Vec<_>>()).unwrap();
}

/// Two words per chunk, language as asked (or `xx`); counts the calls.
#[derive(Default)]
struct FakeAsr {
    calls: usize,
}

impl Asr for FakeAsr {
    fn transcribe(&mut self, audio: &[f32], opts: &AsrOptions) -> mo_meetings_core::Result<Recognized> {
        self.calls += 1;
        let secs = audio.len() as f64 / SAMPLE_RATE as f64;
        let lang = opts.language.clone().unwrap_or_else(|| "xx".into());
        let w = |t: &str, a: f64, b: f64| Word { prob: Some(0.75), ..Word::new(t, a, b) };
        Ok(Recognized { words: vec![w(" a", 0.3, 1.0), w(" b.", 1.2, secs - 0.1)], language: Some(lang) })
    }
    fn model_name(&self) -> String {
        "fake".into()
    }
}

#[derive(Default)]
struct FakeDiarizer {
    calls: usize,
}

impl Diarizer for FakeDiarizer {
    fn diarize(&mut self, _: &[f32], _: &DiarizeOptions) -> mo_meetings_core::Result<Diarization> {
        self.calls += 1;
        // Gaps > 3 s, so each turn is its own chunk.
        let turns =
            vec![Turn::new(1.0, 6.0, 0), Turn::new(11.0, 16.0, 1), Turn::new(21.0, 26.0, 0), Turn::new(31.0, 36.0, 1)];
        let embeddings = vec![Some(vec![0.5, -1.0, 2.0]), None, Some(vec![1.0, 0.0, 0.25]), Some(vec![0.0, 0.0, 1.0])];
        Ok(Diarization { turns, num_speakers: 2, estimated: true, embeddings, ..Default::default() })
    }
}

fn audio() -> Vec<f32> {
    (0..40 * SAMPLE_RATE as usize).map(|i| (i as f32 * 0.05).sin() * 0.3).collect()
}

/// Runs the fake pipeline through `project`; `stop_after` chunks done, then cancels.
fn run(
    project: &mut Project,
    cfg: &Config,
    stop_after: Option<usize>,
    diar: &mut FakeDiarizer,
    asr: &mut FakeAsr,
) -> Result<Outcome> {
    let audio = audio();
    let cancel = AtomicBool::new(false);
    project.transcribe_using(cfg, &mut |_, resume, store| {
        let mut chunks = 0;
        pipeline::transcribe_audio_with(
            &audio,
            diar,
            asr,
            cfg,
            resume,
            store,
            &mut |p| {
                if let Progress::Chunk { .. } = p {
                    chunks += 1;
                    if Some(chunks) == stop_after {
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
            },
            &cancel,
        )
    })
}

fn q(p: &Project, sql: &str) -> i64 {
    p.conn.query_row(sql, [], |r| r.get(0)).unwrap()
}

#[test]
fn creates_and_reopens_a_project_and_rejects_other_files() {
    let d = tmp("open");
    let rec = d.join("meeting.mp4");
    write_recording(&rec, 3);
    let path = d.join("meeting.mom");
    Project::create(&path, &rec).unwrap();
    assert!(matches!(Project::create(&path, &rec), Err(Error::Exists(_))));
    let p = Project::open(&path).unwrap();
    assert_eq!(p.recording().unwrap(), std::path::absolute(&rec).unwrap());
    assert!(p.runs().unwrap().is_empty());

    // An SQLite file that is not a project, and a project from a newer version.
    let other = d.join("other.db");
    Connection::open(&other).unwrap().execute_batch("CREATE TABLE x (y)").unwrap();
    assert!(matches!(Project::open(&other), Err(Error::NotAProject(_))));
    Connection::open(&path).unwrap().pragma_update(None, "user_version", 99).unwrap();
    let e = Project::open(&path).err().unwrap();
    assert!(matches!(e, Error::NewerSchema { found: 99, supported: 1, .. }), "{e}");
    assert!(matches!(Project::open(&d.join("missing.mom")), Err(Error::Core(_))));
}

#[test]
fn finds_a_moved_recording_and_checks_it_is_the_same_file() {
    let d = tmp("moved");
    std::fs::create_dir_all(d.join("a")).unwrap();
    let rec = d.join("a/meeting.mp4");
    write_recording(&rec, 3);
    Project::create(&d.join("a/meeting.mom"), &rec).unwrap();

    // Project and recording moved together: found by the relative path.
    std::fs::rename(d.join("a"), d.join("b")).unwrap();
    let mut p = Project::open(&d.join("b/meeting.mom")).unwrap();
    assert_eq!(p.recording().unwrap(), d.join("b/meeting.mp4"));

    // Recording moved elsewhere: missing until located.
    std::fs::rename(d.join("b/meeting.mp4"), d.join("moved.mp4")).unwrap();
    let e = p.recording().err().unwrap();
    assert!(matches!(e, Error::RecordingMissing(_)) && e.exit_code() == 2, "{e}");
    let other = d.join("other.mp4");
    write_recording(&other, 5);
    assert!(matches!(p.locate(&other), Err(Error::RecordingMismatch { .. })));
    assert_eq!(p.locate(&d.join("moved.mp4")).unwrap(), d.join("moved.mp4"));
    assert_eq!(Project::open(&d.join("b/meeting.mom")).unwrap().recording().unwrap(), d.join("moved.mp4"));

    // Same name and place, other content.
    write_recording(&d.join("moved.mp4"), 7);
    assert!(matches!(p.recording(), Err(Error::RecordingMismatch { .. })));
}

#[test]
fn resumes_an_interrupted_run_from_the_next_chunk() {
    let d = tmp("resume");
    let rec = d.join("meeting.wav");
    write_recording(&rec, 3);
    let cfg = Config::default();

    // Reference: one uninterrupted run in another project.
    let mut whole = Project::create(&d.join("whole.mom"), &rec).unwrap();
    let full = run(&mut whole, &cfg, None, &mut FakeDiarizer::default(), &mut FakeAsr::default()).unwrap();
    assert_eq!(full.transcript.chunks.len(), 4);

    // Cancelled after 2 of 4 chunks: diarization and those chunks are stored.
    let path = d.join("meeting.mom");
    let mut p = Project::create(&path, &rec).unwrap();
    let mut asr = FakeAsr::default();
    let e = run(&mut p, &cfg, Some(2), &mut FakeDiarizer::default(), &mut asr).err().unwrap();
    assert!(matches!(e, Error::Core(mo_meetings_core::Error::Cancelled)), "{e}");
    assert_eq!(asr.calls, 2);
    let runs = p.runs().unwrap();
    assert_eq!((runs.len(), runs[0].status.as_str(), runs[0].chunks, runs[0].chunks_done), (1, "cancelled", 4, 2));
    drop(p);

    // Reopened: same settings resume the run, no diarization, only the remaining chunks.
    let mut p = Project::open(&path).unwrap();
    let (mut diar, mut asr) = (FakeDiarizer::default(), FakeAsr::default());
    let out = run(&mut p, &cfg, None, &mut diar, &mut asr).unwrap();
    assert_eq!((diar.calls, asr.calls), (0, 2));
    assert_eq!((out.resumed_chunks, out.resumed_diarization, out.run_id), (2, true, runs[0].id));
    assert_eq!(out.transcript.cues, full.transcript.cues);
    assert_eq!(out.transcript.words, full.transcript.words);
    assert_eq!(out.transcript.chunk_languages, full.transcript.chunk_languages);
    let runs = p.runs().unwrap();
    assert_eq!((runs.len(), runs[0].status.as_str(), runs[0].chunks_done), (1, "done", 4));
    assert_eq!(runs[0].cues, out.transcript.cues.len());

    // Stored data: every word has a speaker, embeddings round-trip, the run's chunks are selected.
    assert_eq!(q(&p, "SELECT count(*) FROM word WHERE speaker IS NULL"), 0);
    assert_eq!(q(&p, "SELECT count(*) FROM word") as usize, full.transcript.words.len());
    assert_eq!(q(&p, "SELECT count(*) FROM selection"), 4);
    let resume = p.load_resume(runs[0].id).unwrap();
    let (diarization, _) = resume.prepared.unwrap();
    assert_eq!(
        diarization.embeddings,
        FakeDiarizer::default().diarize(&[], &DiarizeOptions::default()).unwrap().embeddings
    );

    // Other settings: a new run from scratch.
    let en = Config { asr: AsrOptions { language: Some("en".into()), prompt: None }, ..Config::default() };
    let (mut diar, mut asr) = (FakeDiarizer::default(), FakeAsr::default());
    let out = run(&mut p, &en, None, &mut diar, &mut asr).unwrap();
    assert_eq!((diar.calls, asr.calls, out.resumed_chunks), (1, 4, 0));
    assert_eq!(p.runs().unwrap().len(), 2);
    assert_eq!(q(&p, "SELECT count(DISTINCT run_id) FROM selection"), 1);
    assert_eq!(q(&p, "SELECT run_id FROM selection LIMIT 1"), out.run_id);
}

#[test]
fn a_run_interrupted_before_the_diarization_starts_over() {
    let d = tmp("early");
    let rec = d.join("meeting.wav");
    write_recording(&rec, 3);
    let cfg = Config::default();
    let mut p = Project::create(&d.join("m.mom"), &rec).unwrap();
    let e = p.transcribe_using(&cfg, &mut |_, _, _| Err(mo_meetings_core::Error::Diarization("boom".into())));
    assert!(e.is_err());
    assert_eq!(p.runs().unwrap()[0].status, "failed");
    let (mut diar, mut asr) = (FakeDiarizer::default(), FakeAsr::default());
    let out = run(&mut p, &cfg, None, &mut diar, &mut asr).unwrap();
    assert_eq!((diar.calls, asr.calls, out.resumed_diarization), (1, 4, false));
    assert_eq!(p.runs().unwrap().len(), 1);
}
