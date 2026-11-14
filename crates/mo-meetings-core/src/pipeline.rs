//! The whole pipeline: decode → diarize → chunk → ASR per chunk → assign speakers → cues.
//!
//! Each step is a public function ([`diarize`], [`speech_chunks`], [`transcribe_chunks`], [`assemble`]);
//! [`run`] chains them with the bundled backends.

use crate::asr::whisper::{WhisperAsr, WhisperParams};
use crate::asr::{Asr, AsrOptions};
use crate::assembly::{self, CueRules};
use crate::audio::{self, Decoder};
use crate::chunk::{self, ChunkRules};
use crate::diarize::sherpa::{SherpaDiarizer, SherpaParams};
use crate::diarize::{Diarization, DiarizeOptions, Diarizer};
use crate::{Chunk, Cue, Result, SAMPLE_RATE, Turn, Word, models};
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Load,
    Decode,
    Diarize,
    Transcribe,
    Assemble,
}

#[derive(Clone, Debug)]
pub enum Progress {
    Started(Stage),
    /// ASR progress in chunks and audio seconds.
    Chunk {
        done: usize,
        total: usize,
        audio_done: f64,
        audio_total: f64,
    },
    Finished(Stage, f64),
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Registry id of the whisper model (see [`models::MODELS`]).
    pub asr_model: String,
    pub asr: AsrOptions,
    pub num_speakers: Option<usize>,
    /// Threads for diarization (ONNX Runtime, CPU) and whisper's CPU parts.
    pub threads: usize,
    pub cue_rules: CueRules,
    pub chunk_rules: ChunkRules,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            asr_model: "whisper-large-v3-turbo".into(),
            asr: AsrOptions::default(),
            num_speakers: None,
            threads: 4,
            cue_rules: CueRules::default(),
            chunk_rules: ChunkRules::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Timings {
    pub load_s: f64,
    pub decode_s: f64,
    pub diarize_s: f64,
    pub asr_s: f64,
    pub assemble_s: f64,
    pub total_s: f64,
}

/// Metadata written next to the VTT (`--json`).
#[derive(Clone, Debug, Serialize)]
pub struct Metadata {
    pub tool: String,
    pub input: String,
    pub audio_duration_s: f64,
    pub decoder: Option<Decoder>,
    pub model: String,
    pub language: Option<String>,
    pub prompt: bool,
    pub num_speakers: usize,
    pub num_speakers_estimated: bool,
    pub num_speakers_hint: Option<usize>,
    pub diarization: serde_json::Value,
    pub chunks: usize,
    pub words: usize,
    pub cues: usize,
    pub timings: Timings,
}

pub struct Transcript {
    pub cues: Vec<Cue>,
    pub words: Vec<Word>,
    /// Non-overlapping turns used for word assignment.
    pub turns: Vec<Turn>,
    pub chunks: Vec<Chunk>,
    /// Language whisper used for each chunk (detected per chunk with auto-detection), parallel to `chunks`.
    pub chunk_languages: Vec<Option<String>>,
    pub diarization: Diarization,
    pub meta: Metadata,
}

// The steps below can be called one by one (e.g. to store intermediate data or re-transcribe some chunks
// with other options); [`transcribe_audio`] and [`run`] chain them.

/// Step 1: speaker turns of the whole recording (16 kHz mono).
pub fn diarize(audio: &[f32], diarizer: &mut dyn Diarizer, num_speakers: Option<usize>) -> Result<Diarization> {
    diarizer.diarize(audio, &DiarizeOptions { num_speakers })
}

/// Step 2: spans of speech sent to the ASR in one call each (see [`chunk::speech_chunks`]).
pub fn speech_chunks(turns: &[Turn], audio: &[f32], rules: &ChunkRules) -> Vec<Chunk> {
    chunk::speech_chunks(turns, audio, rules)
}

/// The words of one transcribed chunk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChunkResult {
    /// Position of the chunk in the slice given to [`transcribe_chunks`].
    pub index: usize,
    pub chunk: Chunk,
    /// Words with absolute times, ending inside the chunk; no speakers yet.
    pub words: Vec<Word>,
    /// Language the ASR used: the requested one, or the one it detected with auto-detection.
    pub language: Option<String>,
}

/// Step 3: transcribes `chunks` (all of a recording, or any subset) with `opts`. The loaded model is reused,
/// so the same `asr` can transcribe other chunks with other options afterwards. `sink` gets each chunk's
/// words as soon as the chunk is done; `cancel` is checked before every chunk and stops with
/// [`crate::Error::Cancelled`].
pub fn transcribe_chunks(
    audio: &[f32],
    chunks: &[Chunk],
    asr: &mut dyn Asr,
    opts: &AsrOptions,
    sink: &mut dyn FnMut(&ChunkResult),
    cancel: &AtomicBool,
) -> Result<Vec<ChunkResult>> {
    let mut out = Vec::with_capacity(chunks.len());
    for (index, c) in chunks.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(crate::Error::Cancelled);
        }
        let a = ((c.start * SAMPLE_RATE as f64) as usize).min(audio.len());
        let b = ((c.end * SAMPLE_RATE as f64) as usize).min(audio.len()).max(a);
        let rec = asr.transcribe(&audio[a..b], opts)?;
        let words = rec
            .words
            .into_iter()
            .map(|mut w| {
                w.start += c.start;
                w.end = (w.end + c.start).min(c.end.max(w.start));
                w
            })
            .collect();
        let r = ChunkResult { index, chunk: *c, words, language: rec.language };
        sink(&r);
        out.push(r);
    }
    Ok(out)
}

/// Words with speakers, the turns they were assigned from, and the cues built from them.
#[derive(Clone, Debug, PartialEq)]
pub struct Assembled {
    /// Sorted by start time, each with a speaker.
    pub words: Vec<Word>,
    /// Non-overlapping turns (see [`assembly::exclusive_turns`]).
    pub turns: Vec<Turn>,
    pub cues: Vec<Cue>,
}

/// Step 4: assigns speakers from the diarization `turns` (may overlap) to `words` and builds the cues. Cheap,
/// so it can be re-run after the turns or words change.
pub fn assemble(mut words: Vec<Word>, turns: &[Turn], rules: &CueRules) -> Assembled {
    words.sort_by(|x, y| x.start.total_cmp(&y.start));
    let turns = assembly::exclusive_turns(turns);
    assembly::assign_speakers(&mut words, &turns);
    let cues = assembly::build_cues(&words, rules);
    Assembled { words, turns, cues }
}

/// Backend-agnostic core: audio (16 kHz mono) → transcript. `meta.input`, `decoder` and load/decode
/// timings are left for the caller to fill. `cancel` is checked between the steps and between chunks.
pub fn transcribe_audio(
    audio: &[f32],
    diarizer: &mut dyn Diarizer,
    asr: &mut dyn Asr,
    cfg: &Config,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<Transcript> {
    let mut timings = Timings::default();
    let check = || if cancel.load(Ordering::Relaxed) { Err(crate::Error::Cancelled) } else { Ok(()) };

    check()?;
    progress(Progress::Started(Stage::Diarize));
    let t = Instant::now();
    let diarization = diarize(audio, diarizer, cfg.num_speakers)?;
    timings.diarize_s = t.elapsed().as_secs_f64();
    progress(Progress::Finished(Stage::Diarize, timings.diarize_s));

    check()?;
    progress(Progress::Started(Stage::Transcribe));
    let t = Instant::now();
    let chunks = speech_chunks(&diarization.turns, audio, &cfg.chunk_rules);
    let audio_total: f64 = chunks.iter().map(|c| c.end - c.start).sum();
    let mut audio_done = 0.0;
    let results = transcribe_chunks(
        audio,
        &chunks,
        asr,
        &cfg.asr,
        &mut |r| {
            audio_done += r.chunk.end - r.chunk.start;
            progress(Progress::Chunk { done: r.index + 1, total: chunks.len(), audio_done, audio_total });
        },
        cancel,
    )?;
    timings.asr_s = t.elapsed().as_secs_f64();
    progress(Progress::Finished(Stage::Transcribe, timings.asr_s));

    progress(Progress::Started(Stage::Assemble));
    let t = Instant::now();
    let chunk_languages = results.iter().map(|r| r.language.clone()).collect();
    let words = results.into_iter().flat_map(|r| r.words).collect();
    let Assembled { words, turns, cues } = assemble(words, &diarization.turns, &cfg.cue_rules);
    timings.assemble_s = t.elapsed().as_secs_f64();
    progress(Progress::Finished(Stage::Assemble, timings.assemble_s));

    let speakers_in_output = assembly::speaker_numbers(&words).len();
    let meta = Metadata {
        tool: format!("mo-meetings {}", env!("CARGO_PKG_VERSION")),
        input: String::new(),
        audio_duration_s: audio.len() as f64 / SAMPLE_RATE as f64,
        decoder: None,
        model: asr.model_name(),
        language: cfg.asr.language.clone(),
        prompt: cfg.asr.prompt.as_deref().is_some_and(|p| !p.trim().is_empty()),
        num_speakers: speakers_in_output,
        num_speakers_estimated: diarization.estimated,
        num_speakers_hint: cfg.num_speakers,
        diarization: {
            let mut v = serde_json::to_value(&diarization.report).unwrap_or_default();
            if let serde_json::Value::Object(m) = &mut v {
                for (k, t) in &diarization.timings {
                    m.insert(k.clone(), (*t).into());
                }
            }
            v
        },
        chunks: chunks.len(),
        words: words.len(),
        cues: cues.len(),
        timings,
    };
    Ok(Transcript { cues, words, turns, chunks, chunk_languages, diarization, meta })
}

fn sherpa_diarizer(cfg: &Config) -> Result<(SherpaDiarizer, SherpaParams)> {
    let seg_path = models::require("pyannote-segmentation-3.0")?;
    let emb_path = models::require("titanet-large")?;
    let mut sp = SherpaParams::new(seg_path, emb_path);
    sp.threads = cfg.threads;
    Ok((SherpaDiarizer::new(sp.clone())?, sp))
}

/// Diarization only (bundled sherpa-onnx backend): returns the diarization and the audio duration in seconds.
pub fn run_diarization(input: &Path, cfg: &Config) -> Result<(Diarization, f64)> {
    if !input.is_file() {
        return Err(crate::Error::InvalidInput(format!("input file not found: {}", input.display())));
    }
    let (mut diarizer, _) = sherpa_diarizer(cfg)?;
    let (audio, _) = audio::load(input)?;
    let d = diarizer.diarize(&audio, &DiarizeOptions { num_speakers: cfg.num_speakers })?;
    Ok((d, audio.len() as f64 / SAMPLE_RATE as f64))
}

/// Full pipeline with the bundled backends (sherpa-onnx diarization, whisper.cpp ASR) and models from the
/// user cache. Missing models are reported before any work is done.
pub fn run(input: &Path, cfg: &Config, progress: &mut dyn FnMut(Progress)) -> Result<Transcript> {
    run_cancellable(input, cfg, progress, &AtomicBool::new(false))
}

/// [`run`] that stops with [`crate::Error::Cancelled`] once `cancel` is set (checked between the steps and
/// between chunks, so a chunk in progress finishes first).
pub fn run_cancellable(
    input: &Path,
    cfg: &Config,
    progress: &mut dyn FnMut(Progress),
    cancel: &AtomicBool,
) -> Result<Transcript> {
    let start = Instant::now();
    if !input.is_file() {
        return Err(crate::Error::InvalidInput(format!("input file not found: {}", input.display())));
    }
    let asr_path = models::require(&cfg.asr_model)?;
    models::require("pyannote-segmentation-3.0")?;
    models::require("titanet-large")?;

    progress(Progress::Started(Stage::Load));
    let t = Instant::now();
    let (mut diarizer, sp) = sherpa_diarizer(cfg)?;
    let dtw = match cfg.asr_model.as_str() {
        "whisper-large-v3" => Some(whisper_rs::DtwModelPreset::LargeV3),
        "whisper-large-v3-turbo" => Some(whisper_rs::DtwModelPreset::LargeV3Turbo),
        _ => None,
    };
    let wp = WhisperParams { threads: cfg.threads, dtw, ..Default::default() };
    let mut asr = WhisperAsr::new(&asr_path, cfg.asr_model.clone(), wp)?;
    let load_s = t.elapsed().as_secs_f64();
    progress(Progress::Finished(Stage::Load, load_s));

    progress(Progress::Started(Stage::Decode));
    let t = Instant::now();
    let (audio, decoder) = audio::load(input)?;
    let decode_s = t.elapsed().as_secs_f64();
    progress(Progress::Finished(Stage::Decode, decode_s));
    if audio.is_empty() {
        return Err(crate::Error::InvalidInput("input has no audio".into()));
    }

    let mut tr = transcribe_audio(&audio, &mut diarizer, &mut asr, cfg, progress, cancel)?;
    tr.meta.input = input.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    tr.meta.decoder = Some(decoder);
    tr.meta.timings.load_s = load_s;
    tr.meta.timings.decode_s = decode_s;
    tr.meta.timings.total_s = start.elapsed().as_secs_f64();
    if let serde_json::Value::Object(map) = &mut tr.meta.diarization {
        map.insert(
            "params".into(),
            serde_json::json!({
                "segmentation": "pyannote-segmentation-3.0",
                "embedding": "titanet-large",
                "threshold": sp.threshold,
                "min_duration_on": sp.min_duration_on,
                "min_duration_off": sp.min_duration_off,
                "merge_min_share": sp.merge.min_share,
                "merge_similarity": sp.merge.similarity,
            }),
        );
    }
    Ok(tr)
}

/// JSON written by `diarize` (CLI and MCP): turns (may overlap) and exclusive turns as `[start, end, label]`
/// with labels `SPEAKER_NN`, plus the merge report and timings.
pub fn diarization_json(d: &Diarization, audio_duration_s: f64, total_s: f64) -> serde_json::Value {
    let label = |s: usize| format!("SPEAKER_{s:02}");
    let turns: Vec<_> = d.turns.iter().map(|t| serde_json::json!([t.start, t.end, label(t.speaker)])).collect();
    let excl: Vec<_> = assembly::exclusive_turns(&d.turns)
        .iter()
        .map(|t| serde_json::json!([t.start, t.end, label(t.speaker)]))
        .collect();
    let timings: serde_json::Map<String, serde_json::Value> =
        d.timings.iter().map(|(k, v)| (k.clone(), (*v).into())).collect();
    serde_json::json!({
        "audio_duration_s": audio_duration_s,
        "num_speakers": d.num_speakers,
        "num_speakers_estimated": d.estimated,
        "report": d.report,
        "timings": timings,
        "total_s": total_s,
        "turns": turns,
        "exclusive": excl,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::Recognized;
    use crate::diarize::merge::MergeReport;

    /// Recognizes one word per chunk: the chunk's length and the language it was asked for (or `xx`).
    #[derive(Default)]
    struct FakeAsr {
        calls: Vec<Option<String>>,
    }

    impl Asr for FakeAsr {
        fn transcribe(&mut self, audio: &[f32], opts: &AsrOptions) -> Result<Recognized> {
            self.calls.push(opts.language.clone());
            let lang = opts.language.clone().unwrap_or_else(|| "xx".into());
            let secs = audio.len() as f64 / SAMPLE_RATE as f64;
            let w = Word { prob: Some(0.5), ..Word::new(format!(" {lang}{secs:.0}"), 0.5, secs + 5.0) };
            Ok(Recognized { words: vec![w], language: Some(lang) })
        }
        fn model_name(&self) -> String {
            "fake".into()
        }
    }

    struct FakeDiarizer(Vec<Turn>);

    impl Diarizer for FakeDiarizer {
        fn diarize(&mut self, _: &[f32], _: &DiarizeOptions) -> Result<Diarization> {
            Ok(Diarization {
                turns: self.0.clone(),
                num_speakers: 2,
                estimated: true,
                report: MergeReport::default(),
                timings: vec![],
            })
        }
    }

    fn audio(secs: f64) -> Vec<f32> {
        (0..(secs * SAMPLE_RATE as f64) as usize).map(|i| (i as f32 * 0.05).sin() * 0.3).collect()
    }

    #[test]
    fn transcribes_a_subset_of_chunks_with_its_own_options_and_streams_each_chunk() {
        let audio = audio(30.0);
        let chunks = [Chunk { start: 2.0, end: 6.0 }, Chunk { start: 20.0, end: 23.0 }];
        let mut asr = FakeAsr::default();
        let en = AsrOptions { language: Some("en".into()), prompt: None };
        let mut seen = Vec::new();
        let out =
            transcribe_chunks(&audio, &chunks, &mut asr, &en, &mut |r| seen.push(r.clone()), &AtomicBool::new(false))
                .unwrap();
        assert_eq!(seen, out);
        assert_eq!(out[1].index, 1);
        assert_eq!(out[1].language.as_deref(), Some("en"));
        // Absolute times, ending inside the chunk.
        assert_eq!((out[1].words[0].start, out[1].words[0].end), (20.5, 23.0));
        assert_eq!(out[1].words[0].text, " en3");

        // Same model, another chunk, other options.
        let auto = AsrOptions { language: None, prompt: None };
        let out =
            transcribe_chunks(&audio, &chunks[..1], &mut asr, &auto, &mut |_| {}, &AtomicBool::new(false)).unwrap();
        assert_eq!(out[0].language.as_deref(), Some("xx"));
        assert_eq!(asr.calls, [Some("en".into()), Some("en".into()), None]);
    }

    #[test]
    fn cancel_stops_before_the_next_chunk() {
        let audio = audio(30.0);
        let chunks =
            [Chunk { start: 0.0, end: 5.0 }, Chunk { start: 10.0, end: 15.0 }, Chunk { start: 20.0, end: 25.0 }];
        let cancel = AtomicBool::new(false);
        let mut asr = FakeAsr::default();
        let mut done = 0;
        let r = transcribe_chunks(
            &audio,
            &chunks,
            &mut asr,
            &AsrOptions::default(),
            &mut |_| {
                done += 1;
                cancel.store(true, Ordering::Relaxed);
            },
            &cancel,
        );
        assert!(matches!(r, Err(crate::Error::Cancelled)), "{r:?}");
        assert_eq!((done, asr.calls.len()), (1, 1));
    }

    #[test]
    fn transcribe_audio_chains_the_steps() {
        let audio = audio(30.0);
        let turns = vec![Turn::new(1.0, 8.0, 0), Turn::new(15.0, 25.0, 1)];
        let mut progress = Vec::new();
        let tr = transcribe_audio(
            &audio,
            &mut FakeDiarizer(turns.clone()),
            &mut FakeAsr::default(),
            &Config::default(),
            &mut |p| progress.push(p),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!tr.chunks.is_empty());
        assert_eq!(tr.chunk_languages, vec![Some("cs".to_string()); tr.chunks.len()]);
        assert!(tr.words.iter().all(|w| w.speaker.is_some() && w.prob == Some(0.5)));
        let direct = assemble(
            tr.words.iter().map(|w| Word { speaker: None, ..w.clone() }).collect(),
            &turns,
            &CueRules::default(),
        );
        assert_eq!((direct.words, direct.cues, direct.turns), (tr.words, tr.cues, tr.turns));
        let chunks: Vec<_> = progress.iter().filter(|p| matches!(p, Progress::Chunk { .. })).collect();
        assert_eq!(chunks.len(), tr.chunks.len());

        let cancelled = transcribe_audio(
            &audio,
            &mut FakeDiarizer(turns),
            &mut FakeAsr::default(),
            &Config::default(),
            &mut |_| {},
            &AtomicBool::new(true),
        );
        assert!(matches!(cancelled, Err(crate::Error::Cancelled)));
    }
}
