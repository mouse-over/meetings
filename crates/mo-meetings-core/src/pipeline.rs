//! The whole pipeline: decode → diarize → chunk → ASR per chunk → assign speakers → cues.

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
    pub diarization: Diarization,
    pub meta: Metadata,
}

/// Backend-agnostic core: audio (16 kHz mono) → transcript. `meta.input`, `decoder` and load/decode
/// timings are left for the caller to fill.
pub fn transcribe_audio(
    audio: &[f32],
    diarizer: &mut dyn Diarizer,
    asr: &mut dyn Asr,
    cfg: &Config,
    progress: &mut dyn FnMut(Progress),
) -> Result<Transcript> {
    let mut timings = Timings::default();

    progress(Progress::Started(Stage::Diarize));
    let t = Instant::now();
    let diarization = diarizer.diarize(audio, &DiarizeOptions { num_speakers: cfg.num_speakers })?;
    timings.diarize_s = t.elapsed().as_secs_f64();
    progress(Progress::Finished(Stage::Diarize, timings.diarize_s));

    progress(Progress::Started(Stage::Transcribe));
    let t = Instant::now();
    let chunks = chunk::speech_chunks(&diarization.turns, audio, &cfg.chunk_rules);
    let audio_total: f64 = chunks.iter().map(|c| c.end - c.start).sum();
    let mut words: Vec<Word> = Vec::new();
    let mut audio_done = 0.0;
    for (i, c) in chunks.iter().enumerate() {
        let a = ((c.start * SAMPLE_RATE as f64) as usize).min(audio.len());
        let b = ((c.end * SAMPLE_RATE as f64) as usize).min(audio.len());
        for mut w in asr.transcribe(&audio[a..b], &cfg.asr)? {
            w.start += c.start;
            w.end = (w.end + c.start).min(c.end.max(w.start));
            words.push(w);
        }
        audio_done += c.end - c.start;
        progress(Progress::Chunk { done: i + 1, total: chunks.len(), audio_done, audio_total });
    }
    timings.asr_s = t.elapsed().as_secs_f64();
    progress(Progress::Finished(Stage::Transcribe, timings.asr_s));

    progress(Progress::Started(Stage::Assemble));
    let t = Instant::now();
    words.sort_by(|x, y| x.start.total_cmp(&y.start));
    let turns = assembly::exclusive_turns(&diarization.turns);
    assembly::assign_speakers(&mut words, &turns);
    let cues = assembly::build_cues(&words, &cfg.cue_rules);
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
    Ok(Transcript { cues, words, turns, chunks, diarization, meta })
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

    let mut tr = transcribe_audio(&audio, &mut diarizer, &mut asr, cfg, progress)?;
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
