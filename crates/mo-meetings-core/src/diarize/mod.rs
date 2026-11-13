//! Speaker diarization: who speaks when.
//!
//! [`Diarizer`] is the backend interface. The bundled backend ([`sherpa::SherpaDiarizer`]) runs sherpa-onnx
//! (pyannote segmentation-3.0 + NeMo TitaNet-Large + fast clustering) and then [`merge::merge_clusters`]
//! over per-turn embeddings computed once.

pub mod merge;
pub mod sherpa;

use crate::{Result, Turn};
use serde::Serialize;

#[derive(Clone, Debug, Default)]
pub struct DiarizeOptions {
    /// Optional hint: merge the most similar clusters until at most this many speakers remain.
    pub num_speakers: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Diarization {
    /// Speaker turns, sorted by start, may overlap. Speaker ids are internal (not display numbers).
    pub turns: Vec<Turn>,
    pub num_speakers: usize,
    /// `false` when `num_speakers` was capped by a hint.
    pub estimated: bool,
    pub report: merge::MergeReport,
    /// Backend-specific timings in seconds (e.g. clustering vs. embedding).
    pub timings: Vec<(String, f64)>,
}

/// A diarization backend. Implementations get 16 kHz mono audio.
pub trait Diarizer {
    fn diarize(&mut self, audio: &[f32], opts: &DiarizeOptions) -> Result<Diarization>;
}

/// A speaker embedding model (used for post-clustering merges).
pub trait SpeakerEmbedder {
    /// Embeddings for the given spans of `audio` (16 kHz); `None` where the span is too short.
    fn embed_spans(&self, audio: &[f32], spans: &[(f64, f64)]) -> Result<Vec<Option<Vec<f32>>>>;
}
