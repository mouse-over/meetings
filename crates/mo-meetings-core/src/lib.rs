//! Local meeting transcription: audio → diarization → ASR per speech chunk → words with speakers → VTT.
//!
//! The steps are independent modules connected by plain data types ([`Turn`], [`Word`], [`Chunk`]); the
//! diarization and ASR backends sit behind the [`diarize::Diarizer`] and [`asr::Asr`] traits so they can be
//! swapped (other models, other runtimes) without touching the assembly or output code.

pub mod asr;
pub mod assembly;
pub mod audio;
pub mod chunk;
pub mod diarize;
mod error;
pub mod models;
pub mod pipeline;
pub mod profile;
pub mod types;
pub mod vtt;

pub use error::{Error, Result};
pub use types::{Chunk, Cue, Turn, Word};

/// All audio in the pipeline is 16 kHz mono f32.
pub const SAMPLE_RATE: u32 = 16_000;
