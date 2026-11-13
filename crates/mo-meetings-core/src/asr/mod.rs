//! Speech recognition backends. [`Asr`] transcribes one chunk of 16 kHz audio into timed words.

pub mod whisper;

use crate::{Result, Word};

#[derive(Clone, Debug)]
pub struct AsrOptions {
    /// ISO 639-1 code, `None` = auto-detect.
    pub language: Option<String>,
    /// Glossary / context prompt, sent with every chunk.
    pub prompt: Option<String>,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self { language: Some("cs".into()), prompt: None }
    }
}

/// An ASR backend. `transcribe` gets one chunk (at most ~30 s) and returns words with times relative to
/// the start of `audio`, in order. Every call is independent (no text carried over between chunks).
pub trait Asr {
    fn transcribe(&mut self, audio: &[f32], opts: &AsrOptions) -> Result<Vec<Word>>;
    /// Model name for metadata.
    fn model_name(&self) -> String;
}
