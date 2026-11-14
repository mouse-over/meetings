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

/// What the backend recognized in one chunk.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recognized {
    /// Words with times relative to the start of the chunk, in order.
    pub words: Vec<Word>,
    /// Language the backend used: the requested one, or the detected one with auto-detection.
    pub language: Option<String>,
}

/// An ASR backend. `transcribe` gets one chunk (at most ~30 s) and returns words with times relative to
/// the start of `audio`, in order. Every call is independent (no text carried over between chunks), so the
/// options may change from call to call without reloading the model.
pub trait Asr {
    fn transcribe(&mut self, audio: &[f32], opts: &AsrOptions) -> Result<Recognized>;
    /// Model name for metadata.
    fn model_name(&self) -> String;
}
