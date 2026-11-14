use serde::Serialize;

/// A diarization turn: speech of one (internal) speaker id in seconds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

impl Turn {
    pub fn new(start: f64, end: f64, speaker: usize) -> Self {
        Self { start, end, speaker }
    }
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// A recognized word with absolute times in seconds. `text` keeps whisper's leading space.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    /// Internal speaker id after assignment (see [`crate::assembly::assign_speakers`]).
    pub speaker: Option<usize>,
    /// Recognition probability in 0..=1 (whisper token probabilities averaged over the word), for
    /// highlighting uncertain words. `None` when the backend does not report one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prob: Option<f64>,
}

impl Word {
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Self { text: text.into(), start, end, speaker: None, prob: None }
    }
}

/// A span of audio sent to the ASR backend in one call.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Chunk {
    pub start: f64,
    pub end: f64,
}

/// One subtitle cue: `speaker` is the 1-based display number (`Speaker N`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
    pub text: String,
}
