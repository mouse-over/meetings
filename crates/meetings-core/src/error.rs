use std::path::PathBuf;

/// Errors of the library. The CLI maps them to exit codes (see [`Error::exit_code`]).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("model `{id}` not found at {path} (run `meetings models download`)")]
    ModelMissing { id: String, path: PathBuf },
    #[error("audio decoding failed: {0}")]
    Audio(String),
    #[error("diarization failed: {0}")]
    Diarization(String),
    #[error("speech recognition failed: {0}")]
    Asr(String),
    #[error("model download failed: {0}")]
    Download(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// 2 = bad options/input, 3 = model missing, 4 = processing failed.
    pub fn exit_code(&self) -> i32 {
        match self {
            Error::InvalidInput(_) => 2,
            Error::ModelMissing { .. } => 3,
            _ => 4,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
