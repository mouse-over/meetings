//! Model registry, user cache location and verified downloads. Models are never stored in the repo.
//!
//! None of the models is gated; no Hugging Face token is needed.

use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Asr,
    Segmentation,
    Embedding,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelInfo {
    pub id: &'static str,
    pub kind: Kind,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
    pub license: &'static str,
    pub attribution: &'static str,
    /// Downloaded by `meetings models download` without arguments.
    pub default: bool,
}

pub const MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "whisper-large-v3-turbo",
        kind: Kind::Asr,
        file: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        size: 1_624_555_275,
        license: "MIT",
        attribution: "OpenAI Whisper large-v3-turbo, ggml conversion by whisper.cpp",
        default: true,
    },
    ModelInfo {
        id: "whisper-large-v3",
        kind: Kind::Asr,
        file: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        sha256: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        size: 3_095_033_483,
        license: "MIT",
        attribution: "OpenAI Whisper large-v3, ggml conversion by whisper.cpp",
        default: false,
    },
    ModelInfo {
        id: "pyannote-segmentation-3.0",
        kind: Kind::Segmentation,
        file: "pyannote-segmentation-3.0.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
        sha256: "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
        size: 5_992_913,
        license: "MIT",
        attribution: "pyannote/segmentation-3.0, Copyright (c) 2023 CNRS; ONNX export by sherpa-onnx",
        default: true,
    },
    ModelInfo {
        id: "titanet-large",
        kind: Kind::Embedding,
        file: "nemo_en_titanet_large.onnx",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx",
        sha256: "d51abcf31717ef28162f26acb9d44dd4127c3d44c9b8624f699f3425daca8e77",
        size: 101_405_493,
        license: "CC-BY-4.0",
        attribution: "NVIDIA NeMo TitaNet-Large (nvidia/speakerverification_en_titanet_large), CC-BY-4.0; ONNX export by sherpa-onnx",
        default: true,
    },
];

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.id == id)
}

/// Model directory: `$MEETINGS_MODELS_DIR`, else `<user cache>/meetings/models`
/// (`~/Library/Caches/meetings/models` on macOS, `~/.cache/meetings/models` on Linux).
pub fn models_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("MEETINGS_MODELS_DIR") {
        return PathBuf::from(d);
    }
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("meetings").join("models")
}

pub fn path(m: &ModelInfo) -> PathBuf {
    models_dir().join(m.file)
}

/// Path of a model that must exist, else [`Error::ModelMissing`].
pub fn require(id: &str) -> Result<PathBuf> {
    let m = find(id).ok_or_else(|| Error::InvalidInput(format!("unknown model `{id}`")))?;
    let p = path(m);
    if p.is_file() { Ok(p) } else { Err(Error::ModelMissing { id: id.to_string(), path: p }) }
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Download `m` into [`models_dir`] (via a `.part` file), verifying SHA-256. Skips a file that already
/// exists with the right size unless `verify_existing` is set, in which case its hash is checked too.
/// `progress(done, total)` is called while downloading.
pub fn download(m: &ModelInfo, verify_existing: bool, progress: &mut dyn FnMut(u64, u64)) -> Result<PathBuf> {
    let dest = path(m);
    if dest.is_file() {
        let ok_size = std::fs::metadata(&dest)?.len() == m.size;
        if ok_size && (!verify_existing || sha256_file(&dest)? == m.sha256) {
            return Ok(dest);
        }
    }
    std::fs::create_dir_all(models_dir())?;
    let part = dest.with_extension("part");
    let resp = ureq::get(m.url).call().map_err(|e| Error::Download(format!("{}: {e}", m.url)))?;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(m.size);
    let mut reader = resp.into_body().into_reader();
    let mut file = std::fs::File::create(&part)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = reader.read(&mut buf).map_err(|e| Error::Download(format!("{}: {e}", m.url)))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    file.sync_all()?;
    drop(file);
    let got = hex(&hasher.finalize());
    if got != m.sha256 {
        std::fs::remove_file(&part).ok();
        return Err(Error::Download(format!("{}: SHA-256 mismatch (expected {}, got {got})", m.id, m.sha256)));
    }
    std::fs::rename(&part, &dest)?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_consistent() {
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{}", m.id);
            assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
            assert!(m.url.starts_with("https://"));
            assert!(!m.license.is_empty() && !m.attribution.is_empty());
        }
        for kind in [Kind::Asr, Kind::Segmentation, Kind::Embedding] {
            assert!(MODELS.iter().any(|m| m.kind == kind && m.default));
        }
    }

    #[test]
    fn sha256_of_file() {
        let p = std::env::temp_dir().join(format!("meetings-sha-{}", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        std::fs::remove_file(&p).ok();
    }
}
