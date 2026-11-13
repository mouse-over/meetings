//! Audio loading: any container/codec symphonia supports (MP4/M4A with AAC from Teams, WAV, MP3, FLAC, …) →
//! 16 kHz mono f32, decoded and resampled in a streaming way. Falls back to `ffmpeg` on PATH when symphonia
//! cannot open or decode the file.

use crate::{Error, Result, SAMPLE_RATE};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use std::path::Path;
use std::process::{Command, Stdio};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Which decoder produced the audio (for metadata).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Decoder {
    Symphonia,
    Ffmpeg,
}

/// Decode `path` to 16 kHz mono. Tries symphonia first, then `ffmpeg` if available.
pub fn load(path: &Path) -> Result<(Vec<f32>, Decoder)> {
    if !path.is_file() {
        return Err(Error::InvalidInput(format!("input file not found: {}", path.display())));
    }
    match decode_symphonia(path) {
        Ok(a) => Ok((a, Decoder::Symphonia)),
        Err(e) => match decode_ffmpeg(path) {
            Ok(a) => Ok((a, Decoder::Ffmpeg)),
            Err(fe) => Err(Error::Audio(format!("{e}; ffmpeg fallback: {fe}"))),
        },
    }
}

/// Pure-Rust decoding with symphonia.
pub fn decode_symphonia(path: &Path) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())
        .map_err(|e| Error::Audio(format!("unsupported format: {e}")))?;
    let track = format.default_track(TrackType::Audio).ok_or_else(|| Error::Audio("no audio track".into()))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| Error::Audio("missing audio codec parameters".into()))?
        .clone();
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| Error::Audio(format!("unsupported codec: {e}")))?;

    let mut resampler: Option<Resample16k> = None;
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymError::ResetRequired) => break,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(Error::Audio(format!("read error: {e}"))),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            Err(SymError::DecodeError(_)) => continue, // skip a corrupt packet
            Err(e) => return Err(Error::Audio(format!("decode error: {e}"))),
        };
        let rate = buf.spec().rate();
        let channels = buf.spec().channels().count().max(1);
        interleaved.resize(buf.samples_interleaved(), 0.0);
        buf.copy_to_slice_interleaved(&mut interleaved);
        let r = match &mut resampler {
            Some(r) => r,
            None => resampler.insert(Resample16k::new(rate)?),
        };
        r.push(interleaved.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32))?;
    }
    // The resampler is created with the first decoded packet.
    let Some(r) = resampler else {
        return Err(Error::Audio("no audio decoded".into()));
    };
    r.finish()
}

/// Decode with an external `ffmpeg` binary (raw f32le 16 kHz mono on stdout).
pub fn decode_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "-f", "f32le", "-"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::Audio(format!("cannot run ffmpeg: {e}")))?;
    if !out.status.success() {
        return Err(Error::Audio(format!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr).trim())));
    }
    Ok(out.stdout.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

/// Streaming mono resampler to 16 kHz (FFT based, exact for integer ratios like 48 k → 16 k).
pub struct Resample16k {
    inner: Option<Fft<f32>>,
    pending: Vec<f32>,
    out: Vec<f32>,
    scratch: Vec<f32>,
    input_total: usize,
    ratio: f64,
}

impl Resample16k {
    pub fn new(rate: u32) -> Result<Self> {
        let inner = if rate == SAMPLE_RATE {
            None
        } else {
            Some(
                Fft::<f32>::new(rate as usize, SAMPLE_RATE as usize, 1024, 1, FixedSync::Input)
                    .map_err(|e| Error::Audio(format!("resampler: {e}")))?,
            )
        };
        let scratch = vec![0.0; inner.as_ref().map_or(0, |r| r.output_frames_max())];
        Ok(Self {
            inner,
            pending: Vec::new(),
            out: Vec::new(),
            scratch,
            input_total: 0,
            ratio: SAMPLE_RATE as f64 / rate as f64,
        })
    }

    pub fn push(&mut self, samples: impl Iterator<Item = f32>) -> Result<()> {
        let before = self.pending.len();
        self.pending.extend(samples);
        self.input_total += self.pending.len() - before;
        let Some(r) = self.inner.as_mut() else {
            self.out.append(&mut self.pending);
            return Ok(());
        };
        let mut offset = 0;
        while self.pending.len() - offset >= r.input_frames_next() {
            let n = r.input_frames_next();
            let input = InterleavedSlice::new(&self.pending[offset..offset + n], 1, n).map_err(buffer_err)?;
            let frames = self.scratch.len();
            let mut output = InterleavedSlice::new_mut(&mut self.scratch, 1, frames).map_err(buffer_err)?;
            let (used, produced) =
                r.process_into_buffer(&input, &mut output, None).map_err(|e| Error::Audio(format!("resample: {e}")))?;
            self.out.extend_from_slice(&self.scratch[..produced]);
            offset += used;
        }
        self.pending.drain(..offset);
        Ok(())
    }

    pub fn finish(mut self) -> Result<Vec<f32>> {
        let Some(mut r) = self.inner.take() else { return Ok(self.out) };
        let expected = (self.input_total as f64 * self.ratio).ceil() as usize;
        let delay = r.output_delay();
        let mut tail = std::mem::take(&mut self.pending);
        // Last partial chunk, then silence until the delay is flushed.
        loop {
            let n = r.input_frames_next();
            let valid = tail.len().min(n);
            tail.resize(n, 0.0);
            let input = InterleavedSlice::new(&tail[..n], 1, n).map_err(buffer_err)?;
            let frames = self.scratch.len();
            let mut output = InterleavedSlice::new_mut(&mut self.scratch, 1, frames).map_err(buffer_err)?;
            let idx =
                Indexing { input_offset: 0, output_offset: 0, partial_len: Some(valid), active_channels_mask: None };
            let (_, produced) = r
                .process_into_buffer(&input, &mut output, Some(&idx))
                .map_err(|e| Error::Audio(format!("resample: {e}")))?;
            self.out.extend_from_slice(&self.scratch[..produced]);
            tail.clear();
            if self.out.len() >= expected + delay {
                break;
            }
        }
        let mut out = self.out.split_off(delay.min(self.out.len()));
        out.truncate(expected);
        Ok(out)
    }
}

/// The buffers are sized from the resampler, so this only fires on a bug.
fn buffer_err(e: impl std::fmt::Display) -> Error {
    Error::Audio(format!("resampler buffer: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, freq: f32, seconds: f32) -> Vec<f32> {
        (0..(rate as f32 * seconds) as usize)
            .map(|i| (i as f32 / rate as f32 * freq * std::f32::consts::TAU).sin())
            .collect()
    }

    #[test]
    fn passthrough_at_16k() {
        let s = sine(16_000, 440.0, 0.5);
        let mut r = Resample16k::new(16_000).unwrap();
        r.push(s.iter().copied()).unwrap();
        assert_eq!(r.finish().unwrap(), s);
    }

    #[test]
    fn resamples_48k_with_exact_length_and_no_delay() {
        let input = sine(48_000, 440.0, 2.0);
        let mut r = Resample16k::new(48_000).unwrap();
        // feed in odd-sized pieces like a decoder would
        for piece in input.chunks(1000) {
            r.push(piece.iter().copied()).unwrap();
        }
        let out = r.finish().unwrap();
        assert_eq!(out.len(), 32_000);
        let expected = sine(16_000, 440.0, 2.0);
        // compare away from the edges
        let err: f32 =
            out[2000..30_000].iter().zip(&expected[2000..30_000]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 0.02, "max error {err}");
    }

    #[test]
    fn resamples_44k1() {
        let mut r = Resample16k::new(44_100).unwrap();
        r.push(sine(44_100, 300.0, 1.0).into_iter()).unwrap();
        assert_eq!(r.finish().unwrap().len(), 16_000);
    }

    #[test]
    fn wav_roundtrip_via_symphonia() {
        let dir = std::env::temp_dir().join(format!("mo-meetings-audio-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        let samples = sine(48_000, 440.0, 1.0);
        // 16-bit stereo WAV
        let mut data = Vec::new();
        for s in &samples {
            let v = (s * 0.5 * i16::MAX as f32) as i16;
            data.extend_from_slice(&v.to_le_bytes());
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&2u16.to_le_bytes()); // channels
        wav.extend_from_slice(&48_000u32.to_le_bytes());
        wav.extend_from_slice(&(48_000u32 * 4).to_le_bytes());
        wav.extend_from_slice(&4u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        std::fs::write(&path, wav).unwrap();
        let out = decode_symphonia(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(out.len(), 16_000);
        let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!((peak - 0.5).abs() < 0.05, "peak {peak}");
    }

    #[test]
    fn missing_file_is_invalid_input() {
        let e = load(Path::new("/nonexistent/x.mp4")).unwrap_err();
        assert_eq!(e.exit_code(), 2);
    }
}
