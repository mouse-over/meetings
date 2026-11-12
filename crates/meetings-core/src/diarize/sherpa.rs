//! sherpa-onnx backend: pyannote segmentation-3.0 (fp32) + NeMo TitaNet-Large + fast clustering,
//! then [`super::merge::merge_clusters`] over TitaNet embeddings computed once per turn.

use super::merge::{MergeParams, merge_clusters};
use super::{Diarization, DiarizeOptions, Diarizer, SpeakerEmbedder};
use crate::{Error, Result, SAMPLE_RATE, Turn};
use serde::Serialize;
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig, SpeakerEmbeddingExtractor,
    SpeakerEmbeddingExtractorConfig,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize)]
pub struct SherpaParams {
    pub segmentation_model: PathBuf,
    pub embedding_model: PathBuf,
    /// `FastClustering` distance threshold (higher = fewer clusters).
    pub threshold: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
    /// CPU threads for ONNX Runtime. More than the performance cores slows it down.
    pub threads: usize,
    /// Turns shorter than this are not embedded (s).
    pub min_embed_len: f64,
    pub merge: MergeParams,
}

impl SherpaParams {
    pub fn new(segmentation_model: PathBuf, embedding_model: PathBuf) -> Self {
        Self {
            segmentation_model,
            embedding_model,
            threshold: 1.05,
            min_duration_on: 0.3,
            min_duration_off: 0.5,
            threads: 4,
            min_embed_len: 0.5,
            merge: MergeParams::default(),
        }
    }
}

pub struct SherpaDiarizer {
    params: SherpaParams,
    diarizer: OfflineSpeakerDiarization,
    embedder: SherpaEmbedder,
}

fn path_str(p: &Path) -> Result<String> {
    if !p.is_file() {
        return Err(Error::Diarization(format!("model file not found: {}", p.display())));
    }
    Ok(p.to_string_lossy().into_owned())
}

impl SherpaDiarizer {
    pub fn new(params: SherpaParams) -> Result<Self> {
        let threads = params.threads.max(1) as i32;
        let embedding = SpeakerEmbeddingExtractorConfig {
            model: Some(path_str(&params.embedding_model)?),
            num_threads: threads,
            provider: Some("cpu".into()),
            ..Default::default()
        };
        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(path_str(&params.segmentation_model)?),
                    ..Default::default()
                },
                num_threads: threads,
                provider: Some("cpu".into()),
                ..Default::default()
            },
            embedding,
            clustering: FastClusteringConfig { num_clusters: -1, threshold: params.threshold, ..Default::default() },
            min_duration_on: params.min_duration_on,
            min_duration_off: params.min_duration_off,
        };
        let diarizer = OfflineSpeakerDiarization::create(&config)
            .ok_or_else(|| Error::Diarization("cannot create sherpa-onnx diarizer (bad model?)".into()))?;
        if diarizer.sample_rate() != SAMPLE_RATE as i32 {
            return Err(Error::Diarization(format!("unexpected model sample rate {}", diarizer.sample_rate())));
        }
        let embedder = SherpaEmbedder::new(&params.embedding_model, params.threads, params.min_embed_len)?;
        Ok(Self { params, diarizer, embedder })
    }

    pub fn params(&self) -> &SherpaParams {
        &self.params
    }
}

impl Diarizer for SherpaDiarizer {
    fn diarize(&mut self, audio: &[f32], opts: &DiarizeOptions) -> Result<Diarization> {
        let result =
            self.diarizer.process(audio).ok_or_else(|| Error::Diarization("sherpa-onnx returned no result".into()))?;
        let turns: Vec<Turn> = result
            .sort_by_start_time()
            .into_iter()
            .map(|s| Turn::new(s.start as f64, s.end as f64, s.speaker.max(0) as usize))
            .collect();
        let spans: Vec<(f64, f64)> = turns.iter().map(|t| (t.start, t.end)).collect();
        let emb = self.embedder.embed_spans(audio, &spans)?;
        let merge = MergeParams { max_speakers: opts.num_speakers, ..self.params.merge.clone() };
        let (turns, report) = merge_clusters(&turns, &emb, &merge);
        Ok(Diarization { num_speakers: report.clusters_final, estimated: opts.num_speakers.is_none(), turns, report })
    }
}

/// TitaNet embeddings via sherpa-onnx, computed in parallel (one ONNX thread per worker).
pub struct SherpaEmbedder {
    extractor: SpeakerEmbeddingExtractor,
    workers: usize,
    min_len: f64,
}

impl SherpaEmbedder {
    pub fn new(model: &Path, workers: usize, min_len: f64) -> Result<Self> {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(path_str(model)?),
            num_threads: 1,
            provider: Some("cpu".into()),
            ..Default::default()
        };
        let extractor = SpeakerEmbeddingExtractor::create(&config)
            .ok_or_else(|| Error::Diarization("cannot create speaker embedding extractor".into()))?;
        Ok(Self { extractor, workers: workers.max(1), min_len })
    }

    fn embed_one(&self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        let stream = self.extractor.create_stream().ok_or_else(|| Error::Diarization("cannot create stream".into()))?;
        stream.accept_waveform(SAMPLE_RATE as i32, samples);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return Ok(None);
        }
        Ok(self.extractor.compute(&stream))
    }
}

impl SpeakerEmbedder for SherpaEmbedder {
    fn embed_spans(&self, audio: &[f32], spans: &[(f64, f64)]) -> Result<Vec<Option<Vec<f32>>>> {
        let sr = SAMPLE_RATE as f64;
        let slice = |&(s, e): &(f64, f64)| {
            let a = ((s * sr) as usize).min(audio.len());
            let b = ((e * sr) as usize).min(audio.len()).max(a);
            &audio[a..b]
        };
        let mut out: Vec<Option<Vec<f32>>> = vec![None; spans.len()];
        let per = spans.len().div_ceil(self.workers).max(1);
        std::thread::scope(|scope| -> Result<()> {
            let handles: Vec<_> = out
                .chunks_mut(per)
                .zip(spans.chunks(per))
                .map(|(dst, src)| {
                    scope.spawn(move || -> Result<()> {
                        for (d, span) in dst.iter_mut().zip(src) {
                            if span.1 - span.0 >= self.min_len {
                                *d = self.embed_one(slice(span))?;
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                h.join().map_err(|_| Error::Diarization("embedding worker panicked".into()))??;
            }
            Ok(())
        })?;
        Ok(out)
    }
}
