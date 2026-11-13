//! whisper.cpp via `whisper-rs` (Metal on macOS with the `metal` feature, CPU elsewhere).

use super::{Asr, AsrOptions};
use crate::{Error, Result, Word};
use std::path::Path;
use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
    WhisperState,
};

#[derive(Clone, Debug)]
pub struct WhisperParams {
    pub threads: usize,
    pub beam_size: usize,
    /// Word times from DTW over cross-attention (more precise than whisper.cpp token timestamps).
    pub dtw: Option<DtwModelPreset>,
    pub use_gpu: bool,
}

impl Default for WhisperParams {
    fn default() -> Self {
        Self { threads: 4, beam_size: 5, dtw: Some(DtwModelPreset::LargeV3Turbo), use_gpu: true }
    }
}

pub struct WhisperAsr {
    name: String,
    params: WhisperParams,
    eot: i32,
    state: WhisperState,
    // keep the context alive as long as the state
    _ctx: WhisperContext,
}

impl WhisperAsr {
    pub fn new(model: &Path, name: impl Into<String>, params: WhisperParams) -> Result<Self> {
        whisper_rs::install_logging_hooks();
        let mut cp = WhisperContextParameters::default();
        cp.use_gpu(params.use_gpu);
        cp.flash_attn(false); // DTW does not work with flash attention
        if let Some(preset) = params.dtw.clone() {
            cp.dtw_parameters(DtwParameters {
                mode: DtwMode::ModelPreset { model_preset: preset },
                ..Default::default()
            });
        }
        let ctx = WhisperContext::new_with_params(model, cp)
            .map_err(|e| Error::Asr(format!("cannot load {}: {e}", model.display())))?;
        let state = ctx.create_state().map_err(|e| Error::Asr(format!("cannot create whisper state: {e}")))?;
        let eot = ctx.token_eot();
        Ok(Self { name: name.into(), params, eot, state, _ctx: ctx })
    }
}

/// One decoded token: raw bytes (may split UTF-8 characters) and times in seconds.
#[derive(Clone, Debug)]
pub(crate) struct Tok {
    pub bytes: Vec<u8>,
    pub t0: f64,
    pub t1: f64,
    pub dtw: Option<f64>,
}

/// Join BPE tokens of one segment into words. A token starting with a space starts a new word.
/// With DTW times a word spans from its first token's DTW time to the next word's start (capped by the
/// token-timestamp end of its last token), otherwise it uses token timestamps.
pub(crate) fn words_from_tokens(toks: &[Tok], seg_end: f64) -> Vec<Word> {
    let mut groups: Vec<Vec<&Tok>> = Vec::new();
    for t in toks {
        if t.bytes.is_empty() {
            continue;
        }
        if groups.is_empty() || t.bytes[0] == b' ' {
            groups.push(Vec::new());
        }
        groups.last_mut().expect("pushed").push(t);
    }
    let starts: Vec<f64> = groups.iter().map(|g| g[0].dtw.unwrap_or(g[0].t0)).collect();
    let mut words = Vec::with_capacity(groups.len());
    for (i, g) in groups.iter().enumerate() {
        let text =
            String::from_utf8_lossy(&g.iter().flat_map(|t| t.bytes.iter().copied()).collect::<Vec<u8>>()).into_owned();
        if text.trim().is_empty() {
            continue;
        }
        let start = starts[i];
        let last = g[g.len() - 1];
        let end = if g[0].dtw.is_some() {
            let next = starts.get(i + 1).copied().unwrap_or(seg_end);
            next.min(last.t1.max(start + 0.1)).max(start + 0.02)
        } else {
            last.t1.max(start + 0.02)
        };
        words.push(Word::new(text, start, end));
    }
    words
}

impl Asr for WhisperAsr {
    fn transcribe(&mut self, audio: &[f32], opts: &AsrOptions) -> Result<Vec<Word>> {
        let mut fp =
            FullParams::new(SamplingStrategy::BeamSearch { beam_size: self.params.beam_size as i32, patience: -1.0 });
        fp.set_n_threads(self.params.threads as i32);
        fp.set_language(opts.language.as_deref());
        if let Some(p) = opts.prompt.as_deref().filter(|p| !p.trim().is_empty()) {
            fp.set_initial_prompt(p);
        }
        fp.set_no_context(true); // = condition_on_previous_text=False
        fp.set_token_timestamps(true);
        fp.set_print_progress(false);
        fp.set_print_realtime(false);
        fp.set_print_special(false);
        fp.set_print_timestamps(false);
        fp.set_suppress_blank(true);
        self.state.full(fp, audio).map_err(|e| Error::Asr(format!("whisper failed: {e}")))?;
        let mut words = Vec::new();
        for seg in self.state.as_iter() {
            let seg_end = seg.end_timestamp() as f64 / 100.0;
            let mut toks = Vec::new();
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else { continue };
                let d = tok.token_data();
                if d.id >= self.eot {
                    continue; // special and timestamp tokens
                }
                toks.push(Tok {
                    bytes: tok.to_bytes().map(<[u8]>::to_vec).unwrap_or_default(),
                    t0: d.t0 as f64 / 100.0,
                    t1: d.t1 as f64 / 100.0,
                    dtw: (d.t_dtw >= 0 && self.params.dtw.is_some()).then(|| d.t_dtw as f64 / 100.0),
                });
            }
            words.extend(words_from_tokens(&toks, seg_end));
        }
        Ok(words)
    }

    fn model_name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: &[u8], t0: f64, t1: f64, dtw: Option<f64>) -> Tok {
        Tok { bytes: s.to_vec(), t0, t1, dtw }
    }

    #[test]
    fn joins_subword_tokens_and_split_utf8() {
        // "č" = 0xC4 0x8D split across two tokens
        let toks = [
            tok(b" Dob", 0.0, 0.2, None),
            tok(b"r\xC4", 0.2, 0.3, None),
            tok(b"\x8De", 0.3, 0.5, None),
            tok(b".", 0.5, 0.5, None),
            tok(b" Ano", 0.8, 1.0, None),
        ];
        let w = words_from_tokens(&toks, 1.0);
        assert_eq!(w.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), [" Dobrče.", " Ano"]);
        assert_eq!((w[0].start, w[0].end), (0.0, 0.5));
        assert_eq!((w[1].start, w[1].end), (0.8, 1.0));
    }

    #[test]
    fn dtw_times_end_at_next_word_or_token_end() {
        let toks = [tok(b" a", 0.0, 0.5, Some(0.1)), tok(b" b", 0.5, 3.0, Some(0.4)), tok(b" c", 3.0, 3.2, Some(2.9))];
        let w = words_from_tokens(&toks, 3.5);
        assert_eq!((w[0].start, w[0].end), (0.1, 0.4));
        assert_eq!((w[1].start, w[1].end), (0.4, 2.9));
        assert_eq!((w[2].start, w[2].end), (2.9, 3.2));
    }

    #[test]
    fn skips_empty_words() {
        let w = words_from_tokens(&[tok(b" ", 0.0, 0.1, None), tok(b" x", 0.1, 0.2, None)], 1.0);
        assert_eq!(w.len(), 1);
    }
}
