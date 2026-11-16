# mo-meetings architecture

Local meeting transcription: recording → WebVTT with speakers, no Python, no cloud. Today a library, a CLI
(milestone M0) and an MCP server; later a Tauri desktop app in the style of [Handy](https://github.com/cjpais/Handy)
on top of the same library.

## Crates

```text
crates/mo-meetings-core   library: audio, diarization, ASR, chunking, assembly, VTT/JSON, models, pipeline
crates/mo-meetings-cli    binary `mo-meetings` (clap): transcribe, diarize, models list|download|path
apps/desktop           (M1) Tauri v2 app using mo-meetings-core directly; src-tauri/ is the crate, Vite + TS front-end
crates/mo-meetings-mcp    binary `mo-meetings-mcp`: MCP server over stdio (rmcp) exposing transcribe/diarize as tools
```

The core has no CLI or UI dependencies; progress is reported through a callback (`pipeline::Progress`), which the
CLI turns into stderr output and the app will turn into Tauri events.

## Pipeline

```text
input (mp4/m4a/wav/…)
  │ audio::load            symphonia decode → downmix → streaming FFT resample → 16 kHz mono f32
  │                        (ffmpeg on PATH as fallback when symphonia cannot open/decode the file)
  ▼
diarize::Diarizer          sherpa-onnx: segmentation-3.0 → TitaNet embeddings → FastClustering (thr 1.05)
  │                        then merge::merge_clusters over TitaNet embeddings of each turn (computed once):
  │                          1. clusters < 1 % of speech → nearest big cluster (cosine to centroid)
  │                          2. merge clusters with centroid cosine > 0.7, repeat
  │                          (--num-speakers N: keep merging the most similar pair until ≤ N)
  ▼ Vec<Turn>
chunk::speech_chunks       union of turns → regions; regions > 29.6 s cut at the quietest 100 ms frame;
  │                        regions joined while ≤ 29.6 s and gap ≤ 3 s; 0.2 s padding into pauses
  ▼ Vec<Chunk> (≤ 30 s)
asr::Asr (per chunk)       whisper.cpp: language, glossary prompt in every chunk, no_context
  │                        (= condition_on_previous_text=false), beam 5, DTW token timestamps
  ▼ Vec<Word> (absolute times)
assembly                   exclusive turns → speaker per word by overlap (fallback nearest turn) →
  │                        Speaker N by first appearance → cues (speaker change, pause > 1 s, > 7 s,
  │                        sentence end after 3 s)
  ▼ Vec<Cue>
vtt::render + Metadata     Teams-shaped VTT (`<v Speaker N>`, HTML entities) and JSON metadata
```

Data between steps are plain structs (`Turn`, `Chunk`, `Word`, `Cue` in `types.rs`), so each step is unit-tested
on synthetic data and steps can be re-run separately (the app will re-run assembly after the user edits
speakers without re-running ASR).

### Interfaces

```rust
pub trait Diarizer { fn diarize(&mut self, audio: &[f32], opts: &DiarizeOptions) -> Result<Diarization>; }
pub trait SpeakerEmbedder { fn embed_spans(&self, audio: &[f32], spans: &[(f64, f64)]) -> Result<Vec<Option<Vec<f32>>>>; }
pub trait Asr { fn transcribe(&mut self, audio: &[f32], opts: &AsrOptions) -> Result<Recognized>; fn model_name(&self) -> String; }

// Steps, callable one by one
pub fn pipeline::diarize(audio, &mut dyn Diarizer, num_speakers) -> Result<Diarization>;
pub fn pipeline::speech_chunks(&[Turn], audio, &ChunkRules) -> Vec<Chunk>;
pub fn pipeline::transcribe_chunks(audio, &[Chunk], &mut dyn Asr, &AsrOptions, sink, cancel: &AtomicBool) -> Result<Vec<ChunkResult>>;
pub fn pipeline::assemble(Vec<Word>, &[Turn], &CueRules) -> Assembled;   // words with speakers, exclusive turns, cues

// The steps chained
pub fn pipeline::transcribe_audio(audio, &mut dyn Diarizer, &mut dyn Asr, &Config, progress, cancel) -> Result<Transcript>;
pub fn pipeline::run(input, &Config, progress) -> Result<Transcript>;   // bundled backends + cached models
pub fn pipeline::run_cancellable(input, &Config, progress, cancel) -> Result<Transcript>;
```

`transcribe_audio` is backend-agnostic; `run` wires sherpa-onnx + whisper.cpp and the model cache.
`transcribe_chunks` takes any subset of chunks and per-call options, so one loaded model can re-transcribe some
chunks differently; `sink` gets each chunk's words (with `Word.prob` and the language used or detected) as soon as
the chunk is done. `cancel` is checked before every chunk and between the steps and ends the run with
`Error::Cancelled`. Errors carry the CLI exit code (`Error::exit_code`: 2 input, 3 model missing, 4 processing).

## Dependency choices (versions checked on crates.io, 2026-11-13)

| Need | Choice | Why / alternatives |
|---|---|---|
| ASR | `whisper-rs` 0.16.0 (whisper.cpp) | Maintained (Codeberg), features `metal`/`cuda`/`vulkan`/`coreml`, exposes token data incl. DTW timestamps, initial prompt, `no_context`. Handy uses whisper.cpp too. |
| Diarization | `sherpa-onnx` 2.0.1 (official k2-fsa crate) | Released 2026-10-27 together with sherpa-onnx itself, safe wrappers for `OfflineSpeakerDiarization` and `SpeakerEmbeddingExtractor`, build script downloads prebuilt libs (static/shared, also for Tauri iOS/Android). `sherpa-rs` 0.6.8 was last released 2025-10 and lags upstream; own FFI over the C API would duplicate the official crate. |
| ONNX Runtime | Microsoft release 1.28.2 on Apple Silicon | The onnxruntime inside the sherpa-onnx release (also 1.28.2) ran diarization 2.3× slower (37 min: 288 s vs 124 s; embeddings 22 s vs 7 s). On macOS sherpa-onnx is linked as shared libraries and `mo-meetings-core/build.rs` fetches Microsoft's build (pinned SHA-256), sets its install name and puts it first in the rpath. Linux uses static linking. |
| Decoding | `symphonia` 0.6.1 (`isomp4`, `aac`, `wav`, `mp3`, `flac`, …) | Pure Rust; AAC-LC in Teams MP4 (16 kHz mono) decodes fine. `ffmpeg` fallback only when on PATH. |
| Resampling | `rubato` 5.0.1 (`Fft`) | Synchronous FFT resampler, exact for 48 k/44.1 k → 16 k, streamed so a 1 h recording never sits in memory at the source rate. |
| CLI | `clap` 4.6, `indicatif` 0.18 | |
| Downloads | `ureq` 3.5 (rustls), `sha2` 0.11, `dirs` 7 | Small blocking client, no async runtime in the core. |
| MCP | `rmcp` 3.10 (`transport-io`), `tokio` | Official Rust SDK; `#[tool]` macros with JSON Schema from `schemars`, stdio transport, progress notifications. |
| Later | `tauri` 2.12 | Desktop shell (M1). |

### Embeddings once

The sherpa-onnx C API does not return the embeddings its clustering used. The pipeline therefore runs sherpa's
diarization (segmentation + embeddings + clustering) and then computes one TitaNet embedding per resulting turn
(≥ 0.5 s, in parallel, ~7 s per 37 min of audio); both merge steps reuse these. Doing segmentation and clustering
ourselves (sherpa only for the two ONNX models) would save sherpa's internal embedding pass and is an option if
diarization time matters more later.

## Models

Downloaded by `mo-meetings models download` into the user cache (`dirs::cache_dir()/mo-meetings/models`, override
`MO_MEETINGS_MODELS_DIR`), verified by SHA-256, never in the repo, none gated. Registry in `models.rs`, licenses in
`NOTICE`: whisper large-v3-turbo (default) / large-v3 (MIT), pyannote segmentation-3.0 ONNX (MIT, © CNRS),
NeMo TitaNet-Large ONNX (CC-BY-4.0, NVIDIA).

## Settings profiles

`profile.rs` loads `<name>.toml` presets from `<config>/profiles` and the `profile_dirs` of `<config>/config.toml`
(`dirs::config_dir()/mo-meetings`, override `MO_MEETINGS_CONFIG_DIR`); a later directory wins on the same name.
`Profiles::resolve` follows `extends` (cycle check), overrides scalars, merges glossaries and reads `prompt_file`;
`Resolved::apply` writes the result into a `pipeline::Config`, and callers apply explicit options after it.
`Profiles::suggest` matches the recording's file name against `match` globs. After whisper loads, `run` checks the
prompt against whisper.cpp's prompt window (`n_text_ctx / 2 - 1` tokens) and sends `Progress::Warning` with the
start that would be dropped.

## MCP server

`mo-meetings-mcp` speaks MCP over stdio (stdout is the protocol, logs go to stderr). Tools:

- `transcribe(path, profile?, language?, prompt?, num_speakers?, output?)`: runs `pipeline::run` on a blocking thread,
  writes the VTT and the JSON metadata (default next to the input) and returns their paths with duration, speakers
  and cue count (also as structured content), plus `Progress::Warning` messages as `warnings`.
- `profiles(path?)`: the settings profiles, their files, and which match the recording's file name.
- `diarize(path, num_speakers?, output?)`: writes the same JSON as `mo-meetings diarize` (default
  `<input>.turns.json`) and returns its path.

When the request carries a progress token, `pipeline::Progress` is forwarded as `notifications/progress` on a
0-100 scale (load 0-5, decode 5-10, diarize 10-40, ASR 40-95 by audio seconds, assembly 95-100). Errors come back
as tool results with `isError`; a missing model names `mo-meetings models download`.

## Desktop app

`apps/desktop` (Vite + vanilla TypeScript front-end, `src-tauri/` in the Cargo workspace) uses the core directly:

- `run.rs` calls `pipeline::run` on a worker thread and forwards `Progress` as events; the front-end shows stages
  and chunk progress, then plays the recording (HTML `<video>` from an asset URL) next to the synced transcript,
  click-to-seek. `export.rs` writes the last run's VTT and JSON.
- `models.rs` is the first-run setup: `model_setup` lists the models a run with the chosen whisper model needs
  (turbo or large-v3, plus the diarization models), their licenses and `NOTICE`; `download_models` fetches the
  missing ones on a worker thread with `models::download` (SHA-256 checked, three attempts per model, each resuming
  the `.part` file with an HTTP range request) and reports `download-progress` events.
- sherpa-onnx/onnxruntime dylibs go to `Contents/Frameworks` (the CLI already has that rpath).

## Milestones

Recommended order of the open issues: [`roadmap.md`](roadmap.md).

- **M0 (this)**: workspace, library, CLI through the whole pipeline, unit tests, CI (fmt, clippy, tests), quality
  check against Teams VTT.
- **M1 Tauri shell**: open a recording, run the pipeline with progress, play it with the synced transcript, export
  VTT/JSON; first-run model download.
- **M1.5 projects**: per-meeting SQLite project file, settings profiles, extra passes and comparison, segment
  re-transcription, Markdown export for agents.
- **M2 speakers**: rename speakers (name ↔ `Speaker N`), merge/split speakers, reassign a cue; re-run assembly
  only. Optional voice profiles (TitaNet embeddings) to suggest names across meetings.
- **M3 scenes**: scene detection on the video (screen-share changes), key frames next to the transcript, export
  of a condensed transcript with frames.
