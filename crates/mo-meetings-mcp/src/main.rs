//! `mo-meetings-mcp`: MCP server over stdio exposing `transcribe` and `diarize` from `mo-meetings-core`.
//!
//! stdout carries the MCP protocol; logs go to stderr. Long runs report progress through MCP progress
//! notifications when the client sends a progress token.

use mo_meetings_core::pipeline::{self, Config, Progress, Stage};
use mo_meetings_core::{Error, vtt};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{
    Implementation, ProgressNotificationParam, ProgressToken, RequestMetaObject, ServerCapabilities, ServerConfig,
};
use rmcp::{Json, Peer, RoleServer, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::sync::mpsc;

#[derive(Deserialize, JsonSchema)]
struct TranscribeParams {
    /// Recording to transcribe (mp4, m4a, wav, mp3, …). Prefer an absolute path; relative paths are resolved
    /// against the server's working directory.
    path: String,
    /// ISO 639-1 language code (e.g. `cs`, `en`) or `auto`. Default `cs`.
    #[serde(default)]
    language: Option<String>,
    /// Glossary / context prompt (domain terms, names) given to whisper with every chunk.
    #[serde(default)]
    prompt: Option<String>,
    /// At most this many speakers (by default the number is estimated).
    #[serde(default)]
    #[schemars(range(min = 1, max = 64))]
    num_speakers: Option<u32>,
    /// Output VTT path. Default: the input with a `.vtt` extension. The JSON metadata goes next to it (`.json`).
    #[serde(default)]
    output: Option<String>,
}

#[derive(Serialize, JsonSchema)]
struct TranscribeResult {
    /// WebVTT with `<v Speaker N>` voice tags.
    vtt_path: String,
    /// JSON metadata (model, language, speakers, timings).
    json_path: String,
    audio_duration_s: f64,
    num_speakers: usize,
    cues: usize,
    processing_s: f64,
}

#[derive(Deserialize, JsonSchema)]
struct DiarizeParams {
    /// Recording to diarize. Prefer an absolute path.
    path: String,
    /// At most this many speakers (by default the number is estimated).
    #[serde(default)]
    #[schemars(range(min = 1, max = 64))]
    num_speakers: Option<u32>,
    /// Output JSON path. Default: the input with a `.turns.json` extension.
    #[serde(default)]
    output: Option<String>,
}

#[derive(Serialize, JsonSchema)]
struct DiarizeResult {
    /// JSON with `turns` and non-overlapping `exclusive` turns as `[start_s, end_s, "SPEAKER_NN"]`.
    json_path: String,
    audio_duration_s: f64,
    num_speakers: usize,
    processing_s: f64,
}

#[derive(Clone)]
struct MeetingsServer {
    #[expect(dead_code, reason = "read by the code that #[tool_handler] generates")]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MeetingsServer {
    fn new() -> Self {
        Self { tool_router: Self::tool_router() }
    }

    /// Transcribe a meeting recording locally (diarization + whisper) into WebVTT with `Speaker N` labels.
    #[tool(description = "Transcribe a meeting recording on this machine into WebVTT with speaker labels \
                       (`<v Speaker N>`) plus JSON metadata. Writes both files and returns their paths. Takes \
                       minutes for long recordings; progress is reported.")]
    async fn transcribe(
        &self,
        Parameters(p): Parameters<TranscribeParams>,
        meta: RequestMetaObject,
        peer: Peer<RoleServer>,
    ) -> Result<Json<TranscribeResult>, String> {
        let input = absolute(&p.path)?;
        let output = match &p.output {
            Some(o) => absolute(o)?,
            None => input.with_extension("vtt"),
        };
        let json_path = output.with_extension("json");
        if output == input || json_path == input {
            return Err(message(Error::InvalidInput("output would overwrite the input".into())));
        }
        if output.extension().is_some_and(|e| e.eq_ignore_ascii_case("json")) {
            return Err(message(Error::InvalidInput("output must not have a .json extension".into())));
        }
        let mut cfg = Config::default();
        cfg.asr.language = parse_language(p.language.as_deref())?;
        cfg.asr.prompt = p.prompt.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        cfg.num_speakers = p.num_speakers.map(|n| n as usize);

        let reporter = ProgressForwarder::spawn(meta.get_progress_token(), peer, progress_point);
        let tx = reporter.sender();
        let result = tokio::task::spawn_blocking(move || {
            let tr = pipeline::run(&input, &cfg, &mut |pr| {
                if let Some(tx) = &tx {
                    let _ = tx.send(pr);
                }
            })?;
            std::fs::write(&output, vtt::render(&tr.cues))?;
            let meta = serde_json::to_string_pretty(&tr.meta).map_err(|e| Error::InvalidInput(e.to_string()))?;
            std::fs::write(&json_path, meta + "\n")?;
            Ok::<_, Error>(TranscribeResult {
                vtt_path: output.display().to_string(),
                json_path: json_path.display().to_string(),
                audio_duration_s: tr.meta.audio_duration_s,
                num_speakers: tr.meta.num_speakers,
                cues: tr.meta.cues,
                processing_s: tr.meta.timings.total_s,
            })
        })
        .await;
        reporter.finish().await;
        Ok(Json(result.map_err(|e| format!("transcription task failed: {e}"))?.map_err(message)?))
    }

    /// Speaker diarization only: who spoke when.
    #[tool(description = "Speaker diarization of a recording on this machine (who spoke when). Writes speaker \
                       turns as JSON and returns its path.")]
    async fn diarize(
        &self,
        Parameters(p): Parameters<DiarizeParams>,
        meta: RequestMetaObject,
        peer: Peer<RoleServer>,
    ) -> Result<Json<DiarizeResult>, String> {
        let input = absolute(&p.path)?;
        let output = match &p.output {
            Some(o) => absolute(o)?,
            None => input.with_extension("turns.json"),
        };
        if output == input {
            return Err(message(Error::InvalidInput("output would overwrite the input".into())));
        }
        let cfg = Config { num_speakers: p.num_speakers.map(|n| n as usize), ..Default::default() };

        // Diarization has no finer progress than start/finish.
        let reporter = ProgressForwarder::spawn(meta.get_progress_token(), peer, diarize_progress_point);
        let tx = reporter.sender();
        let result = tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            if let Some(tx) = &tx {
                let _ = tx.send(Progress::Started(Stage::Diarize));
            }
            let (d, duration) = pipeline::run_diarization(&input, &cfg)?;
            let total_s = start.elapsed().as_secs_f64();
            if let Some(tx) = &tx {
                let _ = tx.send(Progress::Finished(Stage::Diarize, total_s));
            }
            let v = pipeline::diarization_json(&d, duration, total_s);
            let text = serde_json::to_string_pretty(&v).map_err(|e| Error::InvalidInput(e.to_string()))?;
            std::fs::write(&output, text + "\n")?;
            Ok::<_, Error>(DiarizeResult {
                json_path: output.display().to_string(),
                audio_duration_s: duration,
                num_speakers: d.num_speakers,
                processing_s: total_s,
            })
        })
        .await;
        reporter.finish().await;
        Ok(Json(result.map_err(|e| format!("diarization task failed: {e}"))?.map_err(message)?))
    }
}

#[tool_handler]
impl ServerHandler for MeetingsServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("mo-meetings-mcp", env!("CARGO_PKG_VERSION")).with_title("mo-meetings"),
            )
            .with_instructions(
                "Local meeting transcription with speaker diarization. `transcribe` writes a WebVTT (+ JSON \
                 metadata) next to the recording and returns the paths; read the VTT for the text. Runs take \
                 roughly 5-15 % of the recording length on Apple Silicon. Models must be downloaded once with \
                 `mo-meetings models download`.",
            )
    }
}

/// Error text for the tool result. Missing models already say `run mo-meetings models download`.
fn message(e: Error) -> String {
    e.to_string()
}

fn absolute(p: &str) -> Result<PathBuf, String> {
    let p = Path::new(p.trim());
    if p.as_os_str().is_empty() {
        return Err(message(Error::InvalidInput("empty path".into())));
    }
    std::path::absolute(p).map_err(|e| message(Error::InvalidInput(format!("bad path {}: {e}", p.display()))))
}

fn parse_language(l: Option<&str>) -> Result<Option<String>, String> {
    match l.map(str::trim) {
        None | Some("") => Ok(Config::default().asr.language),
        Some("auto") => Ok(None),
        Some(l) if l.len() == 2 && l.chars().all(|c| c.is_ascii_lowercase()) => Ok(Some(l.to_string())),
        Some(l) => Err(message(Error::InvalidInput(format!("invalid language `{l}` (use a 2-letter code or `auto`)")))),
    }
}

/// Share of the 0-100 progress scale per stage (diarization and ASR dominate).
fn stage_range(s: Stage) -> (f64, f64) {
    match s {
        Stage::Load => (0.0, 5.0),
        Stage::Decode => (5.0, 10.0),
        Stage::Diarize => (10.0, 40.0),
        Stage::Transcribe => (40.0, 95.0),
        Stage::Assemble => (95.0, 100.0),
    }
}

fn stage_name(s: Stage) -> &'static str {
    match s {
        Stage::Load => "loading models",
        Stage::Decode => "decoding audio",
        Stage::Diarize => "diarizing",
        Stage::Transcribe => "transcribing",
        Stage::Assemble => "assembling",
    }
}

/// Overall progress on a 0-100 scale, monotonic over a pipeline run, with a short message.
fn progress_point(p: &Progress) -> (f64, String) {
    match *p {
        Progress::Started(s) => (stage_range(s).0, format!("{}…", stage_name(s))),
        Progress::Finished(s, secs) => (stage_range(s).1, format!("{} done in {secs:.1} s", stage_name(s))),
        Progress::Chunk { done, total, audio_done, audio_total } => {
            let frac = if audio_total > 0.0 { (audio_done / audio_total).clamp(0.0, 1.0) } else { 1.0 };
            let (a, b) = stage_range(Stage::Transcribe);
            (a + (b - a) * frac, format!("transcribing: {done}/{total} chunks ({audio_done:.0}/{audio_total:.0} s)"))
        }
    }
}

/// Progress of the `diarize` tool, which only reports the start and end of diarization.
fn diarize_progress_point(p: &Progress) -> (f64, String) {
    match *p {
        Progress::Finished(_, secs) => (100.0, format!("diarizing done in {secs:.1} s")),
        _ => (0.0, "diarizing…".into()),
    }
}

/// Forwards `pipeline::Progress` from the blocking worker to the client as MCP progress notifications.
struct ProgressForwarder {
    tx: Option<mpsc::UnboundedSender<Progress>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl ProgressForwarder {
    fn spawn(token: Option<ProgressToken>, peer: Peer<RoleServer>, scale: fn(&Progress) -> (f64, String)) -> Self {
        let Some(token) = token else { return Self { tx: None, task: None } };
        let (tx, mut rx) = mpsc::unbounded_channel::<Progress>();
        let task = tokio::spawn(async move {
            let mut last = -1.0;
            while let Some(p) = rx.recv().await {
                let (value, msg) = scale(&p);
                if value <= last {
                    continue; // the spec requires increasing progress
                }
                last = value;
                let param = ProgressNotificationParam::new(token.clone(), value).with_total(100.0).with_message(msg);
                if peer.notify_progress(param).await.is_err() {
                    break;
                }
            }
        });
        Self { tx: Some(tx), task: Some(task) }
    }

    fn sender(&self) -> Option<mpsc::UnboundedSender<Progress>> {
        self.tx.clone()
    }

    /// Waits until queued notifications are sent, so none arrive after the tool result.
    async fn finish(mut self) {
        self.tx.take();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).is_some_and(|a| a == "--version" || a == "-V") {
        println!("mo-meetings-mcp {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let service = MeetingsServer::new().serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_is_monotonic_over_a_run() {
        let run = [
            Progress::Started(Stage::Load),
            Progress::Finished(Stage::Load, 1.0),
            Progress::Started(Stage::Decode),
            Progress::Finished(Stage::Decode, 1.0),
            Progress::Started(Stage::Diarize),
            Progress::Finished(Stage::Diarize, 1.0),
            Progress::Started(Stage::Transcribe),
            Progress::Chunk { done: 1, total: 2, audio_done: 10.0, audio_total: 30.0 },
            Progress::Chunk { done: 2, total: 2, audio_done: 30.0, audio_total: 30.0 },
            Progress::Finished(Stage::Transcribe, 1.0),
            Progress::Started(Stage::Assemble),
            Progress::Finished(Stage::Assemble, 1.0),
        ];
        let values: Vec<f64> = run.iter().map(|p| progress_point(p).0).collect();
        assert!(values.windows(2).all(|w| w[0] <= w[1]), "{values:?}");
        assert_eq!(values.first(), Some(&0.0));
        assert_eq!(values.last(), Some(&100.0));
    }

    #[test]
    fn language_defaults_and_validation() {
        assert_eq!(parse_language(None).unwrap(), Some("cs".into()));
        assert_eq!(parse_language(Some("auto")).unwrap(), None);
        assert_eq!(parse_language(Some("en")).unwrap(), Some("en".into()));
        assert!(parse_language(Some("english")).is_err());
    }

    #[test]
    fn missing_model_error_points_to_download() {
        let e = Error::ModelMissing { id: "titanet-large".into(), path: "/x".into() };
        assert!(message(e).contains("mo-meetings models download"));
    }
}
