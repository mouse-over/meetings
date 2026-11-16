//! `mo-meetings-mcp`: MCP server over stdio exposing `transcribe` and `diarize` from `mo-meetings-core`.
//!
//! stdout carries the MCP protocol; logs go to stderr. Long runs report progress through MCP progress
//! notifications when the client sends a progress token.

use mo_meetings_core::pipeline::{self, Config, Progress, Stage};
use mo_meetings_core::profile::Profiles;
use mo_meetings_core::{Error, asr, vtt};
use mo_meetings_project::{Error as ProjectError, Project};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{
    Implementation, ProgressNotificationParam, ProgressToken, RequestMetaObject, ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{Json, Peer, RoleServer, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::mpsc;

#[derive(Deserialize, JsonSchema)]
struct TranscribeParams {
    /// Recording to transcribe (mp4, m4a, wav, mp3, …). Prefer an absolute path; relative paths are resolved
    /// against the server's working directory. Optional with an existing `project` (give it when the recording
    /// moved).
    #[serde(default)]
    path: Option<String>,
    /// Project file (e.g. `meeting.mom`, SQLite): created for the recording when missing. Stores the run, and a
    /// run that was interrupted or cancelled with the same settings continues from the next chunk.
    #[serde(default)]
    project: Option<String>,
    /// Settings profile (see the `profiles` tool). The other parameters override its values.
    #[serde(default)]
    profile: Option<String>,
    /// ISO 639-1 language code (e.g. `cs`, `en`) or `auto`. Default: the profile's, else `cs`.
    #[serde(default)]
    language: Option<String>,
    /// Glossary / context prompt (domain terms, names) given to whisper with every chunk. Replaces the
    /// profile's prompt and glossary.
    #[serde(default)]
    prompt: Option<String>,
    /// At most this many speakers (by default the number is estimated).
    #[serde(default)]
    #[schemars(range(min = 1, max = 64))]
    num_speakers: Option<u32>,
    /// Output VTT path. Default: the recording with a `.vtt` extension. The JSON metadata goes next to it (`.json`).
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
    /// The settings profile used.
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<String>,
    /// The project file the run was saved in.
    #[serde(skip_serializing_if = "Option::is_none")]
    project_path: Option<String>,
    /// The run's id in the project.
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<i64>,
    /// Chunks taken over from an interrupted run of the project (0 for a new run).
    #[serde(skip_serializing_if = "Option::is_none")]
    resumed_chunks: Option<usize>,
    /// Problems that did not stop the run (e.g. a prompt longer than whisper keeps).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct ProfilesParams {
    /// A recording: profiles whose file name patterns match it are marked `matches`.
    #[serde(default)]
    path: Option<String>,
}

#[derive(Serialize, JsonSchema)]
struct ProfileEntry {
    name: String,
    description: Option<String>,
    extends: Option<String>,
    /// Profile file.
    source: String,
    /// The recording's file name matches the profile's patterns.
    matches: bool,
}

#[derive(Serialize, JsonSchema)]
struct ProfilesResult {
    profiles: Vec<ProfileEntry>,
    /// Directories searched, later wins on the same name.
    dirs: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
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
                       minutes for long recordings; progress is reported. With `project`, the run is saved in \
                       that project file and an interrupted or cancelled run continues where it stopped.")]
    async fn transcribe(
        &self,
        Parameters(p): Parameters<TranscribeParams>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<Json<TranscribeResult>, String> {
        let input = p.path.as_deref().map(absolute).transpose()?;
        let project = p.project.as_deref().map(absolute).transpose()?;
        let output = p.output.as_deref().map(absolute).transpose()?;
        if output.as_ref().is_some_and(|o| o.extension().is_some_and(|e| e.eq_ignore_ascii_case("json"))) {
            return Err(message(Error::InvalidInput("output must not have a .json extension".into())));
        }
        let mut cfg = Config::default();
        let mut warnings = Vec::new();
        if let Some(name) = p.profile.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
            let profiles = Profiles::load().map_err(message)?;
            warnings.extend(profiles.warnings.iter().cloned());
            profiles.resolve(name).and_then(|r| r.apply(&mut cfg)).map_err(message)?;
        }
        if let Some(l) = p.language.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
            cfg.asr.language = asr::parse_language(l).map_err(message)?;
        }
        if let Some(prompt) = p.prompt {
            cfg.asr.prompt = Some(prompt.trim().to_string()).filter(|s| !s.is_empty());
        }
        if let Some(n) = p.num_speakers {
            cfg.num_speakers = Some(n as usize);
        }

        // A cancelled request stops the run after the chunk in progress; in a project it stays resumable.
        let cancel = Arc::new(AtomicBool::new(false));
        let watcher = {
            let (ct, cancel) = (ctx.ct.clone(), cancel.clone());
            tokio::spawn(async move {
                ct.cancelled().await;
                cancel.store(true, Ordering::Relaxed);
            })
        };
        let reporter = ProgressForwarder::spawn(ctx.meta.get_progress_token(), ctx.peer, progress_point);
        let tx = reporter.sender();
        let result = tokio::task::spawn_blocking(move || {
            let mut on_progress = |pr| match pr {
                Progress::Warning(w) => warnings.push(w),
                pr => {
                    if let Some(tx) = &tx {
                        let _ = tx.send(pr);
                    }
                }
            };
            let mut project = match &project {
                Some(path) => Some(Project::open_or_create(path, input.as_deref()).map_err(project_message)?.0),
                None => None,
            };
            let input = match (&project, input) {
                (Some(p), _) => p.recording().map_err(project_message)?,
                (None, Some(input)) => input,
                (None, None) => {
                    return Err(message(Error::InvalidInput("give the recording (`path`) or a `project`".into())));
                }
            };
            let output = output.unwrap_or_else(|| input.with_extension("vtt"));
            let json_path = output.with_extension("json");
            if output == input || json_path == input {
                return Err(message(Error::InvalidInput("output would overwrite the input".into())));
            }
            let (tr, saved) = match &mut project {
                Some(p) => {
                    let out = p.transcribe(&cfg, &mut on_progress, &cancel).map_err(project_message)?;
                    let saved = (p.path().display().to_string(), out.run_id, out.resumed_chunks);
                    (out.transcript, Some(saved))
                }
                None => (pipeline::run_cancellable(&input, &cfg, &mut on_progress, &cancel).map_err(message)?, None),
            };
            let write = |path: &Path, text: String| std::fs::write(path, text).map_err(|e| message(Error::Io(e)));
            write(&output, vtt::render(&tr.cues))?;
            let meta = serde_json::to_string_pretty(&tr.meta).map_err(|e| e.to_string())?;
            write(&json_path, meta + "\n")?;
            let (project_path, run_id, resumed_chunks) = match saved {
                Some((path, id, resumed)) => (Some(path), Some(id), Some(resumed)),
                None => (None, None, None),
            };
            Ok(TranscribeResult {
                vtt_path: output.display().to_string(),
                json_path: json_path.display().to_string(),
                audio_duration_s: tr.meta.audio_duration_s,
                num_speakers: tr.meta.num_speakers,
                cues: tr.meta.cues,
                processing_s: tr.meta.timings.total_s,
                profile: tr.meta.profile,
                project_path,
                run_id,
                resumed_chunks,
                warnings,
            })
        })
        .await;
        watcher.abort();
        reporter.finish().await;
        Ok(Json(result.map_err(|e| format!("transcription task failed: {e}"))??))
    }

    /// Settings profiles available to `transcribe`.
    #[tool(description = "List the settings profiles (named presets of language, prompt/glossary, model, number \
                       of speakers) that `transcribe` accepts as `profile`. Give `path` to see which profiles \
                       match a recording's file name.")]
    async fn profiles(&self, Parameters(p): Parameters<ProfilesParams>) -> Result<Json<ProfilesResult>, String> {
        let profiles = Profiles::load().map_err(message)?;
        let matching = match &p.path {
            Some(path) => profiles.suggest(&absolute(path)?).into_iter().map(str::to_string).collect(),
            None => Vec::new(),
        };
        Ok(Json(ProfilesResult {
            profiles: profiles
                .list()
                .map(|i| ProfileEntry {
                    name: i.name.clone(),
                    description: i.description.clone(),
                    extends: i.extends.clone(),
                    source: i.path.display().to_string(),
                    matches: matching.contains(&i.name),
                })
                .collect(),
            dirs: profiles.dirs.iter().map(|d| d.display().to_string()).collect(),
            warnings: profiles.warnings,
        }))
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
                 metadata) next to the recording and returns the paths; read the VTT for the text. Give `project` \
                 (a `.mom` file) to keep the run and resume it after an interruption. `profiles` \
                 lists settings presets; pass one as `profile` when it matches the recording. Runs take \
                 roughly 5-15 % of the recording length on Apple Silicon. Models must be downloaded once with \
                 `mo-meetings models download`.",
            )
    }
}

/// Error text for the tool result. Missing models already say `run mo-meetings models download`.
fn message(e: Error) -> String {
    e.to_string()
}

/// Error text for a project error; a recording that moved is given as `path`, not on the command line.
fn project_message(e: ProjectError) -> String {
    match e {
        ProjectError::RecordingMissing(p) => {
            format!("recording not found at {} (moved? give its new path as `path` with the project)", p.display())
        }
        e => e.to_string(),
    }
}

fn absolute(p: &str) -> Result<PathBuf, String> {
    let p = Path::new(p.trim());
    if p.as_os_str().is_empty() {
        return Err(message(Error::InvalidInput("empty path".into())));
    }
    std::path::absolute(p).map_err(|e| message(Error::InvalidInput(format!("bad path {}: {e}", p.display()))))
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
        Progress::Warning(ref w) => (0.0, format!("warning: {w}")),
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
    fn missing_model_error_points_to_download() {
        let e = Error::ModelMissing { id: "titanet-large".into(), path: "/x".into() };
        assert!(message(e).contains("mo-meetings models download"));
    }
}
