//! Runs `pipeline::run` on a worker thread and reports to the front-end through events:
//! `run-progress` ([`ProgressEvent`]), then either `run-finished` ([`RunResult`]) or `run-failed` ([`RunError`]).

use crate::export::{Finished, LastRun};
use mo_meetings_core::pipeline::{self, Config, Progress, Stage};
use mo_meetings_core::{Cue, Error};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};

/// Whether a run is in progress; one run at a time (the models take several GB of memory).
#[derive(Default)]
pub struct Running(Arc<AtomicBool>);

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunOptions {
    /// Registry id of the whisper model (`whisper-large-v3-turbo` or `whisper-large-v3`).
    asr_model: String,
    /// ISO 639-1 code (`cs`, `en`, …) or `auto`.
    language: String,
    /// Glossary prompt given to whisper with every chunk.
    prompt: String,
    /// At most this many speakers; estimated when absent.
    num_speakers: Option<u32>,
}

#[derive(Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ProgressEvent {
    Started {
        stage: Stage,
    },
    Chunk {
        done: usize,
        total: usize,
        #[serde(rename = "audioDone")]
        audio_done: f64,
        #[serde(rename = "audioTotal")]
        audio_total: f64,
    },
    Finished {
        stage: Stage,
        seconds: f64,
    },
}

impl From<Progress> for ProgressEvent {
    fn from(p: Progress) -> Self {
        match p {
            Progress::Started(stage) => Self::Started { stage },
            Progress::Chunk { done, total, audio_done, audio_total } => {
                Self::Chunk { done, total, audio_done, audio_total }
            }
            Progress::Finished(stage, seconds) => Self::Finished { stage, seconds },
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunResult {
    audio_duration_s: f64,
    num_speakers: usize,
    processing_s: f64,
    cues: Vec<Cue>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ErrorKind {
    /// The recording or the options: the user can fix it by picking another file or changing an option.
    Input,
    /// Models are not downloaded yet.
    ModelMissing,
    /// The pipeline failed on a valid input.
    Processing,
    /// A run (or a download) is already in progress.
    Busy,
    /// Downloading the models failed.
    Download,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunError {
    pub(crate) kind: ErrorKind,
    message: String,
}

impl RunError {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl From<Error> for RunError {
    fn from(e: Error) -> Self {
        match e {
            Error::InvalidInput(m) => Self::new(ErrorKind::Input, format!("This recording can't be used: {m}.")),
            Error::Audio(m) => Self::new(
                ErrorKind::Input,
                format!("No audio could be read from this file. Pick a video or audio recording.\n\nDetails: {m}"),
            ),
            Error::ModelMissing { id, path } => Self::new(
                ErrorKind::ModelMissing,
                format!(
                    "The model `{id}` is not downloaded (expected at {}). Download it below and try again.",
                    path.display()
                ),
            ),
            e => Self::new(ErrorKind::Processing, format!("Processing failed: {e}")),
        }
    }
}

fn parse_language(l: &str) -> Result<Option<String>, RunError> {
    match l.trim() {
        "auto" => Ok(None),
        l if l.len() == 2 && l.chars().all(|c| c.is_ascii_lowercase()) => Ok(Some(l.to_string())),
        l => Err(RunError::new(ErrorKind::Input, format!("Unknown language `{l}`: use a 2-letter code or `auto`."))),
    }
}

fn config(o: RunOptions) -> Result<Config, RunError> {
    let mut cfg = Config::default();
    crate::models::needed(&o.asr_model)?;
    cfg.asr_model = o.asr_model;
    cfg.asr.language = parse_language(&o.language)?;
    cfg.asr.prompt = Some(o.prompt.trim().to_string()).filter(|s| !s.is_empty());
    cfg.num_speakers = match o.num_speakers {
        Some(0) => return Err(RunError::new(ErrorKind::Input, "The number of speakers must be at least 1.")),
        n => n.map(|n| n as usize),
    };
    Ok(cfg)
}

/// Clears the running flag when the worker ends, also on panic.
struct ClearOnDrop(Arc<AtomicBool>);

impl Drop for ClearOnDrop {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Starts a run and returns at once; progress and the outcome arrive as events.
#[tauri::command]
pub fn start_run<R: Runtime>(
    app: AppHandle<R>,
    running: State<'_, Running>,
    last: State<'_, LastRun>,
    path: PathBuf,
    options: RunOptions,
) -> Result<(), RunError> {
    let cfg = config(options)?;
    if running.0.swap(true, Ordering::SeqCst) {
        return Err(RunError::new(ErrorKind::Busy, "A recording is already being processed."));
    }
    last.set(None);
    let guard = ClearOnDrop(running.0.clone());
    std::thread::Builder::new()
        .name("pipeline".into())
        .spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline::run(&path, &cfg, &mut |p| {
                    let _ = app.emit("run-progress", ProgressEvent::from(p));
                })
            }));
            // Clear the flag before the outcome reaches the front-end, so it can start the next run right away.
            drop(guard);
            let _ = match outcome {
                Ok(Ok(tr)) => {
                    let result = RunResult {
                        audio_duration_s: tr.meta.audio_duration_s,
                        num_speakers: tr.meta.num_speakers,
                        processing_s: tr.meta.timings.total_s,
                        cues: tr.cues.clone(),
                    };
                    // Stored before the event, so the front-end can export as soon as it shows the transcript.
                    app.state::<LastRun>().set(Some(Finished { input: path, cues: tr.cues, meta: tr.meta }));
                    app.emit("run-finished", result)
                }
                Ok(Err(e)) => app.emit("run-failed", RunError::from(e)),
                Err(_) => {
                    app.emit("run-failed", RunError::new(ErrorKind::Processing, "Processing failed unexpectedly."))
                }
            };
        })
        .map_err(|e| RunError::new(ErrorKind::Processing, format!("Could not start processing: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(language: &str, num_speakers: Option<u32>) -> RunOptions {
        RunOptions {
            asr_model: "whisper-large-v3-turbo".into(),
            language: language.into(),
            prompt: "  ".into(),
            num_speakers,
        }
    }

    #[test]
    fn options_map_to_config() {
        let cfg = config(RunOptions { prompt: " Kubernetes ".into(), ..opts("en", Some(3)) }).unwrap();
        assert_eq!(cfg.asr_model, "whisper-large-v3-turbo");
        assert_eq!(cfg.asr.language.as_deref(), Some("en"));
        assert_eq!(cfg.asr.prompt.as_deref(), Some("Kubernetes"));
        assert_eq!(cfg.num_speakers, Some(3));

        let cfg = config(opts("auto", None)).unwrap();
        assert_eq!(cfg.asr.language, None);
        assert_eq!(cfg.asr.prompt, None);
        assert_eq!(cfg.num_speakers, None);
    }

    #[test]
    fn bad_options_are_input_errors() {
        assert_eq!(config(opts("english", None)).unwrap_err().kind, ErrorKind::Input);
        assert_eq!(config(opts("cs", Some(0))).unwrap_err().kind, ErrorKind::Input);
        let o = RunOptions { asr_model: "titanet-large".into(), ..opts("cs", None) };
        assert_eq!(config(o).unwrap_err().kind, ErrorKind::Input);
    }

    #[test]
    fn errors_map_to_kinds() {
        let e = RunError::from(Error::ModelMissing { id: "titanet-large".into(), path: "/m/x.onnx".into() });
        assert_eq!(e.kind, ErrorKind::ModelMissing);
        assert!(e.message.contains("Download it below"), "{}", e.message);
        assert_eq!(RunError::from(Error::InvalidInput("input file not found: /x".into())).kind, ErrorKind::Input);
        assert_eq!(RunError::from(Error::Audio("unsupported codec".into())).kind, ErrorKind::Input);
        assert_eq!(RunError::from(Error::Asr("boom".into())).kind, ErrorKind::Processing);
    }

    #[test]
    fn progress_events_are_tagged() {
        let v = serde_json::to_value(ProgressEvent::from(Progress::Chunk {
            done: 1,
            total: 4,
            audio_done: 30.0,
            audio_total: 120.0,
        }))
        .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"kind": "chunk", "done": 1, "total": 4, "audioDone": 30.0, "audioTotal": 120.0})
        );
        let v = serde_json::to_value(ProgressEvent::from(Progress::Started(Stage::Diarize))).unwrap();
        assert_eq!(v, serde_json::json!({"kind": "started", "stage": "diarize"}));
    }

    mod ipc {
        use super::super::*;
        use serde_json::{Value, json};
        use std::sync::mpsc;
        use std::time::Duration;
        use tauri::Listener;
        use tauri::test::{MockRuntime, mock_builder, mock_context, noop_assets};

        fn app() -> (tauri::App<MockRuntime>, tauri::WebviewWindow<MockRuntime>) {
            let app = mock_builder()
                .manage(Running::default())
                .manage(LastRun::default())
                .invoke_handler(tauri::generate_handler![start_run, crate::export::export_transcript])
                .build(mock_context(noop_assets()))
                .unwrap();
            let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default()).build().unwrap();
            (app, webview)
        }

        fn invoke(webview: &tauri::WebviewWindow<MockRuntime>, args: Value) -> Result<Value, Value> {
            call(webview, "start_run", args)
        }

        fn call(webview: &tauri::WebviewWindow<MockRuntime>, cmd: &str, args: Value) -> Result<Value, Value> {
            tauri::test::get_ipc_response(
                webview,
                tauri::webview::InvokeRequest {
                    cmd: cmd.into(),
                    callback: tauri::ipc::CallbackFn(0),
                    error: tauri::ipc::CallbackFn(1),
                    url: "tauri://localhost".parse().unwrap(),
                    body: tauri::ipc::InvokeBody::Json(args),
                    headers: Default::default(),
                    invoke_key: tauri::test::INVOKE_KEY.to_string(),
                },
            )
            .map(|b| b.deserialize::<Value>().unwrap())
        }

        /// Events in the order the front-end receives them, as `(name, payload)`, up to the outcome.
        fn collect(app: &tauri::App<MockRuntime>) -> mpsc::Receiver<(String, Value)> {
            let (tx, rx) = mpsc::channel();
            for name in ["run-progress", "run-finished", "run-failed"] {
                let tx = tx.clone();
                app.listen_any(name, move |e| {
                    let _ = tx.send((name.to_string(), serde_json::from_str(e.payload()).unwrap()));
                });
            }
            rx
        }

        fn wait_outcome(rx: &mpsc::Receiver<(String, Value)>, timeout: Duration) -> Vec<(String, Value)> {
            let mut events = Vec::new();
            loop {
                let ev = rx.recv_timeout(timeout).expect("no outcome event");
                let done = ev.0 != "run-progress";
                events.push(ev);
                if done {
                    return events;
                }
            }
        }

        #[test]
        fn missing_file_fails_with_an_input_error_and_frees_the_runner() {
            let (app, webview) = app();
            let rx = collect(&app);
            let args = json!({"path": "/nonexistent/meeting.mp4", "options": {"asrModel": "whisper-large-v3-turbo", "language": "cs", "prompt": ""}});
            assert_eq!(invoke(&webview, args.clone()), Ok(Value::Null));
            let events = wait_outcome(&rx, Duration::from_secs(10));
            let (name, payload) = events.last().unwrap();
            assert_eq!(name, "run-failed");
            assert_eq!(payload["kind"], "input");
            assert!(payload["message"].as_str().unwrap().contains("/nonexistent/meeting.mp4"), "{payload}");
            // The flag is cleared before the outcome is emitted, so the next run starts right away.
            assert_eq!(invoke(&webview, args), Ok(Value::Null));
        }

        #[test]
        fn a_second_run_is_refused_while_one_is_running() {
            let (app, webview) = app();
            app.state::<Running>().0.store(true, Ordering::SeqCst);
            let args = json!({"path": "/x.mp4", "options": {"asrModel": "whisper-large-v3-turbo", "language": "cs", "prompt": "", "numSpeakers": 2}});
            assert_eq!(invoke(&webview, args).unwrap_err()["kind"], "busy");
        }

        /// Full run with the real models. `MO_MEETINGS_DESKTOP_SAMPLE=/path/to/recording cargo test -p
        /// mo-meetings-desktop -- --ignored --nocapture` prints the events as JSON lines.
        #[test]
        #[ignore = "needs the models and a recording in MO_MEETINGS_DESKTOP_SAMPLE"]
        fn real_run_reports_every_stage_then_the_transcript() {
            let path = std::env::var("MO_MEETINGS_DESKTOP_SAMPLE").expect("set MO_MEETINGS_DESKTOP_SAMPLE");
            let (app, webview) = app();
            let rx = collect(&app);
            let args = json!({"path": path, "options": {"asrModel": "whisper-large-v3-turbo", "language": "en", "prompt": "Kubernetes, backlog", "numSpeakers": null}});
            assert_eq!(invoke(&webview, args), Ok(Value::Null));
            let events = wait_outcome(&rx, Duration::from_secs(600));
            for (name, payload) in &events {
                println!("{}", json!({"event": name, "payload": payload}));
            }
            let started: Vec<&str> = events
                .iter()
                .filter(|(_, p)| p["kind"] == "started")
                .map(|(_, p)| p["stage"].as_str().unwrap())
                .collect();
            assert_eq!(started, ["load", "decode", "diarize", "transcribe", "assemble"]);
            assert!(events.iter().any(|(_, p)| p["kind"] == "chunk"));
            let (name, result) = events.last().unwrap();
            assert_eq!(name, "run-finished", "{result}");
            assert!(!result["cues"].as_array().unwrap().is_empty(), "{result}");

            // Export the run as the front-end does after the save dialog; the files stay for inspection.
            let vtt = std::env::temp_dir().join("mo-meetings-desktop-export.vtt");
            let out = call(&webview, "export_transcript", json!({"path": vtt})).unwrap();
            println!("{}", json!({"exported": out}));
            let text = std::fs::read_to_string(&vtt).unwrap();
            assert!(text.starts_with("WEBVTT\n\n1\n"), "{text}");
            assert_eq!(text.matches("<v Speaker ").count(), result["cues"].as_array().unwrap().len());
            let meta: Value =
                serde_json::from_str(&std::fs::read_to_string(vtt.with_extension("json")).unwrap()).unwrap();
            assert_eq!(meta["num_speakers"], result["numSpeakers"]);
            let name = std::path::Path::new(&path).file_name().unwrap().to_str().unwrap();
            assert_eq!(meta["input"], name);
        }
    }
}
