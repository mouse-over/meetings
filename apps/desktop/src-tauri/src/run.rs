//! Runs `pipeline::run` on a worker thread and reports to the front-end through events:
//! `run-progress` ([`ProgressEvent`]), then either `run-finished` ([`RunResult`]) or `run-failed` ([`RunError`]).

use mo_meetings_core::pipeline::{self, Config, Progress, Stage};
use mo_meetings_core::{Cue, Error};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Runtime, State};

/// Whether a run is in progress; one run at a time (the models take several GB of memory).
#[derive(Default)]
pub struct Running(Arc<AtomicBool>);

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunOptions {
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
    /// A run is already in progress.
    Busy,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunError {
    kind: ErrorKind,
    message: String,
}

impl RunError {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
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
                    "The model `{id}` is not downloaded (expected at {}). Download the models with \
                     `mo-meetings models download` and try again.",
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
    path: PathBuf,
    options: RunOptions,
) -> Result<(), RunError> {
    let cfg = config(options)?;
    if running.0.swap(true, Ordering::SeqCst) {
        return Err(RunError::new(ErrorKind::Busy, "A recording is already being processed."));
    }
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
                Ok(Ok(tr)) => app.emit(
                    "run-finished",
                    RunResult {
                        audio_duration_s: tr.meta.audio_duration_s,
                        num_speakers: tr.meta.num_speakers,
                        processing_s: tr.meta.timings.total_s,
                        cues: tr.cues,
                    },
                ),
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
        RunOptions { language: language.into(), prompt: "  ".into(), num_speakers }
    }

    #[test]
    fn options_map_to_config() {
        let cfg = config(RunOptions { prompt: " Kubernetes ".into(), ..opts("en", Some(3)) }).unwrap();
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
    }

    #[test]
    fn errors_map_to_kinds() {
        let e = RunError::from(Error::ModelMissing { id: "titanet-large".into(), path: "/m/x.onnx".into() });
        assert_eq!(e.kind, ErrorKind::ModelMissing);
        assert!(e.message.contains("mo-meetings models download"), "{}", e.message);
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
        use tauri::test::{MockRuntime, mock_builder, mock_context, noop_assets};
        use tauri::{Listener, Manager};

        fn app() -> (tauri::App<MockRuntime>, tauri::WebviewWindow<MockRuntime>) {
            let app = mock_builder()
                .manage(Running::default())
                .invoke_handler(tauri::generate_handler![start_run])
                .build(mock_context(noop_assets()))
                .unwrap();
            let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default()).build().unwrap();
            (app, webview)
        }

        fn invoke(webview: &tauri::WebviewWindow<MockRuntime>, args: Value) -> Result<Value, Value> {
            tauri::test::get_ipc_response(
                webview,
                tauri::webview::InvokeRequest {
                    cmd: "start_run".into(),
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
            let args = json!({"path": "/nonexistent/meeting.mp4", "options": {"language": "cs", "prompt": ""}});
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
            let args = json!({"path": "/x.mp4", "options": {"language": "cs", "prompt": "", "numSpeakers": 2}});
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
            let args = json!({"path": path, "options": {"language": "en", "prompt": "Kubernetes, backlog", "numSpeakers": null}});
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
        }
    }
}
