//! First-run setup: which models a run needs, whether they are in the shared cache, and downloading the missing
//! ones on a worker thread. Reports through events: `download-progress` ([`DownloadEvent`]), then either
//! `download-finished` or `download-failed` ([`RunError`]).

use crate::run::{ErrorKind, RunError};
use mo_meetings_core::Error;
use mo_meetings_core::models::{self, Kind, ModelInfo};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Runtime, State};

/// Third-party models and libraries with their licenses, shown in the app.
const NOTICE: &str = include_str!("../../../../NOTICE");

/// Attempts per model before the download fails and the user is offered a retry.
const ATTEMPTS: u32 = 3;

/// Whether a download is in progress; one at a time.
#[derive(Default)]
pub struct Downloading(Arc<AtomicBool>);

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    id: &'static str,
    kind: &'static str,
    size: u64,
    license: &'static str,
    attribution: &'static str,
    downloaded: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Setup {
    /// Where the models live (`MO_MEETINGS_MODELS_DIR` or the user cache), shared with the CLI and MCP server.
    dir: PathBuf,
    /// The models a run with the chosen whisper model needs, in download order.
    needed: Vec<ModelStatus>,
    notice: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum DownloadEvent {
    /// Bytes of model `index` (1-based) of `count` received so far.
    Progress { id: &'static str, index: usize, count: usize, done: u64, total: u64 },
    /// Attempt `attempt` failed; the next one starts after a short pause.
    Retrying { id: &'static str, attempt: u32, message: String },
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Asr => "asr",
        Kind::Segmentation => "segmentation",
        Kind::Embedding => "embedding",
    }
}

/// The whisper model `asr_model` (must be a speech recognition model) and the default diarization models.
pub fn needed(asr_model: &str) -> Result<Vec<&'static ModelInfo>, RunError> {
    let asr = models::find(asr_model)
        .filter(|m| m.kind == Kind::Asr)
        .ok_or_else(|| RunError::new(ErrorKind::Input, format!("Unknown speech recognition model `{asr_model}`.")))?;
    let mut list = vec![asr];
    list.extend(models::MODELS.iter().filter(|m| m.kind != Kind::Asr && m.default));
    Ok(list)
}

/// The models a run with `asr_model` needs and whether each is downloaded, the model directory and the license
/// notice.
#[tauri::command]
pub fn model_setup(asr_model: String) -> Result<Setup, RunError> {
    Ok(Setup {
        dir: models::models_dir(),
        needed: needed(&asr_model)?
            .into_iter()
            .map(|m| ModelStatus {
                id: m.id,
                kind: kind_name(m.kind),
                size: m.size,
                license: m.license,
                attribution: m.attribution,
                downloaded: models::path(m).is_file(),
            })
            .collect(),
        notice: NOTICE,
    })
}

type Fetch<'a> = dyn FnMut(&ModelInfo, &mut dyn FnMut(u64, u64)) -> mo_meetings_core::Result<PathBuf> + 'a;

/// Downloads `list` one after another with `fetch`, retrying each up to [`ATTEMPTS`] times with `pause` between
/// attempts (a failed attempt resumes from its partial file).
fn download_all(
    list: &[&'static ModelInfo],
    fetch: &mut Fetch<'_>,
    emit: &mut dyn FnMut(DownloadEvent),
    pause: &dyn Fn(u32),
) -> Result<(), RunError> {
    let count = list.len();
    for (i, m) in list.iter().enumerate() {
        let mut attempt = 1;
        loop {
            // At most ~10 progress events a second, plus the last one.
            let mut last: Option<Instant> = None;
            let mut on_progress = |done: u64, total: u64| {
                if done < total && last.is_some_and(|t| t.elapsed() < Duration::from_millis(100)) {
                    return;
                }
                last = Some(Instant::now());
                emit(DownloadEvent::Progress { id: m.id, index: i + 1, count, done, total });
            };
            match fetch(m, &mut on_progress) {
                Ok(_) => break,
                Err(e) if attempt < ATTEMPTS => {
                    emit(DownloadEvent::Retrying { id: m.id, attempt, message: e.to_string() });
                    pause(attempt);
                    attempt += 1;
                }
                Err(e) => return Err(download_error(m, e)),
            }
        }
    }
    Ok(())
}

fn download_error(m: &ModelInfo, e: Error) -> RunError {
    RunError::new(
        ErrorKind::Download,
        format!(
            "Downloading `{}` failed after {ATTEMPTS} attempts. Check the connection and retry.\n\nDetails: {e}",
            m.id
        ),
    )
}

/// Clears the downloading flag when the worker ends, also on panic.
struct ClearOnDrop(Arc<AtomicBool>);

impl Drop for ClearOnDrop {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Starts downloading the models a run with `asr_model` needs and are missing; returns at once.
#[tauri::command]
pub fn download_models<R: Runtime>(
    app: AppHandle<R>,
    downloading: State<'_, Downloading>,
    asr_model: String,
) -> Result<(), RunError> {
    let missing: Vec<_> = needed(&asr_model)?.into_iter().filter(|m| !models::path(m).is_file()).collect();
    if downloading.0.swap(true, Ordering::SeqCst) {
        return Err(RunError::new(ErrorKind::Busy, "The models are already being downloaded."));
    }
    let guard = ClearOnDrop(downloading.0.clone());
    std::thread::Builder::new()
        .name("download".into())
        .spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                download_all(
                    &missing,
                    &mut |m, progress| models::download(m, false, progress),
                    &mut |ev| {
                        let _ = app.emit("download-progress", ev);
                    },
                    &|attempt| std::thread::sleep(Duration::from_secs(2 * u64::from(attempt))),
                )
            }));
            drop(guard);
            let _ = match outcome {
                Ok(Ok(())) => app.emit("download-finished", ()),
                Ok(Err(e)) => app.emit("download-failed", e),
                Err(_) => {
                    app.emit("download-failed", RunError::new(ErrorKind::Download, "The download failed unexpectedly."))
                }
            };
        })
        .map_err(|e| RunError::new(ErrorKind::Download, format!("Could not start the download: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(list: &[&ModelInfo]) -> Vec<&'static str> {
        list.iter().map(|m| m.id).collect()
    }

    #[test]
    fn a_run_needs_the_chosen_whisper_model_and_the_diarization_models() {
        let all = ["pyannote-segmentation-3.0", "titanet-large"];
        assert_eq!(ids(&needed("whisper-large-v3-turbo").unwrap()), [&["whisper-large-v3-turbo"][..], &all].concat());
        assert_eq!(ids(&needed("whisper-large-v3").unwrap()), [&["whisper-large-v3"][..], &all].concat());
        assert_eq!(needed("titanet-large").unwrap_err().kind, ErrorKind::Input);
        assert_eq!(needed("nope").unwrap_err().kind, ErrorKind::Input);
    }

    #[test]
    fn setup_lists_the_needed_models_with_their_licenses_and_the_notice() {
        let s = model_setup("whisper-large-v3".into()).unwrap();
        assert_eq!(s.needed.iter().map(|m| m.id).collect::<Vec<_>>(), ids(&needed("whisper-large-v3").unwrap()));
        assert!(s.needed.iter().all(|m| !m.license.is_empty() && !m.attribution.is_empty()));
        assert!(s.notice.contains("CC-BY-4.0") && s.notice.contains("OpenAI Whisper"));
        let v = serde_json::to_value(&s.needed[0]).unwrap();
        assert_eq!(v["id"], "whisper-large-v3");
        assert_eq!(v["kind"], "asr");
        assert_eq!(v["size"], 3_095_033_483u64);
        assert_eq!(model_setup("nope".into()).unwrap_err().kind, ErrorKind::Input);
    }

    /// Runs `download_all` over the default turbo set with a fake fetch that fails `failures[i]` times for model `i`.
    fn run(failures: &[u32]) -> (Result<(), RunError>, Vec<DownloadEvent>, Vec<u32>) {
        let list = needed("whisper-large-v3-turbo").unwrap();
        let mut left: Vec<u32> = failures.to_vec();
        let mut events = Vec::new();
        let pauses = std::cell::RefCell::new(Vec::new());
        let out = download_all(
            &list,
            &mut |m, progress| {
                let i = list.iter().position(|x| x.id == m.id).unwrap();
                progress(m.size / 2, m.size);
                if left[i] > 0 {
                    left[i] -= 1;
                    return Err(Error::Download("connection reset".into()));
                }
                progress(m.size, m.size);
                Ok(PathBuf::from(m.file))
            },
            &mut |e| events.push(e),
            &|a| pauses.borrow_mut().push(a),
        );
        (out, events, pauses.into_inner())
    }

    #[test]
    fn downloads_each_model_in_order_with_progress() {
        let (out, events, pauses) = run(&[0, 0, 0]);
        out.unwrap();
        assert!(pauses.is_empty());
        let done: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                DownloadEvent::Progress { id, index, count: 3, done, total } if done == total => Some((*id, *index)),
                _ => None,
            })
            .collect();
        assert_eq!(done, [("whisper-large-v3-turbo", 1), ("pyannote-segmentation-3.0", 2), ("titanet-large", 3)]);
    }

    #[test]
    fn a_failed_attempt_is_retried() {
        let (out, events, pauses) = run(&[0, 2, 0]);
        out.unwrap();
        assert_eq!(pauses, [1, 2]);
        let retries: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                DownloadEvent::Retrying { id, attempt, .. } => Some((*id, *attempt)),
                _ => None,
            })
            .collect();
        assert_eq!(retries, [("pyannote-segmentation-3.0", 1), ("pyannote-segmentation-3.0", 2)]);
    }

    #[test]
    fn gives_up_after_the_last_attempt_and_names_the_model() {
        let (out, events, _) = run(&[0, 0, ATTEMPTS]);
        let e = out.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Download);
        let v = serde_json::to_value(&e).unwrap();
        let msg = v["message"].as_str().unwrap();
        assert!(msg.contains("`titanet-large`") && msg.contains("connection reset"), "{msg}");
        let retries =
            events.iter().filter(|e| matches!(e, DownloadEvent::Retrying { id: "titanet-large", .. })).count();
        assert_eq!(retries, ATTEMPTS as usize - 1);
        let v =
            serde_json::to_value(events.iter().rev().find(|e| matches!(e, DownloadEvent::Retrying { .. })).unwrap())
                .unwrap();
        assert_eq!(
            v,
            serde_json::json!({"kind": "retrying", "id": "titanet-large", "attempt": 2, "message": "model download failed: connection reset"})
        );
    }
}
