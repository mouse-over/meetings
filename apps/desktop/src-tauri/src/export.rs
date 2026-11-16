//! Saves the last finished run like `mo-meetings transcribe --json`: the Teams-shaped VTT at the chosen path and
//! the metadata JSON next to it (same name, `.json`).

use crate::run::{ErrorKind, RunError};
use mo_meetings_core::pipeline::Metadata;
use mo_meetings_core::{Cue, vtt};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::State;

/// What an export writes, kept from the last finished run.
pub struct Finished {
    /// The recording the transcript belongs to.
    pub input: PathBuf,
    pub cues: Vec<Cue>,
    pub meta: Metadata,
}

/// The last finished run; cleared when the next run starts, so an export never mixes two runs.
#[derive(Default)]
pub struct LastRun(pub Mutex<Option<Finished>>);

impl LastRun {
    pub fn set(&self, run: Option<Finished>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = run;
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Exported {
    vtt: PathBuf,
    json: PathBuf,
}

fn write(run: &Finished, vtt_path: &Path) -> Result<Exported, RunError> {
    let json_path = vtt_path.with_extension("json");
    if vtt_path == run.input || json_path == run.input {
        return Err(RunError::new(ErrorKind::Input, "Pick another name: this would overwrite the recording."));
    }
    let json = serde_json::to_string_pretty(&run.meta)
        .map_err(|e| RunError::new(ErrorKind::Processing, format!("Could not write the metadata: {e}")))?
        + "\n";
    let save = |path: &Path, text: &str| {
        std::fs::write(path, text)
            .map_err(|e| RunError::new(ErrorKind::Input, format!("Could not save {}: {e}", path.display())))
    };
    save(vtt_path, &vtt::render(&run.cues))?;
    save(&json_path, &json)?;
    Ok(Exported { vtt: vtt_path.to_path_buf(), json: json_path })
}

/// Writes the VTT to `path` and the metadata JSON next to it; returns both paths.
#[tauri::command]
pub fn export_transcript(last: State<'_, LastRun>, path: PathBuf) -> Result<Exported, RunError> {
    let guard = last.0.lock().unwrap_or_else(|e| e.into_inner());
    let run = guard.as_ref().ok_or_else(|| RunError::new(ErrorKind::Input, "There is no transcript to export yet."))?;
    write(run, &path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mo_meetings_core::pipeline::Timings;

    fn finished(input: &Path) -> Finished {
        Finished {
            input: input.to_path_buf(),
            cues: vec![
                Cue { start: 0.5, end: 2.0, speaker: 1, text: "Ahoj <všichni> & vítejte".into() },
                Cue { start: 2.0, end: 3.25, speaker: 2, text: "Díky.".into() },
            ],
            meta: Metadata {
                tool: "mo-meetings test".into(),
                input: input.display().to_string(),
                audio_duration_s: 3.25,
                decoder: None,
                model: "whisper-large-v3-turbo".into(),
                profile: None,
                language: Some("cs".into()),
                prompt: false,
                num_speakers: 2,
                num_speakers_estimated: true,
                num_speakers_hint: None,
                diarization: serde_json::json!({}),
                chunks: 1,
                words: 5,
                cues: 2,
                timings: Timings::default(),
            },
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mo-meetings-export-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn writes_the_cli_vtt_and_metadata_json() {
        let dir = tmp("ok");
        let run = finished(&dir.join("meeting.mp4"));
        let out = write(&run, &dir.join("meeting.vtt")).unwrap();
        assert_eq!(out.json, dir.join("meeting.json"));
        assert_eq!(std::fs::read_to_string(&out.vtt).unwrap(), vtt::render(&run.cues));
        let vtt = std::fs::read_to_string(&out.vtt).unwrap();
        assert!(vtt.starts_with(
            "WEBVTT\n\n1\n00:00:00.500 --> 00:00:02.000\n<v Speaker 1>Ahoj &lt;všichni&gt; &amp; vítejte</v>"
        ));
        let json = std::fs::read_to_string(&out.json).unwrap();
        assert_eq!(json, serde_json::to_string_pretty(&run.meta).unwrap() + "\n");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&json).unwrap()["num_speakers"], 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_to_overwrite_the_recording() {
        let dir = tmp("overwrite");
        for input in ["meeting.vtt", "meeting.json"] {
            let e = write(&finished(&dir.join(input)), &dir.join("meeting.vtt")).unwrap_err();
            assert_eq!(e.kind, ErrorKind::Input);
        }
        assert!(!dir.join("meeting.vtt").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn exporting_before_a_run_is_an_input_error() {
        use tauri::Manager;
        let app = tauri::test::mock_builder()
            .manage(LastRun::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let e = export_transcript(app.state::<LastRun>(), "/tmp/x.vtt".into()).unwrap_err();
        assert_eq!(e.kind, ErrorKind::Input);
    }
}
