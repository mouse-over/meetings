// No console window next to the app on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// Linked from the start so the app builds against the pipeline; the commands that call it come next.
use mo_meetings_core as _;

fn main() {
    tauri::Builder::default().run(tauri::generate_context!()).expect("error while running the desktop app");
}
