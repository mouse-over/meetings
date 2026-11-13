// No console window next to the app on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod run;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(run::Running::default())
        .invoke_handler(tauri::generate_handler![run::start_run])
        .run(tauri::generate_context!())
        .expect("error while running the desktop app");
}
