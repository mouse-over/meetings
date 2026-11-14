// No console window next to the app on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod export;
mod models;
mod run;

fn main() {
    let result = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(run::Running::default())
        .manage(export::LastRun::default())
        .manage(models::Downloading::default())
        .invoke_handler(tauri::generate_handler![
            run::start_run,
            export::export_transcript,
            models::model_setup,
            models::download_models
        ])
        .run(tauri::generate_context!());
    if let Err(e) = result {
        eprintln!("error while running the desktop app: {e}");
        std::process::exit(1);
    }
}
