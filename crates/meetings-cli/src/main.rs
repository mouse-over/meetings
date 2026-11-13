//! `meetings`: local meeting transcription with speakers → WebVTT.
//!
//! Exit codes: 0 ok, 2 bad options or input, 3 model missing, 4 processing failed.

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use meetings_core::pipeline::{self, Config, Progress, Stage};
use meetings_core::{Error, models, vtt};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "meetings", version, about = "Local meeting transcription with speaker diarization")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Transcribe a recording (mp4, m4a, wav, …) into WebVTT with speakers
    Transcribe(TranscribeArgs),
    /// Diarize only: write speaker turns as JSON
    Diarize {
        /// Input recording
        input: PathBuf,
        /// Output JSON (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Hint: at most this many speakers
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=64))]
        num_speakers: Option<u32>,
        /// CPU threads
        #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
        threads: u32,
    },
    /// Manage models in the user cache
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ModelChoice {
    /// whisper large-v3-turbo (default; large-v3 was 2.5× slower and not better)
    Turbo,
    /// whisper large-v3
    Large,
}

impl ModelChoice {
    fn id(self) -> &'static str {
        match self {
            ModelChoice::Turbo => "whisper-large-v3-turbo",
            ModelChoice::Large => "whisper-large-v3",
        }
    }
}

#[derive(clap::Args)]
struct TranscribeArgs {
    /// Input recording
    input: PathBuf,
    /// Output VTT (default: input with .vtt extension)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Also write metadata JSON next to the VTT
    #[arg(long)]
    json: bool,
    /// Language code, or `auto`
    #[arg(long, default_value = "cs")]
    language: String,
    /// Glossary / context prompt given to whisper for every chunk
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,
    /// Read the prompt from a file
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Hint: at most this many speakers (by default the number is estimated)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=64))]
    num_speakers: Option<u32>,
    /// Whisper model
    #[arg(long, value_enum, default_value_t = ModelChoice::Turbo)]
    model: ModelChoice,
    /// CPU threads (diarization; more than the performance cores is slower)
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    threads: u32,
    /// Write words and diarization turns as JSON (for evaluation)
    #[arg(long, hide = true)]
    dump: Option<PathBuf>,
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// List known models, licenses and whether they are downloaded
    List,
    /// Download models (default: those needed by `transcribe` with the turbo model)
    Download {
        /// Model ids (see `meetings models list`)
        ids: Vec<String>,
        /// Download all models
        #[arg(long, conflicts_with = "ids")]
        all: bool,
        /// Re-check SHA-256 of files that are already present
        #[arg(long)]
        verify: bool,
    },
    /// Print the model directory
    Path,
}

fn main() -> ExitCode {
    let cli = Cli::parse(); // exits with 2 on bad options
    let result = match cli.command {
        Command::Transcribe(args) => transcribe(args),
        Command::Diarize { input, output, num_speakers, threads } => diarize(&input, output, num_speakers, threads),
        Command::Models { command } => models_cmd(command),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            let code = e.downcast_ref::<Error>().map_or(4, Error::exit_code);
            ExitCode::from(code as u8)
        }
    }
}

fn invalid(msg: impl Into<String>) -> anyhow::Error {
    Error::InvalidInput(msg.into()).into()
}

fn transcribe(args: TranscribeArgs) -> anyhow::Result<()> {
    let prompt = match (&args.prompt, &args.prompt_file) {
        (Some(p), _) => Some(p.clone()),
        (None, Some(f)) => Some(
            std::fs::read_to_string(f).map_err(|e| invalid(format!("cannot read prompt file {}: {e}", f.display())))?,
        ),
        _ => None,
    };
    let output = args.output.clone().unwrap_or_else(|| args.input.with_extension("vtt"));
    if output == args.input {
        return Err(invalid("output would overwrite the input"));
    }
    let language = match args.language.trim() {
        "auto" => None,
        l if l.len() == 2 && l.chars().all(|c| c.is_ascii_lowercase()) => Some(l.to_string()),
        l => return Err(invalid(format!("invalid language `{l}` (use a 2-letter code or `auto`)"))),
    };
    let mut cfg = Config { asr_model: args.model.id().into(), threads: args.threads as usize, ..Default::default() };
    cfg.asr.language = language;
    cfg.asr.prompt = prompt.map(|p| p.trim().to_string()).filter(|p| !p.is_empty());
    cfg.num_speakers = args.num_speakers.map(|n| n as usize);

    let mut reporter = Reporter::new();
    let tr = pipeline::run(&args.input, &cfg, &mut |p| reporter.on(p))?;
    reporter.done();

    std::fs::write(&output, vtt::render(&tr.cues))?;
    eprintln!(
        "wrote {} ({} cues, {} speakers, {:.0} s audio in {:.0} s)",
        output.display(),
        tr.cues.len(),
        tr.meta.num_speakers,
        tr.meta.audio_duration_s,
        tr.meta.timings.total_s
    );
    if args.json {
        let path = output.with_extension("json");
        std::fs::write(&path, serde_json::to_string_pretty(&tr.meta)? + "\n")?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = &args.dump {
        write_dump(path, &tr)?;
    }
    Ok(())
}

fn diarize(input: &Path, output: Option<PathBuf>, num_speakers: Option<u32>, threads: u32) -> anyhow::Result<()> {
    let cfg =
        Config { threads: threads as usize, num_speakers: num_speakers.map(|n| n as usize), ..Default::default() };
    let start = std::time::Instant::now();
    eprintln!("diarizing…");
    let (d, duration) = pipeline::run_diarization(input, &cfg)?;
    let label = |s: usize| format!("SPEAKER_{s:02}");
    let turns: Vec<_> = d.turns.iter().map(|t| serde_json::json!([t.start, t.end, label(t.speaker)])).collect();
    let excl: Vec<_> = meetings_core::assembly::exclusive_turns(&d.turns)
        .iter()
        .map(|t| serde_json::json!([t.start, t.end, label(t.speaker)]))
        .collect();
    let timings: serde_json::Map<String, serde_json::Value> =
        d.timings.iter().map(|(k, v)| (k.clone(), (*v).into())).collect();
    let v = serde_json::json!({
        "audio_duration_s": duration,
        "num_speakers": d.num_speakers,
        "num_speakers_estimated": d.estimated,
        "report": d.report,
        "timings": timings,
        "total_s": start.elapsed().as_secs_f64(),
        "turns": turns,
        "exclusive": excl,
    });
    let text = serde_json::to_string_pretty(&v)? + "\n";
    match output {
        Some(p) => std::fs::write(p, text)?,
        None => print!("{text}"),
    }
    eprintln!("{} speakers in {:.1} s", d.num_speakers, start.elapsed().as_secs_f64());
    Ok(())
}

/// Evaluation dump in the format of the spike scripts (`segments[].words[]`, `turns`, `exclusive`).
fn write_dump(path: &Path, tr: &pipeline::Transcript) -> anyhow::Result<()> {
    let words: Vec<_> = tr
        .words
        .iter()
        .map(|w| serde_json::json!({"word": w.text, "start": w.start, "end": w.end, "speaker": w.speaker}))
        .collect();
    let label = |s: usize| format!("SPEAKER_{s:02}");
    let turns: Vec<_> =
        tr.diarization.turns.iter().map(|t| serde_json::json!([t.start, t.end, label(t.speaker)])).collect();
    let excl: Vec<_> = tr.turns.iter().map(|t| serde_json::json!([t.start, t.end, label(t.speaker)])).collect();
    let chunks: Vec<_> = tr.chunks.iter().map(|c| serde_json::json!([c.start, c.end])).collect();
    let v = serde_json::json!({
        "segments": [{"words": words}],
        "turns": turns,
        "exclusive": excl,
        "chunks": chunks,
        "meta": tr.meta,
    });
    std::fs::write(path, serde_json::to_string(&v)?)?;
    Ok(())
}

struct Reporter {
    bar: Option<ProgressBar>,
    tty: bool,
    last_pct: usize,
}

impl Reporter {
    fn new() -> Self {
        Self { bar: None, tty: std::io::stderr().is_terminal(), last_pct: 0 }
    }

    fn on(&mut self, p: Progress) {
        match p {
            Progress::Started(stage) => {
                let msg = match stage {
                    Stage::Load => "loading models",
                    Stage::Decode => "decoding audio",
                    Stage::Diarize => "diarizing",
                    Stage::Transcribe => "transcribing",
                    Stage::Assemble => "assembling",
                };
                if self.tty && stage != Stage::Transcribe {
                    let bar = ProgressBar::new_spinner().with_message(msg);
                    bar.enable_steady_tick(std::time::Duration::from_millis(120));
                    self.bar = Some(bar);
                } else if !self.tty {
                    eprintln!("{msg}…");
                }
            }
            Progress::Chunk { done, total, audio_done, audio_total } => {
                if self.tty {
                    let bar = self.bar.get_or_insert_with(|| {
                        let b = ProgressBar::new(total as u64);
                        b.set_style(
                            ProgressStyle::with_template("transcribing [{bar:30}] {pos}/{len} chunks {msg} eta {eta}")
                                .expect("valid template")
                                .progress_chars("=> "),
                        );
                        b
                    });
                    bar.set_position(done as u64);
                    bar.set_message(format!("({:.0}/{:.0} s)", audio_done, audio_total));
                } else {
                    let pct = done * 100 / total.max(1);
                    if pct / 10 > self.last_pct / 10 || done == total {
                        eprintln!("  {done}/{total} chunks ({pct} %)");
                    }
                    self.last_pct = pct;
                }
            }
            Progress::Finished(stage, secs) => {
                if let Some(bar) = self.bar.take() {
                    bar.finish_and_clear();
                }
                eprintln!("{:?} done in {:.1} s", stage, secs);
            }
        }
    }

    fn done(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}

fn human(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.0} MB", bytes as f64 / (1u64 << 20) as f64)
    }
}

fn models_cmd(cmd: ModelsCommand) -> anyhow::Result<()> {
    match cmd {
        ModelsCommand::Path => println!("{}", models::models_dir().display()),
        ModelsCommand::List => {
            println!("model directory: {}\n", models::models_dir().display());
            for m in models::MODELS {
                let present = if models::path(m).is_file() { "downloaded" } else { "-" };
                let def = if m.default { "default" } else { "" };
                println!(
                    "{:<28} {:>8}  {:<10} {:<10} {:<7}  {}",
                    m.id,
                    human(m.size),
                    m.license,
                    present,
                    def,
                    m.attribution
                );
            }
        }
        ModelsCommand::Download { ids, all, verify } => {
            let selected: Vec<&models::ModelInfo> = if all {
                models::MODELS.iter().collect()
            } else if ids.is_empty() {
                models::MODELS.iter().filter(|m| m.default).collect()
            } else {
                ids.iter()
                    .map(|id| models::find(id).ok_or_else(|| invalid(format!("unknown model `{id}`"))))
                    .collect::<anyhow::Result<_>>()?
            };
            for m in selected {
                let bar = ProgressBar::new(m.size);
                bar.set_style(
                    ProgressStyle::with_template("{msg:28} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}")
                        .expect("valid template")
                        .progress_chars("=> "),
                );
                bar.set_message(m.id);
                let path = models::download(m, verify, &mut |done, total| {
                    bar.set_length(total);
                    bar.set_position(done);
                })?;
                bar.finish_and_clear();
                eprintln!("{:<28} ok  {}", m.id, path.display());
            }
        }
    }
    Ok(())
}
