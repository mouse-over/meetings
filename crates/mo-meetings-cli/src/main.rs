//! `mo-meetings`: local meeting transcription with speakers → WebVTT.
//!
//! Exit codes: 0 ok, 2 bad options or input, 3 model missing, 4 processing failed.

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use mo_meetings_core::pipeline::{self, Config, Progress, Stage};
use mo_meetings_core::profile::{self, Profiles};
use mo_meetings_core::{Error, asr, models, vtt};
use mo_meetings_project::Project;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "mo-meetings",
    version,
    about = "mo-meetings (mouse-over meetings): local meeting transcription with speaker diarization"
)]
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
    /// List and inspect settings profiles
    Profiles {
        #[command(subcommand)]
        command: ProfilesCommand,
    },
}

#[derive(Subcommand)]
enum ProfilesCommand {
    /// List the profiles, where each comes from, and which match a recording
    List {
        /// Mark the profiles whose `match` patterns fit this recording's file name
        #[arg(long = "for", value_name = "RECORDING")]
        recording: Option<PathBuf>,
    },
    /// Show a profile with everything it extends merged in
    Show { name: String },
    /// Print the profile directories (later wins on the same name)
    Path,
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
    /// Input recording (with an existing `--project`: optional, the project's recording, or its new path when
    /// it moved)
    #[arg(required_unless_present = "project")]
    input: Option<PathBuf>,
    /// Project file (e.g. `meeting.mom`): created for the recording when missing; stores the run and resumes an
    /// interrupted run with the same settings
    #[arg(long)]
    project: Option<PathBuf>,
    /// Output VTT (default: recording with .vtt extension)
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Also write metadata JSON next to the VTT
    #[arg(long)]
    json: bool,
    /// Settings profile (see `mo-meetings profiles list`); the options below override it
    #[arg(long, short = 'p')]
    profile: Option<String>,
    /// Language code, or `auto` [default: cs]
    #[arg(long)]
    language: Option<String>,
    /// Glossary / context prompt given to whisper for every chunk
    #[arg(long, conflicts_with = "prompt_file")]
    prompt: Option<String>,
    /// Read the prompt from a file
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Hint: at most this many speakers (by default the number is estimated)
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=64))]
    num_speakers: Option<u32>,
    /// Whisper model [default: turbo]
    #[arg(long, value_enum)]
    model: Option<ModelChoice>,
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
        /// Model ids (see `mo-meetings models list`)
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
        Command::Profiles { command } => profiles_cmd(command),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            let code = match (e.downcast_ref::<Error>(), e.downcast_ref::<mo_meetings_project::Error>()) {
                (Some(e), _) => e.exit_code(),
                (_, Some(e)) => e.exit_code(),
                _ => 4,
            };
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
    let mut project = match &args.project {
        Some(path) => {
            let (p, created) = Project::open_or_create(path, args.input.as_deref())?;
            if created {
                eprintln!("created project {}", p.path().display());
            }
            Some(p)
        }
        None => None,
    };
    let input = match (&project, &args.input) {
        (Some(p), _) => p.recording()?,
        (None, Some(input)) => input.clone(),
        (None, None) => return Err(invalid("no recording given")),
    };
    let output = args.output.clone().unwrap_or_else(|| input.with_extension("vtt"));
    if std::path::absolute(&output)? == std::path::absolute(&input)? {
        return Err(invalid("output would overwrite the input"));
    }
    let mut cfg = Config { threads: args.threads as usize, ..Default::default() };
    match &args.profile {
        Some(name) => {
            let profiles = Profiles::load()?;
            warn_all(&profiles.warnings);
            let resolved = profiles.resolve(name)?;
            resolved.apply(&mut cfg)?;
            eprintln!(
                "profile {name} ({})",
                resolved.sources.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" → ")
            );
        }
        None => suggest_profile(&input),
    }
    if let Some(l) = &args.language {
        cfg.asr.language = asr::parse_language(l)?;
    }
    if let Some(p) = prompt {
        cfg.asr.prompt = Some(p.trim().to_string()).filter(|p| !p.is_empty());
    }
    if let Some(m) = args.model {
        cfg.asr_model = m.id().into();
    }
    if let Some(n) = args.num_speakers {
        cfg.num_speakers = Some(n as usize);
    }

    let mut reporter = Reporter::new();
    let tr = match &mut project {
        Some(p) => {
            let out = p.transcribe(&cfg, &mut |pr| reporter.on(pr), &std::sync::atomic::AtomicBool::new(false));
            reporter.done();
            let out = out?;
            if out.resumed_diarization {
                eprintln!(
                    "resumed run {}: diarization and {} of {} chunks from the interrupted run",
                    out.run_id,
                    out.resumed_chunks,
                    out.transcript.chunks.len()
                );
            }
            eprintln!("saved run {} in {}", out.run_id, p.path().display());
            out.transcript
        }
        None => {
            let tr = pipeline::run(&input, &cfg, &mut |p| reporter.on(p))?;
            reporter.done();
            tr
        }
    };

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

fn warn_all(warnings: &[String]) {
    for w in warnings {
        eprintln!("warning: {w}");
    }
}

/// Without `--profile`, points to profiles whose `match` patterns fit the recording. Problems with the
/// profile files are not reported here: the run does not use them.
fn suggest_profile(input: &Path) {
    let Ok(profiles) = Profiles::load() else { return };
    let names = profiles.suggest(input);
    if !names.is_empty() {
        eprintln!("hint: profile {} matches this recording (use --profile <name>)", names.join(", "));
    }
}

fn profiles_cmd(cmd: ProfilesCommand) -> anyhow::Result<()> {
    match cmd {
        ProfilesCommand::Path => {
            println!("config: {}", profile::config_dir().join("config.toml").display());
            for d in profile::profile_dirs(&profile::config_dir())? {
                println!("{}", d.display());
            }
        }
        ProfilesCommand::List { recording } => {
            let profiles = Profiles::load()?;
            warn_all(&profiles.warnings);
            let matching = recording.as_deref().map(|r| profiles.suggest(r)).unwrap_or_default();
            if profiles.list().next().is_none() {
                let dirs: Vec<String> = profiles.dirs.iter().map(|d| d.display().to_string()).collect();
                eprintln!("no profiles in {}", dirs.join(", "));
            }
            for p in profiles.list() {
                let mark = if matching.contains(&p.name.as_str()) { "*" } else { " " };
                let extends = p.extends.as_deref().map(|e| format!(" (extends {e})")).unwrap_or_default();
                println!("{mark} {:<20} {}{extends}", p.name, p.description.as_deref().unwrap_or(""));
                println!("    {}", p.path.display());
                for s in &p.shadowed {
                    println!("    overrides {}", s.display());
                }
            }
            if recording.is_some() {
                eprintln!("* = matches the recording's file name");
            }
        }
        ProfilesCommand::Show { name } => {
            let profiles = Profiles::load()?;
            warn_all(&profiles.warnings);
            let r = profiles.resolve(&name)?;
            let mut out = serde_json::to_value(&r)?;
            // Show the values a run would use, not only the ones the profile sets.
            let mut cfg = Config::default();
            r.apply(&mut cfg)?;
            if let serde_json::Value::Object(m) = &mut out {
                m.insert(
                    "effective".into(),
                    serde_json::json!({
                        "language": cfg.asr.language.as_deref().unwrap_or("auto"),
                        "model": cfg.asr_model,
                        "num_speakers": cfg.num_speakers,
                        "chunks": {"max_len": cfg.chunk_rules.max_len, "max_gap": cfg.chunk_rules.max_gap,
                                   "pad": cfg.chunk_rules.pad, "search": cfg.chunk_rules.search},
                        "cues": {"max_pause": cfg.cue_rules.max_pause, "max_duration": cfg.cue_rules.max_duration,
                                 "sentence_min_duration": cfg.cue_rules.sentence_min_duration},
                    }),
                );
            }
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
    }
    Ok(())
}

fn diarize(input: &Path, output: Option<PathBuf>, num_speakers: Option<u32>, threads: u32) -> anyhow::Result<()> {
    let cfg =
        Config { threads: threads as usize, num_speakers: num_speakers.map(|n| n as usize), ..Default::default() };
    let start = std::time::Instant::now();
    eprintln!("diarizing…");
    let (d, duration) = pipeline::run_diarization(input, &cfg)?;
    let v = pipeline::diarization_json(&d, duration, start.elapsed().as_secs_f64());
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
                                .unwrap_or_else(|_| ProgressStyle::default_bar())
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
            Progress::Warning(w) => match &self.bar {
                Some(bar) => bar.suspend(|| eprintln!("warning: {w}")),
                None => eprintln!("warning: {w}"),
            },
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
                        .unwrap_or_else(|_| ProgressStyle::default_bar())
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
