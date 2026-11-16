# mo-meetings

mo-meetings (mouse-over meetings): local meeting transcription with speaker diarization. A recording (Teams MP4, M4A, WAV, …) → WebVTT with
`Speaker N` labels, entirely on the machine and without Python. Long-term a desktop app in the style of
[Handy](https://github.com/cjpais/Handy) (Rust + Tauri); today a CLI and an MCP server.

Pipeline: 16 kHz mono audio → diarization (sherpa-onnx: pyannote segmentation-3.0 + NeMo TitaNet-Large,
clustering, cluster merging) → speech chunks of at most 30 s cut in pauses → whisper.cpp per chunk (glossary
prompt in every chunk) → words assigned to speakers → cues → VTT. See [`doc/architecture.md`](doc/architecture.md).

## Build

Rust 1.89+, CMake and a C/C++ toolchain (whisper.cpp is compiled from source). On macOS Metal is on by default.

```sh
cargo build --release                         # macOS (Metal)
cargo build --release --no-default-features   # Linux, Windows (MSVC) / CPU only
```

The sherpa-onnx build script downloads its prebuilt libraries; on Apple Silicon `mo-meetings-core`'s build script
also fetches Microsoft's ONNX Runtime 1.28.2 (pinned SHA-256), which runs diarization about 2.3× faster than
the copy in the sherpa-onnx release. The binary expects `lib*.dylib` and `onnxruntime/` next to it
(`target/release/`).

If linking fails on macOS with `tapi error: malformed file … unknown architecture arm64e.x1`, the Command Line
Tools are older than the default SDK; point to an older SDK, e.g.
`export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.

Prebuilt archives (macOS Apple Silicon with Metal, Linux x86_64 CPU, Windows x86_64 CPU) are attached to
[GitHub releases](https://github.com/mouse-over/meetings/releases). Keep the files together: on macOS the binary
loads `libsherpa-onnx-c-api.dylib` and `onnxruntime/` from its own directory. The binaries are not signed;
after downloading in a browser run `xattr -dr com.apple.quarantine mo-meetings-*/`.

## Use

```sh
mo-meetings models download          # ~1.7 GB into the user cache (whisper turbo, segmentation, TitaNet)
mo-meetings models list              # models, licenses, status
mo-meetings transcribe meeting.mp4 -o meeting.vtt --json --prompt-file glossary.txt
mo-meetings diarize meeting.mp4 -o turns.json
```

`transcribe` options: `--profile NAME` (see below), `--language cs` (or `auto`), `--prompt "…"` / `--prompt-file`, `--num-speakers N` (a cap,
the number is estimated otherwise), `--model turbo|large`, `--threads 4`. Progress goes to stderr. Exit codes:
0 ok, 2 bad options or input, 3 model missing, 4 processing failed.

A glossary prompt helps whisper with domain terms and names; it is sent with every chunk. Example:

```text
Schůzka o vývoji aplikace. Termíny: backend, frontend, deploy, pull request, sprint, backlog, Kubernetes.
```

A prompt longer than whisper keeps (223 tokens for the large models) loses its start; the run warns and shows the
part that is ignored.

### Settings profiles

A profile sets up a run for a type of meeting in one step: `mo-meetings transcribe sync.mp4 --profile team-sync`.
Options given on the command line override the profile. Profiles are `<name>.toml` files in `<config>/profiles`
and in the directories listed in `<config>/config.toml`, so a team can keep its profiles in its own repository
(a later directory wins on the same name). `<config>` is `~/Library/Application Support/mo-meetings` (macOS),
`~/.config/mo-meetings` (Linux) or `%APPDATA%\mo-meetings` (Windows); override with `MO_MEETINGS_CONFIG_DIR`.

```toml
# <config>/config.toml
profile_dirs = ["~/work/team-repo/meeting-profiles"]   # relative paths resolve against <config>
```

```toml
# team-sync.toml
description = "Weekly sync of the backend team"
extends = "company"                 # values here override the parent's, glossaries are merged
match = ["*weekly*sync*"]           # suggest this profile for recordings with matching file names
language = "cs"                     # or "auto"
prompt = "Weekly sync of the backend team."   # or prompt_file = "sync.txt", relative to this file
glossary = ["Kubernetes", "pull request", "backlog"]   # appended to the prompt
model = "turbo"                     # turbo, large or a model id
num_speakers = 8                    # at most this many speakers

[chunks]                            # advanced: max_len, max_gap, pad, search (seconds)
max_gap = 2.0

[cues]                              # advanced: max_pause, max_duration, sentence_min_duration (seconds)
max_duration = 6.0
```

`mo-meetings profiles list [--for recording.mp4]` lists the profiles, their files and which ones match a recording;
`profiles show NAME` prints a profile with everything it extends merged in; `profiles path` prints the directories.
Without `--profile`, `transcribe` prints a hint when a profile matches the recording's file name.

### Project files

`mo-meetings transcribe meeting.mp4 --project meeting.mom` keeps the work on a meeting in one SQLite file: each run
with its settings, the diarization turns (with speaker embeddings), the speech chunks, the words with their
probabilities and the cues. The words are saved as each chunk finishes, so a run that was interrupted (killed,
crashed, failed) continues from the next chunk when you run the same command with the same settings again;
different settings start a new run. The recording is referenced, not copied: when it has moved, give its new
path (`mo-meetings transcribe new/place.mp4 --project meeting.mom`) and the project checks it is the same file
(size and a hash of its start). A project that moved together with its recording finds it on its own. With an
existing project the recording can be left out: `mo-meetings transcribe --project meeting.mom`.

Models live in `~/Library/Caches/mo-meetings/models` (macOS) `~/.cache/mo-meetings/models` (Linux) or
`%LOCALAPPDATA%\mo-meetings\models` (Windows); override
with `MO_MEETINGS_MODELS_DIR`. None of them is gated, no Hugging Face token is needed. Licenses: whisper MIT,
segmentation-3.0 MIT (© CNRS), TitaNet-Large CC-BY-4.0 (attribution NVIDIA); see [`NOTICE`](NOTICE).

## MCP server

`mo-meetings-mcp` (in the same archive, built with `cargo build --release -p mo-meetings-mcp`) exposes the pipeline
to MCP clients over stdio:

- `transcribe(path, profile?, language?, prompt?, num_speakers?, output?)` writes `<input>.vtt` and `<input>.json`
  (or next to `output`) and returns their paths, the audio duration, the number of speakers and cues, plus any
  warnings (e.g. a prompt that is too long). `language` defaults to the profile's, else `cs`; use `auto` to detect.
  With `project` (e.g. `meeting.mom`) the run is saved in that project file as with the CLI's `--project`:
  the project is created for `path` when missing, `path` can be left out once it exists (or gives the recording's
  new place when it moved), and the result adds `project_path`, `run_id` and `resumed_chunks`. A cancelled tool
  call stops after the chunk in progress; calling `transcribe` again with the same project and settings continues
  from there.
- `profiles(path?)` lists the settings profiles and, given a recording, which ones match its file name.
- `diarize(path, num_speakers?, output?)` writes speaker turns to `<input>.turns.json` and returns the path.

Long runs send MCP progress notifications (stage and transcribed chunks). Download the models first with
`mo-meetings models download`; if they are missing, the tool returns an error that says so. Pass absolute paths:
relative ones are resolved against the server's working directory.

Register it with the absolute path of the binary inside the unpacked archive (it loads its libraries from there):

```sh
claude mcp add mo-meetings -- /path/to/mo-meetings-0.1.0-aarch64-apple-darwin/mo-meetings-mcp   # Claude Code
claude mcp add --scope user mo-meetings -- /path/to/…/mo-meetings-mcp                            # all projects
```

Other clients (Claude Desktop `claude_desktop_config.json`, Cursor, VS Code, …) take the usual stdio entry; add
`MO_MEETINGS_MODELS_DIR` under `env` if the models are not in the default cache:

```json
{ "mcpServers": { "mo-meetings": { "command": "/path/to/mo-meetings-mcp", "args": [] } } }
```

A transcription can take minutes; if the client times out tool calls, raise its limit (Claude Code:
`MCP_TOOL_TIMEOUT=1800000` in milliseconds).

## Desktop app

`apps/desktop` is a Tauri v2 app (M1): open a recording, transcribe it with progress, play it next to the synced
transcript and export VTT and JSON. On first start (or when the chosen whisper model is missing) it downloads the
models into the same cache as the CLI (`MO_MEETINGS_MODELS_DIR` is respected), with progress, SHA-256 verification
and automatic retries that resume the partial file. Building it needs Rust 1.90+, Node.js 22 and, on Linux, the
[Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) (`libwebkit2gtk-4.1-dev`, `libxdo-dev`,
`librsvg2-dev`).

```sh
cd apps/desktop
npm ci
npm run tauri dev     # or `cargo tauri dev` with tauri-cli 2 installed
npm run tauri build -- --no-bundle -- --no-default-features   # Linux, Windows: CPU only
```

Releases attach the desktop app too: `mo-meetings-desktop-<version>-aarch64-apple-darwin.dmg` (Metal, with the
sherpa-onnx and onnxruntime dylibs in `Contents/Frameworks`), `.deb` and `.AppImage` for Linux x86_64 and a
`-setup.exe` (NSIS) for Windows x86_64. The macOS app is signed ad hoc only, not notarised: after downloading run
`xattr -dr com.apple.quarantine /Applications/mo-meetings.app` (or right-click → Open).

## Development

```sh
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
MO_MEETINGS_TEST_AUDIO=/path/to/meeting.mp4 cargo test --release -p mo-meetings-core --features metal -- --ignored
python3 -I scripts/eval.py LABEL dump.json teams.vtt   # score `transcribe --dump` against a Teams VTT
```

Release: commits on `main` follow [Conventional Commits](https://www.conventionalcommits.org). Run
*Actions → release-pr → Run workflow* (bump `auto`: `feat:` → minor, other changes → patch; or pick
patch/minor/major). It opens a release PR with the workspace version bump and `CHANGELOG.md` from
[git-cliff](https://git-cliff.org) ([`cliff.toml`](cliff.toml)); merging it pushes tag `vX.Y.Z`, and
`.github/workflows/release.yml` builds, smoke-tests and publishes the archives. Manual fallback: bump `version` in
the root `Cargo.toml` and merge; the tag is pushed the same way.

- `spikes/sherpa-diarization/`: does sherpa-onnx (ONNX) diarization match pyannote community-1? Results in
  [`RESULTS.md`](spikes/sherpa-diarization/RESULTS.md).

No recordings, transcripts or outputs derived from them are stored in this repo (see `.gitignore`).
