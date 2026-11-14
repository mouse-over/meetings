# mo-meetings

mo-meetings (mouse-over meetings): local meeting transcription with speaker diarization. A recording (Teams MP4, M4A, WAV, …) → WebVTT with
`Speaker N` labels, entirely on the machine and without Python. Long-term a desktop app in the style of
[Handy](https://github.com/cjpais/Handy) (Rust + Tauri); today a CLI and an MCP server.

Pipeline: 16 kHz mono audio → diarization (sherpa-onnx: pyannote segmentation-3.0 + NeMo TitaNet-Large,
clustering, cluster merging) → speech chunks of at most 30 s cut in pauses → whisper.cpp per chunk (glossary
prompt in every chunk) → words assigned to speakers → cues → VTT. See [`doc/architecture.md`](doc/architecture.md).

## Build

Rust 1.88+, CMake and a C/C++ toolchain (whisper.cpp is compiled from source). On macOS Metal is on by default.

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

`transcribe` options: `--language cs` (or `auto`), `--prompt "…"` / `--prompt-file`, `--num-speakers N` (a cap,
the number is estimated otherwise), `--model turbo|large`, `--threads 4`. Progress goes to stderr. Exit codes:
0 ok, 2 bad options or input, 3 model missing, 4 processing failed.

A glossary prompt helps whisper with domain terms and names; it is sent with every chunk. Example:

```text
Schůzka o vývoji aplikace. Termíny: backend, frontend, deploy, pull request, sprint, backlog, Kubernetes.
```

Models live in `~/Library/Caches/mo-meetings/models` (macOS) `~/.cache/mo-meetings/models` (Linux) or
`%LOCALAPPDATA%\mo-meetings\models` (Windows); override
with `MO_MEETINGS_MODELS_DIR`. None of them is gated, no Hugging Face token is needed. Licenses: whisper MIT,
segmentation-3.0 MIT (© CNRS), TitaNet-Large CC-BY-4.0 (attribution NVIDIA); see [`NOTICE`](NOTICE).

## MCP server

`mo-meetings-mcp` (in the same archive, built with `cargo build --release -p mo-meetings-mcp`) exposes the pipeline
to MCP clients over stdio:

- `transcribe(path, language?, prompt?, num_speakers?, output?)` writes `<input>.vtt` and `<input>.json` (or next to
  `output`) and returns their paths, the audio duration, the number of speakers and cues. `language` defaults to
  `cs`; use `auto` to detect.
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
