# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2](https://github.com/mouse-over/meetings/compare/v0.2.1...v0.2.2) - 2026-11-14

### Added

- *(desktop)* Export the transcript as VTT and JSON ([#43](https://github.com/mouse-over/meetings/pull/43))
- *(desktop)* Download the models on first start, with progress and resume ([#44](https://github.com/mouse-over/meetings/pull/44))
- *(core)* Split the pipeline into steps with a word sink, cancellation and word probability ([#45](https://github.com/mouse-over/meetings/pull/45))

## [0.2.0](https://github.com/mouse-over/meetings/compare/v0.1.0...v0.2.0) - 2026-11-13

### Added

- Scaffold Tauri v2 desktop app in apps/desktop ([#21](https://github.com/mouse-over/meetings/pull/21))
- *(desktop)* Open a recording and run the pipeline with progress ([#27](https://github.com/mouse-over/meetings/pull/27))
- *(desktop)* Play the recording next to a synced, click-to-seek transcript ([#37](https://github.com/mouse-over/meetings/pull/37))

### Documentation

- Implementation roadmap with recommended order of open issues

## [0.1.0](https://github.com/mouse-over/meetings/releases/tag/v0.1.0) - 2026-11-13

### Added

- Add rust workspace with mo-meetings library and cli (M0)
- MCP server so agents can transcribe and diarize meetings locally ([#18](https://github.com/mouse-over/meetings/pull/18))
