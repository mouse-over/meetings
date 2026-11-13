# M0 evaluation

Date 2026-11-13, Apple M5 (4P + 6E cores, 32 GB), release build, whisper large-v3-turbo on Metal, diarization on
CPU with 4 threads, Czech, glossary prompt. Reference: the Teams VTT of each recording. Recordings are anonymised
(A–D); A is the recording from the diarization spike. Measured with `scripts/eval.py` (speaker agreement via the
spike's `assemble.py`).

| | Length | Speakers (Teams) | Speaker agreement | Teams cues covered | Speech lost in gaps ≥ 10 s | Diarization | ASR | Total | RTF |
|---|---|---|---|---|---|---|---|---|---|
| A | 49 min | 5 (5) | **97 %** | 97.9 % | 2 s | 185 s | 284 s | 470 s | 0.16 |
| B | 59 min | 6 (7) | **97 %** | 98.7 % | 2 s | 213 s | 335 s | 549 s | 0.16 |
| C | 47 min | 3 (4) | **97 %** | 99.4 % | 3 s | 169 s | 291 s | 462 s | 0.16 |
| D | 37 min | 3 (3) | **97 %** | 95.2 % | 9 s | 122 s | 187 s | 315 s | 0.14 |

- Agreement: word-level speaker match after a greedy 1:1 mapping of speakers (spike metric). The spike reached
  94–95 % on A (pyannote community-1: 95 %).
- Missed speakers: B's 7th Teams speaker has 4 words; C's 4th has 27 words and is merged into another speaker.
- Uncovered Teams cues are all short backchannels (≤ 0.8 s, 1–3 words in D); no lost transcription windows.
- Matched words (exact match after normalisation, Teams ASR vs whisper) are 57–65 %; this mixes errors of both
  transcripts and is only a rough proxy.
- Peak memory 4.3–4.9 GB for the whole process (whisper turbo on Metal + diarization). Diarization alone on D:
  2.6 GB with Microsoft's onnxruntime, 1.6 GB with the slower one bundled in the sherpa-onnx release.
- Diarization time was 288 s instead of 124 s on D with the onnxruntime from the sherpa-onnx release; see
  `doc/architecture.md`.
