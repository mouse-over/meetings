# sherpa-onnx diarization spike

Measurement only (Python bindings of sherpa-onnx), results in [`RESULTS.md`](RESULTS.md).
Meeting audio, whisper outputs and Teams transcripts are local and not part of the repo.

## Reproduce

```sh
uv venv -p 3.12 .venv && uv pip install -p .venv sherpa-onnx soundfile numpy
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
mkdir -p models/seg models/emb out
(cd models/seg && curl -LO $B/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 && tar xjf *.tar.bz2)
(cd models/emb && curl -LO $B/speaker-recongition-models/nemo_en_titanet_large.onnx)

export DATA_DIR=/path/to/data   # clip16.wav, full16.wav, out-turbo-carry.json, out-full16.json
export TEAMS_VTT=/path/to/teams.vtt

# diarize: wav out.json emb.onnx num_clusters(-1 = threshold) threshold provider threads
.venv/bin/python -I scripts/diar.py $DATA_DIR/full16.wav out/full.json models/emb/nemo_en_titanet_large.onnx -1 1.05 cpu 4
# merge clusters with < 20 s of speech into the nearest big one
.venv/bin/python -I scripts/merge_small.py $DATA_DIR/full16.wav out/full.json out/full-merge20.json models/emb/nemo_en_titanet_large.onnx 20
# score vs Teams (prints no names), error analysis with anonymised speakers
.venv/bin/python -I scripts/score.py full out/full-merge20.json
.venv/bin/python -I scripts/errors.py full out/full-merge20.json
```

`scripts/assemble.py <whisper.json> <diar.json> <teams.vtt> <offset_s> <dur_s> <out.vtt>` assigns whisper words to
speakers and computes word-level agreement with Teams; note that it prints Teams speaker names, so keep its raw output local.
