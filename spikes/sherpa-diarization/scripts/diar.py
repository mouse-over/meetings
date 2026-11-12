"""sherpa-onnx offline diarization -> JSON in pyannote spike format. args: wav out.json emb.onnx num_clusters threshold provider threads [seg]"""
import json, resource, sys, time
import numpy as np, soundfile as sf, sherpa_onnx as so
wav, out, emb, nc, th, prov, nt = sys.argv[1:8]
seg = sys.argv[8] if len(sys.argv) > 8 else "models/seg/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
nc, th, nt = int(nc), float(th), int(nt)
t0 = time.perf_counter()
cfg = so.OfflineSpeakerDiarizationConfig(
    segmentation=so.OfflineSpeakerSegmentationModelConfig(
        pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg), num_threads=nt, provider=prov),
    embedding=so.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=nt, provider=prov),
    clustering=so.FastClusteringConfig(num_clusters=nc, threshold=th),
    min_duration_on=0.3, min_duration_off=0.5)
assert cfg.validate()
sd = so.OfflineSpeakerDiarization(cfg)
t1 = time.perf_counter()
a, sr = sf.read(wav, dtype="float32"); assert sr == sd.sample_rate
r = sd.process(a).sort_by_start_time()
t2 = time.perf_counter()
turns = [[x.start, x.end, f"SPEAKER_{x.speaker:02d}"] for x in r]
# exclusive: sherpa output is already mostly non-overlapping; resolve overlaps by cutting at the later start
ex = []
for s, e, k in sorted(turns):
    if ex and s < ex[-1][1]:
        if e <= ex[-1][1]:  # nested: split previous
            ps, pe, pk = ex.pop(); ex += [[ps, s, pk], [s, e, k], [e, pe, pk]]; continue
        ex[-1][1] = s
    ex.append([s, e, k])
ex = [t for t in ex if t[1] - t[0] > 1e-3]
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
json.dump({"load_s": t1 - t0, "diarize_s": t2 - t1, "max_rss_mb": rss, "provider": prov, "threads": nt,
           "num_clusters": nc, "threshold": th, "emb": emb, "turns": turns, "exclusive": ex}, open(out, "w"), indent=1)
print(f"load {t1-t0:.1f}s diarize {t2-t1:.1f}s rss {rss:.0f}MB speakers={len({t[2] for t in turns})}")
