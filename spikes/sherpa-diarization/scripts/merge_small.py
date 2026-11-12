"""Post-process sherpa diarization: reassign segments of small clusters (< min_s of speech) to the nearest big
cluster centroid (cosine), using sherpa-onnx SpeakerEmbeddingExtractor. args: wav in.json out.json emb.onnx min_s"""
import json, sys, time
import numpy as np, soundfile as sf, sherpa_onnx as so
wav, dj, out, emb, min_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
d = json.load(open(dj)); a, sr = sf.read(wav, dtype="float32")
t0 = time.perf_counter()
ex = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=4))
def embed(s, e):
    st = ex.create_stream(); st.accept_waveform(sr, a[int(s*sr):int(e*sr)]); st.input_finished()
    v = np.array(ex.compute(st)); return v / np.linalg.norm(v)
dur = {}
for s, e, k in d["turns"]: dur[k] = dur.get(k, 0) + e - s
big = {k for k, v in dur.items() if v >= min_s}
segs = [t for t in d["turns"] if t[1] - t[0] >= 0.5]
E = {i: embed(s, e) for i, (s, e, k) in enumerate(segs)}
cent = {k: np.mean([E[i] * (segs[i][1]-segs[i][0]) for i in E if segs[i][2] == k], 0) for k in big}
cent = {k: v / np.linalg.norm(v) for k, v in cent.items()}
def nearest(s, e, k):
    if k in big: return k
    if e - s < 0.5:  # too short to embed: nearest-in-time big segment
        m = (s + e) / 2; return min((t for t in d["turns"] if t[2] in big), key=lambda t: min(abs(t[0]-m), abs(t[1]-m)))[2]
    v = embed(s, e); return max(cent, key=lambda c: float(v @ cent[c]))
remap = lambda L: [[s, e, nearest(s, e, k)] for s, e, k in L]
d["turns"], d["exclusive"] = remap(d["turns"]), remap(d["exclusive"])
d["merge_small_s"] = time.perf_counter() - t0; d["merge_min_s"] = min_s
json.dump(d, open(out, "w"), indent=1)
print(f"merged {len(dur)-len(big)} small clusters into {len(big)} in {d['merge_small_s']:.1f}s")
