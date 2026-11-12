"""Assign diarization speakers to whisper words, write Teams-shaped VTT, score vs Teams speakers."""
import collections, difflib, html, json, re, sys, unicodedata
wj, dj, vtt_in, off, dur, vtt_out = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
turns = json.load(open(dj))["exclusive"]
words = [dict(w) for s in json.load(open(wj))["segments"] for w in s.get("words", [])]
def spk(a, b):
    ov = collections.Counter()
    for s, e, k in turns:
        o = min(b, e) - max(a, s)
        if o > 0: ov[k] += o
    if ov: return ov.most_common(1)[0][0]
    m = (a + b) / 2  # fill nearest
    return min(turns, key=lambda t: min(abs(t[0] - m), abs(t[1] - m)))[2]
for w in words: w["spk"] = spk(w["start"], w["end"])
# relabel SPEAKER_xx -> Speaker N by first appearance
order = list(dict.fromkeys(w["spk"] for w in words))
lab = {k: f"Speaker {i+1}" for i, k in enumerate(order)}
# cues: break on speaker change, pause > 1s, sentence end after >= 3s, or > 7s
cues, cur = [], []
for w in words:
    if cur and (w["spk"] != cur[-1]["spk"] or w["start"] - cur[-1]["end"] > 1.0
                or w["start"] - cur[0]["start"] > 7.0
                or (cur[-1]["word"].rstrip().endswith((".", "?", "!")) and cur[-1]["end"] - cur[0]["start"] > 3.0)):
        cues.append(cur); cur = []
    cur.append(w)
cues.append(cur)
ts = lambda t: f"{int(t//3600):02d}:{int(t%3600//60):02d}:{t%60:06.3f}"
with open(vtt_out, "w", encoding="utf-8") as f:
    f.write("WEBVTT\n\n")
    for i, c in enumerate(cues):
        txt = "".join(w["word"] for w in c).strip()
        f.write(f"{i+1}\n{ts(c[0]['start']+off)} --> {ts(c[-1]['end']+off)}\n<v {lab[c[0]['spk']]}>{html.escape(txt, quote=False)}</v>\n\n")
# score vs Teams
def norm(w): return unicodedata.normalize("NFKD", w.lower()).encode("ascii", "ignore").decode().strip(" .,?!…-\"'")
rx = re.compile(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)\s*\n<v ([^>]+)>(.*?)(?:</v>|\n\n)", re.S)
T = lambda h, m, s, ms: int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
tc = [(T(*c[:4]) - off, html.unescape(c[8]), c[9]) for c in rx.findall(open(vtt_in, encoding="utf-8").read())]
tc = [c for c in tc if 0 <= c[0] < dur]
tw = [(norm(x), c[1]) for c in tc for x in c[2].split() if norm(x)]
ww = [(norm(w["word"]), w["spk"]) for w in words if norm(w["word"])]
sm = difflib.SequenceMatcher(None, [t[0] for t in tw], [w[0] for w in ww], autojunk=False)
pairs = [(tw[a+j][1], ww[b+j][1]) for a, b, n in sm.get_matching_blocks() for j in range(n)]
conf = collections.Counter(pairs)
# greedy one-to-one mapping by co-occurrence
mp, used = {}, set()
for (t, p), n in conf.most_common():
    if t not in mp and p not in used: mp[t] = p; used.add(p)
ok = sum(n for (t, p), n in conf.items() if mp.get(t) == p)
print(f"matched words={len(pairs)}  attribution agreement with Teams={ok/len(pairs):.0%}")
print("Teams words per speaker:", dict(collections.Counter(t[1] for t in tw)))
for t in sorted({t for t, _ in pairs}):
    row = {lab[p]: n for (tt, p), n in conf.items() if tt == t}
    print(f"  {t[:22]:22} -> {lab.get(mp.get(t), '-'):10} {row}")
print("pyannote speech per label (s):", {lab.get(k, k): round(sum(e - s for s, e, kk in turns if kk == k), 1) for k in sorted({t[2] for t in turns})})
print(f"cues written={len(cues)}")
