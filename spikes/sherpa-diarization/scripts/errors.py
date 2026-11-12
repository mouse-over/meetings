"""Error analysis vs Teams, speakers anonymised. needs env DATA_DIR, TEAMS_VTT. args: clip|full diarA.json [diarB.json ...]"""
import collections, difflib, html, json, re, sys, unicodedata
import os
base = os.environ["DATA_DIR"].rstrip("/") + "/"  # folder with whisper JSONs
which = sys.argv[1]
wj, off, dur = ("out-turbo-carry.json", 2010, 240) if which == "clip" else ("out-full16.json", 0, 99999)
vtt = os.environ["TEAMS_VTT"]
words0 = [dict(w) for s in json.load(open(base + wj))["segments"] for w in s.get("words", [])]
def norm(w): return unicodedata.normalize("NFKD", w.lower()).encode("ascii", "ignore").decode().strip(" .,?!…-\"'")
rx = re.compile(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)\s*\n<v ([^>]+)>(.*?)(?:</v>|\n\n)", re.S)
T = lambda h, m, s, ms: int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
tc = [(T(*c[:4]) - off, html.unescape(c[8]), c[9]) for c in rx.findall(open(vtt, encoding="utf-8").read())]
tc = [c for c in tc if 0 <= c[0] < dur]
order = [k for k, _ in collections.Counter(c[1] for c in tc).most_common()]
anon = {k: "ABCDEFGH"[i] for i, k in enumerate(order)}  # A = most cues
tw = [(norm(x), anon[c[1]]) for c in tc for x in c[2].split() if norm(x)]
ww_idx = [i for i, w in enumerate(words0) if norm(w["word"])]
sm = difflib.SequenceMatcher(None, [t[0] for t in tw], [norm(words0[i]["word"]) for i in ww_idx], autojunk=False)
pairs = [(a + j, ww_idx[b + j]) for a, b, n in sm.get_matching_blocks() for j in range(n)]
# rapid-exchange flag: Teams speaker differs within +-3 words
def rapid(ti): return any(tw[k][1] != tw[ti][1] for k in range(max(0, ti-3), min(len(tw), ti+4)))
for dj in sys.argv[2:]:
    d = json.load(open(dj)); turns = d["exclusive"]; allt = d["turns"]
    def spk(a, b):
        ov = collections.Counter()
        for s, e, k in turns:
            o = min(b, e) - max(a, s)
            if o > 0: ov[k] += o
        if ov: return ov.most_common(1)[0][0]
        m = (a + b) / 2
        return min(turns, key=lambda t: min(abs(t[0]-m), abs(t[1]-m)))[2]
    def overl(a, b): return len({k for s, e, k in allt if min(b, e) - max(a, s) > 0}) > 1
    hyp = {wi: spk(words0[wi]["start"], words0[wi]["end"]) for _, wi in pairs}
    conf = collections.Counter((tw[ti][1], hyp[wi]) for ti, wi in pairs)
    mp, used = {}, set()
    for (t, p), n in conf.most_common():
        if t not in mp and p not in used: mp[t] = p; used.add(p)
    inv = {p: t for t, p in mp.items()}
    err = collections.Counter(); st = collections.Counter()
    for ti, wi in pairs:
        t = tw[ti][1]; p = hyp[wi]; ok = mp.get(t) == p
        r = rapid(ti); o = overl(words0[wi]["start"], words0[wi]["end"])
        st["rapid_n"] += r; st["rapid_err"] += r and not ok
        st["ovl_n"] += o; st["ovl_err"] += o and not ok
        st["n"] += 1; st["err"] += not ok
        if not ok: err[(t, inv.get(p, "false:" + p[-2:]))] += 1
    print(f"== {dj.split('/')[-1]}  agree={1-st['err']/st['n']:.1%}  err={st['err']}/{st['n']}")
    print(f"   words near Teams speaker change (±3 words): {st['rapid_n']}, errors there {st['rapid_err']} ({st['rapid_err']/max(1,st['err']):.0%} of all errors)")
    print(f"   words in overlapped diar regions: {st['ovl_n']}, errors there {st['ovl_err']}")
    print("   Teams talker words:", dict(collections.Counter(t[1] for t in tw)), " mapping:", {t: mp.get(t, '-') for t in sorted(anon.values())})
    print("   top confusions (true -> assigned as):", err.most_common(8))
