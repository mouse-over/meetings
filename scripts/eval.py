"""Score a `mo-meetings transcribe --dump` result against a reference Teams VTT. Prints one line of numbers only
(no names, no text), safe to paste into notes. Speaker agreement uses the spike's assemble.py unchanged.

usage: python3 -I scripts/eval.py LABEL dump.json teams.vtt
  agree      word-level speaker agreement with Teams (greedy 1:1 speaker mapping, spike metric)
  words      matched words / Teams words (how much of the Teams text whisper reproduced, ~WER proxy)
  cues       share of Teams cues with at least one output word within the cue time (+-0.5 s)
  lost       Teams speech (s) in gaps >= 10 s without any output word; number of such gaps
"""
import html, json, os, re, subprocess, sys, tempfile

label, dump, vtt = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
assemble = os.path.join(here, "..", "spikes", "sherpa-diarization", "scripts", "assemble.py")
with tempfile.TemporaryDirectory() as tmp:
    out = subprocess.run([sys.executable, "-I", assemble, dump, dump, vtt, "0", "999999", os.path.join(tmp, "a.vtt")],
                         capture_output=True, text=True, check=True).stdout
agree = re.search(r"agreement with Teams=(\d+)%", out).group(1)
matched = int(re.search(r"matched words=(\d+)", out).group(1))
unmapped = len(re.findall(r"-> -", out))
teams_spk = len(re.findall(r"^\s+.+ -> ", out, re.M))

T = lambda h, m, s, ms: int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
rx = re.compile(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)\s*\n<v ([^>]+)>(.*?)(?:</v>|\n\n)", re.S)
cues = [(T(*c[:4]), T(*c[4:8]), html.unescape(c[9])) for c in rx.findall(open(vtt, encoding="utf-8").read())]
teams_words = sum(len(c[2].split()) for c in cues)

d = json.load(open(dump))
words = sorted((w["start"], w["end"]) for s in d["segments"] for w in s["words"])
meta = d.get("meta", {})
starts = [w[0] for w in words]
import bisect
def has_word(a, b):
    i = bisect.bisect_left(starts, a - 5.0)
    return any(s <= b and e >= a for s, e in words[i:bisect.bisect_right(starts, b)])
covered = sum(has_word(a - 0.5, b + 0.5) for a, b, _ in cues)

# lost windows: gaps between consecutive output words (and edges) >= 10 s, Teams speech inside them
edges = [(-1e9, 0.0)] + words + [(1e9, 1e9)]
lost_s, lost_n = 0.0, 0
for (s0, e0), (s1, e1) in zip(edges, edges[1:]):
    if s1 - e0 >= 10:
        sp = sum(max(0.0, min(b, s1) - max(a, e0)) for a, b, _ in cues)
        if sp > 0:
            lost_s += sp; lost_n += 1

t = meta.get("timings", {})
dz = meta.get("diarization", {})
print(f"{label}\tdur={meta.get('audio_duration_s', 0) / 60:.0f}min\tagree={agree}%\tspk={meta.get('num_speakers')}"
      f"(teams {teams_spk}, unmapped {unmapped}, raw {dz.get('clusters_raw')})"
      f"\twords={matched}/{teams_words}={matched / max(teams_words, 1):.0%}\tcues={covered}/{len(cues)}={covered / max(len(cues), 1):.1%}"
      f"\tlost={lost_s:.0f}s/{lost_n}\tdiar={t.get('diarize_s', 0):.0f}s\tasr={t.get('asr_s', 0):.0f}s\ttotal={t.get('total_s', 0):.0f}s")
