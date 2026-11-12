"""Run assemble.py (needs env DATA_DIR, TEAMS_VTT) on a diar json, emit one summary line (no names). args: which(clip|full) diar.json"""
import json, re, subprocess, sys
which, dj = sys.argv[1], sys.argv[2]
import os
base = os.environ["DATA_DIR"]  # folder with whisper JSONs (out-turbo-carry.json, out-full16.json)
vtt = os.environ["TEAMS_VTT"]  # reference Teams transcript
here = os.path.dirname(os.path.abspath(__file__))
w, off, dur = ("out-turbo-carry.json", "2010", "240") if which == "clip" else ("out-full16.json", "0", "99999")
o = subprocess.run([sys.executable, here + "/assemble.py", base + "/" + w, dj, vtt, off, dur, "/tmp/sherpa-score.vtt"],
                   capture_output=True, text=True, check=True).stdout
agr = re.search(r"agreement with Teams=(\d+)%", o).group(1)
mapped = set(re.findall(r"-> (Speaker \d+)", o))
sp = eval(re.search(r"speech per label \(s\): (\{.*\})", o).group(1))
false = sorted((v for k, v in sp.items() if k not in mapped), reverse=True)
unmapped_teams = len(re.findall(r"-> -", o))
d = json.load(open(dj))
print(f"{which}\t{d['emb'].split('/')[-1].replace('.onnx','')}\tnc={d['num_clusters']}\tth={d['threshold']}\tagree={agr}%\tspk={len(sp)}\tfalse_s={false}\tteams_unmapped={unmapped_teams}\tdiar={d['diarize_s']:.1f}s\trss={d.get('max_rss_mb',0):.0f}MB\t{d['provider']}/{d['threads']}t")
