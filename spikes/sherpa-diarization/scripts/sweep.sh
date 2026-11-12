#!/bin/zsh
# run from the spike dir (needs .venv, models/, out/; env DATA_DIR, TEAMS_VTT). usage: sweep.sh clip|full "emb1 emb2" "nc:th ..." provider threads
which=$1; embs=(${=2}); vars=(${=3}); prov=${4:-cpu}; nt=${5:-4}
wav=$DATA_DIR/clip16.wav; [[ $which == full ]] && wav=$DATA_DIR/full16.wav
for e in $embs; do for v in $vars; do
  nc=${v%%:*}; th=${v##*:}; o=out/$which-$e-nc$nc-th$th-$prov.json
  [[ -f $o ]] || .venv/bin/python -I scripts/diar.py $wav $o models/emb/$e.onnx $nc $th $prov $nt >/dev/null 2>out/err.log || { echo "FAIL $o"; continue; }
  .venv/bin/python -I scripts/score.py $which $o | tee -a out/results.tsv
done; done
