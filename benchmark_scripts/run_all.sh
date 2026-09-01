#!/bin/bash
# Full clean sweep: all 7 tasks x methods at n=100, apples-to-apples.
# autoLRP configs baked in: wiki/imdb softmax=jacobian ; roberta/bert gamma=1.0 ; t5 gamma=0.001(best).
# Resumable: each cell writes results/n100/, skips if already has n>=N.
source /workspace/repro/bin/activate
export HF_HOME=/workspace/.hf_home MPLBACKEND=Agg
export PYTHONPATH=/workspace/benchmark_scripts:/workspace/benchmark:/workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/benchmark_scripts
N=100
L=/workspace/full_n100.log
RD=results/n100
mkdir -p "$RD"
echo "START $(date +%H:%M:%S)  full n=$N sweep (fix: wiki/imdb softmax=jac ; roberta/bert g=1.0 ; t5 g=0.001)" > "$L"

done_morf () { [ -f "$1" ] && python3 -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if len(d.get('records',[]))>=$N else 1)" 2>/dev/null; }
done_eqa  () { [ -f "$1" ] && python3 -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if len(d.get('per_example',[]))>=$N else 1)" 2>/dev/null; }

abpc () { python3 - "$1" "$2" "$3" <<PY >>"$L" 2>&1
import sys,json,numpy as np
from sklearn.metrics import auc
f,task,method=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(f)); v=[]
for r in d["records"]:
    a=np.array(r["morf"],float); b=np.array(r["lerf"],float); x=np.linspace(0,1,len(a)); v.append(auc(x,b)-auc(x,a))
print("CELL %-7s %-8s ABPC=%+.4f (n=%d)"%(task,method,float(np.mean(v)),len(d["records"])))
PY
}
readeqa () { python3 - "$1" "$2" "$3" <<PY >>"$L" 2>&1
import sys,json
f,task,method=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(f)); s=d.get("summary",{}); n=len(d.get("per_example",[]))
print("CELL %-7s %-8s TGS=%.4f TPS=%.4f (n=%d)"%(task,method,s.get("tgs",float('nan')),s.get("tps",float('nan')),n))
PY
}

run_morf () { local task=$1 method=$2; shift 2
  local out="$RD/${task}_${method}.json"
  if done_morf "$out"; then echo "SKIP $task $method (n>=$N)">>"$L"; abpc "$out" "$task" "$method"; return; fi
  echo ">>> MORF $task $method $(date +%H:%M:%S)">>"$L"
  python -u run_morf_lerf.py --task "$task" --method "$method" --n $N --max-steps 256 \
     --autolrp-repo /workspace --out-json "$out" "$@" >>"$L" 2>&1
  abpc "$out" "$task" "$method"
}
run_eqa () { local task=$1 method=$2; shift 2
  local out="$RD/${task}_${method}_tgstps.json"
  if done_eqa "$out"; then echo "SKIP $task $method (n>=$N)">>"$L"; readeqa "$out" "$task" "$method"; return; fi
  echo ">>> EQA $task $method $(date +%H:%M:%S)">>"$L"
  python -u run_tgs_tps.py --task "$task" --method "$method" --n $N \
     --autolrp-repo /workspace --out-json "$out" "$@" >>"$L" 2>&1
  readeqa "$out" "$task" "$method"
}

# ---- EQA first (fast) ----
run_eqa roberta attnlrp
run_eqa roberta autolrp --gamma-linear 1.0
run_eqa bert    attnlrp
run_eqa bert    autolrp --gamma-linear 1.0
run_eqa t5      attnlrp
run_eqa t5      autolrp --gamma-linear 0.001
# ---- LLaMA text (headline) ----
run_morf wiki attnlrp
run_morf wiki autolrp --softmax jacobian
run_morf imdb attnlrp
run_morf imdb autolrp --softmax jacobian
# ---- Vision ----
run_morf vit  attnlrp
run_morf vit  autolrp
run_morf vgg  attnlrp
run_morf vgg  autolrp
echo "FULL_DONE $(date +%H:%M:%S)" >>"$L"
