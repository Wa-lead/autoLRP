#!/bin/bash
# v1 library verification: autolrp on all 7 tasks at n=100, same configs/seeds as
# the v0 run. Compare to saved v0 numbers (results/n100):
#   roberta 0.909/1.000  bert 0.800/1.000  t5 0.911/0.978
#   wiki +0.3972  imdb +0.2707  vit +0.4127  vgg +0.4131
source /workspace/repro/bin/activate
export HF_HOME=/workspace/.hf_home MPLBACKEND=Agg
export PYTHONPATH=/workspace/benchmark_scripts:/workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/benchmark_scripts
N=100
L=/workspace/v1_verify.log
RD=results/v1
mkdir -p "$RD"
echo "START $(date +%H:%M:%S)  v1 verification sweep (autolrp only, n=$N)" > "$L"

abpc () { python3 - "$1" "$2" <<PY >>"$L" 2>&1
import sys,json,numpy as np
from sklearn.metrics import auc
f,task=sys.argv[1],sys.argv[2]
d=json.load(open(f)); v=[]
for r in d["records"]:
    a=np.array(r["morf"],float); b=np.array(r["lerf"],float); x=np.linspace(0,1,len(a)); v.append(auc(x,b)-auc(x,a))
print("CELL %-7s autolrp ABPC=%+.4f (n=%d)"%(task,float(np.mean(v)),len(d["records"])))
PY
}
readeqa () { python3 - "$1" "$2" <<PY >>"$L" 2>&1
import sys,json
f,task=sys.argv[1],sys.argv[2]
d=json.load(open(f)); s=d.get("summary",{}); n=len(d.get("per_example",[]))
print("CELL %-7s autolrp TGS=%.4f TPS=%.4f (n=%d)"%(task,s.get("tgs",float('nan')),s.get("tps",float('nan')),n))
PY
}

run_eqa () { local task=$1; shift
  local out="$RD/${task}_autolrp_tgstps.json"
  echo ">>> EQA $task $(date +%H:%M:%S)">>"$L"
  python -u run_tgs_tps.py --task "$task" --method autolrp --n $N \
     --autolrp-repo /workspace --out-json "$out" "$@" >>"$L" 2>&1
  readeqa "$out" "$task"
}
run_morf () { local task=$1; shift
  local out="$RD/${task}_autolrp.json"
  echo ">>> MORF $task $(date +%H:%M:%S)">>"$L"
  python -u run_morf_lerf.py --task "$task" --method autolrp --n $N --max-steps 256 \
     --autolrp-repo /workspace --out-json "$out" "$@" >>"$L" 2>&1
  abpc "$out" "$task"
}

run_eqa roberta --gamma-linear 1.0
run_eqa bert    --gamma-linear 1.0
run_eqa t5      --gamma-linear 0.001
run_morf wiki
run_morf imdb
run_morf vit
run_morf vgg --imagenet-dir /workspace/imagenette2-320/val
echo "V1_VERIFY_DONE $(date +%H:%M:%S)" >>"$L"
