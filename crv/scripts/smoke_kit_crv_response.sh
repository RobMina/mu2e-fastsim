#!/bin/bash
# Smoke test of the fastsimkit task crv_response on a GPVM CPU (no EAF): render the task for a few events with a local
# runtime (A/T = the pyenv-ana python, CVMFS lookup tables), drop --jax (no JAX in pyenv ana) and run the body.
# usage: smoke_kit_crv_response.sh CODE_DIR OUT_DIR [EVENTS=300] [KNOB=VALUE ...]
CODE=${1:?umbrella-like dir with core/ and crv/}; OUT=${2:?out dir}; N=${3:-300}; shift 3 || shift $#
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh >/dev/null 2>&1; pyenv ana >/dev/null 2>&1
mkdir -p "$OUT" && rm -f "$OUT"/{metrics,objective,r3}.json
KN="{}"; [ $# -gt 0 ] && KN=$(python3 -c 'import json,sys; print(json.dumps({a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[1:]}))' "$@")
PYTHONPATH=$CODE/core:$CODE/crv python3 - "$OUT" "$N" "$KN" > "$OUT/job.sh" <<'EOF'
import json, sys
from mu2efastsim.endpoints import kit as K
out, n, kn = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
class Ep:
    name = "smoke"; fnal_data = "/exp/mu2e/data/users/ram2aq/crv_fastsim"
    def runtime(self):
        return {"a": 'A() { local c=$1; shift; [ "$c" = python ] && c=python3; "$c" "$@"; }\n',
                "t": 'T() { ( export CRVFS_LOOKUP_DIR="$L"; A "$@" ); }'}
body, _ = K.FastsimKit(endpoint=Ep(), state_dir=out + "/kit").render({"task": "crv_response", "seed": 1, "events": n, **kn})
body = body.replace(" --jax ", " ").replace("module load apptainer >/dev/null 2>&1\n", "")
body = body.replace("nvidia-smi -L", "true").replace("sys.exit(jax.default_backend() != 'gpu')", "sys.exit(0)")
print(body)
EOF
cd "$OUT" && M2FS_OUT="$OUT" PYTHONPATH=$CODE/core:$CODE/crv bash job.sh > job.log 2>&1; echo "job rc=$?"
tail -4 job.log; ls "$OUT"; python3 -c "import json; m=json.load(open('$OUT/metrics.json')); print(len(m), 'metrics; objective', m['objective'], '+-', m['objective_sigma'], 'events', m['events_with_steps'])"
