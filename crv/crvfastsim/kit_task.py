"""fastsimkit task `crv_response`: one CRV response-tuning point of the fast chain (reco-r3) scored against KPP data.

Registered in mu2efastsim.endpoints.kit by `register(...)` when crvfastsim is importable. FNAL endpoints only (EAF): the
CrvStep tables, CRVCalib/CRVPhoton CSVs, background model, time calibration and data reference are Mu2e-internal and
read in place under the endpoint's fnal_data; the lookup tables are fetched from CVMFS into node tmpfs (copy-compute-
delete, cleanup record). One point = one reco-r3 process on one GPU: `--jax --objective --v05` on the cached MDC2025as
CrvSteps, then crvresponse.objective against the period's data reference.

Params
  every chain.KNOBS name (speed_EX, atten_T10, photon_yield_scale, ...): optional; absent or empty = nominal
  period      P4 (run 125026, module time calibration corr_P4_v2mod; default), P3 (124336 + 124571, corr_P3_v2mod,
              dt_cluster_T2 excluded: the P3 T2 timing defect) or raw (merged 3-run reference, no time calibration)
  tables      vr5 (the five V-R5 / fiberprop_v1 files 990-1122, 112,304 events with CRV steps; default) or all
              (the 20 prod_as tables)
  events      events with CRV steps to process (default 0 = one pass over the tables)
  background  1 = add the out-of-time background model (background/fit_v1/bg.npz; default), 0 = none
  seed
Metrics: objective (size-free total of the used terms) and objective_sigma, objective_all (every term), n_terms,
events_with_steps, wall_s, and term_<name> (chi2ndf scale) for every objective term.
"""
from . import chain
from .objective import crvresponse as R

KNOB_PARAMS = tuple(sorted(chain.KNOBS))
FIXED = ("period", "tables", "events", "background", "seed")
PARAMS = KNOB_PARAMS + FIXED
DEFAULTS = {**{k: "" for k in KNOB_PARAMS}, "period": "P4", "tables": "vr5", "events": 0, "background": 1}
# nlayers of T1, T2 and the taggers have one populated bin (in-time clusters always have every layer): no ndf, no term
NO_TERM = ("nlayers_T1", "nlayers_T2", "nlayers_TAG")
TERMS = tuple(sorted([f"{n}_{s}" for n in R.SHAPES for s in R.SLOTS_OF.get(n, R.SLOTS) if f"{n}_{s}" not in NO_TERM]
                     + [f"oot_rate_{s}" for s in R.SLOTS]))
METRICS = ("objective", "objective_sigma", "objective_all", "n_terms", "events_with_steps", "wall_s") + \
    tuple(f"term_{t}" for t in TERMS)
VR5_TABLES = ("00000990", "00001023", "00001056", "00001089", "00001122")
VR5_EVENTS = 112304


def _knob_lines():
    """One bash line per knob: append NAME=VALUE to KN when the param is non-empty."""
    return "".join(f'[ -n @p_{k}@ ] && KN+=("{k}=@p_{k}@")\n' for k in KNOB_PARAMS)


BODY_HEAD = r'''D=@fnal_data@
I=$D/prod_as_tables; G=$D/geometry; CAL=$G/crvcalib_sim_best_v1_5_run1400.csv; DEVF=$G/crvphoton_sim_best_v1_5_run1400.csv
case @p_period@ in
  P4) TS=$D/timecal/corr_P4_v2mod.json; REF="$D/r3/dataref/125026/*.json"; EXCL="" ;;
  P3) TS=$D/timecal/corr_P3_v2mod.json; REF="$D/r3/dataref/124336/*.json $D/r3/dataref/124571/*.json"; EXCL="dt_cluster_T2" ;;
  raw) TS=""; REF="$D/r3/dataref/dataref_3runs.json"; EXCL="" ;;
  *) echo "unknown period @p_period@ (P4, P3, raw)"; exit 2 ;;
esac
case @p_tables@ in
  vr5) FILES=(); for q in ''' + " ".join(VR5_TABLES) + r'''; do FILES+=("$I/crvsteps.001400_$q.root"); done; NDEF=''' + str(VR5_EVENTS) + r''' ;;
  all) FILES=("$I"/crvsteps.001400_*.root); NDEF=0 ;;
  *) echo "unknown tables @p_tables@ (vr5, all)"; exit 2 ;;
esac
for f in "${FILES[@]}" "$CAL" "$DEVF" ${TS:+"$TS"}; do [ -r "$f" ] || { echo "missing $f"; exit 3; }; done
N=@p_events@; [ "$N" = 0 ] && N=$NDEF
[ "$N" -gt 0 ] || { echo "events must be > 0 with tables=@p_tables@"; exit 2; }
BG=(); [ @p_background@ = 1 ] && BG=(--background "$D/background/fit_v1/bg.npz")
TSA=(); [ -n "$TS" ] && TSA=(--time-shift "$TS")
KN=()
'''

BODY_TAIL = r'''KA=(); [ ${#KN[@]} -gt 0 ] && KA=(--knob "${KN[@]}")
echo "period @p_period@ tables @p_tables@ (${#FILES[@]}) events $N background @p_background@ knobs ${KN[*]:-nominal}"
T python -m crvfastsim reco-r3 "${FILES[@]}" --calib "$CAL" --deviation "$DEVF" --jax --objective --v05 --target "$N" \
  --seed @p_seed@ "${BG[@]}" "${TSA[@]}" "${KA[@]}" --emit "$M2FS_OUT/r3.json" \
  > "$M2FS_OUT/r3.txt" 2>&1 || { tail -30 "$M2FS_OUT/r3.txt"; exit 6; }
tail -2 "$M2FS_OUT/r3.txt"
A python - "$M2FS_OUT" "$EXCL" $REF <<'EOF'
import glob, json, os, sys
from crvfastsim.kit_task import TERMS
from crvfastsim.objective import crvresponse as R
o, excl, refs = sys.argv[1], [x for x in sys.argv[2].split(",") if x], sys.argv[3:]
j = json.load(open(os.path.join(o, "r3.json")))
files = sorted(f for r in refs for f in (glob.glob(r) or [r]))
ref = R.merge_states([json.load(open(f)) for f in files])
ob = R.objective(j["response_state"], ref, exclude=excl)
m = {"objective": ob["objective"], "objective_sigma": ob["sigma"], "objective_all": ob["objective_all"],
     "n_terms": ob["n_terms"], "events_with_steps": j["events_with_steps"], "wall_s": j["wall_s"]}
m.update({"term_" + k: v["chi2ndf"] for k, v in ob["terms"].items()})
missing = [t for t in TERMS if "term_" + t not in m]
if missing:
    print("terms missing (empty histogram): " + " ".join(missing))
json.dump({"knobs": j.get("knobs", {}), "reference_files": len(files), "excluded": ob["excluded"],
           "objective": ob}, open(os.path.join(o, "objective.json"), "w"), indent=1)
json.dump({k: float(v) for k, v in m.items()}, open(os.path.join(o, "metrics.json"), "w"))
print(f"objective {m['objective']:.3f} +- {m['objective_sigma']:.3f} ({m['n_terms']:.0f} terms, {len(files)} reference files)")
EOF
'''


def register(tasks, task_cls, resources_cls, fnal_tasks, table_tasks=None, common="", tables=""):
    """Add crv_response to a fastsimkit TASKS dict. `common` and `tables` are the kit's GPU-check and copy-compute-
    delete snippets (kit._COMMON, kit._TABLES), so the lookup tables and photon map land in node tmpfs."""
    tab = tables.replace("@tables@", "LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root")
    tasks["crv_response"] = task_cls(
        params=PARAMS, metrics=METRICS,
        resources=resources_cls(cores=8, mem_gb=48, gpus=1, gpu_type="a100", minutes=60),
        body=common + tab + BODY_HEAD + _knob_lines() + BODY_TAIL, defaults=dict(DEFAULTS))
    fnal_tasks.add("crv_response")
    if table_tasks is not None:
        table_tasks.add("crv_response")
