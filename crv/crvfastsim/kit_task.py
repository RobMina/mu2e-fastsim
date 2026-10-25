"""fastsimkit task `crv_response`: one CRV response-tuning point of the fast chain (reco-r3) scored against KPP data.

Registered in mu2efastsim.endpoints.kit by `register(...)` when crvfastsim is importable. One point = one reco-r3 process
on one GPU: `--jax --objective --v05` on the cached MDC2025as CrvSteps, then crvresponse.objective against the period's
data reference. The inputs (CrvStep tables, CRVCalib/CRVPhoton CSVs, background model, time calibration, data
reference) are Mu2e-internal:
  - on an endpoint that mounts FNAL storage (EAF) the rendered body reads them in place (X = fnal_data);
  - on a Slurm-over-SSH endpoint (UVA; crvfastsim.kit_uva) the kit streams exactly `inputs(params)` from the FNAL host
    into the job's tmpfs (X = that dir), md5-checked, deleted by the job's EXIT trap (cleanup record).
The lookup tables come from CVMFS into node tmpfs on both (copy-compute-delete).

Params
  every chain.KNOBS name (speed_EX, atten_T10, photon_yield_scale, ...): optional; absent or empty = nominal
  period      P4 (run 125026, module time calibration corr_P4_v2mod; default), P3 (124336 + 124571, corr_P3_v2mod,
              dt_cluster_T2 excluded: the P3 T2 timing defect) or raw (merged 3-run reference, no time calibration)
  tables      vr5 (the five V-R5 / fiberprop_v1 files 990-1122, 112,304 events with CRV steps; default) or all
              (the 20 prod_as tables; EAF only)
  events      events with CRV steps to process (default 0 = one pass over the tables)
  background  1 = add the out-of-time background model (background/fit_v1/bg.npz; default), 0 = none
  seed
Metrics: objective (size-free total of the used terms) and objective_sigma, objective_all (every term), n_terms,
events_with_steps, wall_s, and term_<name> (chi2ndf scale) for every objective term.
"""
# Static names (no numpy/awkward import: the kit process on a laptop may lack them); tests/test_kit_uva.py keeps them
# equal to chain.KNOBS and the crvresponse shapes.
_PROP_KEYS = (("EX", "T1", "T2", "TAG") + tuple(f"EX{i}" for i in range(4)) + ("T10", "T11", "T20", "T21")
              + tuple(f"M{i}" for i in range(1, 9)))
KNOB_PARAMS = tuple(sorted(("background_scale", "cross_talk", "min_voltage", "pe_stretch", "photon_yield_scale",
                            "time_constant", "wf_noise") + tuple(f"yield_{n}" for n in ("EX", "T1", "T2", "TAG"))
                           + tuple(f"{q}_{n}" for q in ("speed", "atten") for n in _PROP_KEYS)))
_SLOTS = ("EX", "T1", "T2", "TAG")
_SHAPES = {"pe_along": _SLOTS, "pe_path": _SLOTS, "asym": ("EX", "T2"), "tdiff": ("EX", "T2"), "tspread": _SLOTS,
           "dt_cluster": ("T1", "T2", "TAG"), "nlayers": _SLOTS, "pe_layer": _SLOTS, "oot_pe": _SLOTS}
FIXED = ("period", "tables", "events", "background", "seed")
PARAMS = KNOB_PARAMS + FIXED
DEFAULTS = {**{k: "" for k in KNOB_PARAMS}, "period": "P4", "tables": "vr5", "events": 0, "background": 1}
# nlayers of T1, T2 and the taggers have one populated bin (in-time clusters always have every layer): no ndf, no term
NO_TERM = ("nlayers_T1", "nlayers_T2", "nlayers_TAG")
TERMS = tuple(sorted([f"{n}_{s}" for n, sl in _SHAPES.items() for s in sl if f"{n}_{s}" not in NO_TERM]
                     + [f"oot_rate_{s}" for s in _SLOTS]))
METRICS = ("objective", "objective_sigma", "objective_all", "n_terms", "events_with_steps", "wall_s", "input_wait_s",
           "inputs_reused") + \
    tuple(f"term_{t}" for t in TERMS)
VR5_TABLES = ("00000990", "00001023", "00001056", "00001089", "00001122")
VR5_EVENTS = 112304

# inputs relative to the FNAL data dir (crv_fastsim)
STEPS = tuple(f"prod_as_tables/crvsteps.001400_{q}.root" for q in VR5_TABLES)
CALIB = "geometry/crvcalib_sim_best_v1_5_run1400.csv"
DEVIATION = "geometry/crvphoton_sim_best_v1_5_run1400.csv"
BACKGROUND = "background/fit_v1/bg.npz"
SIZE_MB = {**{f: 100 for f in STEPS}, BACKGROUND: 58}        # streaming order (largest first); others < 1 MB
PERIODS = {"P4": ("timecal/corr_P4_v2mod.json", "r3/dataref/dataref_P4.json", ""),
           "P3": ("timecal/corr_P3_v2mod.json", "r3/dataref/dataref_P3.json", "dt_cluster_T2"),
           "raw": ("", "r3/dataref/dataref_3runs.json", "")}


def score_result(result, reference, exclude=()):
    """Score only a complete, lossless R3 result against the frozen response-term manifest."""
    from .objective import crvresponse as R
    if result["events_with_steps"] != result["target"] or result["target"] <= 0:
        raise ValueError("crv_response: incomplete event target")
    if result.get("events_without_steps") is None:
        raise ValueError("crv_response: missing generated-event exposure")
    if result["timers"].get("response_overflow_batches", 0):
        raise ValueError("crv_response: residual response overflow; refusing to score truncated output")
    return R.objective(result["response_state"], reference, exclude=exclude, required=TERMS)


def _base(rel):
    return rel.rsplit("/", 1)[1]


def inputs(params):
    """{FNAL relative path: file name in the job's input dir} that a point reads (UVA streams exactly these)."""
    period = params.get("period", DEFAULTS["period"]) or DEFAULTS["period"]
    if period not in PERIODS:
        raise ValueError(f"crv_response: period must be one of {sorted(PERIODS)}, got {period!r}")
    if (params.get("tables", DEFAULTS["tables"]) or "vr5") != "vr5":
        raise ValueError("crv_response: tables=all reads 20 tables in place (EAF only); stream tables=vr5")
    ts, ref, _ = PERIODS[period]
    files = STEPS + (CALIB, DEVIATION, ref) + ((ts,) if ts else ())
    if int(float(params.get("background", DEFAULTS["background"]))):
        files += (BACKGROUND,)
    return {f: _base(f) for f in files}


def _period_case():
    rows = []
    for k, (ts, ref, excl) in PERIODS.items():
        t = f'TS=$(P {ts})' if ts else 'TS=""'
        rows.append(f'  {k}) {t}; REF=$(P {ref}); EXCL="{excl}" ;;')
    return "\n".join(rows)


BODY_HEAD = ("D=@fnal_data@; X=@inputs_dir@   # X: the inputs (D itself on EAF, the streamed tmpfs dir on UVA)\n"
             'P() { if [ "$X" = "$D" ]; then echo "$D/$1"; else echo "$X/${1##*/}"; fi; }\n'
             f"CAL=$(P {CALIB}); DEVF=$(P {DEVIATION})\n"
             "case @p_period@ in\n" + _period_case() + "\n"
             '  *) echo "unknown period @p_period@ (P4, P3, raw)"; exit 2 ;;\nesac\n'
             "case @p_tables@ in\n"
             f'  vr5) FILES=(); for q in {" ".join(VR5_TABLES)}; do FILES+=("$(P prod_as_tables/crvsteps.001400_$q.root)"); done; NDEF={VR5_EVENTS} ;;\n'
             '  all) FILES=("$D"/prod_as_tables/crvsteps.001400_*.root); NDEF=0 ;;\n'
             '  *) echo "unknown tables @p_tables@ (vr5, all)"; exit 2 ;;\nesac\n'
             'for f in "${FILES[@]}" "$CAL" "$DEVF" "$REF" ${TS:+"$TS"}; do [ -r "$f" ] || { echo "missing $f"; exit 3; }; done\n'
             'N=@p_events@; [ "$N" = 0 ] && N=$NDEF\n'
             '[ "$N" -gt 0 ] || { echo "events must be > 0 with tables=@p_tables@"; exit 2; }\n'
             f'BG=(); [ @p_background@ = 1 ] && BG=(--background "$(P {BACKGROUND})")\n'
             'TSA=(); [ -n "$TS" ] && TSA=(--time-shift "$TS")\n'
             "KN=()\n")


def _knob_lines():
    """One bash line per knob: append NAME=VALUE to KN when the param is non-empty."""
    return "".join(f'[ -n @p_{k}@ ] && KN+=("{k}="@p_{k}@)\n' for k in KNOB_PARAMS)


BODY_TAIL = r'''KA=(); [ ${#KN[@]} -gt 0 ] && KA=(--knob "${KN[@]}")
echo "period @p_period@ tables @p_tables@ (${#FILES[@]}) events $N background @p_background@ knobs ${KN[*]:-nominal}"
T python -m crvfastsim reco-r3 "${FILES[@]}" --calib "$CAL" --deviation "$DEVF" --jax --objective --v05 --target "$N" \
  --seed @p_seed@ "${BG[@]}" "${TSA[@]}" "${KA[@]}" --emit "$M2FS_OUT/r3.json" \
  > "$M2FS_OUT/r3.txt" 2>&1 || { tail -30 "$M2FS_OUT/r3.txt"; exit 6; }
tail -2 "$M2FS_OUT/r3.txt"
A python - "$M2FS_OUT" "$EXCL" "$REF" <<'EOF'
import json, os, sys
from crvfastsim.kit_task import score_result
o, excl, ref_path = sys.argv[1], [x for x in sys.argv[2].split(",") if x], sys.argv[3]
j = json.load(open(os.path.join(o, "r3.json")))
ob = score_result(j, json.load(open(ref_path)), exclude=excl)
m = {"objective": ob["objective"], "objective_sigma": ob["sigma"], "objective_all": ob["objective_all"],
     "n_terms": ob["n_terms"], "events_with_steps": j["events_with_steps"], "wall_s": j["wall_s"]}
m.update({"term_" + k: v["chi2ndf"] for k, v in ob["terms"].items()})
json.dump({"knobs": j.get("knobs", {}), "reference": os.path.basename(ref_path), "excluded": ob["excluded"],
           "objective": ob}, open(os.path.join(o, "objective.json"), "w"), indent=1)
json.dump({k: float(v) for k, v in m.items()}, open(os.path.join(o, "metrics.json"), "w"))
print(f"objective {m['objective']:.3f} +- {m['objective_sigma']:.3f} ({m['n_terms']:.0f} terms, {os.path.basename(ref_path)})")
EOF
'''

LOOKUP = "LookupTable_6000_0 LookupTable_2370_1 LookupTable_3200_0 photonMap.root"


def body(common, tables):
    """The point's bash: the kit's GPU check (common), lookup tables to tmpfs (tables), then the reco and objective.
    @inputs_dir@ is left for the caller (EAF: @fnal_data@; UVA: the streamed tmpfs dir)."""
    return common + tables.replace("@tables@", LOOKUP) + BODY_HEAD + _knob_lines() + BODY_TAIL


def register(tasks, task_cls, resources_cls, fnal_tasks, table_tasks=None, common="", tables="", delegates=None):
    """Add crv_response to a fastsimkit TASKS dict (FNAL endpoints: inputs in place) and, with `delegates`, the
    streaming handler for Slurm-over-SSH endpoints (crvfastsim.kit_uva)."""
    tasks["crv_response"] = task_cls(
        params=PARAMS, metrics=METRICS,
        resources=resources_cls(cores=8, mem_gb=48, gpus=1, gpu_type="a100", minutes=60),
        body=body(common, tables).replace("@inputs_dir@", "@fnal_data@"), defaults=dict(DEFAULTS))
    fnal_tasks.add("crv_response")
    if table_tasks is not None:
        table_tasks.add("crv_response")
    if delegates is not None:
        from . import kit_uva
        delegates["crv_response"] = kit_uva.Delegate
