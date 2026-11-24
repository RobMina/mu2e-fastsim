"""fastsimkit task `crv_design`: one CRV module design point (crvfastsim.design, W9) on one CPU node.

One point = `python -m crvfastsim.design.kit_task run`: the design applied to every Run 2 sector, the figures of
merit on a pinned CE-proxy muon sample, metrics.json for Mu2eBO. No GPU. The Run 2 lookup tables (the 12 v6_0 tables
of RUN2_SECTORS, about 0.3 GB) are Mu2e-internal: fetched per job from the CVMFS Stratum-1 into node tmpfs and
deleted by the EXIT trap (out/cleanup.txt, required by `results`). The muon sample is an endpoint-resident file
pinned by sha256 in the submission manifest and checked by the job before use (delegate: crvfastsim.design.kit_uva).

Params (Mu2eBO sends numbers as floats; empty = the stock value of each sector)
  design knobs  n_layers, thickness, width, max_length, absorber, tilt_deg, fiber_diameter, counters_per_module,
                coincidence_layers or spare_layers (k = N - spare in every sector), pe_threshold (module.Design)
  absorber_material  G4_Al, MildSteel, G4_Pb, G4_POLYETHYLENE or empty (stock); one mode spec per material
  sample        the muon sample name (SAMPLES)
  dtype         float64 or float32
  pe_scale      optics calibration (0.847: stock Offline, Run 2 conditions; docs/design.md)
  kappa, f_acc, beta  deadtime model (deadtime.DeadtimeConfig)
  aging_c1, aging_ref light-yield constraint (fom.FomConfig)
  cost_w_scint, cost_w_channels, cost_w_absorber, cost_w_fiber  cost-proxy weights (fom.FomConfig.cost_weights);
                all empty = the plain mean of the four terms; set all four or none
  max_muons     0 = the whole sample
  place_policy  empty = no placement (module.build on the stock sector positions, as before); "first" or "each":
                lay the design out and place it in the detector envelope (design.placement), scoring every placed
                variant ("each": the base placement, or every remedy alone when it does not place)
  place_stagger_max, place_clearance, place_max_trim, place_max_shift  placement ranges [mm] (PlacementConfig)
  place_remedies  ';'-separated remedies, move:<name>:<S1>,<S2>:<dx>,<dy>,<dz>:<max_mm> (move sectors),
                thin:<name>:<sheet>|<sheet>:<min_mm> (thin borated-poly sheets; sheet = G4 name, type<k> or label
                text) or movethin:<name>:<S1>,<S2>:<dx>,<dy>,<dz>:<max_mm>:<sheet>|<sheet>:<min_mm> (both)
  coinc_groups  empty = every sector searched alone for coincidences (as before); "sector_type": Offline's
                CrvCoincidenceFinder grouping, every sector searched with the others of its crs.sectorType, each hit
                with its own sector's cuts (fom.sector_type_groups; placed fillers join their parent's group)
Metrics: METRICS (inefficiency and its error, deadtime_rate_ratio, ly_margin, cost and its four terms, ...); with a
placement also PLACEMENT_METRICS (feasible, largest shift, sheet move and thinnest sheet, shield modified, remedy).
"""
from __future__ import annotations

import re

# static names only: the kit process imports this module without numpy
INT_KNOBS = ("n_layers", "counters_per_module", "coincidence_layers", "spare_layers")
FLOAT_KNOBS = ("thickness", "width", "max_length", "absorber", "tilt_deg", "fiber_diameter", "pe_threshold")
KNOBS = INT_KNOBS + FLOAT_KNOBS
ABSORBER_MATERIALS = ("G4_Al", "MildSteel", "G4_Pb", "G4_POLYETHYLENE")   # = module.ABSORBER_MATERIALS (tested)
COST_WEIGHTS = {"cost_w_scint": "scint_m3", "cost_w_channels": "channels", "cost_w_absorber": "absorber_t",
                "cost_w_fiber": "fiber_m_mm2"}   # param -> fom.COST_TERMS key
PLACE = ("place_policy", "place_stagger_max", "place_clearance", "place_max_trim", "place_max_shift", "place_remedies")
PLACE_POLICIES = ("first", "each")       # = placement.POLICIES (tested)
COINC_GROUPS = ("sector_type",)          # coinc_groups values (empty = every sector alone)
FIXED = ("absorber_material", "sample", "dtype", "pe_scale", "kappa", "f_acc", "beta", "aging_c1", "aging_ref",
         "max_muons") + tuple(COST_WEIGHTS) + PLACE + ("coinc_groups",)
PARAMS = KNOBS + FIXED
DEFAULTS = {**{k: "" for k in KNOBS}, "absorber_material": "", "sample": "run2_ce_proxy_v1", "dtype": "float64",
            "pe_scale": 0.847, "kappa": 1.0, "f_acc": 0.5, "beta": 1.0, "aging_c1": 0.94 ** 5, "aging_ref": 0.86,
            "max_muons": 0, **{k: "" for k in COST_WEIGHTS}, "place_policy": "", "place_stagger_max": 60.0,
            "place_clearance": 3.0, "place_max_trim": 120.0, "place_max_shift": 500.0, "place_remedies": "",
            "coinc_groups": ""}
# file names under the endpoint's crv_design data dir: generated lines (v1) or CRV-entry states (design.sample)
SAMPLES = {"run2_ce_proxy_v1": "run2_ce_proxy_v1.npz", "run2_ce_entry_v1": "run2_ce_entry_v1.npz"}
METRICS = ("inefficiency", "inefficiency_err", "inefficiency_covered", "uncovered_fraction", "stopped_fraction",
           "deadtime_rate_ratio",
           "ly_margin", "cost", "cost_scint", "cost_channels", "cost_absorber", "cost_fiber", "n_muons", "n_bars",
           "wall_s")
PLACEMENT_METRICS = ("placement_feasible", "placement_shift_max_abs_mm", "placement_sheet_move_max_mm",
                     "placement_sheet_min_thickness_mm", "placement_shield_modified", "placement_remedy_mm",
                     "placement_n_variants")
METRICS = METRICS + PLACEMENT_METRICS
TABLES = tuple(f"LookupTable_{t}" for t in ("1045_1", "1550_1", "2100_1", "2370_1", "3040_0", "3200_0", "4550_0",
                                             "5000_1", "5700_0", "6000_0", "6000_1", "6900_1"))
RESOURCES = dict(cores=2, mem_gb=8, gpus=0, gpu_type="", minutes=30)

BODY = ('MU=@sample_path@\n'
        'echo "@sample_sha256@  $MU" | sha256sum -c - > /dev/null || { echo "sample $MU does not match its pin"; exit 3; }\n'
        'T python -m crvfastsim.design.kit_task run --out "$M2FS_OUT" --muons "$MU" --lookup-dir "$L" '
        + " ".join(f"--{k.replace('_', '-')} @p_{k}@" for k in ("dtype", "pe_scale", "kappa", "f_acc", "beta",
                                                               "aging_c1", "aging_ref", "max_muons"))
        + " --design " + " ".join(f"{k}=@p_{k}@" for k in KNOBS + ("absorber_material",))
        + " --cost-weights " + " ".join(f"{k}=@p_{k}@" for k in COST_WEIGHTS)
        + " " + " ".join(f"--{k.replace('_', '-')} @p_{k}@" for k in PLACE + ("coinc_groups",))
        + ' > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }\n'
        'tail -c 400 "$M2FS_OUT/point.txt"\n')


def design_dict(pairs):
    """module.Design keys from KEY=VALUE words (empty value = stock, omitted)."""
    out = {}
    for w in pairs:
        k, _, v = w.partition("=")
        if k not in KNOBS + ("absorber_material",):
            raise ValueError(f"crv_design: unknown design key {k!r}")
        if v == "":
            continue
        out[k] = v if k == "absorber_material" else (int(float(v)) if k in INT_KNOBS else float(v))
    return out


def cost_weights(pairs):
    """fom.FomConfig.cost_weights from KEY=VALUE words over COST_WEIGHTS: () when every value is empty (equal
    weights), all four when every value is set; a partial set raises."""
    vals = {}
    for w in pairs:
        k, _, v = w.partition("=")
        if k not in COST_WEIGHTS:
            raise ValueError(f"crv_design: unknown cost weight {k!r}")
        vals[k] = v
    given = {k: v for k, v in vals.items() if v != ""}
    if not given:
        return ()
    if set(given) != set(COST_WEIGHTS):
        raise ValueError(f"crv_design: set all of {tuple(COST_WEIGHTS)} or none, got {tuple(given)}")
    return tuple((COST_WEIGHTS[k], float(given[k])) for k in COST_WEIGHTS)


_NUM, _SHEET = r"[-+0-9.eE]+", r"[\w][\w .-]*(\|[\w][\w .-]*)*"
_MOVE = rf"[\w]+(,[\w]+)*:{_NUM},{_NUM},{_NUM}:{_NUM}"
_REMEDY = re.compile(rf"^(move:[\w.-]+:{_MOVE}|thin:[\w.-]+:{_SHEET}:{_NUM}|movethin:[\w.-]+:{_MOVE}:{_SHEET}:{_NUM})$")


def remedy_words(text):
    """The place_remedies items (placement.parse_remedies syntax), checked without numpy; raises ValueError."""
    items = [x.strip() for x in (text or "").split(";") if x.strip()]
    for it in items:
        if not _REMEDY.match(it):
            raise ValueError(f"crv_design: remedy {it!r}: expected move:name:S1,S2:dx,dy,dz:max, "
                             "thin:name:sheet|sheet:min_mm or movethin:name:S1,S2:dx,dy,dz:max:sheet|sheet:min_mm")
    names = [it.split(":")[1] for it in items]
    if len(set(names)) != len(names):
        raise ValueError("crv_design: remedy names must be unique")
    return items


def metrics_of(r):
    m = {k: r[k] for k in ("inefficiency", "inefficiency_err", "inefficiency_covered", "uncovered_fraction", "stopped_fraction",
                           "deadtime_rate_ratio", "ly_margin", "cost", "n_muons", "n_bars", "wall_s")}
    m.update(cost_scint=r["cost_rel"]["scint_m3"], cost_channels=r["cost_rel"]["channels"],
             cost_absorber=r["cost_rel"]["absorber_t"], cost_fiber=r["cost_rel"]["fiber_m_mm2"])
    if "placement" in r:                       # only with a placement: the default metrics are unchanged
        pl = r["placement"]
        m.update(placement_feasible=pl["feasible"], placement_shift_max_abs_mm=pl["shift_max_abs_mm"],
                 placement_sheet_move_max_mm=pl["sheet_move_max_mm"],
                 placement_sheet_min_thickness_mm=pl["sheet_min_thickness_mm"],
                 placement_shield_modified=pl["shield_modified"],
                 placement_remedy_mm=(pl["remedy"] or {}).get("amount_mm", 0.0),
                 placement_n_variants=len(r["placement_variants"]))
    return {k: float(v) for k, v in m.items()}


def placement_config(policy, stagger_max, clearance, max_trim, max_shift, remedies):
    """fom.FomConfig.placement from the place_* params; None when policy is empty."""
    if not policy:
        return None
    from . import placement as P
    top = float(stagger_max)
    stag = tuple(sorted({top - 10.0 * i for i in range(int(top // 10.0) + 1)} | {0.0}, reverse=True)) if top > 0 \
        else (0.0,)
    return P.PlacementConfig(stagger_mm=stag, clearance=float(clearance), max_trim=float(max_trim),
                             max_shift=float(max_shift), remedies=P.parse_remedies(remedies), policy=policy)


def coincidence_groups(mode, counters):
    """fom.FomConfig.coincidence_groups from the coinc_groups param: () when empty, Offline's sector-type grouping
    of the counters file for "sector_type"."""
    if not mode:
        return ()
    if mode not in COINC_GROUPS:
        raise ValueError(f"crv_design: coinc_groups {mode!r}: expected one of {COINC_GROUPS} or empty")
    from . import fom
    return fom.sector_type_groups(counters)


def run(argv=None):
    """The job side: evaluate one design point, write eval.json and metrics.json into --out."""
    import argparse
    import dataclasses
    import json
    import os

    import numpy as np

    from . import deadtime as DT
    from . import fom
    from . import module as M
    from . import optics as O
    from .approximations import WORKFLOW
    p = argparse.ArgumentParser(prog="python -m crvfastsim.design.kit_task run")
    p.add_argument("--out", required=True)
    p.add_argument("--muons", required=True)
    p.add_argument("--lookup-dir", required=True)
    p.add_argument("--dtype", required=True, choices=("float32", "float64"))
    for k in ("pe_scale", "kappa", "f_acc", "beta", "aging_c1", "aging_ref"):
        p.add_argument("--" + k.replace("_", "-"), type=float, required=True)
    p.add_argument("--max-muons", type=int, default=0)
    p.add_argument("--design", nargs="*", default=[])
    p.add_argument("--cost-weights", nargs="*", default=[])
    p.add_argument("--place-policy", default="", choices=("",) + PLACE_POLICIES)
    for k in ("place_stagger_max", "place_clearance", "place_max_trim", "place_max_shift"):
        p.add_argument("--" + k.replace("_", "-"), type=float, default=DEFAULTS[k])
    p.add_argument("--place-remedies", default="")
    p.add_argument("--coinc-groups", default="", choices=("",) + COINC_GROUPS)
    a = p.parse_args(argv)
    design = M.Design(**design_dict(a.design))
    remedy_words(a.place_remedies)
    ctx = fom.Context(lookup_dir=a.lookup_dir)
    cfg = fom.FomConfig(optics=O.OpticsConfig(pe_scale=a.pe_scale),
                        deadtime=DT.DeadtimeConfig(kappa=a.kappa, f_acc=a.f_acc, beta=a.beta),
                        aging_c1=a.aging_c1, aging_ref=a.aging_ref, cost_weights=cost_weights(a.cost_weights),
                        placement=placement_config(a.place_policy, a.place_stagger_max, a.place_clearance,
                                                   a.place_max_trim, a.place_max_shift, a.place_remedies),
                        coincidence_groups=coincidence_groups(a.coinc_groups, ctx.counters))
    z = np.load(a.muons)
    n = len(z["p"]) if a.max_muons <= 0 else min(a.max_muons, len(z["p"]))
    muons = dict(origin=z["origin"][:n], direction=z["direction"][:n], p=z["p"][:n])
    r = fom.evaluate(design, muons, ctx, cfg, None, np.float64 if a.dtype == "float64" else np.float32)
    if r["table_substitutions"]:
        raise SystemExit(f"crv_design: lookup tables missing from {a.lookup_dir}: {sorted(r['table_substitutions'])}")
    if "counters_text" in r:                   # the placed layout: what a stock reference of this point is built on
        with open(os.path.join(a.out, "crv_counters_placed.txt"), "w") as f:
            f.write(r.pop("counters_text"))
    r["config"] = dict(fom=dataclasses.asdict(cfg), muons=os.path.basename(a.muons), design=design.as_dict())
    r["approximations"] = WORKFLOW.select("default").record()
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(r, f, indent=1, default=float)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics_of(r), f)
    print(json.dumps(metrics_of(r)))
    return 0


def register(tasks, task_cls, resources_cls, delegates=None):
    """Add crv_design to a fastsimkit TASKS dict; with `delegates`, its Slurm-over-SSH handler (kit_uva)."""
    tasks["crv_design"] = task_cls(params=PARAMS, metrics=METRICS, resources=resources_cls(**RESOURCES),
                                   body=BODY, defaults=dict(DEFAULTS))
    if delegates is not None:
        from . import kit_uva
        delegates["crv_design"] = kit_uva.Delegate


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        raise SystemExit("usage: python -m crvfastsim.design.kit_task run ...")
    sys.exit(run(sys.argv[2:]))
