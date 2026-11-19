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
  max_muons     0 = the whole sample
Metrics: METRICS (inefficiency and its error, deadtime_rate_ratio, ly_margin, cost and its four terms, ...).
"""
from __future__ import annotations

# static names only: the kit process imports this module without numpy
INT_KNOBS = ("n_layers", "counters_per_module", "coincidence_layers", "spare_layers")
FLOAT_KNOBS = ("thickness", "width", "max_length", "absorber", "tilt_deg", "fiber_diameter", "pe_threshold")
KNOBS = INT_KNOBS + FLOAT_KNOBS
ABSORBER_MATERIALS = ("G4_Al", "MildSteel", "G4_Pb", "G4_POLYETHYLENE")   # = module.ABSORBER_MATERIALS (tested)
FIXED = ("absorber_material", "sample", "dtype", "pe_scale", "kappa", "f_acc", "beta", "aging_c1", "aging_ref",
         "max_muons")
PARAMS = KNOBS + FIXED
DEFAULTS = {**{k: "" for k in KNOBS}, "absorber_material": "", "sample": "run2_ce_proxy_v1", "dtype": "float64",
            "pe_scale": 0.847, "kappa": 1.0, "f_acc": 0.5, "beta": 1.0, "aging_c1": 0.94 ** 5, "aging_ref": 0.86,
            "max_muons": 0}
# file names under the endpoint's crv_design data dir: generated lines (v1) or CRV-entry states (design.sample)
SAMPLES = {"run2_ce_proxy_v1": "run2_ce_proxy_v1.npz", "run2_ce_entry_v1": "run2_ce_entry_v1.npz"}
METRICS = ("inefficiency", "inefficiency_err", "inefficiency_covered", "uncovered_fraction", "stopped_fraction",
           "deadtime_rate_ratio",
           "ly_margin", "cost", "cost_scint", "cost_channels", "cost_absorber", "cost_fiber", "n_muons", "n_bars",
           "wall_s")
TABLES = tuple(f"LookupTable_{t}" for t in ("1045_1", "1550_1", "2100_1", "2370_1", "3040_0", "3200_0", "4550_0",
                                             "5000_1", "5700_0", "6000_0", "6000_1", "6900_1"))
RESOURCES = dict(cores=2, mem_gb=8, gpus=0, gpu_type="", minutes=30)

BODY = ('MU=@sample_path@\n'
        'echo "@sample_sha256@  $MU" | sha256sum -c - > /dev/null || { echo "sample $MU does not match its pin"; exit 3; }\n'
        'T python -m crvfastsim.design.kit_task run --out "$M2FS_OUT" --muons "$MU" --lookup-dir "$L" '
        + " ".join(f"--{k.replace('_', '-')} @p_{k}@" for k in ("dtype", "pe_scale", "kappa", "f_acc", "beta",
                                                               "aging_c1", "aging_ref", "max_muons"))
        + " --design " + " ".join(f"{k}=@p_{k}@" for k in KNOBS + ("absorber_material",))
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


def metrics_of(r):
    m = {k: r[k] for k in ("inefficiency", "inefficiency_err", "inefficiency_covered", "uncovered_fraction", "stopped_fraction",
                           "deadtime_rate_ratio", "ly_margin", "cost", "n_muons", "n_bars", "wall_s")}
    m.update(cost_scint=r["cost_rel"]["scint_m3"], cost_channels=r["cost_rel"]["channels"],
             cost_absorber=r["cost_rel"]["absorber_t"], cost_fiber=r["cost_rel"]["fiber_m_mm2"])
    return {k: float(v) for k, v in m.items()}


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
    a = p.parse_args(argv)
    design = M.Design(**design_dict(a.design))
    cfg = fom.FomConfig(optics=O.OpticsConfig(pe_scale=a.pe_scale),
                        deadtime=DT.DeadtimeConfig(kappa=a.kappa, f_acc=a.f_acc, beta=a.beta),
                        aging_c1=a.aging_c1, aging_ref=a.aging_ref)
    z = np.load(a.muons)
    n = len(z["p"]) if a.max_muons <= 0 else min(a.max_muons, len(z["p"]))
    muons = dict(origin=z["origin"][:n], direction=z["direction"][:n], p=z["p"][:n])
    ctx = fom.Context(lookup_dir=a.lookup_dir)
    r = fom.evaluate(design, muons, ctx, cfg, None, np.float64 if a.dtype == "float64" else np.float32)
    if r["table_substitutions"]:
        raise SystemExit(f"crv_design: lookup tables missing from {a.lookup_dir}: {sorted(r['table_substitutions'])}")
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
