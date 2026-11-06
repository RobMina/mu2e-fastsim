"""Bounded Run1A development study: all positive source entrances, never survivor-selected."""
import argparse
import json
from pathlib import Path

import numpy as np

from .tscompare import BINS, _charge, _hist, _keys, _load, _paired, _validate
from .tsdiagnose import FATES, _join, _overlap

SEEDS = (20261111, 20261112, 20261113, 20261114)
EXPECTED, EXPECTED_SHARDS = 12832, 100
VARIANTS = {"baseline": (2.5, .25, .05), "half_material": (2.5, .125, .025), "half_step": (1.25, .25, .05)}
WARNINGS = ["Development only; no tuning to stock, no significance from four seeds.",
            "half_material combines material-step and fractional-loss tightening; not an isolated cause.",
            "Changing step settings changes RNG consumption: equal seed labels are NOT common-random histories.",
            "Selecting positive inputs changes RNG streams versus the original all-charge batch; no bitwise comparison.",
            "Saved full-fast 8/15 is historical, not this cohort's baseline; a fresh baseline is required.",
            "Stock-only mean lengths describe FAST trajectories, not stock material histories or mismatch causes.",
            "Stock-only membership changes across settings/seeds: differences of conditional means are not per-lineage path effects.",
            "Yield-ratio errors use paired independent shards, not four-seed significance; time bins are elapsed ns, momentum MeV/c."]


def cohort(reference):
    """Validate the full saved reference, retaining event exposure before selecting charge."""
    r = _load(reference)
    _validate(r, True)
    for key in ("id", "event_id", "parent_shard_ids"):
        a = r[key]
        width = {"id": 4, "event_id": 3, "parent_shard_ids": 2}[key]
        if a.ndim != 2 or a.shape[1] != width or a.dtype.kind not in "iu" or len(np.unique(_keys(a))) != len(a):
            raise ValueError(f"invalid/duplicate {key}")
        if np.any((a[:, 0] != 1430) | (a[:, 1] < 920000) | (a[:, 1] > 920099)):
            raise ValueError("Run1A development subruns 920000..920099 only")
    if len(r["parent_shard_ids"]) != EXPECTED_SHARDS:
        raise ValueError(f"full {EXPECTED_SHARDS}-shard development reference required")
    units, counts = np.unique(r["event_id"][:, :2], axis=0, return_counts=True)
    order = np.argsort(_keys(r["parent_shard_ids"]))
    if not np.array_equal(units, r["parent_shard_ids"][order]) or not np.array_equal(counts, r["parent_shard_event_counts"][order]):
        raise ValueError("event exposure and parent counts disagree")
    if np.any(r["shard"] != r["id"][:, 1]) or not np.all(np.isin(_keys(r["id"][:, :3]), _keys(r["event_id"]))):
        raise ValueError("source shard/event IDs disagree")
    if np.any(r.get("endpoint_ancestry_incomplete", False)) or np.any(r.get("stock_multiplicity", r["stock_reached"]) != r["stock_reached"]):
        raise ValueError("unresolved/multiple stock endpoints")
    q = _charge(r["pdg"])
    stock = {str(c): {str(p): int(np.sum((q == c) & r["stock_reached"].astype(bool) & (r["stock_pdg"] == p)))
                      for p in (-13, 13, -211, 211)} for c in (-1, 1)}
    mask = q == 1
    if mask.sum() != EXPECTED:
        raise ValueError(f"expected ALL {EXPECTED} positive entrances, got {mask.sum()}")
    rows = ("id", "shard", "x", "p", "t", "pdg", "stock_reached", "stock_x", "stock_p", "stock_t", "stock_pdg")
    selected = {k: r[k][mask].copy() for k in rows}
    selected.update({k: r[k].copy() for k in ("event_id", "parent_shard_ids", "parent_shard_event_counts")})
    return selected, stock


def report(r, f):
    f = _join(r, f)
    group = np.searchsorted(np.sort(_keys(r["parent_shard_ids"])), _keys(r["id"][:, :2]))
    k = len(r["parent_shard_ids"])
    count = lambda m: np.bincount(group[m], minlength=k)
    n = count(np.ones(len(group), bool))
    out = {"metrics": {}, "endpoints": {}, "fates": dict(zip(FATES, np.bincount(f["fate"], minlength=len(FATES)).tolist()))}
    for name, pdg in (("mu+", -13), ("pi+", 211), ("ds", None)):
        s, a = r["stock_reached"].astype(bool), f["fate"] == 1
        if pdg is not None:
            s, a = s & (r["stock_pdg"] == pdg), a & (f["pdg"] == pdg)
        overlap = _overlap(s, a)
        metrics = {**overlap, **{f"stock_only_mean_{v}_mm": float(f[v][s & ~a].mean()) if np.any(s & ~a) else None for v in ("path", "material_path")}}
        out["metrics"].update({f"{name}.{v}": value for v, value in metrics.items()})
        out["endpoints"][name] = {"overlap": overlap, "paired_shards": _paired(count(s), count(a), n), "histograms": {}}
        for variable, edges in BINS.items():
            sv, fv = ((np.linalg.norm(r["stock_p"], axis=1), np.linalg.norm(f["p"], axis=1)) if variable == "momentum" else (r["stock_t"] - r["t"], f["t"] - r["t"]))
            hs, sf = _hist(sv, s, group, k, edges)
            hf, ff = _hist(fv, a, group, k, edges)
            out["endpoints"][name]["histograms"][variable] = dict(edges=edges.tolist(), stock=hs.sum(0).tolist(), fast=hf.sum(0).tolist(), stock_flow=sf, fast_flow=ff)
    return out


def stock_muplus_lineages(paths, runs, reference, gdml=None):
    """Columnar stock-survivor-selected diagnostic; never a rate estimator."""
    from .tsdiagnose import _terminal_materials
    if Path(reference).resolve() != Path(runs[0]["inputs"]["reference"]).resolve():
        raise ValueError("reference differs from saved study inputs")
    if gdml is not None and Path(gdml).resolve() != Path(runs[0]["inputs"]["gdml"]).resolve():
        raise ValueError("gdml differs from saved study inputs")
    r, _ = cohort(reference)
    selected = r["stock_reached"].astype(bool) & (r["stock_pdg"] == -13)
    geometry = None
    if gdml is not None:
        from .tsgeometry import Geometry
        geometry = Geometry(gdml)
    out = dict(label="Stock-survivor-selected diagnostic, not a rate estimator; rates remain all positive inputs.",
               limitations="Fast fates/paths are not stock histories or mismatch causes; persistent loss means only across supplied runs.",
               rate_ninput=len(r["id"]), nstock_muplus=int(selected.sum()),
               id_columns=["run", "subrun", "event", "track"], id=r["id"][selected].tolist(),
               source_pdg=r["pdg"][selected].tolist(), source_p=r["p"][selected].tolist(),
               source_pmag=np.linalg.norm(r["p"][selected], axis=1).tolist(),
               stock_ds_p=r["stock_p"][selected].tolist(), stock_ds_pmag=np.linalg.norm(r["stock_p"][selected], axis=1).tolist(),
               units={"momentum": "MeV/c", "material_path": "mm"}, fast_fates=list(FATES), runs=[], by_variant={})
    survived = []
    for path, run in zip(paths, runs):
        if run["ninput"] != EXPECTED:
            raise ValueError("saved study must contain all positive inputs")
        f = _join(r, _load(Path(path).with_suffix(".npz")))
        fast = (f["fate"] == 1) & (f["pdg"] == -13)
        if int(np.sum(fast & selected)) != run["metrics"]["mu+.both"] or int(np.sum(fast & ~selected)) != run["metrics"]["mu+.fast_only"]:
            raise ValueError("saved JSON/NPZ mu+ counts disagree")
        survived.append(fast[selected])
        row = dict(variant=run["variant"], seed=run["seed"], fast_fate=f["fate"][selected].tolist(),
                   fast_pdg=f["pdg"][selected].tolist(), survived_muplus=fast[selected].tolist(),
                   material_path=f["material_path"][selected].tolist(),
                   fate_counts=np.bincount(f["fate"][selected], minlength=len(FATES)).tolist(),
                   fast_only=run["metrics"]["mu+.fast_only"], terminal_material_bitmask=None,
                   capped_ids=f['id'][f['fate'] == 6].tolist(),
                   capped_stock_reached=r['stock_reached'][f['fate'] == 6].tolist(),
                   capped_path_mm=f['path'][f['fate'] == 6].tolist())
        if geometry is not None:
            bits, labels = _terminal_materials(f["x"][selected], geometry, f["fate"][selected], ~fast[selected])
            row["terminal_material_bitmask"] = bits.tolist()
            out["terminal_materials"] = {k: labels[k] for k in ("label", "limitations", "bit_to_material")}
        out["runs"].append(row)
    survived = np.asarray(survived, dtype=bool)
    for variant in VARIANTS:
        mask = np.asarray([run["variant"] == variant for run in runs])
        if not np.any(mask):
            continue
        counts = survived[mask].sum(axis=0)
        out["by_variant"][variant] = dict(seeds=[run["seed"] for run in runs if run["variant"] == variant],
            survived_by_seed=survived[mask].sum(axis=1).tolist(), survived_runs_per_lineage=counts.tolist(),
            survival_count_histogram=np.bincount(counts, minlength=int(mask.sum()) + 1).tolist(),
            never_survived_ids=r["id"][selected][counts == 0].tolist())
    out["never_survived_ids"] = r["id"][selected][~survived.any(axis=0)].tolist()
    return out


def summarize(paths, reference=None, gdml=None):
    if gdml is not None and reference is None:
        raise ValueError("summarize --gdml requires --reference")
    runs = [json.loads(Path(p).read_text()) for p in paths]
    if not runs or len({(r["variant"], r["seed"]) for r in runs}) != len(runs):
        raise ValueError("nonempty unique variant/seed runs required")
    if len({json.dumps(r["inputs"], sort_keys=True) for r in runs}) != 1:
        raise ValueError("cannot summarize different reference/field/geometry inputs")
    for r in runs:
        if r["seed"] not in SEEDS or r["settings"] != list(VARIANTS[r["variant"]]):
            raise ValueError("non-prespecified seed/settings")
    def stats(values):
        a = np.asarray([v for v in values if v is not None], float)
        return dict(n=len(a), mean=float(a.mean()) if len(a) else None, sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                    min=float(a.min()) if len(a) else None, max=float(a.max()) if len(a) else None)
    grouped, paired = {}, {}
    baseline = {r["seed"]: r for r in runs if r["variant"] == "baseline"}
    for variant in VARIANTS:
        rows = [r for r in runs if r["variant"] == variant]
        if not rows:
            continue
        grouped[variant] = {"seeds": [r["seed"] for r in rows], "metrics": {m: stats([r["metrics"][m] for r in rows]) for m in rows[0]["metrics"]}}
        if variant != "baseline":
            differences = {str(r["seed"]): {m: r["metrics"][m] - baseline[r["seed"]]["metrics"][m] if r["metrics"][m] is not None and baseline[r["seed"]]["metrics"][m] is not None else None for m in r["metrics"]} for r in rows if r["seed"] in baseline}
            paired[variant] = {"setting_minus_baseline_by_seed": differences, "metrics": {m: stats([d[m] for d in differences.values()]) for m in rows[0]["metrics"]}}
    out = dict(warnings=WARNINGS, prespecified_seeds=list(SEEDS), seed_statistics=grouped, paired_setting_differences=paired)
    if reference is not None:
        out["stock_muplus_lineages"] = stock_muplus_lineages(paths, runs, reference, gdml)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "gdml", "field-config", "offline-root", "maps", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="*", choices=SEEDS, help=f"required for runs; bare flag uses fixed seeds {SEEDS}")
    parser.add_argument("--summarize", nargs="+", type=Path, help="existing run JSON files only; optional --reference adds exact-ID lineages, --gdml terminal materials; no transport")
    args = parser.parse_args(argv)
    if args.summarize:
        if any(getattr(args, k) is not None for k in ("field_config", "offline_root", "maps", "variant", "seeds")):
            parser.error("summarize accepts only --reference, --gdml and --out; settings come from each saved JSON")
        text = json.dumps(summarize(args.summarize, args.reference, args.gdml), indent=2, allow_nan=False)
        if args.out:
            with args.out.open("x") as handle:
                handle.write(text + "\n")
        print(text)
        return
    if any(getattr(args, k) is None for k in ("reference", "gdml", "field_config", "offline_root", "maps", "out", "variant", "seeds")):
        parser.error("runs require --variant, --seeds and all reference/field/geometry/output paths")
    if args.out.exists() or "heldout" in str(args.reference).lower().replace("-", "").replace("_", ""):
        parser.error("fresh output directory and development reference required")
    seeds = args.seeds or SEEDS
    if len(set(seeds)) != len(seeds):
        parser.error("duplicate seeds")
    r, stock = cohort(args.reference)
    order = np.argsort(_keys(r["parent_shard_ids"]))
    for key in ("parent_shard_ids", "parent_shard_event_counts"):
        r[key] = r[key][order]
    inputs = {k: str(getattr(args, k).resolve()) for k in ("reference", "gdml", "field_config", "offline_root", "maps")}
    from . import tsfield, tsgeometry, tstransport
    field, geometry = tsfield.from_config(args.field_config, args.offline_root, args.maps), tsgeometry.Geometry(args.gdml)
    args.out.mkdir(parents=True, exist_ok=False)
    step, material, loss = VARIANTS[args.variant]
    for seed in seeds:
        f = tstransport.transport({k: r[k].copy() for k in ("x", "p", "t", "pdg")}, field, geometry, seed=seed, step_mm=step, material_step_mm=material, loss_fraction=loss)
        f.update({k: r[k] for k in ("id", "shard", "event_id", "parent_shard_ids", "parent_shard_event_counts")})
        result = dict(report(r, f), variant=args.variant, seed=seed, prespecified_seeds=list(SEEDS), settings=list(VARIANTS[args.variant]), inputs=inputs, warnings=WARNINGS,
                      ninput=len(r["id"]), events=len(r["event_id"]), shards=r["parent_shard_ids"].tolist(), stock_per_source_charge=stock, seconds=float(f["seconds"]))
        stem = args.out / f"{args.variant}-{seed}"
        with stem.with_suffix(".npz").open("xb") as handle:
            np.savez_compressed(handle, **f)
        with stem.with_suffix(".json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        print(json.dumps(dict(variant=args.variant, seed=seed, metrics=result["metrics"])))


if __name__ == "__main__":
    main()
