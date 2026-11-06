"""Development-only exact-ID diagnosis of saved TS reference and fast NPZ outputs.

No transport, raw input, or held-out analysis. Counts are descriptive paired
lineage outcomes, not independent-particle significance or causal attribution.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .tscompare import _charge, _keys, _load, _validate

# Saved transport format: do not import or execute the transport implementation.
FATES = ("live", "ds", "rangeout", "mu_decay", "wall", "upstream", "path_limit", "field_missing")
PDGS = (-211, -13, 13, 211)


def _join(r, f):
    rk, fk = _validate(r, True), _validate(f, False)
    if len(rk) != len(fk):
        raise ValueError("reference and fast must contain the same complete ID cohort")
    order = np.argsort(fk)
    pos = np.searchsorted(fk[order], rk)
    if np.any(pos == len(fk)) or np.any(fk[order[pos]] != rk):
        raise ValueError("reference and fast IDs differ")
    row_fields = {'id', 'shard', 'x', 'p', 't', 'pdg', 'fate', 'path', 'material_path', 'pion_decay', 'steps'}
    f = {k: v[order[pos]] if k in row_fields and v.ndim and len(v) == len(fk) else v for k, v in f.items()}
    if np.any(r["shard"] != f["shard"]) or np.any(r["shard"] != r["id"][:, 1]):
        raise ValueError("shard mismatch for exact IDs")
    if np.any(~np.isin(f["fate"], np.arange(len(FATES)))):
        raise ValueError("unknown fast fate")
    if np.any(~np.isin(r["stock_pdg"][r["stock_reached"].astype(bool)], PDGS)):
        raise ValueError("unknown stock endpoint PDG")
    _charge(f["pdg"])
    for k in ("path", "material_path"):
        if k not in f or f[k].shape != (len(rk),) or np.any(~np.isfinite(f[k]) | (f[k] < 0)):
            raise ValueError(f"invalid fast {k}")
    for k in ("x", "p", "t"):
        if np.any(~np.isfinite(f[k])):
            raise ValueError(f"nonfinite fast terminal {k}")
    if np.any(r.get("endpoint_ancestry_incomplete", False)):
        raise ValueError("unresolved DS ancestry cannot define stock losses")
    if "stock_multiplicity" in r and np.any(r["stock_multiplicity"] != r["stock_reached"]):
        raise ValueError("stock multiplicity is not aligned single-endpoint")
    return f


def _overlap(stock, fast):
    counts = np.bincount(2 * stock.astype(int) + fast, minlength=4)
    return dict(zip(("neither", "fast_only", "stock_only", "both"), counts.tolist()),
                stock=int(stock.sum()), fast=int(fast.sum()))


def _contingency(stock, fast, fate, mask):
    return {"ninput": int(mask.sum()), **_overlap(stock[mask], fast[mask]),
            "stock_ds_by_fast_fate": np.bincount(
                len(FATES) * stock[mask].astype(int) + fate[mask],
                minlength=2 * len(FATES)).reshape(2, len(FATES)).tolist()}


def _association(gap, survived, mask):
    g, s = gap[mask], survived[mask]
    table = np.bincount(2 * g.astype(int) + s, minlength=4).reshape(2, 2)
    n = table.sum(axis=1)
    rate = np.divide(table[:, 1], n, out=np.full(2, np.nan), where=n > 0)
    a, b, c, d = table.ravel().astype(float)
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {"gap_by_survival": table.tolist(),
            "survival_fraction_no_gap": float(rate[0]) if n[0] else None,
            "survival_fraction_gap": float(rate[1]) if n[1] else None,
            "difference_gap_minus_no_gap": float(rate[1] - rate[0]) if np.all(n) else None,
            "phi": float((a * d - b * c) / denominator) if denominator else None}


def _terminal_materials(x, geometry, fate, missed):
    """Vectorized TERMINAL POSITION membership, never a trajectory/history label."""
    names = sorted(geometry.materials)
    if len(names) > 64:
        raise ValueError("terminal material bitmask supports at most 64 selected materials")
    bits = {name: np.uint64(1 << i) for i, name in enumerate(names)}
    mask = np.zeros(len(x), dtype=np.uint64)
    for solid, R, t, key in geometry._matter:
        mask[solid.contains((x - t) @ R)] |= bits[key]
    values, inverse = np.unique(mask[missed], return_inverse=True)
    counts = np.bincount(len(values) * fate[missed] + inverse,
                         minlength=len(FATES) * len(values)).reshape(len(FATES), len(values))
    return mask, {
        "label": "TERMINAL POSITION classification",
        "limitations": "Selected-material containment only; positions can lie on boundaries. Not trajectory, material history or cause; zero means no selected material contains the position, not necessarily vacuum.",
        "bit_to_material": {str(int(bit)): name for name, bit in bits.items()},
        "stock_only_muplus_material_bitmasks": values.tolist(),
        "stock_only_muplus_fast_fate_by_material_bitmask": counts.tolist(),
        "table_axes": ["fast_fates", "stock_only_muplus_material_bitmasks"]}


def diagnose(reference, fast, geometry=None):
    """Return JSON-ready summary and column arrays in reference exact-ID order."""
    r, f = _load(reference), _load(fast)
    f = _join(r, f)
    n, q = len(r["id"]), _charge(r["pdg"])
    s, a = r["stock_reached"].astype(bool), f["fate"] == 1
    groups = {"all": np.ones(n, bool)}
    groups.update({f"charge_{c}": q == c for c in (-1, 1)})
    groups.update({f"pdg_{p}": r["pdg"] == p for p in PDGS})
    out = {"ninput": n, "id_columns": ["run", "subrun", "event", "track"],
           "join": "exact unique (run, subrun, event, source-track), complete cohort",
           "units": {"position_path": "mm", "momentum": "MeV/c", "time": "ns"},
           "fast_fates": list(FATES), "contingency_rows": ["stock_not_ds", "stock_ds"],
           "source_groups": {}, "muons": {}, "ancestry": None,
           "limitations": [
               "Saved development outcomes only; no physics tuning or transport rerun.",
               "Fast terminal fate describes that trajectory, not the cause of a stock/fast mismatch.",
               "Independent stochastic histories can disagree per lineage; one fast seed cannot separate stochastic migration from systematic bias.",
               "Field/geometry/boundary integration, material/decay approximations, omitted interactions and reference ancestry remain alternatives; saved endpoints do not identify their contributions.",
               "material_path is cumulative material length, not material identity or an interaction history; stock loss histories are not saved.",
               "Upstream ancestry gaps can hide additional nesting; an observed gap association is descriptive, not causal. No raw ancestry was read.",
               "Sparse mu+ endpoints do not support a precision efficiency claim; no particle-independent significance is computed."]}
    for name, mask in groups.items():
        out["source_groups"][name] = _contingency(s, a, f["fate"], mask)
    outcomes = {"stock_ds": s, "fast_ds": a, "both_ds": s & a,
                "stock_only_ds": s & ~a, "fast_only_ds": ~s & a}
    for name, pdg in (("mu-", 13), ("mu+", -13)):
        sm, fm = s & (r["stock_pdg"] == pdg), a & (f["pdg"] == pdg)
        missed = sm & ~fm
        out["muons"][name] = {**_overlap(sm, fm),
            "source_groups": {g: _overlap(sm[m], fm[m]) for g, m in groups.items()},
            "stock_only_fast_fates": np.bincount(f["fate"][missed], minlength=len(FATES)).tolist(),
            "ids": {g: r["id"][m].tolist() for g, m in
                    (("both", sm & fm), ("stock_only", missed), ("fast_only", ~sm & fm))}}
        outcomes["stock_" + name], outcomes["fast_" + name] = sm, fm
    gap = r.get("source_parent_ancestry_incomplete")
    if gap is not None:
        if gap.shape != (n,) or np.any(~np.isin(gap, [0, 1])):
            raise ValueError("invalid source ancestry-gap flags")
        gap = gap.astype(bool)
        out["ancestry"] = {"flag": "source_parent_ancestry_incomplete", "n_gap": int(gap.sum()),
            "table_axes": ["gap false/true", "survival false/true"],
            "source_groups": {g: {key: _association(gap, v, m) for key, v in outcomes.items()}
                              for g, m in groups.items()}}
    if "event_id" in r:
        events = _keys(r["event_id"])
        if len(np.unique(events)) != len(events):
            raise ValueError("duplicate saved event IDs")
        out["saved_event_count"] = len(events)
        counts = r.get("parent_shard_event_counts")
        if counts is not None and counts.sum() != len(events):
            raise ValueError("saved event IDs and parent counts disagree")
    out["shards"] = np.unique(r["shard"]).tolist()
    if "nested_id" in r:
        out["nested_tracks_represented_by_ancestor"] = len(r["nested_id"])
    columns = dict(zip(out["id_columns"], r["id"].T))
    columns.update(source_charge=q, source_pdg=r["pdg"], stock_reached=s,
                   stock_pdg=r["stock_pdg"], fast_pdg=f["pdg"], fast_fate=f["fate"],
                   ancestry_gap=np.full(n, -1) if gap is None else gap.astype(int))
    for prefix, table, original in (("source", r, ""), ("stock", r, "stock_"), ("fast", f, "")):
        for variable in ("x", "p"):
            for i, axis in enumerate("xyz"):
                columns[f"{prefix}_{variable}{axis}"] = table[original + variable][:, i]
        columns[prefix + "_pmag"] = np.linalg.norm(table[original + "p"], axis=1)
        columns[prefix + "_t"] = table[original + "t"]
    columns.update(stock_elapsed=r["stock_t"] - r["t"], fast_elapsed=f["t"] - r["t"],
                   fast_path=f["path"], fast_material_path=f["material_path"])
    for key in ("steps", "pion_decay"):
        if key in f:
            if f[key].shape != (n,):
                raise ValueError(f"invalid fast {key} shape")
            columns["fast_" + key] = f[key]
    if geometry is not None:
        missed = outcomes["stock_mu+"] & ~outcomes["fast_mu+"]
        columns["fast_terminal_material_bitmask"], out["terminal_position_materials"] = \
            _terminal_materials(f["x"], geometry, f["fate"], missed)
    return out, columns


def _csv(path, columns, mask):
    arrays = [v[mask] for v in columns.values()]
    formats = ["%d" if v.dtype.kind in "biu" else "%.17g" for v in arrays]
    np.savetxt(path, np.rec.fromarrays(arrays, names=list(columns)), fmt=formats,
               delimiter=",", header=",".join(columns), comments="")


def markdown(out):
    """Concise narrative companion to the complete JSON and row tables."""
    lines = ["# Saved development TS diagnosis", "",
             f"Exact-ID joined source lineages: **{out['ninput']:,}**.",
             f"Saved event IDs: {out.get('saved_event_count', 'unavailable')}; no raw-file recount.", "",
             "## DS contingency by source", "",
             "| Source | Inputs | Stock DS | Fast DS | Both | Stock only | Fast only | Neither |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in out["source_groups"].items():
        lines.append(f"| {name} | {c['ninput']} | {c['stock']} | {c['fast']} | {c['both']} | {c['stock_only']} | {c['fast_only']} | {c['neither']} |")
    lines += ["", "## Muon endpoint overlap", ""]
    for name, m in out["muons"].items():
        fates = ", ".join(f"{f}: {v}" for f, v in zip(FATES, m["stock_only_fast_fates"]) if v)
        lines += [f"- {name}: stock **{m['stock']}**, fast **{m['fast']}**, both **{m['both']}**, stock-only **{m['stock_only']}**, fast-only **{m['fast_only']}**.",
                  f"  Stock-only {name} fast fates: {fates or 'none'}."]
    lines += ["", "Stock-only here means missing the specified muon species; a fast DS endpoint of another species is not a terminal loss.",
              "`stock-only.csv/.npz` contains genuine stock-DS/fast-not-DS rows; `muplus-union.csv/.npz` contains every stock or fast DS mu+ lineage, including all overlaps.",
              "Tables retain source, stock endpoint and fast terminal momentum vectors/magnitudes, positions, times, cumulative path/material length and available decay flags.",
              "Fast fate codes follow `fast_fates` in report.json; stock NaNs denote absent stock endpoints, and ancestry -1 denotes unavailable flags.",
              ""]
    if "terminal_position_materials" in out:
        m = out["terminal_position_materials"]
        lines += ["## TERMINAL POSITION classification", "", m["limitations"],
                  f"Material bit map: {m['bit_to_material']}; zero = no selected material.",
                  f"Stock-only mu+ bitmask columns: {m['stock_only_muplus_material_bitmasks']}.",
                  f"Counts by fast_fates row: {m['stock_only_muplus_fast_fate_by_material_bitmask']}.", ""]
    lines += ["", "## Upstream ancestry-gap association", ""]
    if out["ancestry"] is None:
        lines += ["No per-source gap flag is saved; association unavailable."]
    else:
        lines += [f"Flagged inputs: {out['ancestry']['n_gap']:,}. Tables/rates below are descriptive; charge- and PDG-stratified results are in report.json.", ""]
        for name in ("stock_ds", "fast_ds", "stock_mu+", "fast_mu+"):
            c = out["ancestry"]["source_groups"]["all"][name]
            lines += [f"- {name}: [lost, survived] without/with gap = {c['gap_by_survival']}; survival fractions {c['survival_fraction_no_gap']} / {c['survival_fraction_gap']}; phi {c['phi']}."]
    lines += ["", "## Supported interpretation and unresolved alternatives", ""]
    lines += ["The yield difference is a net balance of stock-only and fast-only lineages, not necessarily a subset removed by the fast model. The contingency and terminal-state tables locate the disagreement in this saved realization; they do not isolate a mechanism.", ""]
    lines += ["- " + text for text in out["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory (never overwritten)")
    parser.add_argument("--gdml", type=Path, help="matching GDML for TERMINAL POSITION selected-material classification")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output directory must be fresh")
    for path in (args.reference, args.fast):
        if path.suffix != ".npz" or "heldout" in str(path).lower().replace("-", "").replace("_", ""):
            parser.error("saved development NPZ files only")
    started = time.perf_counter()
    r, f = _load(args.reference), _load(args.fast)
    for table in (r, f):
        shards = table.get("parent_shard_ids", np.column_stack((table["id"][:, 0], table["shard"])))
        if np.any((shards[:, 1] < 920000) | (shards[:, 1] >= 920100)) or np.any((table["shard"] < 920000) | (table["shard"] >= 920100)):
            parser.error("Run1A development subruns 920000..920099 only")
    geometry = None
    if args.gdml is not None:
        from .tsgeometry import Geometry
        geometry = Geometry(args.gdml)
    out, rows = diagnose(r, f, geometry)
    out["inputs"] = {"reference": str(args.reference.resolve()), "fast": str(args.fast.resolve())}
    if args.gdml is not None:
        out["inputs"]["gdml"] = str(args.gdml.resolve())
    out["diagnosis_seconds"] = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=False)
    stock, fast = rows["stock_reached"], rows["fast_fate"] == 1
    masks = {"stock-only": stock & ~fast,
             "muplus-union": (stock & (rows["stock_pdg"] == -13)) | (fast & (rows["fast_pdg"] == -13))}
    for name, mask in masks.items():
        _csv(args.out / (name + ".csv"), rows, mask)
        np.savez_compressed(args.out / (name + ".npz"), **{k: v[mask] for k, v in rows.items()})
    (args.out / "report.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    (args.out / "report.md").write_text(markdown(out))
    print(markdown(out))


if __name__ == "__main__":
    main()
