"""Multi-muon identity audit of donor CrvStep tables: the historical (event, bar) library crossings against the
per-primary ones of tier1b.build(per_primary=True).

    python3 scripts/audit_native_primary_identity.py --config CFG.json --out NEW.json

CFG.json: {"donors": [crvsteps...root], "run_config": "run1a_v01", "counters": optional, "generators": optional}.
Per donor file: primary muons per event, event/bar crossings that merge several primary muons, descendant steps the
historical attachment gives to another primary's crossing (wrong ancestor), descendants of primaries without a
crossing of their own (left out of the per-primary attachment; near-miss material) and parentless steps (no primary
through the SimParticle parent chain: the unattached bin), per sector. The per-primary attachment is checked against
an independent reference (crossings split by primary key, same score among the primary's own crossings).
Optional generator streams report full-stream muon multiplicity. Read only; output is one JSON file.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, geometry, runconfig, tables, tier1b
from crvfastsim.analyses import census

KEY = tables.KEY
ANCESTOR = KEY + ["prim_key"]
SCHEMA = "crvfastsim.native_primary_identity_audit.v2"


def _isin(left, right, columns):
    return pd.MultiIndex.from_frame(left[columns]).isin(pd.MultiIndex.from_frame(right[columns]))


def _primaries(parts):
    return parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == 13)].copy()


def _primary_counts(parts, events):
    prim = _primaries(parts)
    counts = prim.groupby(KEY, sort=False).size()
    counts = counts.reindex(pd.MultiIndex.from_frame(events[KEY].drop_duplicates()), fill_value=0)
    hist = counts.value_counts().sort_index()
    return {"events": int(len(counts)), "muons": int(counts.sum()),
            "multi_muon_events": int((counts > 1).sum()),
            "max_per_event": int(counts.max()) if len(counts) else 0,
            "histogram": {str(int(n)): int(v) for n, v in hist.items()}}


def _measure(frame):
    return {"rows": int(len(frame)), "edep": float(frame.edep.to_numpy(dtype=np.float64).sum()),
            "events": int(len(frame[KEY].drop_duplicates())),
            "ancestors": int(len(frame[ANCESTOR].drop_duplicates()))}


def _select(desc, crossings, identity_safe=False):
    """Historical samebar=-1, midpoint distance, merge order and groupby did idxmin."""
    columns = KEY + ["did", "bar", "x0", "y0", "z0", "x1", "y1", "z1"]
    on = ANCESTOR if identity_safe else KEY
    if identity_safe:
        columns += ["prim_key"]
    xcolumns = on + ["xid", "sp", "bar", "x0", "y0", "z0"]
    candidates = desc[columns].merge(crossings[xcolumns].rename(columns={
        "sp": "selected_primary", "bar": "xbar", "x0": "ex", "y0": "ey", "z0": "ez"}), on=on)
    mid = 0.5 * (candidates[["x0", "y0", "z0"]].to_numpy()
                 + candidates[["x1", "y1", "z1"]].to_numpy())
    dist = np.linalg.norm(mid - candidates[["ex", "ey", "ez"]].to_numpy(), axis=1)
    candidates["score"] = np.where(candidates.bar.to_numpy() == candidates.xbar.to_numpy(), -1., dist)
    best = candidates.loc[candidates.groupby("did").score.idxmin(),
                          ["did", "xid", "selected_primary", "xbar", "score"]]
    return desc.merge(best, on="did")


def _safe_crossings(mu):
    ordered = mu.sort_values(KEY + ["bar", "t0"], kind="stable")
    grouped = ordered.groupby(ANCESTOR + ["bar"], sort=False)
    x = grouped.first()[["sp", "t0", "x0", "y0", "z0"]].reset_index()
    x["xid"] = np.arange(len(x), dtype=np.int64)
    return x


def analyze(steps, parts, geom):
    """Quantify historical identity loss vs an ancestor-restricted diagnostic reference.

    The reference splits crossings by actual primary key, then uses the same score
    only among that ancestor's crossings. It is not a nearest-primary truth join.
    No acceptance thresholds or claims about independent-sample closure are made.
    """
    s = census.classify(steps, parts, geom.sector)
    prim = _primaries(parts).rename(columns={"key": "prim_key"})
    mu = s.loc[s.cls.eq("mu_primary")]
    desc = s.loc[s.cls.eq("mu_descendant")].reset_index(drop=True).copy()
    desc["did"] = np.arange(len(desc), dtype=np.int64)
    historical = tier1b._crossings(mu)
    historical["xid"] = np.arange(len(historical), dtype=np.int64)
    reference = _safe_crossings(mu)
    repaired_crossings = tier1b._crossings(mu, per_primary=True)
    repaired_crossings["xid"] = np.arange(len(repaired_crossings), dtype=np.int64)
    best = tier1b.attach_descendants(desc, repaired_crossings, per_primary=True)
    repaired_attached = desc.merge(best, on="did").merge(
        repaired_crossings[["xid", "sp", "bar"]].rename(columns={"sp": "selected_primary", "bar": "xbar"}), on="xid")
    repaired_unattached = desc.loc[~desc.did.isin(best.did)]
    multiplicity = mu.groupby(KEY + ["bar"], sort=False).prim_key.nunique()
    merged = multiplicity[multiplicity > 1].reset_index()[KEY + ["bar"]]
    affected = mu.loc[_isin(mu, merged, KEY + ["bar"])]
    old = _select(desc, historical)
    safe = _select(desc, reference, identity_safe=True)
    fields = ["did", "selected_primary", "xbar"]
    expected_attachment = safe[fields].sort_values("did").to_numpy(dtype=np.int64)
    actual_attachment = repaired_attached[fields].sort_values("did").to_numpy(dtype=np.int64)
    if not np.array_equal(expected_attachment, actual_attachment):
        raise ValueError("builder per-primary attachment differs from the independent exact-ancestor reference")
    wrong = old.loc[old.prim_key != old.selected_primary]
    no_ancestor = desc.loc[~_isin(desc, mu, ANCESTOR)]
    old_unattached = desc.loc[~_isin(desc, mu, KEY)]
    excluded = no_ancestor.loc[_isin(no_ancestor, mu, KEY)]
    no_crossing = prim.loc[~_isin(prim, mu, ANCESTOR)]
    old_no_crossing = prim.loc[~_isin(prim, mu, KEY)]
    excluded_prim = no_crossing.loc[_isin(no_crossing, mu, KEY)]
    both = old[["did", "selected_primary", "xbar"]].merge(
        safe[["did", "selected_primary", "xbar"]], on="did", suffixes=("_old", "_safe"))
    changed = both.loc[(both.selected_primary_old != both.selected_primary_safe)
                       | (both.xbar_old != both.xbar_safe), ["did"]]
    events = pd.concat([parts[KEY], steps[KEY]], ignore_index=True).drop_duplicates()
    return {"primary_muons": _primary_counts(parts, events),
            "per_primary_builder": {"crossings": len(repaired_crossings), "attached": _measure(repaired_attached),
                "unattached": _measure(repaired_unattached), "independent_reference_identity_equal": True},
            "parentless": _measure(s.loc[s.cls.eq("unknown")].assign(prim_key=-1)),
            "crossings": {"historical_event_bar": int(len(historical)),
                          "ancestor_key_bar_reference": int(len(reference)),
                          "merged_event_bar": int(len(merged)),
                          "merged_events": int(len(merged[KEY].drop_duplicates())),
                          "max_primaries_per_event_bar": int(multiplicity.max()) if len(multiplicity) else 0,
                          "primary_multiplicity_histogram": {str(int(n)): int(v) for n, v in
                                                             multiplicity.value_counts().sort_index().items()},
                          "affected_primary_steps": int(len(affected)),
                          "affected_primary_edep": float(affected.edep.sum()),
                          "affected_primary_length": float(affected.len.sum())},
            "descendants": {"total": _measure(desc), "historical_attached": _measure(old),
                            "historical_unattached": _measure(old_unattached),
                            "identity_safe_attached": _measure(safe),
                            "intended_no_ancestor_crossing": _measure(no_ancestor),
                            "event_only_nearmiss_excluded": _measure(excluded),
                            "historical_wrong_ancestor": _measure(wrong),
                            "historical_wrong_ancestor_samebar": _measure(wrong.loc[wrong.score == -1.]),
                            "historical_wrong_ancestor_distance": _measure(wrong.loc[wrong.score != -1.]),
                            "reference_changed_primary_or_bar": _measure(desc.merge(changed, on="did"))},
            "donor_local_nearmiss": {"identity_no_crossing_muons": int(len(no_crossing)),
                                     "historical_event_only_no_crossing_muons": int(len(old_no_crossing)),
                                     "event_only_excluded_muons": int(len(excluded_prim)),
                                     "excluded_muons_with_descendants": int(len(excluded[ANCESTOR].drop_duplicates()))}}


def run(config, out):
    """Analyze the donors of config (a mapping or JSON file) one at a time and write the JSON report to out."""
    cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else dict(config)
    if set(cfg) - {"donors", "run_config", "counters", "generators"} or not {"donors", "run_config"} <= set(cfg):
        raise ValueError("config requires donors, run_config and optional counters/generators only")
    rc = runconfig.get(cfg["run_config"])
    if cfg.get("counters") is not None:
        rc = replace(rc, counters=str(cfg["counters"]))
    geom = rc.geometry()
    files = []
    for path in cfg["donors"]:
        report = genassoc.validate_donor(path)
        steps = tables.read(path, "steps")
        parts = tables.read(path, "particles", KEY + ["key", "parent", "pdg", "isprim"])
        events = tables.read(path, "events", KEY)
        measurements = analyze(steps, parts, geom)
        measurements["primary_muons"] = _primary_counts(parts, events)
        s = census.classify(steps, parts, geom.sector)
        desc = s.loc[s.cls.eq("mu_descendant")]
        no_own = desc.loc[~_isin(desc, s.loc[s.cls.eq("mu_primary")], ANCESTOR)]
        measurements["identity_census"] = tier1b._identity_census(s, no_own, geom, parts)
        files.append({"path": str(path), "validation": report, "measurements": measurements})
    full_reports = []
    for path in cfg.get("generators", []):
        gen = tables.read(path, "gen", KEY + ["idx", "pdg"])
        events = tables.read(path, "events", KEY)
        mu = gen.loc[gen.pdg.abs().eq(13)]
        counts = mu.groupby(KEY).size().reindex(pd.MultiIndex.from_frame(events[KEY]), fill_value=0)
        full_reports.append({"path": str(path), "events": int(len(counts)), "muons": int(counts.sum()),
            "multi_muon_events": int((counts > 1).sum()), "max_per_event": int(counts.max()) if len(counts) else 0,
            "histogram": {str(int(n)): int(v) for n, v in counts.value_counts().sort_index().items()}})
    result = {"schema": SCHEMA, "run_config": rc.name, "full_generator_multiplicity": full_reports, "files": files}
    Path(out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON: explicit donors, run_config, optional counters")
    parser.add_argument("--out", required=True, help="JSON report")
    args = parser.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
