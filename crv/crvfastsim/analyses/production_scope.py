"""Read-only S1 retained-population diagnostics; unmatched spectra never measure a selection efficiency."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import uproot

from crvfastsim import runconfig, tables
from crvfastsim.analyses import scatter_convergence as SC, transport_validation as TV, v1strata as VS

KEY = tables.KEY
BINS = {
    "len": [0., 5., 10., 15., 19., 20., 21., 25., 30., 40., 50., 75., 100., 200., 500., 1000., 1e9],
    "edep": [0., .1, 1., 2., 3., 4., 5., 6., 8., 10., 15., 20., 30., 50., 100., 1e9],
    "p": [0., 100., 300., 600., 1000., 2000., 5000., 10000., 20000., 100000., 1e12],
    "abs_uy": [0., .05, .1, .2, .3, .5, .7, .9, 1.000001],
}


def crossing_rows(steps, parts, geom):
    """Event/bar totals and earliest primary-muon entry state without any generator-index join."""
    if parts.duplicated(KEY + ["key"]).any():
        raise ValueError("duplicate SimParticle identity")
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == 13), KEY + ["key"]]
    s = steps.merge(prim.rename(columns={"key": "sp"}), on=KEY + ["sp"], validate="many_to_one")
    if not np.isfinite(s[["len", "edep", "t0", "px0", "py0", "pz0"]]).all().all():
        raise ValueError("nonfinite crossing state")
    p = np.linalg.norm(s[["px0", "py0", "pz0"]], axis=1)
    if (p <= 0).any() or ((s.bar < 0) | (s.bar >= len(geom.sector))).any():
        raise ValueError("invalid crossing momentum or bar")
    s = s.assign(p=p, abs_uy=np.abs(s.py0.to_numpy()) / p)
    first = s.sort_values(KEY + ["bar", "t0"], kind="stable").drop_duplicates(KEY + ["bar"])
    x = tables.primary_muon_crossings(steps, parts, geom.sector)
    return x.merge(first[KEY + ["bar", "sp", "p", "abs_uy"]], on=KEY + ["bar"], validate="one_to_one")


def spectra(x):
    """Fixed bins and moments, retaining sparse sectors and explicit out-of-range counts."""
    out = {"crossings": len(x), "contributing_events": len(x[KEY].drop_duplicates()), "variables": {}}
    for name, edges in BINS.items():
        v = x[name].to_numpy(dtype=np.float64)
        out["variables"][name] = {"bins": edges, "counts": np.histogram(v, edges)[0].tolist(),
                                    "underflow": int((v < edges[0]).sum()), "overflow": int((v > edges[-1]).sum()),
                                    "sum": float(v.sum()), "sum2": float(v @ v),
                                    "mean": float(v.mean()) if len(v) else None}
    return out


def pool_spectra(a, b):
    """Add disjoint-file spectra, not their normalized densities."""
    if not a:
        return b
    out = {"crossings": a["crossings"] + b["crossings"],
           "contributing_events": a["contributing_events"] + b["contributing_events"], "variables": {}}
    for name in BINS:
        u, v = a["variables"][name], b["variables"][name]
        row = {"bins": u["bins"], "counts": (np.array(u["counts"]) + v["counts"]).tolist()}
        row.update({k: u[k] + v[k] for k in ("underflow", "overflow", "sum", "sum2")})
        row["mean"] = row["sum"] / out["crossings"] if out["crossings"] else None
        out["variables"][name] = row
    return out


def parse_stage_log(path, census):
    """Read event-labelled stock PrintModule rows; require an explicit collection census for every event."""
    text = Path(path).read_text()
    summaries = re.findall(r"TrigReport Events total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)", text)
    if summaries != [(str(len(census)), str(len(census)), "0")]:
        raise ValueError("scope log must finish the complete selected-event stream")
    product = "ProductPrint mu2e::StepPointMCs_cosmicFilter_crvStage1_Primary"
    if text.count(product) != len(census):
        raise ValueError("scope log must identify the exact retained DS-entry product")
    header = r"^\s*#+\s+PrintModule Run/Subrun/Event\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
    collection = r"^StepPointMCCollection has (\d+) steps\s*$"
    number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    row = r"^\s*(\d+)\s+(\d+)\s+(\d+)" + (r"\s+(" + number + r")") * 8 + r"\s+(\S+)\s*$"
    fields = KEY + ["expected", "index", "sp", "volume", "edep", "nonion", "x", "y", "z", "p", "t", "proper", "process"]
    found = re.findall("(?:" + header + ")|(?:" + collection + ")|(?:" + row + ")", text, re.MULTILINE)
    frame = pd.DataFrame(found, columns=fields).replace("", np.nan)
    is_header = frame.run.notna()
    events = frame.loc[is_header, KEY].astype(np.int64)
    if events.duplicated(KEY).any() or len(events) != len(census):
        raise ValueError("scope event headers incomplete or duplicated")
    joined = census[KEY].merge(events, on=KEY, how="outer", indicator=True, validate="one_to_one")
    if (joined._merge != "both").any():
        raise ValueError("scope event universe differs from extraction")
    frame[KEY] = frame[KEY].ffill()
    expected = frame.loc[frame.expected.notna(), KEY + ["expected"]].astype(np.int64)
    if expected.duplicated(KEY).any() or len(expected) != len(census):
        raise ValueError("scope collection counts incomplete or duplicated")
    hits = frame.loc[frame["index"].notna()].copy()
    hits = hits.drop(columns="expected")
    for name in KEY + ["index", "sp", "volume"]:
        hits[name] = hits[name].astype(np.int64)
    for name in ("edep", "nonion", "x", "y", "z", "p", "t", "proper"):
        hits[name] = hits[name].astype(float)
    counts = hits.groupby(KEY).size().rename("actual").reset_index()
    checked = expected.merge(counts, on=KEY, how="left", validate="one_to_one").fillna({"actual": 0})
    if not (checked.expected == checked.actual).all() or hits.duplicated(KEY + ["index"]).any():
        raise ValueError("scope printed rows do not match collection counts")
    order = hits.groupby(KEY)["index"].agg(["min", "max", "size"])
    if not ((order["min"] == 0) & (order["max"] == order["size"] - 1)).all():
        raise ValueError("scope collection indices are not contiguous")
    return hits


def primary_roots(parts):
    """Resolve parent trees by vectorized pointer doubling, with exact event/particle identity."""
    keys = KEY + ["key"]
    if parts.duplicated(keys).any():
        raise ValueError("duplicate SimParticle identity")
    index = pd.MultiIndex.from_frame(parts[keys])
    parents = parts[KEY + ["parent"]].rename(columns={"parent": "key"})
    has_parent = parts.parent.to_numpy() >= 0
    parent_id = index.get_indexer(pd.MultiIndex.from_frame(parents))
    if (has_parent & (parent_id < 0)).any():
        raise ValueError("missing SimParticle parent identity")
    root = np.where(has_parent, parent_id, np.arange(len(parts)))
    for _ in range(64):
        next_root = root[root]
        if np.array_equal(next_root, root):
            break
        root = next_root
    else:
        raise ValueError("nonconvergent SimParticle ancestry")
    if has_parent[root].any():
        raise ValueError("cyclic SimParticle ancestry")
    out = parts[keys].copy()
    out["root_pdg"] = parts.pdg.to_numpy()[root]
    out["root_isprim"] = parts.isprim.to_numpy()[root]
    return out.rename(columns={"key": "sp"})


def stage_population(hits, parts, steps, geom, groups):
    """Attribute actual retained DS-entry steps by particle identity, never a ray or end-state proxy."""
    p = parts[KEY + ["key", "parent", "pdg", "isprim"]]
    if p.duplicated(KEY + ["key"]).any():
        raise ValueError("duplicate SimParticle identity")
    h = hits.merge(p.rename(columns={"key": "sp"}), on=KEY + ["sp"], how="left", validate="many_to_one")
    h = h.merge(primary_roots(parts), on=KEY + ["sp"], how="left", validate="many_to_one")
    if h.pdg.isna().any():
        raise ValueError("DS-entry step missing retained particle identity")
    if not np.isfinite(h[["p", "t"]]).all().all() or (h.p < 0).any():
        raise ValueError("DS-entry step has invalid printed momentum/time")
    direct = h.loc[(h.isprim == 1) & (h.pdg.abs() == 13)]
    entered = direct.groupby(KEY + ["sp"])["t"].min().rename("ds_entry_t").reset_index()
    x = crossing_rows(steps, parts, geom)
    marked = x.merge(entered, on=KEY + ["sp"], how="left", validate="many_to_one")
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == 13), KEY + ["key"]]
    mu_steps = steps.merge(prim.rename(columns={"key": "sp"}), on=KEY + ["sp"], validate="many_to_one")
    mu_bars = mu_steps.groupby(KEY + ["sp", "bar"], sort=False).agg(t0=("t0", "min")).reset_index()
    mu_bars["sector"] = np.asarray(geom.sector)[mu_bars.bar.to_numpy()]
    mu_bars = mu_bars.merge(entered, on=KEY + ["sp"], how="left", validate="many_to_one")
    event_hits = h[KEY].drop_duplicates()
    event_primary_mu = direct[KEY].drop_duplicates()
    without_direct = event_hits.merge(event_primary_mu, on=KEY, how="left", indicator=True).query('_merge == "left_only"')[KEY]
    n_secondary_only = len(without_direct)
    mu_ancestry = h.loc[(h.root_isprim == 1) & (h.root_pdg.abs() == 13), KEY].drop_duplicates()
    secondary_mu_only = without_direct.merge(mu_ancestry, on=KEY, how="inner", validate="one_to_one")
    result = {"stage_step_rows": len(h), "events_with_stage_steps": len(event_hits),
              "events_with_primary_muon_stage_steps": len(event_primary_mu),
              "events_without_direct_primary_muon_entry": n_secondary_only,
              "without_direct_but_with_primary_muon_ancestor": len(secondary_mu_only),
              "without_primary_muon_ancestor_in_stage_particles": n_secondary_only - len(secondary_mu_only),
              "primary_muon_stage_steps": len(direct), "nonprimary_stage_steps": int((h.isprim != 1).sum()),
              "primary_other_pdg_stage_steps": int(((h.isprim == 1) & (h.pdg.abs() != 13)).sum()), "groups": {},
              "scope": "actual retained DS-entry identity; event/bar earliest primary entry; not rejected-event efficiency",
              "printed_time_precision_ns": .001, "printed_momentum_precision_MeV": .001}
    for name, sectors in groups.items():
        s = marked.loc[marked.sector.isin(sectors)]
        has = s.ds_entry_t.notna()
        before = has & (s.t0 <= s.ds_entry_t + .001)
        per_mu = mu_bars.loc[mu_bars.sector.isin(sectors)]
        own = per_mu.ds_entry_t.notna()
        result["groups"][name] = {"crossings": len(s), "direct_primary_muon_enters_ds": int(has.sum()),
                                   "earliest_crv_entry_before_ds": int(before.sum()),
                                   "earliest_crv_entry_after_ds": int((has & ~before).sum()),
                                   "no_direct_entry_for_earliest_primary": int((~has).sum()),
                                   "particle_bar_crossings": len(per_mu),
                                   "particle_bar_direct_ds_entry": int(own.sum()),
                                   "particle_bar_no_direct_ds_entry": int((~own).sum())}
    return result


def compare(production, reference):
    """Unmatched retained spectra document the scope difference, not its physical attribution."""
    out = {}
    for name, p in production.items():
        r = reference[name]
        row = {"production_crossings": p["crossings"], "reference_crossings": r["crossings"], "variables": {}}
        for v in BINS:
            a, b = p["variables"][v], r["variables"][v]
            ac, bc = np.array(a["counts"], dtype=float), np.array(b["counts"], dtype=float)
            row["variables"][v] = {"production_mean": a["mean"], "reference_mean": b["mean"],
                                      "binned_total_variation": float(.5 * np.abs(ac / ac.sum() - bc / bc.sum()).sum())
                                      if ac.sum() and bc.sum() else None}
        out[name] = row
    return {"groups": out, "selection_matched": False,
            "interpretation": "descriptive unmatched populations; not a fast/G4 residual, efficiency, or correction",
            "exact_reference_ds_entry_selection": {"supported": False,
                "reason": "archived tables lack DS-entry StepPointMCs and end-volume identity; no ray/end-state substitute"}}


def write_summary(result, path):
    """Readable observed-population summary with no equivalence or missing-event claim."""
    lines = ["# Production scope: retained S1 versus full reference", "",
             "Unmatched populations: these are not fast/G4 residuals, selection efficiencies or corrections.", ""]
    ds = result.get("ds_entry_population")
    if ds:
        lines += [f"Retained events with DS-entry steps: {ds['events_with_stage_steps']:,}; "
                  f"without direct primary-muon entry: {ds['events_without_direct_primary_muon_entry']:,}.", "",
                  "| Group | Primary-muon particle/bar crossings | Same muon enters DS | No direct DS entry |",
                  "|---|---:|---:|---:|"]
        for name, row in ds["groups"].items():
            if name.startswith("group:"):
                lines.append(f"| {name[6:]} | {row['particle_bar_crossings']:,} | "
                             f"{row['particle_bar_direct_ds_entry']:,} | {row['particle_bar_no_direct_ds_entry']:,} |")
    lines += ["", "| Group | S1 mean length (mm) | Full mean length (mm) | S1 mean entry p (GeV/c) | Full mean entry p (GeV/c) |",
              "|---|---:|---:|---:|---:|"]
    for name, row in result["comparison"]["groups"].items():
        if name.startswith("group:") and row["production_crossings"] and row["reference_crossings"]:
            v = row["variables"]
            lines.append(f"| {name[6:]} | {v['len']['production_mean']:.3f} | {v['len']['reference_mean']:.3f} | "
                         f"{v['p']['production_mean'] / 1000:.3f} | {v['p']['reference_mean'] / 1000:.3f} |")
    lines += ["", "The stored full-reference tables cannot reconstruct the exact DS-entry selection or missing post-DS trajectories.",
              "Field, physics-list, world and selection effects are not separated by this comparison."]
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--production-table", required=True)
    ap.add_argument("--production-audit", required=True)
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--stage-log")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    audit = json.loads(Path(a.production_audit).read_text())
    artifact = audit["artifacts"].get(a.production_table)
    if artifact is None or artifact["sha256"] != TV.fingerprint(a.production_table)["sha256"]:
        raise ValueError("production table differs from validated extraction")
    rc = runconfig.get("run2_v10")
    geom, groups = rc.geometry(), VS.groups_of(rc, rc.geometry())
    with uproot.open(a.production_table) as f:
        census = f["dump/events"].arrays(KEY, library="pd")
    steps, parts = (tables.read(a.production_table, name) for name in ("steps", "particles"))
    x = crossing_rows(steps, parts, geom)
    prod = {n: spectra(x.loc[x.sector.isin(s)]) for n, s in groups.items()}
    result = {"schema": "crv-production-scope-v1", "analysis_source": TV.fingerprint(__file__),
              "artifacts": {p: TV.fingerprint(p) for p in (a.production_table, a.production_audit)},
              "production": {"events": len(census), "groups": prod}, "reference": {"groups": {}}}
    if a.stage_log:
        hits = parse_stage_log(a.stage_log, census)
        if len(hits[KEY].drop_duplicates()) != len(census):
            raise ValueError("selected S1 event missing DS-entry steps")
        result["ds_entry_population"] = stage_population(hits, parts, steps, geom, groups)
        result["artifacts"][a.stage_log] = TV.fingerprint(a.stage_log)
    manifest, rows = SC.reference_shards(a.reference_root, list(range(101, 121)))
    result["reference"]["manifest"] = TV.fingerprint(manifest)
    result["reference"]["records"] = []
    for subrun, paths in rows:
        s, p = (tables.read(paths["crvsteps"], name) for name in ("steps", "particles"))
        accepted = len(p[KEY].drop_duplicates())
        if accepted != paths["expected_accepted"]:
            raise ValueError("reference accepted census differs from manifest")
        result["reference"]["records"].append({"subrun": subrun, "accepted": accepted})
        rx = crossing_rows(s, p, geom)
        for n, sectors in groups.items():
            row = spectra(rx.loc[rx.sector.isin(sectors)])
            result["reference"]["groups"][n] = pool_spectra(result["reference"]["groups"].get(n), row)
    result["comparison"] = compare(prod, result["reference"]["groups"])
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    write_summary(result, out.with_suffix(".md"))
    print(json.dumps({"production_crossings": len(x), "all_sectors": len(geom.sector_names),
                      "selection_matched": False, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
