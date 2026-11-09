"""Audit persisted transport tables and compare all sectors with event-paired count/mean errors.

Run with python3 -m crvfastsim transport-validation --help. Entry momentum spectra are
truth diagnostics, independently binned in each arm; they are not conditional efficiency estimates.
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from crvfastsim import geometry, runconfig, tables
from mu2efastsim import eloss

P_EDGES = np.array([0., 300., 600., 1000., 2000., 5000., np.inf])


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def audit(path):
    out = fingerprint(path)
    out["trees"] = {}
    with uproot.open(path) as f:
        for name in ("gen", "particles", "steps", "subruns"):
            if f"dump/{name}" not in f:
                continue
            tree = f[f"dump/{name}"]
            cycles = sorted(int(k.rsplit(";", 1)[1]) for k in f["dump"].keys()
                            if k.startswith(name + ";"))
            keys = tables.KEY if name != "subruns" else tables.KEY[:2]
            ids = tree.arrays(keys, library="pd")
            out["trees"][name] = {"cycles": cycles, "selected_cycle": max(cycles),
                                    "rows": int(tree.num_entries), "unique_ids": len(ids.drop_duplicates())}
    return out


def primary_crossings(path, gen, geom, fast_format=False):
    steps, parts = tables.read(path, "steps"), tables.read(path, "particles")
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == 13)].copy()
    if "genidx" in prim:
        identity = prim[tables.KEY + ["key", "genidx"]]
    else:
        if not fast_format:
            raise ValueError(f"reference particles require genidx in {path}")
        gm = gen.loc[gen.pdg.abs() == 13].sort_values(tables.KEY + ["idx"], kind="stable").copy()
        gm["key"] = gm.groupby(tables.KEY, sort=False).cumcount() + 1
        identity = prim[tables.KEY + ["key"]].merge(
            gm[tables.KEY + ["key", "idx"]].rename(columns={"idx": "genidx"}),
            on=tables.KEY + ["key"], how="left", validate="one_to_one")
    if identity.genidx.isna().any() or (identity.genidx < 0).any():
        raise ValueError(f"missing primary identity in {path}")
    if identity.duplicated(tables.KEY + ["genidx"]).any():
        raise ValueError(f"duplicate primary identity in {path}")
    s = steps.merge(identity.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], validate="many_to_one")
    s["p"] = np.linalg.norm(s[["px0", "py0", "pz0"]], axis=1)
    s = s.sort_values(tables.KEY + ["genidx", "bar", "t0"], kind="stable")
    x = s.groupby(tables.KEY + ["genidx", "bar"], sort=False).agg(
        p=("p", "first"), edep=("edep", "sum"), length=("len", "sum"), t0=("t0", "first")).reset_index()
    source_ids = gen[tables.KEY + ["idx"]].rename(columns={"idx": "genidx"})
    valid = x.merge(source_ids, on=tables.KEY + ["genidx"], how="left", indicator=True, validate="many_to_one")
    if (valid._merge != "both").any():
        raise ValueError(f"crossing source identity absent from generator in {path}")
    x["sector"] = geom.sector[x.bar.to_numpy()]
    if not np.isfinite(x[["p", "edep", "length", "t0"]]).all().all():
        raise ValueError(f"nonfinite primary crossings in {path}")
    return x


def paired_count(a, b, universe):
    """Relative A/B count shift and delta-method SE clustered by generated event, including zero rows."""
    keys = tables.KEY
    ac = a.groupby(keys).size().rename("a")
    bc = b.groupby(keys).size().rename("b")
    both = universe.merge(ac, on=keys, how="left").merge(bc, on=keys, how="left").fillna(0.)
    av, bv = both.a.to_numpy(), both.b.to_numpy()
    n, sa, sb = len(both), av.sum(), bv.sum()
    moments = {"events": n, "a2": float(av @ av), "b2": float(bv @ bv), "ab": float(av @ bv)}
    return count_summary(int(sa), int(sb), moments)


def count_summary(sa, sb, moments):
    n = moments["events"]
    out = {"a": sa, "b": sb, "relative": None, "se": None, "moments": moments}
    if sb and n > 1:
        ratio = sa / sb
        ss = max(moments["a2"] - 2 * ratio * moments["ab"] + ratio ** 2 * moments["b2"], 0.)
        out.update(relative=float(ratio - 1), se=float(np.sqrt(n * ss / (n - 1)) / sb))
    return out


def ratio_moments(x, components):
    """Additive raw moments of per-event [sum_a, weight_a, sum_b, weight_b]."""
    totals, cross = x.sum(axis=0), x.T @ x
    if not np.isfinite(totals).all() or not np.isfinite(cross).all():
        raise ValueError("nonfinite paired ratio moments")
    return {"events": len(x), "contributing_events": int(np.count_nonzero((x[:, 1] > 0) | (x[:, 3] > 0))),
            "components": components, "totals": totals.tolist(), "cross_products": cross.tolist()}


def finite_ratio(numerator, denominator):
    value = numerator / denominator if denominator else None
    return float(value) if value is not None and np.isfinite(value) else None


def ratio_summary(moments):
    """Ratio of weighted means; delta-method SE with generated events as clusters.

    The gradient is with respect to sample totals. Its dot product with those
    totals vanishes, so n/(n-1) * g' (sum_i x_i x_i') g is the centered
    influence covariance without explicitly subtracting the mean. Zero-crossing
    events enter n. A zero numerator is supported; zero denominators are not.
    """
    sa, wa, sb, wb = moments["totals"]
    out = {"a": finite_ratio(sa, wa), "b": finite_ratio(sb, wb),
           "relative": None, "se": None}
    if out["a"] is None or out["b"] is None or not wa or not wb or not sb:
        return out
    scale = wb / wa
    ratio = (sa / sb) * scale
    gradient = np.array([scale / sb, -ratio / wa, -ratio / sb, ratio / wb])
    if not np.isfinite(ratio) or not np.isfinite(gradient).all():
        return out
    out["relative"] = float(ratio - 1)
    n = moments["events"]
    if n > 1 and moments["contributing_events"] > 1:
        cross = np.asarray(moments["cross_products"])
        # Pair the two arms before the final dot product: identical arms cancel exactly.
        projected = cross[:, :2] @ gradient[:2] + cross[:, 2:] @ gradient[2:]
        contraction = float(gradient @ projected)
        tolerance = 64 * np.finfo(float).eps * float(np.abs(gradient) @ np.abs(cross) @ np.abs(gradient))
        if contraction < -tolerance:
            raise ValueError("materially negative paired ratio variance")
        variance = max(contraction, 0.) * n / (n - 1)
        out["se"] = float(np.sqrt(variance)) if np.isfinite(variance) else None
    return out


def add_measure_summaries(out):
    """Reconstruct all mean/path ratios from additive moments (also used on merge)."""
    for name, moments in out["paired_measure_moments"].items():
        for field, value in ratio_summary(moments).items():
            out[name + "_" + field] = value
    n = out["moments"]["events"]
    out["crossings_per_generated_event"] = {arm: out[arm] / n if n else None for arm in ("a", "b")}
    return out


def compare_measures(a, b, universe):
    """Count, crossing means and sum(edep)/sum(length), without identity matching."""
    event_values = []
    for frame in (a, b):
        grouped = frame.groupby(tables.KEY).agg(n=("length", "size"), length=("length", "sum"),
                                                edep=("edep", "sum"))
        event_values.append(universe.merge(grouped, on=tables.KEY, how="left")
                            [["n", "length", "edep"]].fillna(0.).to_numpy(dtype=float))
    av, bv = event_values
    ac, bc = av[:, 0], bv[:, 0]
    out = count_summary(int(ac.sum()), int(bc.sum()),
                        {"events": len(universe), "a2": float(ac @ ac),
                         "b2": float(bc @ bc), "ab": float(ac @ bc)})
    out["sums"] = {col: {"a": float(av[:, i].sum()), "b": float(bv[:, i].sum())}
                   for i, col in enumerate(("length", "edep"), 1)}
    out["paired_measure_moments"] = {}
    for name, i, j, weight in (("length_mean", 1, 0, "n"), ("edep_mean", 2, 0, "n"),
                                ("energy_per_path", 2, 1, "length")):
        col = "length" if i == 1 else "edep"
        x = np.column_stack((av[:, i], av[:, j], bv[:, i], bv[:, j]))
        out["paired_measure_moments"][name] = ratio_moments(
            x, [col + "_a", weight + "_a", col + "_b", weight + "_b"])
    return add_measure_summaries(out)


def selection_summary(count, sums):
    """Unmatched-arm totals and crossing means; no unpaired uncertainty claim."""
    return {"count": count, "sums": sums,
            "means": {col: value / count if count else None for col, value in sums.items()},
            "energy_per_path": finite_ratio(sums["edep"], sums["length"])}


def compare(a, b, universe):
    out = compare_measures(a, b, universe)
    out["scope"] = ("Truth diagnostics, not efficiencies or physics gates; matched_only conditions on "
                    "common crossing identities; a_only/b_only follow the comparison arm labels.")
    out["entry_p_bins"] = [paired_count(a[(a.p >= lo) & (a.p < hi)],
                                          b[(b.p >= lo) & (b.p < hi)], universe)
                           for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:])]
    join = tables.KEY + ["genidx", "bar"]
    joined = a.merge(b, on=join, how="outer", suffixes=("_a", "_b"),
                     indicator=True, validate="one_to_one")
    matched = joined.loc[joined._merge == "both"]

    def arm_frame(frame, arm):
        cols = {col + "_" + arm: col for col in ("p", "length", "edep")}
        return frame[join + list(cols)].rename(columns=cols)

    out["identity_matched"] = len(matched)
    out["matched_only"] = compare_measures(arm_frame(matched, "a"), arm_frame(matched, "b"), universe)
    for arm, label in (("a", "left_only"), ("b", "right_only")):
        frame = arm_frame(joined.loc[joined._merge == label], arm)
        out[arm + "_only"] = selection_summary(len(frame),
                                               {col: float(frame[col].sum()) for col in ("length", "edep")})
    out["entry_p_difference_MeV_quantiles"] = (np.quantile(matched.p_a - matched.p_b, [.1, .5, .9]).tolist()
                                               if len(matched) else None)
    return out


def repo_state(module):
    root = Path(module.__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    tracked = ("transport.py", "tables.py", "eloss.py", "materials.py", "world.py", "crossings.py",
               "runconfig.py", "geometry.py", "analyses/transport.py")
    package = Path(module.__file__).parent
    hashes = {name: fingerprint(package / name)["sha256"] for name in tracked if (package / name).is_file()}
    return {"root": str(root), "head": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "source_sha256": hashes}


def accepted_count_evidence(persisted, expected):
    if expected is not None:
        if isinstance(expected, bool) or not isinstance(expected, (int, np.integer)) or expected < 0:
            raise ValueError("expected accepted count must be a nonnegative integer")
        if persisted != expected:
            raise ValueError(f"reference accepted event count mismatch: persisted {persisted}, expected {expected}")
    return {"persisted_reference_accepted_events": persisted,
            "expected_accepted": int(expected) if expected is not None else None,
            "accepted_count_gate": True if expected is not None else None,
            "reference_completeness": (
                "Persisted accepted-event count matches supplied external successful-worker count; "
                "count agreement does not verify accepted-event identities." if expected is not None else
                "Persisted accepted events counted; completeness requires external job evidence.")}


def print_comparisons(sectors):
    for name, values in sectors.items():
        c = values["candidate_vs_reference"]
        fields = [name, "candidate/ref", c["a"], c["b"], c["relative"], "+/-", c["se"]]
        if "candidate_vs_baseline" in values:
            d = values["candidate_vs_baseline"]
            fields.extend(["candidate/base", d["relative"], "+/-", d["se"]])
        print(*fields)


def run(gen_path, reference, baseline, candidate, config, expected_events, output, gdml=None, artifacts=(),
        expected_accepted=None):
    rc = runconfig.get(config)
    geom = rc.geometry()
    gen = tables.read(gen_path, "gen")
    universe = gen[tables.KEY].drop_duplicates()
    if len(universe) != expected_events or gen.duplicated(tables.KEY + ["idx"]).any():
        raise ValueError("generated event count or generator identity mismatch")
    paths = {"gen": gen_path, "reference": reference}
    if baseline is not None:
        paths["baseline"] = baseline
    paths["candidate"] = candidate
    inventories = {k: audit(p) for k, p in paths.items()}
    accepted = accepted_count_evidence(inventories["reference"]["trees"]["particles"]["unique_ids"],
                                       expected_accepted)
    for arm, path in paths.items():
        if arm == "gen":
            continue
        saved = tables.read(path, "gen")
        m = saved.merge(gen, on=tables.KEY + ["idx"], how="left", suffixes=("_s", "_g"),
                        indicator=True, validate="one_to_one")
        if (m._merge != "both").any():
            raise ValueError(f"{arm}: generated identities absent from source")
        for c in ("pdg", "px", "py", "pz", "x", "y", "z", "t"):
            if not np.allclose(m[c + "_s"], m[c + "_g"], rtol=1e-10, atol=1e-8):
                raise ValueError(f"{arm}: source mismatch in {c}")
        if arm in ("baseline", "candidate") and len(saved) != len(gen):
            raise ValueError(f"{arm}: incomplete generator coverage")
    x = {k: primary_crossings(p, gen, geom, fast_format=(k != "reference"))
         for k, p in paths.items() if k != "gen"}
    groups = {name: [i] for i, name in enumerate(geom.sector_names)}
    cfg = rc.counters_cfg()
    for typ, name in runconfig.sector_type_names(geom, cfg).items():
        groups["group:" + name] = [i for i, s in enumerate(geom.sector_names) if int(cfg[f"crs.sectorType{s}"]) == typ]
    comparisons = [("candidate_vs_reference", "candidate", "reference")]
    if baseline is not None:
        comparisons.insert(0, ("baseline_vs_reference", "baseline", "reference"))
        comparisons.append(("candidate_vs_baseline", "candidate", "baseline"))
    result = {}
    for name, sectors in groups.items():
        selected = {k: frame[frame.sector.isin(sectors)] for k, frame in x.items()}
        result[name] = {label: compare(selected[a], selected[b], universe) for label, a, b in comparisons}
    event_ranges = universe.groupby(["run", "subrun"]).event.agg(["min", "max", "size"]).reset_index().to_dict("records")
    out = {"run_config": config, "expected_events": expected_events, "persisted_generated_events": len(universe),
           "event_ranges": event_ranges, "diagnostic_sha256": fingerprint(__file__)["sha256"],
           **accepted,
           "inputs": inventories, "code": {"crv": repo_state(runconfig), "core": repo_state(eloss)},
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "uproot": uproot.__version__},
           "geometry": [fingerprint(geometry.find_counters(rc.counters))] + ([fingerprint(gdml)] if gdml else []),
           "artifacts": [fingerprint(p) for p in artifacts],
           "p_bin_edges_MeV": [0, 300, 600, 1000, 2000, 5000, None],
           "scope": "Truth diagnostic; event-cluster paired count/mean/path SE, independent per-arm entry-p bins, no G4/BO certification.",
           "crossing_key": tables.KEY + ["genidx", "bar"], "sectors": result}
    with Path(output).open("x") as stream:
        stream.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"{config}: {len(universe)} persisted generated events; output {output}")
    for label, value in inventories.items():
        print(label, json.dumps(value["trees"]))
    print_comparisons(result)
    return out


def merge_counts(rows):
    moments = {k: sum(r["moments"][k] for r in rows) for k in ("events", "a2", "b2", "ab")}
    out = count_summary(sum(r["a"] for r in rows), sum(r["b"] for r in rows), moments)
    if "sums" in rows[0]:
        out["sums"] = {col: {arm: sum(r["sums"][col][arm] for r in rows) for arm in ("a", "b")}
                       for col in ("length", "edep")}
        out["paired_measure_moments"] = {}
        for name, first in rows[0]["paired_measure_moments"].items():
            ms = [r["paired_measure_moments"][name] for r in rows]
            if any(m["components"] != first["components"] for m in ms):
                raise ValueError("incompatible paired moment components")
            out["paired_measure_moments"][name] = {
                "events": sum(m["events"] for m in ms),
                "contributing_events": sum(m["contributing_events"] for m in ms), "components": first["components"],
                **{key: np.sum([m[key] for m in ms], axis=0).tolist()
                   for key in ("totals", "cross_products")}}
        add_measure_summaries(out)
    if "identity_matched" in rows[0]:
        out["scope"] = rows[0]["scope"]
        out["identity_matched"] = sum(r["identity_matched"] for r in rows)
        out["matched_only"] = merge_counts([r["matched_only"] for r in rows])
        for arm in ("a", "b"):
            selected = [r[arm + "_only"] for r in rows]
            out[arm + "_only"] = selection_summary(sum(r["count"] for r in selected),
                {col: sum(r["sums"][col] for r in selected) for col in ("length", "edep")})
        out["entry_p_bins"] = [merge_counts([r["entry_p_bins"][i] for r in rows]) for i in range(len(P_EDGES) - 1)]
        # Quantiles are not additive. Preserve each shard's diagnostic, never average them.
        out["entry_p_difference_MeV_quantiles"] = rows[0]["entry_p_difference_MeV_quantiles"] if len(rows) == 1 else None
        out["entry_p_difference_MeV_shard_quantiles"] = [
            q for r in rows for q in r.get("entry_p_difference_MeV_shard_quantiles",
                                           [r["entry_p_difference_MeV_quantiles"]])]
    return out


def merge(paths, output):
    rows = [json.loads(Path(p).read_text()) for p in tables.expand(paths)]
    if not rows or len({r["run_config"] for r in rows}) != 1:
        raise ValueError("merge requires one nonempty run configuration")
    if len({r["diagnostic_sha256"] for r in rows}) != 1 or any(
            r["p_bin_edges_MeV"] != rows[0]["p_bin_edges_MeV"] for r in rows):
        raise ValueError("incompatible diagnostic implementation or momentum bins")
    comparison_sets = {frozenset(values) for r in rows for values in r["sectors"].values()}
    if len(comparison_sets) != 1:
        raise ValueError("incompatible comparison sets")
    sources = [r["inputs"]["gen"]["sha256"] for r in rows]
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate generator source in merge")
    identities = [(r["run"], r["subrun"]) for row in rows for r in row["event_ranges"]]
    if len(set(identities)) != len(identities):
        raise ValueError("merge requires disjoint run/subrun IDs")
    provenance = [(r["code"]["crv"]["head"], r["code"]["core"]["head"],
                   json.dumps({k: v["source_sha256"] for k, v in r["code"].items()}, sort_keys=True),
                   tuple(g["sha256"] for g in r["geometry"])) for r in rows]
    if len(set(provenance)) != 1 or any(r["sectors"].keys() != rows[0]["sectors"].keys() for r in rows):
        raise ValueError("incompatible code, geometry or sector coverage")
    expected_accepted = (sum(r["expected_accepted"] for r in rows)
                         if all(r.get("expected_accepted") is not None for r in rows) else None)
    accepted = accepted_count_evidence(
        sum(r["inputs"]["reference"]["trees"]["particles"]["unique_ids"] for r in rows), expected_accepted)
    out = {"run_config": rows[0]["run_config"], "shards": len(rows), **accepted,
           "persisted_generated_events": sum(r["persisted_generated_events"] for r in rows),
           "p_bin_edges_MeV": rows[0]["p_bin_edges_MeV"], "scope": rows[0]["scope"],
           "source_audits": [fingerprint(p) for p in tables.expand(paths)],
           "sectors": {s: {arm: merge_counts([r["sectors"][s][arm] for r in rows])
                            for arm in rows[0]["sectors"][s]} for s in rows[0]["sectors"]},
           "inventory_totals": {arm: {tree: {key: sum(r["inputs"][arm]["trees"][tree][key] for r in rows)
                                             for key in ("rows", "unique_ids")}
                                      for tree in rows[0]["inputs"][arm]["trees"]} for arm in rows[0]["inputs"]}}
    with Path(output).open("x") as stream:
        stream.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print_comparisons(out["sectors"])
    print("inventory_totals", json.dumps(out["inventory_totals"]))
    print("persisted_generated_events", out["persisted_generated_events"], "shards", len(rows))
    return out


def show_summary(path):
    data = json.loads(Path(path).read_text())
    print(data["run_config"], "events", data["persisted_generated_events"], "(truth diagnostics)")
    for name, values in data["sectors"].items():
        if not name.startswith("group:"):
            continue
        print(name)
        for arm, v in values.items():
            print(arm, "count", v["a"], v["b"], "shift", v["relative"], "+/-", v["se"],
                  "length", v["length_mean_relative"], "+/-", v.get("length_mean_se"),
                  "energy", v["edep_mean_relative"], "+/-", v.get("edep_mean_se"))
            if "matched_only" in v:
                m = v["matched_only"]
                print("  matched", v["identity_matched"], "a-only/b-only", v["a_only"]["count"], v["b_only"]["count"],
                      "length", m["length_mean_relative"], "+/-", m["length_mean_se"],
                      "energy", m["edep_mean_relative"], "+/-", m["edep_mean_se"],
                      "edep/path", m["energy_per_path_relative"], "+/-", m["energy_per_path_se"])
            for i, b in enumerate(v["entry_p_bins"][:2]):
                print("  p", P_EDGES[i], P_EDGES[i + 1], "counts", b["a"], b["b"],
                      "shift", b["relative"], "+/-", b["se"])


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--summary" in argv:
        p = argparse.ArgumentParser(description="Summarize an existing transport audit without rereading ROOT")
        p.add_argument("--summary", required=True)
        show_summary(p.parse_args(argv).summary)
        return
    if "--merge" in argv:
        p = argparse.ArgumentParser(description="Merge disjoint-shard transport diagnostic audits")
        p.add_argument("--merge", nargs="+", required=True)
        p.add_argument("--output", required=True)
        a = p.parse_args(argv)
        merge(a.merge, a.output)
        return
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("gen", "reference", "candidate", "config", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--baseline", help="Optional pre-fix fast baseline; omitted for candidate/reference-only audits")
    p.add_argument("--expected-events", type=int, required=True)
    p.add_argument("--expected-accepted", type=int,
                   help="External successful-worker accepted-event count; must match unique reference particle events")
    p.add_argument("--gdml")
    p.add_argument("--artifact", action="append", default=[])
    a = p.parse_args(argv)
    run(a.gen, a.reference, a.baseline, a.candidate, a.config, a.expected_events, a.output, a.gdml, a.artifact,
        expected_accepted=a.expected_accepted)


if __name__ == "__main__":
    main()
