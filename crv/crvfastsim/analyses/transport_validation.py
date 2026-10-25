"""Audit persisted transport tables and compare all sectors with event-paired counting errors.

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


def compare(a, b, universe):
    out = paired_count(a, b, universe)
    out["sums"] = {col: {"a": float(a[col].sum()), "b": float(b[col].sum())} for col in ("length", "edep")}
    for col in ("length", "edep"):
        ma, mb = a[col].mean(), b[col].mean()
        out[col + "_mean_relative"] = float(ma / mb - 1) if len(a) and len(b) and mb else None
    out["entry_p_bins"] = [paired_count(a[(a.p >= lo) & (a.p < hi)],
                                          b[(b.p >= lo) & (b.p < hi)], universe)
                           for lo, hi in zip(P_EDGES[:-1], P_EDGES[1:])]
    join = tables.KEY + ["genidx", "bar"]
    matched = a.merge(b, on=join, suffixes=("_a", "_b"), validate="one_to_one")
    out["identity_matched"] = len(matched)
    out["entry_p_difference_MeV_quantiles"] = (np.quantile(matched.p_a - matched.p_b, [.1, .5, .9]).tolist()
                                               if len(matched) else None)
    return out


def repo_state(module):
    root = Path(module.__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    tracked = ("transport.py", "tables.py")
    hashes = {name: fingerprint(Path(module.__file__).parent / name)["sha256"] for name in tracked}
    return {"root": str(root), "head": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "source_sha256": hashes}


def run(gen_path, reference, baseline, candidate, config, expected_events, output, gdml=None, artifacts=()):
    rc = runconfig.get(config)
    geom = rc.geometry()
    gen = tables.read(gen_path, "gen")
    universe = gen[tables.KEY].drop_duplicates()
    if len(universe) != expected_events or gen.duplicated(tables.KEY + ["idx"]).any():
        raise ValueError("generated event count or generator identity mismatch")
    paths = {"gen": gen_path, "reference": reference, "baseline": baseline, "candidate": candidate}
    inventories = {k: audit(p) for k, p in paths.items()}
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
    result = {}
    for name, sectors in groups.items():
        selected = {k: frame[frame.sector.isin(sectors)] for k, frame in x.items()}
        result[name] = {label: compare(selected[a], selected[b], universe) for label, a, b in
                       (("baseline_vs_reference", "baseline", "reference"),
                        ("candidate_vs_reference", "candidate", "reference"),
                        ("candidate_vs_baseline", "candidate", "baseline"))}
    event_ranges = universe.groupby(["run", "subrun"]).event.agg(["min", "max", "size"]).reset_index().to_dict("records")
    out = {"run_config": config, "expected_events": expected_events, "persisted_generated_events": len(universe),
           "event_ranges": event_ranges, "diagnostic_sha256": fingerprint(__file__)["sha256"],
           "reference_completeness": "Persisted accepted events counted; completeness requires external job evidence.",
           "inputs": inventories, "code": {"crv": repo_state(runconfig), "core": repo_state(eloss)},
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "uproot": uproot.__version__},
           "geometry": [fingerprint(geometry.find_counters(rc.counters))] + ([fingerprint(gdml)] if gdml else []),
           "artifacts": [fingerprint(p) for p in artifacts],
           "p_bin_edges_MeV": [0, 300, 600, 1000, 2000, 5000, None],
           "scope": "Truth diagnostic; event-cluster paired SE, independent per-arm entry-p bins, no G4/BO certification.",
           "crossing_key": tables.KEY + ["genidx", "bar"], "sectors": result}
    with Path(output).open("x") as stream:
        stream.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"{config}: {len(universe)} persisted generated events; output {output}")
    for label, value in inventories.items():
        print(label, json.dumps(value["trees"]))
    for name, res in result.items():
        c, d = res["candidate_vs_reference"], res["candidate_vs_baseline"]
        print(name, "candidate/ref", c["a"], c["b"], c["relative"], "+/-", c["se"],
              "candidate/base", d["relative"], "+/-", d["se"])
    return out


def merge_counts(rows):
    moments = {k: sum(r["moments"][k] for r in rows) for k in ("events", "a2", "b2", "ab")}
    out = count_summary(sum(r["a"] for r in rows), sum(r["b"] for r in rows), moments)
    if "sums" in rows[0]:
        out["sums"] = {col: {arm: sum(r["sums"][col][arm] for r in rows) for arm in ("a", "b")}
                       for col in ("length", "edep")}
        for col, sums in out["sums"].items():
            out[col + "_mean_relative"] = (sums["a"] / out["a"] / (sums["b"] / out["b"]) - 1
                                           if out["a"] and out["b"] and sums["b"] else None)
        out["identity_matched"] = sum(r["identity_matched"] for r in rows)
        out["entry_p_bins"] = [merge_counts([r["entry_p_bins"][i] for r in rows]) for i in range(len(P_EDGES) - 1)]
    return out


def merge(paths, output):
    rows = [json.loads(Path(p).read_text()) for p in tables.expand(paths)]
    if not rows or len({r["run_config"] for r in rows}) != 1:
        raise ValueError("merge requires one nonempty run configuration")
    if len({r["diagnostic_sha256"] for r in rows}) != 1 or any(
            r["p_bin_edges_MeV"] != rows[0]["p_bin_edges_MeV"] for r in rows):
        raise ValueError("incompatible diagnostic implementation or momentum bins")
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
    out = {"run_config": rows[0]["run_config"], "shards": len(rows),
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
    for sector, values in out["sectors"].items():
        c, d = values["candidate_vs_reference"], values["candidate_vs_baseline"]
        print(sector, "candidate/ref", c["relative"], "+/-", c["se"], "candidate/base", d["relative"], "+/-", d["se"])
    print("inventory_totals", json.dumps(out["inventory_totals"]))
    print("persisted_generated_events", out["persisted_generated_events"], "shards", len(rows))
    return out


def show_summary(path):
    data = json.loads(Path(path).read_text())
    print(data["run_config"], "events", data["persisted_generated_events"])
    for name, values in data["sectors"].items():
        if not name.startswith("group:"):
            continue
        print(name)
        for arm in ("baseline_vs_reference", "candidate_vs_reference", "candidate_vs_baseline"):
            v = values[arm]
            print(arm, "count", v["a"], v["b"], "shift", v["relative"], "+/-", v["se"],
                  "length", v["length_mean_relative"], "energy", v["edep_mean_relative"])
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
    for flag in ("gen", "reference", "baseline", "candidate", "config", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--expected-events", type=int, required=True)
    p.add_argument("--gdml")
    p.add_argument("--artifact", action="append", default=[])
    a = p.parse_args(argv)
    run(a.gen, a.reference, a.baseline, a.candidate, a.config, a.expected_events, a.output, a.gdml, a.artifact)


if __name__ == "__main__":
    main()
