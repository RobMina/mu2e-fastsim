"""Compare a completed retained-step pilot with frozen references using the presentation statistic.

Requires the optional fsdeck package on PYTHONPATH, read-only. Output stays under a
new directory; original shard histograms supply flows and position-table variances.
"""
import argparse
import json
from pathlib import Path

import uproot

from . import compare as VC
from . import paired_rates
from .descendant_audit import digest


def paths(data, new, shard):
    data, new = Path(data), Path(new)
    s = f"001400_{shard:08d}"
    v = data / "crv_fastsim/valhist"
    g = v / "g4ref_v1"
    old = v / f"thread_b_expand_20261025_s{shard:08d}"
    hist = {name: g / name / f"valhist.{s}.root" for name in ("a0", "a0p", "a0g")}
    hist.update(old_a1=old / "a1/valhist.root", a1=new / "a1/valhist.root", e2e=new / "e2e/valhist.root",
                old_e2e=data / "crv_oppside_20261026/valhist/e2e_on" / f"valhist.{s}.root")
    tables = {"a0": g / "a0_dump" / f"valhist.{s}.root", "old_a1": hist["old_a1"],
              "a1": hist["a1"], "e2e": hist["e2e"],
              "old_e2e": data / "crv_oppside_20261026/tables/e2e_on" / f"reco.{s}.root"}
    return hist, tables, g / "g4all_tables" / f"g4all.{s}.root", old / "tables" / f"fast1b.{s}.root"


def run(data, new, shard, outdir, control=()):
    from fsdeck import crv as FC
    from fsdeck import crv_pos as CP
    from fsdeck.common import _clean
    out = Path(outdir)
    if out.exists():
        raise FileExistsError(out)
    new = Path(new)
    completion = json.loads((new / "completion.json").read_text())
    if not completion.get("completed"):
        raise ValueError("pilot is not complete")
    manifest = json.loads((new / "manifest.json").read_text())
    hist, tables, g4steps, oldsteps = paths(data, new, shard)
    from .descendant_inputs import bind_pilot
    bind_pilot(manifest, oldsteps)
    for arm in ("a1", "e2e"):
        if digest(hist[arm]) != completion["arms"][arm]["sha256"]:
            raise ValueError("completed histogram payload changed")
    controls = {}
    for item in control:
        name, sep, directory = item.partition("=")
        if (not sep or not name.isidentifier() or name in hist or name in controls or
                any(name + "_" + arm in hist for arm in ("a1", "e2e"))):
            raise ValueError("controls must have unique NAME=completed-directory labels")
        root = Path(directory)
        cm = json.loads((root / "manifest.json").read_text())
        done = json.loads((root / "completion.json").read_text())
        if not done.get("completed") or done["expected_generated_events"] != completion["expected_generated_events"]:
            raise ValueError("incomplete or different control population")
        for arm in ("a1", "e2e"):
            key = name + "_" + arm
            hist[key] = root / arm / "valhist.root"
            tables[key] = hist[key]
            if (digest(hist[key]) != done["arms"][arm]["sha256"] or
                    done["arms"][arm]["trees"]["events"]["event_keys_sha256"] !=
                    completion["arms"][arm]["trees"]["events"]["event_keys_sha256"]):
                raise ValueError("control hash or event identity mismatch")
        controls[name] = (root, Path(cm["steps"]))
    files = list(hist.values()) + list(tables.values()) + [g4steps, oldsteps, Path(manifest["steps"])]
    files += [p for root, step in controls.values() for p in (step, root / "manifest.json", root / "completion.json")]
    hashes = {str(p): digest(p) for p in files}
    hashes.update({str(p): digest(p) for p in (Path(FC.__file__), Path(CP.__file__), Path(__file__))})
    out.mkdir(parents=True)
    saved = FC.arm, CP.TABLES, CP.PROXY, CP.factors
    try:
        FC.arm = lambda name: hist[name]
        CP.TABLES = {("clusters", name): ([shard], lambda k, p=p: p, "crvdump/clusters")
                     for name, p in tables.items()}
        for name, p in (("g4", g4steps), ("old", oldsteps), ("new", Path(manifest["steps"]))):
            CP.TABLES[("steps", name)] = ([shard], lambda k, p=p: p, "dump/steps")
        CP.PROXY = {name: {"steps": ("new" if name in ("a1", "e2e") else
                                             "old" if name.startswith("old_") else "g4", [shard], True),
                          "clusters": (name if name in tables else "a0", [shard], name in tables)}
                    for name in hist}
        for name, (root, step) in controls.items():
            CP.TABLES[("steps", name)] = ([shard], lambda k, p=step: p, "dump/steps")
            for arm in ("a1", "e2e"):
                key = name + "_" + arm
                CP.PROXY[key] = {"steps": (name, [shard], True), "clusters": (key, [shard], True)}
        variance = out / "position_variance.json"
        CP.extract(variance)
        pos = saved[3](variance)
        CP.factors = lambda *args, **kwargs: pos
        comparisons = {}
        pairs = [("old_vs_a0g", "a0g", "old_e2e"), ("new_vs_a0g", "a0g", "e2e"),
                 ("old_vs_a1", "old_a1", "old_e2e"), ("new_vs_a1", "a1", "e2e")]
        pairs += [(name + "_vs_a0g", "a0g", name + "_e2e") for name in controls]
        pairs += [(name + "_vs_a1", name + "_a1", name + "_e2e") for name in controls]
        for name, ref, cand in pairs:
            per, factors = FC.compare_arms(VC, ref, cand, ("a0", "a0p"))
            flagged = [k for k, v in per.items() if v["flag"]]
            with uproot.open(hist[ref]) as r, uproot.open(hist[cand]) as c:
                flows = {k: {"ref": VC.flow(r, k), "cand": VC.flow(c, k)} for k in per
                         if per[k].get("status") != "binning_differs"}
            comparisons[name] = {"ref": ref, "cand": cand, "n_compared": len(per), "n_flagged": len(flagged),
                                 "flagged": flagged, "per_hist": per, "time_factors": factors, "flows": flows}
            print(f"{name}: {len(flagged)}/{len(per)}", flush=True)
        paired_rates.run(str(hist["a1"]), str(hist["e2e"]), str(out / "paired_rates.json"))
        for name in controls:
            paired_rates.run(str(hist[name + "_a1"]), str(hist[name + "_e2e"]),
                             str(out / ("paired_" + name + ".json")))
        counts = {name: FC._totals("a0g", name) for name in ["old_e2e", "e2e"] + [n + "_e2e" for n in controls]}
        result = {"shards": [shard], "histograms": {k: str(v) for k, v in hist.items()}, "hashes": hashes,
                  "comparisons": comparisons, "counts_vs_a0g": counts,
                  "limitations": "One-shard pilot, fixed library; A0g/floor cluster position variance borrowed from A0. No donor covariance or full-sample promotion."}
        (out / "comparison.json").write_text(json.dumps(_clean(result), indent=2) + "\n")
        return result
    finally:
        FC.arm, CP.TABLES, CP.PROXY, CP.factors = saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "new", "outdir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--control", action="append", default=[])
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
