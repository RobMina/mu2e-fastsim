"""Deck arms re-made on a new shard set: merge the per-shard art histograms, then compare with the deck statistic.

Arms (one art file per shard under DQM/<arm>/valhist.<shard>.root, each with its crvdump tables): a0 / a0p (G4 steps,
stock response and reco, two digitisation seeds: the floor pair), a0g (G4 steps injected on the gen stream), and
fast-step arms a1_<set> (fast steps, stock response and reco) and e2e_<set> (fast steps, response and reco) for each
tier-1b set. The comparison runs `fsdeck.crv.compare_arms` (deck statistic: first-bin cut, clustered time errors,
exact per-bin position variance refilled from each arm's own tables, floor a0 vs a0p, flag rule) read-only, with the
arm paths and tables pointed at this shard set. As in the deck, a0g and a0p borrow a0's cluster position variance.

    python -m valhist.deck_rerun merge --dqm DIR --shards 22,24 --tag v2even22-40 --outdir MERGED
    python -m valhist.deck_rerun compare --dqm DIR --merged MERGED --tag v2even22-40 --shards 22,24 \
        --steps a0g='G4/crvsteps.{s}.root' --steps new='T/fast1b.{s}.root' --sets new,legacy --out NEW.json
"""
import argparse
import json
from pathlib import Path

import uproot

from . import compare as VC
from .descendant_audit import digest
from .steplayers import shards as parse_shards

A1_ALIAS = ["CrvSteps_fastCrvSteps_noName=CrvSteps_steps_noName",
            "SimParticlemv_fastCrvSteps_noName=SimParticlemv_steps_noName"]
G4_ARMS = ("a0", "a0p", "a0g")


def shard(k):
    return f"001400_{k:08d}"


def arms_of(sets):
    return list(G4_ARMS) + [f"{p}_{s}" for s in sets for p in ("a1", "e2e")]


def merge(dqm, shards, tag, outdir, sets):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for a in arms_of(sets):
        files = [str(Path(dqm) / a / f"valhist.{shard(k)}.root") for k in shards]
        missing = [f for f in files if not Path(f).is_file()]
        if missing:
            raise SystemExit(f"{a}: missing {len(missing)} shard files, e.g. {missing[0]}")
        dest = out / f"{a}.{tag}.root"
        if dest.exists():
            raise FileExistsError(dest)
        VC.merge(str(dest), files, A1_ALIAS if a == "a0g" or a.startswith("a1_") else ())
        made[a] = {"path": str(dest), "sha256": digest(dest), "inputs": {f: digest(f) for f in files}}
        print(f"merged {a}: {dest}", flush=True)
    (out / f"merge.{tag}.json").write_text(json.dumps(made, indent=2) + "\n")
    return made


def comparisons(sets):
    pairs = [("floor_a0p_vs_a0", "a0", "a0p", None)]
    for s in sets:
        pairs += [(f"a1_{s}_vs_a0g", "a0g", f"a1_{s}", ("a0", "a0p")),
                  (f"e2e_{s}_vs_a0g", "a0g", f"e2e_{s}", ("a0", "a0p")),
                  (f"e2e_{s}_vs_a1_{s}", f"a1_{s}", f"e2e_{s}", ("a0", "a0p"))]
    return pairs


def compare(dqm, merged, tag, shards, steps, sets, out):
    from fsdeck import crv as FC
    from fsdeck import crv_pos as CP
    from fsdeck.common import _clean
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    hist = {a: Path(merged) / f"{a}.{tag}.root" for a in arms_of(sets)}
    table = lambda a: (lambda k, a=a: Path(dqm) / a / f"valhist.{shard(k)}.root")
    step_pat = lambda name: (lambda k, p=steps[name]: Path(p.format(s=shard(k))))
    saved = FC.arm, CP.TABLES, CP.PROXY, CP.factors
    try:
        FC.arm = lambda name: hist[name]
        CP.TABLES = {("clusters", a): (shards, table(a), "crvdump/clusters") for a in arms_of(sets) if a != "a0g"}
        CP.TABLES.update({("steps", n): (shards, step_pat(n), "dump/steps") for n in steps})
        CP.PROXY = {"a0": {"steps": ("a0g", shards, True), "clusters": ("a0", shards, True)},
                    "a0p": {"steps": ("a0g", shards, True), "clusters": ("a0", shards, False)},
                    "a0g": {"steps": ("a0g", shards, True), "clusters": ("a0", shards, False)}}
        for s in sets:
            for p in ("a1", "e2e"):
                CP.PROXY[f"{p}_{s}"] = {"steps": (s, shards, True), "clusters": (f"{p}_{s}", shards, True)}
        variance = out.with_suffix(".position_variance.json")
        CP.extract(variance)
        pos = saved[3](variance)
        CP.factors = lambda *a, **kw: pos
        res = {}
        for name, ref, cand, floor in comparisons(sets):
            per, kf = FC.compare_arms(VC, ref, cand, floor)
            flagged = sorted(k for k, v in per.items() if v["flag"])
            with uproot.open(hist[ref]) as r, uproot.open(hist[cand]) as c:
                flows = {k: {"ref": VC.flow(r, k), "cand": VC.flow(c, k)} for k in per
                         if per[k].get("status") != "binning_differs"}
            res[name] = {"ref": ref, "cand": cand, "floor": floor, "n_compared": len(per), "n_flagged": len(flagged),
                         "flagged": flagged, "per_hist": per, "time_factors": kf, "flows": flows,
                         "totals": FC._totals(ref, cand)}
            print(f"{name}: {len(flagged)}/{len(per)} flagged", flush=True)
        result = {"shards": shards, "tag": tag, "histograms": {k: str(v) for k, v in hist.items()},
                  "hashes": {str(p): digest(p) for p in hist.values()},
                  "code": {str(p): digest(p) for p in (Path(FC.__file__), Path(CP.__file__), Path(__file__))},
                  "comparisons": res,
                  "limitations": "a0g/a0p cluster position variance borrowed from a0 (as the deck); flags are "
                                 "descriptive, not calibrated p-values; library conditional"}
        out.write_text(json.dumps(_clean(result), indent=2) + "\n")
        return result
    finally:
        FC.arm, CP.TABLES, CP.PROXY, CP.factors = saved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("merge", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--dqm", required=True)
        s.add_argument("--shards", required=True)
        s.add_argument("--tag", required=True)
        s.add_argument("--sets", default="new,legacy", help="comma-separated tier-1b arm suffixes")
    sub.choices["merge"].add_argument("--outdir", required=True)
    c = sub.choices["compare"]
    c.add_argument("--merged", required=True)
    c.add_argument("--steps", action="append", required=True, metavar="NAME=PATTERN",
                   help="step tables per shard ({s}); a0g plus one per set")
    c.add_argument("--out", required=True)
    a = p.parse_args()
    shards, sets = parse_shards(a.shards), [s for s in a.sets.split(",") if s]
    if a.cmd == "merge":
        merge(a.dqm, shards, a.tag, a.outdir, sets)
    else:
        steps = dict(x.split("=", 1) for x in a.steps)
        if set(steps) != {"a0g", *sets}:
            raise SystemExit("--steps needs a0g and one pattern per set")
        compare(a.dqm, a.merged, a.tag, shards, steps, sets, a.out)


if __name__ == "__main__":
    main()
