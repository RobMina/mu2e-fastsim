"""The validation deck's histogram flag rule applied to ValCrvStep refills (`valhist.stephist` accumulators).

The deck compares two arms per histogram with chi2/ndf over the booked bins with entries and flags a histogram when
chi2/ndf > max(2, 3 x floor), where the floor is the same statistic for a pair that differs only by its random seed
(threshold 2 without a floor). The variance per bin depends on the histogram:

  t, t2, lt        k1 n1 + k2 n2, k = <m^2>/<m> of the per-event step multiplicity m (events with m > 0), taken from
                   the arm's NStep histogram at its bin centres, as the deck does
  posx, y/z keys   the exact event-clustered variance w2_1 + w2_2
  everything else  Poisson n1 + n2

The lowest bin of E, SMom and SMom2 is left out. `stephist.compare` keeps its own "deck" field (Poisson for every
non-y/z key); this module reproduces the deck's variance choices so a step-level comparison can be read against the
deck's threshold. It reproduces the deck's published fast-steps vs Geant4 statistics to 1e-9 when given the deck's
merged ValCrvStep histograms (`read_valcrvstep`); the unit tests use synthetic accumulators.

    python -m valhist.deck_rule SUMMARY.json [...] [--ref numpy --cand jax] [--floor-ref numpy --floor-cand numpy_reseed
        [--floor-summary FLOOR.json ...]] [--out OUT.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import stephist as H

CLUSTERED_T = ("t", "t2", "lt")
EXACT = ("posx",)


def clustering_factor(nstep):
    """<m^2>/<m> over events with m > 0 from an NStep accumulator (flow-inclusive counts), at the bin centres."""
    nb, lo, hi = H.BOOK["NStep"]
    edges = np.linspace(lo, hi, nb + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    v = np.asarray(nstep, float)[1:-1]
    m = c > 0.5
    return float((v[m] * c[m] ** 2).sum() / max((v[m] * c[m]).sum(), 1e-300))


def variance_kind(key):
    if key in CLUSTERED_T:
        return "clustered_k"
    if key in EXACT or "/" in key:
        return "exact"
    return "poisson"


def statistic(ref, cand, key):
    """(chi2/ndf, ndf, variance kind, (k_ref, k_cand)) of one histogram key for two accumulators."""
    (n1, w1), (n2, w2) = ref[key]["all"], cand[key]["all"]
    kind = variance_kind(key)
    k = (1.0, 1.0)
    if kind == "clustered_k":
        k = (clustering_factor(ref["NStep"]["all"][0]), clustering_factor(cand["NStep"]["all"][0]))
        var = k[0] * np.asarray(n1, float) + k[1] * np.asarray(n2, float)
    elif kind == "exact":
        var = np.asarray(w1, float) + np.asarray(w2, float)
    else:
        var = np.asarray(n1, float) + np.asarray(n2, float)
    value, ndf = H.chi2(n1, n2, var, key in H.FIRST_BIN_CUT)
    return value, ndf, kind, k


def compare(ref, cand, floor_ref=None, floor_cand=None):
    """Per histogram key (yields excluded): the deck statistic, its floor and the flag."""
    out = {}
    for key in ref:
        if key.startswith("_total") or key not in cand:
            continue
        value, ndf, kind, k = statistic(ref, cand, key)
        r = {"chi2ndf": value, "ndf": ndf, "variance": kind}
        if kind == "clustered_k":
            r["k"] = list(k)
        floor = None
        if floor_ref is not None and floor_cand is not None and key in floor_ref and key in floor_cand:
            floor = statistic(floor_ref, floor_cand, key)[0]
        r["floor_chi2ndf"] = floor
        r["threshold"] = max(2.0, 3.0 * floor) if floor is not None else 2.0
        r["flag"] = value is not None and value > r["threshold"]
        out[key] = r
    return out


def read_valcrvstep(path, directory="Validation/CrvSteps_steps_noName", keys=tuple(H.BOOK)):
    """Accumulators {key: {"all": (n, n)}} from a merged ValCrvStep ROOT file (flow-inclusive counts; no per-event
    second moments, so only the Poisson and clustering-factor keys are meaningful)."""
    import uproot
    out = {}
    with uproot.open(path) as f:
        for k in keys:
            n = f[f"{directory}/{k}"].values(flow=True).astype(float)
            out[k] = {"all": (n, n.copy())}
    return out


def from_summaries(paths, arms):
    """Per arm, the stephist accumulators summed over the summary JSONs (disjoint shard sets)."""
    out = {a: {} for a in arms}
    shards = []
    for path in paths:
        d = json.loads(Path(path).read_text())
        shards += list(d.get("shards") or [r["shard"] for r in d.get("records", [])])
        for a in arms:
            H.add(out[a], H.from_json(d["hists"][a]))
    if len(set(shards)) != len(shards):
        raise ValueError("summaries overlap in shards")
    return out, sorted(shards)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary", nargs="+", help="JSONs with hists.<arm> stephist accumulators (disjoint shards)")
    p.add_argument("--ref", default="numpy")
    p.add_argument("--cand", default="jax")
    p.add_argument("--floor-ref")
    p.add_argument("--floor-cand")
    p.add_argument("--floor-summary", nargs="+", help="read the floor arms from these JSONs instead of SUMMARY")
    p.add_argument("--out")
    a = p.parse_args()
    arms, shards = from_summaries(a.summary, (a.ref, a.cand))
    floor, floor_shards = (None, None), None
    if a.floor_ref and a.floor_cand:
        f, floor_shards = from_summaries(a.floor_summary or a.summary, (a.floor_ref, a.floor_cand))
        floor = (f[a.floor_ref], f[a.floor_cand])
        if floor_shards != shards:
            raise ValueError("floor and comparison cover different shards")
    res = compare(arms[a.ref], arms[a.cand], *floor)
    result = {"summaries": a.summary, "floor_summaries": a.floor_summary, "shards": shards, "ref": a.ref,
              "cand": a.cand, "floor": [a.floor_ref, a.floor_cand],
              "flagged": sorted(k for k, v in res.items() if v["flag"]), "per_hist": res}
    if a.floor_summary and a.floor_ref == a.ref:
        result["ref_identical_in_floor"] = all(np.array_equal(arms[a.ref][k]["all"][0], floor[0][k]["all"][0])
                                               for k in arms[a.ref])
    text = json.dumps(result, indent=1, allow_nan=False)
    if a.out:
        if Path(a.out).exists():
            raise FileExistsError(a.out)
        Path(a.out).write_text(text)
    for k, v in res.items():
        print(f"{k:10s} {v['variance']:12s} chi2/ndf {v['chi2ndf'] if v['chi2ndf'] is None else round(v['chi2ndf'], 4)}"
              f"  floor {v['floor_chi2ndf']}  threshold {v['threshold']:.3f}  {'FLAG' if v['flag'] else ''}")
    print("flagged:", result["flagged"], "ref identical in floor:", result.get("ref_identical_in_floor"))


if __name__ == "__main__":
    main()
