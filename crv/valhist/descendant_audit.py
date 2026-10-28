"""Diagnose crossing-conditioned descendant support without changing the sampler.

Truth-ancestry diagnostics only; full integer event/particle keys define membership.
Event-clustered Poisson-bootstrap moments condition on the retained library and inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, tier1b
from crvfastsim.analyses.census import classify
from . import steplayers as S

KINDS = ("own_bar", "own_other_bar", "no_own_crossing_other_muon", "no_event_crossing")
VIEWS = {"sector": ["grp", "support"], "layer": ["grp", "lay", "support"]}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def annotate(steps, parts, geom):
    """Partition descendants by whether their true ancestor crosses their bar or any bar."""
    pk = tables.KEY + ["key"]
    if parts.duplicated(pk).any():
        raise ValueError("duplicate particle identities")
    for frame, cols in ((parts, pk + ["parent"]), (steps, tables.KEY + ["bar", "sp"])):
        values = frame[cols].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("non-integer or non-finite identity")
    keys = pd.MultiIndex.from_frame(parts[pk])
    parents = parts.loc[parts.parent >= 0, tables.KEY + ["parent"]]
    if not pd.MultiIndex.from_frame(parents).isin(keys).all():
        raise ValueError("missing parent identity")
    if not pd.MultiIndex.from_frame(steps[tables.KEY + ["sp"]]).isin(keys).all():
        raise ValueError("missing step particle identity")
    if ((steps.bar < 0) | (steps.bar >= geom.nbars)).any():
        raise ValueError("bar outside geometry")
    if not np.isfinite(steps.edep).all() or (steps.edep < 0).any():
        raise ValueError("invalid visible energy")
    s = classify(steps, parts, geom.sector)
    if s.cls.eq("unknown").any():
        raise ValueError("unresolved or cyclic step ancestry")
    x = s.loc[s.cls.eq("mu_primary"), tables.KEY + ["prim_key", "bar"]].drop_duplicates()
    d = s.loc[s.cls.eq("mu_descendant")].copy()
    ownbar = pd.MultiIndex.from_frame(d[tables.KEY + ["prim_key", "bar"]]).isin(
        pd.MultiIndex.from_frame(x[tables.KEY + ["prim_key", "bar"]]))
    own = pd.MultiIndex.from_frame(d[tables.KEY + ["prim_key"]]).isin(
        pd.MultiIndex.from_frame(x[tables.KEY + ["prim_key"]]))
    event = pd.MultiIndex.from_frame(d[tables.KEY]).isin(pd.MultiIndex.from_frame(x[tables.KEY]))
    d["support"] = np.select([ownbar, own, event], KINDS[:3], default=KINDS[3])
    d["grp"] = d["group"]
    d["lay"] = tier1b.layer_rank(geom)[d.bar.to_numpy(dtype=np.int64)]
    d["w"] = d.edep.to_numpy(dtype=np.float64)
    crossed = x.groupby(tables.KEY + ["bar"], sort=False).prim_key.nunique()
    audit = {"steps": len(s), "descendant_steps": len(d), "primary_crossings_by_ancestor": len(x),
             "event_bar_crossings": len(crossed), "multi_muon_bars": int((crossed > 1).sum()),
             "collapsed_crossings": int((crossed - 1).sum())}
    return d, audit


def paired_views(reference, candidate):
    """Accumulate exact paired event second moments without packed event identifiers."""
    ids = pd.concat([reference[tables.KEY], candidate[tables.KEY]], ignore_index=True)
    ev = ids.groupby(tables.KEY, sort=False).ngroup().to_numpy()
    r, c = reference.assign(ev=ev[:len(reference)]), candidate.assign(ev=ev[len(reference):])
    return {name: S.accumulate(r, c, cols) for name, cols in VIEWS.items()}


def summarize(views):
    out = {}
    for name, frame in views.items():
        f = frame.copy()
        for quantity in ("n", "e"):
            r, c = f[f"{quantity}_ref"], f[f"{quantity}_cand"]
            r2, c2 = f[f"{quantity}2_ref"], f[f"{quantity}2_cand"]
            diff2 = f[f"d{quantity}2"]
            ratio = np.divide(c, r, out=np.full(len(r), np.nan), where=r.to_numpy() > 0)
            residual = c2 + ratio ** 2 * r2 - ratio * (c2 + r2 - diff2)
            f[f"{quantity}_ratio"] = ratio
            f[f"{quantity}_ratio_se"] = np.sqrt(np.maximum(residual, 0)) / r.where(r > 0)
            f[f"{quantity}_delta"] = c - r
            f[f"{quantity}_delta_se"] = np.sqrt(diff2)
        out[name] = json.loads(f.to_json(orient="records"))
    return out


def run(reference_paths, candidate_paths, out, counters=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if candidate_paths is not None and len(reference_paths) != len(candidate_paths):
        raise ValueError("reference/candidate file counts differ")
    if len(set(map(str, reference_paths))) != len(reference_paths):
        raise ValueError("repeated reference path")
    counter_path = geometry.find_counters(counters or S.COUNTERS)
    geom = geometry.from_offline_config(counter_path)
    acc, inventory, seen = {}, [], set()
    started = time.monotonic()
    for i, ref in enumerate(reference_paths):
        tick = time.monotonic()
        rsteps, rparts = S._read(ref)
        r, ra = annotate(rsteps, rparts, geom)
        cand = None if candidate_paths is None else candidate_paths[i]
        if cand is None:
            c, ca = r.iloc[:0].copy(), None
        else:
            csteps, cparts = S._read(cand)
            c, ca = annotate(csteps, cparts, geom)
        subruns = pd.concat([r[tables.KEY[:2]], c[tables.KEY[:2]]]).drop_duplicates()
        current = set(subruns.itertuples(index=False, name=None))
        if seen.intersection(current):
            raise ValueError("overlapping run/subrun across input pairs")
        seen.update(current)
        acc = S._merge_views(acc, paired_views(r, c))
        inventory.append({"reference": str(ref), "reference_sha256": digest(ref), "reference_audit": ra,
                          "candidate": str(cand) if cand is not None else None,
                          "candidate_sha256": digest(cand) if cand is not None else None, "candidate_audit": ca,
                          "seconds": time.monotonic() - tick})
        print(f"{Path(ref).name}: {ra}; {inventory[-1]['seconds']:.2f} s", flush=True)
    result = {"schema": 1, "diagnostic": "truth-ancestor crossing support, not an efficiency selection",
              "uncertainty": "paired event Poisson-bootstrap; no library or denominator uncertainty",
              "candidate_present": candidate_paths is not None, "inputs": inventory,
              "geometry": str(counter_path), "geometry_sha256": digest(counter_path),
              "source_sha256": digest(__file__), "seconds": time.monotonic() - started,
              "views": summarize(acc)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"wrote {out}; {result['seconds']:.2f} s", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reference", help="reference path pattern, {s}=001400_NNNNNNNN")
    inputs.add_argument("--donor-manifest", help="library JSON with immutable inputs list")
    p.add_argument("--candidate", help="candidate path pattern, paired with reference")
    p.add_argument("--shards", default="11-20")
    p.add_argument("--counters")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.donor_manifest:
        if a.candidate:
            p.error("donor manifest is a reference-only census")
        refs, cands = json.loads(Path(a.donor_manifest).read_text())["inputs"], None
    else:
        shards = S.shards(a.shards)
        refs = [a.reference.format(s=S.shard(k), k=k) for k in shards]
        cands = [a.candidate.format(s=S.shard(k), k=k) for k in shards] if a.candidate else None
    run(refs, cands, a.out, a.counters)


if __name__ == "__main__":
    main()
