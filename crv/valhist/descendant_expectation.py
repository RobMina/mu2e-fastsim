"""Exact pre-placement other-bar descendant moments, conditioned on observed fast crossings.

Donor destination sector/layer is retained as a truth diagnostic; this is NOT the
expectation after geometry placement and is not a replacement simulation model.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

from crvfastsim import geometry, tables, tier1b as T
from crvfastsim.analyses.census import classify, SECTOR_GROUP
from . import descendant_audit as A
from . import steplayers as S
from .shower_moments import conditional_moments

GROUPS = T.GROUPS
DIM = 32


def cell_codes(lx):
    """Sampler's effective (bin,context) cells, with pooled-bin fallback codes."""
    if lx.xid.duplicated().any() or not np.array_equal(lx.xid, np.arange(len(lx))):
        raise ValueError("requires ordered contiguous library crossing IDs")
    if not lx.ctx.between(0, T.N_CTX - 1).all():
        raise ValueError("invalid crossing context")
    return lx.bin.to_numpy(dtype=np.int64) * (T.N_CTX + 1) + lx.ctx.to_numpy(dtype=np.int64)


def feature_matrix(lx, desc, donor_bars, geom):
    """One row per donor crossing, zero-deposit crossings included, 16 count + 16 energy cells."""
    if len(donor_bars) != len(lx):
        raise ValueError("donor crossing count mismatch")
    d = desc.loc[desc.samebar.eq(0)]
    ids = d.xid.to_numpy(dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= len(lx)):
        raise ValueError("unknown descendant crossing")
    centers = geom.center[np.asarray(donor_bars)[ids]] + d[["dcx", "dcy", "dcz"]].to_numpy()
    b = T.bar_containing(centers, geom)
    if (b < 0).any() or not np.allclose(centers, geom.center[np.maximum(b, 0)], rtol=0, atol=1e-7):
        raise ValueError("donor target is not an exact geometry bar center")
    gi = T._group_index(geom, b)
    if (gi < 0).any():
        raise ValueError("unsupported sector")
    category = gi * 4 + T.layer_rank(geom)[b]
    energy = d.edep.to_numpy(dtype=np.float64)
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("invalid descendant energy")
    return sparse.coo_matrix((np.r_[np.ones(len(d)), energy],
                              (np.tile(ids, 2), np.r_[category, category + 16])), shape=(len(lx), DIM)).tocsr()


def pool(lx, features):
    """Duplicate only the sparse features into a pooled bin cell used on sparse-context fallback."""
    code = cell_codes(lx)
    pooled = lx.bin.to_numpy(dtype=np.int64) * (T.N_CTX + 1) + T.N_CTX
    return sparse.vstack([features, features], format="csr"), np.r_[code, pooled]


def evaluation_codes(fx, lx, bins, geom, context):
    p = np.linalg.norm(fx[["px0", "py0", "pz0"]].to_numpy(), axis=1)
    b = T._bin(bins, T._group_index(geom, fx.bar), p, fx.len.to_numpy())
    ncell = len(GROUPS) * T.N_PBINS * T.N_LBINS * (T.N_CTX + 1)
    counts = np.bincount(cell_codes(lx), minlength=ncell)
    code = b * (T.N_CTX + 1) + T.crossing_context(fx, geom) if context else b * (T.N_CTX + 1) + T.N_CTX
    fallback = counts[code] < T.MIN_CTX_CROSSINGS if context else np.ones(len(fx), dtype=bool)
    code[fallback] = b[fallback] * (T.N_CTX + 1) + T.N_CTX
    return code, int(fallback.sum())


def reference_features(steps, parts, geom):
    d, audit = A.annotate(steps, parts, geom)
    d = d.loc[d.support.eq("own_other_bar")]
    idx = T._group_index(geom, d.bar) * 4 + d.lay.to_numpy()
    result = np.r_[np.bincount(idx, minlength=16), np.bincount(idx, weights=d.w, minlength=16)]
    return result, audit


def build_cache(manifest, out, counters=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tick = time.monotonic()
    m = json.loads(Path(manifest).read_text())
    library = str(Path(manifest).with_suffix(".root"))
    counter_path = geometry.find_counters(counters or S.COUNTERS)
    geom = geometry.from_offline_config(counter_path)
    lib = T.read_library(library)
    lx = lib["libxing"]
    hashes = {str(p): A.digest(p) for p in (manifest, library, counter_path, __file__, T.__file__)}
    bars, offset, donor_ids, records = [], 0, [], []
    for f in m["inputs"]:
        start = time.monotonic()
        steps, parts = S._read(f)
        tagged = classify(steps, parts, geom.sector)
        x = T._crossings(tagged.loc[tagged.cls.eq("mu_primary")])
        z = lx.iloc[offset:offset + len(x)]
        ctx = T.crossing_context(x, geom)
        b = T._bin(lib["libbins"], T._group_index(geom, x.bar), x.p.to_numpy(), x.len.to_numpy())
        np.testing.assert_array_equal(z.ctx, ctx)
        np.testing.assert_array_equal(z.bin, b)
        np.testing.assert_allclose(z[["ux", "uy", "uz"]], T._unit(x[["px0", "py0", "pz0"]]), rtol=1e-7, atol=1e-12)
        bars.append(x.bar.to_numpy(dtype=np.int64))
        donor_ids.append(parts[tables.KEY].drop_duplicates())
        hashes[str(f)] = A.digest(f)
        records.append({"path": f, "crossings": len(x), "seconds": time.monotonic() - start})
        offset += len(x)
        print(f"{Path(f).name}: {len(x)} exact crossing bins/contexts; {records[-1]['seconds']:.2f} s", flush=True)
    if offset != len(lx):
        raise ValueError("library crossing count differs from donor reconstruction")
    features = feature_matrix(lx, lib["libdesc"], np.concatenate(bars), geom)
    out.mkdir(parents=True)
    sparse.save_npz(out / "features.npz", features)
    lx.to_csv(out / "crossings.csv", index=False)
    lib["libbins"].to_csv(out / "bins.csv", index=False)
    pd.concat(donor_ids).drop_duplicates().to_csv(out / "donor_events.csv", index=False)
    hashes.update({str(out / p): A.digest(out / p) for p in ("features.npz", "crossings.csv", "bins.csv", "donor_events.csv")})
    result = {"library": library, "geometry": str(counter_path), "hashes": hashes, "donors": records,
              "seconds": time.monotonic() - tick, "meaning": "pre-placement truth-destination other-bar features"}
    (out / "cache.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out}; {result['seconds']:.2f} s", flush=True)


def evaluate(cache, transport, reference, shards, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tick = time.monotonic()
    root = Path(cache)
    meta = json.loads((root / "cache.json").read_text())
    for p, sha in meta["hashes"].items():
        if A.digest(p) != sha:
            raise ValueError(f"cache/input drift: {p}")
    geom = geometry.from_offline_config(meta["geometry"])
    lx, bins = pd.read_csv(root / "crossings.csv"), pd.read_csv(root / "bins.csv")
    donor_ids = pd.MultiIndex.from_frame(pd.read_csv(root / "donor_events.csv"))
    x, cell = pool(lx, sparse.load_npz(root / "features.npz"))
    ncell = len(GROUPS) * T.N_PBINS * T.N_LBINS * (T.N_CTX + 1)
    n = np.bincount(cell, minlength=ncell)
    totals = {a: np.zeros(ncell, dtype=np.int64) for a in ("default", "context")}
    ref, refcov = np.zeros(DIM), np.zeros((DIM, DIM))
    from .descendant_inputs import population, unique_shards
    records, files, seen = [], {}, set()
    for k in unique_shards(shards):
        f, r = (p.format(s=S.shard(k), k=k) for p in (transport, reference))
        fx = tables.read(f, "steps")
        parts = tables.read(f, "gen", tables.KEY).drop_duplicates()
        st, pa = S._read(r)
        population(parts, fx, pa, donor_ids, seen)
        v, audit = reference_features(st, pa, geom)
        d, _ = A.annotate(st, pa, geom)
        d = d.loc[d.support.eq("own_other_bar")].copy()
        d["feature"] = T._group_index(geom, d.bar) * 4 + d.lay
        g = d.groupby(tables.KEY + ["feature"]).w.agg(["size", "sum"]).reset_index()
        event = g.groupby(tables.KEY, sort=False).ngroup().to_numpy()
        ef = sparse.coo_matrix((np.r_[g["size"], g["sum"]], (np.tile(event, 2), np.r_[g.feature, g.feature + 16])),
                               shape=(len(parts), DIM)).tocsr()
        ref += v
        refcov += (ef.T @ ef).toarray()
        record = {"shard": k, "crossings": len(fx), "generated_events": len(parts), "reference": audit}
        for arm, context in (("default", False), ("context", True)):
            code, fallback = evaluation_codes(fx, lx, bins, geom, context)
            if (n[code] == 0).any():
                raise ValueError("empty selected donor cell")
            totals[arm] += np.bincount(code, minlength=ncell)
            record[arm + "_fallback"] = fallback
        records.append(record)
        files.update({p: A.digest(p) for p in (f, r)})
    result = {"cache": str(root), "cache_sha256": A.digest(root / "cache.json"), "input_sha256": files,
              "shards": records, "reference": ref.tolist(), "reference_event_poisson_cov": refcov.tolist(), "arms": {}}
    project = np.zeros((8, DIM))
    for j in range(8):
        project[j, j * 4:(j + 1) * 4] = 1
    for arm, neval in totals.items():
        mom = conditional_moments(x, cell, n, neval)
        mean, cov = mom["total_mean"], mom["sampling_cov"]
        sector, rc = project @ mean, project @ ref
        sd = np.sqrt(np.maximum(np.diag(project @ cov @ project.T), 0))
        result["arms"][arm] = {"mean": mean.tolist(), "sampling_cov": cov.tolist(), "sector_mean": sector.tolist(),
                               "sector_reference": rc.tolist(), "sector_ratio": (sector / rc).tolist(),
                               "sector_sampling_sd": sd.tolist()}
        print(arm, "other-bar pre-placement energy/G4", dict(zip(GROUPS, (sector / rc)[4:])), flush=True)
    result.update(seconds=time.monotonic() - tick, source_sha256=A.digest(__file__),
                  limitations=["Pre-placement donor destination, not a placed prediction or closure gate.",
                               "Exact conditional sampler mean/covariance; no donor or denominator uncertainty.",
                               "Reference covariance is event-Poisson, not a paired comparison to the expected output."])
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"wrote {out}; {result['seconds']:.2f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("cache")
    b.add_argument("manifest")
    b.add_argument("out")
    b.add_argument("--counters")
    e = sub.add_parser("evaluate")
    for name in ("cache", "transport", "reference", "out"):
        e.add_argument("--" + name, required=True)
    e.add_argument("--shards", default="11-20")
    a = vars(p.parse_args())
    command = a.pop("command")
    (build_cache if command == "cache" else evaluate)(**a)


if __name__ == "__main__":
    main()
