"""Exact additive-observable shower sampling and corrected-source donor diagnostics.

This is an analysis of the retained sampler, not a response model or rate correction.
It holds muon-family output fixed and never uses event/bar occupancy as an additive feature.
"""
import argparse
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
import uproot

from crvfastsim import genassoc, geometry, tables, tier1b
from crvfastsim.analyses.census import SECTOR_GROUP, classify, primary_of
from valhist import shower_moments as M
from valhist import denominator_provenance as P
from valhist.genassoc_campaign import _new_json, verify_library
from valhist.library_variants import sha256
from valhist.steplayers import booking, booking_edges, shard

GROUPS = ("EX", "T1", "T2", "M")
KEY = tables.KEY
BIN = ["apdg", "kebin"]


def fingerprint(paths):
    return {str(Path(p).resolve(strict=True)): sha256(p) for p in dict.fromkeys(paths)}


def verify(hashes):
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("input or code drift: " + path)


def code_identity():
    return fingerprint([__file__, M.__file__, genassoc.__file__, tier1b.__file__, geometry.__file__,
                        P.__file__, inspect.getfile(primary_of), inspect.getfile(tables.read)])


def event_index(keys):
    frame = keys[KEY].copy()
    for k in KEY:
        a = frame[k].to_numpy()
        if a.dtype.kind not in "iu" or (a < 0).any() or (a > np.iinfo(np.uint32).max).any():
            raise ValueError("invalid full-width event identities")
        frame[k] = a.astype(np.int64)
    return pd.MultiIndex.from_frame(frame)


def event_rows(frame, census):
    index = event_index(census)
    if not index.is_unique:
        raise ValueError("duplicate event census")
    rows = index.get_indexer(event_index(frame))
    if (rows < 0).any():
        raise ValueError("physics rows outside generated event census")
    return rows


def bin_frame(gen):
    g = gen.loc[gen.pdg.abs() != 13].copy()
    g["apdg"] = g.pdg.abs().astype(np.int64)
    ke = g.e - g.apdg.map(tier1b.MASS).fillna(0.0)
    if not np.isfinite(ke).all() or (ke < 0).any():
        raise ValueError("nonfinite or negative native generator KE")
    g["kebin"] = tier1b._ke_bin(ke.to_numpy())
    return g


def cell_rows(frame, cells):
    index = pd.MultiIndex.from_frame(cells[BIN])
    if not index.is_unique:
        raise ValueError("duplicate cell definition")
    rows = index.get_indexer(pd.MultiIndex.from_frame(frame[BIN]))
    if (rows < 0).any():
        raise ValueError("unaccounted species/KE cell")
    return rows


def feature_layout(edges):
    names = [f"{g}/{q}" for g in GROUPS for q in ("steps", "edep")]
    blocks = {}
    for g in GROUPS:
        for axis in ("y", "z"):
            n = len(edges[axis]) + 1
            blocks[f"{g}/{axis}"] = [len(names), len(names) + n]
            names += [f"{g}/{axis}/{i}" for i in range(n)]
    return names, blocks


def features(steps, row, nrows, sector, edges):
    """Sparse row-by-feature matrix; all steps processed by array operations."""
    names, blocks = feature_layout(edges)
    row = np.asarray(row)
    if row.dtype.kind not in "iu" or len(row) != len(steps) or (row < 0).any() or (row >= nrows).any():
        raise ValueError("invalid step feature row")
    bars = steps.bar.to_numpy()
    if bars.dtype.kind not in "iu" or (bars < 0).any() or (bars >= len(sector)).any():
        raise ValueError("invalid bar")
    group = pd.Index(GROUPS).get_indexer(SECTOR_GROUP[sector[bars]])
    if (group < 0).any():
        raise ValueError("uncovered CRV sector")
    edep = steps.edep.to_numpy(dtype=float)
    if not np.isfinite(edep).all() or (edep < 0).any():
        raise ValueError("invalid step energy")
    rows = [row, row]
    cols = [2 * group, 2 * group + 1]
    weights = [np.ones(len(row)), edep]
    for axis in ("y", "z"):
        pos = steps[axis + "0"].to_numpy().astype(np.float32).astype(np.float64)
        if not np.isfinite(pos).all():
            raise ValueError("nonfinite position")
        offsets = np.asarray([blocks[f"{g}/{axis}"][0] for g in GROUPS])
        rows.append(row)
        cols.append(offsets[group] + np.searchsorted(edges[axis], pos, side="right"))
        weights.append(np.ones(len(row)))
    result = sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(nrows, len(names))).tocsr()
    result.sum_duplicates()
    return result


def grouped_features(matrix, rows, nrows):
    mapping = sparse.coo_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))),
                               shape=(nrows, len(rows))).tocsr()
    return (mapping @ matrix).tocsr()


def prepare(campaign, outdir):
    """Bind a completed strict campaign and compress every persisted shower template exactly."""
    start = time.monotonic()
    root = Path(campaign).resolve(strict=True)
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError(out)
    verify_library(root)
    meta_path = root / "library/library.json"
    meta = json.loads(meta_path.read_text())
    plan = json.loads(Path(meta["plan"]).read_text())
    hashes = fingerprint([meta_path, root / "library/library.root", meta["plan"], meta["geometry"], booking(),
                          *meta["inputs"], *meta["gen"]])
    code = code_identity()
    geom = geometry.from_offline_config(meta["geometry"])
    edges = booking_edges()
    names, blocks = feature_layout(edges)
    lib = {name: tables.read(str(root / "library/library.root"), name)
           for name in ("libshower", "libshowersteps", "libgen")}
    sh = lib["libshower"]
    if not np.array_equal(sh.shid.to_numpy(), np.arange(len(sh))):
        raise ValueError("non-contiguous or reordered library shower ids")
    columns = ["shid", "apdg", "kebin", "ke"]
    projections, profile, fileid, records, donor_keys = [], [], [], [], []
    offset = 0
    for f, row in enumerate(plan["donors"]):
        done_path = root / "logs" / f"dump.{row['sequence']}.done.json"
        done = json.loads(done_path.read_text())
        hashes[str(done_path)] = sha256(done_path)
        if done["exit"] != 0 or done["output_sha256"] != hashes[row["new_table"]]:
            raise ValueError("uncertified donor completion")
        checked = genassoc.validate_donor(row["new_table"])
        census = tables.read(row["new_table"], "events", KEY)
        donor_keys.append(event_index(census))
        showers, _ = P.donor_showers(row["new_table"])
        if showers[["pdg_disagreement", "gen_join_missing", "negative_ke", "nonfinite"]].to_numpy().any():
            raise ValueError("defective corrected shower projection")
        showers["shid"] += offset
        projections.append(showers[columns])
        profile.append(showers[["apdg", "kebin", "ke", "gen_px", "gen_py", "gen_pz", "gen_x", "gen_y", "gen_z"]])
        fileid.append(np.full(len(showers), f, np.int64))
        offset += len(showers)
        records.append({"path": row["new_table"], "showers": len(showers), "validation": checked})
        print(f"projected donor {f + 1}/{len(plan['donors'])}: {len(showers)} showers", flush=True)
    pd.testing.assert_frame_equal(pd.concat(projections, ignore_index=True), sh[columns], check_exact=True, check_dtype=False)
    rows = pd.Index(sh.shid).get_indexer(lib["libshowersteps"].shid)
    if (rows < 0).any():
        raise ValueError("orphan library shower steps")
    x = features(lib["libshowersteps"], rows, len(sh), geom.sector, edges)
    cells = lib["libgen"].sort_values(BIN).reset_index(drop=True)
    c = cell_rows(sh, cells)
    observed = np.bincount(c, minlength=len(cells))
    if not np.array_equal(observed, cells.nshower.to_numpy()) or (observed > cells.ngen).any():
        raise ValueError("invalid persisted reach support")
    fileid = np.concatenate(fileid)
    denom_counts, denom_events, identities = [], [], []
    for path in meta["gen"]:
        genassoc.validate_generator_stream(path)
        gen = tables.read(path, "gen")
        census = genassoc.generator_event_keys(path)
        ids = event_index(census)
        identities.append(ids)
        denom_counts.append(np.bincount(cell_rows(bin_frame(gen), cells), minlength=len(cells)))
        denom_events.append(len(census))
    allids = identities[0].append(identities[1:])
    if not allids.is_unique:
        raise ValueError("overlapping denominator shards")
    donor_ids = donor_keys[0].append(donor_keys[1:])
    if not donor_ids.is_unique:
        raise ValueError("overlapping donor census identities")
    expected = np.rint(np.sum(denom_counts, axis=0) * (plan["generated"] / sum(denom_events))).astype(np.int64)
    if not np.array_equal(expected, cells.ngen.to_numpy()):
        raise ValueError("persisted ngen differs from unchanged bank/exposure")
    verify({**hashes, **code})
    out.mkdir()
    sparse.save_npz(out / "templates.npz", x)
    sparse.save_npz(out / "file_cell_features.npz", grouped_features(x, fileid * len(cells) + c,
                                                                   len(records) * len(cells)))
    np.savez_compressed(out / "index.npz", cell=c, file_index=fileid, ngen=cells.ngen.to_numpy(),
                        apdg=cells.apdg.to_numpy(), kebin=cells.kebin.to_numpy(),
                        gen_counts=np.asarray(denom_counts), gen_events=np.asarray(denom_events),
                        donor_events=donor_ids.to_frame(index=False).to_numpy(dtype=np.int64),
                        profile=pd.concat(profile, ignore_index=True).to_numpy())
    result = {"campaign": str(root), "library": str(root / "library/library.root"), "generated": plan["generated"],
              "generated_per_file": plan["generated_per_file"], "donors": records, "gen": meta["gen"],
              "geometry": meta["geometry"], "geometry_sha256": meta["geometry_sha256"],
              "names": names, "blocks": blocks, "edges": {k: v.tolist() for k, v in edges.items()},
              "profile_columns": list(profile[0].columns), "input_sha256": hashes, "code_sha256": code,
              "output_sha256": fingerprint([out / p for p in ("templates.npz", "file_cell_features.npz", "index.npz")]),
              "wall_seconds": time.monotonic() - start,
              "limitation": "Additive, time-independent step observables only; not unique event/bar hits or response."}
    _new_json(out / "cache.json", result)
    print(f"prepared {len(sh)} templates, {x.shape[1]} features, {x.nnz} nonzeros in {result['wall_seconds']:.2f}s", flush=True)


def load_cache(path):
    root = Path(path).resolve(strict=True)
    meta = json.loads((root / "cache.json").read_text())
    verify(meta["output_sha256"])
    index = dict(np.load(root / "index.npz", allow_pickle=False))
    cells = pd.DataFrame({k: index[k] for k in ("apdg", "kebin")})
    return meta, index, cells, sparse.load_npz(root / "templates.npz"), sparse.load_npz(root / "file_cell_features.npz")


def reference_pairing(reference, legacy, fullgen):
    """Certify retyped reference rows and actual generator payload, not merely reused event labels."""
    from valhist.g4table import retype
    steps, parts = retype(tables.read(legacy, "steps"), tables.read(legacy, "particles"))
    for name, expected in (("steps", steps), ("particles", parts)):
        pd.testing.assert_frame_equal(expected, tables.read(reference, name), check_exact=True, check_dtype=False)
    accepted_gen = tables.read(legacy, "gen")
    if accepted_gen.duplicated(KEY + ["idx"]).any() or fullgen.duplicated(KEY + ["idx"]).any():
        raise ValueError("duplicate generator identity in reference pairing")
    index = pd.MultiIndex.from_frame(fullgen[KEY + ["idx"]])
    rows = index.get_indexer(pd.MultiIndex.from_frame(accepted_gen[KEY + ["idx"]]))
    if (rows < 0).any():
        raise ValueError("reference generator missing from full population")
    pd.testing.assert_frame_equal(accepted_gen.reset_index(drop=True),
                                  fullgen.iloc[rows][accepted_gen.columns].reset_index(drop=True),
                                  check_exact=True, check_dtype=False)
    return {"reference": reference, "legacy_donor": legacy, "retyped_payload_identical": True,
            "generator_payload_identical": True, "generator_rows": len(accepted_gen)}


def evaluation(cache, corrected, reference, generated, selected, outdir):
    """Sufficient event statistics for exact conditional paired residuals, including all zero-step events."""
    out = Path(outdir).resolve()
    if out.exists():
        raise FileExistsError(out)
    code = code_identity()
    meta, index, cells, _, _ = load_cache(cache)
    geom = geometry.from_offline_config(meta["geometry"])
    edges = {k: np.asarray(v) for k, v in meta["edges"].items()}
    extra_cells = [cells]
    for k in selected:
        gen = tables.read(generated.format(s=shard(k), k=k), "gen")
        extra_cells.append(bin_frame(gen)[BIN].drop_duplicates())
    cells = pd.concat(extra_cells).drop_duplicates().sort_values(BIN).reset_index(drop=True)
    C, D = len(cells), len(meta["names"])
    HTH = np.zeros((C, C)); HTR = np.zeros((2, C, D)); RTR = np.zeros((2, D, D))
    HS = np.zeros(C); RS = np.zeros((2, D)); sample_sum = np.zeros(D); fixed_sum = np.zeros(D); g4_sum = np.zeros(D)
    g4_nonmu_sum = np.zeros(D); g4_second = np.zeros((D, D)); event_total = 0
    hashes, inventory, allids = {}, [], []
    train_ids = []
    for path in meta["gen"]:
        train_ids.append(event_index(genassoc.generator_event_keys(path)))
    train_index = train_ids[0].append(train_ids[1:]).union(event_index(pd.DataFrame(index["donor_events"], columns=KEY)))
    for k in selected:
        start = time.monotonic()
        paths = [p.format(s=shard(k), k=k) for p in (corrected, reference, generated)]
        hashes.update(fingerprint(paths))
        cand, ref, genpath = paths
        gen_role = genassoc.validate_generator_stream(genpath)
        gen = tables.read(genpath, "gen")
        legacy = str(Path(genpath).with_name(Path(genpath).name.replace("gen.", "crvsteps.", 1)))
        hashes.update(fingerprint([legacy]))
        pairing = reference_pairing(ref, legacy, gen)
        census = genassoc.generator_event_keys(genpath)
        ids = event_index(census)
        if not ids.is_unique or len(ids.intersection(train_index)):
            raise ValueError("evaluation overlaps training bank or duplicates itself")
        allids.append(ids)
        cand_gen = tables.read(cand, "gen")
        pd.testing.assert_frame_equal(gen.reset_index(drop=True), cand_gen[gen.columns].reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
        pop = bin_frame(gen)
        h = sparse.coo_matrix((np.ones(len(pop)), (event_rows(pop, census), cell_rows(pop, cells))),
                              shape=(len(census), C)).tocsr()
        mats = {}
        for label, path in (("cand", cand), ("ref", ref)):
            steps = tables.read(path, "steps", KEY + ["bar", "sp", "edep", "y0", "z0"])
            parts = tables.read(path, "particles", KEY + ["key", "parent", "pdg"])
            classified = classify(steps, parts, geom.sector)
            if classified.cls.eq("unknown").any():
                raise ValueError("unknown step ancestry")
            er = event_rows(classified, census)
            nonmu = classified.cls.str.startswith("nonmu").to_numpy()
            mats[label] = features(classified, er, len(census), geom.sector, edges)
            mats[label + "_nonmu"] = features(classified.loc[nonmu], er[nonmu], len(census), geom.sector, edges)
        fixed = mats["cand"] - mats["cand_nonmu"]
        residuals = [-mats["ref_nonmu"], fixed - mats["ref"]]
        HTH += (h.T @ h).toarray(); HS += np.asarray(h.sum(axis=0)).ravel()
        for i, r in enumerate(residuals):
            HTR[i] += (h.T @ r).toarray(); RTR[i] += (r.T @ r).toarray(); RS[i] += np.asarray(r.sum(axis=0)).ravel()
        sample_sum += np.asarray(mats["cand_nonmu"].sum(axis=0)).ravel()
        fixed_sum += np.asarray(fixed.sum(axis=0)).ravel(); g4_sum += np.asarray(mats["ref"].sum(axis=0)).ravel()
        g4_nonmu_sum += np.asarray(mats["ref_nonmu"].sum(axis=0)).ravel()
        g4_second += (mats["ref"].T @ mats["ref"]).toarray()
        event_total += len(census)
        inventory.append({"shard": k, "events": len(census), "nonmu_primaries": len(pop),
                          "paths": paths, "pairing": pairing, "generator_validation": gen_role,
                          "seconds": time.monotonic() - start})
        print(f"evaluation {k}: {len(census)} generated, {len(pop)} nonmu primaries, {inventory[-1]['seconds']:.2f}s", flush=True)
    if not allids[0].append(allids[1:]).is_unique:
        raise ValueError("cross-shard evaluation overlap")
    verify({**hashes, **code})
    out.mkdir()
    np.savez_compressed(out / "sufficient.npz", hth=HTH, htr=HTR, rtr=RTR, counts=HS, rsum=RS,
                        sampled=sample_sum, fixed=fixed_sum, g4=g4_sum, g4_nonmu=g4_nonmu_sum, g4_second=g4_second)
    _new_json(out / "evaluation.json", {"cache": str(Path(cache).resolve()), "cache_sha256": sha256(Path(cache) / "cache.json"),
                                       "names": meta["names"], "cells": cells.to_dict(orient="list"),
                                       "events": event_total, "shards": inventory, "input_sha256": hashes,
                                       "code_sha256": code, "output_sha256": sha256(out / "sufficient.npz"),
                                       "train_eval_overlap": 0,
                                       "legacy_census_limit": "Legacy rows infer event census; require independent v2 fullgen certificate before interpretation."})


def compact_draw(gen, libshower, libgen, rng):
    """Exactly reproduce the current sampler's acceptance and uniform-template random draws."""
    g = bin_frame(gen).merge(libgen, on=BIN, how="left").fillna({"ngen": 0, "nshower": 0})
    prob = np.divide(g.nshower, np.maximum(g.ngen, 1), out=np.zeros(len(g)), where=g.ngen.to_numpy() > 0)
    if (prob > 1).any():
        raise ValueError("invalid shower probability")
    accepted = g.loc[rng.random(len(g)) < prob].reset_index(drop=True)
    ls = libshower.sort_values(BIN, kind="stable")
    code = ls.apdg.to_numpy() * 100 + ls.kebin.to_numpy()
    want = accepted.apdg.to_numpy() * 100 + accepted.kebin.to_numpy()
    lo, hi = np.searchsorted(code, want, "left"), np.searchsorted(code, want, "right")
    if ((hi - lo) <= 0).any():
        raise ValueError("accepted unsupported shower")
    return ls.shid.to_numpy()[lo + (rng.random(len(accepted)) * (hi - lo)).astype(np.int64)]


def sampler_check(cache, generated, out, draws=256, seed=20261027):
    """Compare compressed draws with actual emitted steps, then independent compact replicas."""
    meta, index, cells, x, _ = load_cache(cache)
    hashes = fingerprint([generated, meta["library"], Path(cache) / "cache.json"])
    lib = {n: tables.read(meta["library"], n) for n in ("libshower", "libshowersteps", "libgen")}
    gen = tables.read(generated, "gen")
    picked = compact_draw(gen, lib["libshower"], lib["libgen"], np.random.default_rng(seed))
    emitted = tier1b._showers_for_gen(gen, lib, np.random.default_rng(seed))
    geom = geometry.from_offline_config(meta["geometry"])
    edges = {k: np.asarray(v) for k, v in meta["edges"].items()}
    actual = np.asarray(features(emitted, np.zeros(len(emitted), np.int64), 1, geom.sector, edges).sum(axis=0)).ravel()
    expected = np.asarray(x[picked].sum(axis=0)).ravel()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)
    samples = np.empty((draws, x.shape[1]))
    for r in range(draws):
        ids = compact_draw(gen, lib["libshower"], lib["libgen"], np.random.default_rng(np.random.SeedSequence([seed, r + 1])))
        samples[r] = np.asarray(x[ids].sum(axis=0)).ravel()
    gen_cells = pd.MultiIndex.from_frame(bin_frame(gen)[BIN])
    cidx = pd.MultiIndex.from_frame(cells[BIN]).get_indexer(gen_cells)
    neval = np.bincount(cidx[cidx >= 0], minlength=len(cells))
    moments = M.conditional_moments(x, index["cell"], index["ngen"], neval)
    result = {"input_sha256": hashes, "actual_emitted_feature_match": True, "selected_showers": len(picked),
              "draws": draws, "seed": seed, "names": meta["names"], "empirical_mean": samples.mean(axis=0).tolist(),
              "empirical_covariance": np.cov(samples, rowvar=False, ddof=1).tolist(),
              "exact": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in moments.items()},
              "limitation": "Independent-seed Monte Carlo cross-check, not a calibrated multivariate goodness-of-fit test."}
    verify(hashes)
    _new_json(out, result)
    print(f"actual-emission feature check passed; compact sampler {draws} replicas complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("campaign"); p.add_argument("outdir")
    p = sub.add_parser("evaluation")
    p.add_argument("cache"); p.add_argument("outdir")
    p.add_argument("--corrected", required=True); p.add_argument("--reference", required=True)
    p.add_argument("--generated", required=True); p.add_argument("--shards", default="11-20")
    p = sub.add_parser("sampler-check")
    p.add_argument("cache"); p.add_argument("generated"); p.add_argument("--out", required=True)
    p.add_argument("--draws", type=int, default=256); p.add_argument("--seed", type=int, default=20261027)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        prepare(**args)
    elif command == "evaluation":
        from valhist.steplayers import shards
        args["selected"] = shards(args.pop("shards"))
        evaluation(**args)
    else:
        sampler_check(**args)


if __name__ == "__main__":
    main()
