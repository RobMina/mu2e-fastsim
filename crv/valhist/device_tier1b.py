"""Strict-v3 fixed-uniform replay and independent-stream device sampling audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, tier1b as T, tier1b_assets
from crvfastsim.tier1b_jax import DeviceRandom, DeviceTier1b, ReplayRandom
from . import descendant_audit as A, descendant_topology as D, device_tier1b_moments as M, placement_closure as P, stephist as H, steplayers as S


class RecordRandom:
    def __init__(self, seed):
        self.rng, self.draws = np.random.default_rng(seed), []

    def random(self, n):
        x = self.rng.random(n)
        self.draws.append(x)
        return x


def compare_frames(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=2e-12, atol=2e-9)
    integer = [c for c in a if a[c].dtype.kind in "iub"]
    for c in integer:
        np.testing.assert_array_equal(a[c], b[c], err_msg=c)
    return max((float(np.max(np.abs(a[c].to_numpy() - b[c].to_numpy()), initial=0, where=np.isfinite(a[c].to_numpy())))
                for c in a if a[c].dtype.kind == "f"), default=0.)


def check_metadata(dev, st):
    """Compare support-defining metadata exactly before any random draws."""
    xx, vv, meta = dev._metadata(st)
    n = len(st)
    mu, vis, visits = T.muon_visits(st, dev.geom)
    np.testing.assert_array_equal(np.asarray(xx["mu"])[:n], mu)
    np.testing.assert_array_equal(np.asarray(xx["vis"])[:n], vis)
    take = np.asarray(vv["key"]) != np.iinfo(np.int64).max
    np.testing.assert_array_equal(np.asarray(vv["key"])[take], visits.mu.astype(np.int64) * 100000 + visits.vis.astype(np.int64))
    for c in ("lid", "wmin", "wmax"):
        np.testing.assert_array_equal(np.asarray(vv[c])[take], visits[c])
    np.testing.assert_array_equal(np.asarray(vv["entry"])[take], visits[["ex", "ey", "ez"]].to_numpy())
    p = np.linalg.norm(st[["px0", "py0", "pz0"]].to_numpy(), axis=1)
    expected = dict(ctx=T.crossing_context(st, dev.geom),
                    bins=T._bin(dev.lib["libbins"], T._group_index(dev.geom, st.bar), p, st.len.to_numpy()),
                    cells=T.uv_cells(st.bar.to_numpy(), st[["x0", "y0", "z0"]].to_numpy(), p,
                                     T.last_crossing(st), T.visit_mask(st, dev.geom), dev.geom),
                    last=T.last_crossing(st), mask=T.visit_mask(st, dev.geom))
    for c, a in expected.items():
        np.testing.assert_array_equal(np.asarray(meta[c])[:n], a, err_msg=c)
    hard_info = None
    if dev.hard is not None:
        xx, _, _ = dev._metadata(st)
        pat = dev.hard_patterns(xx, dev.metadata_constants, dev.hard)
        nmu = int(pat["nmu"])
        order = np.lexsort((st.t0.to_numpy(), mu))
        first = np.r_[True, mu[order][1:] != mu[order][:-1]] if n else np.zeros(0, bool)
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n) - np.maximum.accumulate(np.where(first, np.arange(n), 0)) if n else np.zeros(0, np.int64)
        plo, phi, counts = T._patterns(mu, rank, T._symbols(dev.geom, st.bar), nmu)
        pp = np.zeros(nmu)
        pp[mu[order[first]]] = p[order[first]]
        np.testing.assert_array_equal(np.asarray(pat["pattern"])[:nmu], np.c_[np.searchsorted(T.HARD_PCLASS, pp), plo, phi])
        np.testing.assert_array_equal(np.asarray(pat["count"])[:nmu], counts)
        np.testing.assert_array_equal(np.asarray(pat["rank"])[:n], rank)
        hard_info = dict(muons=nmu, exact=True)
    return dict(crossings=n, visits=int(take.sum()), exact=True, hard=hard_info)


def replay(dev, st, pa, gen, opts, seed):
    rng = RecordRandom(seed)
    tick = time.perf_counter()
    host = T.sample(st, pa, gen, dev.lib, dev.geom, rng, **opts)
    host_s = time.perf_counter() - tick
    rr = ReplayRandom(rng.draws)
    tick = time.perf_counter()
    device = dev.sample(st, pa, gen, rr, **opts)
    device_s = time.perf_counter() - tick
    if rr.index != len(rng.draws):
        raise AssertionError("unused replay draws")
    print(json.dumps({"replay_dtypes": {k: {c: str(f[c].dtype) for c in f} for k, f in
                                       (("desc", dev.ld), ("xing", dev.lx))},
                      "xgeo_dtypes": {k: str(v.dtype) for k, v in dev.xgeo.items()},
                      "replay_max_abs": {c: float(np.max(np.abs(host[0][c].to_numpy() - device[0][c].to_numpy()), initial=0))
                                         for c in host[0]}}), flush=True)
    try:
        error = compare_frames(host[0], device[0])
        compare_frames(host[1], device[1])
    except AssertionError:
        ht, dt = {}, {}
        hd = T._desc_for_crossings(st, dev.lib, dev.geom, ReplayRandom(rng.draws), context=True, contain=True,
                                   point=True, topology=True, trace=ht, drop_unvisited=True, chord=True, hard=opts.get("hard", False))[0]
        dd, *rest = dev.descendants(st, ReplayRandom(rng.draws), trace=dt, hard=opts.get("hard", False))
        differences = {c: float(np.max(np.abs(hd[c].to_numpy() - dd[c].to_numpy()), initial=0))
                       for c in hd if hd[c].dtype.kind in "fiu"}
        print(json.dumps({"desc_max_abs": differences, "donor_exact": bool(np.array_equal(ht["xid"], dt["picks"][dt["ix"]])),
                          "bar_exact": bool(np.array_equal(ht["target_bar"], dt["values"]["bar"])),
                          "why_device": rest[2]}), flush=True)
        raise
    expected_host = host[2]["templates"]["nearmiss"]["expected_reached"]
    expected_device = device[2]["templates"]["nearmiss"]["expected_reached"]
    np.testing.assert_allclose(expected_host, expected_device, rtol=2e-14, atol=1e-13)
    hi, di = json.loads(json.dumps(host[2])), json.loads(json.dumps(device[2]))
    hi["templates"]["nearmiss"].pop("expected_reached")
    di["templates"]["nearmiss"].pop("expected_reached")
    for c in ("hard_expected", "hard_no_key"):
        if c in hi["other_bar"]:
            np.testing.assert_allclose(hi["other_bar"][c], di["other_bar"][c], rtol=2e-14, atol=1e-13)
            hi["other_bar"].pop(c)
            di["other_bar"].pop(c)
    if hi != di:
        raise AssertionError(f"info differs: {host[2]} != {device[2]}")
    return dict(host_s=host_s, device_s=device_s, rows=len(host[0]), max_abs=error)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(out / "summary.json")
    geom = geometry.from_offline_config(args.counters)
    spec = tier1b_assets.resolve(args.tier1b_set, directory=args.library_dir)
    tick = time.perf_counter()
    lib = tier1b_assets.load(spec, geom)
    load_s = time.perf_counter() - tick
    dev = DeviceTier1b(lib, geom, args.batch_size)
    hashes = {str(p): A.digest(p) for p in [*spec.paths(), T.__file__, __file__,
                                         *[Path(T.__file__).with_name(f"tier1b{suffix}.py") for suffix in ("_jax", "_metadata_jax", "_csr_jax", "_hard_jax")]]}
    edges = S.booking_edges()
    hist, pair, topo = {n: {} for n in ("numpy", "jax")}, {}, {n: {} for n in ("numpy", "jax")}
    records = []
    moments, layer_pairs = {n: {} for n in ("numpy", "jax")}, {}
    component_pairs = {c: {} for c in H.COMPONENTS[1:]}
    for shard in args.shards:
        path = str(Path(args.transport) / f"fastC.{S.shard(shard)}.root")
        st, pa, gen = (tables.read(path, tree) for tree in ("steps", "particles", "gen"))
        hashes[path] = A.digest(path)
        record = dict(shard=shard, events=len(gen[tables.KEY].drop_duplicates()), input_steps=len(st))
        record["metadata"] = check_metadata(dev, st)
        record["replay"] = replay(dev, st, pa, gen, spec.options, [shard, 1000])
        if args.warm_repeats and (args.warm_shards is None or shard in args.warm_shards):
            record["warmup"] = {}
            for backend in ("numpy", "jax"):
                tick = time.perf_counter()
                if backend == "numpy":
                    T.sample(st, pa, gen, lib, geom, np.random.default_rng([shard, 1000]), **spec.options)
                else:
                    dev.sample(st, pa, gen, DeviceRandom(shard), **spec.options)
                record["warmup"][backend] = time.perf_counter() - tick
            record["warm"] = {"numpy": [], "jax": []}
            record["warm_boundaries"] = []
            for backend in ("numpy", "jax"):
                for _ in range(args.warm_repeats):
                    tick = time.perf_counter()
                    stream = np.random.default_rng([shard, 1000]) if backend == "numpy" else DeviceRandom(shard)
                    if backend == "numpy":
                        T.sample(st, pa, gen, lib, geom, stream, **spec.options)
                    else:
                        dev.sample(st, pa, gen, stream, **spec.options)
                    record["warm"][backend].append(time.perf_counter() - tick)
                    if backend == "jax":
                        record["warm_boundaries"].append(dict(dev.last))
        if not args.replay_only:
            events = gen[tables.KEY].drop_duplicates()
            arms, labels_of = {}, {}
            for name, rng in (("numpy", np.random.default_rng([shard, 1000])), ("jax", DeviceRandom(shard))):
                tick = time.perf_counter()
                steps, parts, info = (T.sample(st, pa, gen, lib, geom, rng, **spec.options) if name == "numpy"
                                      else dev.sample(st, pa, gen, rng, **spec.options))
                elapsed = time.perf_counter() - tick
                hh, idx = H.fill(steps[H.STEP_COLS], parts, events, edges)
                H.add(hist[name], hh)
                arms[name] = idx
                labels = D.topo_labels(steps, parts, geom)
                labels_of[name] = labels
                M.add(moments[name], M.fill(steps, parts, events, geom))
                topo[name] = S._merge_views(topo[name], {v: S.accumulate(labels, labels, cols) for v, cols in D.VIEWS.items()})
                record[name] = dict(seconds=elapsed, rows=len(steps), particles=len(parts), info=info,
                                    boundary=dev.last if name == "jax" else None)
                if args.write:
                    dest = out / name
                    dest.mkdir(exist_ok=True)
                    tables.write(str(dest / Path(path).name.replace("fastC.", "fast1b.")), steps, parts, gen,
                                 tables.read(path, "subruns"), as_ttree=True)
            H.add_paired(pair, H.paired(arms["numpy"], arms["jax"], edges))
            for component in component_pairs:
                filtered = {name: {k: (ev[c == component], b[c == component], c[c == component])
                                   for k, (ev, b, c) in idx.items()} for name, idx in arms.items()}
                H.add_paired(component_pairs[component], H.paired(filtered["numpy"], filtered["jax"], edges))
            layer_pairs = S._merge_views(layer_pairs, {v: S.accumulate(labels_of["numpy"], labels_of["jax"], cols)
                                                       for v, cols in D.VIEWS.items()})
        records.append(record)
        print(json.dumps(record), flush=True)
    comparison = H.compare(hist["numpy"], hist["jax"], pair) if hist["numpy"] else {}
    print(json.dumps({"comparison": comparison, "moments": {n: M.summarize(v) for n, v in moments.items()}
                      if not args.replay_only else {}}), flush=True)
    result = dict(records=records, tier1b=spec.describe(), hashes=hashes, library_load_s=load_s, device_prepare_s=dev.preparation_s,
                  comparison=comparison, flagged=[k for k, v in comparison.items() if v.get("flag")],
                  layer_census=A.summarize(layer_pairs),
                  component_yields={k: {c: H.yield_stat(hist["numpy"][k], hist["jax"][k], c, component_pairs[c].get(k))
                                            for c in component_pairs} for k in hist["numpy"] if k.startswith("_total")},
                  moments={n: M.summarize(v) for n, v in moments.items()} if not args.replay_only else {},
                  raw_moments={n: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in a.items()}
                               for n, a in moments.items()} if not args.replay_only else {},
                  device_array_bytes=dev.device_bytes, batch_size=args.batch_size,
                  hists={n: H.to_json(h) for n, h in hist.items()}, paired={k: v.tolist() for k, v in pair.items()},
                  topology={n: {v: f.to_dict(orient="list") for v, f in vv.items()} for n, vv in topo.items()},
                  certified=False, boundary="device crossing metadata/main-UV CSR/placement; host showers/near-miss grouping/ancestry publication")
    for path, old in hashes.items():
        if A.digest(path) != old:
            raise ValueError(f"input/code changed during run: {path}")
    (out / "summary.json").write_text(json.dumps(result, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("counters", "library-dir", "transport", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--shards", type=int, nargs="+", default=[22])
    p.add_argument("--tier1b-set", choices=("strict_v3", "strict_v4"), default="strict_v4")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--replay-only", action="store_true")
    p.add_argument("--warm-repeats", type=int, default=0)
    p.add_argument("--warm-shards", type=int, nargs="+", help="Time only these shards; every requested shard is still replayed")
    p.add_argument("--write", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
