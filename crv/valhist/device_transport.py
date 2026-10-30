"""Reproducible held-out transport comparison and cold/warm device timing.

Reads existing inputs in place. Only explicitly requested outputs are written.
The NumPy baseline is regenerated once for its staged-table receipt; stochastic
arms use independent RNGs, and step histograms use the existing clustered rules.

python3 -m valhist.device_transport --inputs DIR --reference DIR --geometry CSV \
    --gdml FILE --counters FILE --shards 22 --out DIR [--repeat 2]
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, transport as nt, transport_jax as jt
from crvfastsim import crossings as nc, crossings_jax as jc
from crvfastsim.analyses import v1compare as v1
from . import stephist as sh, steplayers as sl


def deterministic_check(mu, geom):
    p = np.linalg.norm(mu[["px", "py", "pz"]].to_numpy(), axis=1)
    o = mu[["x", "y", "z"]].to_numpy()
    d = mu[["px", "py", "pz"]].to_numpy() / p[:, None]
    ref = nc.crossings(o, d, geom, backend="numpy", chunk=1024)
    out = []
    for start in range(0, len(mu), 1024):
        xs = jax.device_get(jc.crossings(o[start:start+1024], d[start:start+1024],
                                       geom.center-geom.halflen, geom.center+geom.halflen))
        if xs.overflow:
            raise RuntimeError("deterministic crossing capacity exceeded")
        it, k = np.nonzero(xs.valid)
        out.append(np.c_[it+start, xs.bar[it,k], xs.entry[it,k], xs.exit[it,k], xs.exit[it,k]-xs.entry[it,k]])
    cand = np.concatenate(out) if out else np.zeros((0, 5))
    cand = cand[np.lexsort((cand[:,1], cand[:,0]))]
    exact_ids = np.array_equal(ref[:,:2], cand[:,:2])
    max_delta = float(np.max(np.abs(ref[:,2:]-cand[:,2:]), initial=0.)) if exact_ids else None
    return dict(count_numpy=len(ref), count_jax=len(cand), exact_track_bar=exact_ids, max_distance_delta_mm=max_delta)


def baseline_receipt(a, b):
    if list(a) != list(b) or a.shape != b.shape:
        return dict(same_schema=False, generated_rows=len(a), staged_rows=len(b))
    delta = {}
    for c in a:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if np.issubdtype(x.dtype, np.number):
            delta[c] = float(np.max(np.abs(x.astype(float)-y.astype(float)), initial=0.))
    return dict(same_schema=True, rows=len(a), max_abs=delta)


def fingerprint_frames(*frames):
    h = hashlib.sha256()
    for frame in frames:
        h.update(str(list(frame.columns)).encode())
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    geom = geometry.from_offline_config(args.counters)
    world, ab = nt.load_world(args.gdml)
    dev = jt.DeviceTransport(geom, world, ab, batch_size=args.batch_size)
    edges = {"y": sl.Y_EDGES, "z": sl.Z_EDGES}
    sh._GEOM.update(geom=geom, grp=sl.bar_table(geom).grp.to_numpy())
    result = dict(devices=[str(d) for d in jax.devices()], batch_size=args.batch_size, shards={},
                  timing_note="Synchronized wall times; cold includes compilation; warm reuses prepared geometry.")
    ah, bh, ph = {}, {}, {}
    jax_stock, numpy_stock, port = v1._new_state(), v1._new_state(), v1._new_state()
    for shard in args.shards:
        suffix = f"001400_{shard:08d}.root"
        gen = tables.read(str(Path(args.inputs)/f"gen.{suffix}"), "gen")
        sub = tables.read(str(Path(args.inputs)/f"gen.{suffix}"), "subruns")
        staged = tables.read(str(Path(args.reference)/f"fastC.{suffix}"), "steps")
        a = time.perf_counter()
        if args.reuse_baseline:
            sn, pn = staged, tables.read(str(Path(args.reference)/f"fastC.{suffix}"), "particles")
        else:
            sn, pn = nt.transport(gen, geom, np.random.default_rng([shard, 0]), stage="C", world=world, absorbers=ab)
        numpy_s = None if args.reuse_baseline else time.perf_counter()-a
        numpy_times = [] if args.reuse_baseline else [numpy_s]
        for _ in range(0 if args.reuse_baseline else args.numpy_repeat-1):
            a = time.perf_counter()
            nt.transport(gen, geom, np.random.default_rng([shard, 0]), stage="C", world=world, absorbers=ab)
            numpy_times.append(time.perf_counter()-a)
        receipt = baseline_receipt(sn, staged)
        # The staged table is the baseline once the reproduction receipt has been recorded.
        trials, fingerprints = [], []
        for trial in range(args.repeat+1):
            timing = {}
            a = time.perf_counter()
            sj, pj = jt.transport(gen, geom, np.random.default_rng([shard+100000, 0]), stage="C", prepared=dev, timers=timing)
            trials.append(dict(total_s=time.perf_counter()-a, **timing))
            fingerprints.append(fingerprint_frames(sj, pj))
        if len(set(fingerprints)) != 1:
            (out/f"repeat_failure_{shard}.json").write_text(json.dumps(dict(shard=shard, fingerprints=fingerprints, trials=trials)))
            raise RuntimeError("JAX transport repeat fingerprint changed")
        events = gen[tables.KEY].drop_duplicates()
        ha, ia = sh.fill(sn, pn, events, edges)
        hb, ib = sh.fill(sj, pj, events, edges)
        sh.add(ah, ha); sh.add(bh, hb)
        pair = sh.paired(ia, ib, edges)
        sh.add_paired(ph, pair)
        comparison = sh.compare(ha, hb, pair)
        row = dict(generated_events=len(events), primary_muons=len(pn), numpy_steps=len(sn), jax_steps=len(sj),
                   baseline_receipt=receipt, deterministic=deterministic_check(nt.all_primary_muons(gen), geom),
                   numpy_s=numpy_s, numpy_times_s=numpy_times, trials=trials, histograms=comparison,
                   repeat_fingerprints=fingerprints, repeat_exact=True, baseline_reused=args.reuse_baseline)
        if args.paired_jax_reference:
            previous = tables.read(str(Path(args.paired_jax_reference)/f"fastC_jax.{suffix}"), "steps")
            row["paired_jax_receipt"] = baseline_receipt(sj, previous)
            identity = tables.KEY + ["bar", "sp", "nstep"]
            row["paired_jax_same_identity"] = np.array_equal(sj[identity].to_numpy(), previous[identity].to_numpy())
            row["paired_jax_same_stops"] = np.array_equal(sj.p1.to_numpy() == 0., previous.p1.to_numpy() == 0.)
            if not row["paired_jax_same_identity"] or not row["paired_jax_same_stops"]:
                raise RuntimeError("optimized JAX transport changed crossing/stopping identity")
            np.testing.assert_allclose(sj.to_numpy(), previous.to_numpy(), rtol=2e-10, atol=1e-8)
        candidate = out/f"fastC_jax.{suffix}"
        tables.write(str(candidate), sj, pj, gen, sub, as_ttree=True)
        if args.stock:
            stock = Path(args.stock)/f"g4all.{suffix}"
            state = v1._new_state()
            v1._accumulate(state, [str(candidate)], [str(stock)], geom.sector)
            row["v1_vs_stock"] = v1.summarise(state)
            baseline_state = v1._new_state()
            v1._accumulate(baseline_state, [str(Path(args.reference)/f"fastC.{suffix}")], [str(stock)], geom.sector)
            row["numpy_v1_vs_stock"] = v1.summarise(baseline_state)
            v1._merge_into(jax_stock, state)
            v1._merge_into(numpy_stock, baseline_state)
            v1._accumulate(port, [str(candidate)], [str(Path(args.reference)/f"fastC.{suffix}")], geom.sector)
            result["aggregate_v1_vs_stock"] = v1.summarise(jax_stock)
            result["aggregate_numpy_v1_vs_stock"] = v1.summarise(numpy_stock)
            result["aggregate_port_v1"] = v1.summarise(port)
        result["shards"][str(shard)] = row
        result["aggregate_histograms"] = sh.compare(ah, bh, ph)
        result["histogram_accumulators"] = dict(numpy=sh.to_json(ah), candidate=sh.to_json(bh), paired_d2=ph)
        result["v1_states"] = dict(numpy_stock=numpy_stock, candidate_stock=jax_stock, port=port)
        (out/"summary.json").write_text(json.dumps(result, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))
        print(json.dumps(dict(shard=shard, numpy_s=numpy_s, trials=trials, steps=[len(sn),len(sj)],
                              yield_stat=comparison["_total/all"])), flush=True)
        if args.no_candidate_tables:
            candidate.unlink()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--geometry", help="reserved provenance path; geometry is reconstructed from --counters")
    p.add_argument("--gdml", required=True)
    p.add_argument("--counters", required=True)
    p.add_argument("--stock")
    p.add_argument("--paired-jax-reference", help="previous same-key JAX step tables; require near-exact paired output")
    p.add_argument("--shards", type=int, nargs="+", default=[22])
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--numpy-repeat", type=int, default=1)
    p.add_argument("--no-candidate-tables", action="store_true", help="retain summary only after all comparisons; delete only candidate tables written by this run")
    p.add_argument("--reuse-baseline", action="store_true", help="reuse staged baseline after an independent reproduction receipt; no NumPy timing claim")
    p.add_argument("--batch-size", type=int, default=4096)
    run(p.parse_args())


if __name__ == "__main__":
    main()
