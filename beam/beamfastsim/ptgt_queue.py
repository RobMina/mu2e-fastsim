"""Batched ptgt_point evaluation through a directory queue (opt-in): many designs share one tsdevice pass.

A tsdevice pass on the GPU has a large fixed cost (~3 min on an A100, whatever the number of TS entrants), so one design
per job spends most of its time there. Here each design's job only hands its request to a resident GPU worker and waits;
the worker runs the device chain of every design it has collected (cached per design, as ptgt_point does), then one
tsdevice pass over all of their references, and writes each design's numbers from its rows of that pass (tsdevice
tags each row with its reference's position, `batch`). The numbers are ptgt_point's; only the lanes of the TS + DS
stage are shared, so a design's stops depend on its batch mates through the random stream, not in distribution.

Layout under the queue directory Q: Q/req/<id>.json (waiting), Q/run/<id>.json (taken by a worker), Q/done/<id>/
(request.json, metrics.json, point.json; or error.txt), Q/STOP (a worker stops taking requests).

    # client (what a kit job runs, via ptgt_point --queue)
    python3 -m beamfastsim.ptgt_point --queue Q --rod T RU RM RD --pot N --seed S --dtype float32 --platform gpu --out OUT
    # worker (one GPU job)
    python3 -m beamfastsim.ptgt_queue worker --queue Q --batch 10 --gather-s 90 --idle-s 900 --wall-s 4800 \\
        --bank BANK --slab-bank SLAB --ps-map PSMap.header --cache DIR --gdml TS.gdml --field-config CFG \\
        --offline-root ROOT --maps DIR --ds-geom GEOM --ds-map DSMap.header [--data DIR]
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

POLL_S = 5.0


def _write_atomic(path, text):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def submit_and_wait(queue, request, out, wait_s=None, poll_s=POLL_S):
    """Queue one design and wait for its result: copies metrics.json and point.json to out, returns the metrics.
    Raises RuntimeError with the worker's error text, or TimeoutError after wait_s seconds."""
    q, out = Path(queue), Path(out)
    for sub in ("req", "run", "done"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rid = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:12]}"
    req = dict(request, id=rid, submitted=time.time(), client=socket.gethostname(),
               job=os.environ.get("SLURM_JOB_ID", ""))
    _write_atomic(q / "req" / f"{rid}.json", json.dumps(req, indent=1))
    (out / "queue_request.json").write_text(json.dumps(req, indent=1))
    done, t0 = q / "done" / rid, time.time()
    while True:
        if (done / "error.txt").exists():
            raise RuntimeError(f"ptgt_queue worker: {(done / 'error.txt').read_text()[:2000]}")
        if (done / "metrics.json").exists() and (done / "point.json").exists():
            for name in ("metrics.json", "point.json"):
                shutil.copyfile(done / name, out / name)
            return json.loads((out / "metrics.json").read_text())
        if wait_s is not None and time.time() - t0 > wait_s:
            raise TimeoutError(f"ptgt_queue: no result for {rid} after {wait_s:.0f} s")
        time.sleep(poll_s)


def pending(queue):
    """Waiting requests, oldest first: [(path, request)]."""
    out = []
    for p in sorted(Path(queue, "req").glob("*.json")):
        try:
            out.append((p, json.loads(p.read_text())))
        except (OSError, ValueError):
            continue                      # being written (atomic rename makes this rare) or removed
    return sorted(out, key=lambda x: x[1].get("submitted", 0.0))


def take(queue, batch, gather_s, now=None):
    """Claim up to `batch` requests (moved req -> run) once the batch is full or the oldest has waited gather_s.
    Returns the claimed requests (possibly empty)."""
    now = time.time() if now is None else now
    p = pending(queue)
    if not p or (len(p) < batch and now - p[0][1].get("submitted", now) < gather_s):
        return []
    claimed = []
    for path, req in p[:batch]:
        dst = Path(queue, "run", path.name)
        try:
            os.replace(path, dst)         # a second worker loses the race harmlessly
        except FileNotFoundError:
            continue
        claimed.append(req)
    return claimed


def requeue_orphans(queue):
    """Requests a dead worker had claimed (run/ without done/): back to req/."""
    n = 0
    for p in Path(queue, "run").glob("*.json"):
        if not (Path(queue, "done", p.stem) / "metrics.json").exists():
            os.replace(p, Path(queue, "req", p.name))
            n += 1
    return n


def finish(queue, req, metrics=None, point=None, error=None):
    d = Path(queue, "done", req["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "request.json").write_text(json.dumps(req, indent=1))
    if error is not None:
        _write_atomic(d / "error.txt", error)
    else:
        _write_atomic(d / "point.json", json.dumps(point, indent=1))
        _write_atomic(d / "metrics.json", json.dumps(metrics, indent=1))   # last: the client waits for it
    run = Path(queue, "run", f"{req['id']}.json")
    if run.exists():
        run.unlink()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_data(req, data_sha):
    """None, or why the request's pinned data does not match the worker's."""
    want = req.get("data_sha256") or {}
    bad = sorted(k for k, v in want.items() if data_sha.get(k) != v)
    return None if not bad else f"data sha256 differs for {', '.join(bad)} (worker: " + \
        ", ".join(f"{k}={data_sha.get(k, 'missing')[:12]}" for k in bad) + ")"


_HASHES = {}


def hashes(base):
    """(bank content hash, slab bank content hash, PS map sha256) of the worker's inputs, computed once."""
    k = (base.bank, base.slab_bank, base.ps_map)
    if k not in _HASHES:
        from beamfastsim import ptgt_device, ptgt_point as PP
        _HASHES[k] = (ptgt_device.Bank.load(base.bank).lib.content_hash,
                      ptgt_device.Bank.load(base.slab_bank).lib.content_hash, PP.field_map_sha(base.ps_map))
    return _HASHES[k]


def run_batch(queue, reqs, base, run_chain, run_ts, data_sha):
    """One batch: the chain (cached) of each request, then one tsdevice pass per dtype, then each request's numbers.
    run_chain(argv): ptgt_chain_jax.main; run_ts(argv): runs tsdevice (a subprocess on the GPU). Returns seconds."""
    from beamfastsim import ptgt_point as PP
    t_batch = time.perf_counter()
    bank_hash, slab_hash, map_sha = hashes(base)
    ready = {}
    for req in reqs:
        why = check_data(req, data_sha)
        if why or req.get("platform", base.platform) != base.platform:
            finish(queue, req, error=why or f"platform {req.get('platform')} != worker {base.platform}")
            continue
        a = argparse.Namespace(**vars(base))
        a.rod, a.pot, a.seed, a.dtype = [float(v) for v in req["rod"]], int(req["pot"]), int(req["seed"]), req["dtype"]
        try:
            key = PP.design_key(a.rod, a.pot, a.seed, a.dtype, bank_hash, slab_hash, map_sha)
            ref, cached, chain_s = PP.states(a, key, run_chain)
        except Exception as e:            # noqa: BLE001 - one design's failure must not stop its batch mates
            finish(queue, req, error=f"chain: {type(e).__name__}: {e}")
            continue
        ready.setdefault(a.dtype, []).append((req, a, key, ref, cached, chain_s))
    for dtype, items in ready.items():
        out = Path(queue, "work", f"ts_{uuid.uuid4().hex[:8]}.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        a0 = items[0][1]
        argv = PP.tsdevice_argv(a0, items[0][3], out)
        i = argv.index("--reference")
        argv[i + 1:i + 2] = [str(it[3]) for it in items]
        t0 = time.perf_counter()
        try:
            run_ts(argv)
        except Exception as e:            # noqa: BLE001
            for it in items:
                finish(queue, it[0], error=f"tsdevice: {type(e).__name__}: {e}")
            continue
        ts_s = time.perf_counter() - t0
        members = [it[0]["id"] for it in items]
        for k, (req, a, key, ref, cached, chain_s) in enumerate(items):
            try:
                m = PP.metrics(ref, out, cached, chain_s, ts_s / len(items), batch=k)
                m.update(batch_size=float(len(items)), ts_batch_s=float(ts_s))
                point = dict(key=key, rod=a.rod, pot=a.pot, seed=a.seed, dtype=a.dtype, platform=a.platform,
                             bank=bank_hash, slab_bank=slab_hash, metrics=m, queue=dict(id=req["id"], batch=k,
                             members=members, worker=socket.gethostname(),
                             worker_job=os.environ.get("SLURM_JOB_ID", "")))
                finish(queue, req, metrics=m, point=point)
            except Exception as e:        # noqa: BLE001
                finish(queue, req, error=f"metrics: {type(e).__name__}: {e}")
        out.unlink(missing_ok=True)       # per-muon output: the points keep their numbers only
    return time.perf_counter() - t_batch


def worker(queue, base, batch=10, gather_s=90.0, idle_s=900.0, wall_s=None, run_chain=None, run_ts=None,
           data_sha=None, poll_s=POLL_S, log=print):
    """Serve the queue until STOP, idle_s without requests, or (with wall_s) no time left for another batch."""
    q = Path(queue)
    for sub in ("req", "run", "done", "work"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    n = requeue_orphans(q)
    if n:
        log(f"requeued {n} orphaned request(s)")
    t0, last, slowest, served = time.time(), time.time(), 0.0, 0
    while not (q / "STOP").exists():
        if wall_s is not None and time.time() - t0 + 1.5 * slowest + 60 > wall_s:
            log(f"wall guard: {time.time() - t0:.0f} s used, slowest batch {slowest:.0f} s")
            break
        reqs = take(q, batch, gather_s)
        if not reqs:
            if time.time() - last > idle_s:
                log(f"idle {idle_s:.0f} s: done")
                break
            time.sleep(poll_s)
            continue
        dt = run_batch(q, reqs, base, run_chain, run_ts, data_sha or {})
        slowest, last, served = max(slowest, dt), time.time(), served + len(reqs)
        log(json.dumps(dict(batch=len(reqs), seconds=round(dt, 1), served=served)))
    return served


def _tsdevice_subprocess(argv):
    """tsdevice in a fresh process: its W1 device-loop settings apply (they cannot once the chain made a backend)."""
    subprocess.run([sys.executable, "-m", "beamfastsim.tsdevice", *argv], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker", help="serve a queue on this node")
    w.add_argument("--queue", required=True)
    w.add_argument("--batch", type=int, default=10)
    w.add_argument("--gather-s", type=float, default=90.0, help="wait this long for a full batch")
    w.add_argument("--idle-s", type=float, default=900.0, help="stop after this long without requests")
    w.add_argument("--wall-s", type=float, help="stop taking batches before this many seconds")
    w.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    w.add_argument("--pool", type=int)
    w.add_argument("--data", help="directory whose files' sha256 a request may pin (bank.npz, slab_bank.npz, ts.gdml)")
    for k in ("bank", "slab-bank", "ps-map", "cache", "gdml", "field-config", "offline-root", "maps", "ds-geom",
              "ds-map"):
        w.add_argument(f"--{k}", required=True)
    w.add_argument("--ds-offline-root")
    a = ap.parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if a.platform == "gpu" else "cpu")
    from beamfastsim import ptgt_chain_jax
    data_sha = {p.name: _sha256(p) for p in sorted(Path(a.data).iterdir()) if p.is_file()} if a.data else {}
    base = argparse.Namespace(platform=a.platform, pool=a.pool, bank=a.bank, slab_bank=a.slab_bank, ps_map=a.ps_map,
                              cache=a.cache, gdml=a.gdml, field_config=a.field_config, offline_root=a.offline_root,
                              maps=a.maps, ds_geom=a.ds_geom, ds_offline_root=a.ds_offline_root, ds_map=a.ds_map)
    served = worker(a.queue, base, a.batch, a.gather_s, a.idle_s, a.wall_s, ptgt_chain_jax.main,
                    _tsdevice_subprocess, data_sha, log=lambda s: print(s, flush=True))
    print(json.dumps(dict(served=served)), flush=True)


if __name__ == "__main__":
    main()
