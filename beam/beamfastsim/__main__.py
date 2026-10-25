"""beamfastsim CLI.

  python3 -m beamfastsim g4table --nts F.root [...] --out g4ref.npz
  python3 -m beamfastsim run --g4 g4ref.npz --out fast.npz [--n N] [--seed S] [--ds MM] [--batch B] [--no-msc]
  python3 -m beamfastsim point --g4 g4ref.npz --out DIR [--geom F] [--n N] [--repeat R]   (fastsimkit beam_point)
  python3 -m beamfastsim compare --g4 g4ref.npz --fast fast.npz --out DIR
  python3 -m beamfastsim fieldcheck            (field map lookups vs the text map at sampled grid points)

Inputs from CVMFS: Offline of the reference release (OFFLINE_ROOT, default MDC2025ac's v10_47_00) and the Mau13 maps.
"""
import argparse
import json
import os
import time

import numpy as np

CVMFS = "/cvmfs/mu2e.opensciencegrid.org"
OFFLINE_ROOT = os.environ.get("BEAMFS_OFFLINE_ROOT", f"{CVMFS}/Musings/Offline/v10_47_00")
GEOM_FILE = os.environ.get("BEAMFS_GEOM", "Offline/Mu2eG4/geom/geom_common.txt")
MAPS = os.environ.get("BEAMFS_MAPS", f"{CVMFS}/DataFiles/BFieldMaps/Mau13")
MAP = os.environ.get("BEAMFS_MAP", "DSMap")


def _geom(path=None):
    from . import geometry
    return geometry.from_offline(path or GEOM_FILE, OFFLINE_ROOT)


def cmd_g4table(a):
    from . import g4ref
    t = g4ref.read(a.nts, _geom(a.geom))
    g4ref.save(t, a.out)
    print(f"{len(t['evt'])} DS-stage muons, {np.sum(t['foil'] >= 0)} stopped in a foil -> {a.out}")


def cmd_reference_audit(a):
    from . import reference_audit
    report = reference_audit.audit(a.nts, table=a.g4, artifacts=a.artifact)
    with open(a.out, "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if report["duplicate_primary_ids"] or (a.g4 and not report["table"]["all_raw_columns_match"]):
        raise SystemExit(1)


def cmd_run(a):
    import jax
    from . import fieldmap, g4ref, transport
    g = g4ref.load(a.g4)
    n = min(a.n or len(g["evt"]), len(g["evt"]))
    geom = _geom(a.geom)
    prm = transport.Params(ds=a.ds, msc=not a.no_msc, decay=not a.no_decay, n_slices=a.slices, tcut=a.tcut)
    t0 = time.time()
    fm = fieldmap.to_device(fieldmap.load(f"{MAPS}/{MAP}.header"))
    t_load = time.time() - t0
    x = np.stack([g["x"], g["y"], g["z"]], 1)[:n]
    p3 = np.stack([g["px"], g["py"], g["pz"]], 1)[:n]
    key = jax.random.PRNGKey(a.seed)
    out = {k: [] for k in ("x", "p", "t", "fate", "foil", "nfoil")}
    timings = []
    for b0 in range(0, n, a.batch):
        b1 = min(n, b0 + a.batch)
        kb = jax.random.fold_in(key, b0)
        st = transport.initial_state(jax.random.fold_in(kb, 1), x[b0:b1], p3[b0:b1], g["time"][:n][b0:b1],
                                     g["q"][:n][b0:b1], prm)
        t1 = time.time()
        res, nst = transport.transport_all(fm, geom, prm, jax.random.fold_in(kb, 2), st, chunk=a.chunk)
        timings.append((b1 - b0, time.time() - t1, int(nst)))
        for k in out:
            out[k].append(np.asarray(getattr(res, k)))
    o = {k: np.concatenate(v) for k, v in out.items()}
    for r in range(a.repeat):   # warm passes over the same muons (compiled shapes reused); results not kept
        for b0 in range(0, n, a.batch):
            b1 = min(n, b0 + a.batch)
            kb = jax.random.fold_in(jax.random.fold_in(key, 1000 + r), b0)
            st = transport.initial_state(jax.random.fold_in(kb, 1), x[b0:b1], p3[b0:b1], g["time"][:n][b0:b1],
                                         g["q"][:n][b0:b1], prm)
            t1 = time.time()
            _, nst = transport.transport_all(fm, geom, prm, jax.random.fold_in(kb, 2), st, chunk=a.chunk)
            timings.append((b1 - b0, time.time() - t1, int(nst)))
    nb = -(-n // a.batch)
    warm = timings[nb:] if len(timings) > nb else timings
    rate = sum(m for m, _, _ in warm) / sum(s for _, s, _ in warm)
    meta = dict(n=n, device=str(jax.devices()[0]), field_load_s=t_load, first_batch_s=timings[0][1],
                first_pass_s=sum(s for _, s, _ in timings[:nb]), warm_passes=a.repeat, muons_per_s_warm=rate,
                geom=a.geom or GEOM_FILE, lane_steps=[s for _, _, s in timings], params=prm._asdict(), map=MAP)
    np.savez_compressed(a.out, **o, meta=json.dumps(meta))
    print(json.dumps(meta, indent=1))


def cmd_point(a):
    """run + stopping-target metrics for one geometry -> OUT/metrics.json (flat, for the fastsimkit beam_point task)."""
    os.makedirs(a.out, exist_ok=True)
    a.no_msc = a.no_decay = False
    a.slices, a.tcut, a.ds = 4, 1.0, 5.0
    fast = os.path.join(a.out, "fast.npz")
    out_dir = a.out
    a.out = fast
    cmd_run(a)
    from . import g4ref
    f = dict(np.load(fast))
    meta = json.loads(str(f["meta"]))
    g = g4ref.load(a.g4)
    n = len(f["fate"])
    geom = _geom(a.geom)
    stop = f["fate"] == 1
    nf = len(geom.foil_z)
    neg = g["q"][:n] < 0
    r = np.hypot(f["x"][stop, 0] - geom.axis_x, f["x"][stop, 1])
    m = dict(stop_frac=stop.mean(), stop_frac_err=np.sqrt(stop.mean() * (1 - stop.mean()) / n),
             stop_frac_mu_minus=stop[neg].mean(), stops=stop.sum(), muons=n, foils=nf,
             stops_foil_mean=f["foil"][stop].mean() if stop.any() else -1.0,
             stop_r_mean=r.mean() if stop.any() else -1.0,
             stop_frac_upstream_half=np.mean(f["foil"][stop] < nf / 2) if stop.any() else -1.0,
             muons_per_s_warm=meta["muons_per_s_warm"] if a.repeat else -1.0, first_pass_s=meta["first_pass_s"])
    m = {k: float(v) for k, v in m.items()}
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"))
    json.dump(dict(meta, foil_profile=np.bincount(f["foil"][stop], minlength=nf).tolist()),
              open(os.path.join(out_dir, "point_meta.json"), "w"), indent=1)
    os.remove(fast)
    print(json.dumps(m))


def cmd_compare(a):
    from . import compare, g4ref
    m = compare.run(g4ref.load(a.g4), dict(np.load(a.fast)), _geom(a.geom), a.out)
    print(json.dumps(m, indent=1))


def cmd_fieldcheck(a):
    from . import fieldmap
    import jax.numpy as jnp
    fm = fieldmap.load(f"{MAPS}/{MAP}.header")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, fm.n - 1, size=(2000, 3))
    frac = rng.random((2000, 3))
    pos = fm.lo + (idx + frac) * fm.d
    got = np.asarray(fieldmap.field(fieldmap.to_device(fm), jnp.asarray(pos)))
    # reference: explicit trilinear on the numpy grid (BFGridMap::interpolateTriLinear weights)
    ref = np.zeros_like(pos)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = np.prod(np.where([dx, dy, dz], frac, 1 - frac), axis=1)
                ref += w[:, None] * fm.B[idx[:, 0] + dx, idx[:, 1] + dy, idx[:, 2] + dz]
    print(f"max |dB| {np.abs(got - ref).max():.3e} T; |B| at the target centre "
          f"{np.linalg.norm(np.asarray(fieldmap.field(fieldmap.to_device(fm), jnp.asarray([[-3904., 0., 5871.]])))):.4f} T")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="beamfastsim")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("g4table"); s.add_argument("--nts", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--geom", default=None)
    s = sp.add_parser("run")
    s.add_argument("--g4", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=0); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--ds", type=float, default=5.0); s.add_argument("--batch", type=int, default=20000)
    s.add_argument("--slices", type=int, default=4); s.add_argument("--tcut", type=float, default=1.0)
    s.add_argument("--no-msc", action="store_true"); s.add_argument("--no-decay", action="store_true")
    s.add_argument("--chunk", type=int, default=200)
    s.add_argument("--repeat", type=int, default=0, help="warm passes for the throughput figure")
    s.add_argument("--geom", default=None, help="geometry file (absolute, or relative to the Offline root)")
    s = sp.add_parser("compare")
    s.add_argument("--g4", required=True); s.add_argument("--fast", required=True); s.add_argument("--out", required=True)
    s.add_argument("--geom", default=None)
    s = sp.add_parser("point")
    s.add_argument("--g4", required=True); s.add_argument("--out", required=True); s.add_argument("--geom", default=None)
    s.add_argument("--n", type=int, default=0); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--batch", type=int, default=131072); s.add_argument("--chunk", type=int, default=400)
    s.add_argument("--repeat", type=int, default=0)
    s = sp.add_parser("reference-audit", help="certify persisted reference coverage and raw table identity")
    s.add_argument("--nts", nargs="+", required=True); s.add_argument("--g4")
    s.add_argument("--artifact", action="append", default=[]); s.add_argument("--out", required=True)
    sp.add_parser("fieldcheck")
    a = ap.parse_args(argv)
    {"g4table": cmd_g4table, "run": cmd_run, "point": cmd_point, "compare": cmd_compare,
     "fieldcheck": cmd_fieldcheck, "reference-audit": cmd_reference_audit}[a.cmd](a)


if __name__ == "__main__":
    main()
