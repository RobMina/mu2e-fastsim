"""One production-target design point: device target + PS chain -> TS-entrance states (cached per design) -> tsdevice
-> stops per POT (opt-in; the fastsimkit task `ptgt_point`).

The rod family of the G4beamline deck (Use_Proton_Target=4): length Tlength and outer radii R_up (the end the protons
leave), R_mid, R_dn (the end they enter), linear between, on the deck's placement. The PS stage is the validated
configuration of the device model (ENVELOPE, SLABS below: absorbing PS envelope with the HRS bore, the three TS-window
steel rings as thin slabs, the TS-entrance plane), so a point differs from another only by its rod.

    python3 -m beamfastsim.ptgt_point --rod TLENGTH R_UP R_MID R_DN --pot N --seed S --dtype float64 --platform cpu \
        --bank BANK.npz --slab-bank SLAB.npz --ps-map PSMap.header --cache DIR \
        --gdml TS.gdml --field-config CFG --offline-root ROOT --maps DIR --ds-geom GEOM --ds-map DSMap.header --out DIR

The TS-entrance states are cached under --cache by `design_key` (rod, POT, seed, dtype, the banks' content hashes, the
field map's sha256, the PS configuration and CACHE_VERSION): a repeated point skips the chain. Writes OUT/metrics.json
(numbers only) and OUT/point.json (the full record).
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

CACHE_VERSION = 1     # raise when the chain's output for the same inputs changes
ENVELOPE = dict(env_z=(-9929, -7858.01, -7858, -6658, -4068, -4067.99, -4000),
                env_r=(750, 750, 650, 200, 200, 249.8, 249.8), z_min=-8368.0, r_plane=249.3, z_plane=-4044.45)
SLABS = ((-4147.3, 158.8, 174.6, 6.35), (-4140.7, 158.8, 174.6, 6.35), (-4134.4, 158.8, 174.6, 6.35))
POOL = {"cpu": 16384, "gpu": 262144}
TS_SEED = 1
TS_SIGNAL = (13, -211)        # mu- and pi- at the TS entrance (the board objective)


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def field_map_sha(header):
    """sha256 of a field map .header and the .bin next to it (what the chain reads)."""
    h = Path(header)
    return hashlib.sha256((_sha256(h) + _sha256(h.with_suffix(".bin"))).encode()).hexdigest()


def design_key(rod, pot, seed, dtype, bank_hash, slab_hash, map_sha, envelope=ENVELOPE, slabs=SLABS):
    """Cache key of the TS-entrance states of one design: everything the chain's output depends on."""
    from mu2efastsim.library import hashing
    return hashing.digest(dict(version=CACHE_VERSION, rod=[round(float(v), 6) for v in rod], pot=int(pot),
                               seed=int(seed), dtype=str(dtype), bank=bank_hash, slab_bank=slab_hash, map=map_sha,
                               envelope={k: list(v) if isinstance(v, tuple) else v for k, v in envelope.items()},
                               slabs=[list(s) for s in slabs]))


def chain_argv(a, out):
    """ptgt_chain_jax arguments for one rod (the validated PS configuration)."""
    e = ENVELOPE
    argv = ["--bank", a.bank, "--map", a.ps_map, "--env-z", *map(str, e["env_z"]), "--env-r", *map(str, e["env_r"]),
            "--z-min", str(e["z_min"]), "--r-plane", str(e["r_plane"]), "--z-plane", str(e["z_plane"]),
            "--rod", *map(str, a.rod), "--point", "0", "--pot", str(a.pot), "--warm-pot", "0",
            "--pool", str(a.pool or POOL[a.platform]), "--steps", "64", "--dtype", a.dtype, "--seed", str(a.seed),
            "--slab-bank", a.slab_bank, "--out", str(out)]
    for s in SLABS:
        argv += ["--slab", *map(str, s)]
    return argv


def tsdevice_argv(a, reference, out):
    return ["--reference", str(reference), "--gdml", a.gdml, "--field-config", a.field_config,
            "--offline-root", a.offline_root, "--maps", a.maps, "--ds-geom", a.ds_geom,
            "--ds-offline-root", a.ds_offline_root or a.offline_root, "--ds-map", a.ds_map,
            "--seed", str(TS_SEED), "--dtype", a.dtype, "--out", str(out)]


def states(a, key, run_chain):
    """(reference path, cached, chain seconds): the cached TS-entrance states of this design, made if missing."""
    d = Path(a.cache) / key
    ref = d / "reference.npz"
    if ref.exists() and (d / "key.json").exists():
        return ref, True, 0.0
    tmp = Path(a.cache) / f".{key}.{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    run_chain(chain_argv(a, tmp / "reference.npz"))
    dt = time.perf_counter() - t0
    (tmp / "key.json").write_text(json.dumps(dict(key=key, rod=list(a.rod), pot=a.pot, seed=a.seed, dtype=a.dtype,
                                                  chain_s=dt), indent=1))
    try:
        os.replace(tmp, d)            # atomic publish; a concurrent writer of the same design wins harmlessly
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return ref, False, dt


def metrics(reference, ts_out, cached, chain_s, ts_s):
    """The point's numbers: stops per POT (Poisson error: the reference is one shard), the TS-entrance yields."""
    import numpy as np
    with np.load(reference) as r:
        pot = float(r["parent_shard_event_counts"].sum())
        pdg = r["pdg"]
    with np.load(ts_out) as o:
        stop = o["ds_fate"] == 1
        mu_minus = o["pdg_ds"] == 13
    n = float(stop.sum())
    sig = float(np.isin(pdg, TS_SIGNAL).sum())
    return dict(stops_per_pot=n / pot, stops_per_pot_err=max(n, 1.0) ** 0.5 / pot,
                mu_minus_stops_per_pot=float((stop & mu_minus).sum()) / pot, stops=n, pot=pot,
                ts_entrances=float(len(pdg)), ts_mu_minus_pi_minus_per_pot=sig / pot,
                ts_mu_minus_pi_minus_per_pot_err=max(sig, 1.0) ** 0.5 / pot, ts_mu_pi_per_pot=len(pdg) / pot,
                cached=float(cached), chain_s=float(chain_s), ts_s=float(ts_s))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rod", nargs=4, type=float, required=True, metavar=("TLENGTH", "R_UP", "R_MID", "R_DN"))
    ap.add_argument("--pot", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--pool", type=int, help="device lanes (default: POOL of the platform)")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--slab-bank", required=True)
    ap.add_argument("--ps-map", required=True, help="PS field map .header (the .bin next to it)")
    ap.add_argument("--cache", required=True, help="directory of the cached TS-entrance states")
    ap.add_argument("--gdml", required=True)
    ap.add_argument("--field-config", required=True)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--ds-geom", required=True)
    ap.add_argument("--ds-offline-root")
    ap.add_argument("--ds-map", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.pot < 1 or min(a.rod) <= 0:
        ap.error("need --pot >= 1 and a positive rod")
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if a.platform == "gpu" else "cpu")   # before jax is imported
    from beamfastsim import ptgt_chain_jax, ptgt_device, tsdevice
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bank_hash = ptgt_device.Bank.load(a.bank).lib.content_hash
    slab_hash = ptgt_device.Bank.load(a.slab_bank).lib.content_hash
    key = design_key(a.rod, a.pot, a.seed, a.dtype, bank_hash, slab_hash, field_map_sha(a.ps_map))
    ref, cached, chain_s = states(a, key, ptgt_chain_jax.main)
    t0 = time.perf_counter()
    tsdevice.main(tsdevice_argv(a, ref, out / "tsdevice.npz"))
    ts_s = time.perf_counter() - t0
    m = metrics(ref, out / "tsdevice.npz", cached, chain_s, ts_s)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    (out / "point.json").write_text(json.dumps(dict(key=key, rod=list(a.rod), pot=a.pot, seed=a.seed, dtype=a.dtype,
                                                    platform=a.platform, bank=bank_hash, slab_bank=slab_hash,
                                                    metrics=m), indent=1))
    (out / "tsdevice.npz").unlink()   # per-muon output: the point keeps its numbers only
    print(json.dumps(m))


if __name__ == "__main__":
    main()
