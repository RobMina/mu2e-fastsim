"""Opt-in adapter: G4beamline production-target runs -> TS-entrance references for the device chain (tsdevice).

A G4beamline Mu2E.in deck records charged particles at the virtual detector `Coll_01_DetIn`, the upstream face of
its first TS collimator. This module turns those records into the reference format `tsdevice --reference` reads
(`id`, `shard`, `x`, `p`, `t`, `pdg`, Mu2e frame, mm / MeV/c / ns), with the POT exposure per input file, and scores
saved tsdevice outputs on stops per POT.

Conventions (all checked against the deck, none fitted):

* Frame: the deck's world is the Mu2e frame shifted by (-3904, 0, +7929) mm (production target at G4beamline
  z = 1764.5, Mu2e z = -6164.5; TS axis at G4beamline x = 0, Mu2e x = 3904). Axes are parallel.
* Time: the deck starts the proton 0.5 mm in front of the target's beam-entry face at t = 0; the stock Mu2e POT stage
  also starts the proton at the target at t = 0 and applies the bunch time later. No offset is applied.
* Particles: mu+-, pi+- only (the device chain transports these), first inward crossing (Pz > 0, earliest t) per
  (file, EventID, TrackID), as the stock same-stage reference keeps the first forward VD1 state.
* Plane: the deck's collimator face sits at Mu2e z = -4044.3 (from the tree), the stock TS geometry starts at
  z = -4004 (tsgeometry `z_start`): the deck's PS-to-TS distance is 40.3 mm shorter than stock. `translate` (default)
  moves each state along z by the difference and keeps x, y, p, t: the state keeps its place relative to the
  collimator face. `helix` carries it through the stock field instead (vacuum, no material, RK4 in z), keeping the
  distance from the target. The two bracket the plane-matching uncertainty; report both.
* Exposure: POT per file = completed events of the job log ("Run complete N Events"); a file without a log needs an
  explicit `--events-per-file`.

The board's own objective (mu- + pi- crossings per POT, counted by the campaign) is recomputed from the same
selection for closure. Fields and geometry upstream of the plane are the deck's (its own field maps and PS), not
stock: a G4beamline point scored through the device chain is a G4beamline reference, never mixed with stock samples.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

FRAME_OFFSET = np.array([3904.0, 0.0, -7929.0])   # Mu2e = G4beamline + offset [mm]
TREE = "VirtualDetector/Coll_01_DetIn"
BRANCHES = ("EventID", "TrackID", "ParentID", "PDGid", "x", "y", "z", "Px", "Py", "Pz", "t")
SPECIES = (13, -13, 211, -211)
MASS = {13: 105.6583755, 211: 139.57039}             # MeV/c^2 (PDG 2022, as Geant4 11)
K_FIELD = 0.299792458                               # MeV/c per (T mm e)
C_MM_NS = 299.792458
MODES = ("translate", "helix")
STOP = 1                                            # tsdevice DS fate of a stopped muon (dsstage FATES)
_RUN_COMPLETE = re.compile(r"Run complete\s+(\d+)\s+Events")


def charge(pdg):
    """Particle charge [e] from the PDG code (mu- is +13, pi- is -211)."""
    pdg = np.asarray(pdg)
    return np.where(np.abs(pdg) == 211, np.sign(pdg), -np.sign(pdg)).astype(float)


def mass(pdg):
    return np.where(np.abs(np.asarray(pdg)) == 211, MASS[211], MASS[13])


def read_plane(path, tree=TREE):
    """The virtual-detector rows of one G4beamline ROOT file as float64 / int64 arrays (stored-row order)."""
    import uproot
    with uproot.open(path) as f:
        a = f[tree].arrays(list(BRANCHES), library="np")
    out = {k: np.asarray(a[k], dtype=np.float64) for k in ("x", "y", "z", "Px", "Py", "Pz", "t")}
    for k in ("EventID", "TrackID", "ParentID", "PDGid"):
        v = np.asarray(a[k])
        if v.dtype.kind == "f" and not np.all(v == np.round(v)):
            raise ValueError(f"{path}: {k} is not integral")
        out[k] = np.round(v).astype(np.int64)
    return out


def first_inward(rows):
    """Indices of the first inward (Pz > 0) mu/pi crossing per (EventID, TrackID), ordered by (EventID, TrackID)."""
    keep = np.flatnonzero(np.isin(rows["PDGid"], SPECIES) & (rows["Pz"] > 0))
    if not keep.size:
        return keep
    ev, tr, t = rows["EventID"][keep], rows["TrackID"][keep], rows["t"][keep]
    order = np.lexsort((np.arange(keep.size), t, tr, ev))
    ev, tr = ev[order], tr[order]
    first = np.r_[True, (ev[1:] != ev[:-1]) | (tr[1:] != tr[:-1])]
    return keep[order[first]]


def log_events(path):
    """Completed events of one G4beamline job log, or None when the run did not complete."""
    with open(path, "rb") as f:
        text = f.read().decode("latin-1")
    m = _RUN_COMPLETE.findall(text)
    return int(m[-1]) if m else None


def sequencer(name):
    """The 8-digit sequencer of a Mu2e file name (nts.owner.desc.conf.SEQ.root, log.owner.desc.conf.SEQ-job.log)."""
    m = re.search(r"\.(\d{8})(?:-\d+)?\.(?:root|log)$", Path(name).name)
    if not m:
        raise ValueError(f"no sequencer in {name}")
    return int(m.group(1))


def to_mu2e(x_g4bl):
    return np.asarray(x_g4bl, dtype=float) + FRAME_OFFSET


def _helix_rhs(field, x, p, q):
    """d(x, p, t)/dz in a static field, z as the independent variable (requires pz > 0)."""
    b = field(x)
    pz = p[:, 2]
    dx = np.stack([p[:, 0] / pz, p[:, 1] / pz, np.ones_like(pz)], axis=1)
    dp = (K_FIELD * q / pz)[:, None] * np.cross(p, b)
    return dx, dp


def helix_to_plane(x, p, t, pdg, z_target, field, step_mm=1.0, pz_floor=1.0):
    """Carry states from their z to z_target through `field` (B[N,3] tesla from x[N,3] mm), vacuum, RK4 in z.

    Returns x, p, t at the plane and a bool `returned` for lanes whose pz fell below pz_floor [MeV/c] on the way
    (they would not reach the plane forward; the caller drops them and counts them)."""
    x, p = np.array(x, dtype=float), np.array(p, dtype=float)
    t = np.array(t, dtype=float)
    q, m = charge(pdg), mass(pdg)
    dz_total = z_target - x[:, 2]
    if np.any(dz_total < 0):
        raise ValueError("helix_to_plane carries states downstream only")
    nstep = max(1, int(np.ceil(np.max(dz_total, initial=0.0) / step_mm)))
    h = dz_total / nstep
    returned = p[:, 2] < pz_floor
    for _ in range(nstep):
        live = ~returned
        if not live.any():
            break
        xl, pl, hl, ql = x[live], p[live], h[live][:, None], q[live]
        k1x, k1p = _helix_rhs(field, xl, pl, ql)
        k2x, k2p = _helix_rhs(field, xl + 0.5 * hl * k1x, pl + 0.5 * hl * k1p, ql)
        k3x, k3p = _helix_rhs(field, xl + 0.5 * hl * k2x, pl + 0.5 * hl * k2p, ql)
        k4x, k4p = _helix_rhs(field, xl + hl * k3x, pl + hl * k3p, ql)
        xn = xl + hl / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        pn = pl + hl / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        # |p| is conserved by the field: ds/dz = |p|/pz, trapezoidal in 1/pz (exact when pz is constant)
        pabs = np.linalg.norm(pl, axis=1)
        ds = hl[:, 0] * pabs * 0.5 * (1.0 / pl[:, 2] + 1.0 / np.maximum(pn[:, 2], pz_floor))
        t[live] += ds * np.hypot(pabs, m[live]) / (pabs * C_MM_NS)
        x[live], p[live] = xn, pn
        returned[live] |= pn[:, 2] < pz_floor
    x[:, 2] = np.where(returned, x[:, 2], z_target)
    return x, p, t, returned


def build(files, z_target, mode="translate", point=0, events=None, field=None, reader=read_plane):
    """Reference dict for tsdevice from the G4beamline files of one design point.

    files: ROOT paths (one per job); events: POT per file (list, same order); field: callable B(x) for `helix`.
    Rows are in (file, EventID, TrackID) order; `shard` is the file sequencer, `id` = (point, sequencer, EventID,
    TrackID)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "helix" and field is None:
        raise ValueError("helix needs the stock field")
    if events is None or len(events) != len(files):
        raise ValueError("one POT count per file")
    cols = {k: [] for k in ("id", "x", "p", "t", "pdg", "shard")}
    plane_z, n_returned, n_rows = [], 0, 0
    board_selected = 0
    seqs = []
    for path in files:
        seq = sequencer(path)
        seqs.append(seq)
        rows = reader(path)
        n_rows += len(rows["t"])
        i = first_inward(rows)
        x = to_mu2e(np.stack([rows["x"][i], rows["y"][i], rows["z"][i]], axis=1))
        p = np.stack([rows["Px"][i], rows["Py"][i], rows["Pz"][i]], axis=1)
        t, pdg = rows["t"][i], rows["PDGid"][i]
        board_selected += int(np.sum((pdg == 13) | (pdg == -211)))
        if len(t):
            plane_z.append(x[:, 2])
        if mode == "translate":
            x = x.copy()
            x[:, 2] = z_target
        else:
            x, p, t, ret = helix_to_plane(x, p, t, pdg, z_target, field)
            n_returned += int(ret.sum())
            keep = ~ret
            x, p, t, pdg, i = x[keep], p[keep], t[keep], pdg[keep], i[keep]
        n = len(t)
        cols["id"].append(np.stack([np.full(n, point), np.full(n, seq), rows["EventID"][i], rows["TrackID"][i]],
                                   axis=1).astype(np.int64))
        cols["x"].append(x)
        cols["p"].append(p)
        cols["t"].append(t)
        cols["pdg"].append(pdg.astype(np.int64))
        cols["shard"].append(np.full(n, seq, dtype=np.int64))
    out = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in cols.items()}
    zs = np.concatenate(plane_z) if plane_z else np.zeros(0)
    if zs.size and np.ptp(zs) > 1.0:
        raise ValueError(f"crossings span {np.ptp(zs):.3f} mm in z: not one plane")
    pot = np.asarray(events, dtype=np.int64)
    out["parent_shard_ids"] = np.stack([np.full(len(seqs), point), np.asarray(seqs)], axis=1).astype(np.int64)
    out["parent_shard_event_counts"] = pot
    meta = dict(mode=mode, point=int(point), z_target=float(z_target),
                source_plane_z=float(np.median(zs)) if zs.size else None, files=len(files), pot=int(pot.sum()),
                rows_read=int(n_rows), entrances=int(len(out["t"])), helix_returned=n_returned,
                species={str(s): int(np.sum(out["pdg"] == s)) for s in SPECIES},
                board_objective=board_selected / float(pot.sum()) if pot.sum() else None,
                board_selected=board_selected)
    return out, meta


def save(path, ref, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **ref)


def ratio_error(num, den):
    """Ratio of sums sum(num)/sum(den) and its error from the per-shard spread (n/(n-1) linearised estimator)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    r = num.sum() / den.sum()
    n = len(num)
    if n < 2:
        return r, float("nan")
    var = n / (n - 1) * np.sum((num - r * den) ** 2) / den.sum() ** 2
    return float(r), float(np.sqrt(var))


def score(reference, outputs, batch=None):
    """Stops per POT of one reference from tsdevice outputs (one per seed, same reference rows).

    batch: the reference's position in a tsdevice run over several references (its rows are those with this `batch`
    value). Stops per shard are averaged over the seeds, then the ratio to the shard POT carries the per-shard error
    (inputs and transport together)."""
    with np.load(reference) as d:
        shard_ids = d["parent_shard_ids"][:, 1]
        pot = d["parent_shard_event_counts"].astype(float)
        ref_id = d["id"]
        ref_meta = json.loads(str(d["meta"]))
    per_seed, seeds, dtypes = [], [], set()
    for path in outputs:
        with np.load(path) as o:
            rows = slice(None) if batch is None else o["batch"] == batch
            if not np.array_equal(o["id"][rows], ref_id):
                raise ValueError(f"{path}: rows differ from the reference")
            meta = json.loads(str(o["meta"]))
            stop = o["ds_fate"][rows] == STOP
            mu_minus = o["pdg_ds"][rows] == 13
            shard = o["shard"][rows]
        idx = np.searchsorted(shard_ids, shard)
        if not np.all(shard_ids[idx] == shard):
            raise ValueError(f"{path}: shard outside the exposure table")
        per_seed.append(np.stack([np.bincount(idx, weights=stop, minlength=len(pot)),
                                  np.bincount(idx, weights=stop & mu_minus, minlength=len(pot))]))
        seeds.append(meta["seed"])
        dtypes.add(meta["dtype"])
    if len(dtypes) != 1 or len(set(seeds)) != len(seeds):
        raise ValueError("outputs must share one dtype and use distinct seeds")
    mean = np.mean(per_seed, axis=0)
    stops, stops_err = ratio_error(mean[0], pot)
    mum, mum_err = ratio_error(mean[1], pot)
    seed_totals = [float(s[0].sum()) for s in per_seed]
    return dict(reference=str(reference), point=ref_meta["point"], mode=ref_meta["mode"], dtype=dtypes.pop(),
                seeds=seeds, pot=float(pot.sum()), entrances=ref_meta["entrances"],
                board_objective=ref_meta["board_objective"], stops_per_pot=stops, stops_per_pot_err=stops_err,
                mu_minus_stops_per_pot=mum, mu_minus_stops_per_pot_err=mum_err,
                stops_per_seed=seed_totals,
                seed_sd=float(np.std(seed_totals, ddof=1)) if len(seed_totals) > 1 else None)


def covered(bank):
    """B(x) of a tsfield bank that raises on a point outside every map (never a silent zero field)."""
    def field(x):
        b, ok = bank.evaluate(x)
        if not np.all(ok):
            raise ValueError(f"{int(np.sum(~ok))} points outside the field maps, e.g. {x[~ok][0]}")
        return b
    return field


def _field(a):
    from . import tsfield
    if not (a.field_config and a.offline_root and a.maps):
        raise SystemExit("helix needs --field-config, --offline-root and --maps")
    return covered(tsfield.from_config(a.field_config, a.offline_root, a.maps))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="G4beamline files of one point -> tsdevice reference NPZ")
    b.add_argument("--nts", nargs="+", required=True, help="G4beamline ROOT files of one design point")
    b.add_argument("--logs", nargs="*", default=(), help="job logs (matched to files by sequencer)")
    b.add_argument("--events-per-file", type=int, help="POT per file when a log is missing (never silently assumed)")
    b.add_argument("--point", type=int, default=0)
    b.add_argument("--gdml", help="stock TS GDML: the target plane is its z_start")
    b.add_argument("--plane-z", type=float, help="target plane [mm, Mu2e] (instead of --gdml)")
    b.add_argument("--mode", choices=MODES, default="translate")
    b.add_argument("--field-config")
    b.add_argument("--offline-root")
    b.add_argument("--maps")
    b.add_argument("--out", required=True)
    s = sub.add_parser("score", help="stops per POT from tsdevice outputs (one file per seed) of the references, "
                                     "in the order tsdevice --reference read them; JSON lines, one per reference")
    s.add_argument("--references", nargs="+", required=True)
    s.add_argument("--outputs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if (a.gdml is None) == (a.plane_z is None):
            ap.error("give exactly one of --gdml, --plane-z")
        if a.gdml:
            from .tsgeometry import Geometry
            z_target = Geometry(a.gdml).z_start
        else:
            z_target = a.plane_z
        files = sorted(a.nts, key=sequencer)
        logs = {sequencer(p): p for p in a.logs}
        events = []
        for f in files:
            n = log_events(logs[sequencer(f)]) if sequencer(f) in logs else None
            if n is None:
                if a.events_per_file is None:
                    ap.error(f"no completed log for {f}: give --events-per-file")
                n = a.events_per_file
            events.append(n)
        field = _field(a) if a.mode == "helix" else None
        ref, meta = build(files, z_target, a.mode, a.point, events, field)
        meta["log_pot_files"] = sum(sequencer(f) in logs for f in files)
        save(a.out, ref, meta)
        print(json.dumps(meta, indent=1))
    else:
        many = len(a.references) > 1
        res = [score(r, a.outputs, batch=k if many else None) for k, r in enumerate(a.references)]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in res))
        for r in res:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
