"""Between the GPU stages and the writer (NumPy): fast record + trigger decisions + donor library -> write plan.

Per fast event the plan holds the donor (row of the donor library: file and entry), the rotation about the tracker
axis and the time shift that bring the donor's kinematics to the fast electron, and every value the fast chain
supplies: the reco electron track at TT_Front / TT_Mid / TT_Back (positions, momenta, times and LoopHelix parameters),
its truth at the same surfaces and at birth, the calorimeter clusters and the track-cluster match, the covered
trigger decisions and the straw-digi count.

Reco surfaces. At TT_Front the reco state is the true entry position with the response layer's momentum magnitude
and tan(dip) and the true azimuth of the momentum, at the response layer's t0. At TT_Mid and TT_Back it is the true
state there (from the fast transport) with the same residuals applied: |p| scaled by p_reco / p_true(front), tan(dip)
shifted by tandip_reco - tandip_true(front), time shifted by t0 - t_true(front). LoopHelix parameters follow from the
state in the local field, as the stock InfoStructHelper does per segment: rad = pT / (0.3 Bz), lam = pz / (0.3 Bz),
centre on the bending side, phi0 = phi_mom - z / lam (wrapped), t0 = t - z / v_z, d0 = |r_c - rad|,
maxr = r_c + rad, tanDip = pz / pT.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import donor as D

KB = 0.299792458
ME = 0.51099895
C_MM_NS = 299.792458
MBTIME = 1695.0             # ns: the microbunch; stock reco times of a later microbunch fold back by it (MC times do not)
E_MATCH = 10.0               # MeV: a fast cluster on the contact disk above this is matched to the track
PLAN = "centuple_plan.npz"


def _wrap(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def reco_state(x_true, p_true, t_true, p_scale, dtan, dt):
    """Reco state at a surface from the true state there and the front residuals (see module docstring)."""
    pt = np.hypot(p_true[:, 0], p_true[:, 1])
    tan_r = p_true[:, 2] / np.where(pt > 0, pt, np.nan) + dtan
    pm = np.linalg.norm(p_true, axis=1) * p_scale
    pt_r = pm / np.sqrt(1 + tan_r * tan_r)
    phi = np.arctan2(p_true[:, 1], p_true[:, 0])
    p = np.stack([pt_r * np.cos(phi), pt_r * np.sin(phi), pt_r * tan_r], axis=1)
    return x_true.copy(), p, t_true + dt


def loophelix(x, p, t, bz, q=-1.0):
    """LoopHelix parameters of states (detector frame) in the local field bz: dict of rad, lam, cx, cy, phi0, t0,
    d0, maxr, tanDip."""
    pt = np.hypot(p[:, 0], p[:, 1])
    pm = np.linalg.norm(p, axis=1)
    b = np.abs(bz)
    rad = pt / (KB * b)
    lam = p[:, 2] / (KB * b)
    s = np.sign(q * bz)
    cx = x[:, 0] + s * rad * p[:, 1] / pt
    cy = x[:, 1] - s * rad * p[:, 0] / pt
    rc = np.hypot(cx, cy)
    vz = p[:, 2] / np.sqrt(pm * pm + ME * ME) * C_MM_NS
    return dict(rad=rad, lam=lam, cx=cx, cy=cy, phi0=_wrap(np.arctan2(p[:, 1], p[:, 0]) - x[:, 2] / lam),
                t0=t - x[:, 2] / vz, d0=np.abs(rc - rad), maxr=rc + rad, tanDip=p[:, 2] / pt)


def classes(rec):
    q = rec["trk_qual"].astype(bool)
    f = rec["trk_fit"].astype(bool)
    return np.where(q, D.CLASS_Q, np.where(f, D.CLASS_F, D.CLASS_N)).astype(np.int8)


# Stock calorimeter-hit inclusion of the track fit (KinKal MinCaloClusterEnergy 10 MeV, MaxCaloClusterDOCA 100 mm,
# MaxCaloClusterDt 5 ns): P(trkcalohit | the event has a cluster above 10 MeV) in bins of the leading cluster energy,
# measured on the stock development EventNtuples (default-selected e- tracks, 21 files, 36,672 events; 0.961 overall).
CALO_EFF_EDGES = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, np.inf])
CALO_EFF = np.array([0.559, 0.601, 0.682, 0.785, 0.826, 0.934, 0.989, 0.999, 0.997])


def calo_match(rec, run=None, subrun=None, event=None, seed=None, rule="stock"):
    """Track-cluster match of the fast electron: (has, cluster index).

    rule "stock" (default): the leading cluster above E_MATCH on the disk of the first contact, else the leading one on
    the other disk (stock matches either disk), kept with the stock inclusion efficiency of its energy (CALO_EFF, one
    draw per event from SeedSequence([seed, 3, run, subrun, event])). rule "contact": the leading cluster above E_MATCH on
    the contact disk only, always kept (the rule before)."""
    valid, energy, disk = rec["cl_valid"], rec["cl_energy"], rec["cl_disk"]
    big = valid & (energy > E_MATCH)
    on = big & (disk == rec["calo_disk"][:, None]) & (rec["calo_kind"][:, None] > 0)
    j_on = np.argmax(np.where(on, energy, -1.0), axis=1)
    if rule == "contact":
        return on.any(axis=1), j_on
    if rule != "stock":
        raise ValueError("calo_match rule: 'stock' or 'contact'")
    j_any = np.argmax(np.where(big, energy, -1.0), axis=1)
    j = np.where(on.any(axis=1), j_on, j_any)
    has = big.any(axis=1)
    e = energy[np.arange(len(j)), j]
    b = np.clip(np.searchsorted(CALO_EFF_EDGES, e, side="right") - 1, 0, len(CALO_EFF) - 1)
    u = np.array([np.random.default_rng(np.random.SeedSequence([int(seed), 3, int(r), int(s), int(v)])).random()
                  for r, s, v in zip(run, subrun, event)]) if len(j) else np.zeros(0)
    return has & (u < CALO_EFF[b]), j


def cluster_truth_time(ttrue, ctime, contact_t, valid):
    """caloclustersmc.tprimary per cluster: the earliest true deposit (the contact time where none), folded by the
    microbunch as the cluster's reco time is (stock folds the calorimeter MC times like the reco times)."""
    tt = np.where(np.isfinite(ttrue), ttrue, np.asarray(contact_t, float)[:, None])
    kc = np.round(np.nan_to_num(tt - ctime) / MBTIME)
    return np.where(valid, tt - MBTIME * kc, 0.0)


def build(record, donor_lib, out, *, trigger=None, seed=20261111, trigger_cal=None):
    """Write plan (npz) for every event of the record. Returns the path and counters."""
    rec = dict(np.load(record))
    meta = json.loads(str(rec.pop("meta")))
    lib = D.load(donor_lib)
    n = rec["event"].size
    run, sub, evn = rec["run"], rec["subrun"], rec["event"]
    covered = [str(p) for p in lib["covered_paths"]]
    trig = {}
    trig_meta = None
    if trigger is not None:
        tz = dict(np.load(trigger))
        trig_meta = json.loads(str(tz.pop("meta")))
        if not (np.array_equal(tz["event"], evn) and np.array_equal(tz["subrun"], sub)):
            raise ValueError("centuple: trigger file events differ from the record's")
        trig = {k[5:]: v.astype(bool) for k, v in tz.items() if k.startswith("trig_")}
        if trigger_cal is not None:      # opt-in on-spill response of the device paths (trigcal)
            from . import trigcal
            names = sorted(trig)
            raw = np.stack([trig[k] for k in names], 1)
            new = trigcal.apply(trigcal.load(trigger_cal), names, raw, classes(rec), rec["birth_t"], run, sub, evn,
                                seed)
            trig = {k: new[:, j] for j, k in enumerate(names)}
            trig_meta = dict(trig_meta, trigger_cal=str(trigger_cal),
                             raw_pass={k: int(raw[:, j].sum()) for j, k in enumerate(names)},
                             cal_pass={k: int(new[:, j].sum()) for j, k in enumerate(names)})
    missing = [p for p in covered if p not in trig]
    cov_paths = [p for p in covered if p in trig]
    cov = np.zeros(n, bool)
    for p in cov_paths:
        cov |= trig[p]
    cls = classes(rec)
    pf_true = rec["front_p"]
    pt_true = np.hypot(pf_true[:, 0], pf_true[:, 1])
    tan_true = pf_true[:, 2] / np.where(pt_true > 0, pt_true, np.nan)
    has_cl, jcl = calo_match(rec, run, sub, evn, seed)
    e_m = np.where(has_cl, rec["cl_energy"][np.arange(n), jcl], np.nan)
    keys = dict(tbirth=rec["birth_t"], tandip_t=tan_true, nactive=rec["trk_nactive"], fitcon=rec["trk_fitcon"], calo=has_cl,
                eop=e_m / rec["trk_p_reco"],
                ecl=np.where(rec["cl_valid"].any(axis=1), np.max(np.where(rec["cl_valid"], rec["cl_energy"], 0), 1), 0))
    dfile, drow = D.match(lib, cls, cov, keys, seed, run, sub, evn)
    # trkpid of the electron track: its own draw among class-Q donors with the same match status, nearest in E/p
    _, prow = D.match(lib, np.full(n, D.CLASS_Q, np.int8), np.ones(n, bool), keys, seed, run, sub, evn,
                      use={D.CLASS_Q: ("calo", "eop")}, stream=1)
    track = (cls == D.CLASS_Q) | (cls == D.CLASS_F)     # a fitted e- track (the response layer's observables)
    pid = np.where(track, lib["pid"][prow], np.nan)
    # rotation and time shift donor -> fast (truth at TT_Front)
    phi_f = np.arctan2(pf_true[:, 1], pf_true[:, 0])
    dphi = np.where(np.isfinite(phi_f) & np.isfinite(lib["phi_t"][drow]), _wrap(phi_f - lib["phi_t"][drow]), 0.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    dphi = np.where(np.isfinite(phi_f) & np.isfinite(lib["phi_t"][drow]), dphi, rng.uniform(-np.pi, np.pi, n))
    dt = np.where(np.isfinite(rec["front_t"]) & np.isfinite(lib["t_t"][drow]), rec["front_t"] - lib["t_t"][drow], 0.0)
    # microbunch folds of the reco times (truth entry time of the electron, fast and donor)
    k_f = np.where(np.isfinite(rec["front_t"]), np.floor(np.nan_to_num(rec["front_t"]) / MBTIME), 0.0)
    k_d = np.where(np.isfinite(lib["t_t"][drow]), np.floor(np.nan_to_num(lib["t_t"][drow]) / MBTIME), 0.0)
    dt_reco = np.where(dt != 0.0, dt - MBTIME * (k_f - k_d), 0.0)
    plan = dict(run=run, subrun=sub, event=evn, cls=cls, donor_file=dfile, donor_entry=lib["entry"][drow],
                donor_row=drow, donor_cls=lib["cls"][drow], donor_prim=lib["prim"][drow], dphi=dphi, dt=dt,
                dt_reco=dt_reco, fold=k_f, pid=pid,
                nactive=rec["trk_nactive"], fitcon=rec["trk_fitcon"], chi2ndf=rec["trk_chi2ndf"],
                has_entry=rec["has_entry"], birth_x=rec["birth_x"], birth_p=rec["birth_p"], birth_t=rec["birth_t"])
    # reco surfaces of the fitted electrons (class Q and F)
    p_true_f = np.linalg.norm(pf_true, axis=1)
    scale = rec["trk_p_reco"] / p_true_f
    dtan = rec["trk_tandip"] - tan_true
    dt0 = rec["trk_t0"] - rec["front_t"] - MBTIME * k_f        # reco times fold by the microbunch
    for name in ("front", "mid", "back"):
        found = rec["has_entry"] if name == "front" else rec[name + "_found"]
        x, p, t = reco_state(rec[name + "_x"], rec[name + "_p"], rec[name + "_t"], scale, dtan, dt0)
        ok = found & track & np.all(np.isfinite(p), axis=1)
        plan[f"seg_{name}_ok"] = ok
        plan[f"seg_{name}_pos"], plan[f"seg_{name}_mom"], plan[f"seg_{name}_time"] = x, p, t
        lh = loophelix(x, p, t, rec[name + "_bz"])   # t already folded (dt0), so lh t0 is too
        if name == "front":                 # the response layer's drawn helix quantities at the front
            lh["d0"], lh["maxr"], lh["tanDip"] = rec["trk_d0"], rec["trk_rmax"], rec["trk_tandip"]
        for k, v in lh.items():
            plan[f"lh_{name}_{k}"] = v
        plan[f"mc_{name}_ok"] = found & np.all(np.isfinite(rec[name + "_p"]), axis=1)
        plan[f"mc_{name}_pos"], plan[f"mc_{name}_mom"], plan[f"mc_{name}_time"] = (rec[name + "_x"], rec[name + "_p"],
                                                                                   rec[name + "_t"])
    # calorimeter
    for k in ("valid", "energy", "time", "x", "y", "z", "size", "disk", "split", "edge"):
        plan["cl_" + k] = rec["cl_" + k]
    if "cl_etrue" in rec:
        # cluster truth per crystal: the visible energy of the sampled shower steps in the cluster's hit crystals and
        # the earliest of their times (inner-edge clusters: the table's visible energy and the contact time)
        plan["cl_etot"] = np.where(rec["cl_valid"], rec["cl_etrue"], 0.0)
        plan["cl_tprim"] = cluster_truth_time(rec["cl_ttrue"], rec["cl_time"], rec["calo_contact_t"], rec["cl_valid"])
    else:
        # records without it: the drawn shower's visible energy, shared by the clusters in proportion to their energy
        ev_cl = np.where(rec["cl_valid"], rec["cl_energy"], 0.0)
        tot = ev_cl.sum(axis=1, keepdims=True)
        plan["cl_etot"] = np.where(tot > 0, rec["calo_etrue"][:, None] * ev_cl / np.where(tot > 0, tot, 1.0), 0.0)
        plan["cl_tprim"] = np.where(rec["cl_valid"], rec["calo_contact_t"][:, None], 0.0)
    plan["calo_match"] = has_cl & track
    plan["calo_match_cluster"] = jcl
    plan["calo_contact_x"], plan["calo_contact_t"] = rec["calo_contact_x"], rec["calo_contact_t"]
    plan["calo_disk"] = rec["calo_disk"]
    # trigger: covered decisions (names in trig_names, matrix [n, paths])
    plan["trig_names"] = np.array(cov_paths)
    plan["trig"] = np.stack([trig[p] for p in cov_paths], axis=1) if cov_paths else np.zeros((n, 0), bool)
    plan["nsd"] = rec.get("nsd", np.full(n, -1, np.int64))
    plan["pbtime_mc"] = rec.get("pbtime_mc", np.full(n, np.nan))
    plan["pbtime_reco"] = rec.get("pbtime_reco", np.full(n, np.nan))
    pmeta = dict(stage="assemble", seed=int(seed), record=meta, trigger=trig_meta, donor=str(donor_lib),
                 donor_name=str(lib.get("__name__", D.NAME)), covered_paths=cov_paths, covered_missing=missing,
                 e_match_mev=E_MATCH, calo_match_rule="stock", donor_files=[str(f) for f in lib["file_names"]],
                 donor_sha256=[str(f) for f in lib["file_sha256"]],
                 classes={c: int((cls == v).sum()) for c, v in (("N", 0), ("F", 1), ("Q", 2))})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / PLAN, meta=json.dumps(pmeta, default=str), **plan)
    return out / PLAN, pmeta
