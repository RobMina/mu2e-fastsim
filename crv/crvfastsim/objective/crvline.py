"""crvline -- CRV-only straight lines through EX, T1, T2 and the muon taggers (plan docs/crv_only_alignment_plan.md §2.1).

Per event: the largest-PE EX cluster, plus in each other sector (T1, T2, taggers) the largest-PE cluster within
DT_MATCH of it. Hits are that cluster's FIRST pulses (sequenceIndex == 0; secondary pulses are dropped in data and
MC alike). Two independent weighted straight-line fits in y (the stacking normal of every sector):

    z(y) = z0 + sz * (y - Y_REF)   from EX/T2 bar centres (stacking z)
    x(y) = x0 + sx * (y - Y_REF)   from T1 and tagger bar centres (stacking x) and EX/T2 along-bar x from timing

A hit's stacking coordinate is its bar centre (crvpulses.pos); every layer of every body gets total weight
1/SIG_STACK^2, shared by its pulses in proportion to PE. EX/T2 clusters with end-to-end timing add one along-bar
point (crvcoincs.pos x at the cluster y) with weight 1/SIG_LONG^2. Coordinates are EventNtuple's detector frame
(kppgeom: origin at the tracker centre; the taggers sit at x -395..+395 mm).

Home of this module: crvfastsim.objective (moved from crvextrap.crvline, which re-exports it).

All work is on flat numpy arrays with an event index (np.bincount sums), no per-event loops.

Bodies (alignment units): EX M0-M3 = 0-3, T1 M0-M1 = 4-5, T2 M0-M1 = 6-7, taggers M1-M8 = 8-15.
"""
import awkward as ak
import numpy as np

from . import bars as G

SIG_STACK = 51.3 / np.sqrt(12.0)
SIG_LONG = 150.0
V_FIBER = 140.0                      # mm/ns, CrvCoincidenceFinder.fiberSignalSpeed
DT_MATCH = 20.0
Y_REF = 4431.0                       # EX mid-plane, mm (EX layers 4387..4475)
SLOT_NAME = G.SLOT_NAME
BODY_NAMES = G.BODY_NAMES
N_BODY = G.N_BODY

COINC_BR = ["crvcoincs.sectorType", "crvcoincs.time", "crvcoincs.PEs", "crvcoincs.hitPosAndTimeCalculated",
            "crvcoincs.pos.fCoordinates.fX", "crvcoincs.pos.fCoordinates.fY", "crvcoincs.nLayers", "crvcoincs.nHits",
            "crvcoincs.PEsPerLayer[4]"]
COINC_PE = 8.0           # coincidence PE threshold per layer (epilog_extracted_v04)
PULSE_BR = ["crvpulses.barId", "crvpulses.PEs", "crvpulses.crvHitIndex", "crvpulses.sequenceIndex",
            "crvpulses.SiPMId", "crvpulses.time",
            "crvpulses.pos.fCoordinates.fX", "crvpulses.pos.fCoordinates.fY", "crvpulses.pos.fCoordinates.fZ"]
TRUTH_BR = ["crvpulsesmc.pdgId", "crvpulsesmc.pos.fCoordinates.fX", "crvpulsesmc.pos.fCoordinates.fY",
            "crvpulsesmc.pos.fCoordinates.fZ"]


def _choose_clusters(c, pe_scale=None):
    """Per event, the chosen cluster index per sector slot (-1 if none): EX = max PE, others = max PE in time."""
    st, t, pe = c["crvcoincs.sectorType"], c["crvcoincs.time"], c["crvcoincs.PEs"]
    idx = ak.local_index(st)
    pe_ex = ak.where(st == 1, pe, -1.0)
    iex = ak.argmax(pe_ex, axis=1, keepdims=True)
    has_ex = ak.to_numpy(ak.fill_none(ak.firsts(pe_ex[iex]), -1.0) > 0)
    t_ex = ak.fill_none(ak.firsts(t[iex]), np.nan)
    chosen = {0: np.where(has_ex, ak.to_numpy(ak.fill_none(ak.firsts(idx[iex]), -1)), -1)}
    for code, slot in ((2, 1), (3, 2), (4, 3)):
        ok = (st == code) & (abs(t - t_ex) < DT_MATCH)
        f = (pe_scale or {}).get(SLOT_NAME[slot])
        if f is not None:      # light-yield emulation: the cluster survives only if every layer stays above threshold
            ok = ok & (ak.min(c["crvcoincs.PEsPerLayer[4]"], axis=-1) * f >= COINC_PE)
        pe_s = ak.where(ok, pe, -1.0)
        i = ak.argmax(pe_s, axis=1, keepdims=True)
        good = ak.to_numpy(ak.fill_none(ak.firsts(pe_s[i]), -1.0) > 0)
        chosen[slot] = np.where(good & has_ex, ak.to_numpy(ak.fill_none(ak.firsts(idx[i]), -1)), -1)
    return chosen, has_ex


def _wls(evt, y, u, w, nev):
    """Weighted LSQ u = a + b (y - Y_REF) per event. Returns a, b, chi2, nhit, ok (2 distinct y)."""
    yy = y - Y_REF
    S = np.bincount(evt, w, nev)
    Sy = np.bincount(evt, w * yy, nev)
    Syy = np.bincount(evt, w * yy * yy, nev)
    Su = np.bincount(evt, w * u, nev)
    Syu = np.bincount(evt, w * yy * u, nev)
    D = S * Syy - Sy * Sy
    ok = D > 1e-9 * np.maximum(S * Syy, 1e-300)
    Dz = np.where(ok, D, 1.0)
    b = np.where(ok, (S * Syu - Sy * Su) / Dz, np.nan)
    a = np.where(ok, (Syy * Su - Sy * Syu) / Dz, np.nan)
    r = u - (a[evt] + b[evt] * yy)
    chi2 = np.bincount(evt, w * r * r, nev)
    nhit = np.bincount(evt, np.ones_like(w), nev)
    return a, b, chi2, nhit, ok


def _features(c, chosen, flat, evt, counts, nev):
    """Per-event diagnostics for line cleaning: cluster counts per sector (all, and within DT_MATCH of the EX cluster),
    chosen-cluster time, PE, nLayers and nHits per slot, pulses per chosen cluster vs its nHits (crvHitIndex check)."""
    st, t = c["crvcoincs.sectorType"], c["crvcoincs.time"]
    idx = ak.local_index(st)
    out = {}
    t_ex = ak.to_numpy(ak.fill_none(ak.firsts(t[idx == chosen[0][:, None]]), np.nan))
    for code, slot in ((1, 0), (2, 1), (3, 2), (4, 3)):
        name = SLOT_NAME[slot]
        m = st == code
        out[f"n_{name}"] = ak.to_numpy(ak.sum(m, axis=1))
        out[f"nin_{name}"] = ak.to_numpy(ak.sum(m & (abs(t - t_ex[:, None]) < DT_MATCH), axis=1))
        pick = idx == chosen[slot][:, None]
        for k, br in (("t", "crvcoincs.time"), ("pe", "crvcoincs.PEs"), ("nl", "crvcoincs.nLayers"),
                      ("nh", "crvcoincs.nHits")):
            out[f"{k}_{name}"] = ak.to_numpy(ak.fill_none(ak.firsts(c[br][pick]), np.nan)).astype(np.float64)
        # pulses tagged to the chosen cluster (crvHitIndex), all and first pulses only
        slot_p, _, _ = G.bar_slot_body_layer(flat["crvpulses.barId"])
        hp = (flat["crvpulses.crvHitIndex"] == chosen[slot][evt]) & (chosen[slot][evt] >= 0)
        out[f"np_{name}"] = np.bincount(evt[hp], minlength=nev)
        out[f"np1_{name}"] = np.bincount(evt[hp & (flat["crvpulses.sequenceIndex"] == 0)], minlength=nev)
        out[f"npwrong_{name}"] = np.bincount(evt[hp & (slot_p != slot)], minlength=nev)
    return out


CHUNK = 100_000          # events per read: a 500 k-event production file needs about 6 GB at once


def fit_chunks(tree, truth=False, entry_stop=None, chunk=CHUNK, pe_scale=None):
    """fit_file over consecutive entry ranges (bounded memory). Yields one result dict per chunk."""
    n = tree.num_entries if entry_stop is None else min(entry_stop, tree.num_entries)
    for start in range(0, n, chunk):
        yield fit_file(tree, truth=truth, entry_start=start, entry_stop=min(start + chunk, n), pe_scale=pe_scale)


def fit_file(tree, truth=False, entry_stop=None, entry_start=None, pe_scale=None):
    """Lines for the events of one EventNtuple tree (optionally an entry range). Returns a dict of flat per-event and
    per-hit numpy arrays; event indices are local to the range."""
    c = tree.arrays(COINC_BR, entry_start=entry_start, entry_stop=entry_stop)
    p = tree.arrays(PULSE_BR + (TRUTH_BR if truth else []), entry_start=entry_start, entry_stop=entry_stop)
    return fit_arrays(c, p, truth=truth, pe_scale=pe_scale)


def fit_arrays(c, p, truth=False, pe_scale=None):
    """fit_file on in-memory jagged arrays with the EventNtuple branch names (COINC_BR in c, PULSE_BR in p, one entry
    per event), e.g. the fast chain's output (crvfastsim.reco.ntview)."""
    nev = len(c)
    chosen, has_ex = _choose_clusters(c, pe_scale)
    counts = ak.to_numpy(ak.num(p["crvpulses.barId"]))
    evt = np.repeat(np.arange(nev), counts)
    flat = {k: ak.to_numpy(ak.flatten(p[k])) for k in p.fields}
    slot, body, layer = G.bar_slot_body_layer(flat["crvpulses.barId"])
    hidx = flat["crvpulses.crvHitIndex"]
    chosen_mat = np.stack([chosen[s] for s in range(4)], 1)
    sel = (flat["crvpulses.sequenceIndex"] == 0) & (hidx >= 0) & (hidx == chosen_mat[evt, slot])
    e, s, b, L = evt[sel], slot[sel], body[sel], layer[sel]
    pe = np.maximum(flat["crvpulses.PEs"][sel], 0.0)
    X = flat["crvpulses.pos.fCoordinates.fX"][sel]
    Y = flat["crvpulses.pos.fCoordinates.fY"][sel]
    Z = flat["crvpulses.pos.fCoordinates.fZ"][sel]
    key = (e * N_BODY + b) * 4 + L
    pe_layer = np.bincount(key, pe, nev * N_BODY * 4)[key]
    w = np.where(pe_layer > 0, pe / np.where(pe_layer > 0, pe_layer, 1.0), 0.0) / SIG_STACK ** 2

    zs = s % 2 == 0                     # EX, T2: stacking z
    xs = ~zs                            # T1, TAG: stacking x
    az, bz, chiz, nz, okz = _wls(e[zs], Y[zs], Z[zs], w[zs], nev)

    st = c["crvcoincs.sectorType"]
    long_ok = (c["crvcoincs.hitPosAndTimeCalculated"]) & ((st == 1) | (st == 3))
    cidx = ak.local_index(st)
    pick = (cidx == chosen[0][:, None]) | (cidx == chosen[2][:, None])
    lmask = long_ok & pick
    le = np.repeat(np.arange(nev), ak.to_numpy(ak.num(st[lmask])))
    lx = ak.to_numpy(ak.flatten(c["crvcoincs.pos.fCoordinates.fX"][lmask]))
    ly = ak.to_numpy(ak.flatten(c["crvcoincs.pos.fCoordinates.fY"][lmask]))
    ex_ = np.concatenate([e[xs], le])
    ax, bx, chix, nx, okx = _wls(ex_, np.concatenate([Y[xs], ly]), np.concatenate([X[xs], lx]),
                                 np.concatenate([w[xs], np.full(len(le), 1.0 / SIG_LONG ** 2)]), nev)
    nstack_x = np.bincount(e[xs], minlength=nev)
    okx &= nstack_x > 0                 # x needs at least one position-measured (T1 or tagger) hit

    has = {n: chosen[i] >= 0 for i, n in enumerate(SLOT_NAME)}
    feat = _features(c, chosen, flat, evt, counts, nev)
    out = {"nev": nev, "has_ex": has_ex, "has": has, "okz": okz, "okx": okx,
           "z0": az, "sz": bz, "chi2z": chiz, "nz": nz, "x0": ax, "sx": bx, "chi2x": chix, "nx": nx,
           "hit_evt": e, "hit_body": b, "hit_layer": L, "hit_slot": s, "hit_w": w,
           "hit_res": np.where(zs, Z - (az[e] + bz[e] * (Y - Y_REF)), X - (ax[e] + bx[e] * (Y - Y_REF))),
           "hit_u": np.where(zs, Z, X), "hit_y": Y, "long_evt": le, "long_x": lx, "long_y": ly, "feat": feat,
           "hit_pe": pe, "hit_side": flat["crvpulses.SiPMId"][sel].astype(np.int64) % 2,
           "hit_time": flat["crvpulses.time"][sel], "hit_px": X, "hit_pz": Z}
    # per-FEB-pair timing points (plan 4.1.24): per chosen EX/T2 cluster, module and layer pair (LP 0 = layers 0-1,
    # 1 = layers 2-3; each LP is read by one FEB per side), x = V (t0 - t1)/2 from the PE-weighted first-pulse times of
    # each side, at the PE-weighted y of the LP's pulses. Uncorrected (module x LP offsets are applied downstream).
    zsel = np.isin(s, [0, 2])
    if zsel.any():
        side = flat["crvpulses.SiPMId"][sel][zsel].astype(np.int64) % 2
        tm = flat["crvpulses.time"][sel][zsel]
        ez, bz, lpz, pz, yz = e[zsel], b[zsel], (L[zsel] >= 2).astype(np.int64), pe[zsel], Y[zsel]
        k = ((ez.astype(np.int64) * N_BODY + bz) * 2 + lpz) * 2 + side
        uk, inv = np.unique(k, return_inverse=True)
        Wk = np.bincount(inv, pz)
        Tk = np.bincount(inv, pz * tm) / np.where(Wk > 0, Wk, 1)
        Yk = np.bincount(inv, pz * yz) / np.where(Wk > 0, Wk, 1)
        kk, ss = uk // 2, uk % 2
        both = np.nonzero((kk[1:] == kk[:-1]) & (ss[:-1] == 0) & (ss[1:] == 1) & (Wk[:-1] > 0) & (Wk[1:] > 0))[0]
        kb = kk[both]
        out.update({"fp_evt": (kb // (N_BODY * 2)).astype(np.int32), "fp_body": ((kb // 2) % N_BODY).astype(np.int8),
                    "fp_lp": (kb % 2).astype(np.int8), "fp_x": (V_FIBER * (Tk[both] - Tk[both + 1]) / 2.0).astype(np.float32),
                    "fp_y": (0.5 * (Yk[both] + Yk[both + 1])).astype(np.float32)})
    if truth:
        mu = np.abs(flat["crvpulsesmc.pdgId"][sel]) == 13
        tx = flat["crvpulsesmc.pos.fCoordinates.fX"][sel]
        ty = flat["crvpulsesmc.pos.fCoordinates.fY"][sel]
        tz = flat["crvpulsesmc.pos.fCoordinates.fZ"][sel]
        one = np.ones(mu.sum())
        tz0, tsz, _, tnz, tokz = _wls(e[mu], ty[mu], tz[mu], one, nev)
        tx0, tsx, _, tnx, tokx = _wls(e[mu], ty[mu], tx[mu], one, nev)
        out.update({"t_z0": tz0, "t_sz": tsz, "t_okz": tokz, "t_x0": tx0, "t_sx": tsx, "t_okx": tokx})
    return out


def layer_points(o):
    """Collapse pulses to one point per (event, body, layer): PE-weighted stacking coordinate u, y, total weight.
    Returns a dict of compact arrays for an alignment dump (npz), plus the EX/T2 timing points and per-event slopes."""
    e, b, L, w = o["hit_evt"], o["hit_body"], o["hit_layer"], o["hit_w"]
    key = (e.astype(np.int64) * N_BODY + b) * 4 + L
    uk, inv = np.unique(key, return_inverse=True)
    W = np.bincount(inv, w)
    good = W > 0
    Wd = np.where(good, W, 1.0)
    U = np.bincount(inv, w * o["hit_u"]) / Wd
    Yv = np.bincount(inv, w * o["hit_y"]) / Wd
    out = {"nev": np.int64(o["nev"]),
           "p_evt": (uk // (N_BODY * 4)).astype(np.int32)[good], "p_body": ((uk // 4) % N_BODY).astype(np.int8)[good],
           "p_layer": (uk % 4).astype(np.int8)[good], "p_u": U[good].astype(np.float32),
           "p_y": Yv[good].astype(np.float32), "p_w": W[good].astype(np.float32),
           "l_evt": o["long_evt"].astype(np.int32), "l_x": o["long_x"].astype(np.float32),
           "l_y": o["long_y"].astype(np.float32)}
    for k in ("t_sz", "t_sx"):
        out[k] = (o[k] if k in o else np.full(o["nev"], np.nan)).astype(np.float32)
    for k in ("fp_evt", "fp_body", "fp_lp", "fp_x", "fp_y"):
        if k in o:
            out[k] = o[k]
    return out


def concat_points(parts):
    """Concatenate layer_points dicts of consecutive chunks of one file, offsetting event indices."""
    out, off = {}, 0
    for d in parts:
        for k, v in d.items():
            if k == "nev":
                continue
            v = v + off if k in ("p_evt", "l_evt", "fp_evt") else v
            out.setdefault(k, []).append(v)
        off += int(d["nev"])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["nev"] = np.int64(off)
    return out
