"""Donor library: stock EventNtuple events of a training sample, indexed for the write stage (kind ``pool``).

The write stage clones the stock tree layout and, for every fast event, loads one **donor** event of the stock
training files into the branch buffers. The fast chain then overwrites what it simulates (the conversion electron's
track at the tracker surfaces, its truth, the calorimeter clusters, the trigger paths the device evaluates, the event
identity). Every other value of the event (the fit bookkeeping and MVA outputs of the electron track, the other
fit hypotheses, the extrapolated surfaces outside the tracker, the uncovered trigger paths, the event-level MC
quantities) is the donor's, rotated about the tracker axis to the fast azimuth and shifted to the fast time.

A donor is drawn among the stock events of the same **class** and the same **covered-trigger bit** (any of the paths
the device evaluates fired), nearest in the matching keys:

* class Q (the electron fit passes the response layer's quality: e- hypothesis, downstream at TT_Front, fit
  consistency above the CE cut): entry tan(dip), active hits, fit consistency, calorimeter match and E/p;
* class F (an e- fit that fails the quality): entry tan(dip);
* class N (no e- fit): calorimeter leading-cluster energy.

The index is an ``.npz`` with a library manifest (``mu2efastsim.library``) whose source pins the sha256 of every
stock file; the stock files themselves are the pool (read by entry in the write stage).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TREE = "EventNtuple/ntuple"
SID_FRONT, SID_MID, SID_BACK = 0, 1, 2
FITCON_CUT = 1e-3            # the CE fit-consistency cut of the response layer (recoresp.table.CE_CUTS)
CLASS_N, CLASS_F, CLASS_Q = 0, 1, 2
NAME = "beam.centuple_donor"
K_NEAREST = 8                # a donor is drawn uniformly among this many nearest of its class and trigger bit
# matching scales (distance = sum of ((key - key_donor) / scale)^2 over the class keys)
SCALES = dict(tandip=0.05, nactive=1.0, fitcon=0.15, calo=0.25, eop=0.05, ecl=5.0)
USE = {CLASS_Q: ("tandip", "nactive", "fitcon", "calo", "eop"), CLASS_F: ("tandip",), CLASS_N: ("ecl",)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def covered_any(trig, paths):
    """OR of the trig_<path> booleans of `paths` (dict of arrays)."""
    out = None
    for p in paths:
        v = np.asarray(trig["trig_" + p], bool)
        out = v if out is None else out | v
    return out


def _primary(pdg, status, sid, pz, inb):
    """Per event: index of the first e- fit with status > 0 (or -1), and whether it is downstream at TT_Front."""
    import awkward as ak
    is_e = (pdg == 11) & (status > 0)
    front_dn = ak.any((sid == SID_FRONT) & (pz > 0) & inb, axis=-1)
    idx = ak.local_index(pdg, axis=1)
    first = ak.fill_none(ak.firsts(idx[is_e], axis=1), -1)
    return ak.to_numpy(first).astype(np.int64), front_dn


def index_file(path, covered_paths):
    """Matching keys of every event of one stock EventNtuple file (dict of NumPy arrays, one row per event)."""
    import awkward as ak
    import uproot
    t = uproot.open(path)[TREE]
    a = t.arrays(["trk.pdg", "trk.status", "trk.nactive", "trk.fitcon", "trksegs", "trksegsmc", "trkcalohit.did",
                  "trkcalohit.edep", "caloclusters.energyDep_", "trkpid.result", "trkqual.result", "run", "subrun", "event"]
                 + ["trig_" + p for p in covered_paths], library="ak")
    segs = a["trksegs"]
    sid, inb = segs["sid"], segs["inbounds"]
    pz = segs["mom"]["fCoordinates"]["fZ"]
    prim, front_dn = _primary(a["trk.pdg"], a["trk.status"], sid, pz, inb)
    n = len(prim)
    r = np.arange(n)
    has = prim >= 0
    pi = np.maximum(prim, 0)

    def pick(arr, fill=np.nan, dtype=float):
        sel = arr[ak.local_index(arr, axis=1) == ak.Array(pi)[:, np.newaxis]]
        out = ak.to_numpy(ak.fill_none(ak.firsts(sel, axis=1), fill)).astype(dtype)
        out[~has] = fill
        return out

    nactive = pick(a["trk.nactive"], -1, np.int64)
    fitcon = pick(a["trk.fitcon"])
    dn = np.array([bool(front_dn[i][pi[i]]) if has[i] else False for i in r])
    q = has & dn & (fitcon > FITCON_CUT)
    cls = np.where(q, CLASS_Q, np.where(has, CLASS_F, CLASS_N)).astype(np.int8)
    # truth at TT_Front of the primary track: earliest downstream sid-0 crossing
    mcs = a["trksegsmc"]
    tan_t, t_t, phi_t = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    mc_sid = ak.to_list(mcs["sid"])
    mc_t = ak.to_list(mcs["time"])
    mc_p = {c: ak.to_list(mcs["mom"]["fCoordinates"][c]) for c in ("fX", "fY", "fZ")}
    for i in np.flatnonzero(has):
        j = pi[i]
        if j >= len(mc_sid[i]):
            continue
        best = None
        for k, s in enumerate(mc_sid[i][j]):
            if s == SID_FRONT and mc_p["fZ"][i][j][k] > 0 and (best is None or mc_t[i][j][k] < mc_t[i][j][best]):
                best = k
        if best is None:
            continue
        px, py, pzz = (mc_p[c][i][j][best] for c in ("fX", "fY", "fZ"))
        tan_t[i] = pzz / np.hypot(px, py)
        phi_t[i] = np.arctan2(py, px)
        t_t[i] = mc_t[i][j][best]
    did = pick(a["trkcalohit.did"], -1, np.int64)
    edep = pick(a["trkcalohit.edep"])
    # reco |p| at the first downstream TT_Front segment of the primary (E/p)
    p_front = np.full(n, np.nan)
    sl, pl = ak.to_list(sid), {c: ak.to_list(segs["mom"]["fCoordinates"][c]) for c in ("fX", "fY", "fZ")}
    tl = ak.to_list(segs["time"])
    for i in np.flatnonzero(has):
        j = pi[i]
        ks = [k for k, s in enumerate(sl[i][j]) if s == SID_FRONT and pl["fZ"][i][j][k] > 0]
        if ks:
            k = min(ks, key=lambda k: tl[i][j][k])
            p_front[i] = np.sqrt(sum(pl[c][i][j][k] ** 2 for c in ("fX", "fY", "fZ")))
    pid = pick(a["trkpid.result"])
    ecl = ak.to_numpy(ak.fill_none(ak.max(a["caloclusters.energyDep_"], axis=1), 0.0)).astype(float)
    trig = {"trig_" + p: ak.to_numpy(a["trig_" + p]) for p in covered_paths}
    return dict(run=ak.to_numpy(a["run"]).astype(np.int64), subrun=ak.to_numpy(a["subrun"]).astype(np.int64),
                event=ak.to_numpy(a["event"]).astype(np.int64), entry=r.astype(np.int64), cls=cls,
                prim=prim, nactive=nactive, fitcon=fitcon, tandip_t=tan_t, phi_t=phi_t, t_t=t_t,
                calo=(did >= 0), pid=pid, eop=np.where(did >= 0, edep / p_front, np.nan), ecl=ecl,
                cov=covered_any(trig, covered_paths).astype(bool))


def build(paths, covered_paths, out, *, source=None, generator=None, release=None, split=None):
    """Index every event of the stock files `paths` into the donor library `out` (.npz with manifest)."""
    from mu2efastsim.library import store
    from mu2efastsim.library.axes import Axis
    parts, files = [], []
    for fi, p in enumerate(paths):
        d = index_file(p, covered_paths)
        d["file_index"] = np.full(d["entry"].size, fi, np.int64)
        parts.append(d)
        files.append(dict(name=Path(p).name, sha256=_sha256(p), entries=int(d["entry"].size)))
    arr = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    arr["covered_paths"] = np.array(covered_paths)
    arr["file_names"] = np.array([f["name"] for f in files])
    arr["file_sha256"] = np.array([f["sha256"] for f in files])
    tan_axis = Axis("tandip_t", "1", [-10.0, 10.0], frame="true TT_Front state of the electron (pz / pT)",
                    scale="linear", policy="clip", description="matching key (nearest donors), not a binning")
    arr[tan_axis.array] = tan_axis.edges
    counts = {c: int((arr["cls"] == v).sum()) for c, v in (("N", CLASS_N), ("F", CLASS_F), ("Q", CLASS_Q))}
    fields = dict(
        schema=1, name=NAME + ".run1a_onspill", subsystem="beam", version=1, kind="pool",
        description=("Stock EventNtuple events of a training sample of conversion electrons, indexed by the keys the "
                     "write stage matches a fast event on (class, covered-trigger bit, entry tan dip, active hits, fit "
                     "consistency, calorimeter match and E/p, leading-cluster energy). The pool is the stock files "
                     "themselves, pinned by sha256; a draw returns (file, entry)."),
        axes=[tan_axis.describe()],
        outputs=[dict(name="file_index", unit="1", dtype="int64", description="index into file_names (-1: none)"),
                 dict(name="entry", unit="1", dtype="int64", description="tree entry of the donor event")],
        source=source or dict(samples=files, description="stock EventNtuple files of the training sample"),
        generator=generator or dict(name="stock Mu2e chain", release=None, physics_list=None, configuration=None),
        release=release or dict(repository="mu2e-fastsim", commit=None, entry_point="beamfastsim.centuple.donor"),
        seeds=dict(draw="per event: numpy SeedSequence([seed, run, subrun, event])"),
        geometry=dict(tag="run1a", hash=None),
        config_hash=None,
        split=split or dict(training="the files above", held_out="disjoint files", rule="by file"),
        dtype=dict(storage={}, compute=["float32", "float64"],
                   notes="host-side matching in float64; the fast record's dtype does not enter the draw keys' type"),
        rng=dict(host="numpy.random.Generator", device=None, streams={"donor": "per event"}, draws_per_row=1),
        domain=dict(coverage=f"donor classes N/F/Q = {counts['N']}/{counts['F']}/{counts['Q']}",
                    fallbacks=[dict(name="nearest", levels=1,
                                    description="nearest K donors of the same class and covered-trigger bit; "
                                                "if none, the same class")]),
        validation=dict(status="not validated", record="beam/docs/eventntuple.md (validation)"),
        unrecorded={})
    fields["unrecorded"]["config_hash"] = "no build configuration beyond the listed files and covered paths"
    for k in ("generator", "release"):
        if any(v is None for v in fields[k].values()):
            fields[k] = None
            fields["unrecorded"][k] = "not passed to build()"
    fields["geometry"] = dict(tag="run1a", hash="Offline/Mu2eG4/geom/geom_run1_a.txt")
    store.save(out, arr, fields)
    return dict(events=int(arr["entry"].size), **counts)


def load(path):
    from mu2efastsim.library import store
    return store.load(path).as_dict()


def match(lib, cls, cov, keys, seed, run, subrun, event, k=K_NEAREST, use=None, stream=0):
    """Donor (file, entry) per fast event. cls, cov: class and covered-trigger bit per event; keys: dict of the
    matching keys (tandip_t, nactive, fitcon, calo, eop, ecl) per event. The draw for event e uses
    Generator(SeedSequence([seed, run, subrun, event])), so it does not depend on the batch layout."""
    n = len(cls)
    out_f, out_e = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    dcls, dcov = np.asarray(lib["cls"]), np.asarray(lib["cov"])
    use = use or USE
    dkey = dict(tandip=lib["tandip_t"], nactive=lib["nactive"].astype(float), fitcon=lib["fitcon"],
                calo=lib["calo"].astype(float), eop=lib["eop"], ecl=lib["ecl"])
    fkey = dict(tandip=keys["tandip_t"], nactive=np.asarray(keys["nactive"], float), fitcon=keys["fitcon"],
                calo=np.asarray(keys["calo"], float), eop=keys["eop"], ecl=keys["ecl"])
    for c in (CLASS_N, CLASS_F, CLASS_Q):
        for b in (False, True):
            ev = np.flatnonzero((cls == c) & (cov == b))
            if not ev.size:
                continue
            pool = np.flatnonzero((dcls == c) & (dcov == b))
            if not pool.size:
                pool = np.flatnonzero(dcls == c)
            if not pool.size:
                pool = np.arange(dcls.size)
            for c0 in range(0, ev.size, 2048):
                e = ev[c0:c0 + 2048]
                d2 = np.zeros((e.size, pool.size))
                for name in use[c]:
                    a = np.asarray(fkey[name], float)[e][:, None]
                    bb = np.asarray(dkey[name], float)[pool][None, :]
                    diff = (a - bb) / SCALES[name]
                    d2 += np.where(np.isfinite(diff), diff * diff, np.where(np.isnan(a) & np.isnan(bb), 0.0, 25.0))
                kk = min(k, pool.size)
                near = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
                for i, row in zip(e, near):
                    g = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), int(run[i]),
                                                                      int(subrun[i]), int(event[i])]))
                    j = pool[row[g.integers(row.size)]]
                    out_f[i], out_e[i] = lib["file_index"][j], j
    return out_f, out_e


def describe(path):
    lib = load(path)
    return json.dumps({k: (v.shape, str(v.dtype)) for k, v in lib.items()}, indent=1)
