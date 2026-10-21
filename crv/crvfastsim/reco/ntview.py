"""EventNtuple view of the fast chain's output: crvpulses / crvcoincs jagged arrays with the EventNtuple branch names.

Input: the chunk outputs of chain.clusters(..., detail=True) (cl_* per cluster, cl_pos, the member table mem_* and the
flat pulses). Output: two awkward record arrays, one entry per event of `events`, with the fields of
crvextrap.crvline's COINC_BR and PULSE_BR, so the CRV-only line fit and the response observables run unchanged on the
fast chain and on stock EventNtuples.

Conventions reproduced from EventNtuple CrvInfoHelper:
  - positions in the detector frame (x + solenoidOffset, z - detectorSystemZ0; extracted values);
  - crvpulses.pos is the counter centre, crvcoincs.pos the cluster's average position (length axis from timing when
    hitPosAndTimeCalculated); crvcoincs.time is the cluster's average hit time;
  - crvHitIndex is the index of the pulse's cluster in the event's crvcoincs (-1: unclustered), all pulses kept
    (keepUnclusteredPulses: true);
  - nHits = member pulses, nLayers = distinct layers, PEsPerLayer = member pulse PEs summed per layer (no status
    compensation).
Cluster order within an event is the chain's (sector type, then finder order); only the crvHitIndex link depends on it.
"""
import awkward as ak
import numpy as np

SOLENOID_OFFSET = 3904.0    # mu2e.solenoidOffset
DET_Z0 = 24171.0            # mu2e.detectorSystemZ0, extracted
CL_KEYS = ("cl_event", "cl_type", "cl_pe", "cl_avgtime", "cl_poscalc")
P_KEYS = ("ev", "idx", "bar", "sipm", "seq", "pe", "time")


def to_detector(p):
    """Mu2e-frame positions [n, 3] -> detector frame (extracted)."""
    p = np.asarray(p, float).reshape(-1, 3).copy()
    p[:, 0] += SOLENOID_OFFSET
    p[:, 2] -= DET_Z0
    return p


def concat(outs):
    """Chunk outputs -> one cluster table (mem_cl renumbered) and one pulse table."""
    C = {k: [] for k in CL_KEYS + ("cl_pos", "mem_cl", "mem_idx", "mem_event")}
    P = {k: [] for k in P_KEYS}
    off = 0
    for o in outs:
        n = len(o["cl_event"])
        for k in CL_KEYS + ("cl_pos", "mem_idx", "mem_event"):
            C[k].append(np.asarray(o[k]))
        C["mem_cl"].append(np.asarray(o["mem_cl"], np.int64) + off)
        off += n
        for k in P_KEYS:
            P[k].append(np.asarray(o["pulses"][k]))
    cat = lambda L, shape=(0,): np.concatenate(L) if L else np.zeros(shape)
    C = {k: cat(v, (0, 3) if k == "cl_pos" else (0,)) for k, v in C.items()}
    C["cl_pos"] = C["cl_pos"].reshape(-1, 3)
    return C, {k: cat(v) for k, v in P.items()}


def _jag(flat, ev_sorted, events):
    """Unflatten per-row values (rows sorted by event key) to one list per event of `events` (sorted)."""
    lo = np.searchsorted(ev_sorted, events)
    hi = np.searchsorted(ev_sorted, events, side="right")
    return ak.unflatten(flat, hi - lo)


def build(outs, geom, events=None):
    """(crvcoincs, crvpulses) awkward record arrays over `events` (sorted unique int64 keys; default: every event
    with a pulse). geom: crvfastsim.geometry.CRVGeometry (bar centres and layers in the Mu2e frame)."""
    C, P = concat(outs)
    ev_p = P["ev"].astype(np.int64)
    events = np.unique(ev_p) if events is None else np.asarray(events, np.int64)
    bar = P["bar"].astype(np.int64)
    # cluster local index: clusters ordered by event (stable, so the chain's order within an event)
    ev_c = C["cl_event"].astype(np.int64)
    oc = np.argsort(ev_c, kind="stable")
    rank = np.empty(len(oc), np.int64)
    rank[oc] = np.arange(len(oc))
    first = np.searchsorted(ev_c[oc], ev_c)
    local = rank - first
    # members -> pulses by (event, idx)
    pkey = ev_p * (1 << 20) + P["idx"].astype(np.int64)
    po = np.argsort(pkey, kind="stable")
    mkey = C["mem_event"].astype(np.int64) * (1 << 20) + C["mem_idx"].astype(np.int64)
    j = po[np.clip(np.searchsorted(pkey[po], mkey), 0, max(len(po) - 1, 0))] if len(po) else np.zeros(0, np.int64)
    ok = (pkey[j] == mkey) if len(po) else np.zeros(0, bool)
    j, mcl = j[ok], C["mem_cl"][ok]
    hit = np.full(len(ev_p), -1, np.int64)
    hit[j] = local[mcl]
    # per-cluster member sums
    ncl = len(ev_c)
    lay = geom.layer[bar[j]].astype(np.int64)
    pel = np.zeros((ncl, 4))
    np.add.at(pel, (mcl, lay), P["pe"][j])
    nhits = np.bincount(mcl, minlength=ncl)
    lay_hit = np.zeros((ncl, 4), bool)
    lay_hit[mcl, lay] = True
    cpos = to_detector(C["cl_pos"])
    ppos = to_detector(geom.center[bar])
    # events outside the universe are dropped (rows sorted by event first)
    keep_c = np.isin(ev_c[oc], events)
    occ = oc[keep_c]
    coinc = {"crvcoincs.sectorType": C["cl_type"][occ].astype(np.int32),
             "crvcoincs.time": C["cl_avgtime"][occ].astype(np.float32),
             "crvcoincs.PEs": C["cl_pe"][occ].astype(np.float32),
             "crvcoincs.hitPosAndTimeCalculated": C["cl_poscalc"][occ].astype(bool),
             "crvcoincs.pos.fCoordinates.fX": cpos[occ, 0].astype(np.float32),
             "crvcoincs.pos.fCoordinates.fY": cpos[occ, 1].astype(np.float32),
             "crvcoincs.pos.fCoordinates.fZ": cpos[occ, 2].astype(np.float32),
             "crvcoincs.nLayers": lay_hit[occ].sum(1).astype(np.int32),
             "crvcoincs.nHits": nhits[occ].astype(np.int32),
             "crvcoincs.PEsPerLayer[4]": pel[occ].astype(np.float32)}
    evc = ev_c[occ]
    po2 = np.argsort(ev_p, kind="stable")
    po2 = po2[np.isin(ev_p[po2], events)]
    pul = {"crvpulses.barId": bar[po2].astype(np.int32),
           "crvpulses.SiPMId": P["sipm"][po2].astype(np.int32),
           "crvpulses.PEs": P["pe"][po2].astype(np.float32),
           "crvpulses.time": P["time"][po2].astype(np.float32),
           "crvpulses.crvHitIndex": hit[po2].astype(np.int32),
           "crvpulses.sequenceIndex": P["seq"][po2].astype(np.int32),
           "crvpulses.pos.fCoordinates.fX": ppos[po2, 0].astype(np.float32),
           "crvpulses.pos.fCoordinates.fY": ppos[po2, 1].astype(np.float32),
           "crvpulses.pos.fCoordinates.fZ": ppos[po2, 2].astype(np.float32)}
    evp = ev_p[po2]
    c = ak.zip({k: _jag(v, evc, events) for k, v in coinc.items()}, depth_limit=1)
    p = ak.zip({k: _jag(v, evp, events) for k, v in pul.items()}, depth_limit=1)
    return c, p, events
