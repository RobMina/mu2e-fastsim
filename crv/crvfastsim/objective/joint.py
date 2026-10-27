"""Passive measured-variable tables; see docs/response-joint.md for populations and units."""
import json
from pathlib import Path

import awkward as ak
import numpy as np

from . import bars as G
from . import crvline as CL
from . import crvresponse as R

SCHEMA_VERSION = 1


def _flat(a):
    return ak.to_numpy(ak.flatten(a))


def _identity(keys, evt, source_index):
    return {"run": keys[evt, 0], "subrun": keys[evt, 1], "event": keys[evt, 2],
            "event_index": evt, "source_index": np.full(len(evt), source_index, np.int64)}


def _flags(table, name, bounds=None):
    v = table[name]
    finite = np.isfinite(v)
    table[name + "_finite"] = finite
    table[name + "_nonfinite"] = ~finite
    if bounds is not None:
        lo, hi = bounds
        table[name + "_underflow"] = v < lo
        table[name + "_overflow"] = v > hi
        table[name + "_inrange"] = finite & (v >= lo) & (v <= hi)


def _count(key, mask, size):
    return np.bincount(key[mask], minlength=size).astype(np.int64)


def capture(c, p, o, event_keys, source_index=0):
    """Return owned flat columns from an existing accumulate_arrays result, without fitting or RNG."""
    nev = int(o["nev"])
    keys = np.asarray(event_keys)
    if keys.shape != (nev, 3) or keys.dtype.kind not in "iu":
        raise ValueError("event_keys must be an integer (nev, 3) run/subrun/event array")
    if keys.size and (np.any(keys < 0) or np.any(keys > np.iinfo(np.int64).max)):
        raise ValueError("event keys must be nonnegative signed-int64-compatible integers")
    if not isinstance(source_index, (int, np.integer)) or not 0 <= source_index <= np.iinfo(np.int64).max:
        raise ValueError("source_index must be a nonnegative int64 integer")
    if len(c) != nev or len(p) != nev:
        raise ValueError("c, p and o must describe the same event chunk")
    keys = keys.astype(np.int64, copy=False)
    chosen, _ = CL._choose_clusters(c)
    chosen = np.stack([chosen[k] for k in range(4)], axis=1)
    has = np.stack([o["has"][name] for name in CL.SLOT_NAME], axis=1)
    if not np.array_equal(chosen >= 0, has):
        raise ValueError("o does not match unscaled accumulate_arrays selection")
    pevt = np.repeat(np.arange(nev), ak.to_numpy(ak.num(p["crvpulses.barId"])))
    bar = _flat(p["crvpulses.barId"]).astype(np.int64)
    ps, pb, pl = G.bar_slot_body_layer(bar)
    hidx = _flat(p["crvpulses.crvHitIndex"]).astype(np.int64)
    seq = _flat(p["crvpulses.sequenceIndex"]).astype(np.int64)
    selected = (seq == 0) & (hidx >= 0) & (hidx == chosen[pevt, ps])
    e, s = np.asarray(o["hit_evt"], np.int64), np.asarray(o["hit_slot"], np.int64)
    if not all(np.array_equal(a, b) for a, b in ((e, pevt[selected]), (s, ps[selected]),
               (o["hit_body"], pb[selected]), (o["hit_layer"], pl[selected]))):
        raise ValueError("o selected rows do not match c/p order and identities")
    h = _identity(keys, e, source_index)
    h.update({"slot": s, "body": o["hit_body"], "layer": o["hit_layer"], "side": o["hit_side"],
              "bar_id": bar[selected], "pulse_index": _flat(ak.local_index(p["crvpulses.barId"]))[selected],
              "sipm_id": _flat(p["crvpulses.SiPMId"])[selected], "sequence_index": seq[selected],
              "chosen_cluster": hidx[selected], "pe_raw": _flat(p["crvpulses.PEs"])[selected],
              "pe": o["hit_pe"], "time": o["hit_time"], "y": o["hit_y"],
              "okx": o["okx"][e], "okz": o["okz"][e]})
    zs = np.isin(s, [0, 2])
    half = np.asarray([R.HALF[name] for name in CL.SLOT_NAME])[s]
    yy = o["hit_y"] - CL.Y_REF
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        along_abs = np.where(zs, o["x0"][e] + o["sx"][e] * yy, o["z0"][e] + o["sz"][e] * yy)
        along_mm = along_abs - np.where(zs, o["hit_px"], o["hit_pz"])
        al = along_mm / half
        has_al = np.where(zs, o["okx"][e], o["okz"][e]) & (np.abs(al) <= 1.0)
        path = np.sqrt(1.0 + o["sx"][e] ** 2 + o["sz"][e] ** 2)
        dist = np.where(h["side"] == 0, al + 1.0, 1.0 - al) * half
        tc = h["time"] - dist / R.V_FIBER
        gk = e * 4 + s
        ug, gi = np.unique(gk[has_al], return_inverse=True)
        n = np.bincount(gi, minlength=len(ug))
        mean = np.bincount(gi, tc[has_al], len(ug)) / np.maximum(n, 1)
        m2 = np.bincount(gi, tc[has_al] ** 2, len(ug)) / np.maximum(n, 1)
        rms = np.sqrt(np.maximum(m2 - mean ** 2, 0.0))
        gn = np.zeros(nev * 4, np.int64)
        gm, gr = np.full(nev * 4, np.nan), np.full(nev * 4, np.nan)
        gn[ug], gm[ug], gr[ug] = n, mean, rms
        residual = np.where(has_al, tc - gm[gk], np.nan)
        h.update({"along_mm": along_mm, "along": al, "path": path, "pe_path": h["pe"] / path,
                  "pe_raw_path": h["pe_raw"] / path, "distance": dist, "time_corrected": tc,
                  "time_residual": residual, "group_count": gn[gk], "group_mean": gm[gk],
                  "group_rms": gr[gk], "has_along": has_al, "spread_member": has_al,
                  "both_lines": h["okx"] & h["okz"],
                  "corrected_valid": has_al & np.isfinite(tc),
                  "residual_valid": has_al & np.isfinite(residual)})
    h["pe_along_eligible"] = h["both_lines"] & has_al
    h["pe_path_eligible"] = h["both_lines"]
    h["spread_eligible"] = gn[gk] >= R.MIN_SPREAD_HITS
    for name in ("time", "time_corrected", "along_mm", "distance", "group_mean", "y"):
        _flags(h, name)
    for name, edges in (("along", R.E["along"]), ("path", R.E["path"]), ("pe", R.E["pe"]),
                        ("pe_raw", R.E["pe"]), ("pe_path", R.E["pe"]), ("pe_raw_path", R.E["pe"]),
                        ("group_rms", R.E["tspread"]), ("time_residual", R.FTP_EDGES)):
        _flags(h, name, (edges[0], edges[-1]))
    h["pe_along_inrange"] = h["pe_along_eligible"] & h["along_inrange"] & h["pe_path_inrange"]
    h["pe_path_hist_inrange"] = h["pe_path_eligible"] & h["path_inrange"] & h["pe_inrange"]
    h["ftp_eligible"] = has_al & np.isin(s, [0, 2])
    h["ftp_inrange"] = h["ftp_eligible"] & h["time_residual_inrange"]

    ce = np.repeat(np.arange(nev), ak.to_numpy(ak.num(c["crvcoincs.sectorType"])))
    ci = _flat(ak.local_index(c["crvcoincs.sectorType"])).astype(np.int64)
    st = _flat(c["crvcoincs.sectorType"]).astype(np.int64)
    cs = np.where((st >= 1) & (st <= 4), st - 1, -1)
    ct, cp = _flat(c["crvcoincs.time"]), _flat(c["crvcoincs.PEs"])
    dt = ct - o["feat"]["t_EX"][ce]
    valid_slot = cs >= 0
    timing = np.abs(dt) < CL.DT_MATCH
    csel = valid_slot & (ci == chosen[ce, np.maximum(cs, 0)])
    eligible = valid_slot & (chosen[ce, 0] >= 0) & (cp > 0) & ((cs == 0) | timing)
    q = _identity(keys, ce, source_index)
    q.update({"cluster_index": ci, "sector_type": st, "slot": cs, "time": ct, "pe": cp,
              "dt_ex": dt, "has_ex": chosen[ce, 0] >= 0, "timing_match": timing,
              "pe_positive": cp > 0, "eligible": eligible, "selected": csel,
              "nlayers": _flat(c["crvcoincs.nLayers"]), "nhits": _flat(c["crvcoincs.nHits"]),
              "position_time_calculated": _flat(c["crvcoincs.hitPosAndTimeCalculated"]),
              "position_x": _flat(c["crvcoincs.pos.fCoordinates.fX"]),
              "position_y": _flat(c["crvcoincs.pos.fCoordinates.fY"])})
    q["timing_rejected"] = valid_slot & (cs != 0) & q["has_ex"] & ~timing
    q["timing_outside_window"] = valid_slot & (cs != 0) & q["has_ex"] & np.isfinite(dt) & ~timing
    q["timing_unavailable"] = valid_slot & (cs != 0) & (~q["has_ex"] | ~np.isfinite(dt))
    q["ex_cluster"] = chosen[ce, 0]
    counts = ak.to_numpy(ak.num(c["crvcoincs.sectorType"]))
    offsets = np.r_[0, np.cumsum(counts)]
    linked = (hidx >= 0) & (hidx < counts[pevt])
    cg = offsets[pevt[linked]] + hidx[linked]
    q["pulse_count"] = np.bincount(cg, minlength=len(ce)).astype(np.int64)
    q["first_pulse_count"] = np.bincount(cg[seq[linked] == 0], minlength=len(ce)).astype(np.int64)
    q["body"] = np.full(len(ce), -1, np.int64)
    if len(cg):
        order = np.lexsort((-_flat(p["crvpulses.PEs"])[linked], cg))
        lead = np.r_[True, cg[order][1:] != cg[order][:-1]]
        q["body"][cg[order][lead]] = pb[linked][order][lead]
    q["body_available"] = q["body"] >= 0
    q["pe_layer"] = np.asarray(ak.to_numpy(ak.flatten(c["crvcoincs.PEsPerLayer[4]"], axis=1)), float).reshape(-1, 4)
    for name in ("time", "pe", "nhits", "position_x", "position_y"):
        _flags(q, name)
    _flags(q, "nlayers", (R.E["nl"][0], R.E["nl"][-1]))
    _flags(q, "pe_layer", (R.E["pel"][0], R.E["pel"][-1]))
    q["pe_layer_eligible"] = csel[:, None] & (q["pe_layer"] > 0)
    q["pe_layer_hist_inrange"] = q["pe_layer_eligible"] & q["pe_layer_inrange"]
    _flags(q, "dt_ex", (R.E["dt"][0], R.E["dt"][-1]))

    ee = np.repeat(np.arange(nev), 4)
    es = np.tile(np.arange(4), nev)
    v = _identity(keys, ee, source_index)
    v.update({"slot": es, "selected": has.ravel(), "chosen_cluster": chosen.ravel(),
              "has_ex": o["has_ex"][ee], "okx": o["okx"][ee], "okz": o["okz"][ee],
              "both_lines": (o["okx"] & o["okz"])[ee], "group_count": gn, "group_mean": gm,
              "group_rms": gr, "spread_eligible": gn >= R.MIN_SPREAD_HITS,
              "selected_pulse_count": np.bincount(gk, minlength=nev * 4).astype(np.int64),
              "body": np.full(nev * 4, -1, np.int64)})
    if len(e):
        order = np.lexsort((-o["hit_w"], s, e))
        lead = np.r_[True, gk[order][1:] != gk[order][:-1]]
        v["body"][gk[order][lead]] = o["hit_body"][order][lead]
    ck = ce * 4 + np.maximum(cs, 0)
    for name, mask in (("candidate_count", valid_slot), ("candidate_positive_count", valid_slot & (cp > 0)),
                       ("timing_match_count", valid_slot & timing),
                       ("timing_rejected_count", q["timing_rejected"]), ("eligible_count", eligible)):
        v[name] = _count(ck, mask, nev * 4)
    for name in ("x0", "z0", "sx", "sz"):
        v[name] = o[name][ee]
        _flags(v, name)
    for name, feat in (("cluster_time", "t"), ("cluster_pe", "pe"), ("cluster_nlayers", "nl"),
                       ("cluster_nhits", "nh")):
        v[name] = np.stack([o["feat"][f"{feat}_{slot}"] for slot in CL.SLOT_NAME], axis=1).ravel()
        _flags(v, name)
    _flags(v, "cluster_nlayers", (R.E["nl"][0], R.E["nl"][-1]))
    v["nlayers_inrange"] = v["selected"] & v["cluster_nlayers_inrange"]
    v["dt_ex"] = v["cluster_time"] - o["feat"]["t_EX"][ee]
    _flags(v, "dt_ex", (R.E["dt"][0], R.E["dt"][-1]))
    v["dt_cluster_inrange"] = v["selected"] & (es != 0) & v["dt_ex_inrange"]
    v["dtp_inrange"] = np.isfinite(v["dt_ex"]) & (v["dt_ex"] >= R.DTP_EDGES[0]) & (v["dt_ex"] <= R.DTP_EDGES[-1])
    _flags(v, "group_mean")
    _flags(v, "group_rms", (R.E["tspread"][0], R.E["tspread"][-1]))
    v["tspread_inrange"] = v["spread_eligible"] & v["group_rms_inrange"]
    return {f"{prefix}_{name}": np.array(value, copy=True) for prefix, table in
            (("hit", h), ("event", v), ("candidate", q)) for name, value in table.items()}


def summarize(columns, chunks=1):
    """JSON-compatible flow and flag counts; all bodies and sectors remain explicit."""
    out = {"schema_version": SCHEMA_VERSION, "chunks": int(chunks), "slots": list(CL.SLOT_NAME),
           "body_names": list(CL.BODY_NAMES), "tables": {},
           "definitions": {"grouping": "event_slot", "key_order": ["run", "subrun", "event"],
                           "dt_match_ns": float(CL.DT_MATCH), "min_spread_hits": int(R.MIN_SPREAD_HITS),
                           "fiber_speed_mm_per_ns": float(R.V_FIBER), "half_lengths_mm": dict(R.HALF),
                           "histogram_edges": {name: edges.tolist() for name, edges in R.E.items()},
                           "ftp_edges_ns": R.FTP_EDGES.tolist(), "dtp_edges_ns": R.DTP_EDGES.tolist()}}
    for table in ("hit", "event", "candidate"):
        slot = columns[table + "_slot"]
        body = columns[table + "_body"]
        masks = {"all": np.ones(len(slot), bool)}
        masks.update({name: slot == k for k, name in enumerate(CL.SLOT_NAME)})
        masks["unknown_slot"] = slot < 0
        masks.update({"body_" + name: body == k for k, name in enumerate(CL.BODY_NAMES)})
        masks["body_unavailable"] = body < 0
        booleans = {k[len(table) + 1:]: v for k, v in columns.items()
                    if k.startswith(table + "_") and v.dtype.kind == "b"}
        flows = {}
        for name, mask in masks.items():
            counts = {key: int(np.count_nonzero(value[mask])) for key, value in booleans.items()}
            if table == "event":
                counts.update({key: int(columns["event_" + key][mask].sum()) for key in
                               ("candidate_count", "candidate_positive_count", "timing_match_count",
                                "timing_rejected_count", "eligible_count", "selected_pulse_count", "group_count")})
            flows[name] = {"rows": int(mask.sum()), "counts": counts}
        out["tables"][table] = flows
    out["events"] = int(len(columns["event_slot"]) // 4)
    return out


class JointCapture:
    """Collect owned chunk tables; write once per bounded input shard if memory is limited."""

    def __init__(self):
        self._parts = []

    def add(self, c, p, o, event_keys, source_index=0):
        """Capture one already-accumulated chunk; return self for chaining."""
        columns = capture(c, p, o, event_keys, source_index)
        for table in ("hit", "event", "candidate"):
            columns[table + "_chunk_index"] = np.full(len(columns[table + "_slot"]), len(self._parts), np.int64)
        self._parts.append(columns)
        return self

    def columns(self):
        """Concatenate owned columns; event_index stays chunk-local, full keys remain authoritative."""
        if not self._parts:
            raise ValueError("add at least one chunk (an empty chunk is supported) before writing")
        return {key: np.concatenate([part[key] for part in self._parts], axis=0) for key in self._parts[0]}

    def summary(self):
        return summarize(self.columns(), chunks=len(self._parts))

    def write(self, path):
        """Exclusively create an NPZ containing numeric columns and summary_json; return the summary."""
        columns = self.columns()
        summary = summarize(columns, chunks=len(self._parts))
        columns["summary_json"] = np.asarray(json.dumps(summary, allow_nan=False, sort_keys=True))
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, **columns)
        return summary
