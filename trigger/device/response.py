"""Response layer of the device trigger: per path pass probabilities applied after the device's own cuts.

Binned rows: per path one pass probability (stock passes / device passes of the events in a bin, capped at 1), binned
in the straw hits of the event's best time cluster and, for paths with a helix stage, the momentum of the event's best
device helix. Rows are built by `run response-binned` and read by `chain.Chain` (row key "final"); paths without a
binned row keep their scalar hs/ks factors.

Named responses (`load`): the binned response of each process, shipped in `data/`. "ce" and "dio" are one file, built
from CE and DIO development shards (binned rows for the apr helix paths); "rpc" is built from RPC development shards
(binned rows for every tpr/apr track and helix path). Every other path keeps the scalar factors from CE shards. "none"
is no response layer (probability 1: the development runs of `response`, `response-binned` and trigcal). Any other
value is read as a path to a response JSON file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

APR_HELIX_PATHS = r"^apr_(Trk|Hlx)"   # default binned paths: apr paths with a helix stage (AgnosticHelixFinder not ported)

DATA = Path(__file__).resolve().parent / "data"
NAMED = {"ce": "response_ce_dio.json", "dio": "response_ce_dio.json", "rpc": "response_rpc.json"}
NONE = "none"
DEFAULT = "ce"            # the default where the process is not given: the CE chain's response (also DIO's)
NAMES = (*NAMED, NONE)


def path(spec):
    """File of a named response or a path; None for "none" (or None)."""
    if spec is None or spec == NONE:
        return None
    return DATA / NAMED[spec] if spec in NAMED else Path(spec)


def load(spec=DEFAULT):
    """Response rows {path: row} of a named response ("ce", "dio", "rpc", "none") or a JSON file; checked by
    `validate`. "none" (or None) gives {}: no response layer."""
    p = path(spec)
    if p is None:
        return {}
    if not p.is_file():
        raise FileNotFoundError(f"trigger response {spec!r}: no such file (named responses: {', '.join(NAMES)})")
    return validate(json.loads(p.read_text()))


def validate(resp):
    """Check the rows' form: hs/ks factors and binned factors in [0, 1], increasing edges, factor shapes matching the
    edges (hits per row; with edges_p one row per momentum bin). Returns resp."""
    if not isinstance(resp, dict):
        raise ValueError("trigger response: a JSON object {path: row} expected")
    for name, row in resp.items():
        for k in ("hs", "ks"):
            if k in row and not 0.0 <= float(row[k]) <= 1.0:
                raise ValueError(f"trigger response {name}: {k} = {row[k]} not in [0, 1]")
        fin = row.get("final")
        if fin is None:
            continue
        e = np.asarray(fin["edges"], float)
        f = np.asarray(fin["f"], float)
        shape = (len(e) - 1,)
        if "edges_p" in fin:
            ep = np.asarray(fin["edges_p"], float)
            if np.any(np.diff(ep) <= 0):
                raise ValueError(f"trigger response {name}: edges_p not increasing")
            shape = (len(ep) - 1,) + shape
        if np.any(np.diff(e) <= 0) or f.shape != shape or not np.all((f >= 0) & (f <= 1)):
            raise ValueError(f"trigger response {name}: binned row malformed (edges increasing, f of shape {shape} "
                             f"in [0, 1] expected; f shape {f.shape})")
    return resp


def _merge(cnt, edges, min_count):
    """Merge bins with fewer than min_count device passes into their lower neighbour (the first into the next)."""
    e, c = list(edges), np.array(cnt, float)
    i = 1
    while i < len(c):
        if c[i, 1] < min_count:
            c[i - 1] += c[i]; c = np.delete(c, i, 0); del e[i]
        else:
            i += 1
    if len(c) > 1 and c[0, 1] < min_count:
        c[1] += c[0]; c = np.delete(c, 0, 0); del e[1]
    return e, c


def _f(c):
    return [float(min(1.0, s / d)) if d > 0 else 1.0 for s, d in c]


def binned_rows(acc, edges, edges_p, min_count, base=None):
    """Response rows from per-path bin counts acc {path: {"1d": [nb, 2], "2d": [npb, nb, 2], "helix": bool}} of (stock
    passes, device passes); added to a copy of base (a scalar response), whose other paths and keys are kept."""
    npb = len(edges_p) - 1
    out = {k: dict(v) for k, v in (base or {}).items()}
    for name, a in sorted(acc.items()):
        e1, c1 = _merge(a["1d"], edges, min_count)
        row = {"final": {"var": "tc_nsh", "edges": e1, "f": _f(c1)},
               "basis": {"stock_passes": c1[:, 0].tolist(), "device_passes": c1[:, 1].tolist()}}
        if a["helix"] and npb > 1:
            f2 = []
            for jp in range(npb):
                if a["2d"][jp, :, 1].sum() < min_count:
                    f2.append(_f(c1))                     # fall back to the hits-only factors
                    continue
                # the row's own hit bins, then mapped onto the common (1D merged) edges
                er, cr = _merge(a["2d"][jp], edges, min_count)
                fr = _f(cr)
                f2.append([fr[min(max(int(np.searchsorted(er, lo, side="right")) - 1, 0), len(fr) - 1)] for lo in e1[:-1]])
            row["final"] = {"var": "tc_nsh x helix_p", "edges": e1, "edges_p": edges_p, "f": f2}
            row["basis"]["rows_device_passes"] = a["2d"][:, :, 1].sum(1).tolist()
        out[name] = {**out.get(name, {}), **row}
    return out
