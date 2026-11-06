"""Paired CRV response diagnostics on a complete explicitly ordered event population."""
import awkward as ak
import numpy as np
import uproot

from .compare import paired_counts
from .counts import event_ids, event_keys


def rows_to_events(rows, keys):
    ordered = event_keys(keys)
    query = event_keys(event_ids(rows))
    order = np.argsort(ordered)
    index = np.searchsorted(ordered[order], query)
    if np.any(index >= len(keys)) or not np.array_equal(ordered[order[index]], query):
        raise ValueError("CRV output row is absent from the complete event census")
    return order[index]


def histogram(event, values, edges, nevents):
    """Include underflow/overflow; empty events remain rows of zero counts."""
    event, values = np.asarray(event, np.int64), np.asarray(values)
    if not np.isfinite(values).all() or np.any(event < 0) or np.any(event >= nevents):
        raise ValueError("invalid CRV histogram rows")
    bins = np.searchsorted(edges, values, side="right")
    return np.bincount(event * (len(edges) + 1) + bins,
                       minlength=nevents * (len(edges) + 1)).reshape(nevents, -1)


def compare(reference, keys, D, P, C, geom):
    """Stock reference is read only; independent response draws imply statistical, not exact closure."""
    nevents = len(keys)
    with uproot.open(reference) as root:
        census = root["crvdump/events"].arrays(library="ak")
        rows_to_events(ak.Array({k: keys[:, i] for i, k in enumerate(("run", "subrun", "event"))}), event_ids(census))
        stock = {name: root[f"crvdump/{name}"].arrays(library="ak") for name in ("digis", "pulses", "clusters")}
    output = {}
    for name, fast in (("digis", D), ("pulses", P), ("clusters", C)):
        data = stock[name]
        in_sample = np.isin(event_keys(event_ids(data)), event_keys(keys))
        data = data[in_sample]
        ev = rows_to_events(data, keys)
        fev = np.asarray(fast["ev"], np.int64)
        output[name + "_total"] = paired_counts(np.bincount(ev, minlength=nevents)[:, None],
                                                np.bincount(fev, minlength=nevents)[:, None])
        if name != "clusters":
            nsector = len(geom.sector_names)
            sb = ak.to_numpy(data.bar).astype(np.int64)
            fb = np.asarray(fast["bar"], np.int64)
            sc = np.bincount(ev * nsector + geom.sector[sb], minlength=nevents * nsector).reshape(nevents, nsector)
            fc = np.bincount(fev * nsector + geom.sector[fb], minlength=nevents * nsector).reshape(nevents, nsector)
            output[name + "_sectors"] = {"names": list(geom.sector_names), **paired_counts(sc, fc)}
        variables = {"digis": (("starttdc", np.arange(0, 161, 8)),),
                     "pulses": (("pe", np.r_[0, 5, 10, 15, 20, 30, 50, 80, 120, 200, 400, 1000]),
                                ("time", np.arange(0, 2001, 100))),
                     "clusters": (("pe", np.r_[0, 20, 50, 100, 200, 400, 800, 1600, 3200, 10000]),
                                  ("time", np.arange(0, 2001, 100)))}[name]
        for var, edges in variables:
            reference_field = "avgtime" if name == "clusters" and var == "time" else var
            sv = ak.to_numpy(data[reference_field])
            output[name + "_" + var] = {"edges": edges, "includes_flow_bins": True,
                **paired_counts(histogram(ev, sv, edges, nevents), histogram(fev, fast[var], edges, nevents))}
    return output
