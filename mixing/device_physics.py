"""Physical tracker + CRV model for the resident JAX streaming backend.

Only setup uses host conditions/timing readers. Device kernels receive explicit
per-source timing and a PDG lookup by remapped truth identity, never nearest time.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _crv_pdg(collections):
    """Pure JAX lexicographic (event, particle key) join with checked identities."""
    import jax.numpy as jnp
    from jax import lax

    particles, crv = collections["particles"], collections["crv"]
    pkey, pvalid = particles["columns"]["key"], particles["mask"]
    invalid_event = jnp.iinfo(jnp.int64).max
    order = jnp.lexsort((pkey, jnp.where(pvalid, particles["event"], invalid_event)))
    pe = jnp.where(pvalid[order], particles["event"][order], invalid_event)
    pk = pkey[order]
    qe, qk = crv["event"], crv["columns"]["sp"]
    low, high = jnp.zeros(qk.shape, jnp.int64), jnp.full(qk.shape, len(pk), jnp.int64)
    def search(_, state):
        low, high = state
        mid = (low + high) // 2
        index = jnp.minimum(mid, len(pk) - 1)
        before = (pe[index] < qe) | ((pe[index] == qe) & (pk[index] < qk))
        return jnp.where(before & (low < high), mid + 1, low), jnp.where(~before & (low < high), mid, high)
    low, high = lax.fori_loop(0, int(len(pk)).bit_length() + 1, search, (low, high))
    row = jnp.minimum(low, len(pk) - 1)
    found = (low < len(pk)) & pvalid[order[row]] & (pe[row] == qe) & (pk[row] == qk)
    raw = particles["columns"]["pdg"]
    invalid_pdg = ~jnp.isfinite(raw)
    if raw.dtype.kind == "f":
        invalid_pdg |= (raw != jnp.floor(raw)) | (raw < -(2.**63)) | (raw >= 2.**63)
    elif raw.dtype.kind == "u":
        invalid_pdg |= raw >= jnp.uint64(2**63)
    elif raw.dtype.kind != "i":
        raise ValueError("particle PDG must be a numeric integer-valued column")
    safe = jnp.where(invalid_pdg, 0, raw).astype(jnp.int64)
    pdg = jnp.where(found, safe[order[row]], 0)
    error = jnp.any(crv["mask"] & ~found) | jnp.any(pvalid & invalid_pdg)
    return {**crv, "columns": {**crv["columns"], "pdg": pdg}}, error


def combine(tracker, crv, timing, *, synthetic_inputs=False):
    """Join prepared physical adapters; suitable for real or explicitly synthetic inputs."""
    def apply(batch, assets, crv_key, tracker_key):
        import jax.numpy as jnp
        times = {name: value[batch["signal_indices"]] for name, value in assets["timing"].items()}
        times["event_mask"] = batch["event_mask"]
        detector, reference_error = _crv_pdg(batch["collections"])
        td, te, tu = tracker.apply_with_usage(batch["collections"]["tracker"], times, assets["tracker"], tracker_key)
        cd, ce, cu = crv.apply_with_usage(detector, times, assets["crv"], crv_key)
        trk_columns = {"ev": td["ev"], "mask": td["mask"], "sid": td["sid"].astype(jnp.uint32),
                       "pmp": td["pmp"].astype(jnp.uint32), "adc": td["adc"].astype(jnp.uint32),
                       "nadc": jnp.full(td["sid"].shape, td["adc"].shape[1], jnp.uint32),
                       "flags": jnp.zeros(td["sid"].shape, jnp.uint32)}
        for name in ("tdc", "tot"):
            for end in (0, 1):
                trk_columns[name + str(end)] = td[name][:, end].astype(jnp.uint32)
        errors = {"truth_pdg": reference_error, **{"tracker/" + name: value for name, value in te.items()},
                  **{"crv/" + name: value for name, value in ce.items()}}
        usage = {**{"tracker/" + name: value for name, value in tu.items()},
                 **{"crv/" + name: value for name, value in cu.items()}}
        return {"tracker": trk_columns, **cd, "timing": times, "usage": usage}, errors
    return SimpleNamespace(kind="physics", support=["tracker_digis", "crv_reco"], apply=apply,
                           assets={"tracker": tracker.assets, "crv": crv.assets, "timing": timing},
                           metadata={"scope": "bounded physical tracker digis/ADCs and CRV digis/pulses/coincidences; no tracker reconstruction",
                                     "synthetic_inputs": synthetic_inputs,
                                     "tracker": tracker.metadata, "crv": crv.metadata,
                                     "validation": "kernel/oracle and bounded integration only; no production closure or throughput claim"})


def model(args, source, layout):
    """Load existing saved production assets once; no new transport or conditions inference."""
    from crvfastsim import runconfig
    from crvfastsim.response import photons, sipm
    from trkfastsim.digi import digicond
    from . import crv_inputs, device_tracker, device_crv

    capacity = json.loads(Path(args.device_capacities).read_text())["response"]
    if set(capacity) - {"tracker_reuse"} != {"tracker", "crv"}:
        raise ValueError("physical response capacities require tracker and crv (and optionally tracker_reuse)")
    config = digicond.parse_fhicl_dump(Path(args.expanded).read_text())
    cfg = crv_inputs.build_config(config, jax=True)
    rc = runconfig.get("run1a_v01")
    geom, counters = rc.geometry(), rc.counters_cfg()
    cond = crv_inputs.load_conditions(args.truth, args.photon_csv, geom, counters)
    mc, reco, length = crv_inputs.read_timing(source.signal.event_ids, args.truth, args.stock)
    conditions = digicond.Conditions.load(args.tracker_conditions)
    if "tracker_reuse" in capacity:   # opt-in approximation: donor-frame digi reuse (mixing.device_reuse)
        from . import device_reuse
        from .device_source import pack_source
        data, _ = pack_source(source, layout)
        table = device_reuse.precompute(conditions, capacity["tracker_reuse"], data, float(np.max(length)),
                                        getattr(args, "tracker_seed", 0), rows_capacity=layout.rows["tracker"],
                                        shift=float(np.mean(mc)),
                                        margin=float(np.max(np.abs(np.asarray(mc) - np.mean(mc)))) + 5.0, log=print)
        tracker = device_reuse.prepare(conditions, capacity["tracker_reuse"], layout.batch_events, table,
                                       data["starts"])
    else:
        tracker = device_tracker.prepare(conditions, capacity["tracker"], layout.batch_events)
    from crvfastsim.response import lookup     # the optical assets pinned by the run configuration
    with lookup.pinned(rc.asset_pin_map() if hasattr(rc, "asset_pin_map") else None):
        tabs, pmap = photons.load_tables(cfg.response), sipm.photon_map()
    crv = device_crv.prepare(geom, cond, cfg, counters, capacity["crv"], tabs, pmap)
    return combine(tracker, crv, {"mc_pbtime": np.asarray(mc), "reco_pbtime": np.asarray(reco),
                                  "event_length": np.asarray(length)})
