"""Fixed-capacity JAX sampling and joint-frame overlay over resident CSR pools.

All source validation/CSR construction is setup work. Sampling, donor continuation,
truth-pointer remapping and row packing run inside the caller's compiled function.
Capacity excess is a failed batch, never accepted truncation or a NumPy fallback.
"""
from dataclasses import dataclass

import awkward as ak
import numpy as np
from scipy.special import ndtr

from .frames import IDS
from .sampling import STREAMS


@dataclass(frozen=True)
class Layout:
    batch_events: int
    max_donors: int
    rows: dict
    max_selected_rows: int

    def __post_init__(self):
        for name, value in {"batch_events": self.batch_events, "max_donors": self.max_donors,
                            "max_selected_rows": self.max_selected_rows, **self.rows}.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"device capacity {name} must be a positive integer")


def pack_source(source, layout):
    """Validate and pack five host snapshots once; returned arrays are uploaded once."""
    pools = [source.signal, *(source._pools[name] for name in STREAMS)]
    names = set(pools[0].collections)
    if set(layout.rows) != names or any(set(pool.collections) != names for pool in pools):
        raise ValueError("device row capacities and all pool collection sets must match")
    if layout.batch_events > source.population:
        raise ValueError("device batch exceeds signal population")
    for pool in pools:
        pool.validate_references()
    sizes = np.array([len(pool.event_ids) for pool in pools], np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    data = {"ids": np.concatenate([pool.event_ids.astype(np.uint64) for pool in pools]),
            "sizes": sizes, "starts": starts, "collections": {}}
    maxkey = np.concatenate([ak.to_numpy(ak.fill_none(ak.max(ak.values_astype(
        pool.collections["particles"].key, np.int64), axis=1), -1)) for pool in pools])
    bound = np.iinfo(np.int64).max // (1 + 4 * layout.max_donors)
    if np.any(maxkey >= bound):
        raise ValueError("particle keys exceed safe fixed-capacity remapping range")
    data["maxkey"] = maxkey.astype(np.int64)
    for name in sorted(names):
        fields = set(ak.fields(pools[0].collections[name]))
        if any(set(ak.fields(pool.collections[name])) != fields for pool in pools):
            raise ValueError(f"device pool schema differs: {name}")
        counts = np.concatenate([ak.to_numpy(ak.num(pool.collections[name])) for pool in pools])
        columns = {}
        pointers = {"key", "parent", "genidx"} if name == "particles" else ({"idx"} if name == "gen" else {"sp"})
        for field in sorted(fields):
            arrays = [ak.to_numpy(ak.flatten(pool.collections[name][field])) for pool in pools]
            if any(array.ndim != 1 or array.dtype.kind not in "biuf" for array in arrays):
                raise ValueError(f"device columns must be scalar numeric arrays: {name}.{field}")
            dtype = np.dtype(np.int64) if field in pointers else np.result_type(*(array.dtype for array in arrays))
            if dtype.kind == "f" and any(array.dtype.kind in "iu" for array in arrays):
                raise ValueError(f"unsafe mixed integer dtype in {name}.{field}")
            values = np.concatenate([array.astype(dtype) for array in arrays])
            columns[field] = np.r_[values, np.zeros(1, dtype=dtype)]
        data["collections"][name] = {"offsets": np.r_[0, np.cumsum(counts)], "columns": columns}
    law = source._law
    mu = np.log(law.extended_mean) + 0.5 * law.sigma**2
    lower = ndtr((np.log(law.cut_min) - mu) / law.sigma) if law.cut_min else 0.
    upper = ndtr((np.log(law.cut_max) - mu) / law.sigma)
    if upper - lower < 1e-6:
        raise ValueError("requested truncation has less than 1e-6 probability")
    rates = np.array([source._rates[name] for name in STREAMS], np.float64)
    if np.any(rates * law.cut_max > 2**30):
        raise ValueError("device Poisson means exceed supported range")
    skip = np.array([source._campaign["max_skip"][name] for name in STREAMS], np.int64)
    if np.any(skip == np.iinfo(np.int64).max):
        raise ValueError("inclusive device skip bound exceeds int64")
    data.update(rates=rates, skip=skip)
    constants = {"mu": mu, "sigma": law.sigma, "lower": lower, "upper": upper,
                 "cut_min": law.cut_min, "cut_max": law.cut_max}
    return data, constants


def sample_overlay(pools, state, key, ordinal, size, *, layout, law):
    """Pure JAX: return masked mixed rows, receipt, candidate cursor state and errors."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    b, d = layout.batch_events, layout.max_donors
    event_mask = jnp.arange(b) < size
    slots = ((jnp.arange(b, dtype=jnp.uint64) + ordinal) % pools["sizes"][0].astype(jnp.uint64)).astype(jnp.int64)
    intensity_key, poisson_key, skip_key = jax.random.split(key, 3)
    uniform = jax.random.uniform(intensity_key, (b,), dtype=jnp.float64)
    prob = law["lower"] + (law["upper"] - law["lower"]) * uniform
    prob = jnp.clip(prob, jnp.finfo(jnp.float64).tiny, jnp.nextafter(1., 0.))
    intensity = jnp.rint(jnp.clip(jnp.exp(law["mu"] + law["sigma"] * ndtri(prob)),
                                law["cut_min"], np.nextafter(law["cut_max"], law["cut_min"]))).astype(jnp.int64)
    intensity = jnp.where(event_mask, intensity, 0)
    counts = jax.random.poisson(poisson_key, intensity[:, None] * pools["rates"], dtype=jnp.int64)
    counts = jnp.where(event_mask[:, None], counts, 0)
    count_error = jnp.any((counts < 0) | (counts > d))
    bounded = jnp.clip(counts, 0, d)
    initial = jax.random.randint(skip_key, (4,), 0, pools["skip"] + 1, dtype=jnp.int64)
    cursors = jnp.where(state["first"], initial % pools["sizes"][1:], state["cursors"])
    before = jnp.cumsum(bounded, axis=0) - bounded
    donors = (cursors[None, :, None] + before[:, :, None] + jnp.arange(d)) % pools["sizes"][None, 1:, None]
    donor_mask = jnp.arange(d) < bounded[:, :, None]
    frames = donors + pools["starts"][None, 1:, None]
    occurrence = jnp.concatenate([slots[:, None], frames.reshape(b, 4 * d)], axis=1)
    valid = jnp.concatenate([event_mask[:, None], donor_mask.reshape(b, 4 * d)], axis=1)
    pwidth = jnp.where(valid, pools["maxkey"][occurrence] + 1, 0)
    poffset = (jnp.cumsum(pwidth, axis=1) - pwidth).reshape(-1)
    go = pools["collections"]["gen"]["offsets"]
    gwidth = jnp.where(valid, go[occurrence + 1] - go[occurrence], 0)
    goffset = (jnp.cumsum(gwidth, axis=1) - gwidth).reshape(-1)
    occurrence, valid = occurrence.reshape(-1), valid.reshape(-1)
    mixed, totals, row_errors = {}, {}, []
    for name, collection in pools["collections"].items():
        offsets = collection["offsets"]
        lengths = jnp.where(valid, offsets[occurrence + 1] - offsets[occurrence], 0)
        ends = jnp.cumsum(lengths)
        total = ends[-1]
        capacity = layout.rows[name]
        row = jnp.arange(capacity, dtype=jnp.int64)
        owner = jnp.minimum(jnp.searchsorted(ends, row, side="right"), len(occurrence) - 1)
        row_mask = row < total
        source_row = offsets[occurrence[owner]] + row - (ends[owner] - lengths[owner])
        sentinel = len(next(iter(collection["columns"].values()))) - 1
        source_row = jnp.where(row_mask, source_row, sentinel)
        columns = {field: values[source_row] for field, values in collection["columns"].items()}
        if name == "particles":
            columns["key"] = columns["key"] + poffset[owner]
            columns["parent"] = jnp.where(columns["parent"] >= 0, columns["parent"] + poffset[owner], -1)
            columns["genidx"] = jnp.where(columns["genidx"] >= 0, columns["genidx"] + goffset[owner], -1)
        elif name == "gen":
            columns["idx"] = columns["idx"] + goffset[owner]
        else:
            columns["sp"] = columns["sp"] + poffset[owner]
        mixed[name] = {"columns": columns, "mask": row_mask, "event": owner // (1 + 4 * d)}
        totals[name] = total
        row_errors.append(total > capacity)
    selected_rows = sum(totals.values())
    errors = {"donor_capacity": count_error, "row_capacity": jnp.any(jnp.stack(row_errors)),
              "selected_rows": selected_rows > layout.max_selected_rows}
    receipt = {"signal_indices": slots, "signal_event_ids": pools["ids"][slots],
               "intensity": intensity, "multiplicities": counts, "actual_rows": totals,
               "selected_rows_total": selected_rows, "selections": {}}
    for column, stream in enumerate(STREAMS):
        mask = donor_mask[:, column]
        receipt["selections"][stream] = {
            "counts": counts[:, column], "donors": jnp.where(mask, donors[:, column], -1),
            "mask": mask, "source_event_ids": jnp.where(mask[:, :, None], pools["ids"][frames[:, column]], 0),
            "initial_cursor": cursors[column],
            "next_cursor": (cursors[column] + bounded[:, column].sum()) % pools["sizes"][column + 1]}
    next_state = {"first": jnp.array(False),
                  "cursors": (cursors + bounded.sum(axis=0)) % pools["sizes"][1:]}
    batch = {"keys": pools["ids"][slots], "signal_indices": slots,
             "event_mask": event_mask, "collections": mixed, "ordinal": ordinal, "size": size}
    return batch, receipt, next_state, errors
