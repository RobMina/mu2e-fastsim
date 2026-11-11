"""Sampled tracker rows (opt-in approximation W2-W4 ``tracker_rows=sampled``): the response layer's ``nrow`` drawn
from a library instead of transported.

The reco response (:mod:`trkfastsim.recoresp.device`) reads a track's StrawGasStep rows only through the bin of
``nrow`` it falls in. When the tracker steps themselves are not needed (track observables without digis), the
tracker and albedo transport can be replaced by a draw of ``nrow`` given the entry helix: the library holds, per
cell of (|p|, tan dip, r_max) at the tracker entry, the rows the default device chain produced (tracker transport
plus the calorimeter albedo) on training electrons. A cell without training tracks (fewer than ``min_count``, default
1) falls back to its (|p|, tan dip) cell, then to every track. A fallback that drops r_max biases the sparse
low-momentum cells: with 20 instead of 1, the RPC fit efficiency came out 3 % low.

    tab = build(has_entry, nrow, x, p, axis_xy, bz)        # host, from a training run of the default chain
    save(path, tab); dn = to_device(load(path))
    nrow = sample(dn, key, has_entry, x, p)                 # device, per event (0 without a tracker entry)

The shipped library (``DEFAULT``) was built from the default chain on Run 1A geometry and field (conversion, DIO and
RPC electrons, training draws disjoint from the benchmark inputs); its README records the inputs.

The library follows the shared library core (:mod:`mu2efastsim.library`, core/docs/libraries.md): its manifest is
embedded by :func:`save` (``provenance`` fields) or, for the frozen shipped file, the sidecar
``data/nrow_run1a.manifest.json``; :func:`load` checks the manifest, the arrays and their content hash; the cell axes
are ``Axis`` objects with the ``clip`` policy, and :func:`domain_counts` counts the clipped values and the fallback
levels a set of entries uses.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np

from mu2efastsim.library import axes as LA
from mu2efastsim.library import store as LS
from mu2efastsim.library.domain import DomainCounter
from trkfastsim.recoresp import table as RT

SCHEMA = 1
DEFAULT = Path(__file__).with_name("data") / "nrow_run1a.npz"
NAMED = {"default": DEFAULT}
# cell edges (values outside go to the first or last cell)
P_EDGES = np.concatenate([[0.0, 30.0], np.arange(40.0, 96.0, 2.5), [97.0, 99.0, 100.5, 101.5, 102.5, 103.25, 104.0,
                                                                   104.75, 106.0, 108.0, 110.0, 115.0, 120.0, 200.0]])
TANDIP_EDGES = np.round(np.arange(0.30, 1.501, 0.05), 6)
RMAX_EDGES = np.round(np.arange(400.0, 760.1, 10.0), 6)


def resolve(library):
    """Path of a library: a shipped name (NAMED) or a file."""
    if isinstance(library, str) and library in NAMED:
        return NAMED[library]
    return Path(library)


LIBRARY = "tracker.nrow.run1a"
FALLBACK = "cell_level"          # 0 fine cell, 1 its (|p|, tan dip) cell, 2 every track
_bins = LA.locate                # bin k: edges[k] <= v < edges[k + 1]; outside: the end bins (policy clip)


def build(has_entry, nrow, x, p, axis_xy, bz, *, p_edges=P_EDGES, tandip_edges=TANDIP_EDGES, rmax_edges=RMAX_EDGES,
          min_count=1, reference=""):
    """Library from a training run (host arrays): has_entry [n] bool, nrow [n] rows of the default chain, x, p [n, 3]
    entry state (Mu2e frame), axis_xy and bz as the response table's."""
    has = np.asarray(has_entry, bool)
    nrow = np.asarray(nrow)[has].astype(np.int64)
    if nrow.size == 0:
        raise ValueError("no training track with a tracker entry")
    if nrow.max() > np.iinfo(np.int16).max:
        raise ValueError("nrow beyond int16")
    pm, td, _, rmax = RT.entry_truth(np.asarray(x, float)[has], np.asarray(p, float)[has], axis_xy, bz)
    ip, it, ir = _bins(p_edges, pm), _bins(tandip_edges, td), _bins(rmax_edges, rmax)
    npb, ntb, nrb = len(p_edges) - 1, len(tandip_edges) - 1, len(rmax_edges) - 1
    fine = (ip * ntb + it) * nrb + ir
    coarse = ip * ntb + it
    pools, start = [], 0

    def add(values):
        nonlocal start
        pools.append(np.sort(values))
        s = start
        start += len(values)
        return s, len(values)

    fine_order = np.argsort(fine, kind="stable")
    fine_cells, fine_first, fine_n = np.unique(fine[fine_order], return_index=True, return_counts=True)
    fine_at = {}
    for c, f, k in zip(fine_cells, fine_first, fine_n):
        if k >= min_count:
            fine_at[int(c)] = add(nrow[fine_order[f:f + k]])
    coarse_at = {}
    for c in np.unique(coarse):
        v = nrow[coarse == c]
        if len(v) >= min_count:
            coarse_at[int(c)] = add(v)
    every = add(nrow)
    n_fine = npb * ntb * nrb
    cell_start = np.empty(n_fine, np.int64)
    cell_count = np.empty(n_fine, np.int64)
    level = np.empty(n_fine, np.int8)          # 0 fine, 1 coarse, 2 every track
    for c in range(n_fine):
        if c in fine_at:
            (cell_start[c], cell_count[c]), level[c] = fine_at[c], 0
        elif c // nrb in coarse_at:
            (cell_start[c], cell_count[c]), level[c] = coarse_at[c // nrb], 1
        else:
            (cell_start[c], cell_count[c]), level[c] = every, 2
    used = level[fine]
    return dict(schema=np.int64(SCHEMA), p_edges=np.asarray(p_edges, float), tandip_edges=np.asarray(tandip_edges, float),
                rmax_edges=np.asarray(rmax_edges, float), cell_start=cell_start, cell_count=cell_count,
                cell_level=level, pool=np.concatenate(pools).astype(np.int16), axis_xy=np.asarray(axis_xy, float),
                bz=np.float64(bz), min_count=np.int64(min_count), n_training=np.int64(len(nrow)),
                training_level_fraction=np.array([np.mean(used == k) for k in range(3)]), reference=np.str_(reference))


def axes(tab):
    """The cell axes of a library (mu2efastsim.library.axes.Axis), in cell order."""
    frame = "tracker entry helix (Mu2e frame; axis_xy, bz of the library)"
    return (LA.Axis("p", "MeV/c", tab["p_edges"], frame=frame, array="p_edges", policy="clip",
                    description="|p| at the tracker entry"),
            LA.Axis("tandip", "1", tab["tandip_edges"], frame=frame, array="tandip_edges", policy="clip",
                    description="tan(dip) = p_z / p_T at the tracker entry"),
            LA.Axis("rmax", "mm", tab["rmax_edges"], frame=frame, array="rmax_edges", policy="clip",
                    description="maximum radius of the entry helix from the tracker axis"))


# Provenance of the shipped library (data/README.md; built by trkfastsim.recoresp.nrow.build from a training run)
SHIPPED = dict(
    version=1,
    source=dict(description="training electrons of the default device chain, Run 1A geometry and field: 700,000 "
                            "electrons, 503,361 with a tracker entry",
                samples=[dict(dataset="conversion electrons from the MuminusStopsCat stop table", selection="births "
                              "with seeds 1001+ (one seed per 20,000 births)", events=None),
                         dict(dataset="DIO starts", selection="shards 0-49", events=None),
                         dict(dataset="RPC starts", selection="shards 0-1", events=None)]),
    generator=dict(name="trkfastsim device chain (beamfastsim.cetracker.DeviceChain defaults: tracker transport "
                        "and calorimeter albedo, response 'default')", release="mu2e-fastsim", physics_list=None,
                   configuration="float64, Run 1A geometry and field"),
    seeds=dict(ce_births="1001 + chunk", ce_transport="5001 + chunk", dio_transport="7001 + chunk",
               rpc_transport="8001 + chunk", chunk="20,000 electrons"),
    geometry=dict(tag="Run 1A (geom_run1_a)", hash=None),
    split=dict(training="CE births seeds 1001+, DIO shards 0-49, RPC shards 0-1",
               held_out="benchmark draws: CE seeds 1-3, DIO shards 50-249, RPC shards 2-3",
               rule="disjoint seeds and shards"),
    validation=dict(status="tested, not closed", record="docs/approximations.md (W2-W4 tracker_rows=sampled): CE, DIO "
                    "and RPC fit, quality and CE-selection efficiencies within their paired errors against the "
                    "transported rows (100,000 each, float32 and float64); no declared criterion recorded"),
    unrecorded=dict(release="built from a working tree of the PR #169 branch on main 0cac5df; the commit of the "
                            "build is not recorded"),
    legacy=dict(schema=SCHEMA, format="npz without a manifest (PR #169)"),
)


def manifest_fields(tab, **provenance):
    """Manifest fields of a library (mu2efastsim.library.manifest): the description from the arrays, the provenance
    from ``provenance`` (fields not given are recorded as unrecorded, with the reason)."""
    fields = dict(
        name=LIBRARY, subsystem="tracker", version=1, kind="pool",
        description="Per cell of the tracker-entry helix (|p|, tan dip, r_max), the StrawGasStep row counts (nrow) "
                    "of the default device chain on training electrons; a lookup draws one row count from the "
                    "event's cell (opt-in approximation tracker_rows=sampled).",
        axes=[a.describe() for a in axes(tab)],
        outputs=[dict(name="nrow", unit="1", dtype="int32 (device) / int64 (host)",
                      description="StrawGasStep rows of the track; 0 without a tracker entry")],
        dtype=dict(storage={"pool": "int16", "cell_start": "int64", "cell_count": "int64", "edges": "float64"},
                   compute=["float32", "float64"],
                   notes="edges and the entry helix are cast to the compute dtype; indices int32 on the device"),
        rng=dict(host="numpy.random.Generator: one uniform per row (sample_numpy)",
                 device="jax.random key: one uniform per row (sample); the chain passes fold_in(key, 6)",
                 draws_per_row=1),
        domain=dict(coverage="|p| 0-200 MeV/c, tan dip 0.30-1.50, r_max 400-760 mm; values outside go to the end "
                             "cells (policy clip)",
                    fallbacks=[dict(name=FALLBACK, levels={"0": "fine cell", "1": "(|p|, tan dip) cell",
                                                           "2": "every training track"},
                                    description="a fine cell with fewer than min_count training tracks draws from "
                                                "its coarser cell; cell_level records which")]),
        config_hash=None,
    )
    reasons = dict(source="not given to save()", generator="not given to save()", release="not given to save()",
                   seeds="not given to save()", geometry="not given to save()", split="not given to save()",
                   config_hash="no build configuration manifest")
    fields.update(provenance)
    if "validation" not in provenance:
        fields["validation"] = dict(status="not validated", record=None)
    unrecorded = dict(provenance.get("unrecorded", {}))
    for k, why in reasons.items():
        if fields.get(k) is None:
            fields[k] = None
            unrecorded.setdefault(k, why)
    fields["unrecorded"] = unrecorded
    return fields


def save(path, tab, **provenance):
    """Write a library with its manifest (provenance: manifest fields, see :func:`manifest_fields`)."""
    return LS.save(Path(path), tab, manifest_fields(tab, **provenance))


def write_shipped_manifest(path=DEFAULT):
    """(Re)write the sidecar manifest of the frozen shipped library from SHIPPED; the .npz is not touched."""
    tab = LS.load_arrays(path)
    return LS.write_sidecar(path, manifest_fields(tab, **SHIPPED))


def load_library(library):
    """The checked library (mu2efastsim.library.store.Library: manifest, read-only arrays, provenance)."""
    lib = LS.load(resolve(library), expect={"name": LIBRARY})
    if int(lib.arrays["schema"]) != SCHEMA:
        raise ValueError(f"nrow library schema {int(lib.arrays['schema'])}, expected {SCHEMA}")
    return lib


def load(library):
    """The library's arrays (read-only) after the checks of :func:`load_library`."""
    return load_library(library).as_dict()


def domain_counts(tab, has_entry, x, p, counter=None):
    """DomainCounter of a set of entries (host): values outside each cell axis (clipped) and the fallback level of
    the cells they use. Rows without a tracker entry are not lookups."""
    has = np.asarray(has_entry, bool)
    pm, td, _, rmax = RT.entry_truth(np.asarray(x, float), np.asarray(p, float), tab["axis_xy"], float(tab["bz"]))
    counter = counter or DomainCounter(LIBRARY)
    counter.add_lookups(int(has.sum()))
    for a, v in zip(axes(tab), (pm, td, rmax)):
        counter.add_values(a.name, a.edges, v, mask=has)
    ntb, nrb = len(tab["tandip_edges"]) - 1, len(tab["rmax_edges"]) - 1
    c = (_bins(tab["p_edges"], pm) * ntb + _bins(tab["tandip_edges"], td)) * nrb + _bins(tab["rmax_edges"], rmax)
    counter.add_levels(FALLBACK, tab["cell_level"][c], mask=has)
    return counter


def sample_numpy(tab, rng, has_entry, x, p):
    """Host twin of :func:`sample` (tests)."""
    has = np.asarray(has_entry, bool)
    pm, td, _, rmax = RT.entry_truth(np.asarray(x, float), np.asarray(p, float), tab["axis_xy"], float(tab["bz"]))
    ntb, nrb = len(tab["tandip_edges"]) - 1, len(tab["rmax_edges"]) - 1
    c = (_bins(tab["p_edges"], pm) * ntb + _bins(tab["tandip_edges"], td)) * nrb + _bins(tab["rmax_edges"], rmax)
    cnt = tab["cell_count"][c]
    j = tab["cell_start"][c] + np.minimum((rng.uniform(size=len(c)) * cnt).astype(np.int64), cnt - 1)
    return np.where(has, tab["pool"][j], 0).astype(np.int64)


class DeviceNrow(NamedTuple):
    p_edges: object
    tandip_edges: object
    rmax_edges: object
    cell_start: object
    cell_count: object
    pool: object
    axis_xy: object
    bz: float


def to_device(tab, dtype=np.float64):
    import jax.numpy as jnp
    dt = np.dtype(dtype)
    return DeviceNrow(jnp.asarray(tab["p_edges"], dt), jnp.asarray(tab["tandip_edges"], dt),
                      jnp.asarray(tab["rmax_edges"], dt), jnp.asarray(tab["cell_start"], jnp.int32),
                      jnp.asarray(tab["cell_count"], jnp.int32), jnp.asarray(tab["pool"], jnp.int32),
                      jnp.asarray(tab["axis_xy"], dt), float(tab["bz"]))


def sample(dn, key, has_entry, x, p):
    """nrow per event (int32 device array; 0 without a tracker entry): one uniform draw from the event's cell."""
    import jax
    import jax.numpy as jnp
    from trkfastsim.recoresp import device as RD
    dt = x.dtype
    pm, td, _, rmax = RD.entry_truth(x, p, dn.axis_xy.astype(dt), dn.bz)
    ntb, nrb = dn.tandip_edges.shape[0] - 1, dn.rmax_edges.shape[0] - 1
    c = (LA.locate_jax(dn.p_edges.astype(dt), pm) * ntb + LA.locate_jax(dn.tandip_edges.astype(dt), td)) * nrb \
        + LA.locate_jax(dn.rmax_edges.astype(dt), rmax)
    cnt = dn.cell_count[c]
    u = jax.random.uniform(key, (x.shape[0],), dt)
    j = dn.cell_start[c] + jnp.minimum((u * cnt).astype(jnp.int32), cnt - 1)
    return jnp.where(has_entry, dn.pool[j], 0)
