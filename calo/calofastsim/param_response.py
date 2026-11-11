"""Parametric digi response (opt-in approximation W7 ``digi_response=parametric``): crystal hits drawn from a table
instead of photoelectrons, waveforms, digis and the template fit.

Per (event, crystal) the visible energy ``E`` (Birks and longitudinal-response corrected energy of the accepted
steps, as :func:`calofastsim.device_chain.photoelectrons` computes it) and its energy-weighted folded step time ``T``
form a cell. The table holds, per bin of log ``E``, a pool of the full chain's outcomes on training cells well inside
the digitisation window: whether the cell gave a hit, the ratio of the hit energy to ``E`` and the hit time minus
``T`` (the highest-energy hit of the cell; the reco bunch-time smear off). Near the end of the window a pulse no
longer fits in the waveform: a survival curve in ``v = T - window end`` (the event's end, as the step-time folding
computes it) multiplies the hit probability. A cell draws one pool entry of its energy bin; its hit is
``(E * ratio, T + dt)`` less the event's reco bunch-time smear. The joint draw keeps the correlation of the energy and time response at a
given ``E``; what the table cannot carry: the dependence on how the energy is spread in time inside a crystal, the
second hit of a cell (deposits more than ``time4merge`` apart) and hits without a deposit (noise alone).

    tab = build(energy, has_hit, ratio, dt, v)           # host, from a training run of the full chain
    save(path, tab); tb = to_device(load(path))
    ok, ev, crystal, E, T, over = cells(...); hit, e, t = draw(tb, key, ok, E, T, v)   # device
    (device_chain.make_steps_chain / make_chain with digi_response= run these)

The shipped table (``DEFAULT``) is built from the full device chain with the production response (CAL1: Sim_crystal,
CsI pulse, readout noise, propagation table) on the CAL2 training CE files; data/README.md records the inputs.

The table is a library of the shared core (mu2efastsim.library, core/docs/libraries.md): save() embeds a manifest,
the frozen shipped file has the sidecar data/digi_response_run1a.manifest.json, load() checks the manifest, the
arrays and their content hash, and domain_counts() counts the energies clipped to the end bins, the cells of empty
energy bins (they borrow the nearest filled bin) and the cells past the last v edge.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np

from mu2efastsim.library import axes as LA
from mu2efastsim.library import store as LS
from mu2efastsim.library.domain import DomainCounter

SCHEMA = 2
DEFAULT = Path(__file__).with_name("data") / "digi_response_run1a.npz"
NAMED = {"default": DEFAULT}
E_EDGES = np.geomspace(0.01, 300.0, 61)       # MeV; below the first edge -> first bin, above the last -> last bin
POOL_MAX = 4000
V_EDGES = np.arange(-200.0, 40.01, 2.0)     # ns, T - window end: cells below the first edge are inside the window
SURVIVAL_E_MIN = 3.0                        # MeV: cells used for the survival curve (hit probability ~ 1 inside)
LIBRARY = "calo.digi_response.run1a"


def resolve(table):
    if isinstance(table, str) and table in NAMED:
        return NAMED[table]
    return Path(table)


def build(energy, has_hit, ratio, dt, v, *, e_edges=E_EDGES, v_edges=V_EDGES, pool_max=POOL_MAX, seed=1,
          reference=""):
    """Table from training cells (host arrays, one row per (event, crystal) cell with E > 0): visible energy, whether
    the full chain gave the cell a hit, hit energy / E, hit time - T (NaN without a hit) and v = T - window end. The
    pools take the cells with v below the first v edge; each energy bin keeps at most pool_max cells (a uniform
    subsample). The survival curve is the hit fraction per v bin over the inside hit fraction, cells above
    SURVIVAL_E_MIN; an empty v bin takes the value of the nearest filled bin."""
    energy, has_hit, v = np.asarray(energy, float), np.asarray(has_hit, bool), np.asarray(v, float)
    ratio, dt = np.where(has_hit, ratio, 0.0), np.where(has_hit, dt, 0.0)
    inside = v < v_edges[0]
    hi = energy > SURVIVAL_E_MIN
    p_in = has_hit[inside & hi].mean()
    nv = len(v_edges) - 1
    vb = np.searchsorted(v_edges, v, side="right") - 1
    surv = np.full(nv, np.nan)
    vcount = np.zeros(nv, np.int64)
    for k in range(nv):
        m = hi & (vb == k)
        vcount[k] = m.sum()
        if m.any():
            surv[k] = min(1.0, has_hit[m].mean() / p_in)
    if np.isnan(surv).all():
        raise ValueError("no training cells near the window end")
    filled = np.flatnonzero(~np.isnan(surv))
    surv = surv[filled[np.argmin(np.abs(filled[None, :] - np.arange(nv)[:, None]), axis=1)]]
    energy, has_hit, ratio, dt = energy[inside], has_hit[inside], ratio[inside], dt[inside]
    nb = len(e_edges) - 1
    b = np.clip(np.searchsorted(e_edges, energy, side="right") - 1, 0, nb - 1)
    rng = np.random.default_rng(seed)
    start, count, idx = np.zeros(nb, np.int64), np.zeros(nb, np.int64), []
    for k in range(nb):
        rows = np.flatnonzero(b == k)
        if len(rows) > pool_max:
            rows = np.sort(rng.choice(rows, pool_max, replace=False))
        start[k], count[k] = sum(len(i) for i in idx), len(rows)
        idx.append(rows)
    empty = count == 0
    if empty.all():
        raise ValueError("no training cells")
    # an empty bin borrows the nearest filled bin's pool
    filled = np.flatnonzero(~empty)
    for k in np.flatnonzero(empty):
        j = filled[np.argmin(np.abs(filled - k))]
        start[k], count[k] = start[j], count[j]
    rows = np.concatenate(idx)
    return dict(schema=np.int64(SCHEMA), e_edges=np.asarray(e_edges, float), start=start, count=count,
                has_hit=has_hit[rows], ratio=ratio[rows].astype(np.float32), dt=dt[rows].astype(np.float32),
                v_edges=np.asarray(v_edges, float), survival=surv, survival_count=vcount,
                n_training=np.int64(len(inside)), n_inside=np.int64(inside.sum()), empty_bins=empty,
                reference=np.str_(reference))


def axes(tab):
    """The cell keys (mu2efastsim.library.axes.Axis): visible energy and v = T - window end."""
    frame = "(event, crystal) cell of the calorimeter"
    return (LA.Axis("E", "MeV", tab["e_edges"], frame=frame, array="e_edges", scale="log", policy="clip",
                    description="visible energy of the cell (Birks and longitudinal-response corrected)"),
            LA.Axis("v", "ns", tab["v_edges"], frame=frame, array="v_edges", scale="linear", policy="clip",
                    description="energy-weighted folded step time minus the event's window end; below the first "
                                "edge: inside the window (survival 1), above the last: the last bin"))


# Provenance of the shipped table (data/README.md and the build record of PR #168)
SHIPPED = dict(
    version=1,
    source=dict(description="CaloShowerSteps of ten CeEndpoint training files through the full device chain "
                            "(photoelectrons, digitisation, template fit, hits), float64, production response; "
                            "28,265 events, 312,265 cells (281,080 inside the window), read in sorted file order",
                samples=[dict(dataset="dts.mu2e.CeEndpoint.MDC2025ba (CaloLibraryDumper tables)",
                              selection="001430_00000001, 101, ..., 901", events=28265)]),
    generator=dict(name="calofastsim device chain (make_steps_chain stages) with the production response",
                   release="calofastsim at #168", physics_list=None,
                   configuration="Sim_crystal.txt, CsI-waveform-2020-02-12, ReadoutNoise, CsI-propag-2020-07-29, "
                                 "calo_geometry_run1430; adc2mev stockio.adc_to_mev; PE statistics, travel time and "
                                 "noise on; reco bunch-time smear off; caps per event: steps 80, photoelectrons "
                                 "9000, rows 48, digis 48, hits 32; digi_length 96, event_hits 64, rounds 8; batches "
                                 "of 1,024 events, no overflow"),
    release=dict(repository="RobMina/mu2e-fastsim", commit="856aa3b (branch approx-w7param, merged as #168)",
                 entry_point="calofastsim.param_response.build (defaults: E_EDGES, V_EDGES, SURVIVAL_E_MIN 3 MeV, "
                             "POOL_MAX 4000, seed 1), from a copy of that working tree"),
    seeds=dict(build="numpy default_rng(1) (pool subsampling)",
               chain="jax.random.PRNGKey(4000 + batch), split into (k_pe, k_digi, unused)",
               timing="numpy default_rng(77000 + batch): pbtime = -200 - 5 * integers(0, 5) ns, mbtime 1695 ns, "
                      "event length 1700 ns, marker 200, crystal length 200, on-spill"),
    geometry=dict(tag="Run 1A calorimeter (calo_geometry_run1430)", hash=None),
    split=dict(training="CeEndpoint MDC2025ba 001430_00000001, 101, ..., 901",
               held_out="001430_00001000-1400", rule="disjoint files"),
    validation=dict(status="tested, not closed",
                    record="docs/approximations.md (W7 digi_response=parametric): leading-cluster peak, width, "
                           "yield and shapes on held-out CE events, float32 and float64; the cluster count above "
                           "10 MeV misses (chi2 17/5); no declared criterion recorded"),
    legacy=dict(schema=SCHEMA, format="npz without a manifest (PR #168)"),
    unrecorded=dict(config_hash="no build configuration manifest"),
)


def manifest_fields(tab, **provenance):
    """Manifest fields of a table (mu2efastsim.library.manifest); provenance fields not given are unrecorded."""
    fields = dict(
        name=LIBRARY, subsystem="calo", version=1, kind="pool",
        description="Parametric digi response: per log bin of the visible cell energy, a pool of the full chain's "
                    "outcomes (hit, hit energy / E, hit time - T) on training cells inside the digitisation window, "
                    "and a survival curve in v = T - window end that multiplies the hit probability.",
        axes=[a.describe() for a in axes(tab)],
        outputs=[dict(name="hit", unit="1", dtype="bool", description="the cell gives a hit"),
                 dict(name="e", unit="MeV", dtype="float32", description="hit energy E * ratio"),
                 dict(name="t", unit="ns", dtype="float32", description="hit time T + dt (before the reco "
                                                                       "bunch-time smear)")],
        dtype=dict(storage={"ratio": "float32", "dt": "float32", "edges": "float64", "survival": "float64"},
                   compute=["float32", "float64"],
                   notes="to_device keeps fixed dtypes in both modes: float64 edges, survival and draws, float32 "
                         "ratio and dt; the outputs are float32"),
        rng=dict(host="none (build: numpy default_rng(seed) for the pool subsample)",
                 device="jax.random key (draw): split into the pool draw and the survival draw, one float64 "
                        "uniform each per cell", draws_per_row=2),
        domain=dict(coverage="E 0.01-300 MeV (clipped to the end bins); v -200 to 40 ns (below: inside the window, "
                             "survival 1; above: the last bin)",
                    fallbacks=[dict(name="empty_bin", levels={"0": "the energy bin's own pool",
                                                              "1": "the nearest filled bin's pool"},
                                    description="an empty energy bin borrows the nearest filled bin's pool "
                                                "(empty_bins)")]),
        config_hash=None,
    )
    fields.update(provenance)
    if "validation" not in provenance:
        fields["validation"] = dict(status="not validated", record=None)
    unrecorded = dict(provenance.get("unrecorded", {}))
    for k in ("source", "generator", "release", "seeds", "geometry", "split"):
        if fields.get(k) is None:
            fields[k] = None
            unrecorded.setdefault(k, "not given to save()")
    unrecorded.setdefault("config_hash", "no build configuration manifest")
    fields["unrecorded"] = unrecorded
    return fields


def save(path, tab, **provenance):
    """Write a table with its manifest (provenance: manifest fields, see :func:`manifest_fields`)."""
    return LS.save(Path(path), tab, manifest_fields(tab, **provenance))


def write_shipped_manifest(path=DEFAULT):
    """(Re)write the sidecar manifest of the frozen shipped table from SHIPPED; the .npz is not touched."""
    return LS.write_sidecar(path, manifest_fields(LS.load_arrays(path), **SHIPPED))


def load(table):
    """The table's arrays (read-only) after the library checks (manifest, arrays, content hash) and the schema."""
    tab = LS.load(resolve(table), expect={"name": LIBRARY}).as_dict()
    if int(tab["schema"]) != SCHEMA:
        raise ValueError(f"digi response table schema {int(tab['schema'])}, expected {SCHEMA}")
    return tab


def domain_counts(tab, E, v, ok=None, counter=None):
    """DomainCounter of a set of cells (host): energies and v outside their bins, cells of empty energy bins."""
    E, v = np.asarray(E, float), np.asarray(v, float)
    m = np.ones(E.shape, bool) if ok is None else np.asarray(ok, bool)
    counter = counter or DomainCounter(LIBRARY)
    counter.add_lookups(int(m.sum()))
    counter.add_values("E", tab["e_edges"], E, mask=m)
    counter.add_codes("v", np.where(np.isfinite(v), np.where(v >= tab["v_edges"][-1], LA.ABOVE, LA.IN),
                                    LA.NONFINITE), mask=m)          # below the first edge is in the window, not out
    nb = len(tab["e_edges"]) - 1
    b = np.clip(np.searchsorted(tab["e_edges"], E, side="right") - 1, 0, nb - 1)
    counter.add_levels("empty_bin", tab["empty_bins"][b].astype(np.int64), mask=m)
    return counter


class DeviceTable(NamedTuple):
    log_edges: object
    start: object
    count: object
    has_hit: object
    ratio: object
    dt: object
    v_edges: object
    survival: object


def to_device(tab):
    import jax.numpy as jnp
    return DeviceTable(jnp.asarray(np.log(tab["e_edges"]), jnp.float64), jnp.asarray(tab["start"], jnp.int32),
                       jnp.asarray(tab["count"], jnp.int32), jnp.asarray(tab["has_hit"], bool),
                       jnp.asarray(tab["ratio"], jnp.float32), jnp.asarray(tab["dt"], jnp.float32),
                       jnp.asarray(tab["v_edges"], jnp.float64), jnp.asarray(tab["survival"], jnp.float64))


def cells(ev, crystal, energy, time, keep, n_crystals, n_cells):
    """Per-cell sums of the kept steps on the device: (ok, event, crystal, E, T, overflow); the cell slots are
    sorted by (event, crystal), at most n_cells."""
    import jax
    import jax.numpy as jnp
    big = jnp.iinfo(jnp.int64).max
    k = jnp.where(keep, ev.astype(jnp.int64) * n_crystals + crystal.astype(jnp.int64), big)
    ukey = jnp.unique(k, size=n_cells, fill_value=big)
    slot = jnp.clip(jnp.searchsorted(ukey, k), 0, n_cells - 1)
    found = keep & (ukey[slot] == k)
    e = energy.astype(jnp.float64)
    E = jax.ops.segment_sum(jnp.where(found, e, 0.0), slot, num_segments=n_cells)
    ET = jax.ops.segment_sum(jnp.where(found, e * time, 0.0), slot, num_segments=n_cells)
    ok = (ukey != big) & (E > 0)
    T = jnp.where(ok, ET / jnp.where(E > 0, E, 1.0), 0.0)
    distinct = jnp.sum((jnp.sort(k) != big) & jnp.concatenate([jnp.ones(1, bool), jnp.sort(k)[1:] != jnp.sort(k)[:-1]]))
    return ok, jnp.where(ok, ukey // n_crystals, 0), jnp.where(ok, ukey % n_crystals, 0), E, T, \
        jnp.maximum(distinct - n_cells, 0)


def draw(tb: DeviceTable, key, ok, E, T, v):
    """(hit, e, t) per cell from the table (t before the reco bunch-time smear); v = T - the event's window end."""
    import jax
    import jax.numpy as jnp
    nb = tb.start.shape[0]
    b = jnp.clip(jnp.searchsorted(tb.log_edges, jnp.log(jnp.maximum(E, 1e-30)), side="right") - 1, 0, nb - 1)
    k1, k2 = jax.random.split(key)
    u = jax.random.uniform(k1, E.shape, jnp.float64)
    cnt = tb.count[b]
    j = tb.start[b] + jnp.minimum((u * cnt).astype(jnp.int32), cnt - 1)
    nv = tb.survival.shape[0]
    vb = jnp.searchsorted(tb.v_edges, v, side="right") - 1
    s = jnp.where(vb < 0, 1.0, tb.survival[jnp.clip(vb, 0, nv - 1)])
    hit = ok & tb.has_hit[j] & (jax.random.uniform(k2, E.shape, jnp.float64) < s)
    return hit, (E * tb.ratio[j]).astype(jnp.float32), (T + tb.dt[j]).astype(jnp.float32)
