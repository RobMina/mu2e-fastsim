"""Schema-v2 donor and full-generator fixtures on a tiny two-sector geometry (multi-primary tests)."""
import numpy as np
import pandas as pd
import uproot

from crvfastsim import genassoc, tier1b


def _write(path, frames):
    with uproot.recreate(path) as root:
        for name, frame in frames.items():
            columns = {column: frame[column].to_numpy() for column in frame}
            root.mktree("dump/" + name, {column: values.dtype for column, values in columns.items()})
            if len(frame):
                root["dump/" + name].extend(columns)
    return path


def _tiny_counters(path):
    path.write_text("\n".join([
        'vector<string> crs.sectorNames = {"T1", "T2"};',
        'int crs.nLayers = 4;',
        'double crs.scintillatorBarThickness = 19.8;',
        'double crs.scintillatorBarWidth = 51.34;',
        'double crs.gapSmall = 0;',
        'double crs.gapLarge = 0.2;',
        'double crs.gapBetweenModules = 3.66;',
        'vector<double> crs.gapBetweenLayers = {9.525, 9.525, 9.525};',
        'double crs.layerOffset = 0;',
        *[line for sector, z in (("T1", 0), ("T2", 1000)) for line in (
            f'double crs.scintillatorBarLength{sector} = 6000;',
            f'int crs.nModules{sector} = 1;',
            f'int crs.nCountersPerModule{sector} = 2;',
            f'vector<double> crs.firstCounter{sector} = {{0, 0, {z}}};',
            f'vector<double> crs.layerDirection{sector} = {{0, 1, 0}};',
            f'vector<double> crs.gapDirection{sector} = {{0, 0, 1}};',
            f'vector<double> crs.offsetDirection{sector} = {{0, 0, -1}};',
        )],
    ]) + "\n")
    return path


def _frames(geom, subrun=101):
    # Donors retain crossing, shower and gen-empty events; full census also retains two rejected primaries and an empty event.
    events = pd.DataFrame(dict(run=np.full(6, 1400, dtype=np.uint32),
        subrun=np.full(6, subrun, dtype=np.uint32), event=np.arange(1, 7, dtype=np.uint32),
        schema_version=np.uint32(2), associations_checked=np.uint32(0), full_generator_stream=np.uint32(1),
        gen_product_id=np.uint32(99), sim_product_id=np.uint32(0),
        n_gen=np.array([1, 1, 0, 1, 1, 0], dtype=np.int64), n_particles=np.int64(0), n_steps=np.int64(0)))
    bar = geom.bar_index(0, 0, 0, 0)
    center = geom.center[bar]
    axis = np.eye(3)[geom.local_to_world[bar, 0]]
    half = geom.halflen[bar, geom.local_to_world[bar, 0]]
    gen = events.iloc[[0, 1, 3, 4]][genassoc.KEY + ["gen_product_id"]].copy().reset_index(drop=True)
    gen = gen.assign(idx=np.int64(0), pdg=np.array([-13, 22, 22, -13], dtype=np.int32), genid=np.int32(1),
                     e=np.array([200., 20., 20., 200.]), t=0.)
    position = np.tile(center + (half + 100.) * axis, (4, 1))
    momentum = -np.array([np.sqrt(200.**2 - tier1b.MASS[13]**2), 20., 20., 100.])[:, None] * axis
    for j, column in enumerate(("x", "y", "z")):
        gen[column] = position[:, j]
    for j, column in enumerate(("px", "py", "pz")):
        gen[column] = momentum[:, j]
    donor_events = events.iloc[:3].copy().assign(associations_checked=np.uint32(1),
        full_generator_stream=np.uint32(0), sim_product_id=np.uint32(202))
    donor_events["n_particles"] = np.array([2, 1, 0], dtype=np.int64)
    donor_events["n_steps"] = np.array([2, 1, 0], dtype=np.int64)
    particles = donor_events.iloc[[0, 0, 1]][genassoc.KEY + ["gen_product_id", "sim_product_id"]].copy().reset_index(drop=True)
    particles = particles.assign(key=np.array([7, 8, 9], dtype=np.int64), parent=np.array([-1, 7, -1], dtype=np.int64),
        parent_product_id=np.array([0, 202, 0], dtype=np.uint32), genidx=np.int64(0),
        isprim=np.array([1, 0, 1], dtype=np.int32), pdg=np.array([-13, 11, 22], dtype=np.int32),
        se=np.array([200., 2., 20.]), st=0.)
    steps = particles[genassoc.KEY + ["key", "sim_product_id"]].rename(
        columns={"key": "sp", "sim_product_id": "sp_product_id"}).copy()
    starts = np.array([center + half * axis, center, center])
    ends = np.array([center - half * axis, center - 0.25 * axis, center - 0.5 * axis])
    momenta = -np.array([np.sqrt(200.**2 - tier1b.MASS[13]**2), 1., 20.])[:, None] * axis
    steps = steps.assign(bar=np.int64(bar), t0=np.array([10., 10.1, 20.]), t1=np.array([10.01, 10.11, 20.01]),
                         len=np.linalg.norm(ends - starts, axis=1), edep=np.array([2., 0.2, 0.5]),
                         p1=np.linalg.norm(momenta, axis=1) * 0.99)
    for j, column in enumerate(("x", "y", "z")):
        steps[column + "0"], steps[column + "1"] = starts[:, j], ends[:, j]
        steps["p" + column + "0"] = momenta[:, j]
    donor = dict(events=donor_events, gen=gen.iloc[:2].copy(), particles=particles, steps=steps)
    full = dict(events=events, gen=gen)
    return donor, full
