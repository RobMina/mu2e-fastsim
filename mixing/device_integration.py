"""Run real detector algorithms on explicitly artificial source/CRV calibration data.

Tracker conditions are loaded from an existing saved NPZ. CRV geometry, calibration
and optical probabilities are tiny synthetic fixtures; response algorithms are the
physical kernels, not the arithmetic operators in device_synthetic. No rate here
measures production performance or physical closure.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np

from .frames import FrameBatch
from .sampling import BunchLaw, STREAMS


CAPACITY = {"max_donors": 8, "rows": {"particles": 128, "gen": 128, "tracker": 64, "crv": 128},
            "response": {"tracker": dict(clusters_per_step=64, drift_clusters=2048, straws=16,
                                         clusters_per_end=512, pairs_per_straw=8, electronics_batch=2),
                         "crv": dict(substeps=128, photons=512, thermal=128, ct1=256, ct2=128, ct3=64,
                                     pixel_short=8, pixel_long_streams=16, pixel_long_hits=64,
                                     charge_clusters=64, charges_per_cluster=256, waveform_samples=160,
                                     digis=64, reco_waveforms=64, reco_samples=96, pulse_rounds=8,
                                     coinc_hits=64, coinc_combinations=32)}}


def crv_fixture(template):
    """Synthetic four-layer geometry/lookup, stock response algorithms and template."""
    from crvfastsim import chain as ch
    from crvfastsim.response import lookup, photons, sipm
    from crvfastsim.reco import waveform

    edges = dict(x=np.array([0., 10.]), y=np.array([-25., 25.]), z=np.array([-100., -33., 33., 100.]),
                 beta=np.array([0., 1.]), theta=np.array([0., np.pi]), phi=np.array([0., np.pi]), r=np.array([0., 1.]))
    delay = np.zeros((3, 8), np.uint8)
    delay[np.arange(3), [2, 3, 4]] = 1
    def table():
        return lookup.Table(np.array([.8, .6, .4], np.float32), delay, delay.sum(1),
                            np.ones((3, 1), np.uint8), np.ones(3, np.int64))
    tables = {"tiny": lookup.LookupTable("LookupTable_tiny", (6, 0), 0, 10., 25., 100., 10., 1., 1., .5, 1., .126, 0.,
                    np.array([[1., 0.]]), np.array([[1., 0.]]), edges, {name: table() for name in lookup.TABLES})}
    geom = SimpleNamespace(nbars=4, sector_names=["EX"], sector=np.zeros(4, np.int64), module=np.zeros(4, np.int64),
                           layer=np.arange(4), counter=np.zeros(4, np.int64),
                           center=np.column_stack([np.arange(4) * 30., np.zeros((4, 2))]),
                           local_to_world=np.tile(np.arange(3), (4, 1)), halflen=np.tile([10., 25., 100.], (4, 1)))
    counters = {"crs.sectorNames": ["EX"], "crs.sipmsAtSide0EX": True, "crs.sipmsAtSide1EX": True,
                "crs.sectorTypeEX": 1, "crs.precedingSectorForEX": -1,
                "crs.nCountersPerModuleEX": 1, "crs.nModulesEX": 1}
    z = np.zeros(16)
    cond = ch.Conditions(z + 100, z + 400, z + 11, z + 7, z.astype(int), z.astype(int), z.astype(int), z)
    cfg = ch.ChainConfig(response=photons.ResponseConfig(sectors={"EX": photons.SectorConfig("tiny", 0, 100.)}),
                        sipm=replace(sipm.SiPMConfig(), thermal_rate=1e-4), window=(200., 1500.), noise=False,
                        waveform=replace(waveform.WaveformConfig(), single_pe_file=str(template)))
    return geom, cond, cfg, counters, tables, np.ones((40, 40))


def source(c, *, seed, max_selected_rows):
    """Artificial joint steps at valid tracker coordinates and four CRV layers."""
    from .device_backend import Source
    sizes = (5, 3, 4, 3, 2)
    pools = []
    for pool, size in enumerate(sizes):
        counts = (np.arange(size) % 3 != 0).astype(np.int64)
        events = np.flatnonzero(counts)
        trkcounts, crvcounts = counts, counts * 4
        straw = (520 + np.arange(size))[events]
        mid, direction = c.straw_mid[straw], c.straw_dir[straw]
        radial = np.cross(direction, np.array([0., 0., 1.]))
        fallback = np.cross(direction, np.array([0., 1., 0.]))
        radial = np.where((np.linalg.norm(radial, axis=1) > 1e-10)[:, None], radial, fallback)
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        p0 = mid + 100 * direction + radial
        p1 = mid + 100 * direction - radial
        panel = straw // 96
        trk = dict(sp=np.full(len(events), 3, np.int64), sid=((panel // 6) << 10) | ((panel % 6) << 7) | (straw % 96),
                   shape=np.zeros(len(events), np.int64), ion=np.zeros(len(events), np.int64),
                   eion=np.full(len(events), .002), len=np.full(len(events), 2.), width=np.zeros(len(events)),
                   t=np.full(len(events), 500.))
        trk.update({axis + "0": p0[:, i] for i, axis in enumerate("xyz")})
        trk.update({axis + "1": p1[:, i] for i, axis in enumerate("xyz")})
        bar = np.tile(np.arange(4), len(events))
        crv = dict(sp=np.full(len(bar), 3, np.int64), bar=bar, edep=np.full(len(bar), .5),
                   t0=np.full(len(bar), 300.), t1=np.full(len(bar), 301.), len=np.full(len(bar), .1),
                   x0=bar * 30., x1=bar * 30. + .1, y0=np.zeros(len(bar)), y1=np.zeros(len(bar)),
                   z0=np.zeros(len(bar)), z1=np.zeros(len(bar)), px0=np.zeros(len(bar)), py0=np.zeros(len(bar)),
                   pz0=np.full(len(bar), 1000.), p1=np.full(len(bar), 1000.))
        parts = dict(key=np.full(len(events), 3, np.int64), parent=np.full(len(events), -1, np.int64),
                     genidx=np.zeros(len(events), np.int64), pdg=np.full(len(events), 13, np.int64))
        gen = dict(idx=np.zeros(len(events), np.int64), pdg=np.full(len(events), 13, np.int64))
        ids = np.column_stack([np.full(size, 2**63 + pool, np.uint64), np.ones(size, np.uint64),
                               np.arange(size, dtype=np.uint64) + 2**53])
        pools.append(FrameBatch(ids, {"particles": ak.unflatten(ak.zip(parts), counts),
                                     "gen": ak.unflatten(ak.zip(gen), counts),
                                     "tracker": ak.unflatten(ak.zip(trk), trkcounts),
                                     "crv": ak.unflatten(ak.zip(crv), crvcounts)}))
    result = Source.__new__(Source)
    result._signal, result._pools = pools[0], dict(zip(STREAMS, pools[1:]))
    result._seed, result._max_selected_rows = seed, max_selected_rows
    result._ordinal, result._batch_index = 0, 0
    result._law = BunchLaw.from_sdf(20., .8, 120.)
    result._rates = dict.fromkeys(STREAMS, .01)
    result._campaign = {"max_skip": dict.fromkeys(STREAMS, 3), "mode": "sequential"}
    result.metadata = {"synthetic": True, "description": "artificial steps, true physical response kernels; synthetic CRV optical/calibration assets",
                       "signal_events_full": 5, "pool_event_counts": dict(zip(STREAMS, sizes[1:]))}
    result.tracker_conditions = c
    return result


def model(args, source, layout):
    from .device_tracker import prepare as tracker_prepare
    from .device_crv import prepare as crv_prepare
    from .device_physics import combine
    capacities = json.loads(Path(args.device_capacities).read_text())["response"]
    tracker = tracker_prepare(source.tracker_conditions, capacities["tracker"], layout.batch_events)
    geom, cond, cfg, counters, tables, pmap = crv_fixture(args.template)
    crv = crv_prepare(geom, cond, cfg, counters, capacities["crv"], tables, pmap)
    timing = {"mc_pbtime": np.zeros(source.population), "reco_pbtime": np.full(source.population, 10.),
              "event_length": np.resize(np.array([1675., 1700.]), source.population)}
    out = combine(tracker, crv, timing, synthetic_inputs=True, output=capacities.get("output"))
    out.metadata["crv_calibration"] = "synthetic lookup, geometry and conditions; actual stock pulse template"
    return out


def run_test(args):
    from trkfastsim.digi.digicond import Conditions
    from .stream_benchmark import run
    from .device_backend import Backend
    conditions = Conditions.load(args.tracker_conditions)
    if not conditions.onspill:
        raise ValueError("integration requires existing onspill tracker conditions")
    def source_factory(manifest, **kwargs):
        return source(conditions, **kwargs)
    report = run(args, source_factory, Backend)
    from .inspect import summarize_fast
    totals = {"events": 0, "tracker_digis": 0, "digis": 0, "pulses": 0, "clusters": 0}
    for path in args.out.glob("*/fast.root"):
        census = summarize_fast(path)
        totals["events"] += census["events"]
        totals["tracker_digis"] += census["tracker_digis"]
        for name in ("digis", "pulses", "clusters"):
            totals[name] += census["crv"][name]
    if not all(totals.values()):
        raise ValueError(f"physical integration did not exercise every output: {totals}")
    check = {"finished_totals_including_warmup": totals, "tracker_staged_check": tracker_check(conditions),
             "synthetic_sources_and_crv_calibration": True, "physics_algorithms": True,
             "production_validation": False}
    with (args.out / "integration_check.json").open("x") as stream:
        json.dump(check, stream, indent=2)
    return report


def tracker_check(c):
    """Same-device direct tracker regression, outside the persistent measurement."""
    import jax
    from trkfastsim.digi import digi_jax
    from .device_tracker import prepare
    src = source(c, seed=23, max_selected_rows=448)
    rows = src.signal.collections["tracker"]
    columns = {name: ak.to_numpy(ak.flatten(rows[name])) for name in ak.fields(rows)}
    events = np.repeat(np.arange(src.population), ak.to_numpy(ak.num(rows)))
    timing = {"mc_pbtime": np.zeros(src.population), "reco_pbtime": np.full(src.population, 10.),
              "event_length": np.resize(np.array([1675., 1700.]), src.population),
              "event_mask": np.ones(src.population, bool)}
    model = prepare(c, CAPACITY["response"]["tracker"], src.population)
    args = jax.device_put(({"columns": columns, "event": events, "mask": np.ones(len(events), bool)},
                          timing, model.assets, jax.random.key(19)))
    with jax.transfer_guard("disallow"):
        out, errors = jax.jit(model.apply)(*args)
        jax.block_until_ready((out, errors))
    out, errors = jax.device_get((out, errors))
    if any(errors.values()):
        raise ValueError(f"tracker integration check failed: {errors}")
    expected = digi_jax.digitize({**columns, "ev": events}, src.population, c, jax.random.key(19),
                                event_timing={"pbtime": timing["mc_pbtime"], "event_length": timing["event_length"]})
    max_float_difference = 0.
    for name, value in expected.items():
        actual = out[name][out["mask"]]
        if value.dtype.kind in "iu":
            np.testing.assert_array_equal(actual, value, err_msg=name)
        else:
            np.testing.assert_allclose(actual, value, rtol=0, atol=1e-5, err_msg=name)
            max_float_difference = max(max_float_difference, float(np.max(np.abs(actual - value), initial=0)))
    return {"events": src.population, "digis": len(expected["sid"]), "integer_columns_exact": True,
            "max_float_difference": max_float_difference, "atol": 1e-5}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracker-conditions", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--events", type=int, default=5)
    args = parser.parse_args(argv)
    cap = args.out.parent / (args.out.name + "_capacities.json")
    with cap.open("x") as stream:
        json.dump(CAPACITY, stream)
    cfg = SimpleNamespace(out=args.out, frames_manifest=None, tracker_conditions=args.tracker_conditions,
                          template=args.template, device_platform=args.platform, device_kernel="mixing.device_integration:model",
                          device_capacities=cap, synthetic=False, backend="mixing.device_backend:Backend",
                          events=args.events, batch_events=2, max_selected_rows=448, chunk=2,
                          warmup_batches=1, seed=17, tracker_seed=19, sampling_seed=23, require_stage=[])
    report = run_test(cfg)
    print(json.dumps({name: report[name] for name in ("events", "warmup_events", "measured_batches", "backend")}))


if __name__ == "__main__":
    main()
