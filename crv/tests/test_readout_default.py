"""The physical readout mask is applied by default: loaders, prepared contexts and the extracted no-op."""
import numpy as np
import pytest

from crvfastsim import chain, geometry, readout, runconfig
from test_e2e_bench import R, pipeline  # noqa: F401
from test_prepared_context import prepared_pipeline  # noqa: F401


def ordinary(state, **kw):
    opts = {k: state[k] for k in ("inputs", "outdir", "library", "calib", "gdml")}
    return R.run(**{**opts, **kw})


def _observe_charges(monkeypatch, seen):
    original = R.chain.charges

    def charges(steps, parts, geom, cond, *args, **kw):
        seen.append(None if cond.readout_mask is None else cond.readout_mask.copy())
        return original(steps, parts, geom, cond, *args, **kw)
    monkeypatch.setattr(R.chain, "charges", charges)


def test_runchain_applies_the_counters_mask(pipeline, monkeypatch):  # noqa: F811
    monkeypatch.setattr(R.runconfig.RunConfig, "counters_cfg",
                        lambda self: {"crs.sipmsAtSide0EX": True, "crs.sipmsAtSide1EX": False})
    seen = []
    _observe_charges(monkeypatch, seen)
    ordinary(pipeline)
    assert seen and all(m is not None for m in seen)
    assert seen[0][:8].tolist() == [True, False, True, False] * 2


def test_prepared_context_carries_the_mask(prepared_pipeline, monkeypatch):  # noqa: F811
    state = prepared_pipeline
    context = R.prepare(library=state["library"], calib="nominal", gdml=state["gdml"])
    assert context.payload["cond"].readout_mask is not None
    seen = []
    _observe_charges(monkeypatch, seen)
    ordinary(state, prepared=context)
    assert seen and all(m is not None and m.all() for m in seen)


def test_apply_keeps_extra_channels_present_and_refuses_short_conditions():
    g = geometry.CRVGeometry(["R4"], np.zeros(1, int), np.zeros(1, int), np.zeros(1, int), np.zeros(1, int),
                             np.zeros((1, 3)), np.ones((1, 3)), np.tile([0, 1, 2], (1, 1)))
    cfg = {"crs.sipmsAtSide0R4": True, "crs.sipmsAtSide1R4": False}
    z = np.zeros(6)
    cond = chain.Conditions(z, z, z, z, np.zeros(6, np.int64), z.astype(int), z.astype(int), z)
    masked = readout.apply(cond, g, cfg)
    assert masked.readout_mask.tolist() == [True, False, True, False, True, True]
    assert cond.readout_mask is None and masked.status is cond.status
    np.testing.assert_array_equal(readout.effective_status(masked), [0, 1, 0, 1, 0, 0])
    short = chain.Conditions(*(v[:3] for v in (z, z, z, z, np.zeros(6, np.int64), z.astype(int),
                                                z.astype(int), z)))
    with pytest.raises(ValueError, match="conditions cover 3 channels"):
        readout.apply(short, g, cfg)


@pytest.mark.parametrize("name", ["extracted_v04", "run1a_v01"])
def test_no_photons_reach_absent_sipms_in_extracted_and_run1a(name):
    """The mask is a no-op where every simulated SiPM (reflector routing) is a present one: extracted and Run 1A."""
    rc = runconfig.get(name)
    try:
        g, cfg = rc.geometry(), rc.counters_cfg()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    mask = readout.sensor_mask(g, cfg).reshape(-1, 4)
    sectors = rc.response().sectors
    for i, sector in enumerate(g.sector_names):
        bars = np.flatnonzero(g.sector == i)
        if not len(bars):
            continue
        r = sectors[sector].reflector
        simulated = {0, 1, 2, 3} if r == 0 else ({0, 2} if r > 0 else {1, 3})
        assert all(mask[bars][:, s].all() for s in simulated), sector
