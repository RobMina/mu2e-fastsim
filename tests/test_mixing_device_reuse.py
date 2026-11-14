"""Donor-frame digi reuse (mixing.device_reuse, an approximation): which straws are reused, and that the
straws digitised per batch are exactly the resident tracker's."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mu2efastsim.testing")
trkfastsim = pytest.importorskip("trkfastsim")

from mixing import device_reuse
from mixing.device_tracker import prepare

CAPS = {"clusters_per_step": 64, "drift_clusters": 1024, "straws": 8, "clusters_per_end": 512, "pairs_per_straw": 8,
        "electronics_batch": 2}


@pytest.fixture(autouse=True, scope="module")
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def helpers():
    path = Path(trkfastsim.__file__).parents[1] / "tests" / "test_digi.py"
    if not path.is_file() or not (path.parent / "data" / "digi_job_excerpt.fcl").is_file():
        pytest.skip("tracker checkout with tests/test_digi.py and its public conditions fixture required")
    spec = importlib.util.spec_from_file_location("tracker_fixture_reuse", path)
    module = importlib.util.module_from_spec(spec)
    previous_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    return module


@pytest.fixture(scope="module")
def conditions(helpers):
    from trkfastsim.digi import digicond as dc
    d = dc.parse_fhicl_dump(helpers.FIX.read_text())
    preamp = np.zeros((dc.NUSTRAWS, 6))
    preamp[:, 0] = np.arange(dc.NUSTRAWS)
    preamp[:, 3], preamp[:, 4], preamp[:, 5] = 11.996, 11.632, 1.5e6
    c = dc.assemble(d, preamp, helpers._geom())
    c.s["onspill"] = True
    return c


def _pools(c, helpers):
    """Signal pool (one primary frame) then the four donor pools (MuStop, Ele, MuBeam, Neutrals): A, B in MuStop,
    C in MuBeam, D in Neutrals. Returns pack_source-like host data and each frame's global index."""
    frames = {"P": [helpers._step(c, 520, t=700.)],
              "A": [helpers._step(c, 1000, t=650.)], "B": [helpers._step(c, 1000, t=900.)],
              "C": [helpers._step(c, 92, t=500.)], "D": [helpers._step(c, 523, t=800., eion=.04, ion=1)]}
    pools = [["P"], ["A", "B"], [], ["C"], ["D"]]
    order = [name for pool in pools for name in pool]
    rows = [step for name in order for step in frames[name]]
    columns = {k: np.r_[np.array([r[k] for r in rows]), np.zeros(1, np.asarray(rows[0][k]).dtype)] for k in rows[0]}
    counts = [len(frames[name]) for name in order]
    data = {"sizes": np.array([len(p) for p in pools]), "collections": {"tracker": {
        "offsets": np.r_[0, np.cumsum(counts)], "columns": columns}}}
    data["starts"] = np.r_[0, np.cumsum(data["sizes"])[:-1]]
    return data, {name: i for i, name in enumerate(order)}, frames


def _mixed(c, data, index, frames):
    """Event 0: primary P + donors A and B on one straw (two sources: digitised per batch); event 1: donors C and D
    on their own straws (reused)."""
    events = [["P", "A", "B"], ["C", "D"]]
    st, ev, occ = [], [], []
    for e, names in enumerate(events):
        for name in names:
            st += frames[name]
            ev += [e] * len(frames[name])
            occ += [-1 if name == "P" else index[name]] * len(frames[name])
    capacity = 8
    pad = capacity - len(st)
    columns = {k: np.r_[np.array([r[k] for r in st]), np.zeros(pad, np.asarray(st[0][k]).dtype)] for k in st[0]}
    return {"columns": columns, "mask": np.arange(capacity) < len(st), "event": np.r_[ev, np.zeros(pad, np.int64)],
            "occurrence": np.r_[occ, np.full(pad, -1)]}


def _run(model, collection, times, seed=3):
    out, errors = jax.device_get(jax.jit(model.apply)(collection, times, model.assets, jax.random.key(seed)))
    return {name: np.asarray(v)[np.asarray(out["mask"])] for name, v in out.items() if name != "mask"}, errors


def _reuse(c, helpers, streams=None, shift=-5., **extra):
    data, index, frames = _pools(c, helpers)
    caps = {"compute": CAPS, "precompute": CAPS, "frames_per_batch": 8, "straws": 8,
            **({"streams": streams} if streams else {}), **extra}
    table = device_reuse.precompute(c, caps, data, 1700., 0, rows_capacity=8, shift=shift, margin=15.)
    return device_reuse.prepare(c, caps, 2, table, data["starts"]), table, data, index, frames


def test_reuse_capacities_validated():
    with pytest.raises(ValueError, match="frames_per_batch"):
        device_reuse.ReuseCapacities(CAPS, CAPS, 0, 8)
    with pytest.raises(ValueError, match="straws"):
        device_reuse.ReuseCapacities(CAPS, CAPS, 8, 0)
    with pytest.raises(ValueError, match="streams"):
        device_reuse.ReuseCapacities(CAPS, CAPS, 8, 8, streams=["Cosmics"])
    assert device_reuse.ReuseCapacities(CAPS, CAPS, 8, 8, streams=["MuBeamFlash"]).reused_pools() == [3]
    for bad in (0, 65, 2.0):
        with pytest.raises(ValueError, match="phases"):
            device_reuse.ReuseCapacities(CAPS, CAPS, 8, 8, phases=bad)
    with pytest.raises(ValueError, match="event_jitter"):
        device_reuse.ReuseCapacities(CAPS, CAPS, 8, 8, event_jitter=1)
    caps = device_reuse.ReuseCapacities(CAPS, CAPS, 8, 8)
    assert caps.phases == 1 and caps.event_jitter is False


def test_widened_window(conditions):
    w = device_reuse.widened(conditions, 10.)
    assert w.digitizationStart == conditions.digitizationStart - 10 and w.digitizationEnd == conditions.digitizationEnd + 10
    ticks = int(np.ceil(10. / conditions.TDCLSB))
    assert w.digitizationStartTDC == conditions.digitizationStartTDC - ticks
    assert w.digitizationEndTDC == conditions.digitizationEndTDC + ticks
    assert conditions.digitizationStart != w.digitizationStart   # the input is not modified


def test_straws_digitised_per_batch_equal_the_tracker(conditions, helpers):
    """Primary and two-source straws go through the resident tracker with the same keys: the same digis as the
    tracker on the whole batch. Single-donor straws are the setup digis, TDC shifted by mc_pbtime - shift."""
    c = conditions
    model, table, data, index, frames = _reuse(c, helpers)
    collection = _mixed(c, data, index, frames)
    times = {"event_length": np.array([1700., 1690.]), "mc_pbtime": np.array([-2., -9.]),
             "event_mask": np.ones(2, bool)}
    exact, ee = _run(prepare(c, CAPS, 2), collection, times)
    reuse, re = _run(model, collection, times)
    assert not any(np.any(v) for v in ee.values()) and not any(np.any(v) for v in re.values()), (ee, re)
    sid = {us: helpers.DG.unique_straw_to_sid(us) for us in (520, 1000, 92, 523)}
    for ev, us in ((0, 520), (0, 1000)):          # primary; two donor frames
        a, b = (exact["ev"] == ev) & (exact["sid"] == sid[us]), (reuse["ev"] == ev) & (reuse["sid"] == sid[us])
        assert a.sum() > 0 and a.sum() == b.sum(), (us, a.sum(), b.sum())
        for name in ("adc", "tdc", "tot", "pmp"):
            np.testing.assert_array_equal(exact[name][a], reuse[name][b], err_msg=f"{us} {name}")
    for name, us in (("C", 92), ("D", 523)):    # single donor frames: the setup digis
        rows = (table["frame"] == index[name]) & (table["sid"] == sid[us])
        b = (reuse["ev"] == 1) & (reuse["sid"] == sid[us])
        assert rows.sum() > 0 and rows.sum() == b.sum(), (us, rows.sum(), b.sum())
        shift = np.round((-9. - table["shift"]) / c.TDCLSB).astype(np.int64)
        np.testing.assert_array_equal(table["tdc"][rows] + shift, reuse["tdc"][b])
        for field in ("adc", "tot", "pmp"):
            np.testing.assert_array_equal(table[field][rows], reuse[field][b])
    usage = jax.device_get(jax.jit(model.apply_with_usage)(collection, times, model.assets, jax.random.key(3))[2])
    assert usage["reuse_straws"] == 2


def test_streams_not_reused_are_digitised_per_batch(conditions, helpers):
    """With MuStop only reused, the MuBeam and Neutrals frames' straws are digitised like the tracker."""
    c = conditions
    model, table, data, index, frames = _reuse(c, helpers, streams=["MuStopPileup"])
    assert set(np.unique(table["frame"])) <= {index["A"], index["B"]}
    collection = _mixed(c, data, index, frames)
    times = {"event_length": np.array([1700., 1690.]), "mc_pbtime": np.array([-2., -9.]),
             "event_mask": np.ones(2, bool)}
    exact, _ = _run(prepare(c, CAPS, 2), collection, times)
    reuse, _ = _run(model, collection, times)
    for name in ("ev", "sid", "adc", "tdc", "tot", "pmp"):
        np.testing.assert_array_equal(exact[name], reuse[name], err_msg=name)


def _single_donor_rows(table, reuse, index, sid, name, us, copy=0):
    rows = (table["frame"] == index[name]) & (table["sid"] == sid[us]) & (table["copy"] == copy)
    b = (reuse["ev"] == 1) & (reuse["sid"] == sid[us])
    assert rows.sum() > 0 and rows.sum() == b.sum(), (us, rows.sum(), b.sum())
    return rows, b


@pytest.mark.parametrize("k,m", [(0, 0), (1, 0), (3, -1), (2, 1)])
def test_phase_copies(conditions, helpers, k, m):
    """With phases K each frame is digitised K times, at shift + k P / K; an event at shift + k P / K + m P takes
    copy k, its TDC moved by m whole ADC periods and its waveform, TOT and PMP unchanged. The primary and two-source
    straws stay the tracker's digis."""
    c, K, shift = conditions, 4, -5.
    model, table, data, index, frames = _reuse(c, helpers, shift=shift, phases=K)
    assert int(table["phases"]) == K and set(np.unique(table["copy"])) == set(range(K))
    P = float(c.ADCPeriod)
    np.testing.assert_allclose(device_reuse.copy_shifts(c, shift, K), shift + P * np.arange(K) / K)
    collection = _mixed(c, data, index, frames)
    pb = shift + k * P / K + m * P
    times = {"event_length": np.array([1700., 1690.]), "mc_pbtime": np.array([-2., pb]),
             "event_mask": np.ones(2, bool)}
    exact, _ = _run(prepare(c, CAPS, 2), collection, times)
    reuse, errors = _run(model, collection, times)
    assert not any(np.any(v) for v in errors.values()), errors
    sid = {us: helpers.DG.unique_straw_to_sid(us) for us in (520, 1000, 92, 523)}
    for us in (520, 1000):
        a, b = (exact["ev"] == 0) & (exact["sid"] == sid[us]), (reuse["ev"] == 0) & (reuse["sid"] == sid[us])
        assert a.sum() > 0 and a.sum() == b.sum()
        for name in ("adc", "tdc", "tot", "pmp"):
            np.testing.assert_array_equal(exact[name][a], reuse[name][b], err_msg=f"{us} {name}")
    for name, us in (("C", 92), ("D", 523)):
        rows, b = _single_donor_rows(table, reuse, index, sid, name, us, copy=k)
        np.testing.assert_array_equal(table["tdc"][rows] + int(np.round(m * P / c.TDCLSB)), reuse["tdc"][b])
        np.testing.assert_allclose(table["t_analog"][rows] + m * P, reuse["t_analog"][b], atol=1e-9)
        for field in ("adc", "tot", "pmp"):
            np.testing.assert_array_equal(table[field][rows], reuse[field][b])


def test_event_jitter(conditions, helpers):
    """event_jitter: a reused digi's TDC takes the event's ROC jitter of its panel (the resident tracker's draw for
    that event and panel) in place of the setup event's."""
    c = conditions
    model, table, data, index, frames = _reuse(c, helpers, event_jitter=True)
    collection = _mixed(c, data, index, frames)
    times = {"event_length": np.array([1700., 1690.]), "mc_pbtime": np.array([-2., -9.]),
             "event_mask": np.ones(2, bool)}
    reuse, errors = _run(model, collection, times, seed=3)
    assert not any(np.any(v) for v in errors.values()), errors
    jt = np.asarray(jax.device_get(jax.jit(lambda k: device_reuse.jitter_table(k, 2))(jax.random.key(3))))
    sid = {us: helpers.DG.unique_straw_to_sid(us) for us in (92, 523)}
    for name, us in (("C", 92), ("D", 523)):
        rows, b = _single_donor_rows(table, reuse, index, sid, name, us)
        panel = (sid[us] >> 10) * 6 + ((sid[us] >> 7) & 7)
        moved = -9. - table["shift"] + jt[1, panel] * c.eventWindowMarkerROCJitter - table["jitter"][rows]
        np.testing.assert_array_equal(table["tdc"][rows] + np.round(moved / c.TDCLSB).astype(np.int64)[:, None],
                                      reuse["tdc"][b])
    assert np.all(np.abs(table["jitter"]) < 10 * c.eventWindowMarkerROCJitter + 1e-12)


def test_setup_jitter_is_the_trackers(conditions, helpers):
    """The stored setup jitter is the resident tracker's draw for the setup event and panel: with the jitter scale
    set to zero the setup TDCs move by exactly the stored jitter (to the TDC rounding)."""
    from trkfastsim.digi.digicond import Conditions
    c = conditions
    s = dict(c.s)
    s["eventWindowMarkerROCJitter"] = 0.
    flat = Conditions(s, c.a)
    _, table, _, _, _ = _reuse(c, helpers)
    _, table0, _, _, _ = _reuse(flat, helpers)
    assert np.all(table0["jitter"] == 0)
    np.testing.assert_array_equal(table["frame"], table0["frame"])
    expected = table0["tdc"] + np.round(table["jitter"] / c.TDCLSB).astype(np.int64)[:, None]
    assert np.max(np.abs(table["tdc"] - expected)) <= 1


def test_dead_time_applied_per_event(conditions, helpers):
    """The setup digitises without the digital dead time; each event applies it straw by straw. A donor frame with
    two pulses 200 ns apart on one straw (inside the dead time): the setup keeps both, the event keeps the first, as
    the resident tracker does for the same frame in the event."""
    c = conditions
    assert c.deadTimeDigital > 200.
    two = [helpers._step(c, 700, t=650.), helpers._step(c, 700, t=850.)]
    frames = {"P": [helpers._step(c, 520, t=700.)], "A": two}
    pools = [["P"], ["A"], [], [], []]
    order = [name for pool in pools for name in pool]
    rows = [step for name in order for step in frames[name]]
    columns = {k: np.r_[np.array([r[k] for r in rows]), np.zeros(1, np.asarray(rows[0][k]).dtype)] for k in rows[0]}
    counts = [len(frames[name]) for name in order]
    data = {"sizes": np.array([len(p) for p in pools]), "collections": {"tracker": {
        "offsets": np.r_[0, np.cumsum(counts)], "columns": columns}}}
    data["starts"] = np.r_[0, np.cumsum(data["sizes"])[:-1]]
    caps = {"compute": CAPS, "precompute": CAPS, "frames_per_batch": 8, "straws": 8}
    table = device_reuse.precompute(c, caps, data, 1700., 0, rows_capacity=8, shift=-5., margin=15.)
    sid = helpers.DG.unique_straw_to_sid(700)
    rows_a = (table["frame"] == 1) & (table["sid"] == sid)
    assert rows_a.sum() == 2                       # no dead time in the setup
    assert np.all(np.diff(table["dtime"][rows_a]) > 150.)
    model = device_reuse.prepare(c, caps, 2, table, data["starts"])
    st = frames["P"] + two
    capacity = 8
    pad = capacity - len(st)
    collection = {"columns": {k: np.r_[np.array([r[k] for r in st]), np.zeros(pad, np.asarray(st[0][k]).dtype)]
                              for k in st[0]},
                  "mask": np.arange(capacity) < len(st), "event": np.r_[[0, 1, 1], np.zeros(pad, np.int64)],
                  "occurrence": np.r_[[-1, 1, 1], np.full(pad, -1)]}
    times = {"event_length": np.array([1700., 1700.]), "mc_pbtime": np.array([-2., -5.]),
             "event_mask": np.ones(2, bool)}
    exact, _ = _run(prepare(c, CAPS, 2), collection, times)
    reuse, errors = _run(model, collection, times)
    assert not any(np.any(v) for v in errors.values()), errors
    a, b = (exact["ev"] == 1) & (exact["sid"] == sid), (reuse["ev"] == 1) & (reuse["sid"] == sid)
    assert a.sum() == 1 and b.sum() == 1, (a.sum(), b.sum())
    first = np.argmin(table["dtime"][rows_a])
    np.testing.assert_array_equal(reuse["tdc"][b][0], table["tdc"][rows_a][first])


def test_dead_time_clock(conditions):
    c = conditions
    sid = np.array([(3 << 10) | (2 << 7) | 17])
    tdc = np.array([[1000, 1200]])
    t = device_reuse.dead_time_clock(c, sid, tdc)
    assert t.shape == (1,)
    lo = 1200 * c.TDCLSB - c.electronicsTimeDelay
    if c.overrideDbTimeOffsets:
        assert lo <= t[0] < lo + c.TDCLSB
