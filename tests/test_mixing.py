"""Synthetic frame sampling and pre-response overlay tests, with no site data."""
import awkward as ak
import numpy as np
import pytest
import uproot
from scipy.special import ndtr

from mixing.sampling import BunchLaw, STREAMS, FrameSelection, draw_multiplicities, select_frames, stock_rates
from mixing.frames import (FrameBatch, overlay, read_frames, read_event_log, IDS, PARTICLE_FIELDS,
                           GEN_FIELDS, TRACKER_FIELDS, CRV_FIELDS)


def pool():
    return FrameBatch(np.array([[1, 1, 3], [1, 1, 1], [1, 1, 2]]), {
        "particles": ak.Array([[{"key": 7, "parent": -1, "genidx": 0, "st": 10., "et": 20.},
                                {"key": 9, "parent": 7, "genidx": -1, "st": 11., "et": 19.}],
                               [], [{"key": 7, "parent": -1, "genidx": 0, "st": 30., "et": 40.}]]),
        "gen": ak.Array([[{"idx": 0, "t": 5.}], [], [{"idx": 0, "t": 25.}]]),
        "tracker": ak.Array([[{"sp": 9, "sid": 1, "t": 1695., "eion": .2}],
                             [], [{"sp": 7, "sid": 2, "t": -5., "eion": .3}]]),
        "crv": ak.Array([[{"sp": 7, "bar": 4, "t0": 250., "t1": 251., "edep": .4}],
                         [], [{"sp": 7, "bar": 6, "t0": 9., "t1": 10., "edep": .5}]]),
    })


def test_stock_efficiency_chains():
    eff = {"MuBeamCat": .2, "MuminusStopsCat": .3, "MuStopPileupCat": .4,
           "EleBeamCat": .5, "EleBeamFlashCat": .6, "MuBeamFlashCat": .7,
           "NeutralsCat": .8, "NeutralsFlashCat": .9}
    assert stock_rates(eff, stop_prescale=1000) == pytest.approx({
        "MuStopPileup": 24, "EleBeamFlash": .3, "MuBeamFlash": .14, "NeutralsFlash": .72})
    eff.update({"EarlyEleBeamFlashCat": .006, "EarlyMuBeamFlashCat": .007,
                "EarlyNeutralsFlashCat": .009})
    early = stock_rates(eff, stop_prescale=1000,
                        early_prescales=dict(zip(STREAMS[1:], [1000, 30, 20])))
    assert early == pytest.approx({"MuStopPileup": .24, "EleBeamFlash": .03,
                                  "MuBeamFlash": .00042, "NeutralsFlash": .00144})
    with pytest.raises(KeyError):
        stock_rates({}, stop_prescale=1000)
    with pytest.raises(ValueError):
        stock_rates(eff, stop_prescale=-1)


def test_bunch_law_truncation_and_beam_weight():
    law = BunchLaw.from_sdf(1.58e7, .8, 9.48e7)
    rng = np.random.default_rng(421)
    plain = law.draw(rng, 150000, beam_primary=False)
    beam = law.draw(rng, 150000, beam_primary=True)
    assert plain.mean() == pytest.approx(1.58e7, rel=.005)
    assert beam.mean() / plain.mean() == pytest.approx(1 / .8, rel=.008)
    assert np.all(plain >= 0) and np.all(beam <= 9.48e7)
    assert plain.dtype == np.int64
    limited = BunchLaw(1000, .5, 1200, 500)
    sample = limited.draw(rng, 100000, beam_primary=True)
    mu = np.log(1000) + .5 * .5**2
    cdf = lambda x: ndtr((np.log(x) - mu) / .5)
    expected = (cdf(800.5) - cdf(500)) / (cdf(1200) - cdf(500))
    assert np.mean(sample <= 800) == pytest.approx(expected, abs=.006)
    assert sample.min() >= 500 and sample.max() <= 1200
    with pytest.raises(ValueError):
        BunchLaw.from_sdf(100, 1, 500)
    with pytest.raises(ValueError):
        BunchLaw(100, .5, 20, 30)


def test_common_intensity_conditional_poisson():
    rng = np.random.default_rng(81)
    intensity = BunchLaw(100, .4, 1000).draw(rng, 180000, beam_primary=True)
    rates = dict(zip(STREAMS, [.1, .2, .3, 0]))
    counts = draw_multiplicities(rng, intensity, rates)
    assert counts.shape == (180000, 4)
    assert np.all(counts[:, 3] == 0)
    assert counts[:, 0].mean() == pytest.approx(.1 * intensity.mean(), rel=.005)
    residual = counts[:, :3] - intensity[:, None] * np.array([.1, .2, .3])
    assert np.var(residual[:, 0]) == pytest.approx(.1 * intensity.mean(), rel=.015)
    assert np.cov(counts[:, 0], counts[:, 1])[0, 1] == pytest.approx(
        .1 * .2 * np.var(intensity), rel=.025)
    assert abs(np.corrcoef(residual[:, 0], residual[:, 1])[0, 1]) < .01
    with pytest.raises(ValueError):
        draw_multiplicities(rng, [-1], rates)


def test_sequential_wrap_skip_and_batch_continuity():
    rng = np.random.default_rng(1)
    first, cursor = select_frames(rng, [2, 0, 1], 4, cursor=2)
    np.testing.assert_array_equal(first.donors, [2, 3, 0])
    np.testing.assert_array_equal(first.event_index, [0, 0, 2])
    second, end = select_frames(rng, [3], 4, cursor=cursor)
    np.testing.assert_array_equal(second.donors, [1, 2, 3])
    assert end == 0
    seeded = np.random.default_rng(99)
    skip = np.random.default_rng(99).integers(7)
    result, _ = select_frames(seeded, [2], 8, cursor=None, max_events_to_skip=6)
    np.testing.assert_array_equal(result.donors, [skip, skip + 1])
    with pytest.raises(ValueError, match="exhausted"):
        select_frames(rng, [3], 3, cursor=1, wrap=False)
    with pytest.raises(ValueError):
        select_frames(rng, [1.5], 4)
    with pytest.raises(ValueError):
        select_frames(rng, [1], 0)
    empty, _ = select_frames(rng, [0, 0], 0)
    assert empty.donors.size == 0


def test_random_replace_is_explicit():
    rng = np.random.default_rng(4)
    result, _ = select_frames(rng, [4, 4], 2, mode="randomReplace")
    assert result.donors.size == 8
    assert set(result.donors) == {0, 1}
    with pytest.raises(ValueError, match="supported"):
        select_frames(rng, [1], 2, mode="randomNoReplace")


def test_overlay_zero_donors_preserves_values_and_empty_events():
    signal = pool()
    result = overlay(signal, [])
    np.testing.assert_array_equal(result.event_ids, signal.event_ids)
    for name in signal.collections:
        assert ak.almost_equal(result.collections[name], signal.collections[name], rtol=0, atol=0)
    assert len(result.event_ids) == 3


def test_joint_donor_reuse_remaps_sparse_keys_parent_and_generator():
    signal = pool()
    selection = FrameSelection(np.array([2, 1, 0]), np.array([0, 0, 2]))
    result = overlay(signal, [(pool(), selection)])
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result.collections["tracker"])), [3, 1, 1])
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result.collections["crv"])), [3, 1, 1])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["particles"].key[0]), [7, 9, 17, 19, 27, 29])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["particles"].parent[0]), [-1, 7, -1, 17, -1, 27])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["particles"].genidx[0]), [0, -1, 1, -1, 2, -1])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["tracker"].sp[0]), [9, 19, 29])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["crv"].sp[0]), [7, 17, 27])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["gen"].idx[0]), [0, 1, 2])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["tracker"].t[0]), [1695, 1695, 1695])


def test_time_shift_applies_once_to_donors_only_no_modulo_or_gate():
    selection = FrameSelection(np.array([1, 1, 1]), np.array([0, 1, 2]))
    result = overlay(pool(), [(pool(), selection)], time_offset_ns=np.array([10., 20., 30.]))
    np.testing.assert_array_equal(ak.to_numpy(result.collections["tracker"].t[0]), [1695, 1705])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["tracker"].t[2]), [-5, 25])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["crv"].t0[0]), [250, 260])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["crv"].t1[0]), [251, 261])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["gen"].t[0]), [5, 15])
    assert ak.num(result.collections["particles"])[1] == 0


def test_invalid_references_are_not_cross_event_joined():
    bad = dict(pool().collections)
    bad["tracker"] = ak.with_field(bad["tracker"], ak.Array([[8], [], [9]]), "sp")
    with pytest.raises(ValueError, match="dangling"):
        FrameBatch(pool().event_ids, bad).validate_references()
    with pytest.raises(ValueError, match="duplicate event"):
        FrameBatch(np.array([[1, 1, 1]] * 3), pool().collections)
    with pytest.raises(ValueError):
        FrameSelection(np.array([2]), np.array([0]))


def test_four_stream_overlay_keeps_stream_order_and_does_not_mutate():
    signal = pool()
    selection = FrameSelection(np.array([1, 1, 0]), np.array([0, 1]))
    before = ak.copy(signal.collections["tracker"])
    result = overlay(signal, [(pool(), selection)] * 4)
    np.testing.assert_array_equal(ak.to_numpy(result.collections["tracker"].sp[0]), [9, 19, 29, 39, 49])
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result.collections["tracker"])), [5, 0, 1])
    assert ak.almost_equal(signal.collections["tracker"], before, rtol=0, atol=0)


def test_optional_calo_stays_in_same_frame():
    collections = dict(pool().collections)
    collections["calo"] = ak.Array([[{"sp": 9, "t": 600., "e": 1.}], [], []])
    batch = FrameBatch(pool().event_ids, collections)
    selection = FrameSelection(np.array([1, 1, 0]), np.array([0, 0]))
    result = overlay(batch, [(batch, selection)])
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result.collections["calo"])), [2, 1, 0])
    np.testing.assert_array_equal(ak.to_numpy(result.collections["calo"].sp[0]), [9, 19])


def test_reject_fractional_keys_and_bad_generator_links():
    collections = dict(pool().collections)
    collections["tracker"] = ak.with_field(collections["tracker"],
                                           ak.Array([[9.5], [], [7.]]), "sp")
    with pytest.raises(ValueError, match="integer keys"):
        FrameBatch(pool().event_ids, collections).validate_references()
    collections = dict(pool().collections)
    collections["particles"] = ak.with_field(collections["particles"],
                                             ak.Array([[3, -1], [], [0]]), "genidx")
    with pytest.raises(ValueError, match="generator index"):
        FrameBatch(pool().event_ids, collections).validate_references()


def test_zero_primary_batch():
    empty = FrameBatch(pool().event_ids[:0], {k: v[:0] for k, v in pool().collections.items()})
    result = overlay(empty, [])
    assert len(result.event_ids) == 0
    assert all(len(v) == 0 for v in result.collections.values())


def flat_table(ids, fields, overrides=None):
    arrays = {k: np.asarray(ids)[:, i].astype(np.uint32) for i, k in enumerate(IDS)}
    arrays.update({k: np.zeros(len(ids), dtype=np.int64 if k in
                   {"key", "parent", "genidx", "idx", "sp"} else np.float64) for k in fields})
    arrays.update(overrides or {})
    return arrays


def write_dump(path, *, orphan=False):
    ids = np.array([[1, 1, 3], [1, 1, 1], [1, 1, 2]])
    occupied = ids[[2, 0]]
    if orphan:
        occupied[0, 2] = 99
    with uproot.recreate(path) as root:
        root.mktree("crv/events", flat_table(ids, (), {"n_steps": np.array([1, 0, 1])}))
        root.mktree("crv/particles", flat_table(occupied, PARTICLE_FIELDS,
                     {"key": np.array([7, 7]), "parent": np.array([-1, -1]), "genidx": np.array([0, 0])}))
        root.mktree("crv/gen", flat_table(occupied, GEN_FIELDS, {"idx": np.array([0, 0])}))
        root.mktree("crv/steps", flat_table(occupied, CRV_FIELDS, {"sp": np.array([7, 7])}))
        root.mktree("tracker/sgsteps", flat_table(occupied, TRACKER_FIELDS,
                     {"sp": np.array([7, 7]), "sid": np.array([22, 33]), "t": np.array([-2., 1695.])}))


def test_reader_preserves_census_order_empties_and_highest_cycle(tmp_path):
    path = tmp_path / "frames.root"
    write_dump(path)
    with uproot.update(path) as root:
        ids = np.array([[1, 1, 2]])
        root.mktree("tracker/sgsteps", flat_table(ids, TRACKER_FIELDS,
                     {"sp": np.array([7]), "sid": np.array([44])}))
    result = read_frames(path)
    np.testing.assert_array_equal(result.event_ids[:, 2], [3, 1, 2])
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result.collections["tracker"])), [0, 0, 1])
    np.testing.assert_array_equal(ak.to_numpy(ak.flatten(result.collections["tracker"].sid)), [44])
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result.collections["crv"])), [1, 0, 1])


def test_event_lister_census_keeps_empty_events(tmp_path):
    path = tmp_path / "frames.root"
    write_dump(path)
    log = tmp_path / "frames.log"
    log.write_text("Event: 1 run: 1 subRun: 1 event: 3\n"
                   "Event: 2 run: 1 subRun: 1 event: 1\n"
                   "noise ignored\nEvent: 3 run: 1 subRun: 1 event: 2\n")
    ids = read_event_log(log)
    result = read_frames(path, event_ids=ids)
    np.testing.assert_array_equal(result.event_ids[:, 2], [3, 1, 2])
    np.testing.assert_array_equal(ak.to_numpy(ak.num(result.collections["tracker"])), [1, 0, 1])
    from mixing.inspect import summarize
    summary = summarize(result)
    assert summary["events"] == 3
    assert summary["collections"]["tracker"]["empty_events"] == 1
    assert summary["collections"]["tracker"]["t"]["min"] == -2


def test_saved_frame_diagnostic_preserves_empty_events_and_counts():
    from mixing.check_overlay import diagnostic
    report = diagnostic(pool(), 100, seed=12)
    assert report["events"] == 100
    assert report["sampled_frames"] > 0
    assert report["diagnostic_only"] is True
    assert report["collections"]["tracker"]["rows"] == report["collections"]["crv"]["rows"]


def test_inspection_reports_nonfinite_times_without_dropping_rows():
    from mixing.inspect import summarize
    collections = dict(pool().collections)
    collections["tracker"] = ak.with_field(collections["tracker"],
                                           ak.Array([[np.nan], [], [5.]]), "t")
    result = summarize(FrameBatch(pool().event_ids, collections))
    assert result["collections"]["tracker"]["rows"] == 2
    assert result["collections"]["tracker"]["t"] == {"min": 5., "max": 5., "nonfinite": 1}


def test_reader_rejects_rows_missing_from_census(tmp_path):
    path = tmp_path / "bad.root"
    write_dump(path, orphan=True)
    with pytest.raises(ValueError, match="absent from census"):
        read_frames(path)
