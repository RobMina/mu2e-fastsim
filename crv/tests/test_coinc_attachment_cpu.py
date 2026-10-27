"""Independent membership expectations for the stock opposite-side attachment stage."""
from dataclasses import replace

import numpy as np
import pytest

from crvfastsim.reco import coinc as C


@pytest.fixture(params=[False, True], ids=["scalar-combinations", "vectorized-combinations"])
def vectorized(request):
    return request.param


def geom(both=True):
    layer = np.tile(np.arange(4), 2)
    x = np.repeat([0., 400.], 4)
    y = layer * 10.
    return C.SectorGeom(
        sector=np.zeros(8, dtype=int), layer=layer, counter_in_type=np.repeat([0, 1], 4),
        x=x, y=y, pos=np.column_stack([x, y, np.zeros(8)]), length_dir=np.array([2]),
        sector_type=np.array([1]), sector_names=["EX"], pos_side0=np.array([-700.]),
        pos_side1=np.array([700.]), sipms_side0=np.array([True]), sipms_side1=np.array([both]))


def config(**kw):
    cuts = replace(C.EXTRACTED_V04["EX"], max_dt=10.)
    return C.FinderConfig(cuts={"EX": cuts}, **kw)


def pulse(idx, bar, sipm, time=100., pe=10., seq=0):
    return dict(idx=idx, bar=bar, sipm=sipm, time=time, pe=pe, seq=seq)


def seeds(time=100., start=10, sipm=0):
    return [pulse(start + bar, bar, sipm, time) for bar in range(3)]


def run(rows, vectorized=False, cfg=None, G=None, status=None):
    p = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    return C.find_event(p, geom() if G is None else G, np.zeros(32, int) if status is None else status,
                        config() if cfg is None else cfg, vectorized=vectorized)


def members(rows):
    return [row["pulses"] for row in rows]


def test_attachment_default_and_disabled_legacy(vectorized):
    p = seeds() + [pulse(13, 0, 1, 105.)]
    got = run(p, vectorized)
    assert members(got) == [[10, 11, 12, 13]]
    assert got[0]["pe"] == 40.
    assert got[0]["sidehits"] == [3, 1]
    assert got[0]["sidepe"] == [30., 10.]
    assert got[0]["end"] == 105.
    assert members(run(p, vectorized, config(add_opposite_side_pulses=False))) == [[10, 11, 12]]
    assert members(run(p, vectorized, G=geom(both=False))) == [[10, 11, 12]]


@pytest.mark.parametrize("time,attached", [
    (np.nextafter(120., -np.inf), True), (120., False), (np.nextafter(120., np.inf), False),
    (np.nextafter(80., np.inf), True), (80., False), (np.nextafter(80., -np.inf), False),
])
def test_strict_attachment_window_both_boundaries(vectorized, time, attached):
    got = run(seeds() + [pulse(13, 0, 1, time)], vectorized)
    assert members(got) == [[10, 11, 12] + ([13] if attached else [])]



def test_side0_candidate_and_absent_coincidence_seeds(vectorized):
    p = seeds(sipm=1) + [pulse(13, 0, 0, 105.)]
    assert members(run(p, vectorized)) == [[10, 11, 12, 13]]
    assert run([p[0], p[-1]], vectorized) == []


@pytest.mark.parametrize("time,attached", [(np.nextafter(135., -np.inf), True), (135., False)])
def test_type_wide_half_length_and_corrected_hit_times(vectorized, time, attached):
    G = replace(geom(), sector=np.repeat([0, 1], 4), length_dir=np.array([2, 2]),
                sector_type=np.array([1, 1]), sector_names=["EX", "T1"],
                pos_side0=np.array([-700., -1400.]), pos_side1=np.array([1400., 700.]),
                sipms_side0=np.array([True, True]), sipms_side1=np.array([True, True]))
    cfg = config()
    cfg = replace(cfg, cuts={"EX": cfg.cuts["EX"], "T1": cfg.cuts["EX"]})
    got = run(seeds() + [pulse(13, 0, 1, time)], vectorized, cfg, G)
    assert members(got) == [[10, 11, 12] + ([13] if attached else [])]
    assert got[0]["start"] == 105.


def test_filtered_candidates_only_and_inclusive_filter_boundary(vectorized):
    assert members(run(seeds() + [pulse(13, 0, 1, 105., pe=7.)], vectorized)) == [[10, 11, 12]]
    p = seeds() + [pulse(13, 0, 3, 100., pe=4.), pulse(14, 0, 1, 110., pe=4.)]
    assert members(run(p, vectorized)) == [[10, 11, 12, 14, 13]]
    p[-1]["time"] = np.nextafter(110., np.inf)
    assert members(run(p, vectorized)) == [[10, 11, 12]]


def test_compensated_pe_reaches_attached_properties(vectorized):
    status = np.zeros(32, int)
    status[3] = 1
    got = run(seeds() + [pulse(13, 0, 1, 105., pe=4.)], vectorized, status=status)
    assert members(got) == [[10, 11, 12, 13]]
    assert got[0]["sidepe"] == [30., 8.]


@pytest.mark.parametrize("candidate_secondary,anchor_secondary", [(True, False), (False, True), (True, True)])
def test_secondary_candidate_and_anchor_excluded_with_ignore_false(vectorized, candidate_secondary, anchor_secondary):
    p = seeds() + [pulse(13, 0, 1, 105., seq=int(candidate_secondary))]
    p[0]["seq"] = int(anchor_secondary)
    assert members(run(p, vectorized, config(ignore_secondary_pulses=False))) == [[10, 11, 12]]


def test_original_secondary_member_occupies_side(vectorized):
    p = seeds() + seeds(start=20, sipm=1) + [pulse(30, 0, 3, 115.)]
    p[3]["seq"] = 1
    assert members(run(p, vectorized, config(ignore_secondary_pulses=False))) == [[10, 11, 12, 20, 21, 22]]


def test_attachment_rescues_min_cluster_pes(vectorized):
    cfg = config()
    cfg = replace(cfg, cuts={"EX": replace(cfg.cuts["EX"], min_cluster_pes=35.)})
    p = seeds() + [pulse(13, 0, 1, 105.)]
    assert members(run(p, vectorized, cfg)) == [[10, 11, 12, 13]]
    assert run(p, vectorized, replace(cfg, add_opposite_side_pulses=False)) == []


def test_candidate_tie_and_sorted_bar_sipm_append(vectorized):
    p = seeds() + [pulse(30, 1, 3, 103.), pulse(40, 0, 3, 104.),
                   pulse(90, 0, 1, 102.), pulse(2, 0, 1, 102.), pulse(50, 1, 1, 101.)]
    assert members(run(p, vectorized)) == [[10, 11, 12, 90, 40, 50, 30]]


def test_nearest_cluster_tie_and_loser_not_reassigned(vectorized):
    p = seeds() + seeds(time=130., start=20) + [pulse(30, 0, 1, 115.), pulse(31, 0, 1, 114.)]
    assert members(run(p, vectorized)) == [[10, 11, 12, 31], [20, 21, 22]]
    assert members(run(p[:-1], vectorized)) == [[10, 11, 12, 30], [20, 21, 22]]
    p[-2]["time"] = np.nextafter(115., np.inf)
    assert members(run(p[:-1], vectorized)) == [[10, 11, 12], [20, 21, 22, 30]]


def test_nearest_cluster_order_not_time_or_pulse_index(vectorized):
    p = seeds(time=130., start=10) + seeds(start=20) + [pulse(30, 0, 1, 115.)]
    assert members(run(p, vectorized)) == [[10, 11, 12, 30], [20, 21, 22]]


def test_same_physical_bar_not_counter_or_neighbor(vectorized):
    G = geom()
    G.counter_in_type[4] = G.counter_in_type[0]
    p = seeds() + [pulse(13, 4, 1, 105.), pulse(14, 3, 1, 105.)]
    assert members(run(p, vectorized, G=G)) == [[10, 11, 12]]


def hit(idx, bar=0, sipm=0, t=100., seq=0):
    return dict(idx=idx, bar=bar, sipm=sipm, side=sipm % 2, t=t, seq=seq)


def attached(clusters, candidates):
    out = [list(cl) for cl in clusters]
    C._attach_opposite_side_pulses(out, candidates, 20.)
    return [[h["idx"] for h in cl] for cl in out]


def test_attachment_snapshot_no_cascade_or_clustered_pulse_reuse():
    a = hit(0)
    b = hit(1, sipm=1, t=105.)
    same = hit(2, sipm=2, t=106.)
    other_cluster = hit(3, sipm=3, t=107.)
    assert attached([[a], [other_cluster]], [a, same, b, other_cluster]) == [[0, 1], [3, 2]]
    assert attached([[a]], [a, same, b]) == [[0, 1]]


def test_attachment_supports_both_missing_ends_and_empty_inputs():
    a = hit(0, bar=1, sipm=1)
    b = hit(1, bar=0)
    p = [hit(8, bar=1, sipm=2), hit(9, bar=1), hit(10, bar=0, sipm=3), hit(11, bar=0, sipm=1)]
    assert attached([[a, b]], p) == [[0, 1, 11, 10, 9, 8]]
    assert attached([], p) == []
    assert attached([[a]], []) == [[0]]
    assert attached([[a]], [a]) == [[0]]
    assert attached([[a]], [hit(5, bar=4)]) == [[0]]


def test_filtered_side_order_is_preserved(monkeypatch, vectorized):
    captured = []
    original = C._attach_opposite_side_pulses
    def capture(clusters, hits, max_dt):
        captured.extend((h["idx"], h["sipm"], h["seq"]) for h in hits)
        return original(clusters, hits, max_dt)
    monkeypatch.setattr(C, "_attach_opposite_side_pulses", capture)
    p = [pulse(13, 0, 1, 105.)] + seeds()
    assert members(run(p, vectorized)) == [[10, 11, 12, 13]]
    assert captured == [(10, 0, 0), (11, 0, 0), (12, 0, 0), (13, 1, 0)]


def test_config_schema_and_named_run_explicit_false():
    from crvfastsim import response_config as RC, runconfig
    assert C.FinderConfig().add_opposite_side_pulses is True
    assert RC.parse()["finder"]["add_opposite_side_pulses"] is True
    for enabled in (True, False):
        parsed = RC.parse({"schema": 1, "finder": {"add_opposite_side_pulses": enabled}})
        assert RC.build(parsed).finder.add_opposite_side_pulses is enabled
        assert RC.parse(RC.canonical(parsed))["finder"]["add_opposite_side_pulses"] is enabled
        for name in runconfig.names():
            rc = runconfig.get(name)
            assert rc.finder().add_opposite_side_pulses is True
            assert replace(rc, add_opposite_side_pulses=enabled).finder().add_opposite_side_pulses is enabled
    for value in (0, 1, "false", None):
        with pytest.raises(ValueError, match="boolean"):
            RC.parse({"schema": 1, "finder": {"add_opposite_side_pulses": value}})
