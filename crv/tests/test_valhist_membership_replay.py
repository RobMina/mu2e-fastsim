"""Synthetic full-finder replay, set-alignment, census, and publication regressions."""
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from valhist import membership_replay as M


def tables(groups, events=None, sectors=None, pe=None):
    n = len(groups)
    c = pd.DataFrame(dict(ev=np.zeros(n, dtype=np.int64) if events is None else events,
        cid=np.arange(n), sectortype=np.full(n, 3) if sectors is None else sectors,
        pe=np.arange(n, dtype=float) if pe is None else pe))
    m = pd.DataFrame(dict(cid=np.repeat(np.arange(n), [len(g) for g in groups]),
                          target=np.asarray([p for g in groups for p in g], dtype=np.int64)))
    return c, m


def test_reordered_clusters_match_exact_sets_and_properties():
    a = tables([[0, 2], [1, 3]], pe=[10., 20.])
    b = tables([[3, 1], [2, 0]], pe=[20., 10.])
    got = M.compare_tables(a, b, 4)
    assert got['membership_equal']
    assert got['exact_member_set_matches'] == 2
    assert got['properties']['pe']['exact_mismatches'] == 0
    assert got['both_empty_events'] == 3
    assert got['sectors']['3']['events'] == 4


def test_regrouping_not_hidden_by_equal_member_occupancy():
    got = M.compare_tables(tables([[0, 1], [2, 3]]), tables([[0, 2], [1, 3]]), 2)
    assert got['added_membership_links'] == got['removed_membership_links'] == 0
    assert got['unmatched_reference_clusters'] == got['unmatched_candidate_clusters'] == 2
    assert got['membership_mismatch_events'] == 1
    assert not got['membership_equal']
    assert got['numerical_matched_clusters'] == 0
    assert got['properties']['pe']['rows'] == 0


def test_additions_and_removals_by_sector_and_event():
    a = tables([[0, 1], [0, 2]], events=[0, 1], sectors=[3, 4])
    b = tables([[0, 1, 2], [0]], events=[0, 1], sectors=[3, 4])
    got = M.compare_tables(a, b, 5)
    assert got['events'] == 5 and got['both_empty_events'] == 3
    assert got['membership_mismatch_events'] == 2
    assert got['cluster_count_mismatch_events'] == 0
    assert got['added_membership_links'] == got['removed_membership_links'] == 1
    assert got['sectors']['3']['added_membership_links'] == 1
    assert got['sectors']['4']['removed_membership_links'] == 1


def test_signature_is_not_ambiguous_decimal_concatenation():
    got = M.compare_tables(tables([[1, 23]]), tables([[12, 3]]), 1)
    assert not got['membership_equal']


def test_duplicate_exact_sets_withhold_ambiguous_numerical_pairing():
    got = M.compare_tables(tables([[0], [0]]), tables([[0], [0]], pe=[30., 40.]), 1)
    assert got['membership_equal']
    assert got['ambiguous_exact_matches'] == 2
    assert got['numerical_matched_clusters'] == 0


def test_empty_census_and_missing_cluster():
    assert M.compare_tables(tables([]), tables([]), 0)['membership_equal']
    got = M.compare_tables(tables([[0]], events=[2]), tables([]), 4)
    assert got['cluster_count_mismatch_events'] == 1
    assert got['removed_membership_links'] == 1
    assert got['both_empty_events'] == 3
    assert M.compare_tables(tables([[]]), tables([[]]), 2)['exact_member_set_matches'] == 1


def test_duplicate_member_and_unknown_census_rejected():
    with pytest.raises(ValueError, match='duplicate member'):
        M.compare_tables(tables([[1, 1]]), tables([]), 1)
    with pytest.raises(ValueError, match='outside census'):
        M.compare_tables(tables([[1]], events=[2]), tables([]), 1)


def census():
    return M.event_index(pd.DataFrame(dict(run=[2**40, 2**40, 2**40 + 1],
        subrun=[0, 1, 0], event=[2**30, 0, 2**30])))


def test_event_rows_do_not_pack_or_drop_empty_events():
    ev = census()
    assert ev.ev.tolist() == [0, 1, 2]
    p = ev.iloc[[2, 0]].drop(columns='ev').assign(idx=0)
    assert M._indexed(p, ev).ev.tolist() == [2, 0]
    with pytest.raises(ValueError, match='outside full event census'):
        M._indexed(p.assign(run=0), ev)
    with pytest.raises(ValueError, match='duplicate event census'):
        M.event_index(pd.concat([ev, ev]))


def geometry():
    from crvfastsim.reco import coinc as C
    layer = np.repeat(np.arange(4), 2)
    x = np.tile([0., 40.], 4)
    y = layer * 20.
    return C.SectorGeom(np.zeros(8, int), layer, np.tile([0, 1], 4), x, y,
        np.column_stack([x, y, np.zeros(8)]), np.array([2]), np.array([3]), ['T2'],
        np.array([-1400.]), np.array([1400.]), np.array([True]), np.array([True]))


def pulses(ev):
    p = pd.DataFrame(dict(idx=np.arange(7), bar=[0, 2, 4, 6, 0, 0, 0],
        sipm=[0, 0, 0, 0, 1, 1, 3], time=[100., 100., 100., 100., 104., 102., 103.],
        pe=np.full(7, 10.), seq=np.zeros(7, dtype=np.int64)))
    return p.assign(**{n: ev.iloc[0][n] for n in M.R.IDS})


@pytest.mark.parametrize('tiers,fallback', [(((16, 8, 2),), 0), (((4, 1, 2),), 1)])
def test_full_candidate_enabled_disabled_and_fallback(tiers, fallback):
    pytest.importorskip('jax')
    from crvfastsim.reco import coinc as C
    from crvfastsim.reco import coinc_jax as J
    ev = census()
    p = pulses(ev)
    G, status = geometry(), np.zeros(32, np.int64)
    disabled, md = M.replay(ev, p, G, status, J, replace(C.FinderConfig(), add_opposite_side_pulses=False),
                            tiers=tiers, split=False)
    enabled, me = M.replay(ev, p, G, status, J, C.FinderConfig(), tiers=tiers, split=False)
    assert set(disabled[1].target) == {0, 1, 2, 3}
    assert set(enabled[1].target) == {0, 1, 2, 3, 5, 6}
    assert enabled[0].iloc[0].poscalc
    assert md['fallback_events'] == me['fallback_events'] == fallback
    assert me['empty_pulse_events'] == 2 and me['events'] == 3
    assert me['finder_seconds'] >= 0
    got = M.compare_tables(disabled, enabled, len(ev))
    assert got['added_membership_links'] == 2 and got['removed_membership_links'] == 0
    scalar = C.find_event({n: p[n].to_numpy() for n in ('idx', 'bar', 'sipm', 'time', 'pe', 'seq')}, G, status)
    for name in ('pe', 'start', 'end', 'slope', 'avgtime', 'poscalc'):
        np.testing.assert_allclose(enabled[0][name], scalar[0][name], atol=1e-10)
    np.testing.assert_allclose(enabled[0][['x', 'y', 'z']].to_numpy()[0], scalar[0]['pos'], atol=1e-10)


def test_empty_full_candidate_call():
    pytest.importorskip('jax')
    from crvfastsim.reco import coinc as C
    from crvfastsim.reco import coinc_jax as J
    ev = census()
    got, metrics = M.replay(ev, pulses(ev).iloc[:0], geometry(), np.zeros(32, np.int64), J,
                            C.FinderConfig(), tiers=((16, 8, 2),), split=False)
    assert len(got[0]) == len(got[1]) == 0
    assert metrics['empty_pulse_events'] == 3


def test_finder_membership_event_disagreement():
    out = dict(cl_event=np.array([0]), cl_type=np.array([3]), cl_pos=np.zeros((1, 3)),
               mem_cl=np.array([0]), mem_idx=np.array([0]), mem_event=np.array([1]))
    out.update({f'cl_{n}': np.array([0.]) for n in M.PROPERTIES[:6]})
    with pytest.raises(ValueError, match='event/cluster disagreement'):
        M.output_tables(out, pd.DataFrame(dict(ev=[0], idx=[0])), 2)


def test_existing_and_symlink_output_refused(tmp_path):
    out = tmp_path / 'existing.json'
    out.write_text('old')
    with pytest.raises(FileExistsError):
        M.create('missing', out)
    assert out.read_text() == 'old'
    link = tmp_path / 'link.json'
    link.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        M.create('missing', link)


def test_output_must_be_json_outside_code_attribution_and_inputs(tmp_path):
    source = tmp_path / 'code' / 'valhist' / 'frozen.py'
    watched = {str(source): 'a' * 64}
    for out in (source.parent.parent / 'out.json', tmp_path / 'attribution' / 'out.json'):
        with pytest.raises(ValueError, match='outside'):
            M._output_guard(out, tmp_path / 'attribution', watched, [])
    with pytest.raises(ValueError, match='JSON'):
        M._output_guard(tmp_path / 'out.txt', 'missing', {}, [])


def publication_setup(monkeypatch, tmp_path, mutation=None):
    frozen = tmp_path / 'frozen.dat'
    frozen.write_text('sealed')
    source = tmp_path / 'source.py'
    source.write_text('original')
    core = tmp_path / 'core'
    (core / 'mu2efastsim').mkdir(parents=True)
    watched = {str(frozen): M.digest(frozen)}
    monkeypatch.setattr(M.R, '_validate', lambda _: ({'shards': [7]}, [(7, 'ref', 'hybrid', 3)], dict(watched)))
    monkeypatch.setattr(M, '_inventory', lambda _: {str(source): M.digest(source)})
    monkeypatch.setattr(M, '_check_imports', lambda *args: {})
    J = SimpleNamespace(jax=SimpleNamespace(default_device=lambda _: nullcontext()))
    monkeypatch.setattr(M, '_runtime', lambda *args: (None, None, J, None, None, {}))
    def run(*args):
        if mutation:
            mutation(frozen, source)
        effect = dict(added_membership_links=2, removed_membership_links=0, membership_mismatch_events=1,
                      cluster_count_mismatch_events=0, sectors={})
        return dict(replay={arm: dict(enabled_vs_disabled=effect, controls={'disabled': {
            'comparisons': {'hybrid': {'membership_equal': True}}}})
            for arm in ('a1_persisted_pulses', 'hybrid_persisted_pulses')})
    monkeypatch.setattr(M, 'replay_pair', run)
    return core, watched


def test_publish_binds_sources_and_frozen_inputs(monkeypatch, tmp_path):
    core, watched = publication_setup(monkeypatch, tmp_path)
    out = tmp_path / 'new.json'
    result = M.create(tmp_path / 'attribution', out, core=core)
    assert out.is_file()
    assert result['hashes_verified_before_and_after']
    assert result['disabled_hybrid_memberships_close']
    assert result['attachment_effect']['hybrid_persisted_pulses']['added_membership_links'] == 2
    assert result['frozen_attribution_sha256'] == watched
    assert result['analysis_source_sha256']


@pytest.mark.parametrize('kind', ['frozen', 'source'])
def test_drift_refuses_publication(monkeypatch, tmp_path, kind):
    def mutate(frozen, source):
        (frozen if kind == 'frozen' else source).write_text('changed')
    core, _ = publication_setup(monkeypatch, tmp_path, mutate)
    out = tmp_path / 'new.json'
    with pytest.raises(ValueError, match='changed'):
        M.create(tmp_path / 'attribution', out, core=core)
    assert not out.exists()


def test_exclusive_publish_race_preserves_other_file(monkeypatch, tmp_path):
    out = tmp_path / 'new.json'
    core, _ = publication_setup(monkeypatch, tmp_path, lambda *args: out.write_text('other writer'))
    with pytest.raises(FileExistsError):
        M.create(tmp_path / 'attribution', out, core=core)
    assert out.read_text() == 'other writer'


def test_wrong_tree_import_refused(monkeypatch, tmp_path):
    monkeypatch.setitem(M.sys.modules, 'crvfastsim.unbound_test', SimpleNamespace(__file__=str(tmp_path / 'evil.py')))
    with pytest.raises(ValueError, match='unbound or wrong-tree'):
        M._check_imports(Path(M.__file__).resolve().parents[1], tmp_path, {})


def test_in_memory_pair_runs_both_pulse_arms_and_controls(monkeypatch):
    pytest.importorskip('jax')
    from crvfastsim.reco import coinc as C
    from crvfastsim.reco import coinc_jax as J
    ev = census()
    p = pulses(ev)
    G, status, cfg = geometry(), np.zeros(32, np.int64), C.FinderConfig()
    enabled, _ = M.replay(ev, p, G, status, J, cfg, tiers=((16, 8, 2),), split=False)
    disabled, _ = M.replay(ev, p, G, status, J, replace(cfg, add_opposite_side_pulses=False),
                           tiers=((16, 8, 2),), split=False)
    def persisted(result):
        c, m = result
        c = c.merge(ev, on='ev').drop(columns='ev').rename(columns={'cid': 'idx'})
        links = m.rename(columns={'cid': 'idx'}).merge(c[M.R.KEY], on='idx')
        waveform = p[M.R.KEY].assign(target=p.idx)
        return ({'events': ev[M.R.IDS], 'pulses': p, 'clusters': c},
                {'clusters': links[M.R.KEY + ['target']], 'pulses': waveform}, {})
    data = {'a1': persisted(enabled), 'hybrid': persisted(disabled)}
    monkeypatch.setattr(M.R, '_read', lambda path: data[path])
    got = M.replay_pair('a1', 'hybrid', 3, G, status, J, cfg)
    for arm in ('a1_persisted_pulses', 'hybrid_persisted_pulses'):
        r = got['replay'][arm]
        assert r['controls']['disabled']['comparisons']['hybrid']['membership_equal']
        assert r['controls']['enabled']['comparisons']['a1']['membership_equal']
        assert r['enabled_vs_disabled']['added_membership_links'] == 2
    summary = M.attachment_summary({'7': got, '8': got})
    assert summary['hybrid_persisted_pulses']['added_membership_links'] == 4
    assert summary['a1_persisted_pulses']['sectors']['3']['added_membership_links'] == 4
    assert got['events'] == 3


@pytest.mark.parametrize('bad', [[0.5], [np.nan], [np.inf], [float(2**53)], [-1]])
def test_invalid_finder_integer_arrays(bad):
    with pytest.raises(ValueError, match='finder integer'):
        M._integer_array(bad, 'test')
