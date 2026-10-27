"""Current-main paired replay and exposure diagnostics fail closed."""
import copy

import awkward as ak
import numpy as np
import pytest

from crvfastsim.veto.exposure_audit import audit_manifest, count_identities, pbi_moment
from crvfastsim.veto.mixframes import MIXERS, PBI, PBTIMES
from crvfastsim.veto.replay_audit import equal_pulses, replica_stats, tables
from valhist.membership_replay import compare_tables


def test_replicas_keep_unique_stock_frames():
    a, b, s = np.array([0., .2, 1., 0.]), np.array([.1, 0., 1., 0.]), np.array([0., 0., 1., 0.])
    r = replica_stats(a, b, s, [0, 0, 1, 1])
    assert r['frames'] == r['unique_frames'] == 4
    assert r['delta'] == pytest.approx(.0375)
    assert r['delta_err_paired'] == pytest.approx(np.std((a + b) / 2 - s, ddof=1) / 2)
    assert r['replica_contrast']['delta'] == pytest.approx(-.025)
    assert replica_stats(a, a, s, [0] * 4)['delta_err_blocks'] is None


@pytest.mark.parametrize('a,b,s,g', [([0, 0], [0], [0, 0], [0, 0]),
                                     ([-1], [1], [0], [0]), ([0], [0], [0], [])])
def test_replica_shape_and_each_draw_domain(a, b, s, g):
    with pytest.raises(ValueError):
        replica_stats(a, b, s, g)


def test_ordered_pulses_not_nearest_time():
    p = ak.Array({'time': [1., 2.], 'chan': [0, 1]})
    assert equal_pulses(p, p)
    assert not equal_pulses(p, p[::-1])
    assert not equal_pulses(p, ak.with_field(p, [1., 2.01], 'time'))


def test_full_identity_membership_and_empty_frames():
    events = ak.Array({'run': [1, 1], 'subrun': [0, 1], 'event': [7, 7]})
    c = ak.Array({'run': [1], 'subrun': [1], 'event': [7], 'idx': [0], 'sectortype': [1],
                  'pe': [30.], 'start': [100.], 'end': [110.], 'avgtime': [105.], 'pulses': [[0, 2]]})
    a = tables(c, events)
    assert a[0].ev.iloc[0] == 1
    assert compare_tables(a, a, 2)['both_empty_events'] == 1
    changed = tables(ak.with_field(c, [[0, 1]], 'pulses'), events)
    assert not compare_tables(a, changed, 2)['membership_equal']
    with pytest.raises(ValueError, match='duplicate event'):
        tables(c, ak.concatenate([events, events]))


def test_generator_census_uses_full_tuple_and_exposes_old_collisions():
    result = count_identities({'run': [1, 1, 2], 'subrun': [0, 1, 0], 'event': [2**20, 0, 2**20]})
    assert result['full_event_ids'] == 3
    assert result['historical_packed_ids'] == 1
    assert result['packed_collision']
    assert result['event_outside_20bit'] == 2


def test_pbi_weighted_lognormal_moments_and_truncation():
    assert pbi_moment(0) == pytest.approx(1)
    assert pbi_moment(0, beam_conditioned=True) == pytest.approx(1)
    assert pbi_moment(1, cut_max=1e30) == pytest.approx(1.58e7)
    assert pbi_moment(1, beam_conditioned=True, cut_max=1e30) == pytest.approx(1.58e7 / .8)
    assert 1.24 < pbi_moment(1, beam_conditioned=True) / pbi_moment(1) < 1.25
    with pytest.raises(ValueError):
        pbi_moment(1, sdf=1)


def manifest():
    return {'events': 2, 'pbi': PBI, 'mixers': {k: list(v) for k, v in MIXERS.items()},
            'mean_intensity': 1.58e7, 'mean_frames_per_event': {k: 1.58e7 * e * f for k, (e, f) in MIXERS.items()},
            'sources': {k: [{'path': 'source.parquet', 'sha256': 'a' * 64,
                            'metadata': {'dataset': k, 'frames': 10, 'stepped_frames': 2, 'steps': 8}}] for k in MIXERS}}


def test_duplicate_content_not_hidden_by_disjoint_filenames():
    from crvfastsim.veto.exposure_audit import duplicate_payloads
    a, b = manifest(), manifest()
    b['sources']['NeutralsFlashCat'][0]['path'] = 'different.parquet'
    result = duplicate_payloads([a, b])['NeutralsFlashCat']
    assert result['table_occurrences'] == 2 and result['unique_payloads'] == 1
    assert len(result['duplicate_groups']['a' * 64]) == 2


def test_db_efficiency_payload_missing_tag_rejected(tmp_path):
    from crvfastsim.veto.exposure_audit import efficiency_audit
    path = tmp_path / 'eff.csv'
    path.write_text('tag,eff\nEleBeamCat,0.05532579\n')
    with pytest.raises(KeyError):
        efficiency_audit(path)


def test_saved_exposure_conditional_means():
    r = audit_manifest(manifest(), [1.58e7] * 2, PBTIMES[:2])
    assert r['pools']['EleBeamFlashCat']['conditional_poisson_z'] == 0
    assert r['pools']['MuStopPileupCat']['step_occupancy'] == .2


@pytest.mark.parametrize('field', ['events', 'mean_intensity', 'sources', 'mixers'])
def test_saved_exposure_rejects_contract_mutation(field):
    m = copy.deepcopy(manifest())
    if field in ('events', 'mean_intensity'):
        m[field] += 1
    elif field == 'sources':
        m[field]['EleBeamFlashCat'][0]['metadata']['stepped_frames'] = 11
    else:
        m[field]['EleBeamFlashCat'][1] = 3
    with pytest.raises(ValueError):
        audit_manifest(m, [1.58e7] * 2, PBTIMES[:2])
