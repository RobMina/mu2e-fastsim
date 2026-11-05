"""Synthetic development-inventory contracts; not evidence of physics closure."""
import json

import numpy as np
import pytest

from beamfastsim import target_closure_inventory as T


def test_conditional_tail_original_event_influence_and_denominators():
    groups = np.array([0, 0, 1, 1, 2, 2])
    sa = np.array([1, 1, 1, 0, 0, 0], bool)
    sb = np.array([1, 0, 1, 1, 1, 0], bool)
    ta = np.array([1, 0, 0, 0, 0, 0], bool)
    tb = np.array([0, 0, 1, 0, 1, 0], bool)
    result = T.conditional_difference(sa, sb, ta, tb, groups)
    assert result['g4'] == pytest.approx(1 / 3)
    assert result['fast'] == .5
    assert result['diff'] == pytest.approx(1 / 6)
    # Event influence sums are (-17, 8, 9)/72, including denominator fluctuations.
    assert result['event_se'] == pytest.approx(np.sqrt(651) / 72)
    iid = T.conditional_difference(sa, sb, ta, tb, np.arange(6))
    assert result['event_se'] != pytest.approx(iid['event_se'])
    perm = np.array([5, 0, 3, 1, 4, 2])
    reordered = T.conditional_difference(sa[perm], sb[perm], ta[perm], tb[perm], groups[perm])
    assert reordered == pytest.approx(result)
    assert T.event_se(np.array([0., 0.]), np.array([1, 1])) is None
    empty = T.conditional_difference(sa & False, sb, ta & False, tb, groups)
    assert empty['g4'] is None and empty['event_se'] is None


def test_cdf_ties_and_empty():
    assert T.cdf_sup(np.array([0., 0., 2.]), np.array([0., 1.])) == pytest.approx(1 / 3)
    assert T.cdf_sup(np.array([1., 1.]), np.array([1.])) == 0.
    assert T.cdf_sup(np.array([]), np.array([1.])) is None


def test_joint_full_measure_outside_flow_tv_and_sparse():
    a = dict(p=np.array([70., 70., np.nan, np.nan]), rmax=np.array([600., 600., np.nan, np.nan]))
    b = dict(p=np.array([70., np.nan, np.nan, np.nan]), rmax=np.array([600., np.nan, np.nan, np.nan]))
    sa, sb = np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 0], bool)
    joint = T.joint_tv(a, b, sa, sb, ('p', 'rmax'))
    assert joint['tv'] == .25
    assert sum(joint['g4_counts']) == sum(joint['fast_counts']) == 4
    assert joint['g4_counts'][0] == 2 and joint['fast_counts'][0] == 3
    assert joint['insufficient_cell_support']
    assert joint['g4_zero_cells'] > 0 and joint['sparse_cells'] > 0
    assert T.joint_tv(a, a, sa, sa, ('p', 'rmax'))['tv'] == 0


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    ref, fast, census = (tmp_path / name for name in ('reference.npz', 'fast.npz', 'census.json'))
    n = 6
    xyz = np.c_[np.arange(n, dtype=float), np.zeros(n), np.zeros(n)]
    direction = np.tile([0., 0., 1.], (n, 1))
    rows = dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], px=np.zeros(n), py=np.zeros(n),
        pz=np.full(n, 100.), time=np.zeros(n), run=np.ones(n, int), subrun=np.zeros(n, int),
        evt=np.repeat(np.arange(3), 2), trk=np.tile([1, 2], 3), creation_code=166,
        g4_status=np.zeros(n, int), primary_path_accepted=np.repeat([1, 0, 1], 2),
        isvd_front=np.array([1, 1, 1, 1, 0, 0]), xvd_front=xyz[:, 0], yvd_front=xyz[:, 1],
        zvd_front=xyz[:, 2], pxvd_front=np.zeros(n), pyvd_front=np.zeros(n),
        pzvd_front=np.full(n, 99.), tvd_front=np.ones(n), gencount=np.array([3]))
    idx = np.array([4, 0, 3, 1, 5, 2])
    # The lost G4 daughter has a different fast selection; the last event has no plane rows.
    ok = np.array([0, 1, 0, 1, 0, 1], bool)
    parts = dict(idx=idx, fate=np.where(ok, T.C.PLANE, T.C.STOPPED), x=xyz[idx], u=direction[idx],
        T=np.full(n, np.hypot(99.5, T.MASS) - T.MASS), t=np.ones(n), t0=np.zeros(n),
        x0=xyz[idx], u0=direction[idx], p0=np.full(n, 100.))
    receipt = dict(mode='inclusive-matching-background-filter-diagnostic', generation_census_reconciled=True,
                   generated_events=3, generated_electrons=6, generation_source_sha256='a' * 64)
    def save():
        np.savez(ref, **rows)
        np.savez(fast, **parts)
        receipt['reduced_reference_sha256'] = T.checksum(ref)
        census.write_text(json.dumps(receipt))
    def observables(geom, field, state):
        hit = state['ok']
        return dict(p=state['p'][hit], cos=state['u'][hit, 2], r=state['x'][hit, 0],
                    rmax=np.full(hit.sum(), 600.), t=state['t'][hit],
                    tof=(state['t'] - state['t0'])[hit])
    monkeypatch.setattr(T.CC, 'observables', observables)
    save()
    return ref, fast, census, rows, parts, receipt, save


def test_inventory_own_selection_zero_tail_no_plane_and_planning(pilot):
    ref, fast, census, *_ = pilot
    report = T.inventory(ref, str(fast), census, None, None, 12)
    cut = report['cuts']['450.0']
    counts = cut['yield_per_emitted_electron']
    assert counts['generated_events'] == 3 and counts['generated_electrons'] == 6
    assert counts['g4_count'] == 4 and counts['fast_count'] == 3
    assert counts['ratio'] == .75
    assert counts['event_se'] == pytest.approx(1 / 6)
    assert counts['ratio_precision']['event_se'] == pytest.approx(np.sqrt(3) / 8)
    assert counts['insufficient_yield_support']
    assert counts['projected_event_se'] is None
    assert counts['projected_ci_halfwidth'] is None
    assert counts['ratio_precision']['projected_ci_halfwidth'] is None
    assert T._projection(.1, .5, 4)['projected_ci_halfwidth'] == pytest.approx(.2)
    assert T._projection(0., .5, 4, supported=False)['projected_ci_halfwidth'] is None
    ga = np.hypot(100., T.MASS) - np.hypot(99., T.MASS)
    fb = np.hypot(100., T.MASS) - np.hypot(99.5, T.MASS)
    assert cut['loss_core']['g4'] == pytest.approx([ga] * 3)
    assert cut['loss_core']['fast'] == pytest.approx([fb] * 3)
    assert cut['loss_core']['median_diff'] == pytest.approx(fb - ga)
    assert cut['loss_core']['halfwidth68_diff'] == pytest.approx(0.)
    assert cut['tails']['0.25']['g4_selected'] == 4
    assert cut['tails']['0.25']['fast_selected'] == 3
    zero = cut['tails']['2.0']
    assert zero['g4'] == zero['fast'] == 0
    assert zero['insufficient_tail_support']
    assert zero['event_se'] is None and zero['projected_ci_halfwidth'] is None
    assert zero['planned_events_for_observed_support'] is None
    assert not zero['normal_decision_established']
    assert cut['joints']['p,rmax']['g4_counts'][0] == 2
    assert cut['joints']['p,rmax']['fast_counts'][0] == 3
    assert report['development_only'] and not report['power_established']
    assert not report['certificate_established'] and not report['generated_yield_validated']
    assert 'parent/library' in report['uncertainty_scope']
    assert set(report['cuts']) == {'450.0', '500.0', '550.0'}
    assert set(cut['cdf']) == set(T.METRICS)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('change,match', [
    ('status', 'nonzero G4'), ('shape', 'shapes'), ('duplicate', 'identity/count'),
    ('count', 'identity/count'), ('coverage', 'cover all'), ('fractional_idx', 'cover all'),
    ('birth', 'reference starts'), ('event_decision', 'within event'),
    ('generation_hash', 'generation census hash'), ('reference_hash', 'hash-bound')])
def test_saved_guards(pilot, change, match):
    ref, fast, census, rows, parts, receipt, save = pilot
    if change == 'status':
        rows['g4_status'][0] = 1
    elif change == 'shape':
        rows['evt'] = np.zeros((6, 1), int)
    elif change == 'duplicate':
        rows['trk'][1] = rows['trk'][0]
    elif change == 'count':
        receipt['generated_events'] = 4
    elif change == 'coverage':
        parts['idx'][0] = parts['idx'][1]
    elif change == 'fractional_idx':
        parts['idx'] = parts['idx'].astype(float) + .1
    elif change == 'birth':
        parts['p0'][0] += 1
    elif change == 'event_decision':
        rows['primary_path_accepted'][0] = 0
    elif change == 'generation_hash':
        receipt['generation_source_sha256'] = 'bad'
    save()
    if change == 'reference_hash':
        receipt['reduced_reference_sha256'] = 'bad'
        census.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        T.inventory(ref, str(fast), census, None, None, 30)


@pytest.mark.parametrize('planned,critical', [(0, 4), (1.5, 4), (10, 0), (10, np.inf)])
def test_planning_arguments(planned, critical):
    with pytest.raises(ValueError, match='planned-events'):
        T.inventory('unused', 'unused', 'unused', None, None, planned, critical)


def test_cli_strict_json_and_exclusive_output(tmp_path, monkeypatch):
    out = tmp_path / 'inventory.json'
    monkeypatch.setattr(T.C, 'from_offline', lambda *a: None)
    monkeypatch.setattr(T.C, 'field_maps', lambda *a: None)
    seen = {}
    def inventory(*args):
        seen['arguments'] = args
        return dict(development_only=True, certificate_established=False)
    monkeypatch.setattr(T, 'inventory', inventory)
    argv = ['--ref', 'ref.npz', '--fast', '*.npz', '--census', 'census.json',
            '--offline-root', '/offline', '--maps', 'one', 'two', '--planned-events', '120', '--out', str(out)]
    T.main(argv)
    assert json.loads(out.read_text())['development_only']
    assert seen['arguments'][-2:] == (120, 4.)
    with pytest.raises(FileExistsError):
        T.main(argv)
    out.unlink()
    monkeypatch.setattr(T, 'inventory', lambda *a: {'bad': float('nan')})
    with pytest.raises(ValueError):
        T.main(argv)
    assert not out.exists()
