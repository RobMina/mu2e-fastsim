"""Synthetic CLI regressions for reconstruction profiling; no JAX or detector data."""
import itertools
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import awkward as ak
import numpy as np
import pytest

from calofastsim.showers import prepare_chain, reconstruct
from calofastsim.showers.export import EVENT, keys
from calofastsim.showers.library import save
from test_showers_response import source


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from beamfastsim import cechain
    import beamfastsim

    rows = np.array([1, 3])
    conditions = dict(run=np.array([1, 1, 2, 2]), subrun=np.array([0, 1, 0, 1]), event=np.ones(4, int),
                      pbtime=np.full(4, -200.), mbtime=np.full(4, 1695.), event_length=np.full(4, 1700.),
                      marker=np.full(4, 200.), crystal_length=np.full(4, 200.), spill=np.ones(4, int))
    starts = dict(x0=np.zeros((4, 3)), u0=np.tile([0., 0., 1.], (4, 1)), p0=np.full(4, 100.), t0=np.arange(4) + 1000.)
    stock = {k: np.zeros(4) if k == 'energy' else np.full(4, np.nan) for k in ('energy', 'time', 'x', 'y', 'z')}
    np.savez(tmp_path / 'inputs.npz', **starts, **{'condition_' + k: v for k, v in conditions.items()},
             **{'stock_cluster_' + k: v for k, v in stock.items()})
    np.savez(tmp_path / 'chain.npz', event_row=rows, fate=np.full(2, cechain.STOPPED),
             **{'tracker_' + k: starts[s][rows] for k, s in [('x', 'x0'), ('u', 'u0'), ('T', 'p0'), ('t', 't0')]})
    lib, _, _ = source()
    save(tmp_path / 'library.npz', lib, dict(split_row=2, split='synthetic'))
    state = NS(path=tmp_path, rows=rows, ids=keys(conditions, EVENT), placements=[], calls=[], barriers=[],
               drift=None, empty=False, reco_calls=0, bad_transport=False)
    fake_jax = NS(config=NS(update=lambda *a: None), default_backend=lambda: 'cpu', devices=lambda: ['cpu'],
                  random=NS(PRNGKey=lambda seed: seed), effects_barrier=lambda: state.barriers.append(True))
    monkeypatch.setitem(sys.modules, 'jax', fake_jax)
    clock = itertools.count()
    monkeypatch.setattr(reconstruct.time, 'perf_counter', lambda: float(next(clock) ** 2))
    monkeypatch.setattr(cechain, 'field_maps', lambda *a: [0., 0., 0.])
    monkeypatch.setattr(cechain, 'from_offline', lambda *a: None)

    def transport(seed, geometry, field, **kw):
        state.calls.append(kw)
        for k in starts:
            np.testing.assert_array_equal(kw[k], starts[k][rows])
        return NS(fate=np.full(2, cechain.STOPPED), x=kw['x0'], u=kw['u0'], T=kw['p0'],
                  t=kw['t0'] + int(state.bad_transport))

    monkeypatch.setattr(cechain, 'transport', transport)
    device = NS(transport=transport, device_maps=lambda *a: [0., 0., 0.])
    monkeypatch.setitem(sys.modules, 'beamfastsim.cechain_jax', device)
    monkeypatch.setattr(beamfastsim, 'cechain_jax', device, raising=False)
    handoff = reconstruct.from_cechain

    def recorded_handoff(*args, **kwargs):
        state.placements.append(kwargs['placement'])
        result = handoff(*args, **kwargs)
        assert result['generated_events'] == 2 and result['tracker_entry_events'] == 0
        return result

    monkeypatch.setattr(reconstruct, 'from_cechain', recorded_handoff)

    def run_chain(steps, timing, *args, **kwargs):
        assert len(timing['pbtime']) == 2 and len(steps['event']) == 0
        clusters = dict(event=np.array([0], int), energy=np.array([20.]), time=np.array([1001.]),
                        x=np.array([1.]), y=np.array([2.]), z=np.array([3.]))
        if state.drift and state.reco_calls:
            clusters[state.drift] += 1
        if state.empty:
            clusters = {k: v[:0] for k, v in clusters.items()}
        state.reco_calls += 1
        return dict(clusters=clusters, n_refit_skipped=0, n_nonfinite_reco=0)

    io = NS(load_sim_crystal=lambda *a: dict(lru=np.zeros(12), pe_per_mev=np.ones((12, 2))),
            load_pulse=lambda *a: None, load_noise=lambda *a: None, load_propagation=lambda *a: None,
            adc_per_pe=lambda *a: 1., adc_to_mev=lambda *a: 1.)
    monkeypatch.setattr(reconstruct, 'load_chain', lambda *a: (NS(run_chain=run_chain), io, NS(Geometry=NS(load=lambda *a: None))))

    def run(live=None, parent_repeats=2, extra=()):
        (tmp_path / 'parent.json').write_text(json.dumps({'timing': {'stock_generated_per_second': 23.9,
            'measurements': [{'transport_seconds': 1000.}] * parent_repeats}}))
        args = ['reconstruct', '--chain', str(tmp_path / 'chain.npz'), '--chain-report', str(tmp_path / 'parent.json'),
                '--inputs', str(tmp_path / 'inputs.npz'), '--library', str(tmp_path / 'library.npz'),
                '--reco-code', str(tmp_path), '--reco-inputs', str(tmp_path), '--out', str(tmp_path / 'out'),
                '--maps', 'synthetic', '--repeat', '2', '--device', 'cpu', '--reco-commit', 'synthetic']
        if live:
            args += ['--live-transport', live, '--transport-geometry', str(tmp_path / 'geometry')]
        monkeypatch.setattr(sys, 'argv', args + list(extra))
        reconstruct.main()
        return json.loads((tmp_path / 'out/numbers.json').read_text())

    state.run = run
    return state


@pytest.mark.parametrize('live', [None, 'numpy', 'device'])
@pytest.mark.parametrize('placement', ['geometric', 'crystal'])
def test_cold_warm_exact_observables_and_generated_denominator(profile, live, placement):
    expected = dict(energy=[20., 0.], time=[1001., np.nan], x=[1., np.nan], y=[2., np.nan], z=[3., np.nan])
    path = profile.path / 'expected.npz'
    np.savez(path, event_row=profile.rows, **{'fast_' + k: v for k, v in expected.items()})
    got = profile.run(live, parent_repeats=1 if live else 2, extra=['--placement', placement, '--expect-clusters', str(path)])
    assert got['events'] == 2 and got['fast_selected'] == 1 and got['stock_selected'] == 0
    assert got['acceptance_difference']['fast'] == .5
    assert profile.placements == [placement, placement]
    assert [r['repeat'] for r in got['repeats']] == [0, 1]
    assert len(profile.calls) == (2 if live else 0)
    assert len(profile.barriers) == (4 if live else 2)
    timing = got['timing']
    assert timing['method'].startswith('single-call' if live else 'sum of separately')
    warm = got['repeats'][1]
    assert timing['warm_transport_seconds'] == (warm['transport_seconds'] if live else 1000.)
    total = warm['total_seconds'] if live else 1000. + warm['downstream_seconds']
    assert timing['composed_chain_seconds'] == total
    assert timing['generated_per_second'] == 2 / total
    with np.load(profile.path / 'out/clusters.npz') as saved:
        np.testing.assert_array_equal(saved['event_row'], profile.rows)
        for k, values in expected.items():
            np.testing.assert_array_equal(saved['fast_' + k], values)


def test_expected_event_rows_must_match(profile):
    path = profile.path / 'expected.npz'
    np.savez(path, event_row=profile.rows[::-1])
    with pytest.raises(ValueError, match='different event rows'):
        profile.run(extra=['--expect-clusters', str(path)])


@pytest.mark.parametrize('live', ['numpy', 'device'])
def test_live_transport_must_match_saved_states(profile, live):
    profile.bad_transport = True
    with pytest.raises(RuntimeError, match='saved same-seed transport'):
        profile.run(live)


def test_all_zero_events_retained_and_composed_requires_warm_parent(profile):
    profile.empty = True
    got = profile.run('numpy', parent_repeats=1)
    assert got['events'] == 2 and got['fast_selected'] == got['stock_selected'] == 0
    with pytest.raises(ValueError, match='warm upstream'):
        profile.run(parent_repeats=1)


@pytest.mark.parametrize('field', ['time', 'x', 'y', 'z'])
@pytest.mark.parametrize('reference', ['expected', 'repeat'])
def test_same_energy_different_observable_fails(profile, field, reference):
    extra = []
    if reference == 'expected':
        values = dict(energy=[20., 0.], time=[1001., np.nan], x=[1., np.nan], y=[2., np.nan], z=[3., np.nan])
        values[field][0] += 1
        path = profile.path / 'expected.npz'
        np.savez(path, event_row=profile.rows, **{'fast_' + k: v for k, v in values.items()})
        extra = ['--expect-clusters', str(path)]
    else:
        profile.drift = field
    with pytest.raises(RuntimeError, match='leading-cluster observables'):
        profile.run('numpy', extra=extra)


@pytest.mark.parametrize('bad_ids', [None, 'missing', 'duplicate'])
def test_stock_override_full_identity_before_noncontiguous_selection(profile, monkeypatch, bad_ids):
    order = np.array([3, 0, 2, 1])
    ids = {k: profile.ids[k][order].copy() for k in EVENT}
    if bad_ids == 'missing':
        ids['run'][0] = 99
    elif bad_ids == 'duplicate':
        ids = {k: np.repeat(v[:1], 4) for k, v in ids.items()}
    fields = ('energyDep_', 'time_', 'cog_.fCoordinates.fX', 'cog_.fCoordinates.fY', 'cog_.fCoordinates.fZ')
    data = {'evtinfo': ak.Array(ids), 'caloclusters': {'caloclusters.' + k: ak.Array([[float(i + 11)] for i in order]) for k in fields}}
    tree = NS(classname='TTree', arrays=lambda *a, **kw: data)
    monkeypatch.setattr(prepare_chain.uproot, 'open', lambda *a: nullcontext({'prd/ntuple': tree}))
    reader = prepare_chain.stock_clusters

    def checked_reader(path, ids):
        np.testing.assert_array_equal(ids, profile.ids)
        assert len(np.unique(ids)) == 4 and len(np.unique(ids['event'])) == 1
        return reader(path, ids)

    monkeypatch.setattr(prepare_chain, 'stock_clusters', checked_reader)
    extra = ['--stock-ntuple', str(profile.path / 'stock.root')]
    if bad_ids:
        with pytest.raises(ValueError, match='identity|every input event once'):
            profile.run(extra=extra)
    else:
        profile.run(extra=extra)
        with np.load(profile.path / 'out/clusters.npz') as saved:
            for k in ('energy', 'time', 'x', 'y', 'z'):
                np.testing.assert_array_equal(saved['stock_' + k], [12., 14.])
