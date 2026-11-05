"""Inclusive same-start diagnostics require full primary coverage and status-zero truth."""
import json

import numpy as np
import pytest

from beamfastsim import target_inclusive_transport as T
from beamfastsim.target_parent_audit import checksum


def test_inclusive_transport_hash_coverage_and_status_guards(tmp_path, monkeypatch):
    reference, census = tmp_path / 'ref.npz', tmp_path / 'census.json'
    def write(status=0):
        np.savez(reference, x=np.array([0., 1.]), y=np.zeros(2), z=np.zeros(2),
                 run=np.ones(2), subrun=np.zeros(2), evt=np.array([1, 2]), trk=np.ones(2),
                 creation_code=166, g4_status=np.full(2, status), primary_path_accepted=np.array([1, 0]))
        census.write_text(json.dumps(dict(mode='inclusive-matching-background-filter-diagnostic', generation_census_reconciled=True,
            reduced_reference_sha256=checksum(reference), generated_electrons=2, generated_events=2)))
    write(1)
    with pytest.raises(ValueError, match='nonzero G4'):
        T.compare(reference, 'unused', census, None, None)
    write()
    monkeypatch.setattr(T.CC, 'fast_plane', lambda *a: {'idx': np.array([0, 0])})
    with pytest.raises(ValueError, match='cover all'):
        T.compare(reference, 'unused', census, None, None)
    report = json.loads(census.read_text())
    report['reduced_reference_sha256'] = 'bad'
    census.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='hash-bound'):
        T.compare(reference, 'unused', census, None, None)


def test_inclusive_same_start_normalization_and_retained_replay(tmp_path, monkeypatch):
    from beamfastsim import cechain as C
    reference, census, fast_path = tmp_path / 'ref.npz', tmp_path / 'census.json', tmp_path / 'fast.npz'
    n = 3
    np.savez(reference, x=np.arange(n, dtype=float), y=np.zeros(n), z=np.zeros(n),
             px=np.zeros(n), py=np.zeros(n), pz=np.full(n, 100.), time=np.zeros(n),
             run=np.ones(n), subrun=np.zeros(n), evt=np.array([1, 1, 2]), trk=np.array([1, 2, 1]),
             creation_code=166, g4_status=np.zeros(n), primary_path_accepted=np.array([1, 1, 0]),
             ifile=np.zeros(n), gencount=np.array([2]))
    census.write_text(json.dumps(dict(mode='inclusive-matching-background-filter-diagnostic', generation_census_reconciled=True,
        reduced_reference_sha256=checksum(reference), generated_electrons=n, generated_events=2)))
    state = dict(x=np.c_[np.arange(n), np.zeros(n), np.ones(n)], u=np.tile([0., 0., 1.], (n, 1)),
                 p=np.full(n, 99.), t=np.ones(n), t0=np.zeros(n), ok=np.ones(n, bool), gen=2)
    monkeypatch.setattr(T.CC, 'g4_plane', lambda *a: state)
    def observables(geom, field, s):
        return dict(p=s['p'], cos=np.ones(n), r=s['x'][:, 0], rmax=np.full(n, 600.),
                    tof=s['t'] - s['t0'], t=s['t'], dE=np.zeros(n))
    monkeypatch.setattr(T.CC, 'observables', observables)
    np.savez(fast_path, idx=np.arange(n), fate=np.full(n, C.PLANE), x=state['x'], u=state['u'],
             T=np.full(n, np.hypot(99., 0.51099895) - 0.51099895), t=state['t'], t0=state['t0'],
             nfoil=np.ones(n), nipa=np.zeros(n), nopa=np.zeros(n), path=np.ones(n), wall=1.,
             x0=np.c_[np.arange(n), np.zeros(n), np.zeros(n)], u0=state['u'], p0=np.full(n, 100.))
    report = T.compare(reference, str(fast_path), census, None, None)
    assert report['generated_events'] == 2 and report['generated_electrons'] == 3
    assert report['cuts']['450.0']['inclusive_yield']['ratio'] == 1
    assert report['cuts']['450.0']['inclusive_yield']['event_clustered_ratio_err'] == 0
    assert report['cuts']['450.0']['retained_population_replay']['starts'] == 2
    assert not report['generated_yield_validated']
    json.dumps(report, allow_nan=False)


def test_multi_electron_event_errors_and_shuffled_fast_indices(tmp_path, monkeypatch):
    from beamfastsim import cechain as C
    reference, census, fast_path = tmp_path / 'ref.npz', tmp_path / 'census.json', tmp_path / 'fast.npz'
    n = 4
    np.savez(reference, x=np.arange(n, dtype=float), y=np.zeros(n), z=np.zeros(n), px=np.zeros(n),
             py=np.zeros(n), pz=np.full(n, 100.), time=np.zeros(n), run=np.ones(n), subrun=np.zeros(n),
             evt=np.array([1, 1, 2, 2]), trk=np.array([1, 2, 1, 2]), creation_code=166,
             g4_status=np.zeros(n), primary_path_accepted=np.array([1, 1, 0, 0]), ifile=np.zeros(n))
    census.write_text(json.dumps(dict(mode='inclusive-matching-background-filter-diagnostic',
        generation_census_reconciled=True, reduced_reference_sha256=checksum(reference),
        generated_electrons=n, generated_events=2)))
    idx = np.array([2, 0, 3, 1])
    state = dict(x=np.c_[np.arange(n), np.zeros(n), np.ones(n)], u=np.tile([0., 0., 1.], (n, 1)),
                 p=np.full(n, 99.), t=np.ones(n), t0=np.zeros(n), ok=np.ones(n, bool), gen=2)
    monkeypatch.setattr(T.CC, 'g4_plane', lambda *a: state)
    def observables(geom, field, s):
        ok = s['ok']
        return dict(p=s['p'][ok], cos=s['u'][ok, 2], r=s['x'][ok, 0], rmax=np.full(ok.sum(), 600.),
                    tof=(s['t'] - s['t0'])[ok], t=s['t'][ok], dE=np.zeros(ok.sum()))
    monkeypatch.setattr(T.CC, 'observables', observables)
    np.savez(fast_path, idx=idx, fate=np.array([C.STOPPED, C.PLANE, C.STOPPED, C.PLANE]),
             x=state['x'][idx], u=state['u'][idx], T=np.full(n, np.hypot(99., 0.51099895) - 0.51099895),
             t=state['t'][idx], t0=state['t0'][idx], nfoil=np.ones(n), nipa=np.zeros(n), nopa=np.zeros(n),
             path=np.ones(n), wall=1., x0=np.c_[np.arange(n)[idx], np.zeros(n), np.zeros(n)],
             u0=state['u'][idx], p0=np.full(n, 100.))
    report = T.compare(reference, str(fast_path), census, None, None)
    cut = report['cuts']['450.0']
    assert cut['inclusive_yield']['ratio'] == .5
    assert cut['inclusive_yield']['ratio_err'] == pytest.approx(.5)
    assert cut['inclusive_yield']['diff_err'] == pytest.approx(.5)
    assert cut['inclusive_yield']['row_iid_diagnostic_errors']['diff_err'] != pytest.approx(.5)
    assert cut['retained_population_replay']['fast_selected'] == 2
    assert cut['retained_population_replay']['g4_selected'] == 2
