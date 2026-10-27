"""Synthetic retained-file orchestration and provenance failures, without real detector inputs."""
import copy
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from crvfastsim.objective import joint_shapes as A, joint_shapes_report as Report
from crvfastsim.objective import joint_compare as J, joint_data as D, joint_pairs as P, joint_timing as T
from test_joint_diagnosis import sample
from test_joint_data import reference


@pytest.fixture
def evidence(tmp_path):
    c, p, _ = sample()
    p = ak.Array({name: ak.concatenate([p[name][:1], p[name][:1]], axis=1) for name in p.fields})
    count = ak.to_numpy(ak.num(p['crvpulses.barId']))
    sipm = ak.to_numpy(ak.flatten(p['crvpulses.SiPMId']))
    sipm[len(sipm)//2:] = (sipm[len(sipm)//2:]+2) % 4
    p = ak.with_field(p, ak.unflatten(sipm, count), 'crvpulses.SiPMId')
    for name, value in (('pulseBeta', 14.), ('pulseHeight', 120.), ('PEsPulseHeight', 10.), ('pulseFitChi2', 3.)):
        p = ak.with_field(p, ak.ones_like(p['crvpulses.PEs'])*value, 'crvpulses.'+name)
    c, p = ak.concatenate([c[:1]]*36+[c[-1:]]), ak.concatenate([p]*36+[p[:0]])
    p = ak.concatenate([p, ak.Array({name: ak.unflatten(np.empty(0, ak.to_numpy(ak.flatten(p[name])).dtype), [0])
                                   for name in p.fields})])
    keys = np.column_stack((np.full(37, 125026), np.ones(37, int), np.arange(37)+2**53))
    columns = {name: c[name] for name in c.fields}
    columns.update({name: p[name] for name in p.fields})
    columns.update({'evtinfo/'+name: keys[:, i] for i, name in enumerate(('run', 'subrun', 'event'))})
    source = tmp_path / 'data.root'
    with uproot.recreate(source) as root:
        root.mkrntuple('EventNtuple/ntuple', columns)
    ref = tmp_path / 'reference.json'
    reference(ref, source, chunk=37)
    directory = tmp_path / 'capture'
    D.capture_data([source], directory, chunk=20, reference_accumulator=ref)
    table = np.zeros((2304, 9))
    table[:, 0], table[:, 2], table[:, 3] = np.arange(2304), 394.6, 11.4
    table[:, 6:9] = 1
    calib = tmp_path / 'calib.csv'
    np.savetxt(calib, table, delimiter=',')
    receipts = {}
    columns, summary, provenance = J.load_data(directory, receipts)
    asset = dict(path=str(calib), sha256=A.V.digest(calib))
    mapping = T.channel_map(calib, receipts, asset['sha256'])
    h, flow = T.prepare(columns, mapping)
    rows, pf = P.pair_rows(h)
    strata = {mode: P.cell_table(rows, mode) for mode in ('band', 'exact')}
    support = {}
    for mode, cells in strata.items():
        tables, common = P.support_tables({name: cells for name in (*J.ARMS, 'data')})
        support[mode] = {name: dict(all_pe=P.standardize(tables, common, b)) for b, name in enumerate(T.G.BODY_NAMES)}
    prior = dict(comparison='joint-pairs', passed=True,
                 samples={'data': dict(selected_flow=flow, pair_flow=pf, strata=strata)},
                 data={'events': summary['events'], 'provenance': provenance},
                 map={'asset': asset}, standardization=support, files=list(receipts.values()))
    path = tmp_path / 'previous.json'
    path.write_text(json.dumps(prior))
    return directory, path, source


def test_full_retained_shape_io_and_report(evidence, tmp_path, monkeypatch):
    data, prior, _ = evidence
    for name in ('accumulate_arrays', 'fit_arrays'):
        target = T.R if name == 'accumulate_arrays' else T.R.CL
        monkeypatch.setattr(target, name, lambda *a, **k: pytest.fail('must not re-fit/re-score'))
    result = A.run(data, [prior], tmp_path / 'result.json')
    assert result['completed'] and result['full_file'] and result['events'] == 37
    assert result['original_population_replay']['band']['full_census_and_moments_equal']
    assert result['original_population_replay']['exact']['max_absolute_moment_difference'] == 0
    assert result['shape_flow']['shape_invalid_pulses'] == 0
    assert sum(row['hits'] for row in result['audits']) == 36*48*2
    assert result['fixed_support'][str(prior)]['band']['T10']['shape_pair_coverage'] == 1
    Report.run(tmp_path / 'result.json', tmp_path / 'summary')
    assert 'T1 exact-N' in (tmp_path / 'summary/summary.md').read_text()
    loss = copy.deepcopy(result)
    row = loss['fixed_support'][str(prior)]['exact']['T10']
    row['pairs'] //= 2
    row['shape_pair_coverage'] = .5
    assert '50.00%' in Report.markdown(loss)
    row.update(supported=False, reason='shape-valid subset has empty frozen cells', empty_cells=1)
    assert '1 empty of' in Report.markdown(loss)
    with pytest.raises(ValueError, match='output already exists'):
        A.run(data, [prior], tmp_path / 'result.json')


def test_pilot_is_not_full_evidence_or_summary(evidence, tmp_path):
    data, prior, _ = evidence
    result = A.run(data, [prior], tmp_path / 'pilot.json', max_shards=1)
    assert not result['full_file'] and result['events'] == 20
    assert not result['original_population_replay']['band']['full_census_and_moments_equal']
    with pytest.raises(ValueError, match='full original data'):
        Report.markdown(result)


def test_changed_root_bytes_are_rejected(evidence, tmp_path):
    data, prior, source = evidence
    with source.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='ROOT file differs'):
        A.run(data, [prior], tmp_path / 'rejected.json')
    assert not (tmp_path / 'rejected.json').exists()


def test_changed_capture_metadata_is_rejected(evidence, tmp_path):
    data, prior, _ = evidence
    with (data/'summary.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='metadata hash mismatch'):
        A.run(data, [prior], tmp_path / 'rejected.json')


def test_changed_excluded_all_pair_moment_rejected(evidence, tmp_path):
    data, prior, _ = evidence
    value = json.loads(prior.read_text())
    value['samples']['data']['pair_flow']['T10']['all_second_moment'] += .1
    prior.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='historical all-pair moment'):
        A.run(data, [prior], tmp_path/'rejected.json')


def test_missing_capture_shard_refused(evidence, tmp_path):
    data, prior, _ = evidence
    metadata = json.loads((data/'summary.json').read_text())
    (data/metadata['raw_shards'][0]['path']).unlink()
    with pytest.raises(OSError):
        A.run(data, [prior], tmp_path / 'rejected.json')
