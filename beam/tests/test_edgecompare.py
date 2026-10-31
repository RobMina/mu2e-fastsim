"""Saved-reference comparison guards and bookkeeping without G4 or sampler calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim import edgecompare as ec


def setup_pipeline(tmp_path, monkeypatch, *, equal=True):
    root = tmp_path / 'refs'
    root.mkdir()
    for cell in ec.edgeg4.CELLS:
        for mode in ec.edgeg4.MODES:
            d = root / (cell + '_' + mode)
            d.mkdir()
            receipt = dict(complete=True, sources=[], artifacts=[], tables=[], output={}, case={})
            (d / 'receipt.json').write_text(json.dumps(receipt))
            (d / 'expanded.fcl').write_text('fixture')
    monkeypatch.setattr(ec.edgeg4, 'checked', lambda item: item)
    monkeypatch.setattr(ec.edgeg4, 'signature', lambda *args: {})
    monkeypatch.setattr(ec.g4tables, 'compare', lambda *args, **kwargs:
                        dict(all_arrays_equal=equal, material_equal=True))
    monkeypatch.setattr(ec.physics_tables, 'load_target', lambda *args, **kwargs: ({}, {}, []))
    ref = dict(event=np.arange(1, 5001), track=np.ones(5000), exit_MeV=np.tile([0., .2, .3, .4], 1250),
               exit=np.tile([False, True, True, True], 1250))
    monkeypatch.setattr(ec.foilreport, 'audit', lambda _: (dict(other=0, primaries=5000), ref, {}))
    monkeypatch.setattr(ec.foilreport, 'highest', lambda *args, **kwargs: (None, {}))
    class File:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(ec.uproot, 'open', lambda *args, **kwargs: File())
    direction = np.tile([0., 0., 1.], (5000, 1))
    monkeypatch.setattr(ec.edgeangles, 'first_exit', lambda *args, **kwargs:
                        dict(direction=direction, event=ref['event']))
    historical = dict(inputs=[], references={})
    for cell in ec.edgeg4.CELLS:
        path = tmp_path / (cell + '_stock.npz')
        np.savez(path, exit_MeV=ref['exit_MeV'])
        historical['references'][cell + '_full'] = dict(identity=dict(path=str(path)))
    monkeypatch.setattr(ec.ld, 'analyze', lambda *args: historical)
    report = dict(args=dict(cell='moved_edge', seed=9), reference_configuration_qualification='UNVERIFIED',
                  arms=dict(diffuse=dict(output={})))
    arrays = dict(raw_energy=ref['exit_MeV'], status=np.full(5000, ec.sd.TRANSMITTED), direction=direction)
    monkeypatch.setattr(ec.le, 'read_study', lambda _: (report, dict(diffuse=arrays), {}))
    monkeypatch.setattr(ec.le, 'threshold_array', lambda *args: ref['exit_MeV'])
    return root


def test_saved_reference_pipeline_retains_qualification_and_independent_comparisons(tmp_path, monkeypatch):
    root = setup_pipeline(tmp_path, monkeypatch)
    out = tmp_path / 'new'
    result = ec.run(root, ['saved_fast'], 'recovered', tmp_path / 'package.npz', out)
    assert result['complete'] and not result['physics_closure']
    assert len(result['g4']) == 4 and len(result['g4_controls']) == 4
    assert len(result['fast']) == 1
    assert 'UNVERIFIED' in result['historical_configuration']
    assert result['fast']['saved_fast_diffuse']['references']['full_refined']['distribution']['variance']['state'] == 'INDETERMINATE'
    assert len(result['outputs']) == 4
    assert json.loads((out / 'summary.json').read_text()) == result
    with pytest.raises(FileExistsError):
        ec.run(root, [], 'recovered', tmp_path / 'package.npz', out)


def test_new_table_difference_is_not_silently_compared_to_old_fast(tmp_path, monkeypatch):
    root = setup_pipeline(tmp_path, monkeypatch, equal=False)
    with pytest.raises(ValueError, match='tables differ'):
        ec.run(root, [], 'recovered', tmp_path / 'package.npz', tmp_path / 'new')
