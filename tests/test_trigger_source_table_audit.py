"""Source-table coverage and represented straw geometry diagnostics stay separate from physics closure."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import source_table_audit


def test_source_births_and_straw_geometry_are_independently_reconciled(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ('source.art', 'table.root', 'geometry.gdml')]
    with uproot.recreate(paths[0]) as root:
        root.mktree('Events', {'event': 'uint32'})
    paths[1].write_bytes(b'table fixture')
    paths[2].write_bytes(b'geometry fixture')
    monkeypatch.setattr(source_table_audit.event_keys, 'read', lambda tree: np.array([[1, 1, 1]], dtype=np.uint64))
    geometry = SimpleNamespace(NSTRAWS=96, from_gdml=lambda path: SimpleNamespace(nstraws=1, sid=np.array([0]),
        mid=np.zeros((1, 3)), dir=np.array([[0., 0., 1.]]), radius=2.5, halflen=np.array([100.])), to_mu2e=lambda x, origin: x + origin)
    tracker = ModuleType('trkfastsim')
    tracker.geometry = geometry
    transport = ModuleType('trkfastsim.validation.fieldce')
    births = pd.DataFrame({'run': [1], 'subrun': [1], 'event': [1]})
    transport.starts = lambda tables: births
    monkeypatch.setitem(sys.modules, 'trkfastsim', tracker)
    monkeypatch.setitem(sys.modules, 'trkfastsim.validation.fieldce', transport)
    from mu2efastsim import tables
    steps = pd.DataFrame({'upanel': [0], 'straw': [0], 'sid': [0], 'x0': [2.5], 'y0': [0.], 'z0': [0.],
        'x1': [-2.5], 'y1': [0.], 'z1': [0.]})
    monkeypatch.setattr(tables, 'read_many', lambda *args: steps)
    report = source_table_audit.audit(paths[:1], paths[1:2], paths[2], 1, [0, 0, 0])
    assert report['source_birth_identities_verified'] and report['gdml_straw_ids_verified']
    assert report['endpoints']['0']['outside_radius_plus_1um'] == 0
    births['event'] = 2
    with pytest.raises(ValueError, match='cover every unique'):
        source_table_audit.audit(paths[:1], paths[1:2], paths[2], 1, [0, 0, 0])
