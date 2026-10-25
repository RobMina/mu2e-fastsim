"""Persisted-entry checks never certify a partial tree cycle or a wrong event population."""
import json
import numpy as np
import pytest
import uproot
from valhist.pilot import tree_inventory, verify


def write_events(f, name, events):
    a = np.asarray(events, dtype=np.int32)
    f.mktree(name, {'run': 'int32', 'subrun': 'int32', 'event': 'int32'})
    f[name].extend(dict(run=np.full(len(a), 1400, dtype=np.int32), subrun=np.ones(len(a), dtype=np.int32), event=a))


def test_inventory_highest_cycle_and_duplicates(tmp_path):
    p = tmp_path / 'a.root'
    with uproot.recreate(p) as f:
        write_events(f, 'dump/gen', [1])
    with uproot.update(p) as f:
        write_events(f, 'dump/gen', [1, 2, 2])
    inv = tree_inventory(p, 'dump/gen')
    assert inv['cycles'] == [1, 2] and inv['selected_cycle'] == 2
    assert inv['rows'] == 3 and inv['unique_events'] == 2
    with pytest.raises(ValueError, match='missing'):
        tree_inventory(p, 'missing')


@pytest.mark.parametrize('candidate,rc,completed', [([1, 2], '0', True), ([1], '0', False),
                                                  ([1, 3], '0', False), ([1, 2], '1', False)])
def test_verify_keys_entries_and_exit(tmp_path, candidate, rc, completed):
    gen = tmp_path / 'gen.root'
    with uproot.recreate(gen) as f:
        write_events(f, 'dump/gen', [1, 2])
    manifest = dict(generated=tree_inventory(gen, 'dump/gen'), files={})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'logs').mkdir()
    for arm in ('a1', 'e2e'):
        (tmp_path / arm).mkdir()
        with uproot.recreate(tmp_path / arm / 'valhist.root') as f:
            for name in ('events', 'digis', 'pulses', 'clusters'):
                write_events(f, 'crvdump/' + name, candidate)
        (tmp_path / 'logs' / f'{arm}.log').write_text('Art has completed and will exit with status 0.')
        (tmp_path / 'logs' / f'{arm}.exit').write_text(rc)
    if completed:
        verify(tmp_path)
    else:
        with pytest.raises(SystemExit):
            verify(tmp_path)
    assert json.loads((tmp_path / 'completion.json').read_text())['completed'] == completed


def test_prepare_records_assets_and_refuses_reuse(tmp_path, monkeypatch):
    from valhist import pilot
    from crvfastsim import geometry
    from crvfastsim.response import lookup
    area, core, data = tmp_path / 'area', tmp_path / 'core', tmp_path / 'data'
    cat = area / 'backing/Offline/CRVDQM/data/CRVDQM_binning_v3_extracted.txt'
    cat.parent.mkdir(parents=True)
    cat.write_text('catalogue')
    core.mkdir()
    gen, lib = tmp_path / 'fastC.root', tmp_path / 'lib.root'
    with uproot.recreate(gen) as f:
        write_events(f, 'dump/gen', [1, 2])
    lib.write_bytes(b'library')
    (data / 'geometry').mkdir(parents=True)
    for name in ('crvcalib_sim_best_v1_5_run1400.csv', 'crvphoton_sim_best_v1_5_run1400.csv'):
        (data / 'geometry' / name).write_text('conditions')
    (data / 'lookup').mkdir()
    (data / 'lookup/LUT').write_text('lookup')
    template = data / 'template.txt'
    template.write_text('template')
    monkeypatch.setattr(geometry, 'find_counters', lambda p: str(template))
    monkeypatch.setattr(lookup, 'lookup_dir', lambda: str(data / 'lookup'))
    monkeypatch.setattr(pilot, 'revision', lambda p: dict(head='synthetic', status=''))
    out = data / 'valhist/newpilot'
    pilot.prepare(out, gen, lib, area, core, '/pnfs/known.art', 11)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['generated']['unique_events'] == 2
    assert manifest['files'][str(template)] == pilot.digest(template)
    assert not manifest['asset_paths_missing']
    assert 'mc : false' in (out / 'a1.fcl').read_text()
    with pytest.raises(FileExistsError):
        pilot.prepare(out, gen, lib, area, core, '/pnfs/known.art', 11)
