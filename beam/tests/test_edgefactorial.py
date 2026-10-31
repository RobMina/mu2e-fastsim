"""Step/bin controls keep the original four-mode campaign and defaults intact."""
from pathlib import Path

import numpy as np
import pytest

from beamfastsim import edgefactorial as ef, edgeg4, foilclosure as fc


def test_export_default_is_byte_exact():
    macro, _ = fc.control_text(10, 1, export_tables=True)
    assert macro == ('/control/verbose 2\n/run/particle/setStoredInAscii 1\n'
                     '/process/eLoss/CSDARange true\n/process/em/binsPerDecade 168\n')
    assert edgeg4.MODES == {'full_refined': True, 'full_refined_no_lateral': False}


@pytest.mark.parametrize('mode', ef.MODES)
def test_factorial_macros_have_only_requested_bins_and_step(tmp_path, mode):
    receipt = ef.prepare(tmp_path / mode, 'moved_edge', mode, 10, 1)
    case = receipt['case']
    text = (tmp_path / mode / 'preinit.mac').read_text()
    assert ef.macro_signature(text, case)['bins_per_decade'] == case['bins']
    assert all(line.endswith(' ' + str(case['bins'])) for line in text.splitlines() if 'binsPerDecade' in line)
    assert ('StepFunctionMuHad' in text) == mode.startswith('refined')
    assert 'setMuHadLateralDisplacement: true' in (tmp_path / mode / 'job.fcl').read_text()
    with pytest.raises(ValueError, match='macro'):
        ef.macro_signature(text + '/process/em/binsPerDecade 168\n', case)


@pytest.mark.parametrize('value', [False, 0, -1, 7.5, '7'])
def test_invalid_export_bins(value):
    with pytest.raises(ValueError, match='export_bins'):
        fc.control_text(10, 1, export_tables=True, export_bins=value)


def native(tmp_path, monkeypatch, bins=7):
    grid = np.geomspace(.0001, 1.e8, bins * 12 + 1)
    tables = {name: dict(energy=grid, values=np.ones(len(grid))) for name in ef.pt.TABLES}
    couple = dict(density_g_cm3=2.699, range_cuts_mm={'gamma': .01, 'e-': .01, 'e+': .01, 'proton': 0.},
                  energy_cuts_MeV={'e-': .034105750000000004})
    path = tmp_path / 'table.asc'
    path.write_text('fixture')
    monkeypatch.setattr(ef.pt, 'load_target', lambda *a, **k: (tables, couple, [path]))
    return tables, couple


def test_native_grid_verifies_actual_nodes_not_macro_only(tmp_path, monkeypatch):
    native(tmp_path, monkeypatch)
    result = ef.native_signature(tmp_path, dict(bins=7))
    assert result['grids']['Range']['nodes'] == 85
    with pytest.raises(ValueError, match='binning'):
        ef.native_signature(tmp_path, dict(bins=168))


def test_native_cut_change_refused(tmp_path, monkeypatch):
    _, couple = native(tmp_path, monkeypatch)
    couple['energy_cuts_MeV']['e-'] = 1.
    with pytest.raises(ValueError, match='cuts'):
        ef.native_signature(tmp_path, dict(bins=7))


def test_invalid_mode_does_not_create_output(tmp_path):
    with pytest.raises(KeyError):
        ef.prepare(tmp_path / 'bad', 'moved_edge', 'unknown', 10, 1)
    assert not (tmp_path / 'bad').exists()


def test_large_or_reused_factorial_run_refused(tmp_path):
    with pytest.raises(ValueError, match='bounded'):
        ef.prepare(tmp_path / 'big', 'moved_edge', 'stock_step_bins7', 5001, 1)
    ef.prepare(tmp_path / 'one', 'moved_edge', 'stock_step_bins7', 10, 1)
    with pytest.raises(FileExistsError):
        ef.prepare(tmp_path / 'one', 'moved_edge', 'stock_step_bins7', 10, 1)


def test_seeds_independent_of_old_references():
    seeds = [seed for values in ef.SEEDS.values() for seed in values]
    assert len(set(seeds)) == 6
    assert not set(seeds) & {2026103011, 2026103012, 2026103021, 2026103022}


def test_inherited_final_range_is_not_claimed_as_effective_override(tmp_path):
    receipt = ef.prepare(tmp_path / 'one', 'oblique_edge', 'stock_step_bins7', 10, 7)
    signature = ef.macro_signature((tmp_path / 'one/preinit.mac').read_text(), receipt['case'])
    assert signature['step_override'] is False
    assert signature['step_scope'] == 'inherited released runtime'


def test_no_new_four_control_collection_modes():
    assert len(edgeg4.MODES) == 2
    assert len(ef.MODES) == 3


def test_collect_checks_independence_and_uses_unchanged_loader(tmp_path, monkeypatch):
    import json
    seed = 100
    for cell in edgeg4.CELLS:
        for mode in ef.MODES:
            seed += 1
            tag = cell + '_' + mode
            out = tmp_path / tag
            out.mkdir()
            fraction, final_range, bins = ef.MODES[mode]
            case = dict(seed=seed, events=5000, step_fraction=fraction, final_range=final_range,
                        bins=bins, lateral_displacement=True)
            (out / 'receipt.json').write_text(json.dumps(dict(case=case)))
            (out / 'scan.npz').write_text('fixture')
            macro, _ = fc.control_text(5000, seed, export_tables=True, step_fraction=fraction,
                final_range=final_range, bins=bins, export_bins=bins, lateral_displacement=True)
            (out / 'preinit.mac').write_text(macro)
            run = dict(receipt=edgeg4._identity(out / 'receipt.json'), scan_npz=edgeg4._identity(out / 'scan.npz'))
            (tmp_path / (tag + '.manifest.json')).write_text(json.dumps(dict(complete=True, runs={tag: run})))
            (out / 'factorial_audit.json').write_text(json.dumps(dict(complete=True,
                receipt=run['receipt'], sources=[], native=dict(inputs=[]))))
    monkeypatch.setattr(ef, 'native_signature', lambda *a: {})
    calls = []
    def load(*args):
        calls.append(args)
        return dict.fromkeys(ef.MODES, {}), []
    monkeypatch.setattr(ef.slabrefs, 'load_references', load)
    ef.collect(tmp_path)
    assert len(calls) == 2
    assert len(json.loads((tmp_path / 'manifest.json').read_text())['runs']) == 6
    with pytest.raises(FileExistsError):
        ef.collect(tmp_path)


def test_finalize_refuses_prepared_source_changes_before_root_audit(tmp_path):
    out = tmp_path / 'one'
    receipt = ef.prepare(out, 'moved_edge', 'stock_step_bins7', 10, 1)
    (out / 'preinit.mac').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        ef.finalize(out)
    assert not (out / 'receipt.json').exists()
