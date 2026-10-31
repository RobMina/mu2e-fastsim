"""Bounded edge-reference preparation and explicit mode signatures, no G4."""
import json
from pathlib import Path
import subprocess

import pytest

from beamfastsim import edgeg4, foilclosure


def test_default_control_text_preserves_inherited_lateral_setting():
    macro, config = foilclosure.control_text(10, 7)
    assert macro == '/control/verbose 2\n'
    assert config == '#include "base.fcl"\nsource.maxEvents: 10\nservices.SeedService.baseSeed: 7\nphysics.producers.g4run.g4Macro: "preinit.mac"\n'
    assert 'Lateral' not in config


def test_refined_controls_differ_only_in_explicit_lateral_flag(tmp_path):
    receipts = [edgeg4.prepare(tmp_path / mode, 'moved_edge', mode, 10, 9) for mode in edgeg4.MODES]
    a, b = [tmp_path / mode for mode in edgeg4.MODES]
    for name in ('base.fcl', 'foil_geom.txt', 'foil_gun.txt', 'preinit.mac'):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert (a / 'job.fcl').read_text().replace('true', 'false') == (b / 'job.fcl').read_text()
    assert receipts[0]['case']['lateral_displacement'] is True
    assert receipts[1]['case']['lateral_displacement'] is False
    assert receipts[0]['case']['bins'] == 168
    assert receipts[0]['case']['step_fraction'] == .005
    assert all(not r['complete'] for r in receipts)
    assert json.loads((a / 'prepared.json').read_text()) == receipts[0]


@pytest.mark.parametrize('value', [0, 1, 'false', [], .5])
def test_lateral_type_is_not_truthiness(value):
    with pytest.raises(ValueError, match='boolean'):
        foilclosure.control_text(10, 1, lateral_displacement=value)


def test_preparation_refuses_large_campaign_and_reuse(tmp_path):
    with pytest.raises(ValueError, match='bounded'):
        edgeg4.prepare(tmp_path / 'large', 'moved_edge', 'full_refined', 5001, 1)
    assert not (tmp_path / 'large').exists()
    edgeg4.prepare(tmp_path / 'one', 'moved_edge', 'full_refined', 10, 1)
    with pytest.raises(FileExistsError):
        edgeg4.prepare(tmp_path / 'one', 'moved_edge', 'full_refined', 10, 1)


def expanded(lateral='true'):
    return '''source: { maxEvents: 10 }
physics: { producers: { g4run: { physics: {
physicsListName: "ShieldingM" minRangeCut: 0.01 protonProductionCut: 0
setMuHadLateralDisplacement: ''' + lateral + ''' useDensityEffectInIonizationLossCalc: false
} } } }'''


def test_effective_signature_is_scoped_and_requires_matching_mode():
    case = dict(events=10, lateral_displacement=True)
    assert edgeg4.signature(expanded(), case)['setMuHadLateralDisplacement'] is True
    with pytest.raises(ValueError, match='signature'):
        edgeg4.signature(expanded('false'), case)
    with pytest.raises(ValueError, match='signature'):
        edgeg4.signature(expanded().replace('ShieldingM', 'ShieldingM_EMZ'), case)
    with pytest.raises(ValueError, match='count'):
        edgeg4.signature(expanded().replace('maxEvents: 10', 'maxEvents: 9'), case)


def test_input_identity_detects_tampering(tmp_path):
    receipt = edgeg4.prepare(tmp_path / 'one', 'oblique_edge', 'full_refined', 10, 1)
    item = next(x for x in receipt['prepared'] if x['path'].endswith('preinit.mac'))
    edgeg4.checked(item)
    Path(item['path']).write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        edgeg4.checked(item)


def test_uva_shell_syntax_and_no_build_commands():
    path = Path(edgeg4.__file__).resolve().parents[1] / 'art/edge_offline_uva.sh'
    subprocess.run(['bash', '-n', str(path)], check=True)
    text = path.read_text()
    assert 'mktemp -d /tmp/' in text and 'CVMFSEXEC_DIST' in text
    assert 'MDC2025ac' in text and 'exit_code.txt' in text
    assert 'muse build' not in text and 'muse tarball' not in text


def test_finalize_verbosity_first_command_not_echoed_and_inputs_immutable(tmp_path, monkeypatch):
    import numpy as np
    out = tmp_path / 'run'
    receipt = edgeg4.prepare(out, 'moved_edge', 'full_refined', 10, 7)
    texts = dict(
        exit_code='0\n', expanded=expanded(),
        runtime=foilclosure.RELEASE + '\nal9-prof-e29-p084\n',
        job='\n'.join((out / 'preinit.mac').read_text().splitlines()[1:]) +
            '\nTrigReport Events total = 10 passed = 10 failed = 0\nArt has completed and will exit with status 0.\n',
    )
    for name, text in texts.items():
        suffix = '.fcl' if name == 'expanded' else '.log' if name == 'job' else '.txt'
        (out / (name + suffix)).write_text(text)
    for name in ('runtime.sha256', 'launcher.txt', 'launcher.log', 'total.time', 'cache.txt', 'foil.root'):
        (out / name).write_text('fixture\n')
    (out / 'art.time').write_text('Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.45\n')
    (out / 'tables/couple.dat').write_text('fixture\n')
    monkeypatch.setattr(edgeg4.fr, 'audit', lambda _: (dict(other=0, primaries=10), dict(event=np.arange(1, 11)), {}))
    edgeg4.finalize(out)
    saved = json.loads((out / 'receipt.json').read_text())
    assert saved['complete'] and saved['wall_s'] == 20.45
    assert saved['sources'] == receipt['sources']
    assert saved['finalizer']['sha256'] == edgeg4._identity(edgeg4.__file__)['sha256']
    (out / 'preinit.mac').write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        edgeg4.finalize(out)


def test_collect_independent_modes_through_strict_loader(tmp_path, monkeypatch):
    from beamfastsim import slabrefs
    seed = 10
    for cell in edgeg4.CELLS:
        for mode in edgeg4.MODES:
            tag = cell + '_' + mode
            directory = tmp_path / tag
            directory.mkdir()
            seed += 1
            (directory / 'receipt.json').write_text(json.dumps(dict(case=dict(seed=seed))))
            (directory / 'scan.npz').write_text('fixture')
            run = dict(receipt=edgeg4._identity(directory / 'receipt.json'), scan_npz=edgeg4._identity(directory / 'scan.npz'))
            (tmp_path / (tag + '.manifest.json')).write_text(json.dumps(dict(complete=True, runs={tag: run})))
    calls = []
    def load(*args):
        calls.append(args)
        return dict.fromkeys(edgeg4.MODES, {}), []
    monkeypatch.setattr(slabrefs, 'load_references', load)
    edgeg4.collect(tmp_path)
    assert len(calls) == 2
    assert len(json.loads((tmp_path / 'manifest.json').read_text())['runs']) == 4
    with pytest.raises(FileExistsError):
        edgeg4.collect(tmp_path)
