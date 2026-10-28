"""Read-only follow-up report guards and seed-collision semantics."""
import json

import pytest

from valhist import generator_capture_verify as V, generator_seed_report as R


def seed_log():
    return ('Summary of seeds computed by the SeedService.\nBase Seed: 20\n'
            'Seed Value     ModuleLabel.InstanceName\n20 cosmicTimeOffset\n21 g4run\n22 generate\n'
            '=============================================\n'
            'TrigReport Events total = 0 passed = 0 failed = 0\n'
            'Art has completed and will exit with status 0.\n')


def test_seed_assignment_and_same_engine_collisions():
    base, seeds = R.assignment(seed_log())
    assert base == 20 and seeds['generate'] == 22
    first = dict(mode='crygen', subrun=2001, seeds=seeds)
    second = dict(mode='crygen', subrun=2003, seeds={k: v + 2 for k, v in seeds.items()})
    assert R.overlaps([first, second]) == [dict(first=['crygen', 2001, 'generate'],
        second=['crygen', 2003, 'cosmicTimeOffset'], seed=22)]
    second['seeds'] = dict(generate=50, cosmicTimeOffset=51, g4run=22)
    assert R.overlaps([first, second]) == []


@pytest.mark.parametrize('change', ['heading', 'base', 'module'])
def test_seed_assignment_rejects_missing_evidence(change):
    text = seed_log()
    if change == 'heading':
        text = text.replace('Seed Value', 'Other Value')
    elif change == 'base':
        text += '\nBase Seed: 21\n'
    else:
        text = text.replace('22 generate', '22 other')
    with pytest.raises(ValueError):
        R.assignment(text)


@pytest.mark.parametrize('case', ['promoted', 'exit', 'drift'])
def test_capture_recheck_fails_before_publication(tmp_path, case):
    cert = dict(downstream_authorized=case == 'promoted', art_run=dict(exit=1 if case == 'exit' else 0),
                input_sha256={}, input_sha256_post={'wrong': 'value'} if case == 'drift' else {},
                code_sha256={}, code_sha256_post={})
    path = tmp_path / 'certificate.json'
    path.write_text(json.dumps(cert))
    out = tmp_path / 'result.json'
    with pytest.raises(ValueError):
        V.verify([path], out)
    assert not out.exists()


def test_existing_report_is_never_replaced(tmp_path):
    out = tmp_path / 'result.json'
    out.write_text('protected')
    with pytest.raises(FileExistsError):
        V.verify([], out)
    with pytest.raises(FileExistsError):
        R.run([], tmp_path / 'absent', out)
    assert out.read_text() == 'protected'
