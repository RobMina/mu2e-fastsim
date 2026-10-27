"""Campaign planning never silently reruns a retained library cell."""
import json
from pathlib import Path

import pytest

from valhist import library_campaign as campaign
from valhist.library_variants import _sealed_json


def source(tmp_path):
    path = tmp_path / 'valhist/thread_b_library_plan_20261025'
    path.mkdir(parents=True)
    _sealed_json(path / 'plan.json', {'variants': {'bootstrap_0001': {}, 'bootstrap_0002': {}}})
    return tmp_path


def test_plan_independent_seeds(tmp_path):
    data = source(tmp_path)
    out = tmp_path / 'campaign.json'
    result = campaign.plan(data, out, [1, 2], [11, 12, 20], 'new')
    assert len(result['cells']) == 12
    assert len({c['seed'] for c in result['cells']}) == 12
    assert not any(c['reused'] for c in result['cells'])
    assert json.loads(out.read_text()) == result
    with pytest.raises(FileExistsError):
        campaign.plan(data, out, [1, 2], [11, 12], 'new')


def test_reuse_old_only_when_certified(tmp_path, monkeypatch):
    data = source(tmp_path)
    old = data / 'valhist/thread_b_library_bootstrap_20261025_bootstrap_0001_seed200101_s11'
    old.mkdir()
    monkeypatch.setattr(campaign, 'certify', lambda path: {'manifest': {'seed': 200101}})
    result = campaign.plan(data, tmp_path / 'campaign.json', [1], [11, 12], 'new')
    reused = [c for c in result['cells'] if c['reused']]
    assert len(reused) == 1 and reused[0]['directory'] == str(old)


@pytest.mark.parametrize('indices,shards', [([1, 1], [11, 12]), ([1], [1, 12]), ([1], [11]), ([3], [11, 12])])
def test_invalid_plan(tmp_path, indices, shards):
    source(tmp_path)
    with pytest.raises(ValueError):
        campaign.plan(tmp_path, tmp_path / 'new.json', indices, shards, 'new')
    assert not (tmp_path / 'new.json').exists()


def test_run_uses_frozen_working_directory(tmp_path, monkeypatch):
    source(tmp_path)
    manifest = tmp_path / 'campaign.json'
    record = campaign.plan(tmp_path, manifest, [1], [11, 12], 'new')
    metadata = Path(record['cells'][0]['library_metadata'])
    metadata.parent.mkdir()
    root = metadata.parent / 'library.root'
    root.write_bytes(b'library')
    _sealed_json(metadata, {'provenance': {'plan_sha256': record['donor_plan_sha256'],
        'variant': 'bootstrap_0001', 'library_sha256': campaign.sha256(root)}})
    calls = []
    monkeypatch.setattr(campaign.subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(campaign, 'certify', lambda path: {})
    campaign.run(manifest, [1])
    assert len(calls) == 4
    assert all(kwargs['cwd'] == record['frozen_physics'] for _, kwargs in calls)
    assert all(args[0] == 'bash' for args, _ in calls)


def test_run_refuses_existing_new_cell(tmp_path, monkeypatch):
    source(tmp_path)
    manifest = tmp_path / 'campaign.json'
    result = campaign.plan(tmp_path, manifest, [1], [11, 12], 'new')
    Path(result['cells'][0]['directory']).mkdir()
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(FileExistsError):
        campaign.run(manifest, [1])
