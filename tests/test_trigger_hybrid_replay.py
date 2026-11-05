"""Prepare measured fast/stock migrations only from hash-bound admitted inputs."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import hybrid_replay
import integrity
import replay
from test_trigger_replay import fixture, replay as fixture_replay


def test_hybrid_preserves_all_paths_and_retags_direct_consumers(tmp_path):
    import hashlib
    config = fixture()
    text = 'Digitize: ' + fixture_replay.render(config)
    paths = [tmp_path / 'reference.art', tmp_path / 'candidate.art']
    for path in paths:
        path.write_bytes(b'bound-file')
    admission = {'events': 2, 'same_event_handoff_metadata_passed': True,
        'reference_files': integrity.files(paths[:1]), 'candidate_files': integrity.files(paths[1:]),
        'config_sha256': hashlib.sha256(text.encode()).hexdigest(), 'straw_digi_tag': 'Fast::Injected', 'calo_digi_tag': None}
    fcl, manifest = hybrid_replay.prepare(text, paths[:1], paths[1:], admission, 2)
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'replay.fcl').write_text(fcl)
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    hybrid_replay.preflight(prepared, text)
    result = replay.parse_expanded(fcl)
    assert manifest['full_production_history_certified'] is False
    assert 'stock_output_selection' not in manifest
    assert len(replay.paths(result)) == len(manifest['paths'])
    assert all(binding['replacement'] == 'Fast::Injected' for binding in manifest['digi_substitutions'])
    history = {'equal': True, 'all_reference_events_present': True, 'regenerated_events': 2,
        'config_sha256': admission['config_sha256'], 'candidate_files': admission['reference_files']}
    with pytest.raises(ValueError, match='complete history'):
        hybrid_replay.prepare(text, paths[:1], paths[1:], admission, 2, history)
    paths[1].write_bytes(b'changed')
    with pytest.raises(ValueError, match='bytes'):
        hybrid_replay.preflight(prepared, text)


def test_no_unadmitted_fast_source(tmp_path):
    with pytest.raises(ValueError, match='admission'):
        hybrid_replay.prepare('', [], [], {}, 2)
