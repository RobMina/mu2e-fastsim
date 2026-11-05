"""Embedded schedule resolution refuses missing IDs and reordered or changed menu bindings."""
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import schedule_registry


ID = 'a' * 40
PATHS = [(None, 'DigitizePath', ['Digi']), ('10', 'p', ['Digi', 'Cut'])]


def database(blob='trigger_paths:["0:DigitizePath","10:p"]', suffix='\x00'):
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE ParameterSets(ID PRIMARY KEY, PSetBlob)')
        db.execute('INSERT INTO ParameterSets VALUES (?,?)', (ID + suffix, blob))
        return db.serialize()


@pytest.mark.parametrize('suffix', ['', '\x00'])
def test_schedule_id_resolves_to_exact_ordered_paths(suffix):
    report = schedule_registry.resolve(database(suffix=suffix), ID, PATHS)
    assert report['registry_schedule_verified'] and report['paths'] == 2
    assert len(report['registry_blob_sha256']) == 64


@pytest.mark.parametrize('blob', ['trigger_paths:["10:p","0:DigitizePath"]',
                                 'trigger_paths:["0:DigitizePath","11:p"]',
                                 'trigger_paths:["0:DigitizePath","10:q"]',
                                 'trigger_paths:["0:DigitizePath","10:p"] other:1'])
def test_schedule_reordering_or_different_binding_fails(blob):
    with pytest.raises(ValueError, match='differs'):
        schedule_registry.resolve(database(blob), ID, PATHS)


def test_single_terminator_is_supported_but_embedded_or_repeated_nul_fails():
    blob = 'trigger_paths:["0:DigitizePath","10:p"]'
    assert schedule_registry.resolve(database(blob + '\x00'), ID, PATHS)['registry_schedule_verified']
    for text in [blob + '\x00\x00', 'trigger_paths:\x00[]']:
        with pytest.raises(ValueError, match='string framing'):
            schedule_registry.resolve(database(text), ID, PATHS)


def test_registry_requires_exact_id_and_supported_payload():
    with pytest.raises(ValueError, match='missing'):
        schedule_registry.resolve(database(), 'b' * 40, PATHS)
    with pytest.raises(ValueError, match='payload'):
        schedule_registry.resolve(b'not a database', ID, PATHS)
    with pytest.raises(ValueError, match='parameter-set ID'):
        schedule_registry.resolve(database(), 'bad', PATHS)


def test_ambiguous_terminated_and_unterminated_ids_fail():
    with sqlite3.connect(':memory:') as db:
        db.deserialize(database())
        db.execute('INSERT INTO ParameterSets VALUES (?,?)', (ID, 'trigger_paths:[]'))
        db.commit()
        payload = db.serialize()
    with pytest.raises(ValueError, match='ambiguous'):
        schedule_registry.resolve(payload, ID, PATHS)
