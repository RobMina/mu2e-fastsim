"""Read-only parent/random-state inventory contract."""
import importlib.util
from pathlib import Path
import sys

import pytest

TRIGGER = Path(__file__).resolve().parents[1] / 'trigger'
sys.path.insert(0, str(TRIGGER))
spec = importlib.util.spec_from_file_location('trigger_source_audit', TRIGGER / 'source_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def config(files='"root://door//store/dts.test.a.art"'):
    return f'''Digitize: {{ source: {{ module_type: RootInput fileNames: [ {files} ] }}
      physics: {{ DigitizePath: [ Digi ] producers: {{ Digi: {{ module_type: TestDigi }} }} }}
      services: {{ SeedService: {{ policy: autoIncrement baseSeed: 9 }}
        RandomNumberGenerator: {{ defaultEngineKind: MixMaxRng }} }}
      outputs: {{ Output: {{ SelectEvents: [ Physics ] }} }} }}'''


def test_audit_does_not_pretend_configured_order_is_runtime_order():
    report = audit.audit(config())
    assert report['parents'][0]['did'] == 'mu2e:dts.test.a.art'
    assert report['seed_service']['baseSeed'] == '9'
    assert report['configured_parent_order_not_certified_runtime_order']
    assert not report['full_regeneration_prepared']
    assert report['digitize_modules'] == [{'label': 'Digi', 'module_type': 'TestDigi'}]


@pytest.mark.parametrize('files', ['', '42', '"bad"', '"x.art" "x.art"'])
def test_bad_or_duplicate_sources_fail(files):
    with pytest.raises(ValueError):
        audit.audit(config(files))


def test_missing_random_services_fail():
    with pytest.raises(ValueError, match='random'):
        audit.audit(config().replace('RandomNumberGenerator:', 'Other:'))


def test_catalog_only_gets_explicit_parents():
    report = audit.audit(config())
    class Client:
        def get_file(self, *, did, with_metadata):
            assert did == report['parents'][0]['did'] and with_metadata
            return {'name': 'dts.test.a.art', 'size': 12, 'retired': False}
    result = audit.catalog_parents(report, Client())
    assert result['total_bytes'] == 12 and result['catalog_stored_parent_events'] is None


@pytest.mark.parametrize('record', [None, {'name': 'dts.test.a.art', 'size': 12, 'retired': True},
    {'name': 'wrong', 'size': 12}, {'name': 'dts.test.a.art', 'size': 0}])
def test_catalog_missing_retired_mismatched_and_bad_sizes_fail(record):
    class Client:
        def get_file(self, **kwargs):
            return record
    with pytest.raises(ValueError):
        audit.catalog_parents(audit.audit(config()), Client())
