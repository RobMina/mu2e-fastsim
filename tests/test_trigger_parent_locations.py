"""Resolved storage paths preserve exact parent history and do not read data or write the catalogue."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import parent_locations
import stage_parents


def report():
    return {'parents': [{'did': 'mu2e:dts.test.a.art', 'url': '/srv/indir/dts.test.a.art'},
                        {'did': 'mu2e:dts.test.b.art', 'url': '/srv/indir/dts.test.b.art'}],
            'catalog': {'files': [{'name': f'dts.test.{name}.art', 'size': 12,
                                   'checksums': {'adler32': '00000001'}} for name in ['a', 'b']]}}


def test_read_only_locations_keyed_to_exact_identity_not_return_order():
    original = report()
    saved = deepcopy(original)
    def run(command, **kwargs):
        assert command == ['mdh', 'print-url', '-s', 'path', '-l', 'tape', '-']
        assert kwargs['input'] == 'mu2e:dts.test.a.art\nmu2e:dts.test.b.art\n'
        assert kwargs['check'] and kwargs['timeout'] == 600
        return SimpleNamespace(stdout='/store/dts.test.b.art\n/store/dts.test.a.art\n')
    result = parent_locations.resolve(original, runner=run)
    assert original == saved
    assert result['parents'] == saved['parents']
    assert stage_parents.records(result)[0][1] == '/store/dts.test.a.art'
    assert stage_parents.records(result)[1][1] == '/store/dts.test.b.art'
    result['resolved_source_paths'].reverse()
    with pytest.raises(ValueError, match='identities/order'):
        stage_parents.records(result)


@pytest.mark.parametrize('output', ['', '/store/dts.test.a.art\n',
    '/store/dts.test.a.art\n/store/dts.test.a.art\n', '/store/dts.test.a.art\n/store/unknown.art\n',
    'relative/dts.test.a.art\n/store/dts.test.b.art\n', '/store/../dts.test.a.art\n/store/dts.test.b.art\n',
    '/store/dts.test.a.art\nwarning /store/dts.test.b.art\n'])
def test_missing_ambiguous_or_unsafe_paths_fail(output):
    with pytest.raises(ValueError):
        parent_locations.resolve(report(), runner=lambda *args, **kwargs: SimpleNamespace(stdout=output))


def test_failed_query_propagates_instead_of_guessing():
    import subprocess
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    with pytest.raises(subprocess.CalledProcessError):
        parent_locations.resolve(report(), runner=fail)


def test_duplicate_parent_input_fails_before_query():
    value = report()
    value['parents'][1] = value['parents'][0]
    with pytest.raises(ValueError, match='duplicate'):
        parent_locations.resolve(value, runner=lambda *args, **kwargs: pytest.fail('query not allowed'))
