"""A partial transfer can be measured without relabelling it a complete source receipt."""
import json
from pathlib import Path
import sys
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import catalog_bytes


def test_catalog_prefix_verification_keeps_source_receipt_false(tmp_path):
    payload = b'complete staged parent'
    (tmp_path / 'dts.test.a.art').write_bytes(payload)
    report = {'parents': [{'did': 'mu2e:dts.test.a.art', 'url': '/input/dts.test.a.art'},
                          {'did': 'mu2e:dts.test.b.art', 'url': '/input/dts.test.b.art'}],
              'catalog': {'files': [{'name': f'dts.test.{name}.art', 'size': len(payload),
                 'checksums': {'adler32': f'{zlib.adler32(payload):08x}'}} for name in ['a', 'b']]}}
    result = catalog_bytes.verify(report, tmp_path, 1)
    assert result['catalog_byte_checks_passed'] and not result['source_sha256_receipt_verified']
    assert result['verified_parent_prefix'] == 1 and result['total_original_parents'] == 2
    assert len(result['files'][0]['sha256']) == 64
    for count in [0, -1, 3]:
        with pytest.raises(ValueError, match='bounded'):
            catalog_bytes.verify(report, tmp_path, count)
    (tmp_path / 'dts.test.a.art').write_bytes(b'bad')
    with pytest.raises(ValueError, match='adler32'):
        catalog_bytes.verify(report, tmp_path, 1)
