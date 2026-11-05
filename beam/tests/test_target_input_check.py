"""Staged payload validation fails before admitting truncated or corrupted inputs."""
import hashlib
import json
import zlib

import pytest

from beamfastsim.target_input_check import main, verify


def test_integrity_and_admission(tmp_path):
    data = b'existing-reference-payload' * 100
    part, final, report = tmp_path / 'input.part', tmp_path / 'input.art', tmp_path / 'receipt.json'
    part.write_bytes(data)
    adler = f'{zlib.adler32(data):08x}'
    assert verify(part, len(data), adler)['sha256'] == hashlib.sha256(data).hexdigest()
    main([str(part), '--size', str(len(data)), '--adler32', adler, '--admit-as', str(final), '--out', str(report)])
    assert final.read_bytes() == data and not part.exists()
    assert json.loads(report.read_text())['bytes'] == len(data)
    with pytest.raises(ValueError, match='mismatch'):
        verify(final, len(data) + 1, adler)
    with pytest.raises(ValueError, match='mismatch'):
        verify(final, len(data), '00000001')
