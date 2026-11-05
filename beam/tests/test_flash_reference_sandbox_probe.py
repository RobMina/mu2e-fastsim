"""Safe probe entry refusals; successful isolation needs an actual Linux child."""
from pathlib import Path

import pytest

from beamfastsim.flash_reference_sandbox_probe import main, probe


def test_cli_requires_explicit_persistent_target():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_probe_contract_explicit_no_general_certificate():
    source = Path(__file__).parents[1] / "beamfastsim/flash_reference_sandbox_probe.py"
    text = source.read_text()
    assert "not exhaustive confinement" in text
    assert 'path.open("xb")' in text
    assert 'thread.join()' in text
    assert 'errno.EPERM' in text
