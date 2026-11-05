"""Pure ELF-report parsing and environment guards; never compile or load plugins."""
from pathlib import Path
import importlib.util

import pytest

HERE = Path(__file__).resolve().parents[1] / "recording/current_head"
spec = importlib.util.spec_from_file_location("recording_build_receipt", HERE / "verify_build.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_relocation_parser_retains_missing_and_undefined():
    result = verifier.relocation("""libavailable.so => /fixture/libavailable.so (0x1234)
libabsent.so => not found
undefined symbol: cblas_daxpy (/fixture/libgsl.so)
undefined symbol: cblas_daxpy (/fixture/libgsl.so)
undefined symbol: other_symbol (/fixture/plugin.so)
""")
    assert result["resolved"] == {"libavailable.so": "/fixture/libavailable.so"}
    assert result["missing"] == ["libabsent.so => not found"]
    assert result["undefined"] == ["cblas_daxpy", "other_symbol"]


def test_relocation_parser_clean_is_not_inferred_from_exit_status():
    assert verifier.relocation("") == {"missing": [], "undefined": [], "resolved": {}}
    assert verifier.relocation("undefined symbol: unresolved (/fixture/lib.so)\n")["undefined"] == ["unresolved"]


def test_build_verification_rejects_wrong_workdir_before_reading_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_WORK_DIR", str(tmp_path / "wrong"))
    monkeypatch.setenv("MUSE_STUB", verifier.STUB)
    with pytest.raises(ValueError, match="wrong configured"):
        verifier.verify(tmp_path)


def test_build_receipt_preserves_native_validation_limitations():
    text = (HERE / "verify_build.py").read_text()
    assert '"native_execution_performed": False' in text
    assert '"physics_validation_performed": False' in text
    assert 'stock["undefined"] != linkage["undefined"]' in text
    for forbidden in ("muse build", "muse setup", "muse tarball", "cmake --build"):
        assert forbidden not in text
