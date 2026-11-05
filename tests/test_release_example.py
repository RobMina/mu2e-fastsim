"""setup/RELEASE.example.json: the public placeholder manifest stays valid for mu2efastsim.inputs and names no site."""
from pathlib import Path

from mu2efastsim import inputs

EXAMPLE = Path(__file__).resolve().parents[1] / "setup" / "RELEASE.example.json"


def test_example_manifest_is_valid():
    obj = inputs.manifest(EXAMPLE)
    assert obj["files"] and all(not row["path"].startswith("/") for row in obj["files"])


def test_example_manifest_checks_as_missing(tmp_path):
    rows = inputs.check(inputs.manifest(EXAMPLE), tmp_path)
    assert {r["status"] for r in rows} == {"missing"}
