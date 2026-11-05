"""Synthetic persisted-event cycle checks, not generated exposure."""
import json

import numpy as np
import pytest
import uproot

from beamfastsim import tsentries


def make(path, n, update=False):
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        root.mktree("Events", {"number": "int64"}).extend({"number": np.arange(n)})


def test_highest_cycle_counts_and_cli(tmp_path):
    path = tmp_path / "input.art"
    make(path, 10)
    make(path, 12, update=True)
    result = tsentries.audit([path])
    assert result["saved_events"] == 12
    assert result["files"][0]["cycles"] == {"1": 10, "2": 12}
    assert result["files"][0]["selected_key"] == "Events;2"
    assert result["normalization_unverified"]
    out = tmp_path / "counts.json"
    tsentries.main([str(path), "--out", str(out)])
    assert json.loads(out.read_text())["saved_events"] == 12
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        tsentries.main([str(path), "--out", str(path)])
    assert path.read_bytes() == before


def test_empty_and_duplicate_inputs(tmp_path):
    path = tmp_path / "input.art"
    make(path, 0)
    assert tsentries.audit([path])["saved_events"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        tsentries.audit([path, path])
    with pytest.raises(ValueError, match="At least one"):
        tsentries.audit([])
