"""Shipped device capacity files validate against the adapters' capacity contracts."""
import json
from pathlib import Path

import pytest

from mixing.device_tracker import Capacities

FILES = sorted((Path(__file__).resolve().parents[1] / "mixing" / "capacities").glob("*.json"))


def test_capacity_files_exist():
    assert {path.name for path in FILES} >= {"run1a_mix1bb_batch128.json", "run1a_mix1bb_batch64.json"}


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_capacity_file_contract(path):
    caps = json.loads(path.read_text())
    assert set(caps) == {"max_donors", "rows", "response"}
    assert set(caps["rows"]) == {"particles", "gen", "tracker", "crv"}
    assert all(type(v) is int and v > 0 for v in [caps["max_donors"], *caps["rows"].values()])
    tracker = Capacities(**caps["response"]["tracker"])
    assert tracker.end_tiers and tracker.charge_blocks_per_launch
    assert tracker.straws % tracker.electronics_batch == 0
    assert set(caps["response"]) - {"tracker_reuse", "output"} == {"tracker", "crv"}
    if "output" in caps["response"]:             # terminal output sizes (device_physics.output_capacities)
        from mixing.device_physics import output_capacities
        output = output_capacities(caps["response"]["output"])
        assert output["tracker_rows"] < tracker.straws * tracker.pairs_per_straw
    if "tracker_reuse" in caps["response"]:      # opt-in donor-frame reuse: both tracker capacity sets validate
        from mixing.device_reuse import ReuseCapacities
        reuse = ReuseCapacities(**caps["response"]["tracker_reuse"])
        for part in (reuse.compute, reuse.precompute):
            sized = Capacities(**part)
            assert sized.straws % sized.electronics_batch == 0
    pytest.importorskip("jax")
    pytest.importorskip("crvfastsim")
    from mixing import device_crv
    crv = dict(caps["response"]["crv"])
    options = {name: crv.pop(name) for name in list(crv) if name in device_crv.OPTIONS}
    assert set(crv) == set(device_crv.CAPACITIES)
    assert all(type(v) is int and v > 0 for v in crv.values())
    assert options == {"waveform": "scatter", "digi_starts": "rows", "pulse_scan": "bounded"}
