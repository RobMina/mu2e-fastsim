"""Shipped device capacity files validate against the adapters' capacity contracts."""
import json
from pathlib import Path

import pytest

from mixing.device_tracker import Capacities

FILES = sorted((Path(__file__).resolve().parents[1] / "mixing" / "capacities").glob("*.json"))


def test_capacity_files_exist():
    assert {path.name for path in FILES} >= {"run1a_mix1bb_batch128.json", "run1a_mix1bb_batch64.json",
                                             "run1a_mix1bb_batch256.json", "run1a_mix1bb_batch256_exact.json"}


def test_run1a_crv_coincidence_hits_sized_for_production():
    """CRV pulses per event and sector type have an exponential tail: every Run 1A file takes 96 per event in the
    main hit buffer and up to 384 in its coinc_tiers (README: under 0.01 expected overflows in 1e8 events)."""
    for name in ("run1a_mix1bb_batch64.json", "run1a_mix1bb_batch128.json", "run1a_mix1bb_batch256.json",
                 "run1a_mix1bb_batch256_exact.json", "run1a_mix1bb_batch256_cpu.json",
                 "run1a_mix1bb_batch256_exact_cpu.json"):
        crv = json.loads((FILES[0].parent / name).read_text())["response"]["crv"]
        assert crv["coinc_hits"] == 96 and crv["coinc_tiers"] == [[16, 192], [4, 384]], name


@pytest.mark.parametrize("suffix", ["", "_cpu"])
def test_batch256_default_and_exact_reference(suffix):
    """The batch-256 default file has both W6 approximations (donor-frame reuse, CRV thermal noise on hit counters);
    the exact reference has neither and differs from it only there (and in the CRV buffers they resize)."""
    folder = FILES[0].parent
    default = json.loads((folder / f"run1a_mix1bb_batch256{suffix}.json").read_text())
    exact = json.loads((folder / f"run1a_mix1bb_batch256_exact{suffix}.json").read_text())
    assert "tracker_reuse" in default["response"] and "tracker_reuse" not in exact["response"]
    assert default["response"]["crv"]["thermal_counters"] == "hit"
    reuse = default["response"]["tracker_reuse"]
    assert reuse["shifts"] == "events" and reuse["event_jitter"] is True
    assert "thermal_counters" not in exact["response"]["crv"]
    resized = {"thermal", "ct1", "ct2", "charge_clusters", "thermal_counters"}
    assert {k: v for k, v in default["response"]["crv"].items() if k not in resized} == \
        {k: v for k, v in exact["response"]["crv"].items() if k not in resized}
    assert all(default["response"]["crv"][k] < exact["response"]["crv"][k] for k in resized - {"thermal_counters"})
    for key in ("max_donors", "rows"):
        assert default[key] == exact[key]
    assert default["response"]["tracker"] == exact["response"]["tracker"]
    assert default["response"]["output"] == exact["response"]["output"]


@pytest.mark.parametrize("name", ["run1a_mix1bb_batch256", "run1a_mix1bb_batch256_exact"])
def test_batch256_cpu_files_change_only_the_end_tiers(name):
    """The CPU files are the batch-256 files with finer tracker end tiers (narrow ends of 8 clusters), in the tracker
    and in the reuse `compute` tracker; the reuse `precompute` (part of a saved reuse table's key) is unchanged."""
    folder = FILES[0].parent
    base = json.loads((folder / f"{name}.json").read_text())
    cpu = json.loads((folder / f"{name}_cpu.json").read_text())
    tiered = [("tracker",)] + ([("tracker_reuse", "compute")] if "tracker_reuse" in base["response"] else [])
    for path in tiered:
        b, c = base["response"], cpu["response"]
        for key in path:
            b, c = b[key], c[key]
        assert c["clusters_per_end"] == 8 and len(c["end_tiers"]) == 15 and c["end_tiers"] != b["end_tiers"]
        for d in (b, c):
            d.pop("clusters_per_end"), d.pop("end_tiers")
    assert cpu == base


LAYOUT_DEFAULTS = ("flat_saturation_rows", "flat_saturation_pool", "flat_saturation_tiles", "electronics_launches",
                   "narrow_tier")


def _without_layout_defaults(tracker):
    """The tracker capacities without the exact layout keys, which then take their code defaults."""
    return {k: v for k, v in tracker.items() if k not in LAYOUT_DEFAULTS}


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.name)
def test_capacity_file_contract(path):
    """Every file validates, and its explicit exact layout keys are the code defaults."""
    caps = json.loads(path.read_text())
    assert set(caps) == {"max_donors", "rows", "response"}
    assert set(caps["rows"]) == {"particles", "gen", "tracker", "crv"}
    assert all(type(v) is int and v > 0 for v in [caps["max_donors"], *caps["rows"].values()])
    tracker = Capacities(**caps["response"]["tracker"])
    assert tracker.end_tiers and tracker.charge_blocks_per_launch
    assert tracker == Capacities(**_without_layout_defaults(caps["response"]["tracker"]))
    assert tracker.straws % tracker.electronics_batch == 0
    assert set(caps["response"]) - {"tracker_reuse", "output"} == {"tracker", "crv"}
    if "output" in caps["response"]:             # terminal output sizes (device_physics.output_capacities)
        from mixing.device_physics import output_capacities
        output = output_capacities(caps["response"]["output"])
        assert output["tracker_rows"] < tracker.straws * tracker.pairs_per_straw
    if "tracker_reuse" in caps["response"]:      # donor-frame reuse: both tracker capacity sets validate
        from mixing.device_reuse import ReuseCapacities
        reuse = ReuseCapacities(**caps["response"]["tracker_reuse"])
        for part in (reuse.compute, reuse.precompute):
            sized = Capacities(**part)
            assert sized.straws % sized.electronics_batch == 0
            assert sized == Capacities(**_without_layout_defaults(part))
    pytest.importorskip("jax")
    pytest.importorskip("crvfastsim")
    from mixing import device_crv
    crv = dict(caps["response"]["crv"])
    options = {name: crv.pop(name) for name in list(crv) if name in device_crv.OPTIONS}
    tiers = crv.pop("coinc_tiers", None)     # optional wider coincidence hit rows: [[events, hits], ...]
    if tiers is not None:
        widths = [crv["coinc_hits"]] + [hits for _, hits in tiers]
        assert all(type(v) is int and v > 0 for t in tiers for v in t) and widths == sorted(set(widths))
    assert set(crv) == set(device_crv.CAPACITIES)
    assert all(type(v) is int and v > 0 for v in crv.values())
    expected = {"waveform": "scatter", "digi_starts": "rows", "pulse_scan": "bounded"}
    if path.name in ("run1a_mix1bb_batch256.json", "run1a_mix1bb_batch256_cpu.json"):   # W6 defaults: thermal on hit counters
        expected["thermal_counters"] = "hit"
    assert options == expected
