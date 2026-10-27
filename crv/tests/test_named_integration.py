"""CLI, population, inventory and runchain metadata integration guards."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import cli
from crvfastsim.analyses import named_inventory, runchain
from crvfastsim.timewindow import resolve_runchain_window
from valhist.fasttables import select_reference_events


def test_cli_scatter_and_conditions():
    args = cli.build_parser().parse_args(["runchain", "gen.root", "--run-config", "run2_v10",
        "--conditions-manifest", "manifest.json", "--below-msc", "--below-scatter-in", "--scatter-in-step-mm", "25"])
    assert args.below_scatter_in and args.scatter_in_step_mm == 25
    assert args.conditions_manifest == "manifest.json"


def test_event_reference_keeps_empty_events_and_exact_run(tmp_path):
    ref = tmp_path / "ref.root"
    with uproot.recreate(ref) as f:
        f.mktree("crvdump/events", {"run": np.array([1400, 1400]), "subrun": np.array([101, 101]),
                                  "event": np.array([3, 9])})
    data = pd.DataFrame(dict(run=[1400, 1401, 1400], subrun=[101]*3, event=[3, 3, 6], bar=[0, 1, 2]))
    st, pa, events, run = select_reference_events(data, data, ref)
    assert run == 1400 and st.bar.tolist() == pa.bar.tolist() == [0]
    assert events.tolist() == [(101 << 20) + 3, (101 << 20) + 9]


def test_merge_rejects_mixed_conditions_before_counting(tmp_path):
    _, contract = resolve_runchain_window("run2_v10")
    base = dict(run_config="run2_v10", window=[0, 100000], time_window=contract)
    files = []
    for i in range(2):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps({**base, "conditions": {"hash": str(i)}}))
        files.append(str(path))
    with pytest.raises(ValueError, match="conditions"):
        runchain.merge(files, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_inventory_negative_reflector_uses_odd_sensors(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"run_config": "run2_v10"}))
    geom = SimpleNamespace(nbars=2, sector=np.array([0, 1]), sector_names=["plus", "minus"])
    rc = SimpleNamespace(name="run2_v10", geometry=lambda: geom,
                         counters_cfg=lambda: {"crs.sipmsAtSide0plus": True, "crs.sipmsAtSide1plus": False,
                                               "crs.sipmsAtSide0minus": False, "crs.sipmsAtSide1minus": True},
                         sectors={"plus": SimpleNamespace(reflector=1), "minus": SimpleNamespace(reflector=-1)})
    monkeypatch.setattr(named_inventory.runconfig, "get", lambda name: rc)
    cond = SimpleNamespace(roc=np.array([1,-1,1,-1,-1,1,-1,1]), feb=np.ones(8), status=np.zeros(8, int),
                           timeoffset=np.zeros(8), deviation=np.zeros(8))
    monkeypatch.setattr(named_inventory, "load_conditions", lambda *a: (cond, dict(scope="test", stock_event_identity={}, assets={})))
    result = named_inventory.inventory(manifest)
    assert result["unmapped"] == result["unmapped_reflector_side"] == 4
    assert result["unmapped_simulated"] == 0
