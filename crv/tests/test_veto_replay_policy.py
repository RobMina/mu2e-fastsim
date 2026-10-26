"""Yield ownership is an explicit replay contract, never a fitted response knob."""
import copy
import dataclasses
import json

import pytest

from crvfastsim.response.photons import ResponseConfig, SectorConfig
from crvfastsim.veto.fastframes import run_config
from crvfastsim.veto.replay_policy import (
    INTENDED, LEGACY_STOCK, LEGACY_FRAMES, LEGACY_WRITERS, TIMING,
    apply_policy, metadata_identity, stock_shared_table_yields,
)


def metadata(policy=INTENDED):
    response, contract = apply_policy(run_config("run1a_v01")[0], policy)
    return {"response_contract": contract, "stock_shared_table_yields_diagnostic": policy == LEGACY_STOCK,
            "effective_yields": {k: v.yield_per_mev for k, v in response.sectors.items()},
            "conditions_sha256": "a" * 64, "photon_dev_sha256": "b" * 64, "timing": TIMING}


def test_literal_first_owner_order_nonadjacent_and_reflector():
    r = ResponseConfig(sectors={"A": SectorConfig("dir/table", 0, 10),
                                "B": SectorConfig("other/table", 0, 20),
                                "C": SectorConfig("dir/table", 1, 30)})
    d = stock_shared_table_yields(r)
    assert [s.yield_per_mev for s in d.sectors.values()] == [10, 20, 10]
    assert d.sectors["C"].reflector == 1
    assert r.sectors["C"].yield_per_mev == 30
    assert d.photon_yield_scale == r.photon_yield_scale
    reverse = dataclasses.replace(r, sectors=dict(reversed(list(r.sectors.items()))))
    assert stock_shared_table_yields(reverse).sectors["A"].yield_per_mev == 30


@pytest.mark.parametrize("policy", [INTENDED, LEGACY_STOCK])
def test_policy_copy_and_metadata_round_trip(policy):
    nominal = run_config("run1a_v01")[0]
    before = copy.deepcopy(nominal)
    effective, contract = apply_policy(nominal, policy)
    assert nominal == before
    assert effective.sectors["T2"].yield_per_mev == (39794 if policy == LEGACY_STOCK else 30482)
    identity, digest, recovery = metadata_identity(json.loads(json.dumps(metadata(policy))))
    assert recovery is None and len(digest) == 64
    assert identity["response_contract"] == contract
    assert metadata_identity(metadata(LEGACY_STOCK if policy == INTENDED else INTENDED))[1] != digest


def test_pinned_order_and_profile_fail_closed():
    r = run_config("run1a_v01")[0]
    with pytest.raises(ValueError, match="ordered nominal"):
        apply_policy(dataclasses.replace(r, sectors=dict(reversed(list(r.sectors.items())))), LEGACY_STOCK)
    with pytest.raises(ValueError, match="ordered nominal"):
        apply_policy(dataclasses.replace(r, photon_yield_scale=0.5), INTENDED)
    with pytest.raises(ValueError, match="unknown yield"):
        apply_policy(r, "new_default")


@pytest.mark.parametrize("key,value,match", [
    ("stock_shared_table_yields_diagnostic", True, "flag disagree"),
    ("effective_yields", {"T1": 39794, "T2": 39794}, "effective yields"),
    ("timing", "wrap photons", "timing"),
    ("photon_dev_sha256", None, "photon_dev"),
])
def test_inconsistent_sidecar_rejected(key, value, match):
    m = metadata()
    m[key] = value
    with pytest.raises(ValueError, match=match):
        metadata_identity(m)


def test_contract_mutation_rejected():
    m = metadata()
    m["response_contract"]["yield_owner"]["T2"] = "T1"
    with pytest.raises(ValueError, match="pinned"):
        metadata_identity(m)


@pytest.mark.parametrize("policy", [INTENDED, LEGACY_STOCK])
def test_legacy_recovery_explicit_hashed_and_nonmutating(policy):
    m = metadata(policy)
    del m["response_contract"]
    writer = next(k for k, v in LEGACY_WRITERS.items() if v is None)
    m["code_sha256"] = {"replay_fast.py": writer, "fastframes.py": LEGACY_FRAMES}
    before = copy.deepcopy(m)
    with pytest.raises(ValueError, match="explicit"):
        metadata_identity(m)
    identity, digest, recovery = metadata_identity(m, policy)
    assert recovery and m == before
    assert digest == metadata_identity(metadata(policy))[1]
    with pytest.raises(ValueError, match="disagrees"):
        metadata_identity(m, LEGACY_STOCK if policy == INTENDED else INTENDED)
    m["code_sha256"]["fastframes.py"] = "unknown"
    with pytest.raises(ValueError, match="unrecognized"):
        metadata_identity(m, policy)


def test_old_configured_sidecar_missing_flag_is_known_writer_only():
    m = metadata()
    for k in ("response_contract", "effective_yields", "stock_shared_table_yields_diagnostic"):
        del m[k]
    writer = next(k for k, v in LEGACY_WRITERS.items() if v == INTENDED)
    m["code_sha256"] = {"replay_fast.py": writer, "fastframes.py": LEGACY_FRAMES}
    assert metadata_identity(m, INTENDED)[2]
    with pytest.raises(ValueError, match="disagrees"):
        metadata_identity(m, LEGACY_STOCK)


@pytest.mark.parametrize("field", ["stock_shared_table_yields_diagnostic", "effective_yields"])
def test_policy_aware_sidecar_missing_fields_fails(field):
    m = metadata()
    del m[field]
    with pytest.raises(ValueError):
        metadata_identity(m)
    del m["response_contract"]
    m["code_sha256"] = {"replay_fast.py": next(k for k, v in LEGACY_WRITERS.items() if v is None),
                        "fastframes.py": LEGACY_FRAMES}
    with pytest.raises(ValueError, match="missing yield metadata"):
        metadata_identity(m, INTENDED)


@pytest.mark.parametrize("option,policy", [([], INTENDED), (["--yield-policy", LEGACY_STOCK], LEGACY_STOCK),
                                         (["--stock-shared-table-yields"], LEGACY_STOCK)])
def test_runner_writes_explicit_contract_without_response_mutation(tmp_path, monkeypatch, option, policy):
    import awkward as ak
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from crvfastsim import chain
    from crvfastsim.veto import replay_fast as runner
    from crvfastsim.veto.replay import replay_tables, write_replay, sha256
    path = tmp_path / "mixed.root"
    events = pd.DataFrame({"run": [1430], "subrun": [0], "event": [1], "pbtime": [-204.46], "intensity": [1.]})
    write_replay(path, replay_tables(pd.DataFrame(), events))
    path.with_suffix(".json").write_text(json.dumps({"output_sha256": sha256(path), "events": 1, "run": 1430}))
    conditions, photon = tmp_path / "conditions", tmp_path / "photon"
    conditions.write_text("conditions fixture")
    photon.write_text("photon fixture")
    original = run_config("run1a_v01")[0]
    monkeypatch.setattr(runner.FF, "conditions_from_dump", lambda *a: SimpleNamespace())
    monkeypatch.setattr(runner.FF, "_setup", lambda *a: {"cfg": chain.ChainConfig(response=original)})
    seen = []

    def frames(*args, **kwargs):
        seen.append(kwargs["setup"]["cfg"].response)
        return ak.zip({"run": [1430], "subrun": [0], "event": [1],
                       "coincs": ak.zip({"time": ak.unflatten(np.zeros(0), [0])})}, depth_limit=1)

    monkeypatch.setattr(runner.FF, "frames_from_steps", frames)
    out = tmp_path / "fast.parquet"
    runner.main([str(path), "--conditions", str(conditions), "--photon-dev", str(photon), "--out", str(out), *option])
    meta = json.loads(out.with_suffix(".json").read_text())
    assert metadata_identity(meta)[0]["response_contract"]["policy"] == policy
    assert meta["frames"] == 1 and meta["coincs"] == 0
    assert seen[0].sectors["T2"].yield_per_mev == (39794 if policy == LEGACY_STOCK else 30482)
    assert original.sectors["T2"].yield_per_mev == 30482


def test_cli_mutually_exclusive_policy_options():
    from crvfastsim.veto.replay_fast import main
    with pytest.raises(SystemExit) as exc:
        main(["unused.root", "--conditions", "unused", "--photon-dev", "unused", "--out", "unused",
              "--yield-policy", INTENDED, "--stock-shared-table-yields"])
    assert exc.value.code == 2
