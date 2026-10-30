"""Opt-in real native emitted-shard smoke; staged detector inputs and CRY-enabled runtime required."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("MU2E_CRY_CHAIN_INPUTS"), reason="opt-in staged native-chain smoke")
def test_real_native_shards_emit_and_merge(tmp_path):
    backend = os.environ.get("MU2E_CRY_CHAIN_BACKEND", "native")
    if backend not in ("native", "device"):
        raise ValueError("unsupported chain-smoke generator")
    base = Path(os.environ["MU2E_CRY_CHAIN_INPUTS"])
    out = Path(os.environ.get("MU2E_CRY_CHAIN_OUTPUT", str(tmp_path)))
    out.mkdir(parents=True, exist_ok=True)
    geom = base / "data/geometry"
    emitted, stats = [], []
    for k in range(2):
        shard = out / f"shard{k}.json"
        result_dir = out / f"native{k}"
        command = [sys.executable, "-m", "crvfastsim", "runchain", "native", "--gen", backend,
                   "--gen-events", "5000", "--gen-seed", str(20261100 + k), "--seed", str(20261200 + k),
                   "--gen-subrun", str(k+1), "--emit", str(shard),
                   "--gdml", str(geom / "mu2e_extracted_v04.gdml"),
                   "--calib", str(geom / "crvcalib_sim_best_v1_5_run1400.csv"),
                   "--deviation", str(geom / "crvphoton_sim_best_v1_5_run1400.csv"),
                   "--tier1b", "strict_v3", "--tier1b-dir", str(base / "tier1b_strict_v2"),
                   "--outdir", str(result_dir)]
        if backend == "device":
            command.append("--jax")
        with (out / f"native{k}.txt").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads(shard.read_text())
        result = json.loads((result_dir / "runchain_stats.json").read_text())
        assert record["generated_events"] == result["generated_events"] == 5000
        assert record["native_generator"] == result["native_generator"]
        if backend == "device":
            assert record["native_generator"]["generator"] == "jax-cry-1.7"
            assert record["native_generator"]["device"].startswith("cuda")
        assert record["inputs"] == ["native"]
        assert record["live_seconds"] > 0 and record["timers"]["generation_s"] > 0
        assert record["native_generator"]["event_range"] == {
            "run": 1400, "subrun": k+1, "first_event": 1, "last_event": 5000}
        emitted.append(str(shard))
        stats.append(record)
    with (out / "merge.txt").open("w") as log:
        subprocess.run([sys.executable, "-m", "crvfastsim", "runchain-merge", *emitted,
                        "--outdir", str(out / "merge")], stdout=log, stderr=subprocess.STDOUT, check=True)
    merged = json.loads((out / "merge/runchain_merge_stats.json").read_text())
    assert merged["generated_events"] == 10000
    assert merged["live_seconds"] == pytest.approx(sum(s["live_seconds"] for s in stats))
    assert len(merged["native_generators"]) == 2
    assert merged["stage_s_summed_over_shards"]["generation_s"] > 0
    assert sum(sum(s["cl_type_counts"].values()) for s in stats) > 0
    for sector in {k for s in stats for k in s["cl_type_counts"]}:
        name = stats[0]["type_names"][sector]
        total = sum(s["cl_type_counts"].get(sector, 0) for s in stats)
        assert merged["clusters_per_live_second"][name] == pytest.approx(total / merged["live_seconds"])
