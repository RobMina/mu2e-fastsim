"""uva/throughput_uva.py: a job that ran several processes (PER_GPU > 1) is one row whose events are the sum of its
shards, and a one-process job gives the same numbers as before."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("throughput_uva", Path(__file__).resolve().parents[1] / "uva" / "throughput_uva.py")
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)


def shard(path, generated, accepted, wall, setup):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generated_events": generated, "wall_s": wall,
                                "timers": {"events_with_steps": accepted, "setup_s": setup, "wall_s": wall}}))
    return str(path)


def jobs():
    row = dict(start=0.0, end=1000.0, elapsed_s=1000.0, gpus=1, gpu_type="a100-80", node="n", su=10.0, input_wait_s=100.0)
    return {"1000001": dict(row, jobid="1000001"), "1000002": dict(row, jobid="1000002", start=500.0, end=1500.0)}


def test_processes_of_one_job_add_up(tmp_path):
    shards = [shard(tmp_path / "1000001" / "fpp4" / f"shard_{i}.json", 200_000, 9_000 + i, 600.0 + 100 * i, 100.0 + i)
              for i in range(2)]
    shards.append(shard(tmp_path / "1000002" / "fpp4" / "shard_0.json", 200_000, 8_000, 500.0, 100.0))
    out = T.summarise(jobs(), shards)
    assert out["jobs"] == 2 and out["generated"] == 600_000 and out["accepted"] == 26_001
    two, one = out["per_job"]
    assert (two["processes"], two["generated"], two["accepted"]) == (2, 400_000, 18_001)
    assert two["process_wall_s"] == 700.0 and two["setup_s"] == 101.0
    assert two["accepted_per_gpu_hour"] == pytest.approx(3600.0 * 18_001 / 1000.0)
    assert two["accepted_per_gpu_hour_steady"] == pytest.approx(3600.0 * (9_000 / 500.0 + 9_001 / 599.0))
    assert two["accepted_per_su"] == pytest.approx(1_800.1)
    # one process: unchanged definitions
    assert one["processes"] == 1 and one["accepted_per_gpu_hour_steady"] == pytest.approx(3600.0 * 8_000 / 400.0)
    assert out["makespan_h"] == pytest.approx(1500.0 / 3600.0)
    assert out["accepted_per_su"] == pytest.approx(26_001 / 20.0)
