"""beamfastsim.ptgt_queue: the directory queue (submit, claim, finish, orphans), the data pin, and one batch split
back into per-design numbers from a shared tsdevice output (fake chain and tsdevice; no JAX run)."""
import argparse
import json
import threading
import time

import numpy as np
import pytest

from beamfastsim import ptgt_point as PP
from beamfastsim import ptgt_queue as Q


def _req(q, **kw):
    r = dict(rod=[160.0, 3.0, 3.0, 3.0], pot=1000, seed=1, dtype="float32", platform="gpu", data_sha256=None, **kw)
    return r


def test_submit_waits_for_the_worker_and_copies_its_result(tmp_path):
    q, out = tmp_path / "q", tmp_path / "out"

    def fake_worker():
        while True:
            got = Q.take(q, batch=4, gather_s=0.0)
            if got:
                for r in got:
                    Q.finish(q, r, metrics={"stops_per_pot": 1e-3}, point={"id": r["id"]})
                return
            time.sleep(0.05)
    t = threading.Thread(target=fake_worker)
    t.start()
    m = Q.submit_and_wait(q, _req(q), out, wait_s=10, poll_s=0.05)
    t.join()
    assert m == {"stops_per_pot": 1e-3} and json.loads((out / "point.json").read_text())["id"]
    assert not list((q / "req").iterdir()) and not list((q / "run").iterdir())


def test_worker_error_and_timeout_reach_the_client(tmp_path):
    q = tmp_path / "q"
    threading.Thread(target=lambda: (time.sleep(0.2), [Q.finish(q, r, error="boom") for r in Q.take(q, 1, 0.0)])).start()
    with pytest.raises(RuntimeError, match="boom"):
        Q.submit_and_wait(q, _req(q), tmp_path / "o1", wait_s=10, poll_s=0.05)
    with pytest.raises(TimeoutError):
        Q.submit_and_wait(q, _req(q), tmp_path / "o2", wait_s=0.2, poll_s=0.05)


def test_take_waits_for_a_full_batch_or_the_gather_time_and_orphans_requeue(tmp_path):
    q = tmp_path / "q"
    for sub in ("req", "run", "done"):
        (q / sub).mkdir(parents=True)
    now = time.time()
    for i in range(3):
        (q / "req" / f"r{i}.json").write_text(json.dumps(dict(_req(q), id=f"r{i}", submitted=now - 10 + i)))
    assert Q.take(q, batch=5, gather_s=60, now=now) == []            # not full, oldest waited 10 s < 60 s
    got = Q.take(q, batch=2, gather_s=60, now=now)                     # full batch: the two oldest
    assert [r["id"] for r in got] == ["r0", "r1"] and len(list((q / "run").iterdir())) == 2
    assert Q.requeue_orphans(q) == 2 and len(list((q / "req").iterdir())) == 3
    assert [r["id"] for r in Q.take(q, batch=5, gather_s=5, now=now)] == ["r0", "r1", "r2"]


def test_check_data():
    r = dict(data_sha256={"bank.npz": "a" * 64, "ts.gdml": "b" * 64})
    assert Q.check_data(r, {"bank.npz": "a" * 64, "ts.gdml": "b" * 64}) is None
    assert "bank.npz" in Q.check_data(r, {"bank.npz": "c" * 64, "ts.gdml": "b" * 64})
    assert Q.check_data(dict(data_sha256=None), {}) is None


def _reference(path, n, pot):
    np.savez(path, pdg=np.where(np.arange(n) % 2, 13, -211), parent_shard_event_counts=np.array([pot]),
             build_meta=np.array([json.dumps({"heat": {"peak_Gy_per_pot": 2e-9, "peak_Gy_err": 1e-11,
                                                         "total_MeV_per_pot": 900.0}})]))


def test_run_batch_splits_one_tsdevice_pass_into_per_design_numbers(tmp_path, monkeypatch):
    q = tmp_path / "q"
    for sub in ("req", "run", "done"):
        (q / sub).mkdir(parents=True)
    monkeypatch.setattr(Q, "hashes", lambda base: ("b" * 64, "s" * 64, "m" * 64))
    sizes = {160.0: 40, 200.0: 60}

    def run_chain(argv):
        rod = float(argv[argv.index("--rod") + 1])
        _reference(argv[argv.index("--out") + 1], sizes[rod], int(argv[argv.index("--pot") + 1]))

    def run_ts(argv):
        i = argv.index("--reference")
        refs = argv[i + 1:argv.index("--gdml")]
        batch = np.concatenate([np.full(len(np.load(r)["pdg"]), k) for k, r in enumerate(refs)])
        fate = np.zeros(len(batch), int)
        fate[np.flatnonzero(batch == 0)[:4]] = 1                     # design 0: 4 stops
        fate[np.flatnonzero(batch == 1)[:9]] = 1                     # design 1: 9 stops
        np.savez(argv[argv.index("--out") + 1], batch=batch, ds_fate=fate, pdg_ds=np.full(len(batch), 13))
        assert argv[argv.index("--pool") + 1] == str(PP.TS_POOL["gpu"])

    base = argparse.Namespace(platform="gpu", pool=None, bank="bank.npz", slab_bank="slab.npz", ps_map="PSMap.header",
                              cache=str(tmp_path / "cache"), gdml="ts.gdml", field_config="cfg", offline_root="off",
                              maps="maps", ds_geom="ds.txt", ds_offline_root=None, ds_map="DSMap.header")
    reqs = []
    for i, rod in enumerate((160.0, 200.0)):
        r = dict(_req(q), id=f"r{i}", rod=[rod, 3.0, 3.0, 3.0], pot=1000)
        (q / "run" / f"r{i}.json").write_text(json.dumps(r))
        reqs.append(r)
    bad = dict(_req(q), id="r2", data_sha256={"bank.npz": "x" * 64})
    Q.run_batch(q, reqs + [bad], base, run_chain, run_ts, {"bank.npz": "y" * 64})
    m0 = json.loads((q / "done" / "r0" / "metrics.json").read_text())
    m1 = json.loads((q / "done" / "r1" / "metrics.json").read_text())
    assert (m0["stops"], m1["stops"]) == (4.0, 9.0) and m0["stops_per_pot"] == pytest.approx(4e-3)
    assert m0["ts_entrances"] == 40 and m1["ts_entrances"] == 60 and m0["batch_size"] == 2.0
    assert m0["peak_heat_Gy_per_pot"] == pytest.approx(2e-9) and m0["cached"] == 0.0
    p1 = json.loads((q / "done" / "r1" / "point.json").read_text())
    assert p1["queue"]["batch"] == 1 and p1["queue"]["members"] == ["r0", "r1"]
    assert "bank.npz" in (q / "done" / "r2" / "error.txt").read_text()
    assert not list((q / "run").iterdir())
    # the second evaluation of a design reuses its cached chain output
    Q.run_batch(q, [dict(reqs[0], id="r3")], base, run_chain, run_ts, {})
    assert json.loads((q / "done" / "r3" / "metrics.json").read_text())["cached"] == 1.0
