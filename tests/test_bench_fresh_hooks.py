"""Seed-aware prepare hooks of the shared timer (--fresh-inputs): W6 mixing and W8 trigger bench entry points."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

jax = pytest.importorskip("jax")


# ---------------------------------------------------------------- W6: mixing.device_bench

@pytest.fixture
def mixing_config(tmp_path, monkeypatch):
    from mixing import device_backend, device_bench

    class FakeSource:
        made = []

        def __init__(self, manifest, seed, max_selected_rows):
            self.population, self.seed = 100, seed
            FakeSource.made.append(self)

    class FakeBackend:
        made = []

        def __init__(self, args, source):
            self.jax, self.device, self.args = jax, jax.devices("cpu")[0], args
            self.pools, self.state, self.assets = "pools", "state", "assets"
            self._seeds = np.array([args.sampling_seed, args.seed, args.tracker_seed], np.uint64)
            FakeBackend.made.append(self)

    monkeypatch.setattr(device_backend, "Source", FakeSource)
    monkeypatch.setattr(device_backend, "Backend", FakeBackend)
    monkeypatch.setattr(device_bench, "_SHARED", {})
    values = {name: str(tmp_path / name) for name in device_bench.PATHS}
    values.update(out=str(tmp_path / "run"), batch_events=10, max_selected_rows=5)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    return device_bench, path, FakeSource, FakeBackend


def test_mixing_bench_pipelines_by_default(mixing_config):
    bench, path, _, _ = mixing_config
    assert bench.configuration(path).device_pipeline is True
    assert bench.configuration(path).device_pipeline_writers == 3
    assert bench.configuration(path).output_compression == "zstd1"
    values = json.loads(path.read_text())
    path.write_text(json.dumps({**values, "output_compression": None}))
    assert bench.configuration(path).output_compression is None     # null keeps uproot's ZLIB 1
    path.write_text(json.dumps(values))
    values = json.loads(path.read_text())
    path.write_text(json.dumps({**values, "device_pipeline": False}))
    assert bench.configuration(path).device_pipeline is False


def test_mixing_fixed_prepare_unchanged(mixing_config):
    bench, path, source, backend = mixing_config
    prepared = bench.prepare(20, path)
    assert prepared.shared is None and prepared.args.out.name == "run" and prepared.args.out.is_dir()
    assert prepared.ensure_backend() is backend.made[0]
    with pytest.raises(FileExistsError):
        bench.prepare(20, path)                       # an existing output directory is still refused


def test_mixing_seeded_prepares_share_pools_and_backend(mixing_config):
    bench, path, source, backend = mixing_config
    first, second = bench.prepare(20, path, seed=1), bench.prepare(20, path, seed=2)
    assert len(source.made) == 1 and first.source is second.source is source.made[0]
    assert first.args.out.name == "seed_1" and second.args.out.name == "seed_2"
    assert first.args.out.is_dir() and second.args.out.is_dir() and first.args.out.parent.name == "run"
    b1 = first.ensure_backend()
    assert list(np.asarray(b1._seeds)) == [20261107, 20261107, 20261106]
    b2 = second.ensure_backend()
    assert b1 is b2 is backend.made[0] and len(backend.made) == 1
    assert list(np.asarray(b2._seeds)) == [20261108, 20261108, 20261107]
    assert b2._arguments[3] is b2._seeds
    with pytest.raises(FileExistsError):
        bench.prepare(20, path, seed=1)               # one output directory per seed
    with pytest.raises(ValueError):
        bench.prepare(20, path, seed=-1)


# ---------------------------------------------------------------- W8: trigger.device.bench

@pytest.fixture
def trigger_dumps(monkeypatch, tmp_path):
    from trigger.device import bench, chain as chainmod, menu, run
    sizes = {"a.root": 5, "b.root": 7}
    reads, chains = [], []

    def load_dump(path, tree):
        reads.append(path)
        start = 0 if path == "a.root" else 100
        keys = [(1, 0, start + i) for i in range(sizes[path])]
        return keys, {"sid": [np.arange(1)] * sizes[path], "pbt": np.arange(sizes[path]) + start}, {}

    monkeypatch.setattr(run, "load_dump", load_dump)
    monkeypatch.setattr(run, "conditions", lambda path, tree: "cond")
    monkeypatch.setattr(run, "batches", lambda keys, ev, size, *a, **kw: [(None, len(keys), {"events": [k[2] for k in keys],
                                                                                             "pbt": list(ev["pbt"])})])
    monkeypatch.setattr(menu, "parse_fcl", lambda text: "cfg")
    monkeypatch.setattr(chainmod, "Chain", lambda *a, **kw: chains.append(a) or "chain")
    (tmp_path / "menu.fcl").write_text("")
    monkeypatch.setenv("TRIGGER_DEVICE_DUMPS", "a.root,b.root")
    monkeypatch.setenv("TRIGGER_DEVICE_MENU", str(tmp_path / "menu.fcl"))
    monkeypatch.setenv("TRIGGER_DEVICE_MVA", "mva.xml")
    monkeypatch.setenv("TRIGGER_DEVICE_ORDER", "input")          # the stub batches carry no row indices
    monkeypatch.setattr(bench, "_CHAINS", {})
    monkeypatch.setattr(bench, "_DUMPS", {})
    return bench, reads, chains


def events_of(blocks):
    return [e for file_blocks in blocks for _, b, _ in file_blocks for e in b["events"]]


def test_trigger_fixed_load_takes_the_first_events(trigger_dumps):
    bench, reads, chains = trigger_dumps
    ch, blocks = bench.load(7)
    assert ch == "chain" and events_of(blocks) == [0, 1, 2, 3, 4, 100, 101]


def test_trigger_seeded_slices_are_disjoint_and_built_once(trigger_dumps):
    bench, reads, chains = trigger_dumps
    slices = [events_of(bench.load(4, seed=s)[1]) for s in (0, 1, 2)]
    assert slices == [[0, 1, 2, 3], [4, 100, 101, 102], [103, 104, 105, 106]]
    assert [b for s in slices for b in s] == sorted(set(b for s in slices for b in s))
    assert len(chains) == 1 and sorted(reads) == ["a.root", "b.root"]
    _, blocks = bench.load(4, seed=1)
    assert [nb for file_blocks in blocks for nb, _, _ in file_blocks] == [1, 3]   # per-file blocks kept


def test_trigger_seeded_load_errors_when_the_dumps_run_out(trigger_dumps):
    bench, reads, chains = trigger_dumps
    with pytest.raises(ValueError, match="too few events for seed 2"):
        bench.load(5, seed=2)                         # events [10, 15) of 12
    with pytest.raises(ValueError, match="too few events for seed 3"):
        bench.load(4, seed=3)                         # starts past the end
    with pytest.raises(ValueError):
        bench.load(4, seed=-1)


def test_trigger_hit_order_gives_the_input_order_results(monkeypatch):
    """TRIGGER_DEVICE_ORDER=hits: events batched by digi count, every per-event result back in event order (the
    prescale counter of finish runs in event order), the same as the input-order blocks."""
    from trigger.device import bench, run
    rng = np.random.default_rng(3)
    n = 23
    sizes = rng.integers(1, 300, n)
    keys = [(1, 0, 1000 + i) for i in range(n)]
    ev = {k: [rng.integers(0, 50, m) for m in sizes] for k in run.DIG}
    ev["pbt"], ev["spill"] = np.zeros(n), np.zeros(n, np.int64)

    class Fake:
        def decide(self, b):                                       # a per-event result: its digi count and event
            nh = b["valid"].sum(axis=1)
            return {"p": np.where(b["valid_event"], nh * 100000 + b["event"], -1)}, {"a:tc": b["event"] % 3 == 0}, {}

        def finish(self, dec, aux, n_events):
            return {"p": dec["p"][:n_events], "count": np.cumsum(aux["a:tc"][:n_events])}

    out = {}
    for order in ("input", "hits"):
        monkeypatch.setenv("TRIGGER_DEVICE_ORDER", order)
        blocks = [[(nb, b, rows if order == "hits" else None) for rows, nb, b in
                   (run.sorted_batches(keys, ev, 8, hmin=16) if order == "hits" else run.batches(keys, ev, 8, 16))]]
        out[order] = bench.step((Fake(), blocks))[0]
        if order == "hits":   # sorted blocks pad the hit axis to their own largest event only
            assert [b["valid"].shape[1] for _, b, _ in blocks[0]] == sorted(b["valid"].shape[1] for _, b, _ in blocks[0])
    np.testing.assert_array_equal(out["input"]["p"], sizes * 100000 + 1000 + np.arange(n))
    for k in out["input"]:
        np.testing.assert_array_equal(out["hits"][k], out["input"][k])


def test_trigger_sorted_batches_cover_every_event_once(monkeypatch):
    from trigger.device import run
    sizes = [5, 0, 70, 3, 3, 129, 1]
    ev = {k: [np.arange(m) for m in sizes] for k in run.DIG}
    ev["pbt"], ev["spill"] = np.zeros(len(sizes)), np.zeros(len(sizes), np.int64)
    keys = [(1, 0, i) for i in range(len(sizes))]
    got = list(run.sorted_batches(keys, ev, 3, hmin=4))
    rows = np.concatenate([r for r, _, _ in got])
    assert sorted(rows.tolist()) == list(range(len(sizes))) and [sizes[i] for i in rows] == sorted(sizes)
    for r, nb, b in got:
        assert nb == len(r) and b["event"][:nb].tolist() == r.tolist()
        assert b["valid"].sum(axis=1)[:nb].tolist() == [sizes[i] for i in r]
    parts = [{"x": np.asarray(r) * 10} for r, _, _ in got]
    np.testing.assert_array_equal(run.unsort([r for r, _, _ in got], parts)["x"], np.arange(len(sizes)) * 10)
    from trigger.device import bench
    monkeypatch.setenv("TRIGGER_DEVICE_ORDER", "bogus")
    with pytest.raises(ValueError):
        bench._order()


def test_hooks_take_a_seed_keyword():
    timer = pytest.importorskip("mu2efastsim.bench")
    from mixing import device_bench
    from trigger.device import bench
    if hasattr(timer, "accepts_seed"):
        assert timer.accepts_seed(device_bench.prepare) and timer.accepts_seed(bench.load)


@pytest.mark.parametrize("writers", [1, 3])
def test_mixing_bench_pipeline_writers(tmp_path, writers):
    """device_pipeline_writers W: up to W batches collected at once, every batch collected once in its own directory,
    the summed metrics unchanged."""
    import threading
    import time
    from mixing import device_bench

    class Source:
        def __init__(self):
            self.index = 0

        def next_batch(self, size):
            self.index += 1
            return SimpleNamespace(count=size, batch_index=self.index)

    class Backend:
        def __init__(self):
            self.active = self.peak = 0
            self.lock, self.collected = threading.Lock(), []

        def dispatch(self, request):
            return request

        def collect(self, request, directory):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.05)
            with self.lock:
                self.active -= 1
                self.collected.append((request.batch_index, directory.name))
            return {"events": request.count, "wait_transfer_and_write_s": 0.05, "capacity_usage": {"x": request.batch_index}}

        def sync(self):
            pass

    args = SimpleNamespace(batch_events=10, device_pipeline=True, device_pipeline_writers=writers, out=tmp_path)
    prepared = SimpleNamespace(args=args, source=Source(), events=55, passes=0)
    backend = Backend()
    summary = device_bench._run_pipelined(prepared, backend, tmp_path)
    assert summary["events"] == 55 and summary["batches"] == 6 and summary["capacity_usage_max"] == {"x": 6}
    assert sorted(backend.collected) == [(i, f"batch_{i:08d}") for i in range(1, 7)]
    assert backend.peak == writers
    with pytest.raises(ValueError, match="positive integer"):
        device_bench._run_pipelined(SimpleNamespace(**{**vars(prepared), "args": SimpleNamespace(**{**vars(args), "device_pipeline_writers": 0})}),
                                    backend, tmp_path / "x")
