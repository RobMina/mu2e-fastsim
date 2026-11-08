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
    monkeypatch.setattr(run, "batches", lambda keys, ev, size: [(None, len(keys), {"events": [k[2] for k in keys],
                                                                                   "pbt": list(ev["pbt"])})])
    monkeypatch.setattr(menu, "parse_fcl", lambda text: "cfg")
    monkeypatch.setattr(chainmod, "Chain", lambda *a, **kw: chains.append(a) or "chain")
    (tmp_path / "menu.fcl").write_text("")
    monkeypatch.setenv("TRIGGER_DEVICE_DUMPS", "a.root,b.root")
    monkeypatch.setenv("TRIGGER_DEVICE_MENU", str(tmp_path / "menu.fcl"))
    monkeypatch.setenv("TRIGGER_DEVICE_MVA", "mva.xml")
    monkeypatch.setattr(bench, "_CHAINS", {})
    monkeypatch.setattr(bench, "_DUMPS", {})
    return bench, reads, chains


def events_of(blocks):
    return [e for file_blocks in blocks for _, b in file_blocks for e in b["events"]]


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
    assert [nb for file_blocks in blocks for nb, _ in file_blocks] == [1, 3]   # per-file blocks kept


def test_trigger_seeded_load_errors_when_the_dumps_run_out(trigger_dumps):
    bench, reads, chains = trigger_dumps
    with pytest.raises(ValueError, match="too few events for seed 2"):
        bench.load(5, seed=2)                         # events [10, 15) of 12
    with pytest.raises(ValueError, match="too few events for seed 3"):
        bench.load(4, seed=3)                         # starts past the end
    with pytest.raises(ValueError):
        bench.load(4, seed=-1)


def test_hooks_take_a_seed_keyword():
    timer = pytest.importorskip("mu2efastsim.bench")
    from mixing import device_bench
    from trigger.device import bench
    if hasattr(timer, "accepts_seed"):
        assert timer.accepts_seed(device_bench.prepare) and timer.accepts_seed(bench.load)
