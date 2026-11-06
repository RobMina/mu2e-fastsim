"""Persistent runner tests with real ROOT output and no detector/GPU dependency."""
import json
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from mixing import stream_benchmark as bench
from mixing.fast_detectors import write_output
from test_mixing_direct_frames import manifest as source_manifest


class Source:
    population = 3
    metadata = {"population": 3, "finite_pool": True}

    def __init__(self, *args, **kwargs):
        self.index = 0
        self.ordinal = 0

    def next_batch(self, size):
        ids = np.column_stack([np.full(size, 2**40, np.uint64), np.ones(size, np.uint64),
                               np.arange(size, dtype=np.uint64) + 2**53])
        result = SimpleNamespace(keys=ids, batch_index=self.index, ordinal_start=self.ordinal,
                                 provenance={"signal_event_ids": ids.copy(), "batch_index": self.index})
        self.index += 1
        self.ordinal += size
        return result


class Backend:
    metadata = {"name": "synthetic", "support": ["tracker_digis", "crv_reco"], "residency": "host test only"}

    def __init__(self, args, source):
        self.calls = 0
        self.barriers = 0

    def sync(self):
        self.barriers += 1

    def run(self, batch, directory):
        self.calls += 1
        keys, n = batch.keys, len(batch.keys)
        timing = SimpleNamespace(mc_pbtime=np.zeros(n), reco_pbtime=np.zeros(n), event_length=np.full(n, 1700.))
        empty = np.array([], dtype=np.int64)
        crv = ({"ev": empty, "adc": np.empty((0, 12), np.int16)}, {"ev": empty}, {"ev": empty})
        jagged = ak.unflatten(np.array([], dtype=np.int64), np.zeros(n, np.int64))
        tracker = {**dict(zip(("run", "subrun", "event"), keys.T)), "d_sid": jagged, "d_adc": jagged}
        write_output(directory / "fast.root", keys, timing, crv, tracker)
        with uproot.update(directory / "fast.root") as root:
            root.mktree("benchmark/events", {**dict(zip(("run", "subrun", "event"), keys.T)),
                        "realization": np.arange(batch.ordinal_start, batch.ordinal_start + n, dtype=np.uint64)})
        return {"events": n}


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(out=tmp_path / "output", frames_manifest=tmp_path / "manifest.json",
                           backend="mixing.stream_backend:Backend", events=8, batch_events=3,
                           max_selected_rows=100, chunk=2, warmup_batches=1, seed=1, tracker_seed=2,
                           sampling_seed=3, require_stage=[])


def test_persistent_setup_tail_and_persisted_denominator(args):
    sources, backends = [], []
    def source_factory(*a, **kw):
        source = Source(*a, **kw)
        sources.append(source)
        return source
    def backend_factory(*a):
        backend = Backend(*a)
        backends.append(backend)
        return backend
    report = bench.run(args, source_factory, backend_factory)
    assert len(sources) == len(backends) == 1
    assert backends[0].calls == 4
    assert report["events"] == 8 and report["warmup_events"] == 3 and report["measured_batches"] == 3
    assert report["events_per_s"] == 8 / report["measured_s"]
    assert report["speedup"] is None
    rows = [json.loads(line) for line in (args.out / "batches.jsonl").read_text().splitlines()]
    assert [row["events"] for row in rows] == [3, 3, 3, 2]
    assert [row["ordinal_start"] for row in rows] == [0, 3, 6, 9]
    assert report["pipeline_s"] >= report["setup_s"] + report["warmup_s"] + report["measured_s"]
    with pytest.raises(FileExistsError):
        bench.run(args, source_factory, backend_factory)


@pytest.mark.parametrize("field,value", [("events", 0), ("warmup_batches", -1), ("sampling_seed", -1)])
def test_invalid_before_output(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        bench.run(args, Source, Backend)
    assert not args.out.exists()


def test_unavailable_reco_rejected(args):
    args.require_stage = ["track_reco"]
    with pytest.raises(ValueError, match="requested stages"):
        bench.run(args, Source, Backend)
    assert not (args.out / "stream_report.json").exists()


def test_corrupt_persisted_identity_cannot_publish_rate(args):
    class Wrong(Backend):
        def run(self, batch, directory):
            result = super().run(batch, directory)
            with uproot.update(directory / "fast.root") as root:
                root.mktree("benchmark/events", {**dict(zip(("run", "subrun", "event"), batch.keys.T)),
                            "realization": np.zeros(len(batch.keys), np.uint64)})
            return result
    with pytest.raises(ValueError, match="realization"):
        bench.run(args, Source, Wrong)
    assert not (args.out / "stream_report.json").exists()
    assert (args.out / "measure_00000001" / "fast.root").exists()


def test_failed_batch_preserves_files_no_summary(args):
    class Failed(Backend):
        def run(self, batch, directory):
            raise RuntimeError("synthetic failure")
    with pytest.raises(RuntimeError):
        bench.run(args, Source, Failed)
    assert not (args.out / "stream_report.json").exists()
    assert (args.out / "warmup_00000000" / "sampling.npz").exists()


def test_batch_cannot_exceed_source_population(args):
    args.batch_events = 4
    with pytest.raises(ValueError, match="population"):
        bench.run(args, Source, Backend)


def test_factory_syntax_and_resolution():
    with pytest.raises(ValueError):
        bench.factory("missing-colon")
    assert bench.factory("mixing.stream_benchmark:run") is bench.run


def test_coherently_changed_output_ids_rejected_against_sampling(args):
    class Wrong(Backend):
        def run(self, batch, directory):
            batch.keys[:, 0] += 1
            return super().run(batch, directory)
    with pytest.raises(ValueError, match="sampled source"):
        bench.run(args, Source, Wrong)
    assert not (args.out / "stream_report.json").exists()


def test_real_resident_source_wires_into_runner(source_manifest, args):
    from mixing.stream_source import StreamSource
    args.frames_manifest = source_manifest[0]
    args.max_selected_rows = 100000
    result = bench.run(args, StreamSource, Backend)
    assert result["events"] == 8
    config = json.loads((args.out / "configuration.json").read_text())
    assert config["source"]["signal_events_full"] == 5
