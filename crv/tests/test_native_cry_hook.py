"""Native generator admission leaves table defaults intact and refuses unsupported modes before setup."""
import json

import pytest

from test_e2e_bench import pipeline

from crvfastsim.analyses import runchain
from crvfastsim.cli import build_parser


def test_cli_default_and_native():
    p = build_parser()
    a = p.parse_args(["runchain", "gen.root"])
    assert a.gen == "table" and a.gen_events is None
    a = p.parse_args(["runchain", "native", "--gen", "native", "--gen-events", "100"])
    assert a.gen == "native" and a.gen_events == 100
    a = p.parse_args(["runchain", "native", "--gen", "device", "--gen-events", "100"])
    assert a.gen == "device" and a.gen_events == 100


@pytest.mark.parametrize("options", [
    {"gen": "unknown"}, {"gen": "native"},
    {"gen": "native", "gen_events": -1},
    {"gen": "native", "gen_events": 2**20},
    {"gen": "native", "gen_events": 1, "gen_subrun": -1},
    {"gen": "native", "gen_events": 1, "gen_subrun": 2**28},
    {"gen": "native", "gen_events": 1, "gen_seed": -1},
    {"gen": "native", "gen_events": 1, "run_config": "run2_v10"},
    {"gen": "native", "gen_events": 1, "max_events": 1},
    {"gen": "native", "gen_events": 1, "compare": ["x"]},
    {"gen": "native", "gen_events": 1, "prepared": object()},
    {"gen": "native", "gen_events": 1, "prefetch": True},
    {"gen": "device", "gen_events": 1, "prefetch": True},
    {"gen": "table", "gen_events": 1},
])
def test_admission_precedes_setup(monkeypatch, options):
    monkeypatch.setattr(runchain, "_resolve", lambda *a, **kw: pytest.fail("setup reached"))
    with pytest.raises(ValueError):
        runchain.run(["native"], "unused", **options)


def test_native_rejects_table_inputs():
    with pytest.raises(ValueError, match="placeholder"):
        runchain.run(["gen.root"], "unused", gen="native", gen_events=100)


def test_native_emission_and_merge_preserve_exposure(pipeline, monkeypatch, tmp_path):
    from mu2efastsim import cosmic_cry
    from mu2efastsim.cry_receipts import GENERATOR

    class Source:
        def __init__(self, config, seed):
            self.seed = seed

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def generate(self, count, subrun):
            data = pipeline["gen"].copy()
            data["subrun"] = subrun
            receipt = {"generator": GENERATOR, "seed": self.seed, "accepted_events": count,
                       "live_seconds": 0.5, "raw_batch_size": 100000,
                       "cosmic_config": {"geometry_sha256": "fixture"},
                       "cosmic_time_offset": "fixture", "stock_overenergy_accumulation": True,
                       "event_range": {"run": 1, "subrun": subrun, "first_event": 3, "last_event": 5}}
            return data, None, receipt

    monkeypatch.setattr(cosmic_cry.CosmicConfig, "extracted", lambda _: None)
    monkeypatch.setattr(cosmic_cry, "CosmicSource", Source)
    monkeypatch.setattr(runchain.tables, "read", lambda *a: pytest.fail("native read a gen table"))
    emitted = []
    for k in range(2):
        emit = tmp_path / f"native{k}.json"
        stats = runchain.run(["native"], str(tmp_path / f"out{k}"), gen="native", gen_events=3,
                             gen_seed=k+1, gen_subrun=k+1, emit=str(emit), library=pipeline["library"],
                             calib="nominal", gdml=pipeline["gdml"])
        shard = json.loads(emit.read_text())
        assert shard["native_generator"] == stats["native_generator"]
        assert shard["live_seconds"] == .5 and shard["generated_events"] == 3
        assert shard["timers"]["generation_s"] >= 0
        emitted.append(str(emit))
    merged = runchain.merge(emitted, tmp_path / "merged")
    assert merged["generated_events"] == 6 and merged["live_seconds"] == 1
    assert merged["clusters_per_live_second"]["99"] == 2
    assert len(merged["native_generators"]) == 2
    with pytest.raises(ValueError, match="repeated fresh-source seed"):
        runchain.merge([emitted[0], emitted[0]], tmp_path / "bad")
