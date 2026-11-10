"""mu2efastsim.demo on synthetic inputs: a fake runner writes what each entry point writes, the demo reads the headline,
compares it with its reference and writes the summary; the CE device chain runs on the GPU and the CPU when a GPU is
seen; without the release it skips with a message and exit 2."""
import json
import subprocess
from pathlib import Path

import pytest

from mu2efastsim import demo


@pytest.fixture
def release(tmp_path, monkeypatch):
    inputs = tmp_path / "release"
    for f in [demo.BEAM_TABLE, demo.CRV_GEN, *demo.CRV_FILES]:
        (inputs / f).parent.mkdir(parents=True, exist_ok=True)
        (inputs / f).write_bytes(b"x")
    off, maps = tmp_path / "offline", tmp_path / "maps"
    (off / "Offline/Mu2eG4/geom").mkdir(parents=True)
    (off / "Offline/Mu2eG4/geom/geom_common.txt").write_text("")
    maps.mkdir()
    for m in ("DSMap.header", "TSdMap.header"):
        (maps / m).write_text("")
    monkeypatch.setenv("M2FS_INPUTS", str(inputs))
    monkeypatch.setenv("M2FS_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("BEAMFS_OFFLINE_ROOT", str(off))
    monkeypatch.setenv("BEAMFS_MAPS", str(maps))
    return tmp_path


def fake(shift=0.0, fail=(), rates=None):
    """Runner writing synthetic outputs at the reference values (shifted by `shift` reference-sized units); the CE
    device chain reports its batches at `rates` electrons/s per arm (the first batch 10 s slower)."""
    calls, arms = [], []
    rates = rates or {"gpu": 50000.0, "cpu": 100.0}

    def run(cmd, log, arm=None):
        calls.append(cmd)
        arms.append(arm)
        chain = "beam" if "point" in cmd else "ce" if "beamfastsim.cechain" in cmd else "crv"
        log.write_text("synthetic")
        if chain in fail:
            return 6, "", 0.1
        ref = demo.REFERENCES[chain].value * (1 + shift)
        n = int(cmd[cmd.index("--n" if chain != "crv" else "--max-events") + 1])
        if chain == "beam":
            d = Path(cmd[cmd.index("--out") + 1]); d.mkdir(parents=True)
            (d / "metrics.json").write_text(json.dumps({"stop_frac": ref, "stop_frac_err": 0.003, "muons": n}))
            return 0, "", 1.0
        if chain == "ce":
            b = int(cmd[cmd.index("--batch") + 1])
            wall = b / rates[arm]
            batches = [dict(n=b, wall_s=wall + 10.0)] + [dict(n=b, wall_s=wall)] * (n // b - 1)
            return 0, "log line\n" + json.dumps({"n": n, "fates": {"plane": round(ref * n), "stopped": 1},
                                                 "dtype": cmd[cmd.index("--device-float") + 1], "maps_s": 1.0,
                                                 "batches": batches}), 1.0
        d = Path(cmd[cmd.index("--outdir") + 1]); d.mkdir(parents=True)
        (d / "runchain_stats.json").write_text(json.dumps({"generated_events": n,
                                                           "clusters_per_generated": {"EX": ref * 0.6, "T1": ref * 0.4}}))
        return 0, "", 1.0
    run.calls, run.arms = calls, arms
    return run


def label(arm):
    return f"synthetic {arm}"


def main(argv, run, gpu=True, cores=108):
    return demo.main(argv, runner=run, probe=lambda: cores if gpu else None, label=label)


def test_references_are_filled():
    for name, ref in demo.REFERENCES.items():
        assert ref.source, name
        assert ref.value > 0 and ref.error > 0, name


def test_all_chains_pass_at_the_reference(release, capsys):
    run = fake()
    assert main([], run) == 0
    out = capsys.readouterr().out
    assert out.count(" ok ") == 4
    s = json.loads((release / "data/demo/demo_summary.json").read_text())
    assert s["passed"] and [(r["chain"], r.get("arm")) for r in s["chains"]] == [
        ("beam", None), ("ce", "gpu"), ("ce", "cpu"), ("crv", None)]
    assert run.arms == [None, "gpu", "cpu", None]
    assert {int(c[c.index("--n") + 1]) for c in run.calls if "--n" in c} == {20000, 2 * 1048576, 2 * 4096}
    assert s["ce_gpu_over_cpu"] == 500.0 and "warm rate GPU / CPU: 500x" in out
    gpu = s["chains"][1]
    assert gpu["warm_rate"] == 50000.0 and gpu["setup_s"] == 11.0 and gpu["device"] == "synthetic gpu"


def test_no_gpu_runs_the_cpu_arm_only(release, capsys):
    run = fake()
    assert main(["--only", "ce"], run, gpu=False) == 0
    out = capsys.readouterr().out
    assert run.arms == ["cpu"] and "no GPU visible to JAX" in out and "GPU / CPU" not in out
    s = json.loads((release / "data/demo/demo_summary.json").read_text())
    assert s["ce_gpu_over_cpu"] is None and s["chains"][0]["sample"] == 2 * 4096


def test_device_cpu_skips_the_probe_and_gpu_requires_one(release, capsys):
    run = fake()
    assert demo.main(["--only", "ce", "--device", "cpu"], runner=run, probe=lambda: 1 / 0, label=label) == 0
    assert run.arms == ["cpu"]
    assert main(["--only", "ce", "--device", "gpu"], fake(), gpu=False) == 2
    assert "JAX sees no GPU" in capsys.readouterr().out
    assert main(["--only", "beam", "--device", "gpu"], fake(), gpu=False) == 0


def test_float32_and_gpu_options(release):
    run = fake()
    assert main(["--only", "ce", "--float32"], run) == 0
    g, c = run.calls
    assert g[g.index("--device-float") + 1] == c[c.index("--device-float") + 1] == "float32"
    assert "--unroll" in g and "--unroll" not in c and g[g.index("--lanes") + 1] == "524288" and "--lanes" not in c
    assert g[g.index("--backend") + 1] == c[c.index("--backend") + 1] == "device"


def test_quick_uses_a_quarter(release):
    run = fake()
    assert main(["--quick", "--only", "ce,crv"], run) == 0
    assert [c[c.index("--batch") + 1] for c in run.calls[:2]] == ["262144", "1024"]
    assert run.calls[0][run.calls[0].index("--n") + 1] == str(2 * 262144)


def test_gpu_batch_follows_the_sm_count(release):
    assert [demo.gpu_batch(c) for c in (0, 132, 108, 84, 56, 28, 14, 1)] == [
        1048576, 1048576, 1048576, 524288, 524288, 262144, 131072, 65536]
    run = fake()
    assert main(["--only", "ce"], run, cores=14) == 0
    g = run.calls[0]
    assert g[g.index("--batch") + 1] == "131072" and g[g.index("--lanes") + 1] == "65536"
    assert run.calls[2][run.calls[2].index("--max-events") + 1] == "250"


def test_outside_tolerance_and_failed_chain(release, capsys):
    assert main(["--only", "beam"], fake(shift=0.5)) == 1
    assert main(["--only", "crv"], fake(shift=0.5)) == 1
    assert main(["--only", "ce"], fake(shift=-0.5)) == 1
    assert "OUTSIDE" in capsys.readouterr().out
    assert main(["--only", "ce"], fake(fail=("ce",))) == 3


def test_ce_timing_needs_batches():
    out = json.dumps({"n": 30, "fates": {"plane": 20}, "maps_s": 2.0,
                      "batches": [{"n": 10, "wall_s": 7.0}, {"n": 10, "wall_s": 2.0}, {"n": 10, "wall_s": 3.0}]})
    t = demo.ce_timing(out)
    assert t["warm_rate"] == 4.0 and t["setup_s"] == 6.5
    two = json.dumps({"n": 20, "fates": {"plane": 14}, "maps_s": 1.0,
                      "batches": [{"n": 10, "wall_s": 9.0}, {"n": 10, "wall_s": 4.0}]})
    assert demo.ce_timing(two)["warm_rate"] == 2.5 and demo.ce_timing(two)["setup_s"] == 6.0
    with pytest.raises(ValueError):
        demo.ce_timing(json.dumps({"n": 10, "fates": {"plane": 7}, "batches": [{"n": 10, "wall_s": 1.0}]}))


def test_gpu_probe(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(demo.subprocess, "run", lambda *a, **k: 1 / 0)
    assert demo.gpu_cores() is None
    monkeypatch.delenv("JAX_PLATFORMS")
    for stdout, cores in (("warning\ngpu 108\n", 108), ("cuda 0\n", 0), ("cpu 0\n", None), ("", None)):
        monkeypatch.setattr(demo.subprocess, "run", lambda *a, o=stdout, **k: subprocess.CompletedProcess(a, 0, o, ""))
        assert demo.gpu_cores() == cores, stdout


def test_ce_headline_ignores_other_output():
    out = 'jax: {"not": "it"}\n' + json.dumps({"n": 10, "fates": {"plane": 7}}, indent=1) + "\n"
    v, e, n = demo.headline("ce", Path("/unused"), out)
    assert (v, n) == (0.7, 10) and e > 0
    with pytest.raises(ValueError):
        demo.headline("ce", Path("/unused"), "no json here")


def test_only_checks_the_selected_chains_files(release, capsys):
    (release / "release" / demo.CRV_GEN).unlink()
    assert main(["--only", "beam"], fake()) == 0
    assert main(["--only", "crv"], fake()) == 2
    assert "demo skipped: missing" in capsys.readouterr().out


def test_skips_without_release(monkeypatch, capsys):
    monkeypatch.delenv("M2FS_INPUTS", raising=False)
    assert main([], fake()) == 2
    assert "demo skipped: M2FS_INPUTS is not set" in capsys.readouterr().out


def test_skips_when_a_file_is_missing(release, capsys):
    (release / "release" / demo.BEAM_TABLE).unlink()
    assert main([], fake()) == 2
    assert "demo skipped: missing" in capsys.readouterr().out


def test_commands_use_existing_entry_points(release):
    c = demo.commands("/r", Path("/o"), dict(demo.SAMPLES, ce_gpu=8, ce_cpu=4))
    assert c["beam"][1:4] == ["-m", "beamfastsim", "point"]
    for arm in ("ce_gpu", "ce_cpu"):
        assert c[arm][1:3] == ["-m", "beamfastsim.cechain"] and "--stops" in c[arm] and "--batch" in c[arm]
    assert c["crv"][1:4] == ["-m", "crvfastsim", "runchain"] and "--max-events" in c["crv"]
