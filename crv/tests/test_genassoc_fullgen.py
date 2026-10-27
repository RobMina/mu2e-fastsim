"""Local-only fullgen audit tests: synthetic TTrees and a fake art process; no build/remote IO."""
import json
from pathlib import Path

import numpy as np
import pytest
import uproot

from valhist import genassoc_fullgen as F


def save(path, value):
    path.write_text(json.dumps(value))


def rootfile(path, shard=1, strict=False, mutation=None):
    # Two census events, but only the first has a generator. Use large uint32 IDs.
    keys = {"run": np.array([1400, 1400], dtype="uint32"),
            "subrun": np.array([shard, shard], dtype="uint32"),
            "event": np.array([2**32 - 2, 2**32 - 1], dtype="uint32")}
    gen = {k: a[:1].copy() for k, a in keys.items()}
    gen.update(idx=np.array([0], dtype="int32"), pdg=np.array([13], dtype="int32"),
               e=np.array([100.]), t=np.array([0.]), px=np.array([3.]))
    subruns = {k: a[:1].copy() for k, a in keys.items() if k != "event"}
    subruns.update(livetime=np.array([1.25]), primaries=np.array([2.]))
    events = dict(keys)
    for k, v in dict(schema_version=2, associations_checked=0, full_generator_stream=1,
                     gen_product_id=101, sim_product_id=0).items():
        events[k] = np.full(2, v, dtype="uint32")
    events.update(n_gen=np.array([1, 0], dtype="int64"), n_particles=np.zeros(2, dtype="int64"),
                  n_steps=np.zeros(2, dtype="int64"))
    payload = {"gen": gen, "subruns": subruns}
    if strict:
        gen["gen_product_id"] = np.array([101], dtype="uint32")
        payload["events"] = events
    if mutation:
        mutation(payload)
    with uproot.recreate(path) as root:
        for tree, columns in payload.items():
            types = {c: a.dtype for c, a in columns.items()}
            if tree == "events":
                types["gen_tag"] = "string"
                columns = {**columns, "gen_tag": ["cosmicTimeOffset"] * len(columns["event"])}
            root.mktree("dump/" + tree, types)
            root["dump/" + tree].extend(columns)


def logtext(art):
    return f'Opened input file "{art}"\nArt has completed and will exit with status 0.\n'


def test_log_parsing_is_exact():
    art = F.ART_TEMPLATE.format(sequence="001400_00000002")
    assert F.parse_dump_log(logtext(art), art) == art
    F.parse_driver_log("other\nshard 1 steps_rc=0 gen_rc=0 10:09:09\n")
    for bad in ("", logtext(art).replace("status 0", "status 01"),
                logtext(art) + 'Opened input file "/pnfs/other.art"\n',
                logtext(art).replace("g4ref_v1/", "wrong/"),
                logtext(art) + "Art has completed and will exit with status -1\n"):
        with pytest.raises(ValueError, match="input/completion"):
            F.parse_dump_log(bad, art)
    for bad in ("shard 11 steps_rc=0 gen_rc=0", "shard 1 steps_rc=0 gen_rc=1",
                "shard 1 steps_rc=0 gen_rc=0\nshard 1 steps_rc=2 gen_rc=0"):
        with pytest.raises(ValueError, match="shard 1"):
            F.parse_driver_log(bad)


def test_exact_comparison_counts_empty_events(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "GENERATED_PER_FILE", 2)
    old, new = tmp_path / "old.root", tmp_path / "new.root"
    rootfile(old)
    rootfile(new, strict=True)
    report, keys = F.compare_payload(old, new)
    assert report["event_count"] == 2 and report["genrows"] == 1
    assert report["zero_generator_events"] == 1 and report["legacy_values_identical"]
    assert report["first_event"] == dict(run=1400, subrun=1, event=2**32 - 2)
    assert report["last_event"]["event"] == 2**32 - 1
    assert keys.dtype == np.int64
    monkeypatch.setattr(F, "GENERATED_PER_FILE", 50_000)
    with pytest.raises(ValueError, match="50000"):
        F.compare_payload(old, new)


@pytest.mark.parametrize("case", ["gen", "subruns", "missing_column", "wrong_role", "bad_census", "legacy"])
def test_comparison_failures(tmp_path, monkeypatch, case):
    monkeypatch.setattr(F, "GENERATED_PER_FILE", 2)
    old, new = tmp_path / "old.root", tmp_path / "new.root"
    rootfile(old)

    def mutate(payload):
        if case == "gen":
            payload["gen"]["px"][0] = np.nextafter(3., 4.)  # Not even an ulp may change.
        elif case == "subruns":
            payload["subruns"]["livetime"][0] = 1.5
        elif case == "missing_column":
            del payload["gen"]["px"]
        elif case == "wrong_role":
            payload["events"]["full_generator_stream"][:] = 0
        elif case == "bad_census":
            payload["events"]["n_gen"][1] = 1
    rootfile(new, strict=case != "legacy", mutation=mutate)
    with pytest.raises((ValueError, AssertionError)):
        F.compare_payload(old, new)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "GENERATED_PER_FILE", 2)
    old = tmp_path / "old"
    (old / "logs").mkdir(parents=True)
    paths = []
    for i in range(1, 21):
        seq = f"001400_{i:08d}"
        p = old / f"gen.{seq}.root"
        rootfile(p, i)
        paths.append(str(p))
        if i != 1:
            (old / f"logs/dumpgen.{seq}.log").write_text(logtext(F.ART_TEMPLATE.format(sequence=seq)))
    (old / "logs/driver.log").write_text("shard 1 steps_rc=0 gen_rc=0 10:09:09\n")
    area = tmp_path / "area"
    source = area / "CrvFastSim/src/CrvStepTableDumper_module.cc"
    source.parent.mkdir(parents=True)
    source.write_text("source fixture")
    plugin = area / "build/test/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b"plugin fixture")
    config = area / "CrvFastSim/fcl/dump_gen.fcl"
    config.parent.mkdir(parents=True)
    config.write_text('#include "CrvFastSim/fcl/dump_crvsteps.fcl"\n')
    config.with_name("dump_crvsteps.fcl").write_text("# fixture\n")
    metadata, plan, historical = [tmp_path / p for p in ("metadata.json", "plan.json", "historical.json")]
    save(metadata, {"inputs": [], "gen": paths[:10], "generated_scaled_to": 0})
    save(plan, {"source_metadata": str(metadata), "source_metadata_sha256": F.variants.sha256(metadata),
                "area": str(area), "gen": paths[:10], "gen_sha256": F._hashes(paths[:10]),
                "donors": [], "generated": 0, "generated_per_file": 500_000})
    save(historical, {"evaluation_gen": [{"path": p, "sha256": F.variants.sha256(p)} for p in paths[10:]]})
    monkeypatch.setenv("MUSE_WORK_DIR", str(area))
    monkeypatch.setattr(F.shutil, "which", lambda name: str(source))
    calls = []

    class FakeArt:
        def __init__(self, command, stdout, stderr, env, cwd):
            self.pid = 1234
            i = len(calls) + 1
            calls.append(command)
            assert "-n" not in command  # Read to EOF, not just an apparently successful prefix.
            assert env["LD_DEBUG"] == "libs" and cwd == area
            target = command[command.index("-T") + 1]
            rootfile(target, i, strict=True)
            stdout.write(logtext(command[command.index("-s") + 1]))
            Path(env["LD_DEBUG_OUTPUT"] + ".1234").write_text(f"calling init: {plugin}\n")

        def wait(self):
            return 0

    monkeypatch.setattr(F.subprocess, "Popen", FakeArt)
    return dict(plan=plan, historical=historical, paths=paths, old=old, calls=calls,
                out=tmp_path / "audit", plugin=plugin, source=source)


def run(f):
    return F.certify(f["plan"], f["historical"], f["out"])


def test_twenty_file_certificate_and_no_overwrite(campaign):
    f = campaign
    before = F._hashes(f["paths"])
    result = run(f)
    assert len(f["calls"]) == 20 and len(result["files"]) == 20
    assert result["event_count"] == 40 and result["cross_file_event_overlap"] == 0
    assert result["input_sha256_pre"] == result["input_sha256_post"]
    assert result["code_sha256_pre"] == result["code_sha256_post"]
    assert before == F._hashes(f["paths"])
    done = json.loads((f["out"] / "001400_00000001.done.json").read_text())
    assert "NOT certified" in done["source"]["source_evidence"]["status"]
    assert done["source"]["art"] == F.ART_TEMPLATE.format(sequence="001400_00000001")
    assert done["loaded_plugin"] == str(f["plugin"])
    assert F.variants.sha256(f["plugin"]) == done["code_sha256_post"][str(f["plugin"])]
    assert all(row["zero_generator_events"] == 1 for row in result["files"].values())
    with pytest.raises(FileExistsError):
        run(f)
    assert not (f["out"] / "failed.json").exists()


@pytest.mark.parametrize("case,match", [("old_hash", "historical old hash"),
    ("eval_paths", "derived paths"), ("missing_log", "dumpgen"), ("bad_first", "completion"),
    ("wrong_loader", "unexpected loaded dumper"), ("code_drift", "hash drift")])
def test_failures_preserve_attempts_without_certificate(campaign, monkeypatch, case, match):
    f = campaign
    if case in ("old_hash", "eval_paths"):
        data = json.loads(f["historical"].read_text())
        data["evaluation_gen"][0]["sha256" if case == "old_hash" else "path"] = "wrong"
        save(f["historical"], data)
    elif case == "missing_log":
        (f["old"] / "logs/dumpgen.001400_00000002.log").unlink()
    elif case == "bad_first":
        (f["old"] / "logs/dumpgen.001400_00000001.log").write_text("failed")
    else:
        original = F.subprocess.Popen

        def changed(*args, **kwargs):
            proc = original(*args, **kwargs)
            if case == "wrong_loader":
                Path(kwargs["env"]["LD_DEBUG_OUTPUT"] + ".1234").write_text("trying file=" + str(f["plugin"]))
            else:
                f["source"].write_text("drift")
            return proc
        monkeypatch.setattr(F.subprocess, "Popen", changed)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        run(f)
    assert (f["out"] / "failed.json").exists()
    assert not (f["out"] / "certificate.json").exists()
    if f["calls"]:
        assert (f["out"] / "fullgen.001400_00000001.root").exists()
        assert (f["out"] / "001400_00000001.attempt.json").exists()
    with pytest.raises(FileExistsError):
        run(f)


def test_dangling_output_is_not_followed(campaign):
    campaign["out"].symlink_to(campaign["out"].with_name("missing"))
    with pytest.raises(FileExistsError):
        run(campaign)
    assert not campaign["calls"]


def test_real_first_log_needs_no_driver(campaign):
    f = campaign
    (f["old"] / "logs/dumpgen.001400_00000001.log").write_text(
        logtext(F.ART_TEMPLATE.format(sequence="001400_00000001")))
    (f["old"] / "logs/driver.log").unlink()
    rows, bound = F.original_inputs(f["paths"])
    assert rows[0]["source_evidence"]["status"] == "certified historical dump log"
    assert len(bound) == 20


@pytest.mark.parametrize("case", ["art_failure", "input_drift", "overlap", "short_census"])
def test_runtime_failure_never_certifies(campaign, monkeypatch, case):
    f = campaign
    original = F.subprocess.Popen

    def changed(*args, **kwargs):
        proc = original(*args, **kwargs)
        if case == "art_failure":
            proc.wait = lambda: 7
        elif case == "input_drift":
            Path(f["paths"][0]).write_bytes(b"changed legacy input")
        elif case == "overlap" and len(f["calls"]) == 2:
            # Both legacy and v2 payloads still match; only the zero-row event overlaps.
            def mutate(payload):
                payload["events"]["subrun"][1] = 1
            command = args[0]
            rootfile(command[command.index("-T") + 1], 2, strict=True, mutation=mutate)
        elif case == "short_census":
            monkeypatch.setattr(F, "GENERATED_PER_FILE", 50_000)
        return proc
    monkeypatch.setattr(F.subprocess, "Popen", changed)
    with pytest.raises((ValueError, RuntimeError)):
        run(f)
    assert not (f["out"] / "certificate.json").exists()
    assert (f["out"] / "failed.json").exists()
    assert (f["out"] / "fullgen.001400_00000001.root").exists()


def test_older_sidecar_without_hashes_is_explicit(campaign):
    data = json.loads(campaign["historical"].read_text())
    for record in data["evaluation_gen"]:
        del record["sha256"]
    save(campaign["historical"], data)
    run(campaign)
    manifest = json.loads((campaign["out"] / "inputs.json").read_text())
    assert "no historical checksum" in manifest["old_hash_evidence"][campaign["paths"][10]]
