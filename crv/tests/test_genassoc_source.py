"""Reference source preflight with tiny real TTrees; no art or remote operations."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import uproot

from valhist import genassoc_campaign as C, genassoc_source as S
from valhist.library_variants import sha256

pytestmark = pytest.mark.usefixtures("dcache_site")


def save(path, value):
    path.write_text(json.dumps(value))


def rootfile(path, keys, tree="gen", strict=False, donor=False, floating=False, empty_last=False):
    keys = np.asarray(keys, dtype=np.uint32 if strict else np.int64)
    cols = {name: keys[:, index] for index, name in enumerate(("run", "subrun", "event"))}
    if floating:
        cols["event"] = cols["event"].astype(float)
    if tree == "gen":
        cols.update(idx=np.zeros(len(keys), dtype=np.int32), pdg=np.full(len(keys), 11, dtype=np.int32),
                    e=np.full(len(keys), 10.), t=np.zeros(len(keys)))
    with uproot.recreate(path) as root:
        if strict:
            cols["gen_product_id"] = np.full(len(keys), 101, dtype=np.uint32)
            events = {name: cols[name] for name in ("run", "subrun", "event")}
            for name, value in dict(schema_version=2, associations_checked=int(donor),
                                    full_generator_stream=int(not donor), gen_product_id=101,
                                    sim_product_id=202 if donor else 0).items():
                events[name] = np.full(len(keys), value, dtype=np.uint32)
            for name, value in dict(n_gen=1, n_particles=0, n_steps=0).items():
                events[name] = np.full(len(keys), value, dtype=np.int64)
            if empty_last:
                events["n_gen"][-1] = 0
                cols = {c: a[:-1] for c, a in cols.items()}
            root.mktree("dump/events", {c: a.dtype for c, a in events.items()})
            root["dump/events"].extend(events)
        root.mktree("dump/" + tree, {c: a.dtype for c, a in cols.items()})
        root["dump/" + tree].extend(cols)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    # Only fixture sizes are reduced. The CLI has no exposure/count override.
    monkeypatch.setattr(S, "GENERATED_PER_FILE", 2)
    old = tmp_path / "g4ref_v1_tables"
    (old / "logs").mkdir(parents=True)
    donors, gen = [], []
    for index in range(1, 11):
        seq = f"001400_{index:08d}"
        donor, full = old / f"crvsteps.{seq}.root", tmp_path / f"fullgen.{index}.root"
        keys = [(2**32 - 1, index, 2**53 + 1), (2**32 - 1, index, 2**53 + 2)]
        rootfile(donor, keys[:1], "particles")
        rootfile(full, keys)
        donors.append(str(donor))
        gen.append(str(full))
        if index != 1:
            (old / f"logs/dump.{seq}.log").write_text(
                f'Opened input file "{S.art_template().format(sequence=seq)}"\n'
                'Art has completed and will exit with status 0.\n')
    (old / "logs/driver.log").write_text("shard 1 steps_rc=0 gen_rc=0 10:09:09\n")
    evaluation = tmp_path / "eval.root"
    rootfile(evaluation, [(2**32 - 1, 100, 2**53 + 1), (2**32 - 1, 100, 2**53 + 2)])
    prod_donors = []
    for index in range(20):
        p = tmp_path / f"production.{index}.root"
        rootfile(p, [(17, index, 2**53 + 1)], "particles")
        prod_donors.append({"old_table": str(p), "old_sha256": sha256(p)})
    prod_source, prod_plan = tmp_path / "production_metadata.json", tmp_path / "production_plan.json"
    save(prod_source, {"inputs": [r["old_table"] for r in prod_donors], "gen": gen,
                       "generated_scaled_to": 10_000_000})
    save(prod_plan, {"source_metadata": str(prod_source), "source_metadata_sha256": sha256(prod_source),
                     "area": str(tmp_path / "art"), "generated": 10_000_000,
                     "generated_per_file": 500_000, "donors": prod_donors, "gen": gen,
                     "gen_sha256": {p: sha256(p) for p in gen}})
    historical = tmp_path / "lib_shower_g4train.json"
    save(historical, {"training": [{"path": p} for p in donors],
                      "training_gen": [{"path": p} for p in gen],
                      "evaluation_gen": [{"path": str(evaluation)}],
                      "training_generated_events": 20, "evaluation_generated_events": 2})
    return dict(production=prod_plan, historical=historical, donors=donors, gen=gen,
                evaluation=evaluation, old=old, out=tmp_path / "source_train_strict_v2")


def run(f):
    return S.plan(f["production"], f["historical"], f["out"])


def change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    save(path, value)


def refresh_gen(f):
    change_json(f["production"], lambda p: p.update(gen_sha256={g: sha256(g) for g in f["gen"]}))


def test_plan_loads_in_existing_campaign_and_preserves_inputs(campaign):
    f = campaign
    before = {str(p): sha256(p) for p in [f["historical"], f["production"], *f["donors"], *f["gen"]]}
    result = run(f)
    out, loaded = C._load_plan(f["out"] / "plan.json")
    assert result == loaded and out == f["out"]
    assert result["generated"] == 20 and result["generated_per_file"] == 2
    assert result["area"] == json.loads(f["production"].read_text())["area"]
    assert result["gen_sha256"] == json.loads(f["production"].read_text())["gen_sha256"]
    metadata = json.loads(Path(result["source_metadata"]).read_text())
    assert metadata["inputs"] == f["donors"] and metadata["gen"] == f["gen"]
    assert metadata["provenance"] == result["provenance"]
    evidence = result["donors"][0]["source_evidence"]
    assert "NOT certified" in evidence["status"]
    assert result["donors"][0]["old_log"] is None
    assert result["donors"][0]["art"] == S.art_template().format(sequence="001400_00000001")
    assert len(evidence["logs"]) == 2
    for record in evidence["logs"]:
        assert record["sha256"] == sha256(record["path"])
    assert all("unverified legacy" in r["generator_validation"]["stream_role"]
               for r in result["provenance"]["training_gen"])
    assert result["provenance"]["certification"]["training_persisted_particle_events"] == 10
    assert len((out / "donors.tsv").read_text().splitlines()) == 10
    assert all((out / d).is_dir() for d in ("donors", "logs", "comparisons", "library", "validation"))
    assert before == {p: sha256(p) for p in before}
    with pytest.raises(FileExistsError):
        run(f)
    Path(result["source_metadata"]).write_text("{}")
    with pytest.raises(ValueError, match="metadata drift"):
        C._load_plan(out / "plan.json")


def test_real_first_log_is_certified_not_derived(campaign):
    f = campaign
    (f["old"] / "logs/dump.001400_00000001.log").write_text(
        f'Opened input file "{S.art_template().format(sequence="001400_00000001")}"\n'
        'Art has completed and will exit with status 0.\n')
    (f["old"] / "logs/driver.log").unlink()
    assert run(f)["donors"][0]["source_evidence"]["status"] == "certified historical dump log"


@pytest.mark.parametrize("change,match", [
    (lambda s: s.pop("evaluation_gen"), "missing historical"),
    (lambda s: s["training_gen"].pop(), "ten reference"),
    (lambda s: s.update(training_generated_events=19), "exposure"),
    (lambda s: s.update(training_generated_events=20.0), "integer"),
    (lambda s: s.update(evaluation_generated_events=3), "evaluation generated exposure"),
    (lambda s: s["training"][0].update(path="relative.root"), "absolute"),
    (lambda s: s["training_gen"].__setitem__(0, s["training_gen"][1]), "duplicate"),
    (lambda s: s["training_gen"][0].update(path=s["evaluation_gen"][0]["path"]), "bank must match"),
    (lambda s: s["training"][0].update(unique_events=123), "unique_events mismatch"),
    (lambda s: s["training_gen"][0].update(event_keys_sha256="wrong"), "event_keys_sha256 mismatch"),
    (lambda s: s.update(input_sha256={}), "incomplete historical"),
])
def test_bad_source_contract(campaign, change, match):
    change_json(campaign["historical"], change)
    with pytest.raises(ValueError, match=match):
        run(campaign)
    assert not campaign["out"].exists()


@pytest.mark.parametrize("case,match", [
    ("missing_other", "missing historical donor log"),
    ("missing_driver", "driver.log"),
    ("driver_failure", "shard 1"),
    ("wrong_path", "unexpected original reference art path"),
    ("incomplete", "completion"),
    ("ambiguous", "completion"),
    ("bad_first", "completion"),
    ("wrong_exit", "completion"),
])
def test_log_failures_never_fall_back(campaign, case, match):
    f = campaign
    log = f["old"] / "logs/dump.001400_00000002.log"
    if case == "missing_other":
        log.unlink()
    elif case == "missing_driver":
        (f["old"] / "logs/driver.log").unlink()
    elif case == "driver_failure":
        (f["old"] / "logs/driver.log").write_text("shard 1 steps_rc=1 gen_rc=0\n")
    elif case == "wrong_path":
        log.write_text(log.read_text().replace("g4ref_v1/", "wrong/"))
    elif case == "incomplete":
        log.write_text(log.read_text().splitlines()[0])
    elif case == "ambiguous":
        log.write_text(log.read_text() + 'Opened input file "/dcache-test/other.art"\n')
    elif case == "bad_first":
        (f["old"] / "logs/dump.001400_00000001.log").write_text("failed\n")
    else:
        log.write_text(log.read_text().replace("status 0", "status 1"))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        run(f)
    assert not f["out"].exists()


@pytest.mark.parametrize("case,match", [
    ("train_eval", "training/evaluation full-generator event overlap"),
    ("donor_outside", "outside corresponding"),
    ("gen_overlap", "cross-file event overlap"),
    ("gen_exposure", "per-file exposure"),
    ("float", "must be integer"),
    ("negative", "invalid integer event identity"),
    ("production_overlap", "training/production accepted donor event overlap"),
    ("wrong_role", "associations_checked"),
])
def test_identity_exposure_and_role_guards(campaign, case, match):
    f = campaign
    keys = [(2**32 - 1, 1, 2**53 + 1), (2**32 - 1, 1, 2**53 + 2)]
    if case == "train_eval":
        rootfile(f["evaluation"], keys)
    elif case == "donor_outside":
        rootfile(f["donors"][0], [(1, 1, 1)], "particles")
    elif case == "gen_overlap":
        rootfile(f["gen"][1], keys)
        refresh_gen(f)
    elif case == "gen_exposure":
        rootfile(f["gen"][0], keys[:1])
        refresh_gen(f)
    elif case == "float":
        rootfile(f["gen"][0], keys, floating=True)
        refresh_gen(f)
    elif case == "negative":
        rootfile(f["evaluation"], [(-1, 1, 1)])
    elif case == "production_overlap":
        p = json.loads(f["production"].read_text())["donors"][0]["old_table"]
        rootfile(p, keys[:1], "particles")
        change_json(f["production"], lambda plan: plan["donors"][0].update(old_sha256=sha256(p)))
    else:
        rootfile(f["gen"][0], [(1, 1, 1), (1, 1, 2)], strict=True, donor=True)
        refresh_gen(f)
    with pytest.raises(ValueError, match=match):
        run(f)
    assert not f["out"].exists()


def test_schema_v2_fullgen_census_counts_empty_events(campaign):
    f = campaign
    keys = [(2**32 - 1, 1, 1), (2**32 - 1, 1, 2)]
    rootfile(f["gen"][0], keys, strict=True, empty_last=True)
    rootfile(f["donors"][0], keys[:1], "particles")
    refresh_gen(f)
    record = run(f)["provenance"]["training_gen"][0]
    assert record["events"]["tree"] == "dump/events"
    assert record["events"]["unique_events"] == 2
    assert record["generator_validation"]["counts"]["gen"] == 1
    assert record["generator_validation"]["stream_role"] == "full generator stream"


def test_production_hash_manifest_must_cover_whole_bank(campaign):
    change_json(campaign["production"], lambda p: p["gen_sha256"].pop(campaign["gen"][0]))
    with pytest.raises(ValueError, match="bank must match"):
        run(campaign)
    assert not campaign["out"].exists()


def test_missing_generator_fails_before_output(campaign):
    Path(campaign["gen"][0]).unlink()
    with pytest.raises(FileNotFoundError):
        run(campaign)
    assert not campaign["out"].exists()


def test_full_integer_ids_do_not_alias(campaign):
    # Adjacent >2**53 events and common event numbers in other runs/subruns are distinct.
    rootfile(campaign["evaluation"], [(2**32 - 1, 1, 2**53), (2**32 - 2, 1, 2**53 + 1)])
    assert run(campaign)["provenance"]["certification"]["train_eval_overlap"] == 0


@pytest.mark.parametrize("target,match", [("gen", "denominator drift"), ("historical", "hash drift"),
                                         ("production", "hash drift")])
def test_input_hash_errors(campaign, target, match):
    f = campaign
    if target == "gen":
        Path(f["gen"][0]).write_bytes(b"changed")
    elif target == "historical":
        change_json(f["historical"], lambda s: s.update(input_sha256={
            p: "bad" if p == f["donors"][0] else sha256(p)
            for p in [*f["donors"], *f["gen"], str(f["evaluation"])]}))
    else:
        change_json(f["production"], lambda p: p["donors"][0].update(old_sha256="bad"))
    with pytest.raises(ValueError, match=match):
        run(f)
    assert not f["out"].exists()


def test_cli_plan_and_denominator_drift_guard(campaign, monkeypatch):
    f = campaign
    monkeypatch.setattr(sys, "argv", ["genassoc_source", "plan", "--production-plan", str(f["production"]),
                                     "--historical-source", str(f["historical"]), "--outdir", str(f["out"])])
    S.main()
    Path(f["gen"][0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="denominator drift"):
        C._load_plan(f["out"] / "plan.json")


def test_refuses_dangling_output_symlink(campaign):
    campaign["out"].symlink_to(campaign["out"].with_name("missing"))
    with pytest.raises(FileExistsError):
        run(campaign)
