"""Synthetic saved-audit contracts with real file hashes; no ROOT or response execution."""
import dataclasses
import json
import math
from pathlib import Path
import shutil

import pytest

from crvfastsim.analyses import named_seed_audit as A


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def change(path, update):
    value = json.loads(Path(path).read_text())
    update(value)
    save(path, value)


def metric(events, cm, rm):
    supported = events >= 2 and rm > 0
    return dict(events=events, candidate_mean=cm, reference_mean=rm,
                relative=cm / rm - 1 if supported else None, relative_se=9000 if supported else None,
                status="diagnostic" if supported else "unsupported")


def ensemble(tmp_path, backends=("numpy",), seeds=(1, 2, 3), run="run2_v10"):
    rc = A.runconfig.get(run)
    reference = tmp_path / "reference.root"
    reference.write_bytes(b"fixed stock bytes")
    conditions = {"run_config": run, "config": dataclasses.asdict(rc)}
    conditions["config_sha256"] = A.NC._digest(conditions["config"])
    conditions = json.loads(json.dumps(conditions))
    manifest = tmp_path / "manifest.json"
    save(manifest, conditions)
    paths = []
    for backend in backends:
        for seed in seeds:
            candidate = tmp_path / f"{backend}-{seed}.root"
            candidate.write_bytes(f"synthetic {backend} output seed {seed}".encode())
            cfg = dataclasses.asdict(rc.chain_config(jax=backend != "numpy", fused_response=backend == "jax_fused"))
            cfg["window"] = list(A.runconfig.OFFSPILL_WINDOW)
            cfg = json.loads(json.dumps(cfg))
            meta = dict(schema="crv-fasttables-named-v1", run_config=run, chain_config=cfg,
                        seed=seed, stage="chain", noise=cfg["noise"], jax=cfg["jax"], window_applied=True,
                        configured_window_ns=list(rc.window), effective_window_ns=cfg["window"],
                        window_selection="offspill", conditions=conditions, finder=cfg["finder"],
                        timing_policy="fixed_absolute_crop_no_pbtime_rebase", onspill_equivalence=False,
                        geometry_readout=dict(policy="opt_in_counters_sipmsAtSide", present=19264, absent=2112,
                                              mask_sha256="a" * 64),
                        observer={"arbitrary_unsettled_shape": [seed, backend]})
            save(str(candidate) + ".metadata.json", meta)
            sectors = {}
            for sector in rc.sectors:
                sectors[sector] = {}
                for side in A.SIDES:
                    zero = sector == "R4" and side in ("1", "3")
                    rm, cm = (0., 0.) if zero else (10., 10. + seed + (backend != "numpy"))
                    metrics = {name: metric(4, cm, rm) for name in A.METRICS}
                    median = {"count": 0, "median": None} if zero else {"count": 2, "median": 3.}
                    sectors[sector][side] = dict(paired_event_metrics=metrics,
                                                seq0_pe_median={"candidate": median, "reference": median})
            audit = dict(schema=A.AUDIT_SCHEMA, status=A.AUDIT_STATUS, run_config=run, events=4,
                         event_identity_sha256="e" * 64, conditions_config_sha256=conditions["config_sha256"],
                         sectors=sectors, inputs={"candidate": A._record(candidate), "reference": A._record(reference),
                                                  "conditions_manifest": A._record(manifest)})
            path = tmp_path / f"{backend}-{seed}.audit.json"
            save(path, audit)
            paths.append(path)
    return paths


def meta_path(path):
    return Path(json.loads(path.read_text())["inputs"]["candidate"]["path"] + ".metadata.json")


def get_metric(audit, sector="T1", side="all", name="pulse_pe"):
    return audit["sectors"][sector][side]["paired_event_metrics"][name]


def reject(paths, tmp_path, match=None):
    out = tmp_path / "bad"
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        A.aggregate(paths, out)
    assert not out.exists()


def test_seed_spread_all_sectors_metrics_and_declared_backends(tmp_path):
    paths = ensemble(tmp_path, backends=("numpy", "jax_nonfused", "jax_fused"))
    result = A.aggregate(paths, tmp_path / "report")
    assert json.loads((tmp_path / "report/seed-audit.json").read_text()) == result
    assert set(result["backends"]) == {"numpy", "jax_nonfused", "jax_fused"}
    group = result["backends"]["numpy"]
    assert group["K"] == 3 and group["seeds"] == [1, 2, 3] and group["small_ensemble"]
    assert set(group["sectors"]) == set(A.runconfig.RUN2_SECTORS)
    for sector in group["sectors"].values():
        assert set(sector) == set(A.SIDES)
        for side in sector.values():
            assert set(side["metrics"]) == set(A.METRICS)
    metric = group["sectors"]["T1"]["all"]["metrics"]["pulse_pe"]
    rel = metric["relative"]
    assert rel["mean"] == pytest.approx(.2)
    assert rel["sample_sd"] == pytest.approx(.1)
    assert rel["se_mean"] == pytest.approx(.1 / math.sqrt(3))
    assert rel["min"] == pytest.approx(.1) and rel["max"] == pytest.approx(.3)
    assert metric["candidate_mean"] == dict(K=3, mean=12., sample_sd=1., se_mean=1 / math.sqrt(3), min=11., max=13.)
    assert result["backends"]["jax_fused"]["sectors"]["T1"]["all"]["metrics"]["pulse_pe"]["relative"]["mean"] == pytest.approx(.3)
    assert "jax" not in result["common_metadata"]["chain_config"]
    assert "fused_response" not in result["common_metadata"]["chain_config"]
    assert result["inputs"][0]["metadata"]["observer"] == {"arbitrary_unsettled_shape": [1, "numpy"]}
    forbidden = group["documented_forbidden_channels"]
    assert len(forbidden) == 2 and all(row["candidate_all_zero"] for row in forbidden)
    zero = forbidden[0]["metrics"]["pulse_pe"]
    assert zero["relative"] == dict(K=3, mean=None, sample_sd=None, se_mean=None, min=None, max=None)
    assert zero["reference_zero"] and zero["candidate_mean"]["mean"] == 0
    assert zero["candidate_mean"]["sample_sd"] == 0
    table = (tmp_path / "report/table.md").read_text()
    assert "+20.000 ± 10.000% (K=3)" in table
    assert all(f"| {sector} |" in table for sector in A.runconfig.RUN2_SECTORS)
    before = (tmp_path / "report/seed-audit.json").read_bytes()
    with pytest.raises(FileExistsError):
        A.aggregate(paths, tmp_path / "report")
    assert (tmp_path / "report/seed-audit.json").read_bytes() == before


def test_zero_reference_nonzero_candidate_is_retained(tmp_path):
    paths = ensemble(tmp_path)
    change(paths[1], lambda a: get_metric(a, "R4", "1").update(candidate_mean=3.))
    result = A.aggregate(paths, tmp_path / "report")
    row = result["backends"]["numpy"]["documented_forbidden_channels"][0]
    assert not row["candidate_all_zero"]
    m = row["metrics"]["pulse_pe"]
    assert m["relative"]["mean"] is None
    assert m["candidate_mean"]["mean"] == 1.
    assert m["candidate_mean"]["sample_sd"] == pytest.approx(math.sqrt(3))


def test_eight_seeds_still_small_and_cli(tmp_path, capsys):
    paths = ensemble(tmp_path, seeds=range(1, 9), run="run1a_v01")
    A.main(["--audits", *map(str, paths), "--outdir", str(tmp_path / "report")])
    result = json.loads((tmp_path / "report/seed-audit.json").read_text())
    assert result["backends"]["numpy"]["K"] == 8
    assert result["backends"]["numpy"]["small_ensemble"]
    assert result["backends"]["numpy"]["documented_forbidden_channels"] == []
    assert "conditional seed diagnostics" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["two", "one_short_backend", "seed", "path", "symlink", "hash"])
def test_duplicates_and_insufficient_seeds(tmp_path, mode):
    paths = ensemble(tmp_path, backends=("numpy", "jax_nonfused") if mode == "one_short_backend" else ("numpy",))
    if mode == "two":
        paths = paths[:2]
    elif mode == "one_short_backend":
        paths = paths[:5]
    elif mode == "seed":
        change(meta_path(paths[1]), lambda m: m.update(seed=1))
    else:
        a, b = (json.loads(p.read_text()) for p in paths[:2])
        if mode == "hash":
            shutil.copyfile(a["inputs"]["candidate"]["path"], b["inputs"]["candidate"]["path"])
            record = A._record(b["inputs"]["candidate"]["path"])
        elif mode == "symlink":
            alias = tmp_path / "alias.root"
            alias.symlink_to(a["inputs"]["candidate"]["path"])
            record = {**a["inputs"]["candidate"], "path": str(alias)}
        else:
            record = a["inputs"]["candidate"]
        change(paths[1], lambda a: a["inputs"].update(candidate=record))
    reject(paths, tmp_path, "three distinct seeds|duplicate")


@pytest.mark.parametrize("field,value", [("schema", "wrong"), ("status", "certified"), ("events", 5),
                                           ("events", True), ("event_identity_sha256", "f" * 64),
                                           ("event_identity_sha256", "bad"), ("conditions_config_sha256", "d" * 64),
                                           ("run_config", "run1a_v01")])
def test_identity_drift(tmp_path, field, value):
    paths = ensemble(tmp_path)
    change(paths[1], lambda a: a.update({field: value}))
    reject(paths, tmp_path)


@pytest.mark.parametrize("mode", ["missing_sector", "missing_side", "missing_metric", "extra_metric", "missing_relative",
                                   "wrong_ratio", "wrong_status", "wrong_events", "nonfinite", "string", "bool", "negative_se",
                                   "missing_median", "empty_median", "reference_drift", "zero_bad_relative"])
def test_malformed_metrics(tmp_path, mode):
    paths = ensemble(tmp_path)

    def mutate(a):
        m = get_metric(a)
        if mode == "missing_sector":
            a["sectors"].pop("C2")
        elif mode == "missing_side":
            a["sectors"]["T1"].pop("3")
        elif mode in ("missing_metric", "extra_metric"):
            metrics = a["sectors"]["T1"]["all"]["paired_event_metrics"]
            metrics.pop("pulse_pe") if mode == "missing_metric" else metrics.update(extra=m)
        elif mode == "missing_relative":
            m.pop("relative")
        elif mode == "wrong_ratio":
            m["relative"] = 42
        elif mode == "wrong_status":
            m["status"] = "unsupported"
        elif mode == "wrong_events":
            m["events"] = 3
        elif mode in ("nonfinite", "string", "bool"):
            m["candidate_mean"] = {"nonfinite": "NAN_SENTINEL", "string": "12", "bool": True}[mode]
        elif mode == "negative_se":
            m["relative_se"] = -1
        elif mode == "missing_median":
            a["sectors"]["T1"]["all"].pop("seq0_pe_median")
        elif mode == "empty_median":
            a["sectors"]["T1"]["all"]["seq0_pe_median"]["reference"] = {"count": 0, "median": 2.}
        elif mode == "reference_drift":
            m.update(reference_mean=20, relative=m["candidate_mean"] / 20 - 1)
        elif mode == "zero_bad_relative":
            get_metric(a, "R4", "1")["relative"] = 0

    change(paths[1], mutate)
    if mode == "nonfinite":
        paths[1].write_text(paths[1].read_text().replace('"NAN_SENTINEL"', "NaN"))
    reject(paths, tmp_path)


@pytest.mark.parametrize("mode", ["noise", "stage", "window", "mask", "missing_mask", "malformed_mask", "config",
                                   "missing_config", "missing_config_field", "nested_jax", "config_bool_alias", "diagnostic", "yield",
                                   "jax_mismatch", "fused_without_jax", "bool_seed", "conditions", "finder"])
def test_metadata_drift_and_mixed_mode(tmp_path, mode):
    paths = ensemble(tmp_path)

    def mutate(m):
        if mode == "noise":
            m["noise"] = m["chain_config"]["noise"] = False
        elif mode == "stage":
            m["stage"] = "reco"
        elif mode == "window":
            m["window_selection"] = "onspill"
            m["effective_window_ns"] = m["chain_config"]["window"] = list(A.runconfig.ONSPILL_WINDOW)
        elif mode == "mask":
            m["geometry_readout"]["mask_sha256"] = "b" * 64
        elif mode == "missing_mask":
            m.pop("geometry_readout")
        elif mode == "malformed_mask":
            m["geometry_readout"]["present"] = True
        elif mode == "config":
            m["chain_config"]["sipm"]["cross_talk"] = .876
        elif mode == "missing_config":
            m.pop("chain_config")
        elif mode == "missing_config_field":
            m["chain_config"].pop("rounds")
        elif mode == "nested_jax":
            m["chain_config"]["sipm"]["jax"] = True
        elif mode == "config_bool_alias":
            m["chain_config"]["background_scale"] = True
        elif mode == "diagnostic":
            m["diagnostic_override"] = None
        elif mode == "yield":
            m["chain_config"]["response"]["sectors"]["T5"]["yield_per_mev"] = 123
        elif mode == "jax_mismatch":
            m["jax"] = True
        elif mode == "fused_without_jax":
            m["chain_config"]["fused_response"] = True
        elif mode == "bool_seed":
            m["seed"] = True
        elif mode == "conditions":
            m["conditions"]["config"]["photon_yield_scale"] = .1
        elif mode == "finder":
            m["finder"]["add_opposite_side_pulses"] = False

    change(meta_path(paths[1]), mutate)
    reject(paths, tmp_path)


@pytest.mark.parametrize("mode", ["candidate_bytes", "candidate_size", "candidate_hash", "reference_bytes",
                                   "reference_path", "reference_size", "manifest_bytes", "missing_metadata"])
def test_input_integrity_and_frozen_reference(tmp_path, mode):
    paths = ensemble(tmp_path)
    a = json.loads(paths[1].read_text())
    if mode in ("candidate_bytes", "reference_bytes", "manifest_bytes"):
        key = {"candidate_bytes": "candidate", "reference_bytes": "reference", "manifest_bytes": "conditions_manifest"}[mode]
        Path(a["inputs"][key]["path"]).write_bytes(b"changed")
    elif mode == "reference_path":
        copied = tmp_path / "different-reference.root"
        shutil.copyfile(a["inputs"]["reference"]["path"], copied)
        change(paths[1], lambda a: a["inputs"].update(reference=A._record(copied)))
    elif mode in ("candidate_size", "reference_size"):
        key = mode.split("_")[0]
        change(paths[1], lambda a: a["inputs"][key].update(size=0))
    elif mode == "candidate_hash":
        change(paths[1], lambda a: a["inputs"]["candidate"].update(sha256="0" * 64))
    else:
        meta_path(paths[1]).unlink()
    reject(paths, tmp_path)


def test_relative_paths_and_resolved_reference_alias(tmp_path):
    paths = ensemble(tmp_path)
    alias = tmp_path / "reference-alias.root"
    alias.symlink_to(tmp_path / "reference.root")
    for path in paths:
        change(path, lambda a: [r.update(path=Path(r["path"]).name) for r in a["inputs"].values()])
    change(paths[1], lambda a: a["inputs"]["reference"].update(path=alias.name))
    result = A.aggregate(paths, tmp_path / "report")
    assert result["reference"]["path"] == str(tmp_path / "reference.root")


def test_changed_during_validation_fails_before_output(monkeypatch, tmp_path):
    paths = ensemble(tmp_path)
    original = A._aggregate

    def changed(rows):
        result = original(rows)
        paths[0].write_text("{}")
        return result

    monkeypatch.setattr(A, "_aggregate", changed)
    reject(paths, tmp_path, "input changed")


def test_duplicate_json_and_output_symlink(tmp_path):
    paths = ensemble(tmp_path)
    paths[0].write_text('{"schema":"a", "schema":"b"}')
    reject(paths, tmp_path, "duplicate JSON key")
    out = tmp_path / "output-link"
    out.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        A.aggregate(paths, out)
    assert not (tmp_path / "missing").exists()
