"""Synthetic completed transactions; prior physics verification is explicitly fixture-substituted."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import pilot_membership as M
from test_veto_streaming_pilot_report import archive
from test_veto_pilot_coincidence import geometry as full_geometry


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def saved(archive, tmp_path, monkeypatch):
    source, _, _, summary, _ = archive
    source_root = tmp_path / "inputs"
    source_root.mkdir()
    pilot = source_root / "pilot"
    shutil.copytree(source, pilot)
    counters = source_root / "counters.txt"
    counters.write_text("synthetic counters")
    monkeypatch.setattr(M.S, "one_asset", lambda *a: counters)
    monkeypatch.setattr(M, "ignore_bindings", lambda: {})
    monkeypatch.setattr(M.S.M.P, "COUNTERS_SHA", M.S.C.sha256(counters))
    total = sum(p.stat().st_size for child in pilot.iterdir() for p in child.iterdir())
    v = {k: summary[k] for k in ("hashes", "seeds", "arms", "registry_sha256", "registry_frames", "frames", "batches")}
    v.update(schema=M.T.SCHEMA, summary_sha256=M.S.C.sha256(pilot / "run/summary.json"),
             code_sha256=M.S.C.sha256(M.T.__file__), persisted_path_bytes=total,
             F=None, finite_library_uncertainty=None)
    for k in ("saved_artifact_protocol_exact", "source_admission_exact", "typed_source_fingerprints_exact",
              "photon_timing_fields_exact", "authoritative_all_live_conditions_exact", "whole_selected_frames_exact",
              "all_declared_arm_seeds_complete"):
        v[k] = True
    for k in ("execution_admitted", "full_retained_campaign_admitted", "timing_closure", "regenerated_optics", "regenerated_response"):
        v[k] = False
    verified = source_root / "verified.json"
    dump(verified, v)
    selection = source_root / "selection.parquet"
    ak.to_parquet(ak.Array({"t_mid": [[700., 800.], []], "p_mid": [[100., 110.], []]}), selection)
    reference = source_root / "reference.json"
    card = M.card.VetoCard(dt_lo=0., dt_hi=150.)
    dump(reference, dict(card=asdict(card) | {"sha": card.sha()}, selected=2, pmin=85., pmax=200., F={"grid_ns": .25}))
    paired = source_root / "paired.json"
    dump(paired, dict(selection=str(selection), selection_sha256=M.S.C.sha256(selection),
                      reference=str(reference), reference_sha256=M.S.C.sha256(reference)))
    hashes = dict(expected_pilot_sha256=v["summary_sha256"], expected_verified_sha256=M.S.C.sha256(verified),
                  expected_paired_sha256=M.S.C.sha256(paired))
    geom = full_geometry()
    for name in ("sector", "layer", "counter_in_type", "x", "y", "pos"):
        setattr(geom, name, getattr(geom, name)[:1])
    monkeypatch.setattr(M, "geometry_only", lambda *a: geom)

    def forbidden(*a, **k):
        pytest.fail("saved membership attempted upstream admission or physics regeneration")

    for module, names in ((M.T, ("verify",)), (M.S, ("admission", "resources", "run")),
                          (M.S.I, ("load_input",)), (M.S.O, ("make_batch",)), (M.S.O.K, ("make_photons",)),
                          (M.S.R, ("run_arm",)), (M.S.R.P, ("reconstruct", "reconstruct_matrix")),
                          (M.S.R.W, ("synthesize",)), (M.S.R.S, ("simulate_candidates",))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    return dict(pilot=pilot, verified=verified, paired=paired, out=tmp_path / "result", **hashes)


def admission(args):
    return M.admit(**{k: v for k, v in args.items() if k != "out"})


def test_saved_only_transaction_integration(saved):
    d = admission(saved)
    assert len(d["selected"]) == 2
    result = M.run(**saved)
    assert result["unique_frames"] == 2 and result["ledger_rows"] == 16
    assert len(result["counts"]) == 8 and len(result["comparisons"]) == 6
    assert result["F"] is None and not result["stock_rerun"] and not result["inherited_raw_admission_repeated"]
    assert all(c["coverage_exact"] for c in result["comparisons"].values())
    assert all(c["native_links_exact"] for c in result["comparisons"].values())
    ledger = pd.read_parquet(saved["out"] / "coverage.parquet")
    assert len(ledger) == 16 and ledger.n_coincidences.eq(0).all()
    assert ledger.track_grid_coverage.eq(0).all() and ledger.gate_pass.all()
    assert result["persisted_path_bytes"] <= M.OUTPUT_CAP
    assert (saved["out"] / M.A.COMPLETE).exists()
    with pytest.raises(ValueError, match="new output"):
        M.run(**saved)


@pytest.mark.parametrize("name", ["pilot", "verified", "paired"])
def test_external_hash_required(saved, name):
    saved["expected_" + name + "_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed bound"):
        admission(saved)


@pytest.mark.parametrize("kind", ["child", "pulse", "top", "extra", "failed"])
def test_changed_saved_transaction_rejected(saved, kind):
    summary = M.SR._json(saved["pilot"] / "run/summary.json")
    child = saved["pilot"] / summary["records"][0]["path"]
    path = {"child": child / M.A.COMPLETE, "pulse": child / "historical_pulses.parquet",
            "top": saved["pilot"] / "run/selected.parquet", "extra": child / "extra.txt", "failed": child / M.A.FAILED}[kind]
    path.write_bytes(b"tampered")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        admission(saved)


@pytest.mark.parametrize("kind", ["summary", "code", "scope", "gate", "bytes", "domain"])
def test_pinned_verification_must_agree(saved, kind):
    v = M.SR._json(saved["verified"])
    if kind == "summary":
        v["summary_sha256"] = "0" * 64
    elif kind == "code":
        v["code_sha256"] = "0" * 64
    elif kind == "scope":
        v["execution_admitted"] = True
    elif kind == "gate":
        v["whole_selected_frames_exact"] = False
    elif kind == "bytes":
        v["persisted_path_bytes"] += 1
    else:
        v["frames"] += 1
    dump(saved["verified"], v)
    saved["expected_verified_sha256"] = M.S.C.sha256(saved["verified"])
    with pytest.raises(ValueError):
        admission(saved)


@pytest.mark.parametrize("name", ["selection", "reference"])
def test_same_count_does_not_replace_selection_binding(saved, name):
    pair = M.SR._json(saved["paired"])
    Path(pair[name]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed bound"):
        admission(saved)


@pytest.mark.parametrize("where", ["pilot", "verification", "selection", "symlink"])
def test_output_cannot_enter_frozen_evidence(saved, where):
    if where == "pilot":
        saved["out"] = saved["pilot"] / "new"
    elif where in ("verification", "selection"):
        saved["out"] = saved["verified"].parent / "new"
    else:
        saved["out"].symlink_to(saved["pilot"], target_is_directory=True)
    with pytest.raises(ValueError):
        M.run(**saved)


def test_incomplete_arm_seed_ledger(saved):
    d = admission(saved)
    with pytest.raises(ValueError, match="incomplete"):
        M.validate_ledger(d["saved_ledger"].iloc[:-1], d)
    ledger = d["saved_ledger"].copy()
    ledger.loc[0, "event"] += 1
    with pytest.raises(AssertionError):
        M.validate_ledger(ledger, d)


def test_late_input_change_never_complete(saved, monkeypatch):
    original = M.P.compare
    def mutate(*a):
        result = original(*a)
        saved["verified"].write_bytes(b"late mutation")
        return result
    monkeypatch.setattr(M.P, "compare", mutate)
    with pytest.raises(ValueError, match="changed bound"):
        M.run(**saved)
    assert not (saved["out"] / M.A.COMPLETE).exists()
    assert (saved["out"] / M.A.FAILED).exists()


def test_output_cap_no_truncation(saved, monkeypatch):
    monkeypatch.setattr(M, "OUTPUT_CAP", 10000)
    with pytest.raises(M.A.ByteLimitExceeded):
        M.run(**saved)
    assert not (saved["out"] / M.A.COMPLETE).exists()


def test_selection_header_guard_before_decode(saved, monkeypatch):
    d = admission(saved)
    monkeypatch.setattr(M, "SELECTION_CAP", 1)
    monkeypatch.setattr(ak, "from_parquet", lambda *a, **k: pytest.fail("premature selection decode"))
    with pytest.raises(ValueError, match="byte cap"):
        M.load_grid(d)


@pytest.mark.parametrize("kind", ["extra", "failed", "top", "root"])
def test_late_new_input_entries_never_complete(saved, monkeypatch, kind):
    original = M.P.compare
    summary = M.SR._json(saved["pilot"] / "run/summary.json")
    child = saved["pilot"] / summary["records"][0]["path"]
    path = {"extra": child / "new.txt", "failed": child / M.A.FAILED,
            "top": saved["pilot"] / "run/new.txt", "root": saved["pilot"] / "new.txt"}[kind]
    def mutate(*a):
        result = original(*a)
        path.write_bytes(b"late")
        return result
    monkeypatch.setattr(M.P, "compare", mutate)
    with pytest.raises(ValueError):
        M.run(**saved)
    assert not (saved["out"] / M.A.COMPLETE).exists()


def test_admission_does_not_adopt_new_pulse_hash(saved, monkeypatch):
    original = M.SR._batch
    def mutate(directory, *args):
        result = original(directory, *args)
        if directory.name.startswith("batch"):
            (directory / "historical_pulses.parquet").write_bytes(b"changed after receipt check")
        return result
    monkeypatch.setattr(M.SR, "_batch", mutate)
    with pytest.raises(ValueError):
        admission(saved)


@pytest.mark.parametrize("field", ["F", "finite_library_uncertainty"])
def test_missing_null_scope_is_not_success(saved, field):
    v = M.SR._json(saved["verified"])
    del v[field]
    dump(saved["verified"], v)
    saved["expected_verified_sha256"] = M.S.C.sha256(saved["verified"])
    with pytest.raises(ValueError, match="incomplete prior report"):
        admission(saved)


@pytest.mark.parametrize("counts", [[513], [2049], [-1], [1., 2.], [1, 1, 1, 1, 1]])
def test_pulse_counts_guard_before_decode(counts):
    with pytest.raises(ValueError):
        M.pulse_admission(np.asarray(counts))


def test_saved_pulse_row_cap_before_read(monkeypatch):
    monkeypatch.setattr(M, "_read", lambda *a: pytest.fail("decoded oversized pulses"))
    with pytest.raises(ValueError, match="row cap"):
        M.saved_pulses(Path("missing"), 2049)


@pytest.mark.parametrize("case", ["order", "waveform", "flags"])
def test_collection_order_bridge_must_be_native(tmp_path, case):
    p = pd.DataFrame(dict(frame_id=[0, 0], wf=[0, 1], seq=[0, 0], chan=[0, 4], starttdc=[20, 20],
                          failed=[False, True], zerondf=[False, False], channel_ignored=[False, False]))
    if case == "order":
        p = p.iloc[::-1].reset_index(drop=True)
    elif case == "waveform":
        p["wf"] = [0, 0]
    else:
        p["failed"] = [0, 1]
    path = tmp_path / "p.parquet"
    p.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        M.saved_pulses(path, 2)


def test_json_object_and_byte_caps(tmp_path):
    path = tmp_path / "j.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        M.read_json(path)
    with pytest.raises(ValueError, match="byte cap"):
        M.read_json(path, 1)


def test_nonzero_native_membership_through_persisted_runner(tmp_path, monkeypatch):
    from test_veto_pilot_coincidence import domain
    p, registry, selected, calibration, geom = domain()
    pilot = tmp_path / "source"
    pilot.mkdir()
    p = p.sort_values(["frame_id", "chan", "starttdc", "seq"]).reset_index(drop=True)
    p["wf"] = np.arange(len(p), dtype=np.int64)
    p["failed"], p["zerondf"] = False, False
    records, ledgers, domains, bindings = [], [], {}, {}
    for seed in M.S.SEEDS:
        tag = f"batch0000_seed{seed}"
        child = pilot / tag
        child.mkdir()
        records.append(dict(path=tag, batch=0, seed=seed))
        domains[tag] = selected
        for arm in M.S.ARMS:
            path = child / f"{arm}_pulses.parquet"
            p.to_parquet(path, index=False)
            bindings[str(path)] = M.S.C.sha256(path)
            ledger = selected.assign(batch=0, optical_seed=np.uint64(seed), response_seed=np.uint64(seed), arm=arm)
            ledger["pulses"] = ledger.frame_id.map(p.groupby("frame_id").size()).fillna(0).astype(np.int64)
            ledgers.append(ledger)
    track_dir = tmp_path / "tracks"
    track_dir.mkdir()
    for name in ("selected", "reference"):
        path = track_dir / (name + ".json")
        path.write_text("{}")
    d = dict(registry=registry, selected=selected, calibration=calibration, domains=domains, inputs=bindings,
             saved_ledger=pd.concat(ledgers), summary=dict(records=records, registry_sha256=M.F.digest(registry),
             plan="/absent/plan", cache_audit="/absent/cache", response="/absent/response"),
             pair={n: str(track_dir / (n + ".json")) for n in ("selection", "reference")})
    d["pair"]["selection"] = str(track_dir / "selected.json")
    monkeypatch.setattr(M, "admit", lambda *a, **k: d)
    monkeypatch.setattr(M, "recheck", lambda *a: None)
    monkeypatch.setattr(M, "geometry_only", lambda *a: geom)
    card = M.card.VetoCard(dt_lo=0., dt_hi=150., gate_end_hi=850., min_pe=35.)
    ref = dict(card=asdict(card) | {"sha": card.sha()}, selected=3, pmin=85., pmax=200., F={"grid_ns": 1.})
    grid = M.G.frozen_grid(ref, ak.Array(dict(t_mid=[[720., 750., 800.]], p_mid=[[100., 100., 100.]])))
    monkeypatch.setattr(M, "load_grid", lambda *a: grid)
    out = tmp_path / "result"
    result = M.run(pilot, tmp_path / "verify" / "v.json", tmp_path / "paired" / "p.json", out,
                   expected_pilot_sha256="a" * 64, expected_verified_sha256="b" * 64, expected_paired_sha256="c" * 64)
    assert result["unique_frames"] == 4 and result["ledger_rows"] == 32
    ledger = pd.read_parquet(out / "coverage.parquet")
    assert ledger.n_coincidences.sum() == 24
    assert ledger.n_veto_eligible.sum() == 8
    assert (~ledger.gate_pass).sum() == 8
    assert (ledger.track_grid_coverage > 0).sum() == 16
    assert pd.read_parquet(out / "members.parquet").shape[0] == 80
    assert all(c["exact"] and c["coverage_exact"] for c in result["comparisons"].values())


def test_jagged_leaf_limit_precedes_track_decode(saved, monkeypatch):
    from types import SimpleNamespace
    d = admission(saved)
    leaf = SimpleNamespace(physical_type="DOUBLE", max_repetition_level=1, path="t_mid.list.item")
    group = SimpleNamespace(total_byte_size=16, column=lambda i: SimpleNamespace(num_values=3_000_001))
    meta = SimpleNamespace(num_rows=1, num_row_groups=1, num_columns=1, row_group=lambda i: group)
    fake = SimpleNamespace(metadata=meta, schema=SimpleNamespace(column=lambda i: leaf))
    monkeypatch.setattr(M.pq, "ParquetFile", lambda *a: fake)
    monkeypatch.setattr(ak, "from_parquet", lambda *a, **k: pytest.fail("premature jagged decode"))
    with pytest.raises(ValueError, match="jagged value cap"):
        M.load_grid(d)


@pytest.mark.parametrize("value", ["true", "false", "true\nignoreChannels: true", "TRUE"])
def test_reviewed_ignore_policy_binding(tmp_path, monkeypatch, value):
    source = tmp_path / "source.cc"
    source.write_text("stock source")
    config = tmp_path / "effective_runtime.fcl"
    config.write_text("ignoreChannels: " + value + "\n")
    monkeypatch.setattr(M, "IGNORE_SOURCE", source)
    monkeypatch.setattr(M, "IGNORE_SOURCE_SHA", M.S.C.sha256(source))
    monkeypatch.setattr(M, "IGNORE_CONFIGS", {config: M.S.C.sha256(config)})
    if value == "true":
        assert M.ignore_bindings() == {str(source): M.S.C.sha256(source), str(config): M.S.C.sha256(config)}
    else:
        with pytest.raises(ValueError, match="ignoreChannels contract"):
            M.ignore_bindings()
