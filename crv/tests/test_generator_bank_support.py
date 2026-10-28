"""Synthetic local-only post-capture gate tests; never run the live campaign."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from valhist import generator_bank_support as S


def completed():
    return {"schema": S.G.SCHEMA, "scope": S.G.SCOPE, "assumptions": S.G.ASSUMPTIONS.copy(),
            "limits": S.G.LIMITS.copy(), "conditional_capture_evidence_complete": True,
            "strict_event_census_certified": True, "observed_producer_pairs_match": True,
            "bank_admitted_for_propagation": False, "downstream_authorized": False,
            "selection_independence_certified": False, "files": 94, "events": 94_000_000,
            "input_sha256": {"/data/evidence": "a" * 64},
            "input_sha256_post": {"/data/evidence": "a" * 64}}


def cells(rows):
    return pd.DataFrame(rows, columns=S.U.BIN, dtype=np.int64)


def census(events=(1, 2, 3), subrun=21):
    return pd.DataFrame({"run": np.full(len(events), 1400, dtype=np.int64),
                         "subrun": np.full(len(events), subrun, dtype=np.int64),
                         "event": np.array(events, dtype=np.int64)})


def test_union_includes_bank_only_template_only_and_evaluated_only_cells():
    actual = S.cell_union(cells([(22, 2), (22, 1)]), cells([(2112, 3), (22, 2)]),
                          cells([(2212, 4), (22, 1)]))
    pd.testing.assert_frame_equal(actual, cells([(22, 1), (22, 2), (2112, 3), (2212, 4)]))
    assert S.cell_union().empty


@pytest.mark.parametrize("rows", [[(13, 1)], [(0, 1)], [(22, -1)]])
def test_union_invalid_native_cells(rows):
    with pytest.raises(ValueError):
        S.cell_union(cells(rows))


def test_absent_bank_is_not_no_template_and_present_rounding_zero_is_separate():
    raw = np.array([0, 1, 5, 10, 0, 100])
    denominator = np.rint(raw * 10_000_000 / 94_000_000).astype(np.int64)
    result = S.masks(raw, denominator, np.array([1, 0, 0, 2, 0, 1]))
    np.testing.assert_array_equal(result["absent_bank"], [1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(result["present_but_rounded_zero"], [0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(result["no_template"], [0, 1, 1, 0, 1, 0])
    np.testing.assert_array_equal(result["saturated"], [0, 0, 0, 1, 0, 0])
    assert not (result["absent_bank"] & result["present_but_rounded_zero"]).any()


def test_support_api_saturation_and_round_to_even_are_not_repaired():
    # Two files at 10M fixed generated exposure: 0.5 -> 0, 1.5 -> 2.
    counts = np.array([[1, 3, 4], [0, 0, 0]], dtype=np.int64)
    result = S.D.support_report(counts, np.array([10_000_000] * 2), np.array([1, 3, 1]),
                                S.GENERATED, S.REPLICAS, S.SEED)
    assert result["nominal_scaled_denominator"] == [0., 2., 2.]
    assert result["nominal_missing_support"] == [True, False, False]
    assert result["nominal_saturated"] == [False, True, False]
    assert result["any_invalid_replicas"] > 0
    assert result["replicas"] == 1000
    assert result["downstream_propagation_authorized"] is False
    assert result["ensemble_support_pass"] is False


def test_bootstrap_tail_loss_includes_evaluated_cells_without_templates():
    counts = np.array([[1, 0], [0, 0]], dtype=np.int64)
    events = np.array([47_000_000] * 2, dtype=np.int64)
    support = S.D.support_report(counts, events, np.zeros(2, dtype=np.int64), S.GENERATED, S.REPLICAS, S.SEED)
    tails = S.bootstrap_tails(counts, events, np.array([1, 3]))
    assert support["any_invalid_replicas"] == 0  # No templates is NOT a tail-support pass.
    assert tails["any_evaluated_zero_denominator_replicas"] == 1000
    assert tails["absent_bank_replicas_per_cell"][1] == 1000
    assert tails["present_but_rounded_zero_replicas_per_cell"][0] > 0
    np.testing.assert_array_equal(np.array(tails["absent_bank_replicas_per_cell"]) +
                                  tails["present_but_rounded_zero_replicas_per_cell"],
                                  support["zero_denominator_replicas_per_bin"])


def test_completed_conditional_contract():
    S.completed_contract(completed())


@pytest.mark.parametrize("field,value", [
    ("conditional_capture_evidence_complete", False), ("strict_event_census_certified", False),
    ("observed_producer_pairs_match", False), ("bank_admitted_for_propagation", True),
    ("downstream_authorized", True), ("selection_independence_certified", True),
    ("scope", "joint time/energy"), ("assumptions", []), ("limits", []),
    ("events", 2_000_000), ("files", 2), ("schema", "other"), ("input_sha256_post", {}),
    ("strict_event_census_certified", 1), ("downstream_authorized", 0),
])
def test_incomplete_or_promoted_campaign_rejected(field, value):
    report = completed()
    report[field] = value
    with pytest.raises(ValueError, match="completed conditional"):
        S.completed_contract(report)


def test_live_campaign_contract_fails_before_any_cache_or_worker_read(tmp_path, monkeypatch):
    report = completed()
    report["conditional_capture_evidence_complete"] = False
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report))
    monkeypatch.setattr(S.G, "load_plan", lambda *a: pytest.fail("live manifest read"))
    monkeypatch.setattr(S.U, "load_cache", lambda *a: pytest.fail("cache read before completed contract"))
    out = tmp_path / "new.json"
    with pytest.raises(ValueError, match="completed conditional"):
        S.run(source, "/missing/uncertainty_v2/production", "/missing/cert.json", out)
    assert not out.exists()


def test_partition_disjointness_uses_exact_integer_keys_and_empty_events():
    seen = set()
    assert S.claim_partition(census(), seen, (1400, 21, 3)) == (1400, 21)
    with pytest.raises(ValueError, match="overlapping"):
        S.claim_partition(census((4, 5)), seen)
    assert S.claim_partition(census(subrun=22), seen, (1400, 22, 3)) == (1400, 22)
    with pytest.raises(ValueError, match="range"):
        S.exact_partition(census((1, 2, 4)), (1400, 21, 3))
    with pytest.raises(ValueError, match="empty"):
        S.exact_partition(census(()))
    with pytest.raises(ValueError, match="duplicate"):
        S.exact_partition(census((1, 1)))
    with pytest.raises(ValueError, match="identities"):
        S.exact_partition(census().astype(float))


def test_strict_count_keeps_zero_generator_census_event(tmp_path):
    # Existing purely synthetic ROOT fixture, no production data or external jobs.
    from test_generator_bank_capture import files
    old, new = files(tmp_path)
    counts, report = S.count_strict_file(str(old), str(new), 1400, 2001, 2)
    assert counts.empty  # Only a muon in the tiny fixture.
    assert report["event_count"] == 2 and report["zero_generator_events"] == 1
    assert report["validation"]["schema_version"] == 2


@pytest.mark.parametrize("case", ["alias", "sentinel", "gap", "payload"])
def test_fresh_strict_gate_rejects_schema_census_or_payload_drift(tmp_path, case):
    from test_generator_bank_capture import files
    old, new = files(tmp_path, case)
    with pytest.raises((ValueError, AssertionError)):
        S.count_strict_file(str(old), str(new), 1400, 2001, 2)


def test_candidate_comparison_restricts_only_comparison_not_union():
    series = pd.Series([3, 9], index=pd.MultiIndex.from_frame(cells([(22, 1), (2112, 4)])))
    bins = cells([(22, 1), (2212, 8)]).to_dict("records")
    S.candidate_counts_match(series, bins, [3, 0])
    with pytest.raises(ValueError, match="legacy candidate"):
        S.candidate_counts_match(series, bins, [4, 0])
    with pytest.raises(ValueError, match="duplicate"):
        S.candidate_counts_match(series, bins + bins, [3, 0, 3, 0])


def test_tail_upper_bound_deduplicates_events_and_keeps_all_four_sectors():
    features = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9] * 8, [0] * 8], dtype=float)
    bound = S.tail_bound(np.array([0, 0, 1, 2]), np.array([True, True, False, True]), features)
    assert bound["events"] == 2 and bound["primaries"] == 3
    assert bound["whole_event_nonmu_steps_energy"] == list(range(1, 9))
    assert S.tail_bound(np.array([], dtype=np.int64), np.array([], dtype=bool), features) == S.empty_bound()


def test_whole_event_features_use_identity_not_time_or_positions():
    pop = census()
    parts = pd.DataFrame({"run": [1400] * 6, "subrun": [21] * 6, "event": [1] * 6,
                          "key": [1, 2, 3, 4, 5, 6], "parent": [-1, 1, -1, 3, 3, 3],
                          "pdg": [13, 11, 22, 11, 11, 11]})
    steps = parts.rename(columns={"key": "sp"})[S.U.KEY + ["sp"]].assign(
        bar=[0, 1, 0, 1, 2, 3], edep=[99., 98., 2., 3., 4., 5.], t=0., x=0., y=0., z=0.)
    geom = SimpleNamespace(sector=np.arange(4))
    a = S.H.scalar_features(steps, parts, pop, geom).toarray()
    b = S.H.scalar_features(steps.assign(t=1e9, x=-999, y=999, z=333), parts, pop, geom).toarray()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[0], [1, 2, 1, 3, 1, 4, 1, 5])
    assert not a[1:].any()  # Zero-step/census events preserved.


def evaluation_evidence():
    certificate = {"complete_expected_id_range": True, "train_eval_overlap": 0,
                   "cross_shard_overlap": 0, "strict_art_role_certified": False, "events": S.GENERATED,
                   "shards": [{"shard": k, "events": 50000, "reference_generator_exact": True}
                              for k in range(21, 221)]}
    report = {"events_inferred": S.GENERATED, "cache": "/data/uncertainty_v2/production", "cache_sha256": "a" * 64,
              "shards": [{"shard": k, "events_inferred": 50000, "reference_gen_exact": True} for k in range(21, 221)],
              "input_sha256": {f"/data/{p}.001400_{k:08d}.root": "b" * 64
                               for k in range(21, 221) for p in ("gen", "crvsteps")}}
    return certificate, report


def test_exact_existing_evaluation_schema_and_paths():
    cert, report = evaluation_evidence()
    pairs = S.evaluation_pairs(cert, report, Path(report["cache"]), "a" * 64)
    assert len(pairs) == 200
    assert pairs[0] == (21, "/data/gen.001400_00000021.root", "/data/crvsteps.001400_00000021.root")
    assert pairs[-1][0] == 220


@pytest.mark.parametrize("change", ["shards", "count", "cache", "missing", "ambiguous", "strict_claim", "overlap"])
def test_evaluation_binding_rejection(change):
    cert, report = evaluation_evidence()
    if change == "shards":
        cert["shards"][1]["shard"] = 21
    elif change == "count":
        report["events_inferred"] -= 1
    elif change == "cache":
        report["cache"] += "_new"
    elif change == "missing":
        report["input_sha256"].pop(next(iter(report["input_sha256"])))
    elif change == "ambiguous":
        report["input_sha256"]["/other/gen.001400_00000021.root"] = "b" * 64
    elif change == "strict_claim":
        cert["strict_art_role_certified"] = True
    else:
        cert["train_eval_overlap"] = 1
    with pytest.raises(ValueError):
        S.evaluation_pairs(cert, report, Path("/data/uncertainty_v2/production"), "a" * 64)


def test_hash_binding_conflict_content_and_path_drift(tmp_path):
    a, b, link = tmp_path / "a", tmp_path / "b", tmp_path / "link"
    a.write_text("same")
    b.write_text("same")
    link.symlink_to(a)
    bound = S.Bindings()
    bound.freeze(link)
    bound.verify()
    with pytest.raises(ValueError, match="conflicting"):
        bound.add({str(link): "a" * 64})
    link.unlink()
    link.symlink_to(b)
    with pytest.raises(ValueError, match="drift"):
        bound.verify()
    fresh = S.Bindings()
    fresh.freeze(a)
    a.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        fresh.verify()
    with pytest.raises(ValueError, match="unbound"):
        fresh.read(tmp_path / "not_bound.json")


def test_wrong_hash_fails_before_consumption(tmp_path):
    path = tmp_path / "file.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash drift"):
        S.Bindings().add({str(path): "0" * 64})


def test_exclusive_json_refuses_overwrite_symlink_and_nonjson(tmp_path):
    path = tmp_path / "new.json"
    S.write_new_json(path, {"downstream_authorized": False})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        S.write_new_json(path, {})
    assert path.read_bytes() == original
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        S.write_new_json(link, {})
    with pytest.raises(ValueError, match="NEW JSON"):
        S.write_new_json(tmp_path / "forbidden.root", {})
    assert not (tmp_path / "forbidden.root").exists()


def test_current_code_hashes_include_new_driver_and_existing_readonly_apis():
    hashes = S._current_code()
    for module in (S, S.G, S.D, S.U, S.H, S.E, S.genassoc, S.tables._core):
        assert str(Path(module.__file__).resolve()) in hashes


def test_evaluation_stage_cell_bounds_and_disjointness_without_event_loops(tmp_path, monkeypatch):
    pop = census((1, 2))
    gen = pd.concat([pop.iloc[[0]], pop.iloc[[0]], pop.iloc[[1]]], ignore_index=True).assign(
        idx=[0, 1, 0], pdg=[22, 22, 2112], e=[1., 1., S.U.tier1b.MASS[2112] + 1.], t=0.)
    parts = pop.assign(key=1, parent=-1, pdg=[22, 13])
    steps = pd.concat([pop.iloc[[0]]] * 4, ignore_index=True).assign(
        sp=1, bar=[0, 1, 2, 3], edep=[2., 3., 4., 5.])
    geom = SimpleNamespace(sector=np.arange(4), nbars=4, local_to_world=np.tile([0, 1, 2], (4, 1)),
                           center=np.zeros((4, 3)), module=np.zeros(4, dtype=int), layer=np.zeros(4, dtype=int))
    _, ancestry = S.A.annotate(steps, parts, geom)
    count = S.cell_counts(gen)
    # Photon absent bank WITH a template; neutron present but rounded zero WITHOUT one.
    bank = pd.Series([0, 1], index=count.index, dtype=np.int64)
    showers = pd.Series([1, 0], index=count.index, dtype=np.int64)
    frames = {("gp", "gen"): gen, ("rp", "gen"): gen,
              ("rp", "steps"): steps, ("rp", "particles"): parts}
    monkeypatch.setattr(S.tables, "read", lambda path, name: frames[(path, name)].copy())
    monkeypatch.setattr(S.genassoc, "validate_generator_stream", lambda _: {})
    real_census = S.E.contiguous_census
    monkeypatch.setattr(S.E, "contiguous_census", lambda g, k, n: real_census(g, k, 2))
    real_claim = S.claim_partition
    monkeypatch.setattr(S, "claim_partition", lambda c, seen, expected: real_claim(c, seen, (1400, 21, 2)))
    cert = {"shards": [{"ancestry": ancestry}]}
    heldout = {"shards": [{"generator_rows": 3}]}
    counts, records, bounds, categories = S._evaluation(
        [(21, "gp", "rp")], cert, heldout, bank, showers, geom, set())
    assert counts[0].sum() == 3 and records[0]["events"] == 2
    assert categories["absent_bank"]["primaries"] == 2
    assert categories["absent_bank"]["events"] == 1
    assert categories["absent_bank"]["whole_event_nonmu_steps_energy"] == [1, 2, 1, 3, 1, 4, 1, 5]
    assert categories["present_but_rounded_zero"]["primaries"] == 1
    assert categories["no_template"]["primaries"] == 1
    assert categories["any_tail"]["events"] == 2
    assert sum(v["primaries"] for v in bounds.values()) == 3
    with pytest.raises(ValueError, match="overlapping"):
        S._evaluation([(21, "gp", "rp")], cert, heldout, bank, showers, geom, {(1400, 21)})


def test_source_has_no_capture_or_builder_invocation():
    import ast
    source = ast.parse(Path(S.__file__).read_text())
    called = {node.func.attr for node in ast.walk(source)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called.intersection({"capture", "aggregate", "prepare", "rebuild", "write_library",
                                    "save_npz", "savez", "savez_compressed", "mkdir", "Popen"})


def test_output_refused_before_any_input_access(tmp_path):
    out = tmp_path / "existing.json"
    out.write_text("keep")
    with pytest.raises(FileExistsError):
        S.run("missing", "missing", "missing", out)
    assert out.read_text() == "keep"
    with pytest.raises(ValueError, match="NEW JSON"):
        S.run("missing", "missing", "missing", tmp_path / "library.root")


def test_retained_cache_preflight_when_explicitly_requested():
    import os
    cache = os.environ.get("R2_RETAINED_CACHE_PREFLIGHT")
    if not cache:
        pytest.skip("set R2_RETAINED_CACHE_PREFLIGHT for read-only fixed-cache preflight")
    before = S._current_code()
    bound = S.Bindings()
    meta, cells, showers, donors, old = S._cache(cache, bound, set())
    assert meta["generated"] == 10_000_000
    assert len(donors) == 20 and len(old) == 10 and showers.sum() == 65_400
    assert len(cells) > 0
    assert all(r["historical_dumper_reexecuted"] is False for r in donors)
    assert S._current_code() == before
    bound.verify()
