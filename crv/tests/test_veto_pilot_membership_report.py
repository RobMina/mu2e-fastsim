"""Saved-only verifier tests; inherited raw verification is fixture-substituted, not certified."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import pilot_membership_report as V
from test_veto_pilot_membership import saved, archive
from test_veto_saved_qualified_report import reseal

M, P = V.M, V.P


def forbidden(*a, **k):
    pytest.fail("saved verifier attempted finder or physics regeneration")


@pytest.fixture
def generated(saved, tmp_path, monkeypatch):
    M.run(**saved)
    for module, names in ((M, ("run",)), (P, ("run",)), (P.R, ("recover", "prepare_pulses")),
                          (P.R.C, ("find_event", "_find_clusters", "_check_coincidence_vec", "_properties"))):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    return dict(directory=saved["out"], out=tmp_path / "verified-membership.json",
                expected_summary_sha256=M.S.C.sha256(saved["out"] / "summary.json")), saved


def repin(args):
    reseal(args["directory"])
    args["expected_summary_sha256"] = M.S.C.sha256(args["directory"] / "summary.json")


def summary_change(args, change):
    path = args["directory"] / "summary.json"
    value = M.read_json(path)
    change(value)
    path.write_bytes(M.A._canonical(value))
    repin(args)


def table_change(args, name, change):
    path = args["directory"] / (name + ".parquet")
    table = pd.read_parquet(path)
    value = change(table)
    (table if value is None else value).to_parquet(path, index=False)
    repin(args)


def reject(args, match=None):
    with pytest.raises((ValueError, AssertionError, FileNotFoundError), match=match):
        V.verify(**args)
    assert not args["out"].exists()


def test_complete_saved_only_verification(generated):
    args, _ = generated
    result = V.verify(**args)
    assert json.loads(args["out"].read_text()) == result
    assert result["unique_frames"] == 2 and result["ledger_rows"] == 16
    assert len(result["counts"]) == 8 and len(result["comparisons"]) == 6
    assert result["persisted_path_bytes"] <= 32 * 1024**2
    assert result["native_member_structure_exact"] and result["per_frame_coverage_exact"]
    assert all(result[name] is False for name in V.FALSE_FIELDS)
    assert result["regenerated_finder"] is result["cluster_scalars_independently_recomputed"] is False
    assert result["F"] is result["finite_library_uncertainty"] is None


def test_external_pin_is_required(generated):
    args, _ = generated
    args["expected_summary_sha256"] = "0" * 64
    reject(args, "changed bound")


@pytest.mark.parametrize("field", sorted(V.FALSE_FIELDS | V.NULL_FIELDS | {"scope"}))
def test_scope_is_mandatory_and_exact(generated, field):
    args, _ = generated
    summary_change(args, lambda s: s.pop(field))
    reject(args, "schema/scope")


@pytest.mark.parametrize("field,value", [("execution_admitted", 0), ("F", 0), ("timing_closure", True),
                                          ("stock_rerun", True), ("scope", "promoted"), ("output_cap_bytes", 1)])
def test_scope_cannot_be_promoted_or_coerced(generated, field, value):
    args, _ = generated
    summary_change(args, lambda s: s.update({field: value}))
    reject(args)


@pytest.mark.parametrize("kind", ["extra", "failed", "pending", "code", "summary"])
def test_transaction_receipts_and_snapshots(generated, kind):
    args, _ = generated
    path = args["directory"] / {"extra": "extra.txt", "failed": M.A.FAILED, "pending": M.A.PENDING,
                                "code": "code0.txt", "summary": "summary.json"}[kind]
    path.write_bytes(b"changed")
    reject(args)


@pytest.mark.parametrize("kind", ["code_domain", "input_domain", "counts", "comparisons", "grid_meta", "census"])
def test_resealed_semantic_summary_changes(generated, kind):
    args, _ = generated
    def change(s):
        if kind == "code_domain":
            s["code_sha256"].pop(next(iter(s["code_sha256"])))
        elif kind == "input_domain":
            s["inputs"].pop(next(k for k in s["inputs"] if k.endswith("code0.txt")))
        elif kind == "counts":
            s["counts"][0]["members"] += 1
        elif kind == "comparisons":
            s["comparisons"].clear()
        elif kind == "grid_meta":
            s["grid"]["selected"] += 1
        else:
            s["unique_frames"] += 1
    summary_change(args, change)
    reject(args)


@pytest.mark.parametrize("kind", ["selected", "grid", "pulse", "dtype", "coverage", "domain", "seed", "extra_column"])
def test_resealed_saved_table_changes(generated, kind):
    args, _ = generated
    if kind == "selected":
        table_change(args, "selected", lambda t: t.assign(event=t.event + 1))
    elif kind == "grid":
        table_change(args, "grid", lambda t: t.assign(weight=t.weight + .01))
    elif kind == "pulse":
        table_change(args, "pulses", lambda t: t.assign(wf=t.wf + 1))
    elif kind == "dtype":
        table_change(args, "coverage", lambda t: t.assign(pulses=t.pulses.astype(float)))
    elif kind == "coverage":
        table_change(args, "coverage", lambda t: t.assign(track_grid_coverage=1.))
    elif kind == "domain":
        table_change(args, "coverage", lambda t: t.iloc[:-1])
    elif kind == "seed":
        table_change(args, "coverage", lambda t: t.assign(response_seed=np.uint64(7)))
    else:
        table_change(args, "members", lambda t: t.assign(undeclared=0))
    reject(args)


@pytest.mark.parametrize("where", ["existing", "membership", "pilot", "prior", "symlink"])
def test_report_is_exclusive_outside_evidence(generated, where):
    args, saved = generated
    if where == "existing":
        args["out"].write_text("keep")
    elif where == "membership":
        args["out"] = args["directory"] / "report.json"
    elif where == "pilot":
        args["out"] = saved["pilot"] / "report.json"
    elif where == "prior":
        args["out"] = saved["verified"].parent / "report.json"
    else:
        args["out"].symlink_to(saved["pilot"] / "absent")
    with pytest.raises(ValueError):
        V.verify(**args)
    if where == "existing":
        assert args["out"].read_text() == "keep"


@pytest.mark.parametrize("where", ["member", "receipt", "new_output", "source_pulse", "source_extra", "prior", "root"])
def test_late_file_mutations_never_publish(generated, monkeypatch, where):
    args, saved = generated
    child = saved["pilot"] / M.read_json(saved["pilot"] / "run/summary.json")["records"][0]["path"]
    path = {"member": args["directory"] / "members.parquet", "receipt": args["directory"] / M.A.COMPLETE,
            "new_output": args["directory"] / "late.txt", "source_pulse": child / "historical_pulses.parquet",
            "source_extra": child / "late.txt", "prior": saved["verified"], "root": saved["pilot"] / "late.txt"}[where]
    original = P.compare
    def mutate(*a, **k):
        result = original(*a, **k)
        path.write_bytes(b"late mutation")
        return result
    monkeypatch.setattr(P, "compare", mutate)
    reject(args)


def test_dtype_header_check_precedes_decode(generated, monkeypatch):
    args, _ = generated
    table_change(args, "members", lambda t: t.assign(pulse_idx=t.pulse_idx.astype(float)))
    original = pd.read_parquet
    def guarded(path, *a, **k):
        assert Path(path) != args["directory"] / "members.parquet", "decoded wrong dtype"
        return original(path, *a, **k)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    reject(args, "dtype changed")


@pytest.fixture
def nonzero(monkeypatch):
    from test_veto_pilot_coincidence import domain
    source, registry, frames, calibration, geom = domain()
    source = source.sort_values(["frame_id", *P.PULSE_KEYS]).reset_index(drop=True)
    source["wf"] = np.arange(len(source), dtype=np.int64)
    source["failed"], source["zerondf"] = False, True
    products = P.run(source, registry, frames, calibration, geom, registry_digest=M.F.digest(registry))
    products["pulses"] = products["pulses"].merge(source[[*P.FRAME_KEYS, *P.PULSE_KEYS, "failed", "zerondf"]],
                                                on=[*P.FRAME_KEYS, *P.PULSE_KEYS], validate="one_to_one")
    monkeypatch.setattr(P, "run", forbidden)
    monkeypatch.setattr(P.R, "recover", forbidden)
    return source, frames, registry, calibration, geom, products


def test_nonzero_scalar_member_validation_without_finder(nonzero):
    actual = V._products(*nonzero)
    assert actual["frames"].pulses.tolist() == [4, 3, 3, 0]
    assert actual["frames"].clusters.tolist() == [1, 1, 1, 0]
    assert len(actual["members"]) == 10
    assert actual["members"].iloc[:4].pulse_idx.tolist() == [0, 2, 3, 1]
    # The verifier deliberately does not claim to independently regenerate scalar physics.
    nonzero[-1]["clusters"]["slope"] += .1
    assert len(V._products(*nonzero)["clusters"]) == 3


@pytest.mark.parametrize("kind", ["bounds", "structural", "sector", "layer", "order", "duplicate", "missing", "empty_cluster",
                                   "cluster_index", "cluster_sector", "cluster_dtype", "nonfinite", "negative_pe", "nLayers",
                                   "wf", "flag", "pulse_value", "qualified_key"])
def test_nonzero_invalid_native_structure(nonzero, kind):
    source, frames, registry, calibration, geom, products = nonzero
    p, c, m = (products[name] for name in ("pulses", "clusters", "members"))
    if kind == "bounds":
        m.loc[0, "pulse_idx"] = 999
    elif kind == "structural":
        m.loc[0, "chan"] += 1
    elif kind == "sector":
        m.loc[0, "sector"] += 1
    elif kind == "layer":
        m.loc[0, "layer"] = 3
    elif kind == "order":
        m.loc[0, "member_order"] = 9
    elif kind == "duplicate":
        m.loc[1, "pulse_idx"] = m.loc[0, "pulse_idx"]
    elif kind == "missing":
        m.loc[0, "cluster_idx"] = 99
    elif kind == "empty_cluster":
        products["members"] = m[m.frame_id != 0].reset_index(drop=True)
    elif kind == "cluster_index":
        c.loc[0, "idx"] = 2
    elif kind == "cluster_sector":
        c.loc[0, "sectortype"] = 0
    elif kind == "cluster_dtype":
        products["clusters"] = c.assign(nLayers=c.nLayers.astype(float))
    elif kind == "nonfinite":
        c.loc[0, "slope"] = np.nan
    elif kind == "negative_pe":
        c.loc[0, "pe"] = -1.
    elif kind == "nLayers":
        c.loc[0, "nLayers"] = 4
    elif kind == "wf":
        p.loc[0, "wf"] += 1
    elif kind == "flag":
        p.loc[0, "zerondf"] = False
    elif kind == "pulse_value":
        p.loc[0, "pe"] += .1
    else:
        m.loc[0, "input_sha256"] = "b" * 64
    with pytest.raises((ValueError, AssertionError)):
        V._products(*nonzero)


def test_nonzero_persisted_runner_and_saved_verifier(tmp_path, monkeypatch):
    """Use the runner's nonzero fixture; source admission/grid are explicitly synthetic here."""
    from test_veto_pilot_membership import test_nonzero_native_membership_through_persisted_runner
    original = M.run
    def capture(pilot, verified, paired, out, **hashes):
        d = M.admit()
        (pilot / "run").mkdir()
        # The existing fixture substitutes upstream admission. Give this saved-only
        # test unique pinned source locations, rather than claim raw verification.
        for name, path in (("pilot", pilot / "run/summary.json"), ("verified", verified), ("paired", paired)):
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"synthetic": name}))
            digest = M.S.C.sha256(path)
            hashes["expected_" + name + "_sha256"] = digest
            d["inputs"][str(path)] = digest
        return original(pilot, verified, paired, out, **hashes)
    monkeypatch.setattr(M, "run", capture)
    test_nonzero_native_membership_through_persisted_runner(tmp_path, monkeypatch)
    for module, names in ((M, ("run",)), (P, ("run",)), (P.R, ("recover", "prepare_pulses")),
                          (P.R.C, ("find_event", "_properties", "_find_clusters", "_check_coincidence_vec", "_check_coincidence")),
                          (M.T, ("verify",)), (M.S, ("resources", "run")), (M.S.I, ("load_input",)),
                          (M.S.O.K, ("make_photons",)), (M.S.R, ("run_arm",)),
                          (M.S.R.P, ("reconstruct", "reconstruct_matrix")), (M.S.R.W, ("synthesize",)),
                          (M.S.R.S, ("simulate_candidates",))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    result = V.verify(tmp_path / "result", tmp_path / "nonzero-report.json",
                      expected_summary_sha256=M.S.C.sha256(tmp_path / "result/summary.json"))
    assert result["unique_frames"] == 4 and result["ledger_rows"] == 32
    assert sum(row["coincidences"] for row in result["counts"]) == 24
    assert sum(row["members"] for row in result["counts"]) == 80
    assert sum(row["gate_fail"] for row in result["counts"]) == 8
    assert sum(row["nonzero_coverage"] for row in result["counts"]) == 16
    assert all(row["exact"] and row["coverage_exact"] for row in result["comparisons"].values())


@pytest.mark.parametrize("kind", ["columns", "rows", "groups", "bytes"])
def test_metadata_caps_precede_decode(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace
    path = tmp_path / "table.parquet"
    path.write_bytes(b"not decoded")
    values = dict(num_rows=1, num_row_groups=1, num_columns=1)
    values.update({"columns": {"num_columns": 2}, "rows": {"num_rows": 3},
                   "groups": {"num_row_groups": 4097}, "bytes": {}}[kind])
    meta = SimpleNamespace(**values, row_group=lambda i: SimpleNamespace(total_byte_size=129 * 1024**2 if kind == "bytes" else 1))
    class Header:
        metadata = meta
        schema_arrow = None
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
    monkeypatch.setattr(V.pq, "ParquetFile", lambda *a: Header())
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="cap/schema|byte cap"):
        V._table(path, {"i": "int64"}, 2)


def test_resealed_snapshot_is_not_adopted(generated):
    args, _ = generated
    (args["directory"] / "code0.txt").write_text("changed generation source snapshot")
    repin(args)
    reject(args, "changed bound")


@pytest.mark.parametrize("field,value", [("memory_before", {}), ("memory_after", None),
                                          ("coincidence_coverage_seconds", -1), ("coincidence_coverage_seconds", True)])
def test_measurement_schema_not_a_physics_certification(generated, field, value):
    args, _ = generated
    summary_change(args, lambda s: s.update({field: value}))
    reject(args)


def test_combination_admission_without_finder(nonzero, monkeypatch):
    monkeypatch.setattr(P, "combination_work", lambda *a: P.Limits().max_combinations + 1)
    with pytest.raises(ValueError, match="combination cap"):
        V._products(*nonzero)


def test_zero_sign_grid_bits_rejected(generated):
    args, _ = generated
    def sign(table):
        index = table.index[table.weight == 0][0]
        table.loc[index, "weight"] = -0.
    table_change(args, "grid", sign)
    reject(args, "bits changed")


def test_empty_scope_additional_key_rejected(generated):
    args, _ = generated
    summary_change(args, lambda s: s.update(unexpected=False))
    reject(args, "schema/scope")


def test_unbound_output_record_domain_rejected(generated):
    args, _ = generated
    table_change(args, "coverage", lambda t: t.assign(batch=t.batch + 1))
    reject(args)


def test_ignored_member_is_not_active(nonzero):
    source, frames, registry, calibration, geom, products = nonzero
    chan = int(products["members"].chan.iloc[0])
    source.loc[source.chan == chan, "channel_ignored"] = True
    products["pulses"].loc[products["pulses"].chan == chan, "channel_ignored"] = True
    calibration.loc[(calibration.input_sha256 == frames.input_sha256.iloc[0]) & (calibration.chan == chan), "status"] = 2
    with pytest.raises(ValueError, match="ignored pulse"):
        V._products(*nonzero)


@pytest.mark.parametrize("field,value", [("ignore_channels", False), ("ignore_channels", 1), ("ignored_pulse_policy", "drop")])
def test_ignored_policy_exact(generated, field, value):
    args, _ = generated
    summary_change(args, lambda s: s.update({field: value}))
    reject(args, "ignored pulse policy")


def test_ignored_rows_remain_in_canonical_collection(monkeypatch):
    from test_veto_pilot_coincidence import domain
    source, registry, frames, calibration, geom = domain()
    source = source.sort_values(["frame_id", *P.PULSE_KEYS]).reset_index(drop=True)
    source["wf"] = np.arange(len(source), dtype=np.int64)
    source["failed"], source["zerondf"] = False, True
    source.loc[source.chan == 1, "channel_ignored"] = True
    calibration.loc[(calibration.input_sha256 == frames.input_sha256.iloc[0]) & (calibration.chan == 1), "status"] = 2
    products = P.run(source, registry, frames, calibration, geom, registry_digest=M.F.digest(registry))
    products["pulses"] = products["pulses"].merge(source[[*P.FRAME_KEYS, *P.PULSE_KEYS, "failed", "zerondf"]],
                                                on=[*P.FRAME_KEYS, *P.PULSE_KEYS], validate="one_to_one")
    monkeypatch.setattr(P, "run", forbidden)
    monkeypatch.setattr(P.R, "recover", forbidden)
    monkeypatch.setattr(P.R.C, "find_event", forbidden)
    actual = V._products(source, frames, registry, calibration, geom, products)
    assert len(actual["pulses"]) == 10 and actual["pulses"].channel_ignored.sum() == 1
    assert len(actual["members"]) == 9
    assert actual["members"].query("frame_id == 0").pulse_idx.tolist() == [0, 2, 3]


@pytest.mark.parametrize("dtype", [object, "string"])
@pytest.mark.parametrize("empty", [False, True])
def test_text_storage_representation_not_identity(tmp_path, dtype, empty):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pd.DataFrame({"s": pd.Series([] if empty else ["abc"], dtype=dtype),
                          "n": pd.Series([] if empty else [1], dtype="int64")})
    target = table.assign(s=table.s.astype(object))
    for arrow_type in (pa.string(), pa.large_string()):
        path = tmp_path / (str(arrow_type) + ".parquet")
        pq.write_table(pa.Table.from_pandas(table, schema=pa.schema([("s", arrow_type), ("n", pa.int64())])), path)
        got = V._table(path, {"s": "object", "n": "int64"}, 2)
        V._exact(got, target, "text")


@pytest.mark.parametrize("value", [1, b"abc", None])
def test_text_value_normalization_never_coerces_nonstring(value):
    with pytest.raises(ValueError):
        V._typed(pd.DataFrame({"s": pd.Series([value], dtype=object)}), {"s": "object"})


def test_compact_exposes_empty_domain_limitation(generated):
    args, _ = generated
    result = V.verify(**args)
    compact = V.compact(result)
    assert not compact["nonempty_real_membership_exercised"]
    assert not compact["nonzero_real_coverage_exercised"]
    assert compact["historical_contrasts"] == compact["exact_contrasts"] == 6
    assert compact["F"] is None and not compact["execution_admitted"]
