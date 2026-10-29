"""Saved-only synthetic refit certificates; no real finder, physics or raw admission.

The inherited authority is synthetic, not a certification of the retained data.
Only fixture generation substitutes native finder output; bridge preparation,
Parquets, external pins and BatchArtifacts receipts are real throughout.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import refit_membership_report as V
from test_veto_refit_membership import SavedRefits, sha, write_json
from test_veto_saved_qualified_report import reseal

N, M, P, T = V.N, V.M, V.P, V.V


def forbidden(*args, **kwargs):
    pytest.fail("saved-only verifier attempted finder, physics, raw input, prior verification or coverage")


def typed(table, types):
    return table[list(types)].astype(types)


def synthetic_native(pulses, registry, selected, calibration, geom, *, registry_digest):
    """Invent structure, NOT finder output; exercise 3 clusters and 8/4/5 links."""
    assert registry_digest == N.F.digest(registry)
    p = typed(pulses.rename(columns={"refit_idx": "idx"}), T.PULSES)
    cs, ms = [], []
    sizes = {2505: 8, 281: 4, 2755: 5}
    for _, frame in selected.iterrows():
        members = p[p.frame_id == frame.frame_id].drop_duplicates("chan").iloc[:sizes[int(frame.event)]]
        # Native order need not be pulse-index order, and is not independently regenerated.
        members = pd.concat([members.iloc[1:], members.iloc[:1]])
        identity = {k: frame[k] for k in P.FRAME_KEYS}
        bars = members.chan.to_numpy() // 4
        c = {name: (False if dtype == "bool" else 0) for name, dtype in T.CLUSTERS.items()}
        c.update(identity, idx=0, sectortype=0, nLayers=len(set(geom.layer[bars])),
                 start=490., end=510., pe=25., sidepe0=12., sidepe1=13., slope=.125)
        for side in (0, 1):
            c[f"sidehits{side}"] = int(((members.chan % 2 == side) & (members.pe > 0)).sum())
        c["poscalc"] = bool(c["sidehits0"] and c["sidehits1"])
        cs.append(c)
        for order, (_, pulse) in enumerate(members.iterrows()):
            ms.append(dict(identity, cluster_idx=0, member_order=order, pulse_idx=pulse.idx,
                           **{k: pulse[k] for k in P.PULSE_KEYS},
                           layer=int(geom.layer[int(pulse.chan) // 4]), sector=0))
    c, m = typed(pd.DataFrame(cs), T.CLUSTERS), typed(pd.DataFrame(ms), T.MEMBERS)
    frames = selected.copy()
    for name, table in (("pulses", p), ("clusters", c), ("members", m)):
        frames[name] = frames.frame_id.map(table.groupby("frame_id").size()).fillna(0).astype("int64")
    # dispatch itself adds the original flags; avoid duplicate merge columns here.
    return dict(frames=frames, pulses=p.drop(columns=["failed", "zerondf"]), clusters=c, members=m)


@pytest.fixture
def generated(tmp_path, monkeypatch):
    read = pd.read_parquet
    saved = SavedRefits(tmp_path, monkeypatch)
    monkeypatch.setattr(pd, "read_parquet", read)
    monkeypatch.setattr(N.B, "prepare", saved.original_prepare)
    saved.b["schema"] = "thread-d-optical-boundary-v1"
    for number, raw in saved.raw.items():
        # Include both sides and a zero-PE member, preserving the genuine waveform bridge.
        p, w, s, c = (raw[k] for k in ("stock_adc_refit_pulses", "stock_adc_refit_waveforms",
                                      "stock_pulses", "stock_adc_refit_comparison"))
        for table in (p, w, c):
            table["chan"] += (table.chan // 4) % 2
        s["sipm"] = s.bar % 2
        c["sipm"] = c.bar % 2
        p.loc[0, "pe"] = 0.
        s["pe"] = p.pe.astype("float32")
        s["flags"] = p.failed.astype("int64") + 8 * p.zerondf.astype("int64")
        c["flags"], c["pe_stock"], c["pe_fast"] = s["flags"], s.pe, p.pe
        c["delta_pe"] = c.pe_fast - c.pe_stock
        for suffix in ("stock_adc_refit_pulses", "stock_adc_refit_waveforms", "stock_pulses",
                       "stock_adc_refit_comparison"):
            saved.persist(number, suffix, raw[suffix])
    saved.commit_reports()
    with monkeypatch.context() as generation:
        generation.setattr(N.P, "run", synthetic_native)
        saved.result = saved.run()
    for module, names in (
        (N, ("run", "dispatch")), (P, ("run",)), (P.R, ("recover", "prepare_pulses")),
        (P.R.C, ("find_event", "_properties", "_find_clusters", "_check_coincidence_vec", "_check_coincidence")),
        (N.V, ("verify",)), (N.SR, ("verify",)), (N.M, ("run",)), (N.M.S, ("run",)),
        (N.I.Q, ("run", "admission")), (N.I.Q.R, ("run_arm",)), (N.I.Q.R.Q, ("run_arm",)),
        (N.I.Q.R.W, ("synthesize",)), (N.I.Q.R.P, ("reconstruct", "reconstruct_matrix", "fit_buckets")),
        (N.I.Q.R.P.P, ("reconstruct", "reconstruct_full", "fit_gumbel")),
        (N.I.Q.R.P.LM, ("fit_gumbel",)),
        (N.I.Q.R.S, ("simulate_candidates", "optical_candidates", "thermal_candidates")),
        (N.I.Q.M.K, ("make_photons",)), (N.I.Q.M.P, ("read_input", "run", "verify_metadata", "block_probe")),
        (N.I.D, ("run", "run_arm")), (T.G, ("evaluate", "frozen_grid", "track_histogram", "gate", "coinc_mask")),
        (N.M.T, ("verify",)),
    ):
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    import uproot
    monkeypatch.setattr(uproot, "open", forbidden)
    return dict(directory=saved.out, out=tmp_path / "refit-report.json",
                expected_summary_sha256=sha(saved.out / "summary.json")), saved


def repin(args):
    reseal(args["directory"])
    args["expected_summary_sha256"] = sha(args["directory"] / "summary.json")


def summary_change(args, change):
    path = args["directory"] / "summary.json"
    value = M.read_json(path)
    change(value)
    write_json(path, value)
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


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}


def test_nonempty_saved_only_certificate_and_no_source_edits(generated):
    args, saved = generated
    before = inventory(saved.root) | {"aux/" + k: v for k, v in inventory(saved.aux).items()}
    output_before = inventory(args["directory"])
    result = V.verify(**args)
    assert json.loads(args["out"].read_text()) == result
    assert result["unique_frames"] == 3 and result["registry_frames"] == 9000
    assert [sum(r[k] for r in result["counts"]) for k in ("pulses", "clusters", "members")] == [87, 3, 17]
    assert [r["members"] for r in result["frames"]] == [8, 4, 5]
    assert result["nonempty_real_refit_membership"] is True
    for key in ("saved_artifact_protocol_exact", "inherited_membership_transaction_exact", "source_bridge_rechecked",
                "source_tables_exact", "canonical_pulses_exact", "native_member_structure_exact",
                "positive_pe_side_counts_exact", "qualified_frame_census_exact"):
        assert result[key] is True
    assert all(result[k] is False for k in V.FALSE_FIELDS)
    assert all(result[k] is None for k in V.NULL_FIELDS)
    for key in ("regenerated_finder", "cluster_scalars_independently_recomputed",
                "native_member_order_independently_recomputed"):
        assert result[key] is False
    assert sum(d["pe_changed"] for d in result["stock_refit_differences"]) == 85
    assert all(d["failed_flag_changed"] == d["zerondf_flag_changed"] == 0 for d in result["stock_refit_differences"])
    assert 0 < result["persisted_path_bytes"] <= N.CAP
    assert args["out"].stat().st_size <= V.CAP
    assert before == inventory(saved.root) | {"aux/" + k: v for k, v in inventory(saved.aux).items()}
    assert output_before == inventory(args["directory"])
    for path, text in result["code_snapshots"].items():
        assert hashlib.sha256(text.encode()).hexdigest() == result["code_sha256"][path]
    compact = V.compact(result)
    assert compact["counts"] == result["counts"] and compact["F"] is None
    assert compact["nonempty_real_refit_membership"] is True


def test_external_summary_pin_required(generated):
    args, _ = generated
    args["expected_summary_sha256"] = "0" * 64
    reject(args, "changed bound")


@pytest.mark.parametrize("field", sorted(V.FALSE_FIELDS | V.NULL_FIELDS | {"scope"}))
def test_resealed_missing_scope_field(generated, field):
    args, _ = generated
    summary_change(args, lambda s: s.pop(field))
    reject(args, "schema/scope")


@pytest.mark.parametrize("field,value", [
    ("execution_admitted", 0), ("F", 0), ("timing_closure", True), ("scope", "promoted"),
    ("full_registry_admitted", 1), ("stock_refit_pulse_bridge_exact", False),
    ("registry_frames", 8999), ("unique_frames", True), ("saved_pulses", 86),
    ("saved_waveforms", 79), ("finder_calls", 3), ("output_cap_bytes", 1),
    ("seconds", True), ("seconds", -1), ("memory", {}), ("nonempty_real_refit_membership", 1),
])
def test_resealed_scope_census_and_measurements(generated, field, value):
    args, _ = generated
    summary_change(args, lambda s: s.update({field: value}))
    reject(args)


@pytest.mark.parametrize("kind", ["extra", "failed", "missing_complete", "receipt", "snapshot", "pin"])
def test_transaction_receipts_and_snapshots(generated, kind):
    args, _ = generated
    root = args["directory"]
    if kind == "missing_complete":
        (root / N.A.COMPLETE).unlink()
    else:
        path = root / {"extra": "extra.txt", "failed": N.A.FAILED, "receipt": N.A.COMPLETE,
                       "snapshot": "code0.txt", "pin": "summary.json"}[kind]
        path.write_bytes(b"changed")
        if kind == "snapshot":
            repin(args)
    reject(args)


@pytest.mark.parametrize("kind", ["code_domain", "input_domain", "count", "diagnostics", "frame", "nonempty",
                                  "config", "limits", "extra_key"])
def test_resealed_semantic_summary(generated, kind):
    args, _ = generated
    def change(s):
        if kind == "code_domain":
            s["code_sha256"].pop(next(iter(s["code_sha256"])))
        elif kind == "input_domain":
            s["inputs"].pop(next(k for k in s["inputs"] if k.endswith("code0.txt")))
        elif kind == "count":
            s["records"][0]["counts"]["members"] += 1
        elif kind == "diagnostics":
            s["records"][0]["diagnostics"]["stock_refit_differences"]["pe_changed"] = 0
        elif kind == "frame":
            s["records"][1]["frames"].pop()
        elif kind == "nonempty":
            s["nonempty_real_refit_membership"] = False
        elif kind in ("config", "limits"):
            s["finder_config" if kind == "config" else "coincidence_limits"] = {}
        else:
            s["unexpected"] = False
    summary_change(args, change)
    reject(args)


@pytest.mark.parametrize("kind", ["missing", "extra", "ambiguous"])
def test_source_report_role_domain(generated, kind):
    args, saved = generated
    def change(s):
        if kind == "missing":
            s["inputs"].pop(str(saved.verified))
        else:
            path = saved.root / "other-report.json"
            value = saved.dv if kind == "extra" else dict(saved.dv, schema=V.ROLES["boundary"])
            digest = write_json(path, value)
            if kind == "ambiguous":
                s["inputs"].pop(str(saved.verified))
            s["inputs"][str(path)] = digest
    summary_change(args, change)
    reject(args, "source JSON report domain|ambiguous source report")


@pytest.mark.parametrize("table,column,value", [
    ("saved_pulses", "failed", True), ("saved_pulses", "zerondf", True),
    ("saved_pulses", "wf", 999), ("saved_waveforms", "starttdc", 41),
    ("saved_bridge", "stock_idx", 999), ("saved_bridge", "refit_idx", 999),
    ("saved_bridge", "delta_pe", 0.1), ("pulses", "failed", True),
    ("pulses", "channel_ignored", True), ("pulses", "time_relative", 501.),
    ("pulses", "time_reco", 500.), ("pulses", "run", 8), ("pulses", "subrun", 3),
    ("pulses", "event", 2506), ("pulses", "input_sha256", "f" * 64),
])
def test_resealed_pulse_waveform_bridge_and_identity(generated, table, column, value):
    args, _ = generated
    table_change(args, "block0_" + table, lambda t: t.__setitem__(column, t[column].mask(t.index == 0, value)))
    reject(args)


@pytest.mark.parametrize("table", ["saved_pulses", "saved_waveforms", "saved_bridge", "pulses", "members"])
def test_saved_storage_order_is_not_silently_sorted(generated, table):
    args, _ = generated
    table_change(args, "block0_" + table, lambda t: t.iloc[::-1].reset_index(drop=True))
    reject(args)


@pytest.mark.parametrize("kind", ["bounds", "structural", "sector", "layer", "order", "duplicate", "missing_cluster",
                                  "empty_cluster", "cluster_index", "cluster_sector", "nLayers", "nonfinite",
                                  "negative_pe", "side0", "side1", "cluster_frame_dtype"])
def test_resealed_native_structure(generated, kind):
    args, _ = generated
    if kind == "empty_cluster":
        table_change(args, "block0_members", lambda t: t.iloc[:0])
    elif kind == "cluster_frame_dtype":
        table_change(args, "block0_clusters", lambda t: t.assign(frame_id=t.frame_id.astype("int64")))
    else:
        table, column, value = {
            "bounds": ("members", "pulse_idx", 999), "structural": ("members", "starttdc", 41),
            "sector": ("members", "sector", 1), "layer": ("members", "layer", 3),
            "order": ("members", "member_order", 99), "duplicate": ("members", "pulse_idx", 4),
            "missing_cluster": ("members", "cluster_idx", 99), "cluster_index": ("clusters", "idx", 2),
            "cluster_sector": ("clusters", "sectortype", 1), "nLayers": ("clusters", "nLayers", 1),
            "nonfinite": ("clusters", "slope", np.nan), "negative_pe": ("clusters", "pe", -1.),
            "side0": ("clusters", "sidehits0", 99), "side1": ("clusters", "sidehits1", 99),
        }[kind]
        table_change(args, "block0_" + table, lambda t: t.__setitem__(column, t[column].mask(t.index == 0, value)))
    reject(args)


@pytest.mark.parametrize("kind", ["missing", "extra", "pulses", "clusters", "members", "identity"])
def test_resealed_complete_frame_census(generated, kind):
    args, _ = generated
    def change(t):
        if kind == "missing":
            return t.iloc[:1]
        if kind == "extra":
            return pd.concat([t, t.iloc[:1]], ignore_index=True)
        t.loc[0, "event" if kind == "identity" else kind] += 1
    table_change(args, "block1_frames", change)
    reject(args)


@pytest.mark.parametrize("kind", ["one_ulp", "pe32", "negative_zero", "inverse_bunch"])
def test_exact_floating_storage_not_tolerance(generated, kind):
    args, _ = generated
    table = "block0_pulses"
    def change(t):
        if kind == "one_ulp":
            t.loc[1, "pe"] = np.nextafter(t.loc[1, "pe"], np.inf)
        elif kind == "pe32":
            t["pe"] = t.pe.astype("float32").astype("float64")
        elif kind == "negative_zero":
            t.loc[0, "pe"] = -0.
        else:
            # A tiny inverse-shift perturbation is not licensed by numeric closeness.
            t.loc[0, "time_relative"] = np.nextafter(t.loc[0, "time_relative"], np.inf)
    table_change(args, table, change)
    reject(args)


def test_empty_native_observation_is_valid_with_consistent_counts(generated):
    args, _ = generated
    for number in (0, 1):
        for name in ("clusters", "members"):
            table_change(args, f"block{number}_{name}", lambda t: t.iloc[:0])
        table_change(args, f"block{number}_frames", lambda t: t.assign(clusters=np.int64(0), members=np.int64(0)))
    def change(s):
        s["nonempty_real_refit_membership"] = False
        for record in s["records"]:
            record["counts"].update(clusters=0, members=0)
            for frame in record["frames"]:
                frame.update(clusters=0, members=0)
    summary_change(args, change)
    result = V.verify(**args)
    assert result["nonempty_real_refit_membership"] is False
    assert sum(r["pulses"] for r in result["counts"]) == 87
    assert all(r["clusters"] == r["members"] == 0 for r in result["counts"])


def test_floating_cluster_properties_are_not_independently_recomputed(generated):
    args, _ = generated
    table_change(args, "block0_clusters", lambda t: t.assign(slope=t.slope + .25, pe=t.pe + 3.))
    result = V.verify(**args)
    assert result["native_member_structure_exact"]
    assert result["cluster_scalars_independently_recomputed"] is False


def test_valid_alternative_native_order_is_hash_bound_not_recomputed(generated):
    args, _ = generated
    def change(t):
        t = t.iloc[::-1].reset_index(drop=True)
        t["member_order"] = np.arange(len(t), dtype=np.int64)
        return t
    table_change(args, "block0_members", change)
    result = V.verify(**args)
    assert result["native_member_order_independently_recomputed"] is False


@pytest.mark.parametrize("where", ["existing", "membership", "downstream", "prior", "symlink"])
def test_exclusive_report_outside_evidence(generated, where):
    args, saved = generated
    if where == "existing":
        args["out"].write_text("keep")
    elif where == "membership":
        args["out"] = args["directory"] / "report.json"
    elif where == "downstream":
        args["out"] = saved.downstream / "report.json"
    elif where == "prior":
        args["out"] = saved.verified.parent / "report.json"
    else:
        args["out"].symlink_to(saved.downstream / "absent")
    with pytest.raises(ValueError):
        V.verify(**args)
    if where == "existing":
        assert args["out"].read_text() == "keep"


@pytest.mark.parametrize("where", ["member", "receipt", "new_output", "source_pulse", "prior", "inherited_extra"])
def test_late_mutation_never_publishes(generated, monkeypatch, where):
    args, saved = generated
    path = {"member": args["directory"] / "block0_members.parquet", "receipt": args["directory"] / N.A.COMPLETE,
            "new_output": args["directory"] / "late.txt",
            "source_pulse": saved.downstream / "block0_stock_adc_refit_pulses.parquet",
            "prior": saved.verified, "inherited_extra": saved.membership / "late.txt"}[where]
    original = V.member_side_counts
    def mutate(*a, **k):
        result = original(*a, **k)
        path.write_bytes(b"late mutation")
        return result
    monkeypatch.setattr(V, "member_side_counts", mutate)
    reject(args)


def test_dtype_header_precedes_decode(generated, monkeypatch):
    args, _ = generated
    target = args["directory"] / "block0_members.parquet"
    table_change(args, "block0_members", lambda t: t.assign(pulse_idx=t.pulse_idx.astype(float)))
    original = pd.read_parquet
    def guarded(path, *a, **k):
        assert Path(path) != target, "decoded wrong dtype before metadata admission"
        return original(path, *a, **k)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    reject(args, "dtype changed")


@pytest.mark.parametrize("kind", ["columns", "rows", "groups", "bytes"])
def test_metadata_caps_precede_decode(tmp_path, monkeypatch, kind):
    path = tmp_path / "table.parquet"
    path.write_bytes(b"never decoded")
    values = dict(num_rows=1, num_row_groups=1, num_columns=1)
    values.update({"columns": {"num_columns": 2}, "rows": {"num_rows": 3},
                   "groups": {"num_row_groups": 4097}, "bytes": {}}[kind])
    meta = SimpleNamespace(**values, row_group=lambda i: SimpleNamespace(
        total_byte_size=129 * 1024**2 if kind == "bytes" else 1))
    class Header:
        metadata = meta
        schema_arrow = None
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
    monkeypatch.setattr(T.pq, "ParquetFile", lambda *a: Header())
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="cap/schema|byte cap"):
        T._table(path, {"i": "int64"}, 2)


def test_report_cap_prevents_publication(generated, monkeypatch):
    args, _ = generated
    monkeypatch.setattr(V, "CAP", 1)
    reject(args, "report byte cap")


@pytest.mark.parametrize("kind", ["zero_pe_counted", "poscalc", "negative_zero_sentinel", "populated_zero_pe"])
def test_positive_pe_side_contract_without_floating_recomputation(kind):
    identity = dict(input_sha256="a" * 64, run=7, subrun=2, event=2505, frame_id=2504)
    pulses = pd.DataFrame([dict(identity, idx=0, pe=0.), dict(identity, idx=1, pe=2.)])
    members = pd.DataFrame([dict(identity, cluster_idx=0, pulse_idx=0, chan=1),
                            dict(identity, cluster_idx=0, pulse_idx=1, chan=4)])
    clusters = pd.DataFrame([dict(identity, idx=0, sidehits0=1, sidehits1=0, sidepe0=2., sidepe1=0.,
                                 sidetime0=500., sidetime1=0., poscalc=False)])
    products = dict(pulses=pulses, members=members, clusters=clusters)
    V.member_side_counts(products)
    column, value = {"zero_pe_counted": ("sidehits1", 1), "poscalc": ("poscalc", True),
                     "negative_zero_sentinel": ("sidetime1", -0.), "populated_zero_pe": ("sidepe0", 0.)}[kind]
    clusters.loc[0, column] = value
    with pytest.raises(ValueError, match="side count|both sides|sentinel|positive PE"):
        V.member_side_counts(products)


@pytest.mark.parametrize("kind", ["arms", "seed", "seed_role"])
def test_resealed_receipt_context_cannot_change_protocol(generated, kind):
    args, _ = generated
    for name in (N.A.INCOMPLETE, N.A.COMPLETE):
        path = args["directory"] / name
        receipt = M.read_json(path)
        if kind == "arms":
            receipt["context"]["arms"] = ["stock"]
        elif kind == "seed":
            receipt["context"]["seeds"] = [1]
        else:
            receipt["context"]["batch_identity"]["seed_role"] = "new random exposure"
        write_json(path, receipt)
    repin(args)
    reject(args)


def test_physical_file_cap_precedes_parquet_metadata(tmp_path, monkeypatch):
    path = tmp_path / "oversized.parquet"
    with path.open("wb") as stream:
        stream.truncate(32 * 1024**2 + 1)
    monkeypatch.setattr(T.pq, "ParquetFile", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(ValueError, match="file byte cap"):
        T._table(path, {"i": "int64"}, 2)
