"""Synthetic retained-three-frame robustness, not same-input stock finder parity."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import stock_refit_comparison as S
from test_veto_refit_saved_coverage import SavedCoverage, dump, sha, typed
from test_veto_saved_qualified_report import reseal

C, N, R, V, A = S.C, S.N, S.R, S.V, S.A
TABLES = ("frames", "stock_members", "stock_clusters", "bridge", "pulses", "clusters", "members")


def forbidden(*args, **kwargs):
    pytest.fail("comparison attempted physics, prior verification, raw/track access, admission or coverage")


def guards(monkeypatch):
    modules = ((R, ("verify", "sources", "products")), (N, ("run", "dispatch", "authority", "admission")),
        (N.B, ("prepare",)), (N.P, ("run",)), (N.P.R, ("recover", "prepare_pulses")),
        (N.P.R.C, ("find_event", "_properties", "_find_clusters", "_check_coincidence_vec", "_check_coincidence")),
        (V, ("verify",)), (N.SR, ("verify",)), (S.M, ("run", "geometry_only")), (S.M.S, ("run",)),
        (N.I, ("run",)), (N.I.Q, ("run", "admission")), (N.I.Q.R, ("run_arm",)),
        (N.I.Q.R.Q, ("run_arm",)), (N.I.Q.R.W, ("synthesize",)),
        (N.I.Q.R.P, ("reconstruct", "reconstruct_matrix", "fit_buckets")),
        (N.I.Q.R.P.P, ("reconstruct", "reconstruct_full", "fit_gumbel")), (N.I.Q.R.P.LM, ("fit_gumbel",)),
        (N.I.Q.R.S, ("simulate_candidates", "optical_candidates", "thermal_candidates")),
        (N.I.Q.M.K, ("make_photons",)), (N.I.Q.M.P, ("read_input", "run", "verify_metadata", "block_probe")),
        (N.I.D, ("run", "run_arm")), (S.M.T, ("verify",)),
        (C.G, ("evaluate", "frozen_grid", "track_histogram", "gate", "coinc_mask")),
        (C, ("saved_grid", "run", "evaluate")))
    for module, names in modules:
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    import awkward as ak
    import uproot
    monkeypatch.setattr(uproot, "open", forbidden)
    monkeypatch.setattr(ak, "from_parquet", forbidden)


def zeros(types, n):
    return pd.DataFrame({k: np.zeros(n, dtype=t) for k, t in types.items()})


def synthetic_tables(frames):
    """Assert saved native structure directly; no finder, bridge preparation or verifier."""
    frames = typed(frames.assign(clusters=1, members=4), R.FRAME_TYPES)
    products = {k: [] for k in TABLES[1:]}
    for _, frame in frames.iterrows():
        n = int(frame.pulses)
        identity = {k: frame[k] for k in S.KEYS}
        p = zeros(R.NATIVE_PULSES, n).assign(**identity, idx=np.arange(n),
            chan=np.arange(n) * 4 + np.arange(n) % 2, starttdc=40, seq=0, wf=np.arange(n),
            time_relative=500., time_reco=500.125, pe=12.25)
        p.loc[0, "pe"] = 0.
        p.loc[1, "failed"], p.loc[2, "zerondf"] = True, True
        p = typed(p, R.NATIVE_PULSES)
        b = p[S.KEYS].assign(stock_idx=np.arange(n - 1, -1, -1), refit_idx=p.idx,
            wf=p.wf, chan=p.chan, starttdc=p.starttdc, bar=p.chan // 4, sipm=p.chan % 4,
            seq=p.seq, time=500., pe_stock=np.full(n, 12., dtype=np.float32),
            pe_fast=p.pe, time_reco=p.time_reco, failed=p.failed, zerondf=p.zerondf,
            flags=p.failed.astype("int64") + 8 * p.zerondf.astype("int64"), _merge="both")
        b.loc[0, "pe_stock"] = np.float32(0.)
        selected = b.iloc[[2, 0, 3, 1]].reset_index(drop=True)
        sm = zeros(S.STOCK_MEMBERS, 4)
        for key in S.IDS:
            sm[key] = selected[key]
        sm = sm.assign(idx=selected.stock_idx, bar=selected.bar, sipm=selected.sipm,
                       seq=selected.seq, chan=selected.chan, time=selected.time, pe=selected.pe_stock)
        sm = typed(sm, S.STOCK_MEMBERS)
        sc = zeros(S.STOCK_CLUSTERS, 1).assign(**{k: frame[k] for k in S.IDS},
            idx=0, start=490., end=510., avgtime=500., pe=36., sidehits0=1, sidehits1=2)
        sc = typed(sc, S.STOCK_CLUSTERS)
        c = zeros(V.CLUSTERS, 1).assign(**identity, idx=0, start=490.125, end=510.125,
            avgtime=500.125, pe=36.75, nLayers=3, sidehits0=1, sidehits1=2)
        c = typed(c, V.CLUSTERS)
        m = zeros(V.MEMBERS, 4).assign(**identity, cluster_idx=0, member_order=np.arange(4),
            pulse_idx=selected.refit_idx, chan=selected.chan, starttdc=selected.starttdc,
            seq=selected.seq, layer=selected.bar % 4, sector=0)
        m = typed(m, V.MEMBERS)
        for name, table in zip(TABLES[1:], (sm, sc, b, p, c, m)):
            products[name].append(table)
    return {"frames": frames, **{k: pd.concat(v, ignore_index=True) for k, v in products.items()}}


@pytest.fixture
def domain(monkeypatch):
    guards(monkeypatch)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    frames = pd.DataFrame(dict(input_sha256=["a" * 64] * 3, run=[7, 8, 7], subrun=[2, 2, 3],
        event=[19, 19, 19], frame_id=[10, 20, 30], pbtime=-224.46, intensity=1., pulses=[6, 7, 8]))
    return synthetic_tables(frames)


def compare(domain):
    return S.compare(**domain)


def reject_compare(domain, match=None):
    with pytest.raises((ValueError, AssertionError), match=match):
        compare(domain)


def test_pure_full_identity_nonidentity_bijection_and_unchanged_inputs(domain):
    before = {k: v.copy(deep=True) for k, v in domain.items()}
    rows, differences = compare(domain)
    assert rows.frame_id.tolist() == [10, 20, 30]
    assert rows.common_members.tolist() == [4, 4, 4]
    assert rows.stock_only_members.eq(0).all() and rows.refit_only_members.eq(0).all()
    assert rows.member_sets_exact.all() and differences == []
    assert rows.refit_minus_stock_pe.eq(.75).all()
    assert rows.refit_minus_stock_avgtime.eq(.125).all()
    assert not domain["bridge"].stock_idx.equals(domain["bridge"].refit_idx)
    for key, table in before.items():
        pd.testing.assert_frame_equal(domain[key], table, check_exact=True)


def test_actual_legacy_schema_and_absent_native_ordinals(domain):
    expected_members = {**dict.fromkeys(S.IDS, "uint32"), **dict.fromkeys(("idx", "bar", "sipm"), "int32"),
        "time": "float64", "pe": "float32", "seq": "int32", "w": "float64", "chan": "int32",
        **dict.fromkeys(("time_relative", "propagation_shift", "cluster_hit_time", "distance_to_flash_end",
                        "distance_to_digi_end_relative"), "float64")}
    expected_clusters = {**dict.fromkeys(S.IDS, "uint32"), "idx": "int32",
        **dict.fromkeys(("start", "end", "avgtime"), "float64"), "pe": "float32",
        "sidehits0": "int32", "sidehits1": "int32"}
    assert {k: str(v) for k, v in domain["stock_members"].dtypes.items()} == expected_members
    assert {k: str(v) for k, v in domain["stock_clusters"].dtypes.items()} == expected_clusters
    assert not {"member_order", "member_ordinal", "cluster_idx"} & set(domain["stock_members"])
    compare(domain)


@pytest.mark.parametrize("table", ["stock_members", "stock_clusters", "bridge", "pulses", "clusters"])
def test_unordered_legacy_and_lookup_tables_are_not_positional(domain, table):
    expected, _ = compare(domain)
    domain[table] = domain[table].iloc[::-1].reset_index(drop=True)
    actual, differences = compare(domain)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert differences == []


def test_native_member_permutation_without_ordinals_is_rejected(domain):
    domain["members"] = domain["members"].iloc[::-1].reset_index(drop=True)
    reject_compare(domain, "ordinal")


@pytest.mark.parametrize("field", ["member_order", "member_ordinal", "cluster_idx"])
def test_missing_legacy_ordinals_cannot_be_invented(domain, field):
    domain["stock_members"][field] = np.arange(len(domain["stock_members"]), dtype="int64")
    reject_compare(domain, "columns")


def test_changed_membership_is_a_contrast_not_physics_parity_failure(domain):
    old = domain["members"].loc[0, "pulse_idx"]
    replacement = domain["bridge"].loc[domain["bridge"].frame_id.eq(10) & domain["bridge"].refit_idx.eq(4)].iloc[0]
    domain["members"].loc[0, ["pulse_idx", *N.P.PULSE_KEYS]] = [replacement.refit_idx, *replacement[N.P.PULSE_KEYS]]
    rows, differences = compare(domain)
    assert rows.common_members.tolist() == [3, 4, 4]
    assert rows.stock_only_members.tolist() == rows.refit_only_members.tolist() == [1, 0, 0]
    assert rows.member_sets_exact.tolist() == [False, True, True]
    expected_old = int(domain["bridge"].loc[domain["bridge"].frame_id.eq(10) & domain["bridge"].refit_idx.eq(old), "stock_idx"].iloc[0])
    assert {(d["stock_idx"], d["_merge"]) for d in differences} == {(expected_old, "left_only"), (int(replacement.stock_idx), "right_only")}
    assert all(d["frame_id"] == 10 and d["input_sha256"] == "a" * 64 for d in differences)


@pytest.mark.parametrize("table", ["stock_clusters", "clusters"])
@pytest.mark.parametrize("case", ["missing", "multiple", "nonzero_index"])
def test_one_cluster_is_required_without_legacy_parent_assignment(domain, table, case):
    if case == "missing":
        domain[table] = domain[table].iloc[1:].reset_index(drop=True)
    elif case == "multiple":
        domain[table] = pd.concat([domain[table], domain[table].iloc[:1].assign(idx=np.int32(1))], ignore_index=True)
    else:
        domain[table].loc[0, "idx"] = 1
    reject_compare(domain, "cluster")


@pytest.mark.parametrize("table", list(TABLES))
def test_duplicate_identities_fail_closed(domain, table):
    domain[table] = pd.concat([domain[table], domain[table].iloc[:1]], ignore_index=True)
    reject_compare(domain)


@pytest.mark.parametrize("table", ["stock_members", "bridge", "pulses", "members"])
def test_missing_bridge_or_native_link_is_rejected(domain, table):
    domain[table] = domain[table].iloc[1:].reset_index(drop=True)
    reject_compare(domain)


@pytest.mark.parametrize("table,key", [(t, k) for t in ("bridge", "pulses", "members", "clusters")
    for k in ("input_sha256", "frame_id", "run", "subrun", "event")] +
    [(t, k) for t in ("stock_members", "stock_clusters") for k in ("run", "subrun", "event")])
def test_wrong_full_source_or_event_key_is_rejected(domain, table, key):
    domain[table].loc[0, key] = "f" * 64 if key == "input_sha256" else 999
    reject_compare(domain)


def test_same_legacy_event_under_two_sources_is_ambiguous(domain):
    domain["frames"].loc[1, S.IDS] = domain["frames"].loc[0, S.IDS].to_numpy()
    domain["frames"].loc[1, "input_sha256"] = "b" * 64
    reject_compare(domain, "ambiguous legacy full event")


@pytest.mark.parametrize("field", ["stock_idx", "refit_idx"])
def test_bridge_indices_must_be_bijective(domain, field):
    domain["bridge"].loc[1, field] = domain["bridge"].loc[0, field]
    reject_compare(domain, "bijection")


@pytest.mark.parametrize("field", ["stock_idx", "refit_idx", "wf", "starttdc", "flags"])
def test_bridge_integer_keys_cannot_be_float(domain, field):
    domain["bridge"][field] = domain["bridge"][field].astype("float64")
    reject_compare(domain, "integer")


@pytest.mark.parametrize("field", ["wf", "chan", "starttdc", "seq", "pe_fast", "time_reco", "failed", "zerondf", "flags", "_merge"])
def test_saved_native_bridge_payload_is_exact(domain, field):
    table = domain["bridge"]
    if field == "_merge":
        table.loc[0, field] = "left_only"
    elif field in ("failed", "zerondf"):
        table.loc[0, field] = not table.loc[0, field]
    else:
        table.loc[0, field] += 1
    reject_compare(domain)


@pytest.mark.parametrize("field", ["bar", "sipm", "seq", "chan", "pe", "time"])
def test_copied_stock_member_values_may_not_change(domain, field):
    domain["stock_members"].loc[0, field] += 1
    reject_compare(domain, "stock member")


@pytest.mark.parametrize("table,field,partner,pfield", [
    ("stock_members", "time", "bridge", "time"), ("stock_members", "pe", "bridge", "pe_stock"),
    ("bridge", "pe_fast", "pulses", "pe"), ("bridge", "time_reco", "pulses", "time_reco")])
@pytest.mark.parametrize("change", ["ulp", "signed_zero"])
def test_copied_floats_require_storage_bits_not_tolerance(domain, table, field, partner, pfield, change):
    source = domain[table]
    target = domain[partner]
    if table == "stock_members":
        index = target.index[target.frame_id.eq(10) & target.stock_idx.eq(source.loc[0, "idx"])][0]
    else:
        index = 0
    if change == "signed_zero":
        source.loc[0, field], target.loc[index, pfield] = -0., 0.
    else:
        dtype = source[field].dtype.type
        source.loc[0, field] = np.nextafter(dtype(source.loc[0, field]), dtype(np.inf))
    reject_compare(domain, "storage")


@pytest.mark.parametrize("field", ["pulse_idx", "cluster_idx", "member_order", "chan", "starttdc", "seq"])
def test_native_member_link_and_order_are_not_reconstructed(domain, field):
    domain["members"].loc[0, field] += 100
    reject_compare(domain)


def test_ignored_native_pulse_changes_the_domain(domain):
    domain["pulses"].loc[0, "channel_ignored"] = True
    reject_compare(domain, "ignored")


@pytest.mark.parametrize("table,field", [("frames", "members"), ("frames", "pulses"), ("frames", "clusters"),
    ("stock_clusters", "sidehits0"), ("stock_clusters", "sidehits1")])
def test_saved_frame_and_positive_pe_side_counts_are_exact(domain, table, field):
    domain[table].loc[0, field] += 1
    reject_compare(domain, "census|count|cluster|pulse")


@pytest.mark.parametrize("table,field,dtype", [("stock_members", "run", "int64"), ("stock_members", "idx", "int64"),
    ("stock_members", "pe", "float64"), ("stock_members", "time", "float32"),
    ("stock_clusters", "pe", "float64"), ("stock_clusters", "sidehits0", "int64")])
def test_legacy_storage_schema_is_not_promoted(domain, table, field, dtype):
    domain[table][field] = domain[table][field].astype(dtype)
    reject_compare(domain, "dtype")


@pytest.mark.parametrize("table", list(TABLES))
def test_pure_comparison_caps_before_any_schema_work(domain, table):
    count = 5 if table == "frames" else 2049
    domain[table] = pd.DataFrame({"not_a_schema": np.zeros(count)})
    reject_compare(domain, "cap")


class SavedComparison(SavedCoverage):
    """Real local artifacts and manual synthetic certificates; no inherited verification run."""

    def __init__(self, root, monkeypatch):
        super().__init__(root, monkeypatch)
        self.boundary = root / "boundary"
        self.boundary.mkdir()
        self.blocks = [synthetic_tables(f) for f in self.frames]
        self.frames = [b["frames"] for b in self.blocks]
        self.clusters = [b["clusters"] for b in self.blocks]
        self.boundary_report = dict(schema="thread-d-optical-boundary-v1", blocks=[], output_sha256={})
        for number, block in enumerate(self.blocks):
            for kind in ("frames", "pulses", "clusters", "members", "bridge"):
                name = "saved_bridge" if kind == "bridge" else kind
                block[kind].to_parquet(self.directory / f"block{number}_{name}.parquet", index=False)
            for kind in ("stock_members", "stock_clusters"):
                name = f"block{number}_{kind}.parquet"
                block[kind].to_parquet(self.boundary / name, index=False)
                self.boundary_report["output_sha256"][name] = sha(self.boundary / name)
            self.boundary_report["blocks"].append(dict(block=number, targets=N.TARGETS[number],
                input_sha256={f"/never-open/block{number}/mixed.root": N.INPUTS[number]},
                stock_clusters=block["stock_clusters"].to_dict("records")))
        old_role = self.prior["source_reports"]["boundary"]
        self.refit["inputs"].pop(old_role["path"])
        self.prior["source_reports"]["boundary"] = dict(path=str(self.boundary / "summary.json"), sha256="")
        self.refit["records"] = self.records()
        self.prior["frames"] = pd.concat(self.frames).to_dict("records")
        self.prior["counts"] = [dict(block=r["block"], **r["counts"]) for r in self.refit["records"]]
        self.prior["stock_refit_differences"] = [dict(block=i, pe_changed=int(f.pulses.sum()) - len(f),
            time_changed=int(f.pulses.sum()), failed_flag_changed=0, zerondf_flag_changed=0)
            for i, f in enumerate(self.frames)]
        self.repin_boundary()
        self.args["out"] = root / "comparison.json"

    def repin_boundary(self):
        path = self.boundary / "summary.json"
        digest = dump(path, self.boundary_report)
        self.prior["source_reports"]["boundary"]["sha256"] = digest
        self.refit["inputs"][str(path)] = digest
        self.repin()

    def run(self, **changes):
        self.reads = []
        with self.monkeypatch.context() as guard:
            guards(guard)
            allowed = {self.directory / "registry.parquet"}
            allowed |= {self.directory / f"block{i}_{k}.parquet" for i in (0, 1)
                        for k in ("frames", "clusters", "pulses", "members", "saved_bridge")}
            allowed |= {self.boundary / f"block{i}_{k}.parquet" for i in (0, 1)
                        for k in ("stock_members", "stock_clusters")}
            read, metadata, path_open = pd.read_parquet, V.pq.ParquetFile, Path.open

            def bounded_read(path, *args, **kwargs):
                assert Path(path) in allowed, f"unexpected table decode: {path}"
                self.reads.append(Path(path))
                return read(path, *args, **kwargs)

            def bounded_metadata(path, *args, **kwargs):
                assert Path(path) in allowed, f"unexpected Parquet metadata: {path}"
                return metadata(path, *args, **kwargs)

            def no_raw_or_tracks(path, *args, **kwargs):
                assert Path(path) != self.selection and Path(path).suffix != ".root", f"forbidden open/hash: {path}"
                return path_open(path, *args, **kwargs)

            guard.setattr(pd, "read_parquet", bounded_read)
            guard.setattr(V.pq, "ParquetFile", bounded_metadata)
            guard.setattr(Path, "open", no_raw_or_tracks)
            return S.run(**(self.args | changes))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return SavedComparison(tmp_path, monkeypatch)


def reject_run(saved, match=None):
    with pytest.raises((ValueError, AssertionError, FileNotFoundError, KeyError), match=match):
        saved.run()
    assert not saved.args["out"].exists()


def test_runner_three_real_parquet_frames_frozen_inputs_and_narrow_claims(saved):
    before = {p: sha(p) for p in saved.root.rglob("*") if p.is_file()}
    report = saved.run()
    assert json.loads(saved.args["out"].read_text()) == report
    assert before == {p: sha(p) for p in before}
    assert {p for p in saved.root.rglob("*") if p.is_file()} == {*before, saved.args["out"]}
    assert (report["registry_frames"], report["unique_frames"]) == (9000, 3)
    assert report["schema"] == S.SCHEMA and report["scope"] == S.SCOPE
    rows = pd.DataFrame(report["rows"])
    assert rows.clusters.tolist() == [1, 1, 1] and rows.pulses.sum() == 87
    assert rows.common_members.tolist() == [4, 4, 4]
    assert report["all_member_sets_exact"] is True and report["membership_differences"] == []
    assert report["F"] is report["finite_library_uncertainty"] is None
    assert all(report[k] is False for k in S.FALSE_FIELDS)
    for key in ("same_pulse_stock_validation", "stock_native_order_available", "stock_native_order_compared",
                "coverage_recomputed", "prior_verifier_rerun", "source_admission_repeated", "execution_admitted"):
        assert report[key] is False
    assert "member_ordinal" in report["missing_legacy_fields"]
    assert "cluster_idx on members" in report["missing_legacy_fields"]
    assert "no invented native ordinal" in report["parent_cluster_assignment"]
    assert report["stock_refit_differences"] == saved.prior["stock_refit_differences"]
    assert report["unordered_stock_refit_membership_compared"] is True
    assert report["inherited_native_structure_verified"] is report["saved_artifact_pins_exact"] is True
    for path, text in report["code_snapshots"].items():
        assert hashlib.sha256(text.encode()).hexdigest() == report["code_sha256"][path] == sha(path)
    assert str(Path(S.__file__)) in report["code_snapshots"]
    assert saved.args["out"].stat().st_size <= S.CAP
    assert len(set(saved.reads)) == 15
    assert not {"mean", "uncertainty", "coverage"} & report.keys()


def test_runner_different_member_sets_report_without_requiring_stock_physics_parity(saved):
    table = saved.blocks[0]["members"].copy()
    p = saved.blocks[0]["pulses"].iloc[4]
    table.loc[0, ["pulse_idx", *N.P.PULSE_KEYS]] = [p.idx, *p[N.P.PULSE_KEYS]]
    table.to_parquet(saved.directory / "block0_members.parquet", index=False)
    saved.repin()
    result = saved.run()
    assert result["all_member_sets_exact"] is False
    assert len(result["membership_differences"]) == 2
    assert result["rows"][0]["common_members"] == 3
    assert result["same_pulse_stock_validation"] is False and result["F"] is None


@pytest.mark.parametrize("pin", ["expected_summary_sha256", "expected_verified_sha256", "expected_complete_sha256"])
@pytest.mark.parametrize("case", ["missing", "wrong"])
def test_runner_requires_three_external_pins(saved, pin, case):
    if case == "missing":
        saved.args.pop(pin)
        with pytest.raises(TypeError, match=pin):
            saved.run()
        assert not saved.args["out"].exists()
    else:
        saved.args[pin] = "0" * 64
        reject_run(saved, "changed bound")


@pytest.mark.parametrize("case", ["source_unbound", "source_changed", "snapshot_missing", "snapshot_changed", "generation_map"])
def test_runner_inherited_source_binding_and_snapshot_integrity(saved, case):
    if case == "source_unbound":
        saved.prior["source_reports"]["boundary"]["sha256"] = "0" * 64
        saved.repin_prior()
    elif case == "source_changed":
        (saved.boundary / "summary.json").write_text("changed")
    elif case.startswith("snapshot"):
        key = next(iter(saved.prior["code_snapshots"]))
        if case == "snapshot_missing":
            saved.prior["code_snapshots"].pop(key)
        else:
            saved.prior["code_snapshots"][key] += "\n"
        saved.repin_prior()
    else:
        saved.refit["code_sha256"] = {}
        saved.repin()
    reject_run(saved)


@pytest.mark.parametrize("case", ["extra", "missing_complete", "receipt", "pending", "snapshot", "snapshot_resealed"])
def test_runner_real_transaction_receipts_and_snapshot_files(saved, case):
    if case == "missing_complete":
        (saved.directory / A.COMPLETE).unlink()
    else:
        name = {"extra": "undeclared.txt", "receipt": A.COMPLETE, "pending": A.PENDING,
                "snapshot": "code0.txt", "snapshot_resealed": "code0.txt"}[case]
        (saved.directory / name).write_bytes(b"changed")
        if case == "snapshot_resealed":
            reseal(saved.directory)
            saved.args["expected_complete_sha256"] = sha(saved.directory / A.COMPLETE)
    reject_run(saved)


@pytest.mark.parametrize("case", ["table_pin", "cluster_observation", "raw_identity", "targets"])
def test_runner_legacy_source_contract(saved, case):
    if case == "table_pin":
        path = saved.boundary / "block0_stock_members.parquet"
        table = pd.read_parquet(path)
        table.loc[0, "time"] += 1
        table.to_parquet(path, index=False)
    else:
        record = saved.boundary_report["blocks"][0]
        if case == "cluster_observation":
            record["stock_clusters"][0]["pe"] += 1
        elif case == "raw_identity":
            record["input_sha256"] = {"/never-open/mixed.root": "0" * 64}
        else:
            record["targets"] = []
        saved.repin_boundary()
    reject_run(saved)


@pytest.mark.parametrize("target", ["bridge", "stock_members", "verified", "summary", "complete"])
def test_runner_late_input_change_prevents_publication(saved, monkeypatch, target):
    path = {"bridge": saved.directory / "block0_saved_bridge.parquet",
        "stock_members": saved.boundary / "block0_stock_members.parquet", "verified": saved.verified,
        "summary": saved.directory / "summary.json", "complete": saved.directory / A.COMPLETE}[target]
    original = S.compare
    calls = []

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if not calls:
            path.write_bytes(path.read_bytes() + b"\n")
        calls.append(True)
        return result

    monkeypatch.setattr(S, "compare", changed)
    reject_run(saved, "changed|differs")


def test_runner_output_cap_is_checked_before_new_file(saved, monkeypatch):
    monkeypatch.setattr(S, "CAP", 1)
    reject_run(saved, "cap")


def test_runner_exclusive_new_file_never_overwrites_existing_output(saved):
    saved.args["out"].write_bytes(b"keep existing report")
    with pytest.raises((ValueError, FileExistsError), match="new output|exist"):
        saved.run()
    assert saved.args["out"].read_bytes() == b"keep existing report"


@pytest.mark.parametrize("location", ["directory", "boundary", "sources", "membership"])
def test_runner_output_cannot_enter_inherited_artifact_directories(saved, location):
    saved.args["out"] = getattr(saved, location) / "new-report.json"
    reject_run(saved, "outside")


def test_runner_late_output_symlink_retarget_cannot_write_into_evidence(saved, monkeypatch):
    safe = saved.root / "safe"
    safe.mkdir()
    alias = saved.root / "alias"
    alias.symlink_to(safe, target_is_directory=True)
    saved.args["out"] = alias / "new-report.json"
    original = S.compare

    def retarget(*args, **kwargs):
        result = original(*args, **kwargs)
        if alias.resolve() == safe:
            alias.unlink()
            alias.symlink_to(saved.boundary, target_is_directory=True)
        return result

    monkeypatch.setattr(S, "compare", retarget)
    reject_run(saved, "outside")
    assert not (safe / "new-report.json").exists()
    assert not (saved.boundary / "new-report.json").exists()


def test_empty_frame_domain_is_not_a_vacuous_comparison(domain):
    domain["frames"] = domain["frames"].iloc[:0]
    reject_compare(domain, "cap")


@pytest.mark.parametrize("field", ["time", "pe"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_legacy_values_are_rejected(domain, field, value):
    domain["stock_members"].loc[0, field] = value
    reject_compare(domain, "missing|nonfinite")


@pytest.mark.parametrize("field", ["stock_idx", "refit_idx", "pe_stock", "time_reco"])
def test_saved_bridge_fields_cannot_be_omitted(domain, field):
    domain["bridge"] = domain["bridge"].drop(columns=field)
    reject_compare(domain, "bridge fields")


@pytest.mark.parametrize("field", ["pulses", "clusters"])
def test_runner_per_frame_counts_cannot_hide_behind_unchanged_aggregate(saved, field):
    saved.frames[1].loc[0, field] += 1
    saved.frames[1].loc[1, field] -= 1
    saved.frames[1].to_parquet(saved.directory / "block1_frames.parquet", index=False)
    saved.refit["records"] = saved.records()
    saved.prior["frames"] = pd.concat(saved.frames).to_dict("records")
    saved.prior["counts"] = [dict(block=r["block"], **r["counts"]) for r in saved.refit["records"]]
    saved.repin()
    reject_run(saved, "census|count|cluster|pulse")


@pytest.mark.parametrize("table,column", [("saved_bridge", "time_reco"), ("members", "chan"), ("pulses", "pe")])
def test_runner_resealed_native_payload_must_still_match_bridge(saved, table, column):
    path = saved.directory / f"block0_{table}.parquet"
    value = pd.read_parquet(path)
    value.loc[0, column] += 1
    value.to_parquet(path, index=False)
    saved.repin()
    reject_run(saved, "bridge|structural")


def test_missing_selected_tracks_are_not_opened_or_rehashed(saved):
    saved.selection.unlink()
    result = saved.run()
    assert result["selected_tracks_read"] is False
    assert str(saved.selection) not in result["inputs"]


def test_source_inventory_reports_actual_legacy_dtypes_without_promoting_ordinals(saved, monkeypatch):
    guards(monkeypatch)
    before = {p: sha(p) for p in saved.root.rglob("*") if p.is_file()}
    result = S.inspect_schema(saved.boundary, sha(saved.boundary / "summary.json"))
    assert result["schema"] == "stock-refit-schema-inventory-v1"
    assert result["physics_executed"] is False
    assert len(result["tables"]) == 4
    for number in (0, 1):
        for name in ("stock_members", "stock_clusters"):
            table = saved.blocks[number][name]
            inventory = result["tables"][f"block{number}_{name}.parquet"]
            assert inventory == dict(rows=len(table), columns={k: str(v) for k, v in table.dtypes.items()},
                                     identities=table[S.IDS].drop_duplicates().to_dict("records"))
    assert before == {p: sha(p) for p in before}
    assert not saved.args["out"].exists()


@pytest.mark.parametrize("case", ["pin", "table"])
def test_schema_inventory_still_requires_pinned_bytes(saved, monkeypatch, case):
    guards(monkeypatch)
    pin = sha(saved.boundary / "summary.json")
    if case == "pin":
        pin = "0" * 64
    else:
        path = saved.boundary / "block1_stock_members.parquet"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed bound"):
        S.inspect_schema(saved.boundary, pin)


def test_runner_late_source_snapshot_change_is_checked_without_touching_frozen_code(saved, monkeypatch):
    original = C.bounded_bytes
    source = Path(S.__file__)
    reads = []

    def changed(path, *args, **kwargs):
        payload = original(path, *args, **kwargs)
        if Path(path) == source:
            reads.append(Path(path))
            if len(reads) == 3:
                return payload + b"\n"
        return payload

    monkeypatch.setattr(C, "bounded_bytes", changed)
    reject_run(saved, "source snapshot")
    assert len(reads) == 3


def test_runner_output_parent_cannot_be_a_source_directory(saved, monkeypatch):
    directory = saved.root / "synthetic_code"
    directory.mkdir()
    source = directory / "source.txt"
    source.write_text("synthetic additional source binding\n")
    paths = C.code_paths()
    monkeypatch.setattr(C, "code_paths", lambda: [*paths, source])
    saved.args["out"] = directory / "report.json"
    reject_run(saved, "outside")


def test_full_source_identity_can_differ_when_legacy_event_identity_is_unique(domain):
    for table in ("frames", "bridge", "pulses", "clusters", "members"):
        domain[table].loc[domain[table].frame_id.eq(20), "input_sha256"] = "b" * 64
    rows, differences = compare(domain)
    assert rows.input_sha256.tolist() == ["a" * 64, "b" * 64, "a" * 64]
    assert rows.member_sets_exact.all() and differences == []


def test_bridge_duplicate_column_names_are_ambiguous(domain):
    domain["bridge"] = pd.concat([domain["bridge"], domain["bridge"][["stock_idx"]]], axis=1)
    reject_compare(domain, "bridge fields")


def test_native_flags_are_preserved_not_used_to_filter_members(domain):
    assert domain["pulses"].failed.any() and domain["pulses"].zerondf.any()
    rows, differences = compare(domain)
    assert rows.common_members.tolist() == [4, 4, 4] and differences == []


def test_runner_new_file_appearing_late_is_not_overwritten(saved, monkeypatch):
    original = S.compare

    def appeared(*args, **kwargs):
        result = original(*args, **kwargs)
        saved.args["out"].write_bytes(b"concurrent writer")
        return result

    monkeypatch.setattr(S, "compare", appeared)
    with pytest.raises((ValueError, FileExistsError), match="new output|exist"):
        saved.run()
    assert saved.args["out"].read_bytes() == b"concurrent writer"


def test_runner_actual_publication_uses_exclusive_binary_mode(saved, monkeypatch):
    original = Path.open
    modes = []

    def observed(path, mode="r", *args, **kwargs):
        if Path(path) == saved.args["out"]:
            modes.append(mode)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed)
    saved.run()
    assert modes == ["xb"]
