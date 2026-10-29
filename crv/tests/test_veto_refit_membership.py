"""Synthetic saved-refit runner isolation; real local Parquets, pins and atomic receipts only."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import refit_membership as N


GATES = ("saved_artifact_protocol_exact", "qualified_frame_census_exact", "source_transactions_exact",
         "canonical_pulses_exact", "native_member_structure_exact")


def forbidden(*args, **kwargs):
    pytest.fail("runner attempted upstream physics, raw access, verifier rerun or fallback")


def sha(path):
    return N.M.S.C.sha256(path)


def write_json(path, value):
    path.write_bytes(N.A._canonical(value))
    return sha(path)


def empty(types):
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in types.items()})


class SavedRefits:
    """Fixture assertions are not real prior verification or physical admission evidence."""

    def __init__(self, root, monkeypatch):
        self.root = root / "evidence"
        self.root.mkdir()
        self.aux = root / "authority_inputs"
        self.aux.mkdir()
        self.downstream, self.boundary = self.root / "downstream", self.root / "boundary"
        self.downstream.mkdir()
        self.boundary.mkdir()
        self.out = root / "new_membership"
        self.verified = self.root / "downstream_verified.json"
        self.membership_verified = self.root / "membership_verified.json"
        self.policy = self.aux / "ignore_policy.txt"
        self.policy.write_text("synthetic-only fixed ignoreChannels: true\n")
        self.counters = self.aux / "crv_counters_run1a_v01.txt"
        self.counters.write_text("synthetic 1280-bar geometry, not detector geometry\n")
        inputs = [N.INPUTS[0], N.INPUTS[1], "f" * 64]
        self.registry = N.F.freeze(pd.DataFrame({
            "input_sha256": np.repeat(inputs, 3000), "run": 7, "subrun": 2,
            "event": np.tile(np.arange(1, 3001, dtype=np.int64), 3),
            "pbtime": -224.46, "intensity": 1.,
        }))
        monkeypatch.setattr(N, "REGISTRY_SHA", N.F.digest(self.registry))
        self.calibration = pd.DataFrame({
            "input_sha256": np.repeat(inputs, 5120), "chan": np.tile(np.arange(5120, dtype=np.int64), 3),
            "pedestal": 100., "pulsearea": 1000., "pulseheight": 20., "timeoffset": 2.,
            "status": np.zeros(15360, dtype=np.int64), "roc": 1, "feb": 1,
        })
        self.calibration_path = self.aux / "calibration.parquet"
        self.calibration.to_parquet(self.calibration_path, index=False)
        self.geometry = SimpleNamespace(sector=np.zeros(1280, dtype=np.int64),
                                        layer=np.arange(1280, dtype=np.int64) % 4,
                                        sector_type=np.array([0]), sector_names=["T1"])
        self.geometry_calls = []

        def geometry(sources):
            self.geometry_calls.append(sources)
            assert sources == {"counters": self.counters}
            return self.geometry

        monkeypatch.setattr(N.M, "geometry_only", geometry)
        policy_sha = sha(self.policy)
        monkeypatch.setattr(N.M, "ignore_bindings", lambda: {str(self.policy): policy_sha})
        self.code = {str(path): sha(path) for path in N.M.code_paths()}
        self.summary = dict(
            schema=N.M.SCHEMA, scope=N.M.SCOPE,
            hashes={name: str(i + 1) * 64 for i, name in enumerate(sorted(N.V.HASH_KEYS))},
            inputs={str(p): sha(p) for p in (self.calibration_path, self.counters)}, code_sha256=self.code,
            registry_sha256=N.REGISTRY_SHA, unique_frames=3, ledger_rows=0,
            seeds=list(N.M.S.SEEDS), arms=list(N.M.S.ARMS), counts={}, comparisons={}, grid={},
            finder_config=asdict(N.M.C.finder_config()), coincidence_limits=asdict(N.P.Limits()),
            ignore_channels=True, ignored_pulse_policy=N.V.IGNORED_POLICY, output_cap_bytes=N.M.OUTPUT_CAP,
            memory_before={}, memory_after={}, coincidence_coverage_seconds=0.,
            **dict.fromkeys(N.V.FALSE_FIELDS, False), **dict.fromkeys(N.V.NULL_FIELDS),
        )
        assert set(self.summary) == N.V.SUMMARY_KEYS
        self.kwargs = dict(downstream=self.downstream, boundary=self.boundary, verified=self.verified,
                           membership_verified=self.membership_verified, out=self.out)
        self.generation = 0
        self.publish_authority()
        self.d = dict(schema="thread-d-saved-photon-downstream-v1", blocks=[], output_sha256={})
        self.b = dict(schema="synthetic-boundary-fixture", blocks=[], output_sha256={})
        self.prepared, self.raw = {}, {}
        for number in (0, 1):
            self.make_block(number)
            self.d["blocks"].append(dict(block=number, targets=N.TARGETS[number]))
            self.b["blocks"].append(dict(block=number, targets=N.TARGETS[number],
                                        input_sha256={f"/never-open/block{number}/mixed.root": N.INPUTS[number]}))
        self.commit_reports()
        self.prepares, self.finder_calls, self.reads = [], [], []
        self.original_prepare = N.B.prepare
        monkeypatch.setattr(N.B, "prepare", self.prepare)
        monkeypatch.setattr(N.P, "run", self.finder)
        read_parquet = pd.read_parquet

        def bounded_read(path, *args, **kwargs):
            path = Path(path)
            allowed = {self.membership / "registry.parquet", self.calibration_path}
            allowed |= {self.downstream / name for name in self.d["output_sha256"]}
            allowed |= {self.boundary / name for name in self.b["output_sha256"]}
            assert path in allowed or path.parent == self.out, f"unexpected decode: {path}"
            self.reads.append(path)
            return read_parquet(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", bounded_read)

    def publish_authority(self):
        self.membership = self.root / f"membership{self.generation}"
        self.generation += 1
        tables = {name: empty(types) for name, types in N.V.TABLE_TYPES.items()}
        tables.update(registry=self.registry, selected=self.registry.iloc[:3].copy())
        context = dict(registry_sha256=self.summary["registry_sha256"],
                       batch_identity={"contract": N.M.SCHEMA, **self.summary["hashes"]},
                       arms=list(N.M.S.ARMS), seeds=list(N.M.S.SEEDS))
        names = ["summary.json", *(name + ".parquet" for name in tables),
                 *(f"code{i}.txt" for i in range(len(self.code)))]
        with N.A.BatchArtifacts(self.membership, max_bytes=N.M.OUTPUT_CAP,
                              context=context, expected_artifacts=names) as tx:
            tx.write_json("summary.json", self.summary)
            for name, table in tables.items():
                tx.write_parquet(name + ".parquet", table)
            for i, path in enumerate(self.code):
                with tx.open_artifact(f"code{i}.txt") as stream:
                    stream.write(Path(path).read_bytes())
        self.mv = dict(schema=N.V.SCHEMA, summary_sha256=sha(self.membership / "summary.json"),
                       registry_sha256=N.REGISTRY_SHA, registry_frames=9000,
                       code_sha256=sha(N.V.__file__), **dict.fromkeys(GATES, True))
        self.kwargs.update(membership=self.membership, expected_membership_sha256=self.mv["summary_sha256"],
                           expected_membership_verified_sha256=write_json(self.membership_verified, self.mv))

    def commit_reports(self):
        bsha = write_json(self.boundary / "summary.json", self.b)
        self.d["input_summary_sha256"] = bsha
        dsha = write_json(self.downstream / "summary.json", self.d)
        self.dv = dict(schema="thread-d-downstream-report-v1", input_summary_sha256=dsha,
                       boundary_summary_sha256=bsha, original_inputs_verified=True)
        self.kwargs.update(expected_boundary_sha256=bsha, expected_downstream_sha256=dsha,
                           expected_verified_sha256=write_json(self.verified, self.dv))

    def persist(self, number, suffix, table, boundary=False):
        root, report = (self.boundary, self.b) if boundary else (self.downstream, self.d)
        name = f"block{number}_{suffix}.parquet"
        table.to_parquet(root / name, index=False)
        report["output_sha256"][name] = sha(root / name)
        return root / name

    def make_block(self, number):
        digest = N.INPUTS[number]
        selected = self.registry.loc[self.registry.input_sha256.eq(digest)
                                     & self.registry.event.isin(N.TARGETS[number])].reset_index(drop=True)
        nw, extra = ((23, 2), (57, 5))[number]
        events = selected[["event", "pbtime"]].copy()
        w = pd.DataFrame({"wf": np.arange(nw, dtype=np.int64),
                          "event": selected.event.iloc[np.zeros(nw, dtype=int) if number == 0
                                                       else (np.arange(nw) >= 28).astype(int)].to_numpy(),
                          "chan": np.arange(nw, dtype=np.int64) * 4, "starttdc": 40})
        p = w.loc[w.index.repeat(1 + (w.index < extra).astype(int))].reset_index(drop=True)
        p["seq"] = p.groupby("wf", sort=False).cumcount().astype(np.int64)
        p["time"], p["time_relative"] = 500., 502.
        pb = p.event.map(selected.set_index("event").pbtime).to_numpy().astype(np.float32).astype(float)
        p["time_reco"], p["pe"] = p.time_relative.to_numpy() - pb, 12.1234567890123
        p["channel_ignored"] = False
        p["failed"], p["zerondf"] = p.index == 1, p.index == 2
        qualified = p.merge(selected[N.P.FRAME_KEYS], on="event", sort=False, validate="many_to_one")
        qualified["refit_idx"] = qualified.groupby("frame_id", sort=False).cumcount().astype(np.int64)
        stock = qualified[["run", "subrun", "event", "refit_idx", "chan", "seq", "time_reco", "pe"]].rename(
            columns={"refit_idx": "idx", "time_reco": "time"})
        stock["bar"], stock["sipm"] = stock.chan // 4, stock.chan % 4
        stock["flags"], stock["pe"] = 0, stock.pe.astype(np.float32)
        stock = stock.drop(columns="chan")
        comparison = stock.rename(columns={"pe": "pe_stock"}).copy()
        for name in ("chan", "wf", "time_reco", "failed", "zerondf"):
            comparison[name] = qualified[name]
        comparison["pe_fast"], comparison["_merge"] = qualified.pe, "both"
        comparison["delta_time"] = comparison.time_reco - comparison.time
        comparison["delta_pe"] = comparison.pe_fast - comparison.pe_stock
        conditions = events[["event"]].merge(self.calibration.loc[self.calibration.input_sha256.eq(digest)]
                                               .drop(columns="input_sha256"), how="cross")
        conditions["w"] = -conditions.event.map(selected.set_index("event").pbtime).to_numpy().astype(np.float32).astype(float)
        raw = dict(events=events, stock_pulses=stock, conditions=conditions,
                   stock_adc_refit_pulses=p.drop(columns="starttdc"), stock_adc_refit_waveforms=w,
                   stock_adc_refit_comparison=comparison)
        for name, table in raw.items():
            self.persist(number, name, table, boundary=name == "events")
        self.raw[number] = raw
        self.prepared[number] = dict(
            selected=selected, pulses=qualified,
            waveforms=w.merge(selected[N.P.FRAME_KEYS], on="event", sort=False, validate="many_to_one"),
            bridge=qualified[[*N.P.FRAME_KEYS, *N.P.PULSE_KEYS, "wf", "refit_idx"]].assign(
                stock_idx=qualified.refit_idx), diagnostics={"synthetic_bridge_stub": True},
        )

    def prepare(self, p, w, conditions, events, stock, comparison, registry, calibration, geom, digest):
        number = next(n for n in (0, 1) if N.INPUTS[n] == digest)
        self.prepares.append(number)
        for actual, name in ((p, "stock_adc_refit_pulses"), (w, "stock_adc_refit_waveforms"),
                             (conditions, "conditions"), (events, "events"), (stock, "stock_pulses"),
                             (comparison, "stock_adc_refit_comparison")):
            pd.testing.assert_frame_equal(actual, self.raw[number][name], check_exact=True)
        pd.testing.assert_frame_equal(registry, self.registry, check_exact=True)
        pd.testing.assert_frame_equal(calibration, self.calibration, check_exact=True)
        assert geom is self.geometry
        return {name: value.copy() for name, value in self.prepared[number].items()}

    def finder(self, pulses, registry, selected, calibration, geom, *, registry_digest):
        assert self.prepares == [0, 1], "both complete blocks must be admitted before dispatch"
        assert registry_digest == N.F.digest(registry) == N.REGISTRY_SHA
        assert len(registry) == 9000 and len(calibration) == 15360 and geom is self.geometry
        self.finder_calls.append((pulses.copy(), selected.copy()))
        native = pulses[[*N.P.FRAME_KEYS, "refit_idx", *N.P.PULSE_FIELDS]].rename(columns={"refit_idx": "idx"})
        frames = selected.copy()
        frames["pulses"] = frames.frame_id.map(pulses.groupby("frame_id").size()).astype(np.int64)
        frames["clusters"], frames["members"] = 0, 0
        return dict(frames=frames, pulses=native, clusters=empty(N.V.CLUSTERS), members=empty(N.V.MEMBERS))

    def run(self, **changes):
        return N.run(**(self.kwargs | changes))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    for module, names in (
        (N.M.C, ("recover",)), (N.M.C.C, ("find_event",)), (N.V, ("verify",)),
        (N.SR, ("verify",)), (N.M, ("run",)), (N.M.S, ("run",)),
        (N.I.Q, ("run", "admission")), (N.I.Q.R, ("run_arm",)),
        (N.I.Q.R.Q, ("run_arm",)), (N.I.Q.R.W, ("synthesize",)),
        (N.I.Q.R.P, ("reconstruct",)), (N.I.Q.M.K, ("make_photons",)),
        (N.I.Q.M.P, ("read_input",)), (N.I.D, ("run", "run_arm")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    return SavedRefits(tmp_path, monkeypatch)


def assert_unpublished(saved):
    assert not saved.finder_calls
    assert not saved.out.exists()


def assert_failed(saved):
    assert (saved.out / N.A.FAILED).exists()
    assert not (saved.out / N.A.COMPLETE).exists()
    assert not (saved.out / "summary.json").exists()


def test_synthetic_runner_real_parquets_and_atomic_receipts(saved):
    result = saved.run()
    assert saved.prepares == [0, 1] and len(saved.finder_calls) == 2
    assert [len(p) for p, _ in saved.finder_calls] == [25, 62]
    assert [s.event.tolist() for _, s in saved.finder_calls] == [[2505], [281, 2755]]
    assert [s.input_sha256.unique().tolist() for _, s in saved.finder_calls] == [[N.INPUTS[0]], [N.INPUTS[1]]]
    assert result["registry_frames"] == 9000 and result["unique_frames"] == 3
    assert result["saved_pulses"] == 87 and result["saved_waveforms"] == 80
    assert result["finder_calls"] == 2 and result["full_registry_admitted"] is True
    for key in ("nonempty_real_refit_membership", "stock_native_membership_compared", "same_pulse_stock_validation",
                "regenerated_response", "regenerated_fit", "stock_rerun", "inherited_raw_admission_repeated",
                "prior_verifier_rerun", "coverage_computed", "execution_admitted", "full_retained_campaign_admitted",
                "timing_closure"):
        assert result[key] is False
    assert result["F"] is result["finite_library_uncertainty"] is None
    assert json.loads((saved.out / "summary.json").read_text()) == json.loads(json.dumps({
        k: v for k, v in result.items() if k != "persisted_path_bytes"}))
    receipt = json.loads((saved.out / N.A.COMPLETE).read_text())
    _, size = N.SR._batch(saved.out, receipt["context"], list(receipt["artifacts"]), N.CAP)
    assert 0 < size == result["persisted_path_bytes"] <= N.CAP
    assert not (saved.out / N.A.FAILED).exists()
    assert receipt["context"]["arms"] == ["saved_adc_refit"] and receipt["context"]["seeds"] == [0]
    for number in (0, 1):
        p = pd.read_parquet(saved.out / f"block{number}_pulses.parquet")
        wanted = saved.prepared[number]["pulses"].rename(columns={"refit_idx": "idx"})
        pd.testing.assert_frame_equal(p, wanted[p.columns], check_exact=True)
        assert p.pe.dtype == np.dtype("float64") and p.failed.sum() == p.zerondf.sum() == 1
        assert len(pd.read_parquet(saved.out / f"block{number}_saved_waveforms.parquet")) == (23, 57)[number]
    pd.testing.assert_frame_equal(pd.read_parquet(saved.out / "registry.parquet"), saved.registry, check_exact=True)
    for path, digest in (result["inputs"] | result["code_sha256"]).items():
        assert sha(path) == digest
    assert {v["sha256"] for k, v in receipt["artifacts"].items() if k.startswith("code")} == set(result["code_sha256"].values())


@pytest.mark.parametrize("pin", ["downstream", "boundary", "verified", "membership", "membership_verified"])
def test_each_external_pin_is_required_before_finder(saved, pin):
    with pytest.raises(ValueError, match="changed bound input"):
        saved.run(**{f"expected_{pin}_sha256": "0" * 64})
    assert saved.prepares == []
    assert_unpublished(saved)


@pytest.mark.parametrize("gate", GATES)
def test_each_inherited_verification_gate_is_required(saved, gate):
    saved.mv[gate] = False
    saved.kwargs["expected_membership_verified_sha256"] = write_json(saved.membership_verified, saved.mv)
    with pytest.raises(ValueError, match="missing inherited membership gate"):
        saved.run()
    assert saved.reads == []
    assert_unpublished(saved)


@pytest.mark.parametrize("field,value,match", [
    ("summary_sha256", "0" * 64, "belongs to another"),
    ("registry_sha256", "0" * 64, "unexpected inherited full registry"),
    ("registry_frames", 8999, "unexpected inherited full registry"),
    ("code_sha256", "0" * 64, "verifier source changed"),
])
def test_resealed_inherited_report_cannot_change_authority(saved, field, value, match):
    saved.mv[field] = value
    saved.kwargs["expected_membership_verified_sha256"] = write_json(saved.membership_verified, saved.mv)
    with pytest.raises(ValueError, match=match):
        saved.run()
    assert_unpublished(saved)


@pytest.mark.parametrize("field,value", [("input_summary_sha256", "0" * 64),
                                        ("boundary_summary_sha256", "0" * 64),
                                        ("original_inputs_verified", False)])
def test_resealed_downstream_report_must_preserve_provenance(saved, field, value):
    saved.dv[field] = value
    saved.kwargs["expected_verified_sha256"] = write_json(saved.verified, saved.dv)
    with pytest.raises(ValueError, match="provenance differs"):
        saved.run()
    assert saved.prepares == []
    assert_unpublished(saved)


@pytest.mark.parametrize("mutation,match", [("schedule", "block schedule"), ("target", "target schedule"),
                                           ("input", "input identity"), ("binding", "payload binding")])
def test_second_block_admission_failure_precedes_every_finder_call(saved, mutation, match):
    if mutation == "schedule":
        saved.d["blocks"].reverse()
    elif mutation == "target":
        saved.d["blocks"][1]["targets"] = [281]
    elif mutation == "input":
        saved.b["blocks"][1]["input_sha256"] = {"/never-open/mixed.root": "0" * 64}
    else:
        del saved.d["output_sha256"]["block1_stock_adc_refit_comparison.parquet"]
    saved.commit_reports()
    with pytest.raises(ValueError, match=match):
        saved.run()
    assert_unpublished(saved)


def test_second_bridge_failure_has_no_partial_dispatch_or_fallback(saved, monkeypatch):
    def prepare(*args):
        if args[-1] == N.INPUTS[1]:
            raise ValueError("second synthetic bridge rejected")
        return saved.prepare(*args)

    monkeypatch.setattr(N.B, "prepare", prepare)
    with pytest.raises(ValueError, match="second synthetic bridge rejected"):
        saved.run()
    assert saved.prepares == [0]
    assert_unpublished(saved)


@pytest.mark.parametrize("table", ["pulses", "waveforms", "selected"])
def test_wrong_aggregate_census_never_dispatches(saved, table):
    saved.prepared[1][table] = saved.prepared[1][table].iloc[:-1].copy()
    with pytest.raises(ValueError, match="aggregate census"):
        saved.run()
    assert saved.prepares == [0, 1]
    assert_unpublished(saved)


@pytest.mark.parametrize("mutation,match", [("duplicate", "calibration channel domain"),
                                           ("missing", "complete per-input calibration domain"),
                                           ("input", "complete per-input calibration domain"),
                                           ("status", "calibration status")])
def test_full_calibration_domain_including_unselected_input_is_authoritative(saved, mutation, match):
    c = saved.calibration.copy()
    if mutation == "duplicate":
        c.loc[c.index[-1], "chan"] = 0
    elif mutation == "missing":
        c = c.iloc[:-1]
    elif mutation == "input":
        c.loc[c.input_sha256.eq("f" * 64), "input_sha256"] = "e" * 64
    else:
        c.loc[c.index[-1], "status"] = -1
    c.to_parquet(saved.calibration_path, index=False)
    saved.summary["inputs"][str(saved.calibration_path)] = sha(saved.calibration_path)
    saved.publish_authority()
    with pytest.raises(ValueError, match=match):
        saved.run()
    assert_unpublished(saved)


@pytest.mark.parametrize("suffix", ["calibration.parquet", "crv_counters_run1a_v01.txt"])
def test_authority_inputs_must_have_unique_recorded_sources(saved, suffix):
    path = saved.aux / "duplicate" / suffix
    path.parent.mkdir()
    original = saved.calibration_path if suffix == "calibration.parquet" else saved.counters
    path.write_bytes(original.read_bytes())
    saved.summary["inputs"][str(path)] = sha(path)
    saved.publish_authority()
    with pytest.raises(ValueError, match="missing unique authority"):
        saved.run()
    assert_unpublished(saved)


@pytest.mark.parametrize("mutation", ["row_cap", "column_cap"])
def test_second_block_parquet_metadata_rejected_before_decode(saved, mutation):
    if mutation == "row_cap":
        table = pd.DataFrame({"event": np.zeros(2049, dtype=np.int64)})
    else:
        table = pd.DataFrame({f"c{i}": [0] for i in range(65)})
    path = saved.persist(1, "stock_adc_refit_pulses", table)
    saved.commit_reports()
    with pytest.raises(ValueError, match="row/schema cap exceeded before decode"):
        saved.run()
    assert path not in saved.reads and saved.prepares == [0]
    assert_unpublished(saved)


def test_oversized_geometry_fails_before_geometry_parse(saved, monkeypatch):
    saved.counters.write_bytes(b"x" * (2 * 1024**2 + 1))
    saved.summary["inputs"][str(saved.counters)] = sha(saved.counters)
    saved.publish_authority()
    monkeypatch.setattr(N.M, "geometry_only", forbidden)
    with pytest.raises(ValueError, match="geometry text cap"):
        saved.run()
    assert_unpublished(saved)


@pytest.mark.parametrize("mutation", ["payload", "extra", "receipt"])
def test_changed_inherited_transaction_rejected_before_dispatch(saved, mutation):
    if mutation == "payload":
        with (saved.membership / "selected.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "extra":
        (saved.membership / "undeclared.txt").write_text("not in receipt")
    else:
        value = json.loads((saved.membership / N.A.COMPLETE).read_text())
        value["context"]["seeds"] = [999]
        write_json(saved.membership / N.A.COMPLETE, value)
    with pytest.raises(ValueError):
        saved.run()
    assert_unpublished(saved)


@pytest.mark.parametrize("target", ["consumed_pulse", "calibration", "unused_receipted_table", "receipt", "extra_entry"])
def test_late_mutation_prevents_atomic_completion(saved, monkeypatch, target):
    original = N.A.BatchArtifacts.write_parquet
    changed = []

    def write_then_mutate(tx, name, table, **options):
        original(tx, name, table, **options)
        if tx.directory != saved.out or name != "block1_saved_pulses.parquet" or changed:
            return
        paths = {"consumed_pulse": saved.downstream / "block1_stock_adc_refit_pulses.parquet",
                 "calibration": saved.calibration_path, "unused_receipted_table": saved.membership / "grid.parquet",
                 "receipt": saved.membership / N.A.COMPLETE, "extra_entry": saved.membership / "undeclared.txt"}
        path = paths[target]
        with path.open("ab") as stream:
            stream.write(b"late mutation after admission")
        changed.append(path)

    monkeypatch.setattr(N.A.BatchArtifacts, "write_parquet", write_then_mutate)
    with pytest.raises(ValueError, match="changed bound input|undeclared batch artifact|artifact hash or byte count changed"):
        saved.run()
    assert len(changed) == 1 and len(saved.finder_calls) == 2
    assert_failed(saved)


@pytest.mark.parametrize("location", ["downstream", "boundary", "membership", "reports", "calibration", "symlink"])
def test_output_exclusion_includes_discovered_authority_inputs(saved, tmp_path, location):
    roots = dict(downstream=saved.downstream, boundary=saved.boundary, membership=saved.membership,
                 reports=saved.root, calibration=saved.aux)
    if location == "symlink":
        link = tmp_path / "evidence_alias"
        link.symlink_to(saved.aux, target_is_directory=True)
        out = link / "forbidden_output"
    else:
        out = roots[location] / "forbidden_output"
    with pytest.raises(ValueError, match="outside prior evidence"):
        saved.run(out=out)
    assert not out.exists()
    assert_unpublished(saved)


def test_completed_output_is_exclusive_and_unchanged(saved):
    saved.run()
    before = {p.name: p.read_bytes() for p in saved.out.iterdir()}
    reads, calls = list(saved.reads), len(saved.finder_calls)
    with pytest.raises(ValueError, match="choose a new output directory"):
        saved.run()
    assert before == {p.name: p.read_bytes() for p in saved.out.iterdir()}
    assert saved.reads == reads and len(saved.finder_calls) == calls


@pytest.mark.parametrize("mutation", ["count", "index", "order"])
def test_finder_cannot_change_admitted_collection_bridge(saved, monkeypatch, mutation):
    def finder(*args, **kwargs):
        products = saved.finder(*args, **kwargs)
        p = products["pulses"].copy()
        if mutation == "count":
            p = p.iloc[:-1]
        elif mutation == "index":
            p.loc[p.index[0], "idx"] += 1
        else:
            p = p.iloc[::-1].reset_index(drop=True)
        products["pulses"] = p
        return products

    monkeypatch.setattr(N.P, "run", finder)
    with pytest.raises(ValueError, match="finder pulse census changed|finder changed admitted native index bridge"):
        saved.run()
    assert len(saved.finder_calls) == 1
    assert_failed(saved)


def test_finder_exception_has_no_fallback_and_failed_output_cannot_resume(saved, monkeypatch):
    calls = []

    def finder(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("synthetic finder failure")

    monkeypatch.setattr(N.P, "run", finder)
    with pytest.raises(RuntimeError, match="synthetic finder failure"):
        saved.run()
    assert calls == [True]
    assert_failed(saved)
    before = {p.name: p.read_bytes() for p in saved.out.iterdir()}
    with pytest.raises(ValueError, match="choose a new output directory"):
        saved.run()
    assert calls == [True] and before == {p.name: p.read_bytes() for p in saved.out.iterdir()}


def test_synthetic_runner_accepts_real_pure_bridge_without_running_finder(saved, monkeypatch):
    def prepare(*args):
        number = next(n for n in (0, 1) if args[-1] == N.INPUTS[n])
        result = saved.original_prepare(*args)
        saved.prepares.append(number)
        return result

    monkeypatch.setattr(N.B, "prepare", prepare)
    result = saved.run()
    assert saved.prepares == [0, 1] and len(saved.finder_calls) == 2
    for record in result["records"]:
        diagnostics = record["diagnostics"]
        assert diagnostics["complete_stock_refit_bijection"]
        assert diagnostics["authoritative_saved_conditions_exact"]
        assert diagnostics["original_time_bits_exact"]
        assert diagnostics["all_active"]


@pytest.mark.parametrize("mutation", ["metadata", "consumed_payload"])
def test_mutation_during_last_bridge_is_rechecked_before_finder_and_output(saved, monkeypatch, mutation):
    def prepare(*args):
        result = saved.prepare(*args)
        if args[-1] == N.INPUTS[1]:
            path = (saved.verified if mutation == "metadata"
                    else saved.downstream / "block0_stock_adc_refit_pulses.parquet")
            with path.open("ab") as stream:
                stream.write(b"changed after admission before finder")
        return result

    monkeypatch.setattr(N.B, "prepare", prepare)
    with pytest.raises(ValueError, match="changed bound input"):
        saved.run()
    assert saved.prepares == [0, 1]
    assert_unpublished(saved)


@pytest.mark.parametrize("mutation", ["incomplete", "exposure"])
def test_resealed_registry_cannot_replace_the_complete_frozen_authority(saved, mutation):
    saved.registry = saved.registry.copy()
    if mutation == "incomplete":
        saved.registry = saved.registry.iloc[:-1].copy()
    else:
        saved.registry.loc[saved.registry.index[-1], "intensity"] = 2.
    saved.publish_authority()
    with pytest.raises(ValueError, match="complete registry changed"):
        saved.run()
    assert_unpublished(saved)


def test_second_block_file_size_cap_precedes_parquet_open(saved):
    path = saved.downstream / "block1_stock_adc_refit_pulses.parquet"
    with path.open("r+b") as stream:
        stream.truncate(N.I.FILE_CAP + 1)
    saved.d["output_sha256"][path.name] = sha(path)
    saved.commit_reports()
    with pytest.raises(ValueError, match="refit input file cap exceeded"):
        saved.run()
    assert path not in saved.reads and saved.prepares == [0]
    assert_unpublished(saved)


def test_output_creation_race_never_overwrites_competing_writer(saved, monkeypatch):
    original = N.Evidence.recheck
    calls = []

    def recheck_then_compete(evidence):
        original(evidence)
        if not calls:
            saved.out.mkdir()
            (saved.out / "owner.txt").write_text("competing writer owns this directory")
            calls.append(True)

    monkeypatch.setattr(N.Evidence, "recheck", recheck_then_compete)
    with pytest.raises((FileExistsError, ValueError)):
        saved.run()
    assert not saved.finder_calls
    assert {p.name: p.read_text() for p in saved.out.iterdir()} == {
        "owner.txt": "competing writer owns this directory"}


@pytest.mark.parametrize("field", ["pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb"])
def test_resealed_conditions_on_unoccupied_channel_must_match_full_calibration(saved, monkeypatch, field):
    conditions = saved.raw[1]["conditions"].copy()
    conditions.loc[conditions.index[-1], field] += 1
    saved.persist(1, "conditions", conditions)
    saved.commit_reports()

    def prepare(*args):
        number = next(n for n in (0, 1) if args[-1] == N.INPUTS[n])
        result = saved.original_prepare(*args)
        saved.prepares.append(number)
        return result

    monkeypatch.setattr(N.B, "prepare", prepare)
    with pytest.raises(ValueError, match="conditions differ from authoritative calibration"):
        saved.run()
    assert saved.prepares == [0]
    assert_unpublished(saved)


def test_late_bound_source_change_fails_without_modifying_frozen_sources(saved, monkeypatch):
    source = saved.aux / "synthetic_source.py"
    source.write_text("fixture_marker = 1\n")
    code_paths = N.M.code_paths()
    monkeypatch.setattr(N.M, "code_paths", lambda: [*code_paths, source])
    original = N.A.BatchArtifacts.write_parquet

    def write_then_mutate(tx, name, table, **options):
        original(tx, name, table, **options)
        if tx.directory == saved.out and name == "block1_saved_pulses.parquet":
            source.write_text("fixture_marker = 2\n")

    monkeypatch.setattr(N.A.BatchArtifacts, "write_parquet", write_then_mutate)
    with pytest.raises(ValueError, match="changed bound input"):
        saved.run()
    assert len(saved.finder_calls) == 2
    assert_failed(saved)


@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_adapter_normalized_integer_width_is_not_identity_change(saved, monkeypatch, dtype):
    keys = [*N.P.FRAME_KEYS, *N.P.PULSE_KEYS]
    for block in saved.prepared.values():
        for name in keys[1:]:
            block["pulses"][name] = block["pulses"][name].astype(dtype)
    original = saved.finder

    def normalized(*args, **kwargs):
        result = original(*args, **kwargs)
        for name in keys[1:] + ["idx"]:
            result["pulses"][name] = result["pulses"][name].astype(np.int64)
        return result

    monkeypatch.setattr(N.P, "run", normalized)
    result = saved.run()
    assert result["finder_calls"] == 2


def test_snapshot_bytes_must_match_declared_hash_before_dispatch(saved, monkeypatch):
    source = saved.aux / "snapshot_source.py"
    source.write_text("fixture_marker = 1\n")
    paths = N.M.code_paths()
    monkeypatch.setattr(N.M, "code_paths", lambda: [*paths, source])
    original = Path.read_bytes

    def changed_bytes(path):
        if path == source:
            return b"fixture_marker = 2\n"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    with pytest.raises(ValueError, match="source snapshot hash differs"):
        saved.run()
    assert not saved.finder_calls
    assert_failed(saved)


def test_post_snapshot_evidence_change_precedes_dispatch(saved, monkeypatch):
    source = saved.aux / "snapshot_mutator.py"
    source.write_text("fixture_marker = 1\n")
    paths = N.M.code_paths()
    monkeypatch.setattr(N.M, "code_paths", lambda: [*paths, source])
    original = Path.read_bytes

    def mutate_after_read(path):
        payload = original(path)
        if path == source:
            saved.calibration_path.write_bytes(b"late mutation after valid source snapshot")
        return payload

    monkeypatch.setattr(Path, "read_bytes", mutate_after_read)
    with pytest.raises(ValueError, match="changed bound input"):
        saved.run()
    assert not saved.finder_calls
    assert_failed(saved)


def test_actual_legacy_comparison_schema_through_real_parquets_and_bridge(saved, monkeypatch):
    for number in (0, 1):
        comp = saved.raw[number]["stock_adc_refit_comparison"].drop(columns=["bar", "sipm", "chan"])
        saved.persist(number, "stock_adc_refit_comparison", comp)
    saved.commit_reports()

    def prepare(*args):
        number = next(n for n in (0, 1) if args[-1] == N.INPUTS[n])
        result = saved.original_prepare(*args)
        saved.prepares.append(number)
        return result

    monkeypatch.setattr(N.B, "prepare", prepare)
    result = saved.run()
    assert len(saved.finder_calls) == 2
    assert all(r["diagnostics"]["comparison_channel_from_stock_index"] for r in result["records"])
