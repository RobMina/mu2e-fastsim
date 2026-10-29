"""Synthetic runner integration, NOT admission of either real saved control.

Only geometry construction and the external stock ignore-policy source are local
stand-ins. Pins, Parquet decoding, receipts, inventories, and end-of-run rehashes
are real. Empty unconsumed stages deliberately do not certify response physics.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto import saved_pulse_inventory as I

A, F, Q, M = I.A, I.F, I.Q, I.M
sha256 = M.S.C.sha256


def write_json(path, value):
    path.write_bytes(A._canonical(value))
    return sha256(path)


def pulse_tables(selected):
    # Three active layers, one ignored diagnostic, another side, and a quiet frame.
    p = selected.iloc[[0, 0, 0, 0, 1, 2]][I.KEYS].reset_index(drop=True)
    for name in sorted(Q.R.REQUIRED["pulses"]):
        p[name] = np.full(len(p), 1., dtype=np.float64)
    p["chan"] = np.array([0, 4, 8, 12, 1, 4], dtype=np.int64)
    p["wf"] = np.arange(len(p), dtype=np.int64)
    p["seq"] = np.zeros(len(p), dtype=np.int64)
    p["starttdc"] = np.full(len(p), 40, dtype=np.int64)
    p["time"] = np.full(len(p), 500., dtype=np.float64)
    p["time_relative"] = p.time + 2.
    pb = p.frame_id.map(selected.set_index("frame_id").pbtime).to_numpy().astype(np.float32).astype(float)
    p["time_reco"] = p.time_relative.to_numpy() - pb
    p["pe"] = np.full(len(p), 12., dtype=np.float64)
    p["channel_ignored"] = p.chan.eq(12)
    p["failed"] = np.array([False, False, False, True, False, False])
    p["zerondf"] = np.zeros(len(p), dtype=bool)
    p["pedfromdb"] = np.ones(len(p), dtype=bool)
    w = p[[*I.KEYS, "wf", "chan", "starttdc"]].copy()
    w["n"] = np.full(len(w), 16, dtype=np.int64)
    w["pedestal_used"] = np.full(len(w), 100.)
    w["max_adc_minus_pedestal"] = np.full(len(w), 30.)
    w["peak_margin_adc"] = np.full(len(w), 25.)
    w["near_threshold"] = np.zeros(len(w), dtype=bool)
    assert set(Q.R.REQUIRED["pulses"]) <= set(p)
    assert set(Q.R.REQUIRED["waveforms"]) <= set(w)
    return p, w


class SyntheticSavedControls:
    """One legacy block, two thermal states, four arms; eight all-live children."""

    def __init__(self, directory, monkeypatch):
        self.evidence = directory / "synthetic_evidence"
        self.evidence.mkdir()
        self.response = self.evidence / "response"
        self.response.mkdir()
        self.extreme = self.evidence / "extreme"
        self.extreme.mkdir()
        optical = self.evidence / "optical"
        optical.mkdir()
        self.out = directory / "inventory"
        self.response_verified = self.evidence / "response_verified.json"
        self.extreme_verified = self.evidence / "extreme_verified.json"
        self.registry = F.freeze(pd.DataFrame({
            "input_sha256": "a" * 64, "run": 17, "subrun": 3,
            "event": np.arange(1, 9001, dtype=np.int64), "pbtime": -224.46,
            "intensity": np.linspace(1., 2., 9000),
        }))
        self.selected = self.registry.iloc[[1, 17, 307, 8998]].reset_index(drop=True)
        self.calibration = pd.DataFrame({
            "input_sha256": "a" * 64, "chan": np.arange(5120, dtype=np.int64),
            "pedestal": 100., "pulsearea": 1000., "pulseheight": 20., "timeoffset": 2.,
            "status": np.zeros(5120, dtype=np.int64), "roc": 1, "feb": 1,
        })
        self.calibration.loc[self.calibration.chan.eq(12), "status"] = 2
        counters = self.evidence / "crv_counters_run1a_v01.txt"
        counters.write_text("synthetic 1280-bar geometry; not Run1A admission\n")
        geom = SimpleNamespace(sector=np.zeros(1280, dtype=np.int64),
                               layer=np.arange(1280, dtype=np.int64) % 4,
                               sector_type=np.array([0]), sector_names=["T1"])
        monkeypatch.setattr(M, "geometry_only", lambda d: geom)
        policy = self.evidence / "synthetic_ignore_policy.txt"
        policy.write_text("synthetic fixture only: ignoreChannels: true\n")
        policy_sha = sha256(policy)

        def local_policy():
            M.bind(policy, policy_sha)
            return {str(policy): policy_sha}

        monkeypatch.setattr(M, "ignore_bindings", local_policy)
        self.pulses, self.waveforms = pulse_tables(self.selected)
        frames = self.selected.copy()
        for stage in Q.R.STAGES[1:]:
            frames[stage] = np.zeros(len(frames), dtype=np.int64)
        for name, table in (("pulses", self.pulses), ("waveforms", self.waveforms)):
            frames[name] = frames.frame_id.map(table.groupby("frame_id").size()).fillna(0).astype(np.int64)
        full = {"frames": frames, "pulses": self.pulses, "waveforms": self.waveforms}
        empty = {
            name: pd.DataFrame({field: pd.Series(dtype="float64")
                                for field in sorted(set(I.KEYS) | fields)})
            for name, fields in Q.R.REQUIRED.items()
        }
        self.context = dict(registry_sha256=F.digest(self.registry), batch_identity={
            "plan_sha256": "1" * 64, "smoke_sha256": "2" * 64, "contract": Q.R.CONTRACT,
        }, arms=list(Q.M.P.ARMS), seeds=list(Q.SEEDS))
        self.tags = [f"seed{seed}_{arm}" for seed in Q.SEEDS for arm in Q.M.P.ARMS]
        self.decoded_paths = {self.response / "run" / f"{name}.parquet"
                              for name in ("registry", "selected", "calibration")}
        records, receipts = [], {}
        child_names = [f"{variant}_{stage}.parquet" for variant in Q.VARIANTS for stage in Q.R.STAGES] + ["validation.json"]
        for seed in Q.SEEDS:
            for arm in Q.M.P.ARMS:
                tag = f"seed{seed}_{arm}"
                root = self.response / tag
                context = {**self.context, "batch_identity": {**self.context["batch_identity"],
                           "frame_ids": self.selected.frame_id.tolist(), "variants": list(Q.VARIANTS)},
                           "arms": [arm], "seeds": [seed]}
                with A.BatchArtifacts(root, max_bytes=Q.BATCH_CAP, context=context, expected_artifacts=child_names) as tx:
                    for variant in Q.VARIANTS:
                        for stage in Q.R.STAGES:
                            table = full.get(stage, empty[stage]) if variant == "full" else empty[stage]
                            tx.write_parquet(f"{variant}_{stage}.parquet", table)
                    tx.write_json("validation.json", {"synthetic_fixture_not_real_admission": True})
                receipts[tag] = json.loads((root / A.COMPLETE).read_text())
                records.append(dict(seed=seed, arm=arm, path=tag, receipt_sha256=sha256(root / A.COMPLETE),
                                    counts={"pulses": len(self.pulses)}))
                self.decoded_paths.update(root / f"full_{stage}.parquet" for stage in full)
        # Snapshot a real stable helper; do not claim to have admitted the original generation code.
        source = Path(Q.__file__)
        saved = dict(schema=Q.SCHEMA, frames=4, registry_frames=9000, registry_sha256=F.digest(self.registry),
                     arms=list(Q.M.P.ARMS), seeds=list(Q.SEEDS), F=None, timing_closure=False,
                     code_sha256={str(source): sha256(source)}, inputs={str(counters): sha256(counters)},
                     records=records, synthetic_fixture_not_real_admission=True)
        names = ["summary.json", "registry.parquet", "selected.parquet", "calibration.parquet", "admission.json",
                 *[tag + ".json" for tag in self.tags], "code0.txt"]
        with A.BatchArtifacts(self.response / "run", max_bytes=Q.ROOT_CAP, context=self.context, expected_artifacts=names) as tx:
            tx.write_json("summary.json", saved)
            for name in ("registry", "selected", "calibration"):
                tx.write_parquet(name + ".parquet", getattr(self, name))
            tx.write_json("admission.json", {"synthetic_fixture_not_real_admission": True})
            for tag, receipt in receipts.items():
                tx.write_json(tag + ".json", receipt)
            with tx.open_artifact("code0.txt") as stream:
                stream.write(source.read_bytes())
        response_sha = sha256(self.response / "run/summary.json")
        rv_sha = write_json(self.response_verified, dict(
            schema=I.SR.SCHEMA, summary_sha256=response_sha, registry_sha256=F.digest(self.registry),
            authoritative_all_live_conditions_exact=True, all_tested_physical_partition_fields_exact=True,
            synthetic_fixture_not_real_admission=True))
        events = self.selected.drop(columns=["input_sha256", "frame_id"])
        events.to_parquet(optical / "events.parquet", index=False)
        self.decoded_paths.add(optical / "events.parquet")
        optical_sha = write_json(optical / "summary.json", {"blocks": [{"input_sha256": "a" * 64,
            "artifacts": {"events": {"path": "events.parquet", "sha256": sha256(optical / "events.parquet")}}}]})
        legacy_p = self.pulses.drop(columns=["input_sha256", "run", "subrun", "frame_id"])
        legacy_w = self.waveforms.drop(columns=["input_sha256", "run", "subrun", "frame_id"])
        cond = legacy_p[["event", "chan"]].merge(self.calibration.drop(columns="input_sha256"),
                                                 on="chan", validate="many_to_one")
        cond = cond.merge(events[["event", "pbtime"]], on="event", validate="many_to_one")
        cond["w"] = -cond.pbtime.to_numpy().astype(np.float32).astype(float)
        outputs = {}

        def persist(name, table):
            path = self.extreme / name
            table.to_parquet(path, index=False)
            outputs[name] = sha256(path)
            self.decoded_paths.add(path)

        persist("input0_conditions.parquet", cond)
        models = {}
        for thermal in (0, 1):
            prefix = f"input0_thermal{thermal}"
            models[prefix] = {"arms": {arm: {"pulses": len(legacy_p)} for arm in Q.M.P.ARMS}}
            for arm in Q.M.P.ARMS:
                persist(f"{prefix}_{arm}_pulses.parquet", legacy_p)
                persist(f"{prefix}_{arm}_waveforms.parquet", legacy_w)
        outputs["optical_verified.json"] = write_json(self.extreme / "optical_verified.json",
                                                       {"synthetic_fixture_not_real_admission": True})
        self.old = dict(schema="crv-saved-photon-downstream-v1", F=None, timing_closure=False,
                        blocks=[{"input_sha256": "a" * 64, "seed": 37,
                                 "events": events[list(M.C.IDS)].to_dict("records"), "models": models}],
                        output_sha256=outputs, code_sha256={str(source): sha256(source)},
                        input_summary=str(optical / "summary.json"), input_summary_sha256=optical_sha,
                        synthetic_fixture_not_real_admission=True)
        self.pins = dict(expected_response_sha256=response_sha, expected_response_verified_sha256=rv_sha)
        self.commit_extreme()
        original_read = pd.read_parquet
        self.reads = []

        def bounded_decode(path, *args, **kwargs):
            path = Path(path)
            assert path in self.decoded_paths, f"unexpected payload decode: {path}"
            self.reads.append(path)
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", bounded_decode)

    def commit_extreme(self):
        """Repin intentionally modified fixture metadata, not a bypass of binding checks."""
        digest = write_json(self.extreme / "summary.json", self.old)
        self.pins["expected_extreme_sha256"] = digest
        self.pins["expected_extreme_verified_sha256"] = write_json(self.extreme_verified, dict(
            schema="crv-saved-downstream-report-v1", input_summary_sha256=digest, regenerated_response=False,
            blocks=[{"input_sha256": "a" * 64}], synthetic_fixture_not_real_admission=True))

    def run(self, **pins):
        return I.run(self.response, self.response_verified, self.extreme, self.extreme_verified,
                     self.out, **(self.pins | pins))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved-only inventory attempted finder/response/raw access or verifier rerun")

    for module, names in (
        (I.P, ("run",)), (M.C, ("recover",)), (M.C.C, ("find_event",)),
        (Q, ("run", "admission")), (Q.R, ("run_arm",)), (Q.R.Q, ("run_arm",)),
        (I.D, ("run", "run_arm")), (Q.R.W, ("synthesize",)), (Q.R.P, ("reconstruct",)),
        (Q.M.K, ("make_photons",)), (Q.M.P, ("read_input",)),
        (I.SR, ("verify",)), (I.V, ("verify",)), (uproot, ("open",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    return SyntheticSavedControls(tmp_path, monkeypatch)


def test_persists_synthetic_inventory_with_real_transactions_and_no_regeneration(saved):
    result = saved.run()
    persisted = json.loads((saved.out / "summary.json").read_text())
    assert result == persisted | {"persisted_path_bytes": result["persisted_path_bytes"]}
    receipt = json.loads((saved.out / A.COMPLETE).read_text())
    _, size = I.SR._batch(saved.out, receipt["context"], list(receipt["artifacts"]), I.CAP)
    assert size == result["persisted_path_bytes"] <= I.CAP
    assert not (saved.out / A.FAILED).exists()
    assert len(result["records"]) == 16
    assert {r["sample"] for r in result["records"]} == {"all_live", "extreme_optical_union"}
    for record in result["records"]:
        assert record["pulse_rows"] == record["saved_waveforms"] == 6
        assert record["candidate_tuple_upper_bound"] == 1
        assert record["saved_identity_and_order_valid"] and record["forward_time_bits_exact"]
        assert record["finder_not_run"] and not record["nonempty_membership_established"]
        assert [r["saved_pulses"] for r in record["frames"]] == [4, 1, 1, 0]
        assert [r["ignored_pulses"] for r in record["frames"]] == [1, 0, 0, 0]
        assert [r["finder_input_pulses"] for r in record["frames"]] == [3, 1, 1, 0]
        assert record["occupancy"][0]["distinct_layers"] == 3
    for key in ("regenerated_finder", "regenerated_response", "stock_rerun", "inherited_raw_admission_repeated",
                "nonempty_membership_established", "full_retained_campaign_admitted", "execution_admitted", "timing_closure"):
        assert result[key] is False
    assert result["F"] is result["finite_library_uncertainty"] is None
    assert set(saved.reads) == saved.decoded_paths
    for path, digest in (result["inputs"] | result["code_sha256"]).items():
        assert sha256(path) == digest
    snapshots = {v["sha256"] for k, v in receipt["artifacts"].items() if k.startswith("code")}
    assert snapshots == set(result["code_sha256"].values())


@pytest.mark.parametrize("pin", ["response", "response_verified", "extreme", "extreme_verified"])
def test_wrong_external_pin_fails_before_any_table_decode(saved, pin):
    with pytest.raises(ValueError, match="changed bound input"):
        saved.run(**{f"expected_{pin}_sha256": "0" * 64})
    assert saved.reads == []
    assert not saved.out.exists()


@pytest.mark.parametrize("mutation", ["bound_payload", "undeclared_child_entry"])
def test_late_mutation_leaves_failed_transaction_without_summary(saved, monkeypatch, mutation):
    original = M.check_bindings
    calls = []

    def mutate_then_recheck(bindings):
        calls.append(True)
        assert (saved.out / A.INCOMPLETE).exists()
        if mutation == "bound_payload":
            path = saved.extreme / "input0_thermal1_stock_timing_pulses.parquet"
            with path.open("ab") as stream:
                stream.write(b"late mutation after all payload decodes")
        else:
            (saved.response / saved.tags[0] / "undeclared.txt").write_text("late inventory mutation")
        return original(bindings)

    monkeypatch.setattr(M, "check_bindings", mutate_then_recheck)
    match = "changed bound input" if mutation == "bound_payload" else "undeclared batch artifact"
    with pytest.raises(ValueError, match=match):
        saved.run()
    assert calls == [True]
    assert set(saved.reads) == saved.decoded_paths
    assert (saved.out / A.FAILED).exists()
    assert not (saved.out / A.COMPLETE).exists()
    assert not (saved.out / "summary.json").exists()


def test_missing_required_extreme_binding_fails_even_with_fresh_metadata_pins(saved):
    name = "input0_thermal1_stock_timing_waveforms.parquet"
    del saved.old["output_sha256"][name]
    (saved.extreme / name).unlink()  # Inventory alone would now agree with the reduced manifest.
    saved.commit_extreme()
    with pytest.raises(ValueError, match="missing required extreme artifact binding"):
        saved.run()
    assert not saved.out.exists()


def test_completed_output_is_never_overwritten(saved):
    saved.run()
    before = {path.name: path.read_bytes() for path in saved.out.iterdir()}
    reads = list(saved.reads)
    with pytest.raises(ValueError, match="choose a new output directory"):
        saved.run()
    assert {path.name: path.read_bytes() for path in saved.out.iterdir()} == before
    assert saved.reads == reads
