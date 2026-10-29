"""Real saved-response admission with local synthetic artifacts and no response execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto import saved_qualified_response as S

F, M, R = S.F, S.M, S.R


class SavedInputs:
    """Keep artifact hashes real while replacing only upstream saved validation."""

    def __init__(self, directory, monkeypatch):
        self.directory = directory
        self.plan, self.smoke = directory / "plan", directory / "smoke"
        self.plan.mkdir()
        self.smoke.mkdir()
        self.stock = directory / "stock.root"
        self.raw = directory / "synthetic-source.bin"
        self.raw.write_bytes(b"synthetic source identity; never a real detector input")
        self.source_digest = S.sha256(self.raw)
        self.registry = F.freeze(pd.DataFrame({
            "input_sha256": self.source_digest, "run": 17, "subrun": 3,
            "event": np.arange(1, 9001, dtype=np.int64), "pbtime": -224.46,
            "intensity": np.linspace(1., 2., 9000),
        }))
        self.selected = self.registry.iloc[[1, 17, 307, 8998]].reset_index(drop=True)
        self.events = self.selected.drop(columns=["input_sha256", "frame_id"]).iloc[::-1].reset_index(drop=True)
        self.calibration = pd.DataFrame({
            "channel": np.arange(5120, dtype=np.int32), "pedestal": 100.,
            "pulsearea": 1000., "pulseheight": 20., "timeoffset": 2.,
            "status": np.zeros(5120, dtype=np.int32), "roc": 1, "feb": 1,
        })
        self.write_calibration(self.calibration)
        self.photons = {}
        for seed in S.SEEDS:
            q = self.selected.iloc[[0, 0, 1, 2, 2]][list(F.FRAME_KEYS)].reset_index(drop=True)
            q = q.assign(draw_seed=np.uint64(seed), source_row=np.arange(5, dtype=np.int64),
                         chan=np.array([0, 0, 1, 2, 2]), substep=0, photon_type=0,
                         ordinal=np.arange(5, dtype=np.int64), live_channel=True,
                         time_historical_relative=500., time_raw_wrapped_relative=502.)
            for arm in M.P.ARMS:
                q[f"{arm}_kept"] = True
            q["photon_id"] = M.roots(q)
            self.photons[seed] = q
        self.photon_map, self.template = directory / "map.root", directory / "template.txt"
        with uproot.recreate(self.photon_map) as root:
            root["photonMap"] = np.ones((40, 40)), np.arange(41), np.arange(41)
        self.template.write_text("local preflight-only template fixture\n")
        monkeypatch.setattr(R.D, "MAP_SHA", S.sha256(self.photon_map))
        monkeypatch.setattr(R.D, "TEMPLATE_SHA", S.sha256(self.template))
        self.s = {
            "schema": M.SCHEMA, "seeds": list(S.SEEDS),
            "inputs": {str(self.raw): self.source_digest},
            "assets": {str(self.photon_map): S.sha256(self.photon_map), str(self.template): S.sha256(self.template)},
            "code_sha256": {str(p): S.sha256(p) for p in M.code_paths()},
            "blocks": [{
                "input_sha256": self.source_digest,
                "input_bindings": {str(self.stock): S.sha256(self.stock), str(self.raw): self.source_digest},
                "artifacts": {"events": self.table(self.smoke, "events", self.events)},
                "replicas": [{"seed": seed, "artifacts": {
                    "photons": self.table(self.smoke, f"photons_{seed}", q),
                }} for seed, q in self.photons.items()],
            }],
        }
        self.p = {
            "schema": S.PR.P.SCHEMA, "smoke": str(self.smoke), "inputs": {},
            "registry": {"frames": 9000, "registry_sha256": F.digest(self.registry)},
            "artifacts": {"registry": self.table(self.plan, "registry", self.registry)},
            "code_sha256": {str(p): S.sha256(p) for p in S.code_paths()},
        }
        self.verifier_calls = []

        def saved_verifier(kind, directory, out):
            assert Path(directory) == getattr(self, kind)
            assert not Path(out).exists()
            self.verifier_calls.append(kind)
            receipt = {"synthetic_saved_validation": kind, "summary_sha256": S.sha256(Path(directory) / "summary.json")}
            Path(out).write_text(json.dumps(receipt))
            return receipt

        monkeypatch.setattr(S.PR, "verify", lambda directory, out: saved_verifier("plan", directory, out))
        monkeypatch.setattr(M, "verify", lambda directory, out: saved_verifier("smoke", directory, out))
        self.commit()

    @staticmethod
    def table(directory, name, frame):
        path = directory / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return {"path": path.name, "sha256": S.sha256(path), "rows": len(frame)}

    def write_calibration(self, frame):
        with uproot.recreate(self.stock) as root:
            root.mktree("crvdump/calib", {name: frame[name].to_numpy() for name in frame})

    def commit(self):
        (self.smoke / "summary.json").write_bytes(S.json_bytes(self.s))
        self.smoke_hash = S.sha256(self.smoke / "summary.json")
        self.p["inputs"][str(self.smoke / "summary.json")] = self.smoke_hash
        (self.plan / "summary.json").write_bytes(S.json_bytes(self.p))
        self.plan_hash = S.sha256(self.plan / "summary.json")

    def save_photons(self, seed, frame):
        replica = next(r for r in self.s["blocks"][0]["replicas"] if r["seed"] == seed)
        replica["artifacts"]["photons"] = self.table(self.smoke, f"photons_{seed}", frame)
        self.commit()

    def save_events(self, frame):
        self.s["blocks"][0]["artifacts"]["events"] = self.table(self.smoke, "events", frame)
        self.commit()

    def admit(self, **overrides):
        args = {"expected_plan_sha256": self.plan_hash, "expected_smoke_sha256": self.smoke_hash, **overrides}
        return S.admission(self.plan, self.smoke, **args)

    def preflight(self):
        return S.preflight(self.plan, self.smoke, self.directory / "preflight.json",
                           expected_plan_sha256=self.plan_hash, expected_smoke_sha256=self.smoke_hash,
                           photon_map=self.photon_map, template=self.template)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("admission must not regenerate optics, read raw detector data, or execute response")

    entrypoints = [
        (S, ("run",)),
        (M, ("run", "generate", "prepare")),
        (M.K, ("make_photons",)),
        (M.photons, ("make_photons", "load_tables")),
        (M.P, ("run", "common_photons", "block_probe", "read_input")),
        (S.PR.P, ("run", "workload")),
        (R, ("run_arm",)), (R.Q, ("run_arm",)), (R.D, ("run", "run_arm")),
        (R.S, ("optical_candidates", "thermal_candidates", "simulate_candidates")),
        (R.S.S, ("optical_candidates", "thermal_candidates", "simulate_candidates")),
        (R.W, ("synthesize",)), (R.W.W, ("synthesize", "reconstruct")),
        (R.P, ("reconstruct", "reconstruct_matrix", "fit_buckets")),
        (R.P.P, ("reconstruct", "reconstruct_full", "fit_gumbel")),
        (R.P.LM, ("fit_gumbel",)), (S.SI, ("simulate", "photon_map")),
        (S.WF, ("synthesize", "single_pe_template")),
    ]
    for module, names in entrypoints:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    fixture = SavedInputs(tmp_path, monkeypatch)
    original_open = uproot.open

    def local_stock_only(path, *args, **kwargs):
        assert Path(path) == fixture.stock, "unexpected ROOT input read"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(uproot, "open", local_stock_only)
    return fixture


def test_real_admission_preserves_hashes_full_calibration_and_quiet_frame(saved):
    registry, selected, calibration, photons, bindings, verified = saved.admit()
    pd.testing.assert_frame_equal(registry, saved.registry, check_exact=True)
    pd.testing.assert_frame_equal(selected, saved.selected, check_exact=True)
    expected = saved.calibration.rename(columns={"channel": "chan"}).assign(input_sha256=saved.source_digest)
    pd.testing.assert_frame_equal(calibration, expected, check_exact=True)
    assert len(registry) == 9000 and len(selected) == 4 and len(calibration) == 5120
    for seed, q in photons.items():
        pd.testing.assert_frame_equal(q, saved.photons[seed], check_exact=True)
        assert selected.iloc[-1].event not in set(q.event)
    union = pd.concat(list(photons.values()), ignore_index=True)
    np.testing.assert_array_equal(union.photon_id, M.roots(union))
    assert union.photon_id.is_unique and len(union) == 10
    assert saved.verifier_calls == ["plan", "smoke"]
    for kind in ("plan", "smoke"):
        assert verified[kind]["summary_sha256"] == getattr(saved, f"{kind}_hash")
    expected_bindings = {
        str(saved.plan / "summary.json"): saved.plan_hash,
        str(saved.smoke / "summary.json"): saved.smoke_hash,
        **saved.p["code_sha256"], **saved.s["code_sha256"],
        **saved.s["assets"], **saved.s["inputs"], **saved.s["blocks"][0]["input_bindings"],
    }
    for directory, records in ((saved.plan, saved.p["artifacts"].values()),
                               (saved.smoke, saved.s["blocks"][0]["artifacts"].values())):
        expected_bindings.update({str(directory / a["path"]): a["sha256"] for a in records})
    for replica in saved.s["blocks"][0]["replicas"]:
        expected_bindings.update({str(saved.smoke / a["path"]): a["sha256"] for a in replica["artifacts"].values()})
    assert bindings == expected_bindings
    for path, digest in bindings.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_real_preflight_counts_all_channels_including_zero_photon_frame(saved):
    report = saved.preflight()
    assert report["registry_frames"] == 9000
    assert report["calibration_rows"] == 5120 and report["live_channels"] == 4 * 5120
    assert report["frames"] == saved.selected.to_dict(orient="records")
    assert report["candidate_photons"] == {str(seed): 5 for seed in S.SEEDS}
    assert report["response_executed"] is False and report["execution_admitted"] is False
    assert report["code_sha256"] == saved.p["code_sha256"]
    assert saved.verifier_calls == ["plan", "smoke"]
    assert json.loads((saved.directory / "preflight.json").read_text()) == report


@pytest.mark.parametrize("kind", ["plan", "smoke"])
def test_wrong_expected_summary_hash_fails_before_saved_validation(saved, kind):
    with pytest.raises(ValueError, match="changed bound input"):
        saved.admit(**{f"expected_{kind}_sha256": "0" * 64})
    assert saved.verifier_calls == []


@pytest.mark.parametrize("case", ["missing", "wrong"])
def test_registry_digest_is_required_and_checked(saved, case):
    if case == "missing":
        del saved.p["registry"]["registry_sha256"]
    else:
        saved.p["registry"]["registry_sha256"] = "0" * 64
    saved.commit()
    error, message = (KeyError, "registry_sha256") if case == "missing" else (ValueError, "wrong frozen registry")
    with pytest.raises(error, match=message):
        saved.admit()


@pytest.mark.parametrize("column", ["pbtime", "intensity"])
def test_selected_exposure_mismatch_is_not_hidden_by_valid_artifact_hash(saved, column):
    events = saved.events.copy()
    events.loc[0, column] += 1.
    saved.save_events(events)
    with pytest.raises(ValueError, match="changed frame exposure metadata"):
        saved.admit()


def test_saved_root_collision_is_rejected_without_replacing_root_algorithm(saved):
    seed = S.SEEDS[0]
    q = saved.photons[seed].copy()
    q.loc[1, "photon_id"] = q.loc[0, "photon_id"]
    assert not q.duplicated(list(M.K.KEYS)).any()
    saved.save_photons(seed, q)
    with pytest.raises(AssertionError):
        saved.admit()


def test_duplicate_full_photon_key_is_rejected(saved):
    seed = S.SEEDS[0]
    q = saved.photons[seed]
    saved.save_photons(seed, pd.concat([q, q.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="duplicate full photon key"):
        saved.admit()


def test_wrong_frozen_seed_schedule_is_rejected_before_verification(saved):
    saved.s["seeds"][0] += 100
    saved.commit()
    with pytest.raises(ValueError, match="frozen two-seed smoke"):
        saved.admit()
    assert saved.verifier_calls == []


def test_preflight_rejects_wrong_draw_seed_even_with_recomputed_roots(saved):
    seed = S.SEEDS[0]
    q = saved.photons[seed].assign(draw_seed=np.uint64(seed + 100))
    q["photon_id"] = M.roots(q)
    saved.save_photons(seed, q)
    assert len(saved.admit()[3][seed]) == 5
    with pytest.raises(ValueError, match="mixed or unexpected optical seed"):
        saved.preflight()
    assert not (saved.directory / "preflight.json").exists()


@pytest.mark.parametrize("case", ["truncated", "duplicate"])
def test_preflight_requires_complete_unique_calibration(saved, case):
    c = saved.calibration
    c = c.iloc[:-1] if case == "truncated" else pd.concat([c, c.iloc[:1]], ignore_index=True)
    saved.write_calibration(c)
    saved.s["blocks"][0]["input_bindings"][str(saved.stock)] = S.sha256(saved.stock)
    saved.commit()
    assert len(saved.admit()[2]) == len(c)
    message = "incomplete calibration channel domain" if case == "truncated" else "duplicate calibration identity"
    with pytest.raises(ValueError, match=message):
        saved.preflight()
    assert not (saved.directory / "preflight.json").exists()


def test_repeated_input_calibration_across_disjoint_blocks_fails_preflight(saved):
    original = saved.s["blocks"][0]
    blocks = []
    for index, events in enumerate((saved.events.iloc[:2], saved.events.iloc[2:])):
        block = deepcopy(original)
        block["artifacts"]["events"] = saved.table(saved.smoke, f"events_part{index}", events)
        for replica in block["replicas"]:
            seed = replica["seed"]
            q = saved.photons[seed]
            replica["artifacts"]["photons"] = saved.table(
                saved.smoke, f"photons_part{index}_{seed}", q[q.event.isin(events.event)])
        blocks.append(block)
    saved.s["blocks"] = blocks
    saved.commit()
    _, selected, calibration, photons, _, _ = saved.admit()
    assert len(selected) == 4 and len(calibration) == 2 * 5120
    assert {seed: len(q) for seed, q in photons.items()} == {seed: 5 for seed in S.SEEDS}
    with pytest.raises(ValueError, match="duplicate calibration identity"):
        saved.preflight()
    assert not (saved.directory / "preflight.json").exists()


def test_conflicting_transitive_binding_is_rejected(saved):
    saved.s["inputs"][str(saved.smoke / "summary.json")] = "0" * 64
    saved.commit()
    with pytest.raises(ValueError, match="conflicting hashes for one path"):
        saved.admit()


def test_changed_saved_photon_bytes_are_rejected(saved):
    record = saved.s["blocks"][0]["replicas"][0]["artifacts"]["photons"]
    path = saved.smoke / record["path"]
    path.write_bytes(path.read_bytes() + b"changed after binding")
    with pytest.raises(ValueError, match="changed bound input"):
        saved.admit()


def test_exact_four_frame_census_is_required_even_with_quiet_events(saved):
    saved.save_events(saved.events.iloc[1:])
    with pytest.raises(ValueError, match="exactly the four saved smoke frames"):
        saved.admit()


def test_plan_must_point_to_the_hash_bound_smoke_directory(saved):
    saved.p["smoke"] = str(saved.directory / "other-smoke")
    saved.commit()
    with pytest.raises(ValueError, match="plan and saved photons differ"):
        saved.admit()
    assert saved.verifier_calls == []
