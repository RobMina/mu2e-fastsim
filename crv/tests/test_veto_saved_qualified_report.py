"""Saved-only, nonempty synthetic response evidence and adversarial resealed artifacts."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import saved_qualified_report as T
from crvfastsim.veto import saved_qualified_response as V
from crvfastsim.veto import bounded_response as R, batch_artifacts as A, frame_registry as F
from crvfastsim.veto.replay import sha256


def dump(path, value):
    path.write_bytes(A._canonical(value))


def reseal(batch):
    complete = T._json(batch / A.COMPLETE)
    initial = T._json(batch / A.INCOMPLETE)
    complete["artifacts"] = {name: {"bytes": (batch / name).stat().st_size, "sha256": sha256(batch / name)}
                             for name in initial["expected_artifacts"]}
    complete["artifact_bytes"] = sum(record["bytes"] for record in complete["artifacts"].values())
    complete["incomplete_sha256"] = sha256(batch / A.INCOMPLETE)
    dump(batch / A.COMPLETE, complete)
    dump(batch / A.PENDING, complete)


def seal_child(directory, tag):
    child, top = directory / tag, directory / "run"
    reseal(child)
    dump(top / (tag + ".json"), T._json(child / A.COMPLETE))
    summary = T._json(top / "summary.json")
    next(record for record in summary["records"] if record["path"] == tag)["receipt_sha256"] = sha256(child / A.COMPLETE)
    dump(top / "summary.json", summary)
    reseal(top)


def edit_json(path, change):
    value = T._json(path)
    change(value)
    dump(path, value)


def products(registry, selected, calibration, q, seed, arm):
    digest = F.digest(registry)
    cond = R.Q.conditions(registry, selected.drop(columns="frame_id"), calibration, digest)
    eq, ec = R.Q.admit(q, cond, registry, digest, seed)
    helper = F.helper_seed(registry, seed)
    phases = R.W.W.phase_table(ec, helper).set_index("label").reset_index()
    optical = eq[["event", "chan", "photon_id", "time_historical_relative"]].rename(
        columns={"photon_id": "root", "time_historical_relative": "time"}).assign(px=3, py=4, dark=False)
    dark = ec[ec.chan == 1][["event", "chan"]].assign(root=-1, time=550., px=5, py=6, dark=True)
    candidates = pd.concat([optical, dark], ignore_index=True).sort_values(["event", "chan"]).reset_index(drop=True)
    attempts = candidates.assign(generation=0, primary=True, multiplicity=1, inactive=False, v=3., p=.5,
                                 uniform=.2, fired=True, charge=1., pe=1., photon=np.where(candidates.dark, -1, candidates.root),
                                 root_time=candidates.time, root_px=candidates.px, root_py=candidates.py)
    count = len(candidates)
    phase = phases.set_index(["event", "chan"]).loc[pd.MultiIndex.from_frame(candidates[["event", "chan"]]), "phase"].to_numpy()
    start = np.ceil((candidates.time.to_numpy() - 12.5 - phase) / 12.5).astype(np.int64)
    clusters = candidates[["event", "chan"]].assign(cluster=np.arange(count), phase=phase,
        first_charge=candidates.time, last_charge=candidates.time, charges=1, starttdc=start, length=2, prekeep=True)
    sample_rows = np.repeat(np.arange(count), 3)
    samples = candidates.iloc[sample_rows][["event", "chan"]].reset_index(drop=True)
    samples = samples.assign(cluster=sample_rows, tdc=start[sample_rows] + np.tile(np.arange(3), count),
                             valid=np.tile([True, True, False], count), voltage=.1,
                             time_relative=(start[sample_rows] + np.tile(np.arange(3), count)) * 12.5 + phase[sample_rows],
                             cluster_prekeep=True, voltage_noiseless=.1, noise=0., above_zs=False,
                             zs_margin_v=-.1, precast_adc=120.)
    digi_rows = np.repeat(np.arange(count), 12)
    digis = candidates.iloc[digi_rows][["event", "chan"]].reset_index(drop=True).assign(
        starttdc=start[digi_rows], sample=np.tile(np.arange(12), count), adc=120)
    waveforms = candidates[["event", "chan"]].assign(wf=np.arange(count), starttdc=start, n=12,
        pedestal_used=100., max_adc_minus_pedestal=20., peak_margin_adc=15., near_threshold=False)
    fits = candidates.iloc[digi_rows][["event", "chan"]].reset_index(drop=True).assign(
        wf=digi_rows, tdc=start[digi_rows] + np.tile(np.arange(12), count), adc=120)
    pulses = candidates[["event", "chan"]].assign(wf=np.arange(count), starttdc=start, seq=0,
        pe=1. + list(V.M.P.ARMS).index(arm) * .1, time_relative=500., time_reco=520., failed=False,
        zerondf=False, channel_ignored=False, fit_A=10., peph=1., time=500., height=20., beta=10.,
        chi2=1., letime=490., ped=100., pedfromdb=True, fit_mu=500., fit_b=10.)
    data = dict(conditions=ec, candidates=candidates, attempts=attempts, charges=attempts.copy(), digis=digis,
                pulses=pulses, samples=samples, clusters=clusters, phases=phases, waveforms=waveforms, fit_samples=fits)
    data = {name: F.restore(table, registry) for name, table in data.items()}
    frames = selected.copy()
    for name, table in data.items():
        counts = (table[table.valid] if name == "samples" else table).groupby("frame_id").size()
        frames[name] = frames.frame_id.map(counts).fillna(0).astype(np.int64)
    data["frames"] = frames
    limits = R.Limits()
    wave = dict(contract=R.W.CONTRACT, charges=count, clusters=count, reach=10, charge_reach_cells=count * 10,
                waveform_length_upper=23, waveform_cells_upper=count * 23, limits=asdict(limits.waveform),
                peak_memory_certified=False)
    receipt = dict(contract=R.CONTRACT, registry_sha256=digest, optical_seed=seed, response_seed=seed,
        helper_seed=helper, arm=arm, frames=len(selected), live_channels=len(ec), candidate_photons=len(q),
        photons=len(q), dark_candidates=len(dark), attempts=count, charges=count, digis=count, pulses=count,
        limits=asdict(limits), waveform_admission=wave, fit_contract=R.P.CONTRACT,
        thermal_domain="all status-live detector channels, including quiet frames", execution_admitted=False,
        pulse_fit_partition_exact_certified=False)
    return data, receipt


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    base = tmp_path_factory.mktemp("saved-qualified")
    directory = base / "archive"
    directory.mkdir()
    registry = F.freeze(pd.DataFrame(dict(input_sha256="a" * 64, run=1, subrun=2, event=np.arange(9000),
                                          pbtime=-20., intensity=3.)))
    selected = registry.iloc[[0, 4, 400, 8999]].reset_index(drop=True)
    calibration = pd.DataFrame(dict(input_sha256="a" * 64, chan=np.arange(5120), pedestal=100., pulsearea=1000.,
                                   pulseheight=20., timeoffset=2., status=1, roc=1, feb=1))
    calibration.loc[:1, "status"] = 0
    photons = {}
    for seed in V.SEEDS:
        q = selected[list(F.FRAME_KEYS)].copy().assign(draw_seed=np.uint64(seed), chan=0, substep=0, photon_type=0,
              ordinal=0, source_row=0, time_historical_relative=500., time_raw_wrapped_relative=500., live_channel=True)
        for arm in V.M.P.ARMS:
            q[arm + "_kept"] = True
        q["photon_id"] = V.M.roots(q)
        photons[seed] = q
    inputs = {}
    for label in ("plan", "smoke"):
        parent = base / label
        parent.mkdir()
        dump(parent / "summary.json", {"synthetic": label})
        inputs[str(parent / "summary.json")] = sha256(parent / "summary.json")
    assets = {}
    for label in ("map", "template"):
        path = base / label
        path.write_text("synthetic-" + label)
        assets[str(path)] = sha256(path)
    code = {str(path): sha256(path) for path in V.code_paths()}
    digest = F.digest(registry)
    evidence = {"plan": {"summary_sha256": inputs[str(base / "plan" / "summary.json")]},
                "smoke": {"saved_ledgers_exact": True, "root_union_collision_free": True}}
    context = dict(registry_sha256=digest, batch_identity={"plan_sha256": inputs[str(base / "plan" / "summary.json")],
        "smoke_sha256": inputs[str(base / "smoke" / "summary.json")], "contract": R.CONTRACT},
        arms=list(V.M.P.ARMS), seeds=list(V.SEEDS))
    tags = [f"seed{seed}_{arm}" for seed in V.SEEDS for arm in V.M.P.ARMS]
    names = ["summary.json", "registry.parquet", "selected.parquet", "calibration.parquet", "admission.json"]
    names += [tag + ".json" for tag in tags] + [f"code{i}.txt" for i in range(len(code))]
    records, comparisons = [], {}
    with A.BatchArtifacts(directory / "run", max_bytes=V.ROOT_CAP, context=context, expected_artifacts=names) as top:
        for name, table in (("registry", registry), ("selected", selected), ("calibration", calibration)):
            top.write_parquet(name + ".parquet", table)
        top.write_json("admission.json", evidence)
        for i, path in enumerate(code):
            with top.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
        for seed in V.SEEDS:
            reference = None
            for arm in V.M.P.ARMS:
                tag = f"seed{seed}_{arm}"
                expected = [f"{v}_{s}.parquet" for v in V.VARIANTS for s in R.STAGES] + ["validation.json"]
                child_context = {**context, "batch_identity": {**context["batch_identity"], "frame_ids": selected.frame_id.tolist(),
                                  "variants": list(V.VARIANTS)}, "arms": [arm], "seeds": [seed]}
                data, receipts = {}, {}
                with A.BatchArtifacts(directory / tag, max_bytes=V.BATCH_CAP, context=child_context, expected_artifacts=expected) as tx:
                    for variant, sub in {"full": selected, "part0": selected.iloc[::2], "part1": selected.iloc[1::2]}.items():
                        q = V._photon_subset(photons[seed], sub)
                        data[variant], receipts[variant] = products(registry, sub, calibration, q, seed, arm)
                        for name, table in data[variant].items():
                            tx.write_parquet(f"{variant}_{name}.parquet", table)
                    partition = R.compare_partition(data["full"], [data["part0"], data["part1"]])
                    subset = R.compare_partition(data["full"], [data["part0"]])
                    tx.write_json("validation.json", dict(partition=partition, subset=subset, receipts=receipts))
                if arm == "historical":
                    reference = data["full"]
                else:
                    comparisons[f"seed{seed}_historical_vs_{arm}"] = V.structural_comparisons(reference, data["full"], registry)
                top.write_json(tag + ".json", T._json(directory / tag / A.COMPLETE))
                records.append(dict(path=tag, seed=seed, arm=arm, receipt_sha256=sha256(directory / tag / A.COMPLETE),
                                    elapsed_seconds=.1, counts=receipts["full"], partition=partition, subset=subset))
        summary = dict(schema=V.SCHEMA, scope=V.SCOPE, registry_sha256=digest, frames=4, registry_frames=9000,
            seeds=list(V.SEEDS), arms=list(V.M.P.ARMS), thermal_rate=V.THERMAL_RATE,
            sipm_config=asdict(replace(V.SI.SiPMConfig(), thermal_rate=V.THERMAL_RATE)),
            waveform_config=asdict(replace(V.WF.WaveformConfig(), single_pe_file=str(base / "template"))),
            inputs={**inputs, **assets}, code_sha256=code, records=records, structural_comparisons=comparisons,
            byte_caps=dict(run=V.ROOT_CAP, each_arm_seed=V.BATCH_CAP, maximum_total=V.ROOT_CAP + len(tags) * V.BATCH_CAP),
            all_tested_physical_partition_fields_exact=True, execution_admitted=False, F=None,
            finite_library_uncertainty=None, timing_closure=False)
        top.write_json("summary.json", summary)
    return base, (registry, selected, calibration, photons, inputs, evidence)


@pytest.fixture
def saved(archive, tmp_path, monkeypatch):
    base, admitted = archive
    directory = tmp_path / "saved"
    shutil.copytree(base / "archive", directory)
    calls = []
    def admission(*args):
        calls.append(args)
        return deepcopy(admitted)
    monkeypatch.setattr(V, "admission", admission)
    monkeypatch.setattr(R.D, "MAP_SHA", sha256(base / "map"))
    monkeypatch.setattr(R.D, "TEMPLATE_SHA", sha256(base / "template"))
    def forbidden(*args, **kwargs):
        pytest.fail("saved verification attempted response regeneration")
    for module, names in ((V, ("run",)), (R, ("run_arm",)), (V.M.K, ("make_photons",)), (V.M, ("generate",)),
                          (R.S, ("optical_candidates", "thermal_candidates", "simulate_candidates")),
                          (R.S.S, ("optical_candidates", "thermal_candidates", "simulate_candidates")),
                          (R.W, ("admit", "synthesize")), (R.W.W, ("synthesize", "reconstruct")),
                          (R.P, ("reconstruct", "reconstruct_matrix", "fit_buckets")), (R.P.LM, ("fit_gumbel",)),
                          (R.P.P, ("reconstruct", "reconstruct_full", "fit_gumbel")), (V.SI, ("simulate",)),
                          (V.WF, ("single_pe_template", "synthesize"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    return directory, tmp_path / "report.json", calls


def test_saved_only_report_rebuilds_every_stage_and_structural_comparison(saved):
    directory, out, calls = saved
    result = T.verify(directory, out)
    assert T._json(out) == result
    assert len(calls) == 1
    assert result["frames"] == 4 and result["registry_frames"] == 9000
    assert len(result["records"]) == 8
    assert len(result["frame_counts"]) == 8 and len(result["operation_comparisons"]) == 2
    compact = T.compact(result)
    assert len(compact["counts"]) == 8 and len(compact["comparisons"]) == 8
    assert compact["persisted_path_bytes"] == result["persisted_path_bytes"]
    assert result["observed_input_hash_bound_response"] is True
    assert result["regenerated_optics"] is result["regenerated_response"] is result["execution_admitted"] is False
    assert result["F"] is None
    for record in result["records"]:
        assert record["counts"]["photons"] == 4 and record["counts"]["dark_candidates"] == 4
        assert record["counts"]["pulses"] == 8 and record["counts"]["digis"] == 8
        assert record["partition"]["frame_ids"] == [0, 4, 400, 8999]
        assert record["subset"]["frame_ids"] == [0, 400]
        assert set(record["partition"]["stages"]) == set(R.STAGES)
    assert all(value["pulses"]["changed_common"] == 8 for value in result["structural_comparisons"].values())
    assert result["persisted_path_bytes"] == sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def test_output_is_exclusive_and_no_work_precedes_rejection(saved):
    directory, out, calls = saved
    out.write_text("keep me")
    with pytest.raises(ValueError, match="new saved-report"):
        T.verify(directory, out)
    assert out.read_text() == "keep me" and not calls


@pytest.mark.parametrize("case", ["failed", "missing_stage", "missing_pending", "pending", "hash", "context", "expected", "cap"])
def test_transaction_protocol_tampering_fails(saved, case):
    directory, out, _ = saved
    tag = "seed3027000_historical"
    child = directory / tag
    if case == "failed":
        (child / A.FAILED).write_text("failed")
    elif case == "missing_stage":
        (child / "part1_attempts.parquet").unlink()
    elif case == "missing_pending":
        (child / A.PENDING).unlink()
    elif case == "pending":
        (child / A.PENDING).write_text("{}")
    elif case == "hash":
        with (child / "full_pulses.parquet").open("ab") as stream:
            stream.write(b"altered")
    else:
        def change(value):
            if case == "context":
                value["context"]["seeds"] = [3027001]
            elif case == "expected":
                value["expected_artifacts"].remove("part1_attempts.parquet")
            else:
                value["max_bytes"] += 1
        edit_json(child / A.INCOMPLETE, change)
        seal_child(directory, tag)
    with pytest.raises((ValueError, AssertionError)):
        T.verify(directory, out)
    assert not out.exists()


@pytest.mark.parametrize("case", ["wrong_frame", "domain", "frame_count", "condition", "phase", "optical_root", "thermal_time", "fit_field", "missing_field"])
def test_resealed_stage_corruption_cannot_self_certify(saved, case):
    directory, out, _ = saved
    tag = "seed3027000_historical"
    child = directory / tag
    stage = {"wrong_frame": "part0_frames", "domain": "full_pulses", "frame_count": "full_frames",
             "condition": "full_conditions", "phase": "full_phases", "optical_root": "full_candidates",
             "thermal_time": "full_candidates", "fit_field": "part1_pulses", "missing_field": "part1_pulses"}[case]
    path = child / (stage + ".parquet")
    table = pd.read_parquet(path)
    if case == "wrong_frame":
        table = pd.read_parquet(child / "part1_frames.parquet")
    elif case == "domain":
        table.loc[0, "frame_id"] = 8998
    elif case == "frame_count":
        table.loc[0, "pulses"] += 1
    elif case == "condition":
        table.loc[0, "pedestal"] += 1
    elif case == "phase":
        table.loc[0, "phase"] += 1e-8
    elif case == "optical_root":
        table.loc[~table.dark, "root"] += 1
    elif case == "thermal_time":
        table.loc[table.dark, "time"] = 1600.
    elif case == "missing_field":
        table = table.drop(columns="fit_b")
    else:
        table.loc[0, "fit_A"] += 1e-10
    table.to_parquet(path, index=False)
    seal_child(directory, tag)
    with pytest.raises((ValueError, AssertionError)):
        T.verify(directory, out)
    assert not out.exists()


@pytest.mark.parametrize("case", ["receipt", "receipt_limits", "receipt_wave", "partition", "summary_count", "comparison",
                                  "code_missing", "snapshot", "input_missing", "config", "summary_caps", "seed", "registry"])
def test_resealed_provenance_and_claims_fail(saved, case):
    directory, out, _ = saved
    top = directory / "run"
    tag = "seed3027000_historical"
    if case in ("receipt", "receipt_limits", "receipt_wave", "partition"):
        def change(value):
            if case == "receipt":
                value["receipts"]["full"]["photons"] += 1
            elif case == "receipt_limits":
                value["receipts"]["full"]["limits"]["sipm"]["attempts"] += 1
            elif case == "receipt_wave":
                value["receipts"]["full"]["waveform_admission"]["waveform_cells_upper"] += 1
            else:
                value["subset"]["stages"]["pulses"]["rows"] += 1
        edit_json(directory / tag / "validation.json", change)
        seal_child(directory, tag)
    elif case == "snapshot":
        (top / "code0.txt").write_text("changed source snapshot")
        reseal(top)
    elif case == "registry":
        path = top / "registry.parquet"
        table = pd.read_parquet(path)
        table.loc[8998, "intensity"] += 1
        table.to_parquet(path, index=False)
        reseal(top)
    else:
        def change(value):
            if case == "summary_count":
                value["records"][0]["counts"]["photons"] += 1
            elif case == "comparison":
                value["structural_comparisons"]["seed3027000_historical_vs_arrival_only"]["pulses"]["changed_common"] = 0
            elif case == "code_missing":
                value["code_sha256"].pop(next(iter(value["code_sha256"])))
            elif case == "input_missing":
                value["inputs"].pop(value["waveform_config"]["single_pe_file"])
            elif case == "config":
                value["sipm_config"]["thermal_rate"] = 0.
            elif case == "summary_caps":
                value["byte_caps"]["each_arm_seed"] += 1
            else:
                value["records"][0]["seed"] += 1
        edit_json(top / "summary.json", change)
        reseal(top)
    with pytest.raises((ValueError, AssertionError)):
        T.verify(directory, out)
    assert not out.exists()


def test_changed_current_code_rejected_without_editing_source_files(saved, monkeypatch):
    directory, out, _ = saved
    actual = V.M.P.bound
    target = str(V.code_paths()[0])
    def changed(path, digest):
        if str(path) == target:
            raise ValueError("changed current code")
        return actual(path, digest)
    monkeypatch.setattr(V.M.P, "bound", changed)
    with pytest.raises(ValueError, match="changed current code"):
        T.verify(directory, out)
    assert not out.exists()


def test_persisted_cap_rechecked_from_path_bytes(tmp_path):
    directory = tmp_path / "batch"
    context = dict(registry_sha256="a" * 64, batch_identity={"test": True}, arms=["historical"], seeds=[1])
    with A.BatchArtifacts(directory, max_bytes=10000, context=context, expected_artifacts=["value.json"]) as tx:
        tx.write_json("value.json", {"value": 1})
    for name in (A.INCOMPLETE, A.COMPLETE, A.PENDING):
        edit_json(directory / name, lambda value: value.update(max_bytes=1))
    reseal(directory)
    with pytest.raises(ValueError, match="persisted byte cap"):
        T._batch(directory, context, ["value.json"], 1)


def test_self_consistent_swapped_partitions_still_violate_declared_frame_schedule(saved):
    directory, out, _ = saved
    tag = "seed3027000_historical"
    child = directory / tag
    for stage in R.STAGES:
        left, right = (child / f"{variant}_{stage}.parquet" for variant in ("part0", "part1"))
        a, b = left.read_bytes(), right.read_bytes()
        left.write_bytes(b)
        right.write_bytes(a)
    data = {variant: {stage: pd.read_parquet(child / f"{variant}_{stage}.parquet") for stage in R.STAGES}
            for variant in V.VARIANTS}
    validation = T._json(child / "validation.json")
    validation["receipts"]["part0"], validation["receipts"]["part1"] = validation["receipts"]["part1"], validation["receipts"]["part0"]
    validation["partition"] = R.compare_partition(data["full"], [data["part0"], data["part1"]])
    validation["subset"] = R.compare_partition(data["full"], [data["part0"]])
    dump(child / "validation.json", validation)
    summary = T._json(directory / "run" / "summary.json")
    summary["records"][0]["subset"] = validation["subset"]
    dump(directory / "run" / "summary.json", summary)
    seal_child(directory, tag)
    with pytest.raises(AssertionError):
        T.verify(directory, out)
    assert not out.exists()


def test_changed_actual_bound_input_rejected(saved, monkeypatch):
    directory, out, _ = saved
    summary = T._json(directory / "run" / "summary.json")
    path = next(path for path in summary["inputs"] if Path(path).name == "map")
    actual = V.M.P.bound
    def changed(candidate, digest):
        if str(candidate) == path:
            raise ValueError("changed saved source input")
        return actual(candidate, digest)
    monkeypatch.setattr(V.M.P, "bound", changed)
    with pytest.raises(ValueError, match="changed saved source input"):
        T.verify(directory, out)
    assert not out.exists()


def test_stage_channel_outside_authoritative_conditions_rejected(saved):
    directory, out, _ = saved
    tag = "seed3027000_historical"
    path = directory / tag / "full_pulses.parquet"
    table = pd.read_parquet(path)
    table.loc[0, "chan"] = 2
    table.to_parquet(path, index=False)
    seal_child(directory, tag)
    with pytest.raises(ValueError, match="all-live channel domain"):
        T.verify(directory, out)
    assert not out.exists()


@pytest.mark.parametrize("case", ["uniform_string", "fractional_tdc", "bool_string", "probability", "float_identity"])
def test_resealed_symmetric_invalid_schema_rejected(saved, case):
    directory, out, _ = saved
    tag = "seed3027000_historical"
    child = directory / tag
    for variant in V.VARIANTS:
        stages = ("attempts", "charges") if case in ("uniform_string", "probability") else ("samples", "fit_samples") if case == "fractional_tdc" else ("pulses",)
        for stage in stages:
            path = child / f"{variant}_{stage}.parquet"
            table = pd.read_parquet(path)
            if case == "uniform_string":
                table["uniform"] = "not-a-number"
            elif case == "fractional_tdc":
                table["tdc"] = table.tdc.astype(float) + .25
            elif case == "bool_string":
                table["failed"] = "False"
            elif case == "probability":
                table["p"] = 1.5
            else:
                table["seq"] = table.seq.astype(float)
            table.to_parquet(path, index=False)
    seal_child(directory, tag)
    with pytest.raises(ValueError, match="numeric stage schema|numeric domain"):
        T.verify(directory, out)
    assert not out.exists()


def test_self_consistent_unapproved_frozen_helper_rejected(saved, monkeypatch):
    directory, out, _ = saved
    top = directory / "run"
    helper = str(Path(R.W.W.__file__))
    current = sha256(helper)
    monkeypatch.setattr(V, "FROZEN_WAVEFORM_SHA", "0"*64)
    assert T._json(top / "summary.json")["code_sha256"][helper] == current
    with pytest.raises(ValueError, match="unapproved frozen waveform"):
        T.verify(directory, out)
    assert not out.exists()
