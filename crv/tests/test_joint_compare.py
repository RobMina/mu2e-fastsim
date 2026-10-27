"""Local synthetic saved artifacts only: no response, ROOT, remote inputs or rescoring."""
import hashlib
import json

import numpy as np
import pytest

from crvfastsim import response_config as RC
from crvfastsim.kit_task import TERMS
from crvfastsim.objective import joint, joint_compare as J, validation as V


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def columns():
    # Large exact integer event IDs, deliberately nonmonotonic.
    keys = np.array([[7, 3, 2**53 + 5], [7, 4, 2**53 + 1], [8, 3, 2**53 + 2]], dtype=np.int64)
    out = {}
    for table, event, slot in (("event", np.repeat(np.arange(3), 4), np.tile(np.arange(4), 3)),
                                ("candidate", np.arange(3), np.ones(3, int)),
                                ("hit", np.repeat(np.arange(3), 40), np.ones(120, int))):
        for i, name in enumerate(("run", "subrun", "event")):
            out[f"{table}_{name}"] = keys[event, i]
        out[table + "_slot"] = slot
        out[table + "_body"] = np.full(len(event), 4, np.int64)
    out.update(event_selected=np.array([False, True, False, False] * 3),
               event_spread_eligible=np.array([False, True, False, False] * 3),
               event_group_rms=np.repeat([2., 4., 8.], 4),
               candidate_cluster_index=np.zeros(3, np.int64), candidate_selected=np.ones(3, bool),
               candidate_pe_layer=np.array([[0., 20., 40., 60.], [10., 30., 50., 70.], [20., 40., 60., 80.]]),
               hit_pulse_index=np.tile(np.arange(40), 3),
               hit_bar_id=np.tile(np.arange(256, 296), 3), hit_sipm_id=np.zeros(120, np.int64),
               hit_sequence_index=np.zeros(120, np.int64), hit_side=np.zeros(120, np.int64),
               hit_group_count=np.full(120, 40, np.int64),
               hit_pe=np.full(120, 30.), hit_pe_path=np.full(120, 30. / 1.1),
               hit_along=np.zeros(120), hit_path=np.full(120, 1.1),
               hit_time_residual=np.tile([-2., 2.], 60))
    for name in ("both_lines", "spread_member", "spread_eligible", "pe_along_eligible", "pe_path_eligible"):
        out["hit_" + name] = np.ones(120, bool)
    for name in ("candidate_count", "candidate_positive_count", "timing_match_count", "timing_rejected_count",
                 "eligible_count", "selected_pulse_count", "group_count"):
        out["event_" + name] = np.ones(12, np.int64)
    return out


def data_provenance(events, **extra):
    return dict(complete=True, sample_role="measured_data", counts=dict(events=events),
                time_shift_ns=None, time_policy="unshifted stock data; no recalibration", full_files=True,
                inputs=[dict(complete=True, full_file=True, reference_comparison=dict(
                    status="exact", path="/unavailable/saved-data-reference.json", sha256="a" * 64,
                    matched_scoring_fields=["events", "files", "exposure_ns", "lines_z", "lines_x",
                                            "lines_xz", "h", "oot_n"]))], **extra)


def write_capture(root, cols):
    summary = joint.summarize(cols)
    np.savez_compressed(root / "joint.npz", **cols, summary_json=np.asarray(json.dumps(summary)))
    save(root / "joint.json", summary)
    return {name: dict(path=str(root / name), sha256=V.digest(root / name)) for name in ("joint.json", "joint.npz")}


def mutate(root, arm, filename, function):
    path = root / arm / f"{filename}.json"
    record = json.loads(path.read_text())
    function(record)
    save(path, record)


def replace_capture(root, arm, change):
    with np.load(root / arm / "joint.npz", allow_pickle=False) as archive:
        cols = {k: archive[k] for k in archive.files if k != "summary_json"}
    change(cols)
    records = write_capture(root / arm, cols)
    mutate(root, arm, "provenance", lambda p: p.update(joint_files=records))


@pytest.fixture
def block(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("postprocessing attempted simulation/scoring/original input reads")
    for name in ("run", "verify", "inspect_inputs", "revision", "runtime_assets"):
        monkeypatch.setattr(V, name, forbidden)
    monkeypatch.setattr(V.R, "objective", forbidden)
    manifest = dict(schema=1, events=3, events_with_steps=2, excluded=[],
                    inputs=[dict(path="/unavailable/input.root", sha256="frozen-input", events=3, events_with_steps=2)],
                    assets={k: dict(path=f"/unavailable/{k}", sha256=k) for k in
                            ("calib", "deviation", "reference", "background", "time_shift")})
    sources = {"synthetic.py": "frozen-source"}
    source_sha = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    code = {name: dict(sha=name, mode="diagnostic-worktree", published=False, sources=sources,
                       source_sha256=source_sha) for name in ("crvfastsim", "mu2efastsim")}
    for number, arm in enumerate(J.ARMS):
        root = tmp_path / arm
        root.mkdir()
        config = RC.defaults()
        config["timing"].update(phase_scope="per_event", timeoffset_mode="zero")
        for key in ("calib", "deviation", "background", "time_shift"):
            config["assets"][key] = manifest["assets"][key]["path"]
        knobs = {}
        if arm in ("atten", "joint"):
            knobs["atten_T1"] = J.ATTEN
            config["response"]["atten_ratio"]["T1"] = J.ATTEN
        if arm in ("yield", "joint"):
            config["response"]["sectors"]["T1"]["yield_per_mev"] = J.YIELD_LOW
        result = dict(events_with_steps=2, events_without_steps=1, target=2, seed=123,
                      passes=[dict(events=2)], timers={}, knobs=knobs, response_state=dict(events=3, exposure_ns=300000.),
                      resolved_config=RC.record(config))
        score = dict(objective=10. - number, n_terms=32, excluded=[],
                     terms={name: dict(chi2ndf=10. - number) for name in TERMS})
        cols = columns()
        cols["event_group_rms"] += number
        cols["hit_time_residual"] *= 1 + number
        # Each table is independently reordered: ordinal matching would be wrong.
        rng = np.random.default_rng(number)
        for table in ("event", "candidate", "hit"):
            permutation = rng.permutation(len(cols[table + "_slot"]))
            for key in list(cols):
                if key.startswith(table + "_"):
                    cols[key] = cols[key][permutation]
        files = write_capture(root, cols)
        provenance = dict(manifest=manifest, manifest_sha256="frozen-manifest", code=code, code_after=code,
            seed=123, runtime_assets={"lut": dict(path="/unavailable/lut", sha256="lut")},
            python="test", versions={"numpy": "test"}, knobs=knobs, jax=False, calibration=True,
            response_config=dict(RC.record(config), canonical=RC.canonical(config), requested_file=None),
            diagnostic_code=True, joint_capture=True, code_policy="unpublished-diagnostic-snapshot",
            status="complete", integrity_errors=[], joint_files=files)
        for name, record in (("r3", result), ("comparison", score), ("provenance", provenance)):
            save(root / f"{name}.json", record)
        if arm in ("nominal", "atten"):
            ref = tmp_path / f"saved_{arm}.json"
            save(ref, result)
            V.compare_replay(ref, root / "r3.json", tmp_path / f"{arm}_capture_gate.json")
    return tmp_path


def test_legacy_receipt_missing_only_attachment_field(block):
    point = {k: json.loads((block / 'nominal' / f'{k}.json').read_text()) for k in ('r3', 'provenance')}
    config = point['provenance']['response_config']['values']
    config['finder'].pop('add_opposite_side_pulses')
    point['provenance']['response_config'].update(RC.record(config), canonical=RC.canonical(config))
    point['r3']['resolved_config'] = RC.record(config)
    before = json.dumps(point, sort_keys=True)
    J._config(point, 'nominal')
    assert json.dumps(point, sort_keys=True) == before
    config['pulse_reco'].pop('min_beta')
    with pytest.raises(ValueError, match='unknown or missing'):
        J._config(point, 'nominal')


def test_mixed_attachment_eras_rejected(block):
    def legacy_receipt(p):
        config = p['response_config']['values']
        config['finder'].pop('add_opposite_side_pulses')
        p['response_config'].update(RC.record(config), canonical=RC.canonical(config))
    def legacy_result(r):
        config = r['resolved_config']['values']
        config['finder'].pop('add_opposite_side_pulses')
        r['resolved_config'] = RC.record(config)
    mutate(block, 'yield', 'provenance', legacy_receipt)
    mutate(block, 'yield', 'r3', legacy_result)
    with pytest.raises(ValueError, match='finder schema/settings differ'):
        J.load_block(block, {})


def test_full_cli_reordered_identity_join(block, capsys):
    out = block / "report.json"
    J.main(["--directory", str(block), "--out", str(out)])
    assert "PASS" in capsys.readouterr().out
    report = json.loads(out.read_text())
    assert report["passed"]
    assert report["coordinates"]["yield_fraction"] == .7
    assert report["cohorts"]["equal_exposure_identities"]
    for i, name in enumerate(J.ARMS):
        cohort = report["cohorts"]["groups"]["T1"]["arms"][name]
        assert cohort["common_valid"]["rows"] == 3
        assert cohort["common_valid"]["paired_difference"]["mean"] == i
        assert cohort["common_valid"]["paired_difference"]["rms"] == 0
        assert len(report["arms"][name]["terms"]) == 32
        assert all(t["minus_nominal"] == -i for t in report["arms"][name]["terms"].values())
    assert report["residual_diagnostics"]["standardization"]["status"] == "no_data_standardization"
    assert len(report["files"]) >= 24
    assert report["arms"]["nominal"]["measured"]["by_body"]["T10"]["selected_layer_pe"][0]["mean"] == 10
    original = out.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        J.compare(block, out)
    assert original == out.read_bytes()


@pytest.mark.parametrize("filename,change,match", [
    ("provenance", lambda p: p["manifest"]["inputs"][0].update(sha256="other"), "frozen provenance"),
    ("provenance", lambda p: p.update(manifest_sha256="other"), "frozen provenance"),
    ("provenance", lambda p: p.update(status="running"), "incomplete"),
    ("provenance", lambda p: p.update(integrity_errors=["changed"]), "integrity"),
    ("provenance", lambda p: p["runtime_assets"]["lut"].update(sha256="other"), "frozen provenance"),
    ("provenance", lambda p: p["code_after"]["crvfastsim"].update(sha="other"), "code integrity"),
    ("provenance", lambda p: p.update(unexpected="new provenance field"), "frozen provenance"),
    ("r3", lambda p: p.update(events_with_steps=1), "incomplete"),
    ("r3", lambda p: p.update(seed=5), "seed mismatch"),
    ("r3", lambda p: p["response_state"].update(exposure_ns=123.), "time exposure"),
    ("comparison", lambda p: p.update(n_terms=31), "32-term"),
    ("comparison", lambda p: p["terms"].pop(TERMS[0]), "32 term"),
])
def test_reject_invalid_arms(block, filename, change, match):
    mutate(block, "joint", filename, change)
    out = block / "bad.json"
    with pytest.raises(ValueError, match=match):
        J.compare(block, out)
    assert not out.exists()


@pytest.mark.parametrize("mode", ["duplicate", "different", "missing", "duplicate_hit", "incomplete_slots"])
def test_reject_identity_failures(block, mode):
    def change(cols):
        if mode == "duplicate":
            for name in ("run", "subrun", "event", "slot"):
                cols["event_" + name][1] = cols["event_" + name][0]
        elif mode == "different":
            for table in ("event", "hit", "candidate"):
                cols[table + "_run"] += 50
        elif mode == "duplicate_hit":
            for name in ("run", "subrun", "event", "pulse_index"):
                cols["hit_" + name][1] = cols["hit_" + name][0]
        else:
            # Drop a complete event, or just one slot, with consistent saved summaries.
            keep = np.arange(len(cols["event_slot"])) != 0 if mode == "incomplete_slots" else cols["event_run"] == 7
            for name in list(cols):
                if name.startswith("event_"):
                    cols[name] = cols[name][keep]
    replace_capture(block, "yield", change)
    with pytest.raises(ValueError, match="duplicate|identities differ|orphan|four slots|census"):
        J.compare(block, block / "bad.json")


@pytest.mark.parametrize("mode", ["missing", "fail", "candidate_hash", "reference_hash"])
def test_require_exact_capture_gates(block, mode):
    path = block / "nominal_capture_gate.json"
    if mode == "missing":
        path.unlink()
    else:
        gate = json.loads(path.read_text())
        if mode == "fail":
            gate["passed"] = False
        else:
            gate["candidate" if mode == "candidate_hash" else "reference"]["sha256"] = "changed"
        save(path, gate)
    with pytest.raises((ValueError, FileNotFoundError)):
        J.compare(block, block / "bad.json")


def test_effective_coordinates_not_arm_labels(block):
    # Internally consistent receipts and resolved settings, but wrong intervention.
    def provenance(p):
        p["response_config"]["values"]["response"]["sectors"]["T1"]["yield_per_mev"] = 20000.
        config = p["response_config"]["values"]
        p["response_config"].update(RC.record(config), canonical=RC.canonical(config))
    mutate(block, "yield", "provenance", provenance)
    def result(r):
        config = r["resolved_config"]["values"]
        config["response"]["sectors"]["T1"]["yield_per_mev"] = 20000.
        r["resolved_config"] = RC.record(config)
    mutate(block, "yield", "r3", result)
    with pytest.raises(ValueError, match="unexpected effective response"):
        J.compare(block, block / "bad.json")


def test_selection_gains_losses_and_common_valid(block):
    def change(cols):
        first = (cols["event_subrun"] == 3) & (cols["event_run"] == 7)
        cols["event_selected"][first & (cols["event_slot"] == 1)] = False
        cols["event_selected"][first & (cols["event_slot"] == 0)] = True
        cols["event_group_rms"][(cols["event_run"] == 8) & (cols["event_slot"] == 1)] = np.nan
    replace_capture(block, "yield", change)
    report = J.compare(block, block / "report.json")
    selected = report["cohorts"]["groups"]["all"]["arms"]["yield"]
    assert (selected["gained"], selected["lost"], selected["common_selected"]) == (1, 1, 2)
    assert selected["common_selected_finite"]["rows"] == 1
    assert selected["common_valid"]["paired_difference"]["mean"] == 2


def test_cells_exact_unbinned_flows_and_upper_edges():
    cols = columns()
    cols["hit_along"][:5] = [-1., 1., np.nan, 2., 0.]
    cols["hit_path"][:5] = [1., 3., 1., 1., np.inf]
    cols["hit_pe"][:2] = [0., 200.]
    cols["hit_time_residual"][:2] = [-30., 40.]
    cols["hit_time_residual"][5] = np.nan
    result = J.residual_cells(cols)
    assert result["coordinate_inrange_rows"] == 117
    assert result["flows"]["along"]["overflow"] == 1
    assert result["flows"]["path"]["nonfinite"] == 1
    assert result["residual_flow"]["underflow"] == 1
    assert result["residual_flow"]["overflow"] == 1
    assert sum(c["count"] for c in result["cells"]) == 116
    first = ((4 * 5 + 0) * 4 + 0) * 5 + 0
    last = ((4 * 5 + 4) * 4 + 3) * 5 + 4
    assert result["cells"][first]["mean"] == -30
    assert result["cells"][last]["mean"] == 40  # No residual-range truncation.
    expected = np.array([3., np.nan, -2., 4., 1., 8.])
    cells = J.grouped_stats(np.array([0, 0, 1, 0, 1, 1]), expected, 3)
    assert cells[0]["median"] == 3.5
    assert cells[1]["median"] == 1.
    assert cells[2]["mean"] is None
    assert cells[1]["rms"] == pytest.approx(np.std([-2., 1., 8.]))


@pytest.mark.parametrize("sharded", [False, True])
def test_optional_independent_data_and_standardization(block, sharded):
    data = block / "data"
    data.mkdir()
    cols = columns()
    for table in ("event", "hit", "candidate"):
        cols[table + "_run"] += 100  # Independent population must not be event matched.
    files = write_capture(data, cols)
    provenance = data_provenance(3, joint_files=files)
    save(data / "provenance.json", provenance)
    if sharded:
        (data / "joint.npz").rename(data / "raw.npz")
        save(data / "summary.json", dict(complete=True,
            provenance=dict(path="provenance.json", sha256=V.digest(data / "provenance.json")),
            raw_shards=[dict(path="raw.npz", sha256=files["joint.npz"]["sha256"])]))
    report = J.compare(block, block / "data_report.json", data)
    standard = report["residual_diagnostics"]["standardization"]
    assert standard["common_cell_count"] == 1
    assert standard["data_cell_weights"] == [1.]
    assert standard["samples"]["data"]["data_weighted_second_moment"] == 4.
    assert standard["samples"]["atten"]["data_weighted_second_moment"] == 16.
    assert standard["samples"]["atten"]["data_weighted_minus_data"] == 12.


def test_support_threshold_and_population_conditional_distinction():
    def sample(counts, moments):
        return dict(cells=[dict(count=n, second_moment=m) for n, m in zip(counts, moments)])
    samples = {name: sample([30, 90, 29], [1., 9., 100.]) for name in J.ARMS}
    samples["data"] = sample([90, 30, 90], [1., 9., 100.])
    samples["joint"] = sample([90, 30, 31], [1., 9., 100.])
    report = J.support_report(samples)
    assert report["common_cell_indices"] == [0, 1]
    assert report["data_cell_weights"] == [.75, .25]
    assert report["samples"]["nominal"]["support_matched_second_moment"] == 7.
    assert report["samples"]["joint"]["support_matched_second_moment"] == 3.
    assert report["samples"]["nominal"]["data_weighted_second_moment"] == 3.
    assert report["samples"]["joint"]["data_weighted_second_moment"] == 3.
    samples["data"] = sample([29, 29, 29], [1., 9., 100.])
    report = J.support_report(samples)
    assert report["status"] == "no_common_support"
    assert all(row["data_weighted_second_moment"] is None for row in report["samples"].values())


def test_changed_capture_hash_rejected(block):
    with (block / "joint" / "joint.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="joint receipt mismatch"):
        J.compare(block, block / "bad.json")


def test_nonfinite_objective_term_refused(block):
    path = block / "yield" / "comparison.json"
    score = json.loads(path.read_text())
    score["terms"][TERMS[0]]["chi2ndf"] = float("inf")
    path.write_text(json.dumps(score))
    with pytest.raises(ValueError, match="nonfinite"):
        J.compare(block, block / "bad.json")


def test_capture_mutation_after_read_refused(block, monkeypatch):
    original = J.measured
    changed = False
    def measured(cols):
        nonlocal changed
        if not changed:
            with (block / "yield" / "joint.npz").open("ab") as stream:
                stream.write(b"mutation during report")
            changed = True
        return original(cols)
    monkeypatch.setattr(J, "measured", measured)
    with pytest.raises(ValueError, match="file changed during comparison"):
        J.compare(block, block / "bad.json")
    assert not (block / "bad.json").exists()


def test_diagnostic_source_hash_checked(block):
    def change(p):
        for key in ("code", "code_after"):
            p[key]["crvfastsim"]["sources"]["synthetic.py"] = "changed"
    mutate(block, "joint", "provenance", change)
    with pytest.raises(ValueError, match="source code hash mismatch"):
        J.compare(block, block / "bad.json")


def test_data_cross_shard_duplicate_refused(tmp_path):
    cols = columns()
    files = write_capture(tmp_path, cols)
    (tmp_path / "joint.npz").rename(tmp_path / "raw.npz")
    save(tmp_path / "provenance.json", data_provenance(6))
    save(tmp_path / "summary.json", dict(complete=True,
        provenance=dict(sha256=V.digest(tmp_path / "provenance.json")),
        raw_shards=[dict(path="raw.npz", sha256=files["joint.npz"]["sha256"])] * 2))
    with pytest.raises(ValueError, match="duplicate event/slot"):
        J.load_data(tmp_path, {})


def test_same_channel_membership_is_not_unique_pulse_identity():
    cols = columns()
    cols["hit_bar_id"][:] = 256
    # Distinct event-local pulse_index values remain valid even when memberships repeat.
    summary = joint.summarize(cols)
    keys, order = J._audit(cols, summary)
    assert len(keys) == len(order) == 12


@pytest.mark.parametrize("change,match", [
    (lambda p: p.update(sample_role="MC"), "sample_role"),
    (lambda p: p.pop("sample_role"), "sample_role"),
    (lambda p: p.update(time_shift_ns=0), "unshifted"),
    (lambda p: p.pop("time_shift_ns"), "unshifted"),
    (lambda p: p.update(time_policy="shifted"), "unshifted"),
    (lambda p: p.update(full_files=False), "full-file"),
    (lambda p: p.update(inputs=[]), "full-file"),
    (lambda p: p["inputs"][0]["reference_comparison"].update(status="not_requested"), "exact saved"),
    (lambda p: p["inputs"][0]["reference_comparison"].update(status="skipped_partial_capture"), "exact saved"),
    (lambda p: p["inputs"][0]["reference_comparison"].update(sha256=None), "hashed reference"),
    (lambda p: p["inputs"][0]["reference_comparison"].update(matched_scoring_fields=["events"]), "exact saved"),
])
def test_strict_measured_data_admission(tmp_path, change, match):
    provenance = data_provenance(3)
    change(provenance)
    save(tmp_path / "provenance.json", provenance)
    with pytest.raises(ValueError, match=match):
        J.load_data(tmp_path, {})


def test_data_require_shard_hash(tmp_path):
    write_capture(tmp_path, columns())
    save(tmp_path / "provenance.json", data_provenance(3))
    with pytest.raises(ValueError, match="requires hashed provenance"):
        J.load_data(tmp_path, {})


def test_streamed_preallocation_retains_one_shard(tmp_path, monkeypatch):
    import weakref
    records, expected = [], []
    for i in range(3):
        part = tmp_path / str(i)
        part.mkdir()
        cols = columns()
        for table in ("event", "hit", "candidate"):
            cols[table + "_run"] += i * 100
        expected.append(cols)
        files = write_capture(part, cols)
        records.append(dict(files["joint.npz"], summary=joint.summarize(cols)))
    save(tmp_path / "provenance.json", data_provenance(9))
    save(tmp_path / "summary.json", dict(complete=True,
        provenance=dict(sha256=V.digest(tmp_path / "provenance.json")), raw_shards=records))
    original, previous = J._npz, []
    def load(path, receipts):
        assert all(ref() is None for ref in previous), "previous raw shard still retained"
        cols, summary = original(path, receipts)
        previous[:] = [weakref.ref(v) for v in cols.values()]
        return cols, summary
    monkeypatch.setattr(J, "_npz", load)
    merged, summary, _ = J.load_data(tmp_path, {})
    assert all(ref() is None for ref in previous)
    assert summary["events"] == 9
    for name in merged:
        assert np.array_equal(merged[name], np.concatenate([c[name] for c in expected]))


def test_mc_columns_released_before_data(block, monkeypatch):
    import weakref
    original, refs = J.load_block, []
    def load(*args):
        points, gates = original(*args)
        refs.extend(weakref.ref(v) for p in points.values() for v in p["columns"].values())
        refs.extend(weakref.ref(p[k]) for p in points.values() for k in ("keys", "order"))
        return points, gates
    def data(*args):
        assert refs and all(ref() is None for ref in refs), "MC arrays overlap data allocation"
        cols = columns()
        return cols, joint.summarize(cols), data_provenance(3)
    monkeypatch.setattr(J, "load_block", load)
    monkeypatch.setattr(J, "load_data", data)
    report = J.compare(block, block / "report.json", "synthetic-data")
    assert report["cohorts"]["equal_exposure_identities"]
    assert report["arms"]["joint"]["provenance"]["status"] == "complete"


def test_multiplicity_full_integer_ids_slots_and_nonfinite():
    cols = columns()
    # Repeated event number across run/subrun, all above float64's exact range.
    cols["hit_event"][:] = 2**53 + 5
    cols["hit_slot"][:20] = 0
    cols["hit_time_residual"][:40] = np.nan
    cols["hit_group_count"][:40] = 10
    result = J.residual_cells(cols)
    cell = next(c for c in result["cells"] if c["rows"])
    assert cell["distinct_events"] == 3
    assert cell["distinct_event_slots"] == 4
    assert cell["finite_distinct_events"] == 2
    assert cell["finite_distinct_event_slots"] == 2
    assert cell["hit_group_count"] == dict(rows=120, values=[10, 40], counts=[40, 80])
    assert sum(c["hit_group_count"]["rows"] for c in result["cells"]) == 120
    cols["event_group_count"] = np.tile([0, 40, 0, 0], 3)
    cols["event_group_count"][5] = 10
    body = J.measured(cols)["by_body"]["T10"]
    assert body["event_group_count"] == dict(rows=12, values=[0, 10, 40], counts=[9, 1, 2])
    assert body["selected_event_group_count"] == dict(rows=3, values=[10, 40], counts=[1, 2])


def test_empty_residual_cells_have_zero_multiplicity():
    cols = columns()
    cols["hit_spread_member"][:] = False
    result = J.residual_cells(cols)
    assert all(c["distinct_events"] == c["distinct_event_slots"] == 0 for c in result["cells"])
    assert all(c["hit_group_count"] == dict(rows=0, values=[], counts=[]) for c in result["cells"])


@pytest.mark.parametrize("mode", ["dtype", "columns", "summary"])
def test_data_shard_schema_and_summary_receipts(tmp_path, mode):
    records = []
    for i in range(2):
        part = tmp_path / str(i)
        part.mkdir()
        cols = columns()
        for table in ("event", "hit", "candidate"):
            cols[table + "_run"] += i * 100
        if i == 1 and mode == "dtype":
            cols["hit_pe"] = cols["hit_pe"].astype(np.float32)
        if i == 1 and mode == "columns":
            cols["hit_extra"] = np.zeros(120)
        files = write_capture(part, cols)
        summary = joint.summarize(cols)
        if i == 1 and mode == "summary":
            summary["events"] += 1
        records.append(dict(files["joint.npz"], summary=summary))
    save(tmp_path / "provenance.json", data_provenance(6))
    save(tmp_path / "summary.json", dict(complete=True,
        provenance=dict(sha256=V.digest(tmp_path / "provenance.json")), raw_shards=records))
    with pytest.raises(ValueError, match="schemas differ|summary receipt mismatch"):
        J.load_data(tmp_path, {})


def test_monolithic_data_arrays_are_not_copied(tmp_path, monkeypatch):
    files = write_capture(tmp_path, columns())
    save(tmp_path / "provenance.json", data_provenance(3, joint_files=files))
    original, loaded = J._npz, {}
    def load(path, receipts):
        cols, summary = original(path, receipts)
        loaded.update(cols)
        return cols, summary
    monkeypatch.setattr(J, "_npz", load)
    result, _, _ = J.load_data(tmp_path, {})
    assert all(result[name] is values for name, values in loaded.items())
