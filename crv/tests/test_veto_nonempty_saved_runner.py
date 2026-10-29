"""Local saved-runner integration, NOT real detector or upstream admission evidence.

ROOT TTrees, Parquets, pins, frozen transaction receipts, coverage and output
transactions are real. Only geometry_only is replaced with tiny synthetic
geometry. The inherited membership/verifier reports are fixture assertions, not
outputs of real raw/frozen-pilot admission; unused membership product tables are
empty but typed. No finder, recovery, remote input or physical validation runs.
"""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto import nonempty_saved_control as N


def _forbidden(*args, **kwargs):
    pytest.fail("saved runner executed finder/recovery")


@pytest.fixture(autouse=True)
def no_finder(monkeypatch):
    monkeypatch.setattr(N.R, "recover", _forbidden)
    monkeypatch.setattr(N.R.C, "find_event", _forbidden)


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _sha(path):
    return N.M.S.C.sha256(path)


def _source(path):
    return {"path": str(path), "sha256": _sha(path)}


def _ttree(file, name, types, values):
    tree = file.mktree("crvdump/" + name, types)
    if len(values):
        tree.extend({field: values[field] for field in types})


def _calibration():
    return ak.Array({"channel": np.arange(5120, dtype=np.int32),
                     "status": np.zeros(5120, dtype=np.int32),
                     "gain": np.ones(5120, dtype=np.float64)})


CALIB_TYPES = {"channel": "int32", "status": "int32", "gain": "float64"}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    source = tmp_path / "saved"
    source.mkdir()
    input_sha = "a" * 64
    frames = N.F.freeze(pd.DataFrame({
        "input_sha256": [input_sha] * 3, "run": [7] * 3, "subrun": [2] * 3,
        "event": [10, 11, 12], "pbtime": [1.1, -4.2, 0.], "intensity": [1.] * 3,
    }))
    events = ak.Array({
        "run": np.full(3, 7, dtype=np.uint32), "subrun": np.full(3, 2, dtype=np.uint32),
        "event": np.array([10, 11, 12], dtype=np.uint32),
        "pbtime": frames.pbtime.to_numpy().astype(np.float32).astype(float),
        "spill": np.ones(3, dtype=np.int32), "evlen": np.full(3, 1700.),
        "ndigi": np.zeros(3, dtype=np.int32), "npulse": np.array([4, 2, 0], dtype=np.int32),
        "ncluster": np.array([2, 1, 0], dtype=np.int32),
    })
    event_types = {name: str(ak.to_numpy(events[name]).dtype) for name in events.fields}
    pulse = ak.Array({
        "ev": (2 << 20) + np.array([11, 10, 10, 11, 10, 10]),
        "chan": [12, 8, 1, 16, 4, 9], "time": [-3.25, 8.25, 1.25, -2.5, 6.75, 8.5],
        "pe": [11.1, 10.123456789, 13.2, 12.3, 14.4, 15.5], "seq": [0, 2, 0, 1, 1, 3],
    })
    order = np.array([1, 2, 4, 5, 0, 3])
    columns = {name: np.zeros(6, dtype=dtype) for name, dtype in N.R.PULSE_TYPES.items() if name != "wfidx"}
    columns["wfidx"] = ak.unflatten(np.array([], dtype=np.int32), np.zeros(6, dtype=np.int64))
    chan = ak.to_numpy(pulse.chan)[order]
    native = dict(run=np.full(6, 7), subrun=np.full(6, 2), event=np.array([10] * 4 + [11] * 2),
                  idx=np.array([0, 1, 2, 3, 0, 1]), bar=chan // 4, sipm=chan % 4,
                  seq=ak.to_numpy(pulse.seq)[order], time=ak.to_numpy(pulse.time)[order],
                  pe=ak.to_numpy(pulse.pe)[order])
    columns.update({name: values.astype(N.R.PULSE_TYPES[name]) for name, values in native.items()})
    stock_p = ak.Array(columns)
    rows = []
    for event, idx, sector, members, layers in (
        (10, 0, 1, [0, 2], [2, 1]), (10, 1, 1, [3, 1], [0, 2]), (11, 0, 2, [1, 0], [1, 0]),
    ):
        row = {name: False if dtype == "bool" else 0 if "int" in dtype else 0.
               for name, dtype in N.R.CLUSTER_TYPES.items()}
        row.update(run=7, subrun=2, event=event, idx=idx, sectortype=sector,
                   nLayers=2, sidehits0=1, sidehits1=1, poscalc=True,
                   start=-2., end=10., avgtime=3., pe=25., pulses=members, layers=layers)
        rows.append(row)
    clusters = ak.Array(rows)
    clusters = ak.Array({name: ak.values_astype(clusters[name], dtype)
                         for name, dtype in N.R.CLUSTER_TYPES.items()} |
                        {name: ak.values_astype(clusters[name], np.int32) for name in ("pulses", "layers")})
    cluster_types = N.R.CLUSTER_TYPES | {"poscalc": "int32", "pulses": "var * int32", "layers": "var * int32"}
    paths = {name: source / filename for name, filename in {
        "pulses": "pulses.parquet", "counters": "counters.parquet", "stock": "fixed.root",
        "check": "check.json", "fast_metadata": "metadata.json", "reco": "reco.root",
        "table": "table.root", "recovered": "recovered.parquet",
    }.items()}
    ak.to_parquet(pulse, paths["pulses"])
    ak.to_parquet(clusters, paths["recovered"])
    pd.DataFrame({"bar": np.arange(5)}).to_parquet(paths["counters"], index=False)
    geometry = SimpleNamespace(layer=np.array([0, 1, 2, 0, 1]), sector=np.array([0, 0, 0, 1, 1]),
                               sector_type=np.array([1, 2]))
    monkeypatch.setattr(N.M, "geometry_only", lambda sources: geometry)
    with uproot.recreate(paths["reco"]) as file:
        _ttree(file, "events", event_types, events)
        _ttree(file, "pulses", N.R.PULSE_TYPES, stock_p)
        _ttree(file, "clusters", cluster_types, clusters)
        _ttree(file, "calib", CALIB_TYPES, _calibration())
        file.mktree("crvdump/digis", {"idx": "int32"})
    with uproot.recreate(paths["table"]) as file:
        _ttree(file, "pulses", N.R.PULSE_TYPES, stock_p)
        file.mktree("crvdump/digis", {"idx": "int32"})
    with uproot.recreate(paths["stock"]) as file:
        _ttree(file, "calib", CALIB_TYPES, _calibration())
    conditions = "b" * 64
    _json(paths["fast_metadata"], {"input_sha256": input_sha, "pulse_sha256": _sha(paths["pulses"]),
                                   "conditions_sha256": conditions, "seed": 17})
    _json(paths["check"], {"conditions_equal": True, "conditions_sha256": conditions,
                          "input_sha256": input_sha, "stock_sha256": _sha(paths["stock"]),
                          "fast_metadata_sha256": _sha(paths["fast_metadata"])})
    input_binding = {"path": str(source / "inherited-not-opened.root"), "sha256": input_sha}
    recovery = {
        "schema": "crv-retained-fast-recovery-v1", "frame_closure": {"passed": True},
        "finder_config": asdict(N.R.finder_config()), "frames": 3, "coincs": 3,
        "sources": {"input": input_binding} | {key: _source(paths[key]) for key in
                    ("pulses", "counters", "stock", "check", "fast_metadata")},
        "output_sha256": {"recovered.parquet": _sha(paths["recovered"])},
    }
    control = {
        "schema": "crv-retained-pulse-reco-check-v1", "structural_and_roundtrip_validated": True,
        "nested_membership_validated": True, "clusters": 3,
        "bindings": {"input": input_binding, "reco": _source(paths["reco"]), "table": _source(paths["table"])},
        "events_check": {"on": N.RC.check_events(frames, events, 3)},
        "affected_clusters": [{"run": 7, "subrun": 2, "event": 11, "added_count": 1}],
    }
    recovery_path = _json(source / "recovery.json", recovery)
    control_path = _json(source / "control.json", control)
    card = N.G.VetoCard(name="synthetic-runner-only", dt_lo=0., dt_hi=10., min_layers=2)
    reference = _json(source / "reference.json", {"card": asdict(card) | {"sha": card.sha()}})
    pair = _json(source / "paired.json", {"reference": str(reference), "reference_sha256": _sha(reference)})
    grid = pd.DataFrame({"left": [0., 5., 10., 15.], "right": [5., 10., 15., 20.], "weight": [.25] * 4})
    code = {str(path): _sha(path) for path in N.M.code_paths()}
    summary = dict(
        schema=N.M.SCHEMA, scope=N.M.SCOPE,
        hashes={"expected_pilot_sha256": "c" * 64, "expected_verified_sha256": "d" * 64,
                "expected_paired_sha256": _sha(pair)},
        inputs={str(pair): _sha(pair)}, code_sha256=code, registry_sha256=N.F.digest(frames),
        unique_frames=3, ledger_rows=0, seeds=list(N.M.S.SEEDS), arms=list(N.M.S.ARMS),
        counts={}, comparisons={}, grid={"card_sha": card.sha(), "bins": len(grid)},
        finder_config=asdict(N.M.C.finder_config()), coincidence_limits=asdict(N.V.P.Limits()),
        ignore_channels=True, ignored_pulse_policy=N.V.IGNORED_POLICY, output_cap_bytes=N.M.OUTPUT_CAP,
        memory_before={"peak_rss_bytes_cumulative": 0, "current_rss_bytes": None},
        memory_after={"peak_rss_bytes_cumulative": 0, "current_rss_bytes": None}, coincidence_coverage_seconds=0.,
        **dict.fromkeys(N.V.FALSE_FIELDS, False), **dict.fromkeys(N.V.NULL_FIELDS),
    )
    assert set(summary) == N.V.SUMMARY_KEYS
    tables = {name: pd.DataFrame({field: pd.Series(dtype=dtype) for field, dtype in types.items()})
              for name, types in N.V.TABLE_TYPES.items()}
    tables.update(registry=frames, selected=frames.copy(), grid=grid)
    membership = source / "membership"
    names = ["summary.json", *(name + ".parquet" for name in tables),
             *(f"code{i}.txt" for i in range(len(code)))]
    context = dict(registry_sha256=summary["registry_sha256"],
                   batch_identity={"contract": N.M.SCHEMA, **summary["hashes"]},
                   arms=list(N.M.S.ARMS), seeds=list(N.M.S.SEEDS))
    with N.A.BatchArtifacts(membership, max_bytes=N.M.OUTPUT_CAP, context=context, expected_artifacts=names) as tx:
        tx.write_json("summary.json", summary)
        for name, table in tables.items():
            tx.write_parquet(name + ".parquet", table)
        for i, path in enumerate(code):
            with tx.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
    verified = _json(source / "verified.json", {
        "schema": N.V.SCHEMA, "summary_sha256": _sha(membership / "summary.json"),
        "frozen_grid_exact": True, "saved_artifact_protocol_exact": True,
        "grid": summary["grid"], "registry_sha256": summary["registry_sha256"],
    })
    kwargs = dict(recovery=recovery_path, control=control_path, membership=membership,
                  verified=verified, out=tmp_path / "audit",
                  expected_recovery_sha256=_sha(recovery_path), expected_control_sha256=_sha(control_path),
                  expected_membership_sha256=_sha(membership / "summary.json"), expected_verified_sha256=_sha(verified))
    return SimpleNamespace(kwargs=kwargs, paths=paths, recovery=recovery, membership=membership,
                           frames=frames, out=kwargs["out"])


def test_saved_runner_commits_nonempty_without_finder(evidence):
    result = N.run(**evidence.kwargs)
    assert result["schema"] == N.SCHEMA
    assert result["roles"] == {"nonempty": 0, "opposite_side_attachment": 1,
                               "positive_passing_coverage": 0, "zero_coincidence": 2}
    assert result["selected_frames"] == 3 and result["selected_pulses"] == 6
    assert result["comparison"]["exact_member_set_matches"] == 3
    assert result["comparison"]["stock_links"] == result["comparison"]["fast_links"] == 6
    assert all(value["exact_mismatches"] == 0 for value in result["comparison"]["properties"].values())
    assert result["coverage_exact"] is True and result["calibration_exact"] is True
    assert result["pilot_adapter_admitted"] is False
    assert result["regenerated_finder"] is False and result["execution_admitted"] is False
    assert result["F"] is None and result["finite_library_uncertainty"] is None
    assert result["root_metadata"]["calib"]["entries"] == 5120
    assert result["root_metadata"]["fixed_calib"]["entries"] == 5120
    assert 0 < result["persisted_path_bytes"] <= N.CAP
    assert (evidence.out / N.A.COMPLETE).exists()
    assert not (evidence.out / N.A.FAILED).exists()
    persisted = json.loads((evidence.out / "summary.json").read_text())
    assert persisted == {key: value for key, value in result.items() if key != "persisted_path_bytes"}
    pulses = pd.read_parquet(evidence.out / "pulses.parquet")
    assert pulses.chan.tolist() == [8, 1, 4, 9, 12, 16]
    assert pulses.idx.tolist() == [0, 1, 2, 3, 0, 1]
    assert pulses.pe.iloc[0] == 10.123456789
    assert pulses.stock_pe.iloc[0] == float(np.float32(10.123456789))
    assert not {"wf", "starttdc", "time_relative"} & set(pulses)
    stock = pd.read_parquet(evidence.out / "stock_coverage.parquet")
    fast = pd.read_parquet(evidence.out / "fast_coverage.parquet")
    pd.testing.assert_frame_equal(stock, fast, check_exact=True)
    assert (stock.track_grid_coverage > 0).any()
    quiet = stock.loc[stock.frame_id.eq(2)].iloc[0]
    assert quiet.n_coincidences == 0 and quiet.track_grid_coverage == 0


@pytest.mark.parametrize("target", ["recovered", "membership"])
def test_saved_runner_late_input_mutation_never_completes(evidence, monkeypatch, target):
    original = N.A.BatchArtifacts.write_parquet
    mutated = []

    def mutate_after_output_write(tx, name, table, **options):
        original(tx, name, table, **options)
        if tx.directory == evidence.out and not mutated:
            path = evidence.paths["recovered"] if target == "recovered" else evidence.membership / "grid.parquet"
            with path.open("ab") as stream:
                stream.write(b"late mutation")
            mutated.append(path)

    monkeypatch.setattr(N.A.BatchArtifacts, "write_parquet", mutate_after_output_write)
    with pytest.raises(ValueError, match="changed bound input|artifact hash or byte count changed"):
        N.run(**evidence.kwargs)
    assert mutated
    assert (evidence.out / N.A.FAILED).exists()
    assert not (evidence.out / N.A.COMPLETE).exists()
    assert not (evidence.out / "summary.json").exists()


@pytest.mark.parametrize("pin", ["recovery", "control", "membership", "verified"])
def test_saved_runner_rejects_each_wrong_external_pin(evidence, monkeypatch, pin):
    kwargs = evidence.kwargs | {"expected_" + pin + "_sha256": "0" * 64}
    # Wrong pins must fail before decoding ROOT/Parquet or loading geometry.
    monkeypatch.setattr(N.M, "geometry_only", _forbidden)
    monkeypatch.setattr(N, "tree", _forbidden)
    monkeypatch.setattr(N, "parquet", _forbidden)
    with pytest.raises(ValueError, match="changed bound input"):
        N.run(**kwargs)
    assert not evidence.out.exists()


def test_saved_runner_rejects_changed_calibration_even_with_updated_pins(evidence):
    calibration = _calibration()
    gain = ak.to_numpy(calibration.gain).copy()
    gain[5119] = np.nextafter(gain[5119], np.inf)
    calibration = ak.with_field(calibration, gain, "gain")
    with uproot.recreate(evidence.paths["stock"]) as file:
        _ttree(file, "calib", CALIB_TYPES, calibration)
    # Update all relevant pins: the full condition comparison must catch this,
    # not merely a stale file hash, and not just channel/status comparison.
    evidence.recovery["sources"]["stock"] = _source(evidence.paths["stock"])
    check = json.loads(evidence.paths["check"].read_text())
    check["stock_sha256"] = _sha(evidence.paths["stock"])
    _json(evidence.paths["check"], check)
    evidence.recovery["sources"]["check"] = _source(evidence.paths["check"])
    _json(evidence.kwargs["recovery"], evidence.recovery)
    kwargs = evidence.kwargs | {"expected_recovery_sha256": _sha(evidence.kwargs["recovery"])}
    with pytest.raises(ValueError, match="control/fixed-stock calibration differs"):
        N.run(**kwargs)
    assert not evidence.out.exists()


def test_saved_runner_cannot_overwrite_completed_output(evidence):
    N.run(**evidence.kwargs)
    before = {path.name: path.read_bytes() for path in evidence.out.iterdir()}
    with pytest.raises(ValueError, match="choose a new output directory"):
        N.run(**evidence.kwargs)
    assert {path.name: path.read_bytes() for path in evidence.out.iterdir()} == before


@pytest.mark.parametrize("field", ["input_sha256", "stock_sha256", "fast_metadata_sha256"])
def test_saved_runner_rejects_resealed_inconsistent_check(evidence, field):
    path = evidence.paths["check"]
    value = json.loads(path.read_text())
    value[field] = "f" * 64
    _json(path, value)
    evidence.recovery["sources"]["check"] = _source(path)
    _json(evidence.kwargs["recovery"], evidence.recovery)
    evidence.kwargs["expected_recovery_sha256"] = _sha(evidence.kwargs["recovery"])
    with pytest.raises(ValueError, match="historical check input/stock binding differs"):
        N.run(**evidence.kwargs)
    assert not evidence.out.exists()


def test_saved_runner_rejects_counters_before_parse(evidence, monkeypatch):
    path = evidence.paths["counters"]
    path.write_bytes(b"x" * (2 * 1024**2 + 1))
    evidence.recovery["sources"]["counters"] = _source(path)
    _json(evidence.kwargs["recovery"], evidence.recovery)
    evidence.kwargs["expected_recovery_sha256"] = _sha(evidence.kwargs["recovery"])
    monkeypatch.setattr(N.M, "geometry_only", _forbidden)
    with pytest.raises(ValueError, match="counters text cap exceeded before parse"):
        N.run(**evidence.kwargs)
    assert not evidence.out.exists()
