"""Tiny prospective receipts with mocked strict conditions; no stock production certification."""
import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from crvfastsim import named_conditions, response_receipt as R, runconfig


def json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def keys(events=(1, 2), *, run=1400, subrun=1):
    return {"run": np.full(len(events), run, dtype=np.uint64),
            "subrun": np.full(len(events), subrun, dtype=np.uint64),
            "event": np.asarray(events, dtype=np.uint64)}


def tree(f, name, columns):
    types = {k: ("var * " + str(v.layout.content.dtype) if isinstance(v, ak.Array) else v.dtype)
             for k, v in columns.items()}
    f.mktree(name, types)
    f[name].extend(columns)


def root(path, *, stage="reco", ids=None, stock=False, windows=True, mutate=None):
    ids = keys() if ids is None else ids
    n = len(ids["event"])
    intcol = np.ones(n, np.int32)
    data = {"events": {**ids, "ndigi": intcol.copy(),
                       "npulse": intcol.copy() if stage != "digis" else intcol * 0,
                       "ncluster": intcol.copy() if stage != "digis" else intcol * 0}}
    names = ("digis",) if stage == "digis" else ("digis", "pulses", "clusters")
    for name in names:
        spec = R._COLUMNS[name]
        data[name] = {**{k: v.copy() for k, v in ids.items()},
                      **{k: intcol * 0 for k in spec["int"].split()},
                      **{k: intcol.astype(float) for k in spec["float"].split()}}
        for k in spec["jagged"].split():
            data[name][k] = ak.unflatten(np.zeros(n, np.int16 if k == "adc" else np.int32), np.ones(n, int))
    if stock and windows:
        data["events"].update(spill=intcol * 0, evlen=intcol.astype(float) * 100000, pbtime=intcol * 0.0)
    if mutate:
        mutate(data)
    with uproot.recreate(path) as f:
        for name, cols in data.items():
            tree(f, "crvdump/" + name, cols)
    return path


def steps(path, ids=None):
    ids = keys() if ids is None else ids
    with uproot.recreate(path) as f:
        tree(f, "dump/steps", ids)
        tree(f, "dump/particles", ids)
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    rc = runconfig.get("run1a_v01")
    geom = SimpleNamespace(nbars=1, sector_names=["T1"], sector=np.array([0]), module=np.array([0]),
                           layer=np.array([0]), counter=np.array([0]), center=np.zeros((1, 3)),
                           halflen=np.ones((1, 3)), local_to_world=np.array([[0, 1, 2]]))
    config = json_value(dataclasses.asdict(rc))
    conditions = {"schema": named_conditions.SCHEMA, "scope": named_conditions.SCOPE,
                  "run_config": rc.name, "config": config,
                  "config_sha256": hashlib.sha256(R._canonical(config).encode()).hexdigest(),
                  "geometry": named_conditions._geometry(geom), "stock_event_identity": {"count": 10}}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(conditions))
    cfg = dataclasses.replace(rc.chain_config(), window=runconfig.OFFSPILL_WINDOW)
    metadata = {"schema": R.METADATA_SCHEMA, "run_config": rc.name, "conditions": conditions,
                "conditions_manifest": str(manifest), "configured_window_ns": list(rc.window),
                "effective_window_ns": list(cfg.window), "window_selection": "offspill",
                "finder": json_value(dataclasses.asdict(rc.finder())),
                "timing_policy": "fixed_absolute_crop_no_pbtime_rebase", "onspill_equivalence": False,
                "stage": "reco", "seed": 1, "jax": False, "noise": True, "window_applied": False,
                "chain_config": json_value(dataclasses.asdict(cfg)), "warning": "test metadata preserved"}
    calls = []

    def load(path, requested_rc, requested_geom):
        calls.append((path, requested_rc, requested_geom))
        m = json.loads(path.read_text())
        if m["config"] != json_value(dataclasses.asdict(requested_rc)):
            raise ValueError("strict loader config mismatch")
        if m["geometry"] != named_conditions._geometry(requested_geom):
            raise ValueError("strict loader geometry mismatch")
        return object(), m

    monkeypatch.setattr(named_conditions, "load_conditions", load)
    inp = root(tmp_path / "input.root", stock=True)
    return SimpleNamespace(tmp=tmp_path, rc=rc, geom=geom, metadata=metadata, manifest=manifest,
                           inp=inp, out=tmp_path / "output.root", calls=calls)


def publish(case, *, mutate=None):
    before = R.snapshot_inputs({"input": case.inp})
    root(case.out, stage=case.metadata["stage"], mutate=mutate)
    return R.write_receipt(case.out, case.metadata, before)


def validate(case, receipt, **kw):
    return R.validate_receipt(case.out, receipt, run_config=kw.get("run_config", case.rc),
                              geom=kw.get("geom", case.geom), conditions_manifest=case.manifest)


def test_normal_receipt_preserves_metadata_and_is_not_v2(case):
    original = copy.deepcopy(case.metadata)
    receipt = publish(case)
    assert receipt == case.tmp / "output.root.receipt.json"
    payload = json.loads(receipt.read_text())
    assert payload["metadata"] == original == case.metadata
    assert {"chain.py", "response_receipt.py", "reco/pulsefit.py", "valhist/fasttables.py"} <= payload["sources"].keys()
    evidence = validate(case, receipt)
    assert len(case.calls) == 1
    assert evidence["status"] == "bound_fast_output_and_assets"
    assert evidence["conditional_response_supported"]
    assert evidence["population"]["status"] == "verified_exact_stock_event_universe"
    assert "stock_source_configuration_not_certified" in evidence["limitations"]
    assert not evidence["named_v2_pass"] and not evidence["transport_admitted"] and not evidence["e2e_admitted"]
    assert evidence["events"]["count"] == 2


@pytest.mark.parametrize("which", ["output", "input"])
def test_file_drift_rejected(case, which):
    receipt = publish(case)
    with (case.out if which == "output" else case.inp).open("ab") as f:
        f.write(b"drift")
    with pytest.raises(ValueError, match=which + ".*drift"):
        validate(case, receipt)


def test_input_drift_between_snapshot_and_production(case):
    before = R.snapshot_inputs({"input": case.inp})
    root(case.out)
    with case.inp.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="input drift"):
        R.write_receipt(case.out, case.metadata, before)
    assert not (case.tmp / "output.root.receipt.json").exists()


def test_exclusive_publication_and_no_retroactive_saved_hashes(case):
    root(case.out)
    before = R.snapshot_inputs({"input": case.inp})
    with pytest.raises(ValueError, match="historical"):
        R.write_receipt(case.out, case.metadata, before)
    root(case.out)
    with pytest.raises(ValueError, match="live unused"):
        R.write_receipt(case.out, case.metadata, dict(before))
    receipt = R.write_receipt(case.out, case.metadata, before)
    saved = receipt.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        R.write_receipt(case.out, case.metadata, before)
    assert receipt.read_bytes() == saved
    with pytest.raises(ValueError, match="live unused"):
        R.write_receipt(case.out, case.metadata, before, path=case.tmp / "second.json")


def test_missing_output_and_existing_dangling_receipt(case):
    before = R.snapshot_inputs({"input": case.inp})
    with pytest.raises(FileNotFoundError):
        R.write_receipt(case.out, case.metadata, before)
    root(case.out)
    path = case.tmp / "custom.json"
    path.symlink_to(case.tmp / "nonexistent")
    with pytest.raises(ValueError, match="already exists"):
        R.write_receipt(case.out, case.metadata, before, path=path)


@pytest.mark.parametrize("change", [
    {"schema": "legacy"}, {"stage": "bad"}, {"window_applied": True}, {"finder": {}},
    {"effective_window_ns": [0, float("nan")]}, {"effective_window_ns": [0, float("inf")]},
    {"effective_window_ns": [0, 99999]}, {"configured_window_ns": [0, 100000]},
    {"onspill_equivalence": True}, {"run_config": "run2_v10"}, {"conditions": {}},
])
def test_malformed_metadata_fails_publication(case, change):
    case.metadata.update(change)
    with pytest.raises(ValueError):
        publish(case)


def test_complete_chain_config_required_if_supplied(case):
    case.metadata["chain_config"].pop("sipm")
    with pytest.raises(ValueError, match="incomplete producer chain_config"):
        publish(case)


def test_metadata_conditions_and_requested_config_mismatch(case):
    receipt = publish(case)
    external = json.loads(case.manifest.read_text())
    external["stock_event_identity"]["count"] += 1
    case.manifest.write_text(json.dumps(external))
    with pytest.raises(ValueError, match="embedded/external"):
        validate(case, receipt)
    case.manifest.write_text(json.dumps(case.metadata["conditions"]))
    with pytest.raises(ValueError, match="requested run_config"):
        validate(case, receipt, run_config=runconfig.get("run2_v10"))
    with pytest.raises(ValueError, match="config mismatch"):
        validate(case, receipt, run_config=dataclasses.replace(case.rc, photon_yield_scale=0.1))
    wrong_geom = copy.deepcopy(case.geom)
    wrong_geom.center[0, 0] = 100
    with pytest.raises(ValueError, match="geometry mismatch"):
        validate(case, receipt, geom=wrong_geom)


def test_finder_control_not_admitted(case):
    case.metadata["finder"]["add_opposite_side_pulses"] = False
    case.metadata["chain_config"]["finder"] = case.metadata["finder"]
    receipt = publish(case)
    with pytest.raises(ValueError, match="unsupported finder"):
        validate(case, receipt)


def test_extra_chain_configuration_not_admitted(case):
    case.metadata["chain_config"]["feb_phase_width"] = 1
    receipt = publish(case)
    with pytest.raises(ValueError, match="unsupported producer chain_config"):
        validate(case, receipt)


def test_full_width_identity_is_sorted_and_not_packed(tmp_path):
    ids = keys((1, 1 + 2**20, 2**32 - 1), run=2**32 - 1, subrun=2**32 - 1)
    first = root(tmp_path / "one.root", ids=ids)
    second = root(tmp_path / "two.root", ids={k: v[::-1] for k, v in ids.items()})
    identity = R.event_identity(first)
    assert identity == R.event_identity(second)
    assert identity["count"] == 3
    matrix = np.column_stack([ids[k] for k in R.KEYS]).astype("<u8")
    assert identity["sha256"] == hashlib.sha256(matrix.tobytes()).hexdigest()


@pytest.mark.parametrize("field,value", [("event", -1), ("event", 2**32), ("subrun", 2**32),
                                          ("run", 2**32), ("event", 1.5), ("event", 1.0)])
def test_invalid_identity_range_or_noninteger(tmp_path, field, value):
    ids = keys((1,))
    ids[field] = np.array([value])
    path = root(tmp_path / "bad.root", ids=ids)
    with pytest.raises(ValueError, match="integer"):
        R.event_identity(path)


def test_duplicate_event_id_rejected(case):
    before = R.snapshot_inputs({"input": case.inp})
    root(case.out, ids=keys((1, 1)))
    with pytest.raises(ValueError, match="duplicate event"):
        R.event_identity(case.out)
    with pytest.raises(ValueError, match="duplicate event"):
        R.write_receipt(case.out, case.metadata, before)


@pytest.mark.parametrize("ids", [keys((1, 2**20)), keys(subrun=2**20),
                                 {**keys(), "run": np.array([1400, 1401], np.uint32)}])
def test_producer_capacity_and_one_run(case, ids):
    before = R.snapshot_inputs({"input": case.inp})
    root(case.out, ids=ids)
    with pytest.raises(ValueError, match="capacity|one run"):
        R.write_receipt(case.out, case.metadata, before)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["digis"]["event"].__setitem__(0, 99), "outside events"),
    (lambda d: d.pop("pulses"), "missing stage table"),
    (lambda d: d["digis"].pop("adc"), "missing required columns"),
    (lambda d: d["events"]["npulse"].__setitem__(0, 10), "count mismatch"),
    (lambda d: d["pulses"]["bar"].__setitem__(0, 2), "bar/SiPM range"),
    (lambda d: d["pulses"]["pe"].__setitem__(0, np.nan), "nonfinite"),
    (lambda d: d["clusters"].__setitem__("pulses", ak.Array([[1], [0]])), "collection index range"),
])
def test_invalid_tables_rejected(case, mutate, match):
    with pytest.raises(ValueError, match=match):
        publish(case, mutate=mutate)


@pytest.mark.parametrize("col,value", [("spill", 1), ("evlen", 200), ("pbtime", 1),
                                       ("evlen", np.nan), ("pbtime", np.inf)])
def test_actual_stock_window_checked(case, col, value):
    root(case.inp, stock=True, mutate=lambda d: d["events"][col].__setitem__(0, value))
    with pytest.raises(ValueError, match="stock " + col):
        publish(case)


def test_missing_window_evidence_is_not_inferred(case):
    root(case.inp, stock=True, windows=False)
    evidence = validate(case, publish(case))
    assert not evidence["conditional_response_supported"]
    assert evidence["timing_status"] == "unverified_missing_stock_window_columns"


def test_onspill_never_admitted(case):
    case.metadata.update(effective_window_ns=[200.0, 1500.0], window_selection="onspill")
    case.metadata["chain_config"]["window"] = [200.0, 1500.0]
    evidence = validate(case, publish(case))
    assert not evidence["conditional_response_supported"]
    assert evidence["timing_status"] == "unsupported_onspill"


@pytest.mark.parametrize("stage", ["chain", "digis"])
@pytest.mark.parametrize("reference", [False, True])
def test_step_arms_population_only_when_event_reference_bound(case, stage, reference):
    case.metadata.update(stage=stage, window_applied=True)
    inp = steps(case.tmp / "steps.root")
    mapping = {"input": inp}
    if reference:
        mapping["event_reference"] = case.inp
    before = R.snapshot_inputs(mapping)
    root(case.out, stage=stage)
    receipt = R.write_receipt(case.out, case.metadata, before)
    evidence = validate(case, receipt)
    assert evidence["conditional_response_supported"] is reference
    if not reference:
        assert evidence["population"]["status"] == "unverified_population_no_event_reference"


def test_event_reference_must_match_exactly(case):
    case.metadata.update(stage="chain", window_applied=True)
    inp = steps(case.tmp / "steps.root")
    before = R.snapshot_inputs({"input": inp, "event_reference": case.inp})
    root(case.out, ids=keys((1,)))
    with pytest.raises(ValueError, match="event universe mismatch"):
        R.write_receipt(case.out, case.metadata, before)


def test_old_metadata_is_not_a_receipt(case):
    root(case.out)
    path = case.tmp / "old.metadata.json"
    path.write_text(json.dumps(case.metadata))
    with pytest.raises(ValueError, match="receipt schema"):
        validate(case, path)


def test_missing_full_chain_config_is_limited(case):
    case.metadata.pop("chain_config")
    evidence = validate(case, publish(case))
    assert not evidence["conditional_response_supported"]
    assert "full_producer_chain_configuration_not_recorded" in evidence["limitations"]


def test_source_drift_rejected(case, monkeypatch):
    before = R.snapshot_inputs({"input": case.inp})
    root(case.out)
    original = R._sources
    monkeypatch.setattr(R, "_sources", lambda: {**original(), "new_source.py": {}})
    with pytest.raises(ValueError, match="producer source drift"):
        R.write_receipt(case.out, case.metadata, before)


def test_saved_receipt_tampering_checked(case):
    receipt = publish(case)
    payload = json.loads(receipt.read_text())
    payload["events"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate(case, receipt)


def test_snapshot_records_cannot_be_modified(case):
    before = R.snapshot_inputs({"input": case.inp})
    before["input"]["size"] += 1
    root(case.out)
    with pytest.raises(ValueError, match="snapshot was modified"):
        R.write_receipt(case.out, case.metadata, before)


def test_custom_receipt_destination(case):
    before = R.snapshot_inputs({"input": case.inp})
    root(case.out)
    destination = case.tmp / "new-receipt.json"
    assert R.write_receipt(case.out, case.metadata, before, path=destination) == destination
    assert validate(case, destination)["status"] == "bound_fast_output_and_assets"


def test_real_fasttable_writer_empty_collections(case):
    from valhist import fasttables
    before = R.snapshot_inputs({"input": case.inp})
    empty = np.zeros(0, np.int64)
    digis = {"ev": empty, "bar": empty, "sipm": empty, "starttdc": empty,
             "adc": np.zeros((0, 8), np.int16)}
    pulses = {"ev": empty, **{k: empty for k in "idx bar sipm seq flags pedfromdb".split()},
              **{k: np.zeros(0) for k in R._COLUMNS["pulses"]["float"].split()}, "wfidx": []}
    clusters = {"ev": empty, **{k: [] for k in " ".join(R._COLUMNS["clusters"].values()).split()}}
    fasttables.write(str(case.out), 1400, digis, pulses, clusters,
                     events=np.array([(1 << 20) | 1, (1 << 20) | 2]))
    receipt = R.write_receipt(case.out, case.metadata, before)
    assert validate(case, receipt)["events"]["count"] == 2


def test_input_step_packing_checked_before_any_population_inference(case):
    case.metadata.update(stage="chain", window_applied=True)
    inp = steps(case.tmp / "steps.root", keys((1, 2**20)))
    before = R.snapshot_inputs({"input": inp})
    root(case.out)
    with pytest.raises(ValueError, match="packed event-key capacity"):
        R.write_receipt(case.out, case.metadata, before)


def test_partial_stock_window_never_admitted(case):
    root(case.inp, stock=True, mutate=lambda d: d["events"].pop("pbtime"))
    evidence = validate(case, publish(case))
    assert evidence["population"]["status"] == "verified_exact_stock_event_universe"
    assert not evidence["conditional_response_supported"]


def test_empty_and_missing_event_tree_rejected(tmp_path):
    path = tmp_path / "no-events.root"
    with uproot.recreate(path) as f:
        tree(f, "crvdump/digis", keys())
    with pytest.raises(ValueError, match="missing crvdump/events"):
        R.event_identity(path)
    with uproot.recreate(path) as f:
        f.mktree("crvdump/events", {k: np.uint32 for k in R.KEYS})
    with pytest.raises(ValueError, match="empty event universe"):
        R.event_identity(path)


@pytest.mark.parametrize("name", ["reco/cli.py", "reco/coinc_jax.py", "readout.py"])
def test_bucketed_and_readout_source_drift_checked(case, monkeypatch, name):
    before = R.snapshot_inputs({"input": case.inp})
    assert name in before._sources
    root(case.out)
    sources = R._sources
    def changed():
        records = sources()
        records[name] = {**records[name], "sha256": "0" * 64}
        return records
    monkeypatch.setattr(R, "_sources", changed)
    with pytest.raises(ValueError, match="source drift"):
        R.write_receipt(case.out, case.metadata, before)


@pytest.mark.parametrize("key", ["diagnostic_override", "geometry_readout", "response_execution"])
def test_diagnostic_markers_refused(case, key):
    case.metadata[key] = {}
    with pytest.raises(ValueError, match="overrides"):
        publish(case)
