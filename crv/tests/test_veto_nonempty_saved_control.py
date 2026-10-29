"""Pure saved-only audit tests using synthetic native rows, not physical evidence.

No recovery, finder, ROOT job, or remote input is executed. Geometry is deliberately
minimal; these tests establish contracts, not detector or timing validation.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import nonempty_saved_control as N


def forbidden(*args, **kwargs):
    pytest.fail("saved-only unit test attempted finder/recovery or premature decode")


@pytest.fixture(autouse=True)
def no_finder(monkeypatch):
    monkeypatch.setattr(N.R, "recover", forbidden)
    monkeypatch.setattr(N.R.C, "find_event", forbidden)


@pytest.fixture
def native():
    frames = pd.DataFrame({
        "input_sha256": ["a" * 64] * 3,
        "run": [7] * 3, "subrun": [2] * 3, "event": [10, 11, 12],
        "frame_id": [40, 41, 42],
    })
    events = ak.Array({
        "run": [7] * 3, "subrun": [2] * 3, "event": [10, 11, 12],
        "pbtime": [1.1, -4.2, 0.], "npulse": [4, 2, 0], "ncluster": [2, 1, 0],
    })
    # Event interleaving is allowed, but within-event order must never be sorted.
    pulse = ak.Array({
        "ev": (2 << 20) + np.array([11, 10, 10, 11, 10, 10]),
        "chan": [12, 8, 1, 16, 4, 9],
        "time": [-3.25, 8.25, 1.25, -2.5, 6.75, 8.5],
        "pe": [11.1, 10.123456789, 13.2, 12.3, 14.4, 15.5],
        "seq": [0, 2, 0, 1, 1, 3],
    })
    order = np.array([1, 2, 4, 5, 0, 3])
    columns = {}
    for name, dtype in N.R.PULSE_TYPES.items():
        if name == "wfidx":
            columns[name] = ak.unflatten(np.array([], dtype=np.int32), np.zeros(6, dtype=np.int64))
        else:
            columns[name] = np.zeros(6, dtype=dtype)
    chan = ak.to_numpy(pulse.chan)[order]
    columns.update(run=np.full(6, 7), subrun=np.full(6, 2), event=np.array([10] * 4 + [11] * 2),
                   idx=np.array([0, 1, 2, 3, 0, 1]), bar=chan // 4, sipm=chan % 4,
                   seq=ak.to_numpy(pulse.seq)[order], time=ak.to_numpy(pulse.time)[order],
                   pe=ak.to_numpy(pulse.pe)[order].astype(np.float32))
    stock = ak.Array(columns)
    return SimpleNamespace(frames=frames, events=events, pulse=pulse, stock=stock,
                           adapter=ak.copy(stock), status=np.zeros(5120, dtype=np.int64),
                           geom=SimpleNamespace(layer=np.array([0, 1, 2, 0, 1]),
                                                sector=np.array([0, 0, 0, 1, 1]),
                                                sector_type=np.array([1, 2])))


def bridge(n):
    return N.pulse_bridge(n.pulse, n.stock, n.adapter, n.events, n.frames, n.status)


def clusters():
    rows = []
    for event, idx, sector, members, layers in (
        (10, 0, 1, [0, 2], [2, 1]),
        (10, 1, 1, [3, 1], [0, 2]),
        (11, 0, 2, [1, 0], [1, 0]),
    ):
        row = {name: (False if dtype == "bool" else 0 if "int" in dtype else 0.)
               for name, dtype in N.R.CLUSTER_TYPES.items()}
        row.update(run=7, subrun=2, event=event, idx=idx, sectortype=sector,
                   nLayers=len(layers), sidehits0=1, sidehits1=1, poscalc=True,
                   start=-2., end=10., avgtime=3., pe=25.,
                   pulses=members, layers=layers)
        rows.append(row)
    return ak.Array(rows)


def products(n, array=None):
    return N.cluster_tables(clusters() if array is None else array, n.frames, bridge(n), n.geom)


def replace_at(array, field, index, value):
    values = ak.concatenate([array[field][:index], ak.Array([value]), array[field][index + 1:]])
    return ak.with_field(array, values, field)


def test_bridge_preserves_native_order_float64_time_and_explicit_float32_pe(native):
    p = bridge(native)
    assert p.chan.tolist() == [8, 1, 4, 9, 12, 16]
    assert p.idx.tolist() == [0, 1, 2, 3, 0, 1]
    assert p.frame_id.tolist() == [40] * 4 + [41] * 2
    assert p.time.tolist() == [8.25, 1.25, 6.75, 8.5, -3.25, -2.5]
    assert p.pe.iloc[0] == 10.123456789
    assert p.stock_pe.iloc[0] == float(np.float32(10.123456789))
    assert p.stock_pe.iloc[0] != p.pe.iloc[0]
    assert set(p.input_sha256) == {"a" * 64}
    assert not {"wf", "wfidx", "starttdc", "time_relative"} & set(p)
    assert 42 in set(native.frames.frame_id) and 42 not in set(p.frame_id)


@pytest.mark.parametrize("change", ["missing_seq", "invented_waveform"])
def test_bridge_requires_exact_legacy_schema(native, change):
    native.pulse = (ak.without_field(native.pulse, "seq") if change == "missing_seq"
                    else ak.with_field(native.pulse, np.zeros(6, dtype=int), "wf"))
    with pytest.raises(ValueError, match="legacy pulse schema"):
        bridge(native)


@pytest.mark.parametrize("field", ["idx", "event", "time", "pe"])
def test_bridge_rejects_ordered_identity_and_scalar_changes(native, field):
    target = "adapter" if field in ("time", "pe") else "stock"
    array = getattr(native, target)
    if field == "pe":
        # Original float64 is NOT an acceptable replacement for serialized PE.
        value = 10.123456789
    elif field == "time":
        value = np.nextafter(float(array.time[0]), np.inf)
    else:
        value = int(array[field][0]) + 1
    setattr(native, target, replace_at(array, field, 0, value))
    with pytest.raises(ValueError, match="ordered pulse bridge changed: " + field):
        bridge(native)


def test_bridge_rejects_nonempty_waveform_placeholders_in_either_collection(native):
    for target in ("stock", "adapter"):
        saved = getattr(native, target)
        setattr(native, target, replace_at(saved, "wfidx", 0, [0]))
        with pytest.raises(ValueError, match="waveform-placeholder"):
            bridge(native)
        setattr(native, target, saved)


def test_bridge_rejects_legacy_channel_and_event_outside_domain(native):
    original = native.pulse
    for field, value, message in (("chan", 5120, "pulse chan"),
                                  ("ev", (2 << 20) + 99, "outside all-event table")):
        native.pulse = replace_at(original, field, 0, value)
        with pytest.raises(ValueError, match=message):
            bridge(native)


def test_bridge_checks_collection_length_and_quiet_event_census(native):
    native.adapter = native.adapter[:-1]
    with pytest.raises(ValueError, match="collection count"):
        bridge(native)
    native.adapter = ak.copy(native.stock)
    native.events = replace_at(native.events, "npulse", 2, 1)
    with pytest.raises(ValueError, match="pulse/event count"):
        bridge(native)


def test_bridge_rejects_pe_not_representable_as_float32(native):
    native.pulse = replace_at(native.pulse, "pe", 0, np.finfo(np.float64).max)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="nonfinite serialized PE"):
        bridge(native)


def test_bridge_rejects_negative_pe(native):
    native.pulse = replace_at(native.pulse, "pe", 0, -1.)
    with pytest.raises(ValueError, match="pulse bridge domain"):
        bridge(native)


def test_cluster_tables_join_native_members_by_frame_and_index(native):
    c, m = products(native)
    assert c.frame_id.tolist() == [40, 40, 41]
    assert c.nLayers.tolist() == [2, 2, 2]
    assert m.pulse_idx.tolist() == [0, 2, 3, 1, 1, 0]
    assert m.member_order.tolist() == [0, 1, 0, 1, 0, 1]
    assert m.chan.tolist() == [8, 4, 9, 1, 16, 12]
    assert m.layer.tolist() == [2, 1, 2, 0, 1, 0]
    assert m.sector_type.tolist() == [1, 1, 1, 1, 2, 2]
    assert m.cluster_idx.tolist() == [0, 0, 1, 1, 0, 0]
    assert set(m.input_sha256) == {"a" * 64}


def test_cluster_layer_count_can_be_derived_but_not_contradicted(native):
    c, _ = products(native, ak.without_field(clusters(), "nLayers"))
    assert c.nLayers.tolist() == [2, 2, 2]
    with pytest.raises(ValueError, match="saved layer count"):
        products(native, replace_at(clusters(), "nLayers", 0, 3))


@pytest.mark.parametrize("change,match", [
    ("duplicate", "duplicate cluster identity"),
    ("gap", "noncontiguous cluster indices"),
    ("outside", "outside source-qualified census"),
])
def test_cluster_rejects_invalid_native_identity(native, change, match):
    array = clusters()
    if change == "duplicate":
        array = replace_at(array, "idx", 1, 0)
    elif change == "gap":
        array = replace_at(array, "idx", 1, 2)
    else:
        array = replace_at(array, "event", 0, 99)
    with pytest.raises(ValueError, match=match):
        products(native, array)


@pytest.mark.parametrize("members,match", [
    ([99, 2], "outside pulse collection"),
    ([0, 0], "duplicate native pulse membership"),
    ([-1, 2], "invalid native pulse target"),
    ([], "empty or oversized native membership"),
])
def test_cluster_rejects_invalid_member_targets(native, members, match):
    with pytest.raises(ValueError, match=match):
        products(native, replace_at(clusters(), "pulses", 0, members))


@pytest.mark.parametrize("field,value,match", [
    ("layers", [0, 1], "native layer set mismatch"),
    ("layers", [2.5, 1.0], "invalid integer: layer"),
    ("layers", [4, 1], "declared layer outside geometry domain"),
    ("sectortype", 2, "native member sector mismatch"),
])
def test_cluster_requires_exact_geometry_layer_and_sector_joins(native, field, value, match):
    with pytest.raises(ValueError, match=match):
        products(native, replace_at(clusters(), field, 0, value))


def test_legacy_import_retains_each_status_bit_without_new_filter(native):
    for bit in (1, 2, 4, 8):
        native.status[8] = bit
        _, m = products(native)
        assert m.loc[m.chan.eq(8), "channel_status"].item() == bit
        assert len(m) == 6
    native.status[8] = 2**16
    with pytest.raises(ValueError, match="invalid native member channel status"):
        products(native)


def test_cluster_rejects_invalid_scalar_domains_and_nonboolean_position_flag(native):
    for field, value, message in (("start", 11., "invalid native scalar domain"),
                                  ("pe", -1., "invalid native scalar domain"),
                                  ("poscalc", 1, "invalid position-method flag")):
        array = clusters()
        if field == "poscalc":
            array = ak.with_field(array, np.ones(len(array), dtype=np.int64), field)
        else:
            array = replace_at(array, field, 0, value)
        with pytest.raises(ValueError, match=message):
            products(native, array)


def test_compare_pairs_exact_member_sets_despite_cluster_and_member_permutations(native):
    reference = products(native)
    array = clusters()[[1, 0, 2]]
    array = ak.with_field(array, np.array([0, 1, 0]), "idx")
    array = ak.with_field(array, array.pulses[:, ::-1], "pulses")
    result = N.compare(reference, products(native, array))
    assert result["exact_member_set_matches"] == 3
    assert result["stock_links"] == result["fast_links"] == 6
    assert all(p == {"exact_mismatches": 0, "max_abs_difference": 0.}
               for p in result["properties"].values())


def test_compare_reports_ulp_scalar_and_boolean_discrepancies_without_tolerance(native):
    c, m = products(native)
    candidate = c.copy()
    candidate.loc[0, "avgtime"] = np.nextafter(c.loc[0, "avgtime"], np.inf)
    candidate.loc[1, "poscalc"] = False
    candidate.loc[2, "sidehits0"] += 1
    result = N.compare((c, m), (candidate, m.copy()))
    props = result["properties"]
    delta = candidate.loc[0, "avgtime"] - c.loc[0, "avgtime"]
    assert 0 < delta < 1e-9
    assert props["avgtime"] == {"exact_mismatches": 1, "max_abs_difference": delta}
    assert props["poscalc"] == {"exact_mismatches": 1, "max_abs_difference": 1.}
    assert props["sidehits0"] == {"exact_mismatches": 1, "max_abs_difference": 1.}
    assert result["exact_member_set_matches"] == 3
    assert "tolerance" not in result and "passed" not in result


def test_compare_rejects_different_native_member_sets(native):
    c, m = products(native)
    candidate = m.copy()
    candidate.loc[0, "pulse_idx"] = 3
    with pytest.raises(ValueError, match="saved native member sets differ"):
        N.compare((c, m), (c.copy(), candidate))


def selection_inputs(native):
    c, _ = products(native)
    coverage = pd.DataFrame({"frame_id": [41, 40], "gate_pass": [True, True],
                             "n_veto_eligible": [1, 1], "track_grid_coverage": [.2, .1]})
    control = {"affected_clusters": [
        {"run": 7, "subrun": 2, "event": 11, "added_count": 1},
        {"run": 7, "subrun": 2, "event": 10, "added_count": 1},
    ]}
    return c, coverage, control


def test_choose_earliest_roles_deduplicates_and_keeps_quiet_census(native):
    c, coverage, control = selection_inputs(native)
    selected, roles = N.choose(native.frames.iloc[::-1], c.iloc[::-1], coverage, control)
    assert roles == {"nonempty": 40, "opposite_side_attachment": 40,
                     "positive_passing_coverage": 40, "zero_coincidence": 42}
    pd.testing.assert_frame_equal(selected, native.frames.iloc[[2, 0]])
    assert 42 not in set(c.frame_id) and 42 in set(selected.frame_id)
    selected.loc[42 == selected.frame_id, "event"] = 999
    assert native.frames.loc[native.frames.frame_id.eq(42), "event"].item() == 12


@pytest.mark.parametrize("missing", ["attachment", "positive", "quiet"])
def test_choose_requires_every_control_role(native, missing):
    c, coverage, control = selection_inputs(native)
    frames = native.frames
    if missing == "attachment":
        for row in control["affected_clusters"]:
            row["added_count"] = 0
    elif missing == "positive":
        coverage.loc[0, "gate_pass"] = False
        coverage.loc[1, "n_veto_eligible"] = 0
    else:
        frames = frames.loc[frames.frame_id.ne(42)]
    with pytest.raises(ValueError, match="required control role missing"):
        N.choose(frames, c, coverage, control)


def fake_parquet(monkeypatch, *, rows=6, columns=5, groups=1, sizes=(100,)):
    metadata = SimpleNamespace(num_rows=rows, num_columns=columns, num_row_groups=groups,
                               row_group=lambda i: SimpleNamespace(total_byte_size=sizes[i]))
    monkeypatch.setattr(N.SR, "_regular", lambda path: 10)
    monkeypatch.setattr(N.pq, "ParquetFile", lambda path: nullcontext(SimpleNamespace(metadata=metadata)))
    monkeypatch.setattr(N.ak, "from_parquet", forbidden)


def test_parquet_admits_bounded_metadata_then_decodes_once(monkeypatch):
    fake_parquet(monkeypatch)
    result = ak.Array({"x": [1, 2]})
    calls = []
    monkeypatch.setattr(N.ak, "from_parquet", lambda path: calls.append(path) or result)
    assert N.parquet("synthetic.parquet", 6) is result
    assert calls == ["synthetic.parquet"]


def test_parquet_file_cap_precedes_metadata_open(monkeypatch):
    monkeypatch.setattr(N.SR, "_regular", lambda path: N.READ_CAP + 1)
    monkeypatch.setattr(N.pq, "ParquetFile", forbidden)
    monkeypatch.setattr(N.ak, "from_parquet", forbidden)
    with pytest.raises(ValueError, match="Parquet file cap"):
        N.parquet("synthetic.parquet", 6)


def test_parquet_row_and_schema_caps_precede_decode(monkeypatch):
    for change in ({"rows": 7}, {"columns": 65}, {"groups": 4097}):
        fake_parquet(monkeypatch, **change)
        with pytest.raises(ValueError, match="row/schema cap.*before decode"):
            N.parquet("synthetic.parquet", 6)


def test_parquet_uncompressed_individual_and_total_caps_precede_decode(monkeypatch):
    for sizes in ((N.READ_CAP + 1,), (-1,), (N.READ_CAP // 2 + 1,) * 2):
        fake_parquet(monkeypatch, groups=len(sizes), sizes=sizes)
        with pytest.raises(ValueError, match="uncompressed cap.*before decode"):
            N.parquet("synthetic.parquet", 6)


def test_tree_admits_exact_caps_and_passes_requested_columns():
    calls = []
    result = ak.Array({"idx": [0]})
    t = SimpleNamespace(num_entries=6, uncompressed_bytes=N.READ_CAP,
                        object_path="/crvdump/pulses;1",
                        arrays=lambda columns, library: calls.append((columns, library)) or result)
    array, metadata = N.tree({"crvdump/pulses": t}, "crvdump/pulses", 6, ["idx"])
    assert array is result
    assert calls == [(["idx"], "ak")]
    assert metadata == {"key": t.object_path, "entries": 6, "uncompressed_bytes": N.READ_CAP}


@pytest.mark.parametrize("field,value", [("num_entries", 7), ("uncompressed_bytes", N.READ_CAP + 1)])
def test_tree_caps_reject_before_arrays(field, value):
    t = SimpleNamespace(num_entries=6, uncompressed_bytes=100, arrays=forbidden)
    setattr(t, field, value)
    with pytest.raises(ValueError, match="ROOT tree cap exceeded before decode"):
        N.tree({"crvdump/pulses": t}, "crvdump/pulses", 6)


def test_calibration_numeric_and_integer_validators_reject_invalid_status():
    with pytest.raises(ValueError, match="invalid numeric field: status"):
        N.flat(ak.Array({"channel": [0], "status": [np.nan]}), ["channel", "status"])
    for status in (-1, .5):
        table = N.flat(ak.Array({"channel": [0], "status": [status]}), ["channel", "status"])
        with pytest.raises(ValueError, match="invalid integer: status"):
            N.integers(table, ["channel", "status"])


def test_bridge_rejects_signed_zero_time_change(native):
    native.pulse = replace_at(native.pulse, "time", 1, -0.)
    native.stock = replace_at(native.stock, "time", 0, 0.)
    with pytest.raises(ValueError, match="floating bits changed: time"):
        bridge(native)


def test_bridge_requires_actual_native_pe_float32_storage(native):
    native.stock = ak.with_field(native.stock, ak.values_astype(native.stock.pe, np.float64), "pe")
    with pytest.raises(ValueError, match="storage dtype changed: pe"):
        bridge(native)


def test_stock_int32_boolean_storage_is_validated_before_normalization(native):
    original = clusters()
    encoded = ak.with_field(original, ak.values_astype(original.poscalc, np.int32), "poscalc")
    result = N.compare(products(native, original), products(native, encoded))
    assert result["properties"]["poscalc"]["exact_mismatches"] == 0
    encoded = ak.with_field(encoded, np.full(len(encoded), 2, dtype=np.int32), "poscalc")
    with pytest.raises(ValueError, match="invalid position-method flag"):
        products(native, encoded)
