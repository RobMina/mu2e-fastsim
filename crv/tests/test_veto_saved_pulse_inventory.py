"""Pure saved-identity admission: no finder, fitting, response, or regeneration."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from crvfastsim.reco import coinc as C, pulsefit, waveform
from crvfastsim.response import sipm
from crvfastsim.veto import bounded_response as B, frame_registry as F
from crvfastsim.veto import pilot_coincidence as P, replay_recover as R
from crvfastsim.veto import replay_sipm_control as SC, replay_waveform_control as WC
from crvfastsim.veto import saved_pulse_inventory as I


def forbidden(*args, **kwargs):
    pytest.fail("inventory must not execute finder, fit, optics, or response")


@pytest.fixture(autouse=True)
def no_physics(monkeypatch):
    # Configuration and conservative geometry counting remain permitted.
    for module, names in (
        (P, ("run",)), (R, ("recover", "prepare_pulses")), (C, ("find_event",)),
        (B, ("run_arm",)), (B.Q, ("run_arm",)),
        (I.Q, ("run",)), (I.D, ("run",)),
        (SC, ("optical_candidates", "thermal_candidates", "simulate_candidates")),
        (WC, ("synthesize", "reconstruct")), (sipm, ("simulate",)),
        (waveform, ("synthesize",)), (pulsefit, ("reconstruct", "reconstruct_full")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.fixture
def domain():
    layer = np.arange(4, dtype=np.int64)
    pos = np.column_stack((np.zeros(4), layer * 10., np.zeros(4)))
    geom = C.SectorGeom(
        np.zeros(4, dtype=np.int64), layer, np.zeros(4, dtype=np.int64),
        pos[:, 0], pos[:, 1], pos, np.array([2]), np.array([1]), ["T1"],
        np.array([-700.]), np.array([700.]), np.array([True]), np.array([True]),
    )
    registry = F.freeze(pd.DataFrame({
        "input_sha256": ["a" * 64] * 2, "run": [1430, 1430], "subrun": [0, 0],
        "event": [7, 2**20 + 7], "pbtime": [-224.46, -214.46], "intensity": [1., 1.],
    }))
    pulses = registry.iloc[np.zeros(4, dtype=np.int64)][I.KEYS].reset_index(drop=True)
    pulses = pulses.assign(
        wf=np.arange(10, 14, dtype=np.int64), chan=np.arange(4, dtype=np.int64) * 4,
        starttdc=np.full(4, 90, dtype=np.int64), seq=np.zeros(4, dtype=np.int64),
        time=np.array([500., 501., 502., 503.]),
        time_relative=np.array([500., 501., 502., 503.]),
        pe=np.full(4, 10.123456789), failed=False, zerondf=False, channel_ignored=False,
    )
    pulses["time_reco"] = pulses.time_relative - float(np.float32(registry.pbtime.iloc[0]))
    waveforms = pulses[[*I.KEYS, "wf", "chan", "starttdc"]].copy()
    calibration = pd.DataFrame({"input_sha256": ["a" * 64] * 16,
                                "chan": np.arange(16, dtype=np.int64),
                                "status": np.zeros(16, dtype=np.int64)})
    return SimpleNamespace(p=pulses, w=waveforms, s=registry.copy(), r=registry,
                           c=calibration, g=geom)


def inspect(d):
    return I.inspect(d.p, d.w, d.s, d.r, d.c, d.g)


def test_nonempty_identity_inventory_includes_quiet_frame_without_mutation(domain):
    d = domain
    before = [t.copy(deep=True) for t in (d.p, d.w, d.s, d.r, d.c)]
    out = inspect(d)
    assert out["pulse_rows"] == out["saved_waveforms"] == 4
    assert out["candidate_tuple_upper_bound"] == 4
    assert out["saved_identity_and_order_valid"] is True
    assert out["forward_time_bits_exact"] is True
    assert out["finder_not_run"] is True
    assert out["nonempty_membership_established"] is False
    frames = pd.DataFrame(out["frames"])
    pd.testing.assert_frame_equal(frames[I.KEYS], d.s[I.KEYS])
    np.testing.assert_array_equal(frames.saved_pulses, [4, 0])
    np.testing.assert_array_equal(frames.ignored_pulses, [0, 0])
    np.testing.assert_array_equal(frames.finder_input_pulses, [4, 0])
    assert out["occupancy"] == [{"frame_id": 0, "sector_type": 1, "side": 0,
                                  "pulses": 4, "distinct_layers": 4}]
    for actual, expected in zip((d.p, d.w, d.s, d.r, d.c), before):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_typed_empty_tables_keep_complete_frame_census(domain):
    domain.p, domain.w = domain.p.iloc[:0], domain.w.iloc[:0]
    out = inspect(domain)
    assert out["pulse_rows"] == out["saved_waveforms"] == out["candidate_tuple_upper_bound"] == 0
    assert out["occupancy"] == []
    frames = pd.DataFrame(out["frames"])
    assert len(frames) == 2
    assert frames[["saved_pulses", "ignored_pulses", "finder_input_pulses"]].eq(0).all().all()


def test_secondary_and_fit_flagged_rows_retained_only_ignored_excluded(domain):
    d = domain
    secondary = d.p.iloc[:1].assign(seq=np.int64(1), failed=True, zerondf=True)
    d.p = pd.concat([d.p.iloc[:1], secondary, d.p.iloc[1:]], ignore_index=True)
    d.p.loc[d.p.chan.eq(12), "channel_ignored"] = True
    d.c.loc[d.c.chan.eq(12), "status"] = 2
    out = inspect(d)
    assert out["pulse_rows"] == 5 and out["saved_waveforms"] == 4
    assert out["frames"][0]["saved_pulses"] == 5
    assert out["frames"][0]["ignored_pulses"] == 1
    assert out["frames"][0]["finder_input_pulses"] == 4
    assert out["candidate_tuple_upper_bound"] == 2
    assert out["occupancy"] == [{"frame_id": 0, "sector_type": 1, "side": 0,
                                  "pulses": 4, "distinct_layers": 3}]


@pytest.mark.parametrize("table,column", [("p", "time"), ("p", "failed"), ("w", "wf")])
def test_missing_actual_saved_fields(domain, table, column):
    setattr(domain, table, getattr(domain, table).drop(columns=column))
    with pytest.raises(ValueError, match="missing"):
        inspect(domain)


@pytest.mark.parametrize("case", ["float32_time", "nan_time", "negative_pe", "integer_failed", "null_zerondf"])
def test_invalid_numeric_or_boolean_payload(domain, case):
    if case == "float32_time":
        domain.p["time"] = domain.p.time.astype(np.float32)
    elif case == "nan_time":
        domain.p.loc[0, "time"] = np.nan
    elif case == "negative_pe":
        domain.p.loc[0, "pe"] = -1.
    elif case == "integer_failed":
        domain.p["failed"] = 0
    else:
        domain.p["zerondf"] = pd.array([False, False, False, pd.NA], dtype="boolean")
    with pytest.raises(ValueError, match="float64|negative|boolean"):
        inspect(domain)


@pytest.mark.parametrize("column,value", [("seq", -1), ("starttdc", 90.), ("wf", True)])
def test_strict_integer_identity(domain, column, value):
    domain.p[column] = value
    with pytest.raises(ValueError, match="integer"):
        inspect(domain)


def test_source_identity_cannot_be_inferred_from_matching_event_or_frame(domain):
    domain.p["input_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="census"):
        inspect(domain)


@pytest.mark.parametrize("case", ["one_ulp", "negative_zero"])
def test_forward_time_requires_exact_float_bits(domain, case):
    if case == "one_ulp":
        domain.p.loc[0, "time_reco"] = np.nextafter(domain.p.time_reco.iloc[0], np.inf)
    else:
        rounded_pb = float(np.float32(domain.r.pbtime.iloc[0]))
        domain.p["time_relative"] = rounded_pb
        domain.p["time_reco"] = 0.
        domain.p.loc[0, "time_reco"] = -0.
        assert np.array_equal(domain.p.time_reco, domain.p.time_relative - rounded_pb)
    with pytest.raises(ValueError, match="time|bunch"):
        inspect(domain)


@pytest.mark.parametrize("case", ["stored_permutation", "native_structural_disagreement"])
def test_order_disagreement_is_rejected_not_repaired(domain, case):
    if case == "stored_permutation":
        domain.p = domain.p.iloc[::-1].reset_index(drop=True)
    else:
        domain.p["wf"] = domain.p.wf.to_numpy()[::-1]
        domain.w["wf"] = domain.w.wf.to_numpy()[::-1]
    with pytest.raises(ValueError, match="order"):
        inspect(domain)


@pytest.mark.parametrize("case", ["missing", "duplicate_wf", "duplicate_structure"])
def test_waveform_link_must_be_complete_and_bijective(domain, case):
    if case == "missing":
        domain.w = domain.w.iloc[1:]
    elif case == "duplicate_wf":
        domain.w.loc[1, "wf"] = domain.w.wf.iloc[0]
    else:
        domain.w.loc[1, ["chan", "starttdc"]] = domain.w.loc[0, ["chan", "starttdc"]].to_numpy()
    with pytest.raises(ValueError, match="waveform"):
        inspect(domain)


@pytest.mark.parametrize("case", ["missing_channel", "duplicate_channel", "float_status", "ignored_mismatch", "dead_channel"])
def test_complete_authoritative_calibration_and_channel_policy(domain, case):
    if case == "missing_channel":
        # Even an unoccupied channel is required in the source calibration.
        domain.c = domain.c.iloc[:-1]
    elif case == "duplicate_channel":
        domain.c = pd.concat([domain.c, domain.c.iloc[:1]], ignore_index=True)
    elif case == "float_status":
        domain.c["status"] = domain.c.status.astype(float)
    elif case == "ignored_mismatch":
        domain.c.loc[0, "status"] = 2
    else:
        domain.c.loc[0, "status"] = 1
    with pytest.raises(ValueError, match="calibration|policy"):
        inspect(domain)


def test_qualify_legacy_rows_uses_full_census_and_preserves_stored_order(domain):
    legacy = pd.DataFrame({"event": [2**20 + 7, 7, 7], "chan": [12, 4, 0],
                           "time": np.array([1., 2., 3.])})
    before = legacy.copy(deep=True)
    out = I.qualify(legacy, domain.s)
    pd.testing.assert_frame_equal(out[list(legacy)], legacy, check_exact=True)
    np.testing.assert_array_equal(out.frame_id, [1, 0, 0])
    assert out.input_sha256.eq("a" * 64).all()
    assert out.run.eq(1430).all() and out.subrun.eq(0).all()
    pd.testing.assert_frame_equal(legacy, before, check_exact=True)


@pytest.mark.parametrize("case", ["already_qualified", "ambiguous_census", "unknown_event"])
def test_qualify_never_invents_or_guesses_identity(domain, case):
    legacy = domain.p[["event", "chan", "time"]].copy()
    if case == "already_qualified":
        legacy["frame_id"] = 0
    elif case == "ambiguous_census":
        domain.s.loc[1, "event"] = domain.s.event.iloc[0]
    else:
        legacy.loc[0, "event"] = 999
    with pytest.raises(ValueError, match="qualified|ambiguous|census"):
        I.qualify(legacy, domain.s)


@pytest.mark.parametrize("case", ["file", "rows", "columns", "row_groups", "group_bytes", "total_bytes"])
def test_parquet_caps_fail_before_payload_decode(tmp_path, monkeypatch, case):
    path = tmp_path / "saved.parquet"
    path.write_bytes(b"metadata is mocked; payload must never be decoded")
    size = I.FILE_CAP + 1 if case == "file" else 64
    monkeypatch.setattr(I.SR, "_regular", lambda p: size)
    byte_cap = 128 * 1024**2
    sizes = [1]
    if case == "group_bytes":
        sizes = [byte_cap + 1]
    elif case == "total_bytes":
        sizes = [byte_cap // 2 + 1, byte_cap // 2]
    meta = SimpleNamespace(num_rows=5 if case == "rows" else 4,
                           num_columns=65 if case == "columns" else 4,
                           num_row_groups=4097 if case == "row_groups" else len(sizes),
                           row_group=lambda i: SimpleNamespace(total_byte_size=sizes[i]))
    reader = Mock()
    reader.__enter__ = Mock(return_value=SimpleNamespace(metadata=meta))
    reader.__exit__ = Mock(return_value=False)
    open_parquet = Mock(return_value=reader)
    monkeypatch.setattr(I.pq, "ParquetFile", open_parquet)
    decode = Mock(side_effect=forbidden)
    monkeypatch.setattr(I.pd, "read_parquet", decode)
    with pytest.raises(ValueError, match="cap"):
        I.table(path, 4)
    decode.assert_not_called()
    if case == "file":
        open_parquet.assert_not_called()
    else:
        open_parquet.assert_called_once_with(path)


def test_small_parquet_at_exact_row_cap_decodes_once(tmp_path, monkeypatch):
    path = tmp_path / "saved.parquet"
    expected = pd.DataFrame({"time": np.array([1., 2.]), "failed": [False, True]})
    expected.to_parquet(path, index=False)
    decode = Mock(wraps=pd.read_parquet)
    monkeypatch.setattr(I.pd, "read_parquet", decode)
    pd.testing.assert_frame_equal(I.table(path, 2), expected, check_exact=True)
    decode.assert_called_once_with(path)


def test_total_pulse_cap_precedes_schema_and_work(domain, monkeypatch):
    domain.p = pd.DataFrame(index=np.arange(2049))
    monkeypatch.setattr(P, "combination_work", forbidden)
    with pytest.raises(ValueError, match="cap"):
        inspect(domain)


def test_per_frame_cap_precedes_combination_work(domain, monkeypatch):
    domain.p = domain.p.iloc[np.zeros(513, dtype=np.int64)].reset_index(drop=True)
    domain.p["seq"] = np.arange(513, dtype=np.int64)
    monkeypatch.setattr(P, "combination_work", forbidden)
    with pytest.raises(ValueError, match="per-frame pulse cap"):
        inspect(domain)
