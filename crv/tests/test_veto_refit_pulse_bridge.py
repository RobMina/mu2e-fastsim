"""Synthetic pure saved-refit admission; physics, filesystem decoding and verifiers are forbidden."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.reco import coinc, pulsefit, waveform
from crvfastsim.response import sipm
from crvfastsim.veto import refit_pulse_bridge as R

B, I, F, P = R.B, R.I, R.F, R.P


def forbidden(*args, **kwargs):
    pytest.fail("pure admission attempted physics, raw access, saved decoding or a verifier rerun")


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    for module, names in (
        (P, ("run",)), (coinc, ("find_event",)), (pulsefit, ("reconstruct", "reconstruct_full")),
        (waveform, ("synthesize",)), (sipm, ("simulate",)), (B.M.C, ("recover", "prepare_pulses")),
        (B, ("run",)), (I, ("run",)), (I.Q, ("run", "admission")),
        (I.Q.R, ("run_arm",)), (I.Q.R.Q, ("run_arm",)), (I.D, ("run", "run_arm")),
        (I.Q.M.K, ("make_photons",)), (I.Q.M.P, ("read_input",)),
        (I.SR, ("verify",)), (I.V, ("verify",)), (uproot, ("open",)), (pd, ("read_parquet",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


@pytest.fixture
def domain():
    sha = "b" * 64
    geom = SimpleNamespace(sector=np.zeros(4, dtype=np.int64), layer=np.arange(4, dtype=np.int64),
                           sector_type=np.array([1]), sector_names=["T1"])
    registry = F.freeze(pd.DataFrame({
        "input_sha256": ["a" * 64, sha, sha, sha], "run": [1430] * 4, "subrun": [23] * 4,
        "event": [7, 1, 7, 9], "pbtime": [-224.46] * 4, "intensity": [1.] * 4,
    }))
    events = pd.DataFrame({"event": [7, 9], "pbtime": float(np.float32(-224.46)), "pbtime_raw": -224.46})
    calibration = pd.DataFrame({"input_sha256": sha, "chan": np.arange(16, dtype=np.int64),
                                "pedestal": 100., "pulsearea": 10., "pulseheight": 5., "timeoffset": 2.375,
                                "status": np.int64(0), "roc": np.int64(1), "feb": np.int64(2)})
    conditions = calibration.iloc[[0, 4, 8, 12, 0, 15]].drop(columns="input_sha256").reset_index(drop=True)
    conditions["event"] = [7, 7, 7, 7, 9, 9]
    conditions["w"] = -float(np.float32(-224.46))
    p = pd.DataFrame({"event": [7] * 5 + [9], "wf": [10, 10, 11, 12, 13, 14],
                      "chan": [0, 0, 4, 8, 12, 0], "seq": [0, 1, 0, 0, 0, 0],
                      "time": np.arange(6, dtype=float) + 500.125, "pe": np.arange(6, dtype=float) + 10.123456789,
                      "failed": [False, True, False, False, False, False],
                      "zerondf": [True, False, False, False, False, False], "channel_ignored": False,
                      "height": np.arange(6, dtype=float) + 3.})
    p["time_relative"] = p.time + 2.375
    p["time_reco"] = p.time_relative - float(np.float32(-224.46))
    w = p[["event", "wf", "chan"]].drop_duplicates().reset_index(drop=True).assign(starttdc=np.int64(90))
    stock = pd.DataFrame({"run": np.uint32(1430), "subrun": np.uint32(23), "event": p.event.astype(np.uint32),
                          "idx": np.array([4, 1, 3, 0, 2, 0], np.int32), "bar": (p.chan // 4).astype(np.int32),
                          "sipm": (p.chan % 4).astype(np.int32), "seq": p.seq.astype(np.int32),
                          "time": p.time_reco - 0.125, "pe": (p.pe - 0.25).astype(np.float32),
                          "flags": np.array([0, 1, 8, 0, 0, 0], np.int32)})
    comparison = stock.rename(columns={"pe": "pe_stock"}).copy()
    comparison["chan"], comparison["wf"] = p.chan, p.wf
    comparison["pe_fast"], comparison["time_reco"] = p.pe, p.time_reco
    comparison["failed"], comparison["zerondf"] = p.failed, p.zerondf
    comparison["_merge"] = pd.Categorical(["both"] * len(p), categories=["left_only", "right_only", "both"])
    comparison["delta_time"] = comparison.time_reco - comparison.time
    comparison["delta_pe"] = comparison.pe_fast - comparison.pe_stock
    return SimpleNamespace(p=p, w=w, conditions=conditions, events=events, stock=stock, comparison=comparison,
                           registry=registry, calibration=calibration, geom=geom, input_sha=sha)


def prepare(d):
    return R.prepare(**vars(d))


def test_complete_bijection_preserves_all_columns_indices_full_registry_and_inputs(domain):
    d = domain
    d.comparison = d.comparison.iloc[::-1].reset_index(drop=True)
    d.stock = d.stock.iloc[::-1].reset_index(drop=True)
    d.w = d.w.iloc[::-1]
    d.p.index = [70, 20, 90, 10, 80, 40]
    before = {name: t.copy(deep=True) for name, t in vars(d).items() if isinstance(t, pd.DataFrame)}
    out = prepare(d)
    assert set(out) == {"selected", "pulses", "waveforms", "bridge", "diagnostics"}
    assert out["selected"].frame_id.tolist() == [2, 3]
    assert out["diagnostics"]["registry_sha256"] == F.digest(d.registry)
    assert out["diagnostics"]["complete_stock_refit_bijection"]
    assert out["diagnostics"]["finder_not_run"] and not out["diagnostics"]["nonempty_membership_established"]
    assert out["pulses"].refit_idx.tolist() == [0, 1, 2, 3, 4, 0]
    assert out["pulses"].failed.sum() == out["pulses"].zerondf.sum() == 1
    assert out["pulses"].starttdc.eq(90).all()
    pd.testing.assert_frame_equal(out["pulses"][d.p.columns], d.p.reset_index(drop=True), check_exact=True)
    bridge = out["bridge"].sort_values(["frame_id", "refit_idx"])
    assert bridge.stock_idx.tolist() == [4, 1, 3, 0, 2, 0]
    assert bridge.refit_idx.tolist() == [0, 1, 2, 3, 4, 0]
    assert "idx" not in bridge and set(R.KEYS) <= set(bridge)
    assert set(bridge.input_sha256) == {d.input_sha}
    for name, expected in before.items():
        pd.testing.assert_frame_equal(getattr(d, name), expected, check_exact=True)


def test_optional_raw_pbtime_and_rounded_storage(domain):
    domain.events = domain.events.drop(columns="pbtime_raw")
    domain.events["pbtime"] = domain.events.pbtime.astype(np.float32)
    prepare(domain)


@pytest.mark.parametrize("case", ["missing_quiet_identity", "quiet_refit_only", "ambiguous_stock", "wrong_run", "duplicate_events", "duplicate_registry"])
def test_quiet_and_full_identity_census_fail_closed(domain, case):
    if case == "missing_quiet_identity":
        domain.stock = domain.stock.loc[domain.stock.event.eq(7)]
    elif case == "quiet_refit_only":
        domain.p = domain.p.loc[domain.p.event.eq(7)]
    elif case == "ambiguous_stock":
        extra = domain.stock.iloc[-1:].assign(run=np.uint32(1431))
        domain.stock = pd.concat([domain.stock, extra], ignore_index=True)
    elif case == "wrong_run":
        domain.stock["run"] = np.uint32(1431)
    elif case == "duplicate_events":
        domain.events.loc[1, "event"] = 7
    else:
        domain.registry = pd.concat([domain.registry, domain.registry.iloc[:1]], ignore_index=True)
    with pytest.raises((ValueError, AssertionError)):
        prepare(domain)


@pytest.mark.parametrize("table,column", [("p", "event"), ("p", "wf"), ("p", "seq"), ("p", "chan"),
    ("w", "starttdc"), ("w", "wf"), ("events", "event"), ("stock", "run"), ("stock", "idx"),
    ("stock", "bar"), ("stock", "sipm"), ("comparison", "wf"), ("comparison", "seq"),
    ("comparison", "chan"), ("calibration", "status"), ("conditions", "feb")])
def test_noninteger_identifiers_rejected(domain, table, column):
    t = getattr(domain, table)
    t[column] = t[column].astype(float)
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("field", ["pbtime", "pbtime_raw"])
def test_registry_bunch_bits_are_authoritative(domain, field):
    domain.events.loc[0, field] = (float(np.nextafter(np.float32(domain.events.loc[0, field]), np.float32(np.inf)))
                                   if field == "pbtime" else np.nextafter(domain.events.loc[0, field], np.inf))
    with pytest.raises(ValueError):
        prepare(domain)


def test_float32_negative_zero_is_not_positive_zero(domain):
    domain.p = domain.p.iloc[:0]
    domain.w = domain.w.iloc[:0]
    domain.conditions = domain.conditions.iloc[:0]
    domain.registry["pbtime"] = 0.
    domain.events["pbtime"] = -0.
    domain.events["pbtime_raw"] = 0.
    with pytest.raises(ValueError, match="float32 pbtime"):
        prepare(domain)


@pytest.mark.parametrize("field", ["time", "time_relative", "time_reco", "pe"])
@pytest.mark.parametrize("case", ["float32", "nan", "ulp"])
def test_refit_floating_storage_and_exact_time_relations(domain, field, case):
    if case == "float32":
        domain.p[field] = domain.p[field].astype(np.float32)
    elif case == "nan":
        domain.p.loc[0, field] = np.nan
    else:
        domain.p.loc[0, field] = np.nextafter(domain.p.loc[0, field], np.inf)
    with pytest.raises(ValueError):
        prepare(domain)


def test_relative_negative_zero_fails_exact_bits(domain):
    domain.p.loc[0, "time"] = -2.375
    domain.p.loc[0, "time_relative"] = -0.
    domain.p.loc[0, "time_reco"] = -float(np.float32(-224.46))
    with pytest.raises(ValueError, match="relative time bits"):
        prepare(domain)


@pytest.mark.parametrize("field", ["failed", "zerondf", "channel_ignored"])
@pytest.mark.parametrize("case", ["integer", "nullable"])
def test_boolean_flags_are_not_coerced(domain, field, case):
    domain.p[field] = domain.p[field].astype(int if case == "integer" else "boolean")
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("case", ["missing_wf", "wrong_chan", "duplicate_wf", "duplicate_structure", "invented_start",
                                   "stored_order", "native_order", "start_order", "duplicate_pulse"])
def test_saved_waveform_links_and_native_order_are_not_repaired(domain, case):
    if case == "missing_wf":
        domain.w = domain.w.iloc[1:]
    elif case == "wrong_chan":
        domain.w.loc[0, "chan"] = 1
    elif case == "duplicate_wf":
        domain.w.loc[1, "wf"] = 10
    elif case == "duplicate_structure":
        domain.w.loc[1, "chan"] = 0
    elif case == "invented_start":
        domain.p["starttdc"] = 90
    elif case == "stored_order":
        domain.p = domain.p.iloc[::-1]
    elif case == "native_order":
        domain.p["wf"] = 100 - domain.p.wf
        domain.w["wf"] = 100 - domain.w.wf
    elif case == "start_order":
        domain.p.loc[2, "chan"] = 0
        domain.w.loc[1, "chan"] = 0
        domain.w.loc[1, "starttdc"] = 89
    else:
        domain.p.loc[1, "seq"] = 0
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("field", R.CALIBRATION + ["w"])
def test_all_saved_conditions_including_unoccupied_channels_are_authoritative(domain, field):
    domain.conditions.loc[5, field] += 1
    with pytest.raises(ValueError, match="conditions|calibration"):
        prepare(domain)


@pytest.mark.parametrize("case", ["missing_channel", "duplicate_channel", "status_float", "status_overflow", "negative_area", "nan", "wrong_input"])
def test_full_calibration_domain_is_required(domain, case):
    c = domain.calibration
    if case == "missing_channel":
        domain.calibration = c.iloc[:-1]
    elif case == "duplicate_channel":
        c.loc[15, "chan"] = 14
    elif case == "status_float":
        c["status"] = 0.
    elif case == "status_overflow":
        c.loc[15, "status"] = 2**16
    elif case == "negative_area":
        c.loc[15, "pulsearea"] = -1.
    elif case == "nan":
        c.loc[15, "pedestal"] = np.nan
    else:
        c["input_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("status", [1, 2, 4, 3])
def test_all_active_policy_never_silently_filters(domain, status):
    domain.calibration.loc[0, "status"] = status
    domain.conditions.loc[domain.conditions.chan.eq(0), "status"] = status
    domain.p.loc[domain.p.chan.eq(0), "channel_ignored"] = bool(status & 2)
    with pytest.raises(ValueError, match="all-active|channel policy"):
        prepare(domain)


@pytest.mark.parametrize("field", R.STOCK)
def test_every_copied_stock_value_requires_exact_storage(domain, field):
    domain.comparison.loc[0, "pe_stock" if field == "pe" else field] += 1
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("field", ["pe_stock", "time", "flags", "idx", "run"])
def test_comparison_copied_stock_dtype_is_strict(domain, field):
    domain.comparison[field] = domain.comparison[field].astype(np.float64 if field != "time" else np.float32)
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("field", ["pe_fast", "time_reco", "failed", "zerondf", "delta_time", "delta_pe", "wf", "chan"])
def test_comparison_refit_fields_and_deltas_are_exact(domain, field):
    if field in ("failed", "zerondf"):
        domain.comparison.loc[0, field] = not domain.comparison.loc[0, field]
    elif field in ("wf", "chan"):
        domain.comparison.loc[0, field] += 1
    else:
        domain.comparison.loc[0, field] = np.nextafter(domain.comparison.loc[0, field], np.inf)
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("case", ["left_only", "right_only", "missing_comparison", "extra_comparison", "duplicate_comparison",
                                   "missing_refit", "extra_refit", "noncontiguous_stock", "duplicate_stock"])
def test_complete_bijections_never_use_silent_inner_join(domain, case):
    if case in ("left_only", "right_only"):
        domain.comparison.loc[0, "_merge"] = case
    elif case == "missing_comparison":
        domain.comparison = domain.comparison.iloc[1:]
    elif case in ("extra_comparison", "duplicate_comparison"):
        domain.comparison = pd.concat([domain.comparison, domain.comparison.iloc[:1]], ignore_index=True)
        if case == "extra_comparison":
            domain.comparison.loc[6, "idx"] = 99
    elif case == "missing_refit":
        domain.p = domain.p.iloc[1:]
    elif case == "extra_refit":
        domain.p = pd.concat([domain.p, domain.p.iloc[-1:].assign(seq=1)], ignore_index=True)
    elif case == "noncontiguous_stock":
        domain.stock.loc[0, "idx"] = 5
    else:
        domain.stock.loc[0, "idx"] = 1
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("field", ["pe_fast", "time_reco", "delta_time", "delta_pe"])
def test_comparison_refit_floats_are_never_downcast(domain, field):
    domain.comparison[field] = domain.comparison[field].astype(np.float32)
    with pytest.raises(ValueError):
        prepare(domain)


@pytest.mark.parametrize("table,rows", [("p", 2049), ("w", 2049), ("stock", 2049), ("comparison", 2049),
                                        ("conditions", 20481), ("events", 5), ("registry", 9001), ("calibration", 15361)])
def test_resource_row_caps_precede_schema_and_work(domain, monkeypatch, table, rows):
    setattr(domain, table, pd.DataFrame(index=np.arange(rows)))
    monkeypatch.setattr(P, "combination_work", forbidden)
    with pytest.raises(ValueError, match="cap"):
        prepare(domain)


def test_per_frame_pulse_cap_precedes_geometry_work(domain, monkeypatch):
    domain.p = domain.p.iloc[np.zeros(513, dtype=np.int64)].reset_index(drop=True)
    domain.p["seq"] = np.arange(513, dtype=np.int64)
    monkeypatch.setattr(P, "combination_work", forbidden)
    with pytest.raises(ValueError, match="per-frame.*cap"):
        prepare(domain)


def test_real_geometry_candidate_cap_precedes_bridge(domain):
    p = domain.p.iloc[np.repeat([0, 2, 3, 4], 100)].reset_index(drop=True)
    p["seq"] = np.tile(np.arange(100, dtype=np.int64), 4)
    domain.p = p
    with pytest.raises(ValueError, match="candidate budget"):
        prepare(domain)


def test_conditions_and_pulse_columns_are_not_silently_deduplicated(domain):
    domain.conditions = pd.concat([domain.conditions, domain.conditions.iloc[-1:]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate boundary conditions"):
        prepare(domain)


@pytest.mark.parametrize("table", ["p", "w", "stock", "comparison", "events", "calibration", "conditions"])
def test_duplicate_columns_are_rejected(domain, table):
    t = getattr(domain, table)
    setattr(domain, table, pd.concat([t, t.iloc[:, :1]], axis=1))
    with pytest.raises(ValueError):
        prepare(domain)


def test_quiet_refit_frame_is_counted_but_missing_bijection_is_not_admitted(domain, monkeypatch):
    domain.p = domain.p.loc[domain.p.event.eq(7)]
    inspect = I.inspect
    records = []

    def observe(*args):
        result = inspect(*args)
        records.append(result)
        return result

    monkeypatch.setattr(I, "inspect", observe)
    with pytest.raises(ValueError, match="incomplete.*active refit"):
        prepare(domain)
    assert records[0]["frames"][1]["saved_pulses"] == 0
    assert records[0]["frames"][1]["event"] == 9


def test_empty_stock_cannot_supply_a_quiet_event_identity(domain):
    domain.stock, domain.comparison = domain.stock.iloc[:0], domain.comparison.iloc[:0]
    domain.p, domain.w = domain.p.iloc[:0], domain.w.iloc[:0]
    with pytest.raises(ValueError, match="identity for every boundary target"):
        prepare(domain)


def test_four_frame_general_limit_does_not_relax_three_frame_boundary_contract(domain):
    extra = domain.events.copy()
    extra["event"] += 10
    domain.events = pd.concat([domain.events, extra], ignore_index=True)
    extra_stock = domain.stock.copy()
    extra_stock["event"] += np.uint32(10)
    domain.stock = pd.concat([domain.stock, extra_stock], ignore_index=True)
    with pytest.raises(ValueError, match="boundary inventory row cap"):
        prepare(domain)


def test_no_new_fit_quality_or_stock_flag_agreement_cut(domain):
    domain.p["failed"], domain.p["zerondf"] = True, True
    domain.p.loc[0, "pe"] = 0.
    domain.comparison["failed"], domain.comparison["zerondf"] = True, True
    domain.comparison["pe_fast"] = domain.p.pe
    domain.comparison["delta_pe"] = domain.comparison.pe_fast - domain.comparison.pe_stock
    out = prepare(domain)
    assert len(out["pulses"]) == len(domain.p)
    assert out["pulses"].failed.all() and out["pulses"].zerondf.all()
    assert out["pulses"].pe.iloc[0] == 0.
    assert out["bridge"]["flags"].equals(domain.stock["flags"])


def test_extra_complete_input_calibration_is_allowed_and_validated(domain):
    extra = domain.calibration.assign(input_sha256="a" * 64)
    domain.calibration = pd.concat([extra, domain.calibration], ignore_index=True)
    prepare(domain)
    domain.calibration = domain.calibration.iloc[1:]
    with pytest.raises(ValueError, match="incomplete calibration"):
        prepare(domain)


def test_consistently_changed_relative_times_do_not_override_full_calibration(domain):
    domain.conditions["timeoffset"] += 1.
    domain.p["time_relative"] += 1.
    domain.p["time_reco"] += 1.
    domain.comparison["time_reco"] += 1.
    domain.comparison["delta_time"] += 1.
    with pytest.raises(ValueError, match="authoritative calibration"):
        prepare(domain)


@pytest.mark.parametrize("table", ["stock", "comparison"])
@pytest.mark.parametrize("field", ["input_sha256", "frame_id", "refit_idx", "stock_idx", "starttdc"])
def test_bridge_labels_cannot_be_invented_or_overwritten(domain, table, field):
    getattr(domain, table)[field] = 0
    with pytest.raises(ValueError, match="invented bridge identity"):
        prepare(domain)


def test_nonphysical_waveform_metadata_and_fit_payload_are_preserved(domain):
    domain.w["samples"] = np.arange(len(domain.w), dtype=np.int64) + 12
    domain.p["custom_fit_scalar"] = np.arange(len(domain.p), dtype=np.float64) / 7.
    out = prepare(domain)
    pd.testing.assert_frame_equal(out["waveforms"][domain.w.columns], domain.w, check_exact=True)
    pd.testing.assert_series_equal(out["pulses"].custom_fit_scalar, domain.p.custom_fit_scalar, check_exact=True)


def test_legacy_comparison_recovers_channels_only_by_full_stock_index(domain):
    domain.comparison = domain.comparison.drop(columns=["bar", "sipm", "chan"]).iloc[::-1].reset_index(drop=True)
    domain.stock = domain.stock.iloc[::-1].reset_index(drop=True)
    original = domain.comparison.copy(deep=True)
    out = prepare(domain)
    assert out["diagnostics"]["comparison_channel_from_stock_index"]
    bridge = out["bridge"].sort_values(["frame_id", "refit_idx"])
    assert bridge.chan.tolist() == domain.p.chan.tolist()
    assert bridge.stock_idx.tolist() == [4, 1, 3, 0, 2, 0]
    pd.testing.assert_frame_equal(domain.comparison, original, check_exact=True)


@pytest.mark.parametrize("missing", [["bar"], ["sipm"], ["chan"], ["bar", "sipm"], ["bar", "chan"], ["sipm", "chan"]])
def test_partial_legacy_channel_schema_rejected(domain, missing):
    domain.comparison = domain.comparison.drop(columns=missing)
    with pytest.raises(ValueError, match="comparison columns"):
        prepare(domain)


@pytest.mark.parametrize("mutation", ["wrong_index", "wrong_run", "duplicate_index", "wrong_stock_channel"])
def test_legacy_comparison_cannot_guess_channel_identity(domain, mutation):
    domain.comparison = domain.comparison.drop(columns=["bar", "sipm", "chan"])
    if mutation == "wrong_index":
        domain.comparison.loc[0, "idx"] = 99
    elif mutation == "wrong_run":
        domain.comparison.loc[0, "run"] = 1431
    elif mutation == "duplicate_index":
        domain.comparison.loc[0, "idx"] = domain.comparison.loc[1, "idx"]
    else:
        domain.stock.loc[0, "bar"] = 1
    with pytest.raises(ValueError):
        prepare(domain)


def test_refit_numeric_differences_and_flags_are_diagnostics_not_quality_cuts(domain):
    out = prepare(domain)
    d = out["diagnostics"]["stock_refit_differences"]
    assert d["time_changed"] == d["pe_changed"] == len(domain.p)
    assert d["max_abs_time_ns"] == 0.125
    assert d["max_abs_pe"] == float(domain.comparison.delta_pe.abs().max())
    assert d["failed_flag_changed"] == 0 and d["zerondf_flag_changed"] == 2
    assert len(out["pulses"]) == len(domain.p)
