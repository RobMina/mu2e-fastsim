"""Saved-pulse admission, native attachments and exact structural diagnostics."""
from dataclasses import FrozenInstanceError, replace

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.reco import coinc as C
from crvfastsim.veto import frame_registry as F, pilot_coincidence as P, replay_recover as R


def geometry():
    layer = np.arange(4)
    pos = np.column_stack([np.zeros(4), layer * 10., np.zeros(4)])
    return C.SectorGeom(np.zeros(4, dtype=int), layer, np.zeros(4, dtype=int), pos[:, 0], pos[:, 1], pos,
                        np.array([2]), np.array([1]), ["T1"], np.array([-700.]), np.array([700.]),
                        np.array([True]), np.array([True]))


def domain():
    events = pd.DataFrame({"input_sha256": ["a" * 64] * 4 + ["b" * 64] * 4,
                           "run": [1430] * 8, "subrun": [0, 0, 1, 1] * 2,
                           "event": [7, 2**20 + 7, 7, 8] * 2,
                           "pbtime": [-224.46, -214.46, -204.46, -219.46] * 2, "intensity": 1.})
    registry = F.freeze(events)
    selected = registry.iloc[:4].copy()
    frames = selected.iloc[np.repeat([0, 1, 2], [4, 3, 3])].reset_index(drop=True)
    p = frames[P.FRAME_KEYS].copy()
    p["chan"] = [8, 1, 0, 4, 0, 4, 8, 0, 4, 8]
    p["seq"] = [0] * 7 + [1] * 3
    p["starttdc"] = [90] * 4 + [110] * 3 + [130] * 3
    p["wf"] = np.arange(len(p)) + 50
    p["time_relative"] = [502., 505., 500., 501., 600., 601., 602., 700., 701., 702.]
    p["time_reco"] = p.time_relative - frames.pbtime.to_numpy().astype(np.float32).astype(float)
    p["pe"] = 10.123456789
    p["channel_ignored"] = False
    calibration = pd.DataFrame({"input_sha256": np.repeat(["a" * 64, "b" * 64], 16),
                                "chan": np.tile(np.arange(16), 2), "status": 0})
    return p, registry, selected, calibration, geometry()


def run(data, **kwargs):
    p, r, s, c, g = data
    return P.run(p, r, s, c, g, registry_digest=F.digest(r), **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("unexpected finder or physics regeneration execution")


def test_native_attachment_all_scalar_fields_and_quiet_frame(monkeypatch):
    data = domain()
    p, r, s, c, g = data
    original = R.recover
    recorded = {}

    def recover(batch, G, status, cfg=None, **kw):
        assert kw == {"pe_float32": False}
        assert cfg == R.finder_config()
        assert cfg.add_opposite_side_pulses and not cfg.ignore_secondary_pulses
        np.testing.assert_array_equal(batch.events["run"], 0)
        np.testing.assert_array_equal(batch.events["subrun"], 0)
        np.testing.assert_array_equal(batch.events["event"], s.frame_id)
        np.testing.assert_array_equal(batch.pulses["time"], p.sort_values(["frame_id", *P.PULSE_KEYS]).time_reco)
        recorded["native"] = original(batch, G, status, cfg, **kw)
        return recorded["native"]

    monkeypatch.setattr(R, "recover", recover)
    out = run(data)
    assert set(out) == {"frames", "pulses", "clusters", "members"}
    assert out["frames"].pulses.tolist() == [4, 3, 3, 0]
    assert out["frames"].clusters.tolist() == [1, 1, 1, 0]
    assert out["frames"].members.tolist() == [4, 3, 3, 0]
    assert set(P.SCALARS) <= set(out["clusters"])
    for name in P.SCALARS:
        np.testing.assert_array_equal(out["clusters"][name], ak.to_numpy(recorded["native"][name]))
    first = out["members"].query("frame_id == 0")
    assert first.pulse_idx.tolist() == [0, 2, 3, 1]
    assert first.chan.tolist() == [0, 4, 8, 1]
    assert first.member_order.tolist() == [0, 1, 2, 3]
    assert out["clusters"].sidehits1.tolist() == [1, 0, 0]
    assert out["clusters"].pe.iloc[0] == 4 * p.pe.iloc[0]
    assert out["pulses"].idx.tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]
    for table in out.values():
        joined = table.merge(s[P.FRAME_KEYS], on=P.FRAME_KEYS, validate="many_to_one")
        assert len(joined) == len(table)
    assert 2**20 + 7 in out["clusters"].event.to_numpy()


def test_permutations_source_seed_metadata_and_local_wf():
    data = domain()
    full = run(data)
    p, r, s, c, g = data
    reordered = p.sample(frac=1, random_state=21).assign(draw_seed=np.uint64(3027001), arm="arrival_only")
    out = run((reordered, r, s.iloc[::-1], c.iloc[::-1], g))
    for name in full:
        pd.testing.assert_frame_equal(full[name], out[name], check_exact=True)
    local = reordered.assign(wf=reordered.wf + 100)
    assert P.compare(full, run((local, r, s, c, g)))["exact"]
    changed = reordered.copy()
    changed.loc[changed.index[0], "pe"] = np.nextafter(changed.pe.iloc[0], np.inf)
    result = P.compare(full, run((changed, r, s, c, g)))
    assert not result["exact"] and not result["pulse_identity_exact"]
    assert result["stages"]["pulses"]["field_differences"]["pe"] == 1
    assert result["numerical_tolerances"] is False and result["stock_equivalence_asserted"] is False


def test_duplicate_original_event_tuples_across_input_are_not_identity():
    p, r, s, c, g = domain()
    a = run((p, r, s, c, g))
    other = r.iloc[4:]
    q = p.copy()
    q["input_sha256"] = "b" * 64
    q["frame_id"] += 4
    b = run((q, r, other, c, g))
    result = P.compare(a, b)
    assert not result["exact"]
    assert result["stages"]["pulses"]["left_only"] == len(p)
    assert result["stages"]["members"]["right_only"] == len(p)
    np.testing.assert_array_equal(a["clusters"].pe, b["clusters"].pe)


def test_exact_subset_and_partition_of_existing_run_output():
    data = domain()
    p, r, s, c, g = data
    full = run(data)
    parts = []
    for ids in ([3, 1], [2, 0]):
        selected = s[s.frame_id.isin(ids)].iloc[::-1]
        subset = run((p[p.frame_id.isin(ids)].iloc[::-1], r, selected, c, g))
        expected = {name: table[table.frame_id.isin(ids)].reset_index(drop=True) for name, table in full.items()}
        assert P.compare(expected, subset)["exact"]
        for name in full:
            pd.testing.assert_frame_equal(expected[name], subset[name], check_exact=True)
        parts.append(subset)
    combined = {name: pd.concat([part[name] for part in parts], ignore_index=True) for name in full}
    assert P.compare(full, combined)["exact"]


def test_all_empty_typed_tables_and_no_finder_dispatch(monkeypatch):
    p, r, s, c, g = domain()
    monkeypatch.setattr(C, "find_event", forbidden)
    out = run((p.iloc[:0], r, s, c, g))
    assert len(out["frames"]) == 4
    assert out["frames"][["pulses", "clusters", "members"]].eq(0).all().all()
    assert all(len(out[name]) == 0 for name in ("pulses", "clusters", "members"))
    assert P.compare(out, out)["exact"]


@pytest.mark.parametrize("column", [*P.FRAME_KEYS, *P.PULSE_FIELDS])
def test_mandatory_schema_before_finder(monkeypatch, column):
    p, r, s, c, g = domain()
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises(ValueError, match="columns"):
        run((p.drop(columns=column), r, s, c, g))


@pytest.mark.parametrize("column", ["run", "subrun", "event", "frame_id", "chan", "starttdc", "seq", "wf"])
@pytest.mark.parametrize("kind", ["float", "negative", "null", "bool"])
def test_strict_integer_schema_before_finder(monkeypatch, column, kind):
    p, r, s, c, g = domain()
    if kind == "float":
        p[column] = p[column].astype(float)
    elif kind == "negative":
        p.loc[0, column] = -1
    elif kind == "bool":
        p[column] = False
    else:
        p[column] = p[column].astype("Int64")
        p.loc[0, column] = pd.NA
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises(ValueError, match="integer"):
        run((p, r, s, c, g))


@pytest.mark.parametrize("column", ["time_relative", "time_reco", "pe"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "500", True])
def test_finite_numeric_pulse_schema(monkeypatch, column, value):
    p, r, s, c, g = domain()
    p[column] = value
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises(ValueError, match="numeric"):
        run((p, r, s, c, g))


@pytest.mark.parametrize("case", ["time", "negative_pe", "ignored", "ignored_dtype", "duplicate", "foreign",
                                   "wrong_run", "wrong_frame", "chan", "seq", "starttdc", "wf", "double_column"])
def test_malformed_or_out_of_domain_pulses(monkeypatch, case):
    p, r, s, c, g = domain()
    if case == "time":
        p.loc[0, "time_reco"] = np.nextafter(p.time_reco.iloc[0], np.inf)
    elif case == "negative_pe":
        p.loc[0, "pe"] = -1.
    elif case == "ignored":
        p.loc[0, "channel_ignored"] = True
    elif case == "ignored_dtype":
        p["channel_ignored"] = 0
    elif case == "duplicate":
        p = pd.concat([p, p.iloc[:1]], ignore_index=True)
    elif case == "foreign":
        p["input_sha256"] = "b" * 64
    elif case == "wrong_run":
        p["run"] += 1
    elif case == "wrong_frame":
        p["frame_id"] = 1
    elif case == "double_column":
        p = pd.concat([p, p[["pe"]]], axis=1)
    else:
        p[case] = np.uint64({"chan": 16, "seq": 2**31, "starttdc": 2**31, "wf": 2**63}[case])
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises(ValueError):
        run((p, r, s, c, g))


@pytest.mark.parametrize("case", ["missing", "duplicate", "foreign", "float_chan", "float_status", "bad_status",
                                   "negative_status", "dead", "ignored_mismatch", "ignored", "compensate"])
def test_complete_source_bound_status_and_no_silent_drops(monkeypatch, case):
    p, r, s, c, g = domain()
    if case == "missing":
        c = c.iloc[1:]
    elif case == "duplicate":
        c = pd.concat([c, c.iloc[:1]])
    elif case == "foreign":
        c = c[c.input_sha256.eq("b" * 64)]
    elif case == "float_chan":
        c["chan"] = c.chan.astype(float)
    elif case == "float_status":
        c["status"] = c.status.astype(float)
    elif case == "bad_status":
        c.loc[0, "status"] = 2**16
    elif case == "negative_status":
        c.loc[0, "status"] = -1
    elif case == "compensate":
        c.loc[c.chan.isin([2, 6, 10]), "status"] = 1
        out = run((p, r, s, c, g))
        assert out["clusters"].pe.iloc[1] == 6 * p.pe.iloc[0]
        assert len(out["pulses"]) == len(p)
        return
    else:
        c.loc[0, "status"] = 1 if case == "dead" else 2
        if case == "ignored":
            p.loc[p.chan.eq(0), "channel_ignored"] = True
            out = run((p, r, s, c, g))
            assert len(out["pulses"]) == len(p)
            assert out["pulses"].channel_ignored.sum() == p.chan.eq(0).sum()
            assert not out["members"].chan.eq(0).any()
            return
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises(ValueError):
        run((p, r, s, c, g))


@pytest.mark.parametrize("case", ["multi_input", "selected_float", "selected_changed", "selected_duplicate", "wrong_digest", "pbtime_overflow"])
def test_registry_admission_precedes_finder(monkeypatch, case):
    p, r, s, c, g = domain()
    if case == "multi_input":
        s = r.iloc[[0, 4]]
    elif case == "selected_float":
        s["frame_id"] = s.frame_id.astype(float)
    elif case == "selected_changed":
        s["intensity"] += 1
    elif case == "selected_duplicate":
        s = pd.concat([s.iloc[:1], s.iloc[:1]])
    elif case == "pbtime_overflow":
        r["pbtime"] = 1e300
        s = r.iloc[:4]
        p = p.iloc[:0]
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises((ValueError, AssertionError)):
        P.run(p, r, s, c, g, registry_digest="0" * 64 if case == "wrong_digest" else F.digest(r))


@pytest.mark.parametrize("field,ceiling", [("max_frames", 4), ("max_pulses_per_frame", 512),
                                           ("max_total_pulses", 2048), ("max_combinations", 2_000_000)])
def test_all_limit_types_immutable_and_lowerable(field, ceiling):
    for value in (0, -1, True, 1., np.int64(1), ceiling + 1):
        with pytest.raises(ValueError, match="positive native int"):
            replace(P.Limits(), **{field: value})
    lowered = replace(P.Limits(), **{field: 1})
    assert getattr(lowered, field) == 1
    with pytest.raises(FrozenInstanceError):
        setattr(lowered, field, 2)


@pytest.mark.parametrize("field", ["max_frames", "max_pulses_per_frame", "max_total_pulses", "max_combinations"])
def test_all_caps_fail_before_helper_or_finder(monkeypatch, field):
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    monkeypatch.setattr(R, "recover", forbidden)
    with pytest.raises(ValueError, match="cap"):
        run(domain(), limits=replace(P.Limits(), **{field: 1}))


def test_single_layer_counts_do_not_claim_nonexistent_combinations():
    p, r, s, c, g = domain()
    p = p.iloc[np.zeros(100, dtype=int)].reset_index(drop=True)
    p["seq"] = np.arange(len(p))
    assert P.combination_work(p, g, R.finder_config()) == 0
    result = run((p, r, s, c, g))
    assert len(result["pulses"]) == 100 and len(result["clusters"]) == 0


@pytest.mark.parametrize("case", ["bounds", "duplicate", "sector", "layers", "nLayers", "temporary"])
def test_native_membership_validation_no_nearest_match(monkeypatch, case):
    original = R.recover

    def corrupted(*args, **kwargs):
        native = original(*args, **kwargs)
        if case in ("bounds", "duplicate"):
            links = ak.to_list(native.pulses)
            links[0][-1] = 999 if case == "bounds" else links[0][0]
            return ak.with_field(native, ak.Array(links), "pulses")
        if case == "sector":
            return ak.with_field(native, native.sectortype + 1, "sectortype")
        if case == "temporary":
            return ak.with_field(native, native.run + 1, "run")
        if case == "nLayers":
            return ak.with_field(native, native.nLayers + 1, "nLayers")
        return ak.with_field(native, ak.Array([[0, 1, 3]] * len(native)), "layers")

    monkeypatch.setattr(R, "recover", corrupted)
    with pytest.raises(ValueError, match="native"):
        run(domain())


def test_comparison_reports_native_links_properties_and_exact_nextafter():
    full = run(domain())
    right = {name: table.copy() for name, table in full.items()}
    right["members"].loc[0, "pulse_idx"] += 1
    right["clusters"].loc[0, "avgtime"] = np.nextafter(right["clusters"].avgtime.iloc[0], np.inf)
    result = P.compare(full, right)
    assert result["pulse_identity_exact"]
    assert not result["native_links_exact"] and not result["cluster_properties_exact"]
    assert result["stages"]["clusters"]["field_differences"]["avgtime"] == 1
    right["pulses"].loc[0, "seq"] += 1
    assert not P.compare(full, right)["pulse_identity_exact"]


def test_no_physics_regeneration(monkeypatch):
    from crvfastsim.veto import bounded_response as B, bounded_sipm as S, bounded_waveform as W, diagnostic_fit as D
    from crvfastsim.veto import keyed_optics as K

    for module, name in ((B, "run_arm"), (S, "simulate_candidates"), (S, "thermal_candidates"),
                         (W, "synthesize"), (D, "reconstruct")):
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(K, "make_photons", forbidden)
    assert len(run(domain())["clusters"]) == 3


def test_no_coincidences_with_nonempty_pulses_and_quiet_only_subset():
    p, r, s, c, g = domain()
    sparse = run((p.iloc[:1], r, s, c, g))
    assert sparse["frames"].pulses.tolist() == [1, 0, 0, 0]
    assert sparse["frames"].clusters.eq(0).all()
    assert len(sparse["pulses"]) == 1 and len(sparse["members"]) == 0
    full = run((p, r, s, c, g))
    quiet = run((p.iloc[:0], r, s.iloc[[3]], c, g))
    for name, table in full.items():
        pd.testing.assert_frame_equal(table[table.frame_id.eq(3)].reset_index(drop=True), quiet[name], check_exact=True)


def test_exact_limit_boundary_is_inclusive():
    data = domain()
    assert P.combination_work(data[0], data[4], R.finder_config()) == 3
    limits = P.Limits(max_frames=4, max_pulses_per_frame=4, max_total_pulses=10, max_combinations=3)
    assert P.compare(run(data), run(data, limits=limits))["exact"]


@pytest.mark.parametrize("field,size", [("max_pulses_per_frame", 513), ("max_total_pulses", 2049)])
def test_hard_pulse_ceilings_precede_finder(monkeypatch, field, size):
    p, r, s, c, g = domain()
    p = p.iloc[np.zeros(size, dtype=int)].reset_index(drop=True)
    p["seq"] = np.arange(size)
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises(ValueError, match="pulse cap"):
        run((p, r, s, c, g))


def test_two_native_clusters_in_one_frame_keep_local_ids_and_secondaries():
    p, r, s, c, g = domain()
    p = p[p.frame_id.isin([0, 2])].copy()
    p["frame_id"] = 0
    for name in F.FRAME_KEYS:
        p[name] = s[name].iloc[0]
    p["time_reco"] = p.time_relative - float(np.float32(s.pbtime.iloc[0]))
    out = run((p, r, s.iloc[:1], c, g))
    assert out["clusters"].idx.tolist() == [0, 1]
    assert out["frames"].members.tolist() == [7]
    assert out["members"].groupby("cluster_idx").size().tolist() == [4, 3]
    assert out["members"].groupby("cluster_idx").member_order.min().eq(0).all()


def test_saved_relative_time_is_not_reconstructed_by_lossy_roundtrip(monkeypatch):
    data = domain()
    p, r, s, c, g = data
    original = R.prepare_pulses
    observed = {}
    def prepare(*a, **k):
        batch = original(*a, **k)
        batch.finder_time = batch.finder_time + 1e-10
        return batch
    recover = R.recover
    def exact(batch, *a, **k):
        observed["time"] = batch.finder_time.copy()
        return recover(batch, *a, **k)
    monkeypatch.setattr(R, "prepare_pulses", prepare)
    monkeypatch.setattr(R, "recover", exact)
    run(data)
    np.testing.assert_array_equal(observed["time"], p.sort_values(["frame_id", *P.PULSE_KEYS]).time_relative)


def test_real_float_roundtrip_loss_does_not_shift_finder(monkeypatch):
    p, r, s, c, g = domain()
    p = p.iloc[:4].copy()
    p["time_relative"] = np.array([.1, 20.1, .1, .1])
    p["time_reco"] = p.time_relative - float(np.float32(s.pbtime.iloc[0]))
    assert np.any(p.time_reco.to_numpy() + float(np.float32(s.pbtime.iloc[0])) != p.time_relative.to_numpy())
    original = R.recover
    def exact(batch, *a, **k):
        np.testing.assert_array_equal(batch.finder_time, p.sort_values(["frame_id", *P.PULSE_KEYS]).time_relative)
        return original(batch, *a, **k)
    monkeypatch.setattr(R, "recover", exact)
    result = run((p, r, s, c, g))
    assert result["frames"].pulses.sum() == 4


def test_nonempty_failed_and_zerondf_pulses_are_not_fit_quality_cuts():
    p, r, s, c, g = domain()
    before = run((p, r, s, c, g))
    p["failed"], p["zerondf"] = True, True
    after = run((p, r, s, c, g))
    assert P.compare(before, after)["exact"]
    assert len(after["members"]) == 10


def test_two_sector_types_restore_all_native_members():
    p, r, s, c, g = domain()
    g = replace(g, sector=np.repeat([0, 1], 4), layer=np.tile(g.layer, 2),
                counter_in_type=np.tile(g.counter_in_type, 2), x=np.tile(g.x, 2), y=np.tile(g.y, 2),
                pos=np.tile(g.pos, (2, 1)), length_dir=np.array([2, 2]), sector_type=np.array([1, 2]),
                sector_names=["T1", "T2"], pos_side0=np.array([-700., -700.]), pos_side1=np.array([700., 700.]),
                sipms_side0=np.array([True, True]), sipms_side1=np.array([True, True]))
    second = p.iloc[:4].copy()
    second["chan"] += 16
    second["wf"] += 100
    p = pd.concat([p, second], ignore_index=True)
    c = pd.DataFrame(dict(input_sha256=np.repeat(["a" * 64, "b" * 64], 32),
                          chan=np.tile(np.arange(32), 2), status=0))
    out = run((p, r, s, c, g))
    assert set(out["clusters"].sectortype) == {1, 2}
    assert out["members"].sector.value_counts().to_dict() == {0: 10, 1: 4}


@pytest.mark.parametrize("layers,expected", [(2, 600), (3, 4000), (4, 10000)])
def test_geometry_combination_budget_matches_layer_products(layers, expected):
    g = geometry()
    p = pd.DataFrame(dict(frame_id=np.zeros(40, np.int64), chan=np.repeat(np.arange(4) * 4, 10)))
    cfg = R.finder_config()
    cfg = replace(cfg, cuts={"T1": replace(cfg.cuts["T1"], coincidence_layers=layers)})
    assert P.combination_work(p, g, cfg) == expected
    assert P.combination_work(pd.concat([p, p.assign(chan=p.chan + 1)]), g, cfg) == 2 * expected
    assert P.combination_work(pd.concat([p, p.assign(frame_id=1)]), g, cfg) == 2 * expected


def test_high_single_layer_occupancy_has_no_coincidence_tuples():
    p = pd.DataFrame(dict(frame_id=np.zeros(512, np.int64), chan=np.zeros(512, np.int64)))
    assert P.combination_work(p, geometry(), R.finder_config()) == 0


def test_geometry_combinations_reject_before_finder(monkeypatch):
    p, r, s, c, g = domain()
    p = p.iloc[np.zeros(512, dtype=int)].reset_index(drop=True)
    p["chan"] = np.repeat(np.arange(4) * 4, 128)
    p["seq"] = np.tile(np.arange(128), 4)
    p["wf"] = np.repeat(np.arange(4), 128)
    assert P.combination_work(p, g, R.finder_config()) == 4 * 128**3
    monkeypatch.setattr(R, "prepare_pulses", forbidden)
    with pytest.raises(ValueError, match="combination cap"):
        run((p, r, s, c, g))


def test_ignored_opposite_side_is_retained_but_never_attached():
    p, r, s, c, g = domain()
    c.loc[c.chan.eq(1), "status"] = 2
    p.loc[p.chan.eq(1), "channel_ignored"] = True
    out = run((p, r, s, c, g))
    assert len(out["pulses"]) == 10 and out["pulses"].channel_ignored.sum() == 1
    members = out["members"].query("frame_id == 0")
    assert members.pulse_idx.tolist() == [0, 2, 3]
    assert out["clusters"].query("frame_id == 0").sidehits1.tolist() == [0]


def test_all_ignored_pulses_keep_frame_and_diagnostic_counts():
    p, r, s, c, g = domain()
    c["status"] = 2
    p["channel_ignored"] = True
    out = run((p, r, s, c, g))
    assert len(out["frames"]) == 4 and len(out["pulses"]) == 10
    assert len(out["clusters"]) == len(out["members"]) == 0
    assert out["frames"].pulses.sum() == 10
