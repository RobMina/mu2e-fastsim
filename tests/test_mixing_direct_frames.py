"""Synthetic direct-frame contracts; no native RNG replay or detector response claims."""
import hashlib
import json

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from mixing import direct_frames
from mixing.direct_frames import load_manifest, to_detector_inputs
from mixing.frames import (CRV_FIELDS, GEN_FIELDS, IDS, PARTICLE_FIELDS, TRACKER_FIELDS,
                           FrameBatch, overlay, read_frames)
from mixing.sampling import BunchLaw, CHAINS, STREAMS, FrameSelection, draw_multiplicities, select_frames, stock_rates


KEYS = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2**53 + 1], [1, 1, 9]], dtype=np.uint64)
INTEGER_FIELDS = {"key", "parent", "pdg", "ccode", "scode", "isprim", "genidx", "idx", "genid",
                  "sid", "plane", "panel", "straw", "upanel", "shape", "ion", "sp", "bar"}


def rows(fields, counts, **overrides):
    total = sum(counts)
    flat = {field: np.zeros(total, dtype=np.int64 if field in INTEGER_FIELDS else np.float64)
            for field in fields}
    flat.update(overrides)
    return ak.unflatten(ak.Array(flat), counts)


def batch():
    return FrameBatch(KEYS.copy(), {
        "particles": rows(PARTICLE_FIELDS, [2, 2, 2, 2, 0], key=np.tile([9, 7], 4),
                          parent=np.tile([7, -1], 4), genidx=np.tile([-1, 0], 4),
                          pdg=np.array([-11, 11, -13, 13, 22, 211, -211, 2212])),
        "gen": rows(GEN_FIELDS, [1, 1, 1, 1, 0], pdg=np.array([11, 13, 211, 2212])),
        "tracker": rows(TRACKER_FIELDS, [1, 1, 1, 1, 0], sp=np.full(4, 9),
                        sid=np.array([11, 12, 13, 14]), t=np.array([-2., 1695., 3400., 250.])),
        "crv": rows(CRV_FIELDS, [2, 2, 2, 2, 0], sp=np.tile([7, 9], 4),
                    t0=np.array([-2., 1695., 3400., np.nan, 250., 251., 10000., 0.]),
                    t1=np.array([-1., 1696., 3401., 1., 251., 252., 10001., np.nan])),
    })


def subset(source, indices):
    return FrameBatch(source.event_ids[indices], {name: array[indices] for name, array in source.collections.items()})


def flat(source, name):
    array = source.collections[name]
    ids = np.repeat(source.event_ids, ak.to_numpy(ak.num(array)), axis=0)
    return {**dict(zip(IDS, ids.T)), **{field: ak.to_numpy(ak.flatten(array[field])) for field in ak.fields(array)}}


def write_dump(path, source, census=True, products=None):
    products = products or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root:
        if census:
            root.mktree("crv/events", {**dict(zip(IDS, source.event_ids.T)),
                                      "n_steps": ak.to_numpy(ak.num(source.collections["crv"])),
                                      **products.get("events", {})})
        for name, tree in (("particles", "crv/particles"), ("gen", "crv/gen"),
                           ("tracker", "tracker/sgsteps"), ("crv", "crv/steps")):
            root.mktree(tree, {**flat(source, name), **products.get(name, {})})


def save_manifest(path, manifest):
    path.write_text(json.dumps(manifest))


@pytest.fixture
def manifest(tmp_path):
    source = batch()
    write_dump(tmp_path / "data/signal.root", source)
    write_dump(tmp_path / "data/pool.root", source)
    config = {"version": 1, "signal": {"path": "data/signal.root"},
              "pools": {s: {"path": "data/pool.root"} for s in STREAMS},
              "campaign": {"extended_mean": 10., "sdf": .8, "cut_max": 60., "beam_primary": True,
                           "efficiencies": {tag: .6 for chain in CHAINS.values() for tag in chain},
                           "stop_prescale": 3., "max_skip": {s: 19 for s in STREAMS},
                           "mode": "sequential", "postmix_selection": "universal",
                           "additional_time_offset_ns": 0}, "seed": 421}
    path = tmp_path / "manifest.json"
    save_manifest(path, config)
    return path, config


def test_full_ids_pdg_join_all_fields_unwrapped_and_no_mutation():
    source = batch()
    extra = ak.with_field(source.collections["tracker"], source.collections["tracker"].t + 7, "extra")
    source = FrameBatch(source.event_ids, dict(source.collections, tracker=extra))
    before = {name: ak.copy(value) for name, value in source.collections.items()}
    tracker, crv, pdg = to_detector_inputs(source)
    assert list(tracker.columns) == [*IDS, *TRACKER_FIELDS, "extra"]
    assert list(crv.columns) == [*IDS, *CRV_FIELDS]
    np.testing.assert_array_equal(tracker[list(IDS)], KEYS[:4])
    np.testing.assert_array_equal(crv[list(IDS)], np.repeat(KEYS[:4], 2, axis=0))
    np.testing.assert_array_equal(pdg, [11, -11, 13, -13, 211, 22, 2212, -211])
    for name, table in (("tracker", tracker), ("crv", crv)):
        for field in ak.fields(source.collections[name]):
            np.testing.assert_array_equal(table[field], ak.to_numpy(ak.flatten(before[name][field])))
    for name in before:
        for field in ak.fields(before[name]):
            np.testing.assert_array_equal(ak.to_numpy(ak.flatten(before[name][field])),
                                          ak.to_numpy(ak.flatten(source.collections[name][field])))


def test_reused_donors_both_detectors_truth_ancestry_and_optional_calo():
    source = batch()
    calo = rows(("sp", "t", "e"), [1, 1, 1, 1, 0], sp=np.full(4, 9), t=np.arange(4.) + 3400)
    source = FrameBatch(source.event_ids, dict(source.collections, calo=calo))
    selection = FrameSelection(np.array([2, 1, 0, 0, 1]), np.array([0, 0, 1, 4]))
    mixed = overlay(source, [(source, selection)] * 4)
    tracker, crv, pdg = to_detector_inputs(mixed)
    np.testing.assert_array_equal(ak.to_numpy(mixed.collections["particles"].key[0]),
                                  np.ravel([[9 + 10 * i, 7 + 10 * i] for i in range(9)]))
    np.testing.assert_array_equal(ak.to_numpy(mixed.collections["particles"].parent[0]),
                                  np.ravel([[7 + 10 * i, -1] for i in range(9)]))
    np.testing.assert_array_equal(ak.to_numpy(mixed.collections["particles"].genidx[0]),
                                  np.ravel([[-1, i] for i in range(9)]))
    np.testing.assert_array_equal(ak.to_numpy(mixed.collections["gen"].idx[0]), np.arange(9))
    np.testing.assert_array_equal(tracker.sp[:9], np.arange(9) * 10 + 9)
    np.testing.assert_array_equal(crv.sp[:18], np.ravel([[7 + 10 * i, 9 + 10 * i] for i in range(9)]))
    np.testing.assert_array_equal(pdg[:18], np.tile([11, -11], 9))
    np.testing.assert_array_equal(tracker.t[:9], np.full(9, -2.))
    np.testing.assert_array_equal(crv.t0[:18], np.tile([-2., 1695.], 9))
    np.testing.assert_array_equal(crv.t1[:18], np.tile([-1., 1696.], 9))
    np.testing.assert_array_equal(ak.to_numpy(mixed.collections["calo"].sp[0]), tracker.sp[:9])
    assert len(mixed.event_ids) == 5 and ak.num(mixed.collections["crv"])[-1] == 0
    assert set(mixed.collections) == set(source.collections)
    mixed.validate_references()


@pytest.mark.parametrize("indices", [slice(0, 0), slice(4, 5)])
def test_zero_events_and_zero_step_events(indices):
    source = subset(batch(), indices)
    tracker, crv, pdg = to_detector_inputs(overlay(source, []))
    assert tracker.empty and crv.empty and pdg.shape == (0,) and pdg.dtype == np.int64
    assert list(tracker.columns) == [*IDS, *TRACKER_FIELDS]
    assert list(crv.columns) == [*IDS, *CRV_FIELDS]


@pytest.mark.parametrize("field,value", [("t0", np.inf), ("t1", -np.inf), ("edep", np.nan),
                                         ("x0", np.inf), ("p1", np.nan), ("len", -np.inf)])
def test_strict_crv_payload(field, value):
    source = batch()
    array = source.collections["crv"]
    values = ak.to_numpy(ak.flatten(array[field])).copy()
    values[0] = value
    array = ak.with_field(array, ak.unflatten(values, ak.num(array)), field)
    with pytest.raises(ValueError, match="nonfinite CRV|infinite endpoint"):
        to_detector_inputs(FrameBatch(source.event_ids, dict(source.collections, crv=array)))


def test_invalid_pdg_and_missing_detector_fields():
    source = batch()
    parts = ak.with_field(source.collections["particles"], source.collections["particles"].pdg + .5, "pdg")
    with pytest.raises(ValueError, match="particle pdg"):
        to_detector_inputs(FrameBatch(source.event_ids, dict(source.collections, particles=parts)))
    with pytest.raises(ValueError, match="missing tracker fields"):
        to_detector_inputs(FrameBatch(source.event_ids, dict(source.collections,
                          tracker=ak.without_field(source.collections["tracker"], "width"))))
    bad = ak.with_field(source.collections["crv"], source.collections["crv"].sp + 1, "sp")
    with pytest.raises(ValueError, match="dangling"):
        to_detector_inputs(FrameBatch(source.event_ids, dict(source.collections, crv=bad)))


def test_manifest_delegates_sampler_full_pools_and_reproducible_provenance(manifest, monkeypatch, tmp_path):
    path, config = manifest
    monkeypatch.chdir(tmp_path / "data")
    actual = load_manifest(path, max_events=2)
    mixed, tracker, crv, pdg, provenance = actual
    campaign = config["campaign"]
    rng = np.random.default_rng(config["seed"])
    law = BunchLaw.from_sdf(campaign["extended_mean"], campaign["sdf"], campaign["cut_max"])
    intensity = law.draw(rng, 2, beam_primary=True)
    rates = stock_rates(campaign["efficiencies"], stop_prescale=campaign["stop_prescale"])
    counts = draw_multiplicities(rng, intensity, rates)
    np.testing.assert_array_equal(provenance["intensity"], intensity)
    np.testing.assert_array_equal(provenance["multiplicities"], counts)
    np.testing.assert_array_equal(mixed.event_ids, KEYS[:2])
    assert provenance["pool_event_counts"] == dict.fromkeys(STREAMS, 5)
    assert provenance["signal_events_full"] == 5 and provenance["stream_order"] == STREAMS
    assert "not C++ RNG replay" in provenance["rng"]
    assert provenance["sources"]["signal"]["path"] == str(path.parent / "data/signal.root")
    source = read_frames(path.parent / "data/pool.root")
    backgrounds = []
    for column, stream in enumerate(STREAMS):
        selection, cursor = select_frames(rng, counts[:, column], 5, cursor=None,
                                          max_events_to_skip=campaign["max_skip"][stream])
        receipt = provenance["selections"][stream]
        np.testing.assert_array_equal(receipt["donors"], selection.donors)
        np.testing.assert_array_equal(receipt["source_event_ids"], KEYS[selection.donors])
        np.testing.assert_array_equal(receipt["event_index"], selection.event_index)
        np.testing.assert_array_equal(receipt["counts"], counts[:, column])
        assert receipt["next_cursor"] == cursor
        assert receipt["initial_cursor"] == (cursor - len(selection.donors)) % 5
        assert np.any(selection.donors >= 2)
        backgrounds.append((source, selection))
    expected = overlay(subset(source, slice(0, 2)), backgrounds)
    for name in expected.collections:
        for field in ak.fields(expected.collections[name]):
            np.testing.assert_array_equal(ak.to_numpy(ak.flatten(mixed.collections[name][field])),
                                          ak.to_numpy(ak.flatten(expected.collections[name][field])))
    etracker, ecrv, epdg = to_detector_inputs(expected)
    pd.testing.assert_frame_equal(tracker, etracker)
    pd.testing.assert_frame_equal(crv, ecrv)
    np.testing.assert_array_equal(pdg, epdg)
    again = load_manifest(path, max_events=2)
    pd.testing.assert_frame_equal(tracker, again[1])
    pd.testing.assert_frame_equal(crv, again[2])
    np.testing.assert_array_equal(pdg, again[3])
    np.testing.assert_array_equal(counts, again[4]["multiplicities"])


def test_manifest_zero_primary_events_still_loads_full_pools(manifest, monkeypatch):
    path, _ = manifest
    seen = []
    original = direct_frames.read_frames
    def read(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(len(result.event_ids))
        return result
    monkeypatch.setattr(direct_frames, "read_frames", read)
    mixed, tracker, crv, pdg, provenance = load_manifest(path, max_events=0)
    assert seen == [5] * 5
    assert len(mixed.event_ids) == 0 and tracker.empty and crv.empty and pdg.size == 0
    assert provenance["multiplicities"].shape == (0, 4)
    for stream in STREAMS:
        receipt = provenance["selections"][stream]
        assert receipt["source_event_ids"].shape == (0, 3)
        assert receipt["initial_cursor"] == receipt["next_cursor"]


def test_event_log_relative_path_and_missing_census(manifest):
    path, config = manifest
    source = batch()
    write_dump(path.parent / "data/pool.root", source, census=False)
    log = path.parent / "data/events.log"
    log.write_text("\n".join(f"Event: {i} run: {r} subRun: {s} event: {e}"
                              for i, (r, s, e) in enumerate(KEYS)))
    with pytest.raises(KeyError):
        load_manifest(path, max_events=0)
    for spec in config["pools"].values():
        spec["event_log"] = "data/events.log"
    save_manifest(path, config)
    result = load_manifest(path, max_events=1)
    assert result[4]["pool_event_counts"] == dict.fromkeys(STREAMS, 5)
    assert result[4]["sources"]["pools"][STREAMS[0]]["event_log"] == str(log)


@pytest.mark.parametrize("empty", [True, False])
def test_pool_requires_nonempty_census_not_steps(manifest, empty):
    path, _ = manifest
    source = subset(batch(), slice(0, 0) if empty else slice(4, 5))
    write_dump(path.parent / "data/pool.root", source)
    if empty:
        with pytest.raises(ValueError, match="nonempty event census"):
            load_manifest(path, max_events=0)
    else:
        result = load_manifest(path)
        assert result[4]["pool_event_counts"] == dict.fromkeys(STREAMS, 1)
        np.testing.assert_array_equal(result[0].event_ids, KEYS)
        assert len(result[1]) == 4 and len(result[2]) == 8


@pytest.mark.parametrize("key,value", [("mode", "randomReplace"), ("postmix_selection", "trigger"),
    ("additional_time_offset_ns", 1), ("additional_time_offset_ns", float("nan")),
    ("beam_primary", False), ("beam_primary", 1), ("early", True), ("early_prescales", {}),
    ("early_reduction", .01), ("typo", 0), ("extended_mean", "10"), ("sdf", 1),
    ("cut_min", 1000), ("stop_prescale", -1)])
def test_unsupported_campaign_fails_before_read(manifest, monkeypatch, key, value):
    path, config = manifest
    config["campaign"][key] = value
    save_manifest(path, config)
    monkeypatch.setattr(direct_frames, "read_frames", lambda *a, **k: pytest.fail("invalid config read a file"))
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize("section,key", [(None, "seed"), (None, "campaign"), (None, "signal"),
    ("campaign", "efficiencies"), ("campaign", "mode"), ("campaign", "postmix_selection"),
    ("campaign", "additional_time_offset_ns"), ("campaign", "max_skip"), ("campaign", "stop_prescale"),
    ("campaign", "extended_mean"), ("campaign", "sdf"), ("campaign", "cut_max"),
    ("campaign", "beam_primary"), ("pools", STREAMS[1])])
def test_missing_explicit_config(manifest, section, key):
    path, config = manifest
    del (config if section is None else config[section])[key]
    save_manifest(path, config)
    with pytest.raises(ValueError, match="missing"):
        load_manifest(path)


@pytest.mark.parametrize("mutation", ["factor_missing", "factor_extra", "factor_nan", "skip_missing",
    "skip_fractional", "skip_negative", "pool_extra", "source_extra", "source_empty", "version", "seed"])
def test_bad_manifest_nested_config(manifest, mutation):
    path, config = manifest
    campaign = config["campaign"]
    if mutation == "factor_missing": del campaign["efficiencies"]["MuBeamCat"]
    elif mutation == "factor_extra": campaign["efficiencies"]["EarlyMuBeamFlashCat"] = .1
    elif mutation == "factor_nan": campaign["efficiencies"]["MuBeamCat"] = float("nan")
    elif mutation == "skip_missing": del campaign["max_skip"][STREAMS[0]]
    elif mutation == "skip_fractional": campaign["max_skip"][STREAMS[0]] = .5
    elif mutation == "skip_negative": campaign["max_skip"][STREAMS[0]] = -1
    elif mutation == "pool_extra": config["pools"]["Early"] = config["signal"]
    elif mutation == "source_extra": config["signal"]["offset"] = 0
    elif mutation == "source_empty": config["signal"]["path"] = ""
    elif mutation == "version": config["version"] = True
    else: config["seed"] = -1
    save_manifest(path, config)
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_bad_max_events(manifest, limit):
    with pytest.raises(ValueError, match="max_events"):
        load_manifest(manifest[0], max_events=limit)


def native_products(source, tracker=True):
    """Native v2 CRV metadata, with optionally modern tracker pointer IDs."""
    sim = np.arange(len(source.event_ids), dtype=np.uint32) + 101
    gen = sim + 100
    per_row = lambda name, values: np.repeat(values, ak.to_numpy(ak.num(source.collections[name])))
    parts = flat(source, "particles")
    products = {
        "events": {"sim_product_id": sim, "gen_product_id": gen},
        "particles": {"sim_product_id": per_row("particles", sim),
                      "parent_product_id": np.where(parts["parent"] < 0, 0, per_row("particles", sim)),
                      "gen_product_id": np.where(parts["genidx"] < 0, 0, per_row("particles", gen))},
        "gen": {"gen_product_id": per_row("gen", gen)},
        "crv": {"sp_product_id": per_row("crv", sim)},
    }
    if tracker:
        products["tracker"] = {"sp_product_id": per_row("tracker", sim)}
    return products


@pytest.mark.parametrize("tracker_products", [True, False])
def test_native_products_join_full_ids_before_discard(manifest, tracker_products):
    path, _ = manifest
    source = batch()
    write_dump(path.parent / "data/signal.root", source,
               products=native_products(source, tracker=tracker_products))
    mixed, _, _, _, provenance = load_manifest(path, max_events=1)
    checks = provenance["source_fingerprints"]["signal"]["product_id_checks"]
    assert checks["crv"] == checks["parent"] == checks["generator"] == "checked"
    assert checks["particles_census"] == checks["gen_census"] == "checked"
    assert (checks["tracker"] == "checked") == tracker_products
    if not tracker_products:
        assert "unavailable" in checks["tracker"]
        assert "partial coverage" in checks["tracker"]
    assert ak.fields(mixed.collections["particles"]) == list(PARTICLE_FIELDS)
    assert ak.fields(mixed.collections["tracker"]) == list(TRACKER_FIELDS)


@pytest.mark.parametrize("name,field,index", [
    ("tracker", "sp_product_id", 3), ("crv", "sp_product_id", 6),
    ("particles", "parent_product_id", 6), ("particles", "gen_product_id", 7),
    ("events", "sim_product_id", 3), ("events", "gen_product_id", 3)])
def test_native_product_mismatch_rejected_even_outside_selection(manifest, name, field, index):
    path, _ = manifest
    source = batch()
    products = native_products(source)
    products[name][field][index] += 1
    write_dump(path.parent / "data/pool.root", source, products=products)
    with pytest.raises(ValueError, match="native product ID/key mismatch"):
        load_manifest(path, max_events=0)


@pytest.mark.parametrize("name,field", [("particles", "sim_product_id"), ("gen", "gen_product_id")])
def test_partial_native_product_targets_rejected(manifest, name, field):
    path, _ = manifest
    source = batch()
    products = native_products(source)
    del products[name][field]
    write_dump(path.parent / "data/signal.root", source, products=products)
    with pytest.raises(ValueError, match="partial native product data"):
        load_manifest(path, max_events=0)


@pytest.mark.parametrize("name,field,index,value", [
    ("tracker", "sp_product_id", 0, 0), ("crv", "sp_product_id", 0, 0),
    ("particles", "parent_product_id", 0, 0), ("particles", "gen_product_id", 1, 0),
    ("particles", "parent_product_id", 1, 101), ("particles", "gen_product_id", 0, 201),
    ("particles", "sim_product_id", 0, -1), ("gen", "gen_product_id", 0, .5)])
def test_native_product_values_and_null_links(manifest, name, field, index, value):
    path, _ = manifest
    source = batch()
    products = native_products(source)
    products[name][field] = products[name][field].astype(float if value == .5 else np.int64)
    products[name][field][index] = value
    write_dump(path.parent / "data/signal.root", source, products=products)
    with pytest.raises(ValueError, match="native product ID"):
        load_manifest(path, max_events=0)


@pytest.mark.parametrize("indices", [slice(0, 0), slice(4, 5)])
def test_native_empty_signal_and_empty_collection_metadata(manifest, indices):
    path, _ = manifest
    source = subset(batch(), indices)
    write_dump(path.parent / "data/signal.root", source, products=native_products(source))
    result = load_manifest(path, max_events=0)
    assert result[4]["estimated_selected_rows"] == dict.fromkeys(batch().collections, 0)
    assert result[4]["actual_rows"] == result[4]["estimated_selected_rows"]


def test_fingerprints_and_selected_row_estimates(manifest, monkeypatch):
    path, _ = manifest
    expected_source = hashlib.sha256((path.parent / "data/signal.root").read_bytes()).hexdigest()
    expected_pool = hashlib.sha256((path.parent / "data/pool.root").read_bytes()).hexdigest()
    expected_census = hashlib.sha256(KEYS.astype("<u8").tobytes()).hexdigest()
    monkeypatch.setattr(type(path), "read_bytes", lambda *a: pytest.fail("source hashing must stream"))
    mixed, tracker, crv, _, provenance = load_manifest(path, max_events=2)
    fingerprints = provenance["source_fingerprints"]
    assert fingerprints["signal"]["source_sha256"] == expected_source
    assert fingerprints["signal"]["census_sha256"] == expected_census
    assert all("unavailable" in check for check in fingerprints["signal"]["product_id_checks"].values())
    for stream in STREAMS:
        assert fingerprints["pools"][stream]["source_sha256"] == expected_pool
        assert fingerprints["pools"][stream]["census_sha256"] == expected_census
    source = batch()
    estimated = {name: int(ak.sum(ak.num(rows[:2]))) for name, rows in source.collections.items()}
    for selection in provenance["selections"].values():
        for name, rows in source.collections.items():
            estimated[name] += int(ak.sum(ak.num(rows)[selection["donors"]]))
    assert provenance["estimated_selected_rows"] == estimated == provenance["actual_rows"]
    assert estimated["tracker"] == len(tracker) and estimated["crv"] == len(crv)
    assert estimated == {name: int(ak.sum(ak.num(rows))) for name, rows in mixed.collections.items()}
    assert set(provenance["timing_s"]) == {"read_validate", "sample", "overlay_validate", "adapter_validate"}
    assert all(value >= 0 for value in provenance["timing_s"].values())


def test_event_log_fingerprint_uses_full_ordered_census(manifest):
    path, config = manifest
    source = batch()
    write_dump(path.parent / "data/signal.root", source, census=False)
    log = path.parent / "data/signal.log"
    reordered = KEYS[::-1]
    text = "\n".join(f"Event: {i} run: {r} subRun: {s} event: {e}" for i, (r, s, e) in enumerate(reordered))
    log.write_text(text)
    config["signal"]["event_log"] = "data/signal.log"
    save_manifest(path, config)
    first = load_manifest(path, max_events=0)[4]["source_fingerprints"]["signal"]
    assert first["event_log_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert first["census_sha256"] == hashlib.sha256(reordered.astype("<u8").tobytes()).hexdigest()
    # A path-preserving edit is reflected in both the log and census receipts.
    log.write_text(text + "\nEvent: 5 run: 8 subRun: 8 event: 8\n")
    second = load_manifest(path, max_events=0)[4]["source_fingerprints"]["signal"]
    assert first["source_sha256"] == second["source_sha256"]
    assert first["event_log_sha256"] != second["event_log_sha256"]
    assert first["census_sha256"] != second["census_sha256"]


def test_explicit_event_log_does_not_bypass_native_identity_checks(manifest):
    path, config = manifest
    source = batch()
    products = native_products(source)
    products["events"]["sim_product_id"][0] += 1
    write_dump(path.parent / "data/signal.root", source, products=products)
    log = path.parent / "data/signal.log"
    log.write_text("\n".join(f"Event: {i} run: {r} subRun: {s} event: {e}" for i, (r, s, e) in enumerate(KEYS)))
    config["signal"]["event_log"] = "data/signal.log"
    save_manifest(path, config)
    with pytest.raises(ValueError, match="native product ID/key mismatch for particles_census"):
        load_manifest(path, max_events=0)


def test_source_byte_change_updates_runtime_fingerprint(manifest):
    path, _ = manifest
    first = load_manifest(path, max_events=0)[4]["source_fingerprints"]["signal"]
    # Extra metadata is ignored by FrameBatch but is part of the exact source.
    write_dump(path.parent / "data/signal.root", batch(),
               products={"events": {"metadata": np.arange(len(KEYS))}})
    second = load_manifest(path, max_events=0)[4]["source_fingerprints"]["signal"]
    assert first["source_sha256"] != second["source_sha256"]
    assert first["census_sha256"] == second["census_sha256"]


def test_pointer_metadata_without_particle_identity_fails_before_read_frames(manifest, monkeypatch):
    path, _ = manifest
    source = batch()
    products = {"tracker": native_products(source)["tracker"]}
    write_dump(path.parent / "data/signal.root", source, products=products)
    monkeypatch.setattr(direct_frames, "read_frames", lambda *a, **k: pytest.fail("metadata already discarded"))
    with pytest.raises(ValueError, match="partial native product data for tracker"):
        load_manifest(path, max_events=0)


def test_legacy_census_and_modern_pointer_data_without_native_census_identity(manifest):
    path, _ = manifest
    source = batch()
    products = native_products(source)
    del products["events"]
    write_dump(path.parent / "data/signal.root", source, products=products)
    checks = load_manifest(path, max_events=0)[4]["source_fingerprints"]["signal"]["product_id_checks"]
    assert checks == dict.fromkeys(("tracker", "crv", "parent", "generator"), "checked")


@pytest.mark.parametrize("name,field,index", [
    ("tracker", "sp_product_id", 0), ("crv", "sp_product_id", 0),
    ("particles", "parent_product_id", 0), ("particles", "gen_product_id", 1)])
def test_native_product_cannot_resolve_matching_key_in_another_full_event(manifest, name, field, index):
    path, _ = manifest
    source = batch()
    products = native_products(source)
    # The next event has exactly this product ID and key/index, but a different
    # run (with identical subrun/event). Only a full-ID join rejects it.
    products[name][field][index] += 1
    write_dump(path.parent / "data/signal.root", source, products=products)
    with pytest.raises(ValueError, match="native product ID/key mismatch"):
        load_manifest(path, max_events=0)


@pytest.mark.parametrize("field,label", [("parent_product_id", "parent"), ("gen_product_id", "generator")])
def test_absent_optional_ancestry_products_are_reported_not_assumed(manifest, field, label):
    path, _ = manifest
    source = batch()
    products = native_products(source)
    del products["particles"][field]
    write_dump(path.parent / "data/signal.root", source, products=products)
    checks = load_manifest(path, max_events=0)[4]["source_fingerprints"]["signal"]["product_id_checks"]
    assert "unavailable" in checks[label] and "partial coverage" in checks[label]
    assert checks["tracker"] == checks["crv"] == "checked"


@pytest.mark.parametrize("pdg", [np.array([2**63], dtype=np.uint64), np.array([2.0**63]), np.array([-2.0**64])])
def test_pdg_out_of_int64_rejected_before_cast(pdg):
    source = batch()
    parts = source.collections["particles"]
    source.collections["particles"] = ak.with_field(parts, ak.unflatten(np.repeat(pdg, 8), [2, 2, 2, 2, 0]), "pdg")
    with pytest.raises(ValueError, match="exceeds int64"):
        to_detector_inputs(source)
