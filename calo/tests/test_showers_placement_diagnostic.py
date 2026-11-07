"""Frozen placement diagnostics: saved replay and CSS accounting, never reconstruction."""
import builtins
import hashlib
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from calofastsim.showers import placement_diagnostic as pd
from calofastsim.showers.export import EVENT, keys, read
from calofastsim.showers.library import save
from test_showers_production_compare import case  # noqa: F401 -- shared ROOT fixture
from test_showers_response import source


SEED = 20261105


def frozen(case):
    """Construct the documented saved comparison arrays without running reconstruction."""
    entries, _, geometry, _, _ = read(case["path"], input_mode="primary")
    draw = pd.sample(case["library"], entries, np.random.default_rng(SEED), orientation="disk")
    ids = keys(case["data"]["events"], EVENT)
    saved = dict(full_ids=ids, event_row=np.arange(len(ids)), query_event_rows=entries["event_row"])
    saved.update({"entry_" + k: v for k, v in entries.items()})
    saved.update({"geometry_" + k: v for k, v in geometry.items()})
    saved.update({"sampling_" + k: draw[k] for k in ("donor", "supported", "fallback", "offsets")})
    # The non-query row must remain in the archive, but never enter query diagnostics.
    for arm in ("stockref", "control", *pd.ARMS):
        saved[arm + "_energy"] = np.array([20., 0., 999.])
    return saved


def replay(case, saved):
    return pd.reproduce(case["path"], case["library"], case["metadata"], saved, SEED)


@pytest.mark.parametrize("support", ["normal", "one_unsupported", "all_unsupported"])
def test_reproduce_preserves_full_rows_positive_empty_and_unsupported(case, support):
    if support == "one_unsupported":
        case["library"]["entry_disk"][1] = 2
    elif support == "all_unsupported":
        case["library"]["entry_disk"][:] = 2
    saved = frozen(case)
    with patch.object(pd, "sample", wraps=pd.sample) as sample, \
         patch.object(pd, "place", wraps=pd.place) as geometric, \
         patch.object(pd, "place_crystals", wraps=pd.place_crystals) as crystal:
        entries, original, draw, placed, audit = replay(case, saved)
    assert sample.call_count == crystal.call_count == 1
    assert geometric.call_count == 2  # original CSS round-trip plus one shared draw
    assert sample.call_args.kwargs["orientation"] == "disk"
    assert geometric.call_args.args[1] is crystal.call_args.args[2] is draw
    np.testing.assert_array_equal(saved["event_row"], [0, 1, 2])
    np.testing.assert_array_equal(entries["event_row"], [0, 1])
    np.testing.assert_array_equal(pd.energy_sum(original, 2), [68., 0.])
    np.testing.assert_array_equal(original["disk"], [0, 1])
    np.testing.assert_array_equal(pd.place(case["library"], original)["crystal"], original["crystal"])
    expected_support = {"normal": [True, True], "one_unsupported": [True, False],
                        "all_unsupported": [False, False]}[support]
    np.testing.assert_array_equal(draw["supported"], expected_support)
    np.testing.assert_array_equal(draw["donor"] < 0, ~np.asarray(expected_support))
    np.testing.assert_array_equal(pd.energy_sum(draw, 2), [0. if support == "all_unsupported" else 68., 0.])
    for name in ("donor", "supported", "fallback", "offsets"):
        np.testing.assert_array_equal(draw[name], saved["sampling_" + name])
    result, arrays = pd.diagnose(case["library"], entries, original, draw, placed, saved)
    sampling = result["strata"]["all"]["sampling"]
    assert sampling["unsupported"] == expected_support.count(False)
    assert sampling["empty"] == (support == "normal")
    assert result["strata"]["all"]["entries"] == 2
    assert result["strata"]["all"]["arms"]["geometric"]["selection"]["reco"]["stock_selected"] == 1
    np.testing.assert_array_equal(arrays["event_row"], [0, 1])
    assert audit["census"]["events"] == 3


@pytest.mark.parametrize("family", ["full_ids", "event_row", "query_event_rows", "entry_", "geometry_", "sampling_"])
def test_reproduce_rejects_same_size_stale_saved_arrays(case, family):
    saved = frozen(case)
    names = [k for k in saved if k.startswith(family)] if family.endswith("_") else [family]
    assert names
    for name in names:
        changed = saved[name].copy()
        if changed.dtype.names:
            changed[changed.dtype.names[-1]][0] += 1
        elif changed.dtype.kind == "b":
            changed.flat[0] = not changed.flat[0]
        else:
            changed.flat[0] += 1
        with pytest.raises(ValueError, match="saved comparison mismatch: " + name):
            replay(case, dict(saved, **{name: changed}))


@pytest.mark.parametrize("arm", ["stockref", "control", "geometric", "crystal"])
@pytest.mark.parametrize("bad", ["short", "negative", "nan", "inf"])
def test_reproduce_rejects_invalid_full_population_leading_energy(case, arm, bad):
    saved = frozen(case)
    energy = saved[arm + "_energy"].copy()
    if bad == "short":
        energy = energy[:-1]
    else:
        # Even a non-query row is subject to the saved full-population guard.
        energy[-1] = {"negative": -1., "nan": np.nan, "inf": np.inf}[bad]
    saved[arm + "_energy"] = energy
    with pytest.raises(ValueError, match="invalid saved leading energy: " + arm):
        replay(case, saved)


@pytest.mark.parametrize("fault", ["unmapped", "wrong_crystal"])
def test_original_css_must_roundtrip_to_same_crystal(case, fault):
    saved = frozen(case)
    actual_place = pd.place

    def corrupt_original(library, steps):
        result = actual_place(library, steps)
        if fault == "unmapped":
            result["mapped"][0] = False
        else:
            result["crystal"][0] += 1
        return result

    with patch.object(pd, "place", side_effect=corrupt_original), \
         pytest.raises(ValueError, match="original CSS does not round-trip"):
        replay(case, saved)


def test_reproduce_still_checks_library_geometry_and_provenance(case):
    saved = frozen(case)
    case["library"]["geometry_crystal_position_x"] = case["library"]["geometry_crystal_position_x"] + 1.
    with pytest.raises(ValueError, match="geometry"):
        replay(case, saved)
    case["library"]["geometry_crystal_position_x"] -= 1.
    case["metadata"]["input"] = str(case["path"])
    with pytest.raises(ValueError, match="comparison file"):
        replay(case, saved)


@pytest.fixture
def synthetic():
    """Two-disk deposits with every exclusive loss category and zero-step query rows."""
    library, entry, _ = source()
    n = 5
    entries = {k: np.repeat(v, n, axis=0) for k, v in entry.items()}
    entries["event_row"] = np.array([2, 101, 205, 307, 409])
    entries["radius"] = np.array([449., 450., 550., 1000., 500.])
    entries["disk"] = np.array([0, 0, 1, 1, 1])
    entries["position"][:, 0] = entries["radius"]
    # A nonzero shared transverse origin exposes accidental use of global radius.
    origin = np.array([100., -50.])
    entries["position"][:, :2] += origin
    for a, shift in zip("xy", origin):
        library["geometry_disk_origin_" + a] += shift
        library["geometry_crystal_position_" + a] += shift
        library["entry_position"][:, "xy".index(a)] += shift
    library["geometry_disk_size_x"][:] = 2.
    # Inner and outer also violate axial: radial reasons must win.
    positions = np.array([[0., 0., 99.], [21., 0., 121.], [5., 0., 99.],
                          [5., 0., 110.], [5., 0., 110.], [5., 0., 110.],
                          [5., 0., 110.], [5., 0., 110.], [5., 0., 210.],
                          [5., 0., 210.]])
    positions[:, :2] += origin
    row = np.array([0] * 9 + [1])
    energy = np.array([1., 2., 3., 4., 5., 6., 7., 11., 13., 10.])
    draw = dict(entry=row, energy_birks=energy, energy_g4=energy * 10.,
                position=positions, time=np.arange(len(row), dtype=float),
                disk=np.array([0] * 8 + [1, 1]), crystal=np.array([10] * 8 + [11, 11]),
                donor=np.array([0, 0, 0, -1, 0]), supported=np.array([True, True, True, False, True]),
                fallback=np.array([False, True, False, True, False]), offsets=np.array([0, 9, 10, 10, 10, 10]))
    original = dict(draw, energy_birks=energy * 2.)
    mapped = np.array([False] * 7 + [True] * 3)
    placed = {arm: dict(draw, mapped=mapped.copy()) for arm in pd.ARMS}
    placed["crystal"]["crystal_loss_reason"] = np.array([3, 2, 1, 1, 2, 3, 0, 0, 0, 0])
    saved = {arm + "_energy": np.full(410, 999.) for arm in ("stockref", "control", *pd.ARMS)}
    for value in saved.values():
        value[entries["event_row"]] = [20., 10., 0., 0., 0.]
    saved["geometric_energy"][entries["event_row"]] = [0., 11., 0., 0., 0.]
    return library, entries, original, draw, placed, saved


def test_diagnose_whole_event_birks_balance_and_exclusive_losses(synthetic):
    library, entries, original, draw, placed, saved = synthetic
    report, arrays = pd.diagnose(*synthetic)
    np.testing.assert_array_equal(arrays["event_row"], entries["event_row"])
    np.testing.assert_array_equal(arrays["donor"], [52., 10., 0., 0., 0.])
    np.testing.assert_array_equal(arrays["original"], [104., 20., 0., 0., 0.])
    for arm in pd.ARMS:
        np.testing.assert_array_equal(arrays[arm + "_mapped"], [24., 10., 0., 0., 0.])
        np.testing.assert_array_equal(arrays[arm + "_lost"], [28., 0., 0., 0., 0.])
        np.testing.assert_allclose(arrays[arm + "_mapped"] - arrays["original"],
                                   arrays["donor"] - arrays["original"] - arrays[arm + "_lost"])
        loss = np.stack([arrays[arm + "_loss_" + k] for k in pd.LOSS])
        np.testing.assert_array_equal(loss.sum(axis=0), arrays[arm + "_lost"])
        np.testing.assert_array_equal(loss[:, 0], [1, 2, 3, 22, 0, 0, 0, 0] if arm == "geometric"
                                      else [1, 2, 3, 0, 4, 5, 6, 7])
        stats = report["strata"]["all"]["arms"][arm]
        assert stats["placement_loss"]["fast"] == pytest.approx(28 / 5)
        assert stats["lost_fraction_of_donor"]["ratio"] == pytest.approx(28 / 62)
    assert report["strata"]["all"]["sampling"] == dict(unsupported=1, fallback=2, empty=2)
    assert report["strata"]["all"]["donor_features"]["energy_delta"]["count"] == 4
    selection = report["strata"]["all"]["arms"]["geometric"]["selection"]
    assert selection["reco"]["misses"] == selection["reco"]["gains"] == 1
    # A reconstruction miss is not evidence of crossing the CSS placement threshold.
    assert selection["css_gate"]["donor_to_mapped_down"] == 0
    assert selection["reco"]["misses_with_css_placement_gate_down"] == 0
    assert report["shapes"]["all"]["common_css_gate_entries"] == 1


def test_diagnose_rejects_per_query_energy_change_even_if_total_preserved(synthetic):
    placed = synthetic[4]["geometric"]
    placed["energy_birks"] = placed["energy_birks"].copy()
    placed["energy_birks"][0] += 1.
    placed["energy_birks"][-1] -= 1.
    with pytest.raises(ValueError, match="per-query energy balance"):
        pd.diagnose(*synthetic)


@pytest.mark.parametrize("arm", pd.ARMS)
def test_loss_categories_use_raw_not_remapped_positions_and_mapped_wins(synthetic, arm):
    library, _, _, draw, placed, _ = synthetic
    placed[arm]["position"] = np.full_like(draw["position"], 10000.)
    reason = pd.loss_categories(library, draw, placed[arm], arm)
    np.testing.assert_array_equal(reason, [0, 1, 2, 3, 3, 3, 3, -1, -1, -1] if arm == "geometric"
                                  else [0, 1, 2, 4, 5, 6, 7, -1, -1, -1])
    placed[arm]["mapped"][0] = True
    assert pd.loss_categories(library, draw, placed[arm], arm)[0] == -1


def test_radial_strata_boundaries_and_disk_partition(synthetic):
    masks = pd.strata(synthetic[1])
    for name, rows in [("all", [0, 1, 2, 3, 4]), ("radius_0_450", [0]),
                       ("radius_450_550", [1, 4]), ("radius_550_1000", [2]),
                       ("radius_1000_inf", [3]), ("disk0", [0, 1]), ("disk1", [2, 3, 4])]:
        np.testing.assert_array_equal(np.flatnonzero(masks[name]), rows)
    np.testing.assert_array_equal(sum(v.astype(int) for k, v in masks.items() if k.startswith("radius_")), 1)


@pytest.mark.parametrize("explicit_support", [False, True])
def test_css_threshold_is_strict_and_reco_misses_are_only_associations(explicit_support):
    original = np.array([20., 20., 20., 10., 0., 0.])
    donor = np.array([0., 10., 12., 12., 0., 0.])
    mapped = np.array([0., 10., 9., 12., 0., 0.])
    stock = np.array([20., 20., 20., 10., 20., 0.])
    fast = np.array([0., 10., 0., 11., 0., 0.])
    kwargs = {"supported": np.array([True, True, True, True, False, True])} if explicit_support else {}
    result = pd.selection_accounting(original, donor, mapped, donor - mapped, stock, fast, stock, **kwargs)
    assert result["css_gate"] == dict(original=3, donor=2, mapped=1, donor_to_mapped_down=1,
                                       source_to_donor_down=2, source_to_donor_up=1)
    reco = result["reco"]
    assert (reco["stock_selected"], reco["fast_selected"], reco["misses"], reco["gains"], reco["net_deficit"]) == (4, 1, 4, 1, 3)
    assert reco["miss_partition"] == dict(no_donor_energy=1 if explicit_support else 2,
        unsupported=int(explicit_support), positive_donor_no_loss=1, positive_donor_with_loss=1)
    assert sum(reco["miss_partition"].values()) == reco["misses"]
    assert reco["control_miss_partition"] == reco["miss_partition"]
    assert reco["misses_with_css_placement_gate_down"] == 1


def test_centroid_relative_to_query_and_nonzero_origin_with_both_disks():
    _, entries, steps = source()
    entries = {k: np.repeat(v, 2, axis=0) for k, v in entries.items()}
    origin = np.array([100., -50.])
    entries["position"][:] = [103., -46., 100.]
    steps["position"] = np.array([[105., -45., 110.], [109., -41., 210.]])
    steps["energy_birks"] = np.array([1., 3.])
    centroid = pd.centroid_coordinates(steps, entries, origin)
    # Weighted transverse delta is (5, 4), radial basis (3/5, 4/5).
    assert centroid["radial"][0] == pytest.approx(6.2)
    assert centroid["tangent"][0] == pytest.approx(-1.6)
    np.testing.assert_array_equal(centroid["energy"], [4., 0.])
    assert np.isnan(centroid["radial"][1]) and np.isnan(centroid["tangent"][1])
    accepted = pd.centroid_coordinates(steps, entries, origin, np.array([True, False]))
    assert accepted["radial"][0] == pytest.approx(2.)
    assert accepted["tangent"][0] == pytest.approx(-1.)
    entries["position"][0, :2] = origin
    centered = pd.centroid_coordinates(steps, entries, origin)
    assert centered["radial"][0] == pytest.approx(8.)
    assert centered["tangent"][0] == pytest.approx(8.)


def test_empty_strata_are_strict_json_null_not_nonfinite(case, tmp_path):
    saved = frozen(case)
    entries, original, draw, placed, _ = replay(case, saved)
    result, _ = pd.diagnose(case["library"], entries, original, draw, placed, saved)
    empty = result["strata"]["radius_550_1000"]
    assert empty["entries"] == 0
    assert empty["sampling"] == dict(unsupported=0, fallback=0, empty=0)
    assert empty["donor_features"]["angle_deg"] == dict(count=0, mean=None, quantiles=None)
    for arm in pd.ARMS:
        assert empty["arms"][arm]["placement_loss"] == dict(fast=None, stock=None, difference=None, error=None)
        assert empty["arms"][arm]["lost_fraction_of_donor"] == dict(ratio=None, error=None)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    path = tmp_path / "strict.json"
    pd._json(path, result)
    assert json.loads(path.read_text()) == result
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_cli_saved_only_wiring_outputs_and_no_reconstruction(case, tmp_path, monkeypatch, capsys):
    saved = frozen(case)
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    np.savez_compressed(comparison / "clusters.npz", **saved)
    library_path = tmp_path / "library.npz"
    save(library_path, case["library"], case["metadata"])
    out = tmp_path / "diagnostic"
    monkeypatch.setattr(sys, "argv", ["placement_diagnostic", "--tables", str(case["path"]),
        "--library", str(library_path), "--comparison", str(comparison), "--out", str(out), "--seed", str(SEED)])
    actual_import = builtins.__import__

    def no_reconstruction(name, globals=None, locals=None, fromlist=(), level=0):
        forbidden = {"chain", "reconstruct", "reconstruction"}
        assert not (forbidden & set(name.split("."))), "diagnostic attempted reconstruction import"
        if name.startswith("calofastsim") or (globals or {}).get("__package__", "").startswith("calofastsim"):
            assert not (forbidden & set(fromlist or ())), "diagnostic attempted reconstruction import"
        return actual_import(name, globals, locals, fromlist, level)

    with patch.object(pd, "load", wraps=pd.load) as load, \
         patch.object(pd, "reproduce", wraps=pd.reproduce) as reproduce, \
         patch.object(pd, "diagnose", wraps=pd.diagnose) as diagnose, \
         patch.object(pd, "plot_report") as plot, \
         patch.object(builtins, "__import__", side_effect=no_reconstruction):
        pd.main()
    load.assert_called_once_with(library_path)
    assert reproduce.call_count == diagnose.call_count == 1
    assert reproduce.call_args.args[-1] == SEED
    result = json.loads((out / "numbers.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert summary == {k: v for k, v in result.items() if k != "shapes"}
    assert json.loads(capsys.readouterr().out) == summary
    assert result["seed"] == SEED and result["audit"]["census"]["events"] == 3
    assert "Not a decomposition of leading-reco energy" in result["scope"]
    assert result["inputs"]["clusters"]["sha256"] == hashlib.sha256((comparison / "clusters.npz").read_bytes()).hexdigest()
    with np.load(out / "arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["event_row"], [0, 1])
        np.testing.assert_array_equal(arrays["original"], [68., 0.])
    plot.assert_called_once()
    assert plot.call_args.args == (result, out / "energy_balance.png")


def test_diagnose_reco_partition_distinguishes_empty_and_unsupported(synthetic):
    _, entries, _, _, _, saved = synthetic
    for arm in ("stockref", "control"):
        saved[arm + "_energy"][entries["event_row"][[2, 3]]] = 20.
    report, _ = pd.diagnose(*synthetic)
    reco = report["strata"]["all"]["arms"]["geometric"]["selection"]["reco"]
    assert reco["miss_partition"] == dict(no_donor_energy=1, unsupported=1,
        positive_donor_no_loss=0, positive_donor_with_loss=1)
    assert sum(reco["miss_partition"].values()) == reco["misses"] == 3
    assert reco["control_miss_partition"] == reco["miss_partition"]


@pytest.mark.parametrize("fault", ["different_origins", "rotated_axes"])
def test_diagnose_rejects_unsupported_centroid_geometry(synthetic, fault):
    library = synthetic[0]
    if fault == "different_origins":
        library["geometry_disk_origin_x"][1] += 1.
    else:
        library["geometry_disk_axis_x_x"][:] = -1.
        library["geometry_disk_axis_y_y"][:] = -1.
    with pytest.raises(ValueError, match="common transverse origins and aligned disk axes"):
        pd.diagnose(*synthetic)


def test_donor_features_rotate_about_disk_origin_and_exclude_unsupported():
    library, entries, _ = source()
    origin = np.array([100., -50.])
    for i, axis in enumerate("xy"):
        library["geometry_disk_origin_" + axis][:] = origin[i]
    library["entry_position"][0] = [105., -50., 100.]
    library["entry_direction"][0] = [.6, 0., .8]
    entries = {k: np.repeat(v, 2, axis=0) for k, v in entries.items()}
    entries["position"][:] = [100., -45., 100.]
    entries["direction"][:] = [0., .6, .8]
    entries["energy"][:] = 90.
    entries["radius"][:] = 4.
    entries["cosine"][:] = .8
    features = pd.donor_features(library, entries,
        dict(donor=np.array([0, -1]), supported=np.array([True, False])))
    assert features["angle_deg"][0] == pytest.approx(0., abs=1e-6)
    assert features["radius_delta"][0] == 1.
    assert features["energy_delta"][0] == 10.
    assert features["cosine_delta"][0] == pytest.approx(.2)
    assert all(np.isnan(v[1]) for v in features.values())


@pytest.mark.parametrize("empty_inner", [False, True])
def test_plot_report_accepts_empty_radial_strata(case, tmp_path, empty_inner):
    saved = frozen(case)
    entries, original, draw, placed, _ = replay(case, saved)
    if empty_inner:
        entries["radius"][:] = [450., 550.]
    result, _ = pd.diagnose(case["library"], entries, original, draw, placed, saved)
    path = tmp_path / "empty_strata.png"
    pd.plot_report(result, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_candidate_box_distance_front_depth_and_sparse_two_disks():
    library, _, steps = source()
    steps["position"] = np.array([[13., 14., 125.], [5., 0., 199.]])
    distance = pd.candidate_box_distance(library, steps)
    np.testing.assert_allclose(distance, [np.sqrt(50.), 1.])
    steps["position"] = np.array([[0., 0., 110.], [0., 0., 210.]])
    np.testing.assert_array_equal(pd.candidate_box_distance(library, steps), [0., 0.])


def test_unknown_crystal_guard_is_not_postmap(synthetic):
    library, _, _, draw, placed, _ = synthetic
    placed["crystal"]["crystal_loss_reason"][0] = 4
    with pytest.raises(ValueError, match="unknown crystal loss reason"):
        pd.loss_categories(library, draw, placed["crystal"], "crystal")


def test_loss_partition_guard(synthetic):
    with patch.object(pd, "loss_categories", return_value=np.full(10, 100)), \
         pytest.raises(ValueError, match="loss categories do not partition"):
        pd.diagnose(*synthetic)


def test_invalid_candidate_distance_guard(synthetic):
    with patch.object(pd, "candidate_box_distance", return_value=np.full(10, np.nan)), \
         pytest.raises(ValueError, match="invalid candidate box distance"):
        pd.diagnose(*synthetic)


@pytest.mark.parametrize("field", EVENT)
def test_reproduce_full_identity_includes_run_subrun_and_event(case, field):
    saved = frozen(case)
    saved["full_ids"] = saved["full_ids"].copy()
    saved["full_ids"][field][-1] += 1
    with pytest.raises(ValueError, match="saved comparison mismatch: full_ids"):
        replay(case, saved)
