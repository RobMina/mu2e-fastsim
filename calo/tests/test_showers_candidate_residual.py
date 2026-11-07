"""Synthetic saved-only residual diagnosis: no draws, placement, reconstruction or input reads."""
import builtins
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from calofastsim.showers import candidate_residual as cr


@pytest.fixture(autouse=True)
def forbid_physics_and_input(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved-only residual tests forbid RNG, physics execution and input reads")

    for name in ("default_rng", "RandomState", "Generator", "SeedSequence", "seed", "random", "rand", "randn", "choice", "uniform", "normal", "randint"):
        monkeypatch.setattr(np.random, name, forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    actual_open, actual_path_open = builtins.open, Path.open

    def guarded_open(path, mode="r", *args, **kwargs):
        if "r" in mode and Path(path).suffix not in (".py", ".pyc", ".so", ".dylib"):
            forbidden()
        return actual_open(path, mode, *args, **kwargs)

    def guarded_path_open(path, mode="r", *args, **kwargs):
        if "r" in mode and path.suffix not in (".py", ".pyc", ".so", ".dylib"):
            forbidden()
        return actual_path_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    for name in ("load", "loadtxt", "genfromtxt", "fromfile"):
        monkeypatch.setattr(np, name, forbidden)
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("calofastsim") and module is not None:
            for name in ("sample", "build", "prepare", "reproduce", "place", "place_crystals", "run_chain",
                         "reconstruct", "from_cechain", "read", "load", "response_inputs", "pinned_json", "pinned_npz"):
                if callable(getattr(module, name, None)):
                    monkeypatch.setattr(module, name, forbidden)
    if "uproot" in sys.modules:
        monkeypatch.setattr(sys.modules["uproot"], "open", forbidden)
    actual_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        assert not ({"chain", "reconstruct", "handoff"} & (set(name.split(".")) | set(fromlist or ()))), "physics import forbidden"
        return actual_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def flow_counts(values, edges):
    """Independent numpy histogram oracle; the final edge is in the last interior bin."""
    values = np.asarray(values)
    return np.r_[np.count_nonzero(values < edges[0]), np.histogram(values, edges)[0],
                 np.count_nonzero(values > edges[-1])]


def delete_block_shape_error(fast, stock, group, edges, fm, sm):
    leave = []
    for block in np.unique(group):
        keep = group != block
        f, s = flow_counts(fast[fm & keep], edges), flow_counts(stock[sm & keep], edges)
        if not f.sum() or not s.sum():
            return np.full(len(edges) + 1, np.nan)
        leave.append(f / f.sum() - s / s.sum())
    leave = np.array(leave)
    return np.sqrt((len(leave) - 1) / len(leave) * ((leave - leave.mean(axis=0)) ** 2).sum(axis=0))


def test_hist_own_flow_conservation_and_selected_denominator_delete_block_oracle():
    edges = np.array([0., 1., 2.])
    fast = np.array([-1., 0., 1., 2., 3., np.nan, .5, 1.5, np.nan, np.nan])
    stock = np.array([.5, np.nan, 2., 4., -2., 0., np.nan, .5, np.nan, np.nan])
    fm, sm = np.isfinite(fast), np.isfinite(stock)
    # Last original-source block has no selected queries in either arm; it still belongs to the jackknife universe.
    group = np.array([0, 0, 1, 1, 1, 3, 3, 3, 9, 9])
    result = cr.hist_own(fast, stock, group, edges, fm, sm)
    np.testing.assert_array_equal(result["fast"], [1, 2, 3, 1])
    np.testing.assert_array_equal(result["stock"], [1, 3, 1, 1])
    assert result["fast_selected"] == 7 and result["stock_selected"] == 6
    assert result["flow_bins"] is True
    np.testing.assert_allclose(result["fraction_difference"], np.array([1, 2, 3, 1]) / 7 - np.array([1, 3, 1, 1]) / 6)
    assert result["fraction_difference"].sum() == pytest.approx(0)
    expected = delete_block_shape_error(fast, stock, group, edges, fm, sm)
    np.testing.assert_allclose(result["fraction_error"], expected)
    # A compact selected-only grouping must not accidentally give the same answer.
    keep = fm | sm
    wrong = delete_block_shape_error(fast[keep], stock[keep], group[keep], edges, fm[keep], sm[keep])
    assert not np.allclose(expected, wrong)


@pytest.mark.parametrize("mode", ["empty", "one_block", "leave_empty", "one_arm_empty"])
def test_hist_own_unavailable_errors_are_not_zero(mode):
    fast, stock, group = np.array([0., 1., 2.]), np.array([1., 1., 0.]), np.array([0, 1, 2])
    fm = sm = np.ones(3, bool)
    if mode == "empty":
        fast, stock, group = fast[:0], stock[:0], group[:0]
        fm = sm = np.zeros(0, bool)
    elif mode == "one_block":
        group[:] = 0
    elif mode == "leave_empty":
        fm = np.array([True, False, False])
    else:
        fm = np.zeros(3, bool)
    result = cr.hist_own(fast, stock, group, np.array([0., 1., 2.]), fm, sm)
    assert np.isnan(result["fraction_error"]).all()
    if mode in ("empty", "one_arm_empty"):
        assert np.isnan(result["fraction_difference"]).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_histograms_reject_selected_nonfinite_values(bad):
    values = np.array([bad, 1.])
    with pytest.raises(ValueError, match="nonfinite"):
        cr.hist_own(values, np.ones(2), np.array([0, 1]), np.array([0., 2.]), np.ones(2, bool), np.ones(2, bool))
    with pytest.raises(ValueError, match="nonfinite"):
        cr.cell_contributions(values, np.ones(2), np.array([0, 95]), np.array([0., 2.]))


def test_cell_contributions_96_by_flow_signed_bins_sum_to_global():
    edges = np.array([0., 1., 2.])
    fast = np.array([-1., 0., 1., 2., 3., .5, 1.5, -2., 4.])
    stock = np.array([.5, 1., 2., 3., -1., 1.5, .5, 4., -2.])
    cell = np.array([0, 0, 1, 1, 47, 48, 48, 95, 95])
    result = cr.cell_contributions(fast, stock, cell, edges)
    assert all(value.shape == (96, 4) for value in result.values())
    for index in range(96):
        mask = cell == index
        np.testing.assert_array_equal(result["fast"][index], flow_counts(fast[mask], edges))
        np.testing.assert_array_equal(result["stock"][index], flow_counts(stock[mask], edges))
    np.testing.assert_array_equal(result["difference"], result["fast"] - result["stock"])
    np.testing.assert_array_equal(result["difference"].sum(axis=0), flow_counts(fast, edges) - flow_counts(stock, edges))
    np.testing.assert_array_equal(result["difference"].sum(axis=1), np.zeros(96))
    assert np.any(result["difference"] < 0) and np.any(result["difference"] > 0)
    assert result["fast"].sum() == result["stock"].sum() == len(cell)
    # Equal global shapes can hide opposite signed regional changes; do not collapse to regional chi2.
    np.testing.assert_array_equal(result["difference"][48], np.zeros(4))
    assert result["difference"][0, 0] == 1


def geometry_saved():
    saved = {}
    for axis, values in zip("xyz", ([10., 10.], [-20., -20.], [100., 200.])):
        saved["geometry_disk_origin_" + axis] = np.array(values)
        for coord in "xyz":
            saved["geometry_disk_axis_" + axis + "_" + coord] = np.full(2, float(axis == coord))
    return saved


def test_coordinates_common_origin_both_disks_direction_signs_and_axis_origin():
    saved = geometry_saved()
    saved["entry_position"] = np.array([[13., -20., 100.], [10., -16., 200.], [7., -20., 100.], [10., -20., 200.]])
    saved["entry_direction"] = np.array([[.6, -.2, .7], [.3, -.4, .8], [.1, .2, .9], [0., -.5, .8]])
    _, origin, radial, tangent = cr.coordinates(saved)
    np.testing.assert_array_equal(origin, [10., -20.])
    np.testing.assert_allclose(radial, [.6, -.4, -.1, 0.])
    np.testing.assert_allclose(tangent, [-.2, -.3, -.2, -.5])
    # Different disk z origins are allowed; translation of global transverse coordinates leaves directions unchanged.
    shifted = copy.deepcopy(saved)
    shifted["entry_position"][:, :2] += [91., -13.]
    shifted["geometry_disk_origin_x"] += 91.
    shifted["geometry_disk_origin_y"] -= 13.
    _, _, ur, ut = cr.coordinates(shifted)
    np.testing.assert_allclose(ur, radial)
    np.testing.assert_allclose(ut, tangent)


@pytest.mark.parametrize("field", ["geometry_disk_origin_x", "geometry_disk_origin_y", "geometry_disk_axis_x_y", "geometry_disk_axis_z_z"])
def test_coordinates_reject_noncommon_or_nonaligned_geometry(field):
    saved = geometry_saved()
    saved["entry_position"] = np.zeros((1, 3))
    saved["entry_direction"] = np.array([[0., 0., 1.]])
    saved[field][1] += .01
    with pytest.raises(ValueError, match="aligned disks with common transverse origin"):
        cr.coordinates(saved)


def test_masks_frozen_threshold_boundaries_and_direction_zero():
    entry = dict(disk=np.array([0, 1, 1, 1, 1, 1]), radius=np.array([0., 449., 450., 550., 1000., 449.]),
                 energy=np.array([0., 50., 80., 95., 110., 95.]), cosine=np.array([0., .4, .7, 1., .69, .7]))
    ur, ut = np.array([-1., -0., 0., 1., -1., 0.]), np.array([1., 0., -0., -1., 0., -1.])
    diagnostic = dict(residual_angle_deg=np.array([0., 9.999, 10., 29.999, 30., 90.]), pool_fallback=np.array([False, True, False, False, False, True]))
    result = cr.masks(entry, ur, ut, diagnostic)
    expected = {"radius_0_450": [0, 1, 5], "radius_450_550": [2], "radius_550_1000": [3], "radius_1000_inf": [4],
                "energy_0_50": [0], "energy_50_80": [1], "energy_80_95": [2], "energy_95_110": [3, 5], "energy_110_inf": [4],
                "cosine_0_0.4": [0], "cosine_0.4_0.7": [1, 4], "cosine_0.7_1.000001": [2, 3, 5],
                "angle_0_10": [0, 1], "angle_10_30": [2, 3], "angle_30_inf": [4, 5],
                "radial_direction_negative": [0, 4], "radial_direction_nonnegative": [1, 2, 3, 5],
                "tangent_direction_negative": [3, 5], "tangent_direction_nonnegative": [0, 1, 2, 4],
                "highEinnerdisk1": [5], "problem109": [5], "fallback": [1, 5], "disk1_radius_0_450": [1, 5]}
    for key, indices in expected.items():
        np.testing.assert_array_equal(np.flatnonzero(result[key]), indices, err_msg=key)
    for prefix in ("radius_", "energy_", "cosine_", "angle_"):
        np.testing.assert_array_equal(sum(mask for key, mask in result.items() if key.startswith(prefix)), np.ones(6))


@pytest.fixture
def saved_arrays():
    """All arrays are invented in memory, with sparse original stored-row blocks and nonfront sentinels."""
    rows = np.array([1, 2, 101, 201, 202, 301, 401, 402])
    full, n = 405, len(rows)
    radius = np.array([0., 449., 450., 549., 550., 749., 750., 1000.])
    radial = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]] * 2)
    tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
    ur, ut = np.array([0., -.3, .2, -.2, .1, 0., -.4, .3]), np.array([.1, 0., -.2, .3, -.4, .2, 0., -.1])
    entry = dict(event_row=rows, disk=np.arange(n) % 2, radius=radius,
                 energy=np.array([40., 50., 79., 80., 94., 95., 105., 110.]),
                 cosine=np.array([.2, .4, .69, .7, .8, .9, 1., .3]), time=np.full(n, 500.),
                 position=np.column_stack((radius[:, None] * radial + [10., -20.], 100. + (np.arange(n) % 2) * 100.)),
                 direction=np.column_stack((ur[:, None] * radial + ut[:, None] * tangent, np.full(n, .8))))
    saved = dict(geometry_saved(), event_row=np.arange(full), query_event_rows=rows.copy(), full_ids=np.arange(full) + 10000)
    saved.update({"entry_" + key: value for key, value in entry.items()})
    energies = dict(stockref=[0., 10., 11., 20., 0., 40., 60., 150.], control=[0., 10., 12., 21., 0., 39., 61., 145.],
                    geometric=[12., 10., 0., 22., 15., 30., 140., 0.], crystal=[0., 11., 15., 0., 10., 42., 70., 130.])
    for index, (arm, values) in enumerate(energies.items()):
        values = np.array(values)
        selected = values > 10
        local = radius[:, None] * radial + (np.arange(n) * 500 - 1500 + index * 30)[:, None] * radial + (np.arange(n) * -450 + 1500 + index * 20)[:, None] * tangent
        fields = dict(energy=values, time=500. + np.arange(n) * 10 - 30 + index,
                      x=local[:, 0], y=local[:, 1], z=np.full(n, 100.))
        for field, value in fields.items():
            out = np.full(full, 9999. if field == "energy" else np.nan)
            out[rows] = value if field == "energy" else np.where(selected, value, np.nan)
            saved[arm + "_" + field] = out
    original = np.array([0., 10., 30., 40., 20., 50., 60., 150.])
    donor = original + np.array([15., 0., -5., 10., 5., -5., 100., 0.])
    diagnostic = {key: saved[key].copy() for key in ("event_row", "query_event_rows", "full_ids")}
    diagnostic.update(residual_event_row=rows.copy(), residual_original=original, residual_donor=donor,
                      residual_original_time_accepted=original * .8, residual_entry_folded_time=np.full(n, 500.),
                      residual_angle_deg=np.array([0., 10., 30., 3., 12., 35., 0., 90.]), pool_fallback=np.arange(n) == 3)
    for arm, scale in (("geometric", .1), ("crystal", .2)):
        diagnostic["residual_" + arm + "_lost"] = donor * scale
        diagnostic["residual_" + arm + "_time_rejected"] = donor * .15
        diagnostic["residual_" + arm + "_time_accepted"] = donor * (1 - scale - .15)
    return saved, diagnostic


def test_css_stage_correlations_keep_own_positive_css_denominators():
    arrays = dict(event_row=np.array([0, 101, 202, 303]), direction_radial=np.array([-1., -.5, .5, 1.]),
                  direction_tangent=np.array([0., .2, .4, .6]))
    diagnostic = {}
    for arm, shift in (("original", 0.), ("donor", 1.), ("geometric", 2.), ("crystal", 3.)):
        diagnostic[f"css_{arm}_centroid_energy"] = np.array([0., 10., 20., 30.]) + shift
        diagnostic[f"css_{arm}_centroid_radial"] = np.array([np.nan if not shift else 1., 2., 3., 4.])
        diagnostic[f"css_{arm}_centroid_tangent"] = np.array([np.nan if not shift else 4., 3., 2., 1.])
    result = cr.stage_correlations(diagnostic, arrays)
    for arm in ("donor", "geometric", "crystal"):
        assert result["arms"][arm]["radial"]["stock"]["n"] == 3
        assert result["arms"][arm]["radial"]["fast"]["n"] == 4
        assert result["arms"][arm]["energy_radial_direction"]["stock"]["n"] == 4
        np.testing.assert_allclose(result["arms"][arm]["radial"]["fast"]["correlation"],
                                   np.corrcoef([1., 2., 3., 4.], arrays["direction_radial"])[0, 1])


def test_telescoping_signed_energy_and_same_truth_momentum_identity(saved_arrays):
    saved, diagnostic = saved_arrays
    rows = saved["query_event_rows"]
    terms = cr.telescoping(saved, diagnostic, rows)
    p = np.sqrt(saved["entry_energy"] ** 2 - .51099895 ** 2)
    for arm, value in terms.items():
        assert set(value) == {"donor", "spatial", "timing", "downstream", "control"}
        np.testing.assert_allclose(value["donor"], diagnostic["residual_donor"] - diagnostic["residual_original"])
        np.testing.assert_allclose(value["spatial"], -diagnostic[f"residual_{arm}_lost"])
        np.testing.assert_allclose(value["timing"], diagnostic["residual_original"] - diagnostic["residual_original_time_accepted"] - diagnostic[f"residual_{arm}_time_rejected"])
        np.testing.assert_allclose(value["control"], saved["control_energy"][rows] - saved["stockref_energy"][rows])
        target = saved[arm + "_energy"][rows] - saved["stockref_energy"][rows]
        np.testing.assert_allclose(sum(value.values()), target, atol=1e-12)
        np.testing.assert_allclose(sum(term / p for term in value.values()), target / p, atol=1e-12)
        assert np.any(value["donor"] < 0) and np.any(value["donor"] > 0)


def test_telescoping_rejects_broken_css_accounting(saved_arrays):
    saved, diagnostic = saved_arrays
    diagnostic["residual_geometric_time_accepted"][2] += .1
    with pytest.raises(ValueError, match="telescoping response identity"):
        cr.telescoping(saved, diagnostic, saved["query_event_rows"])


def test_analyze_saved_arrays_flow_cells_truth_ep_and_no_mutation(saved_arrays):
    saved, diagnostic = saved_arrays
    before = copy.deepcopy(saved_arrays)
    result, arrays = cr.analyze(saved, diagnostic)
    rows = saved["query_event_rows"]
    assert result["queries"] == 8 and result["stored_events"] == 405
    assert result["downstream_term_is_causal"] is False
    assert result["regions"]["all"]["source_blocks"] == 5
    assert sum(result["regions"][f"cell_{i:02d}"]["entries"] for i in range(96)) == 8
    assert result["cell_coordinates"].shape == (96, 4)
    p = np.sqrt(saved["entry_energy"] ** 2 - .51099895 ** 2)
    np.testing.assert_allclose(arrays["momentum"], p)
    for arm in ("stockref", *cr.ARMS):
        np.testing.assert_allclose(arrays[arm + "_truth_ep"], saved[arm + "_energy"][rows] / p)
    for arm in cr.ARMS:
        item = result["regions"]["all"]["arms"][arm]
        f, s = saved[arm + "_energy"][rows], saved["stockref_energy"][rows]
        fs, ss = f > 10, s > 10
        assert item["selected"] == fs.sum() and item["stock_selected"] == ss.sum()
        assert item["gains"] == (fs & ~ss).sum() and item["misses"] == (ss & ~fs).sum()
        assert item["energy"]["difference"] == pytest.approx(np.mean(f - s))
        for name, edges in cr.EDGES.items():
            shape = item["shapes"][name]
            inclusive = name in ("energy", "truth_ep")
            assert shape["fast"].sum() == (len(rows) if inclusive else fs.sum())
            assert shape["stock"].sum() == (len(rows) if inclusive else ss.sum())
            if inclusive:
                cells = result["cell_histograms"][arm][name]
                np.testing.assert_array_equal(cells["difference"].sum(axis=0), shape["fast"] - shape["stock"])
                global_hist = result["global_histograms"][arm][name]
                np.testing.assert_array_equal(cells["fast"].sum(axis=0)[1:-1], global_hist["fast"])
                assert cells["fast"][:, -1].sum() == global_hist["overflow"]["fast"]
        if arm != "control":
            assert sum(v["difference"] for v in item["energy_terms"].values()) == pytest.approx(item["energy"]["difference"])
            assert sum(v["difference"] for v in item["ep_terms"].values()) == pytest.approx(item["truth_ep"]["difference"])
    # Saved centroids are disk-local, whereas entry positions are global.
    selected = arrays["stockref_energy"] > 10
    np.testing.assert_allclose(arrays["stockref_radial"][selected], (np.arange(8) * 500 - 1500)[selected])
    np.testing.assert_allclose(arrays["stockref_tangent"][selected], (np.arange(8) * -450 + 1500)[selected])
    json.dumps(cr.clean(result), allow_nan=False)
    for after, prior in zip(saved_arrays, before):
        assert after.keys() == prior.keys()
        for key in prior:
            np.testing.assert_array_equal(after[key], prior[key], err_msg=key)


@pytest.mark.parametrize("where,key", [(0, "query_event_rows"), (1, "query_event_rows"), (1, "full_ids"), (1, "event_row"), (1, "residual_event_row")])
def test_analyze_rejects_saved_alignment_drift(saved_arrays, where, key):
    saved_arrays[where][key][-1] += 1
    with pytest.raises(ValueError, match="saved residual mismatch"):
        cr.analyze(*saved_arrays)


@pytest.mark.parametrize("fault", ["negative", "duplicate", "descending", "outside", "float"])
def test_analyze_rejects_invalid_query_rows(saved_arrays, fault):
    saved, diagnostic = saved_arrays
    rows = saved["entry_event_row"].copy()
    if fault == "negative":
        # Keep the aliased last row otherwise valid: rejection must be of the row ID,
        # not an accidental downstream NaN or telescoping-accounting error.
        for arm in ("stockref", *cr.ARMS):
            for field in ("energy", "time", "x", "y", "z"):
                saved[arm + "_" + field][-1] = saved[arm + "_" + field][rows[0]]
        rows[0] = -1
    elif fault == "duplicate":
        rows[1] = rows[0]
    elif fault == "descending":
        rows[:2] = rows[:2][::-1]
    elif fault == "outside":
        rows[-1] = len(saved["event_row"])
    else:
        rows = rows.astype(float)
    saved["entry_event_row"] = saved["query_event_rows"] = diagnostic["query_event_rows"] = diagnostic["residual_event_row"] = rows
    with pytest.raises(ValueError, match="invalid query rows"):
        cr.analyze(saved, diagnostic)


def moment_oracle(y, x):
    if not len(y):
        return dict(mean=None, std=None, covariance=None, correlation=None)
    dy, dx = y - y.mean(), x - x.mean()
    std, xstd = np.sqrt(np.mean(dy ** 2)), np.sqrt(np.mean(dx ** 2))
    cov = np.mean(dy * dx)
    return dict(mean=float(y.mean()), std=float(std), covariance=float(cov),
                correlation=float(cov / std / xstd) if std > 0 and xstd > 0 else None)


def test_analyze_own_joint_moments_preserve_original_blocks_and_selection_denominators(saved_arrays):
    saved, diagnostic = saved_arrays
    result, arrays = cr.analyze(saved, diagnostic)
    group = saved["query_event_rows"] // 100
    blocks = np.unique(group)
    for arm in cr.ARMS:
        fs, ss = arrays[arm + "_energy"] > 10, arrays["stockref_energy"] > 10
        item = result["regions"]["all"]["arms"][arm]
        for name, x in (("radial", arrays["direction_radial"]), ("tangent", arrays["direction_tangent"]),
                        ("time", diagnostic["residual_entry_folded_time"])):
            f, s = arrays[arm + "_" + name], arrays["stockref_" + name]
            for population, fm, sm in (("own", fs, ss), ("joint", fs & ss, fs & ss)):
                actual = item["spatial"][name][population]
                expected_f, expected_s = moment_oracle(f[fm], x[fm]), moment_oracle(s[sm], x[sm])
                assert actual["fast"]["n"] == fm.sum() and actual["stock"]["n"] == sm.sum()
                leaves = []
                for block in blocks:
                    keep = group != block
                    of, os = moment_oracle(f[fm & keep], x[fm & keep]), moment_oracle(s[sm & keep], x[sm & keep])
                    leaves.append({k: None if of[k] is None or os[k] is None else of[k] - os[k] for k in of})
                for key in expected_f:
                    for side, expected in (("fast", expected_f), ("stock", expected_s)):
                        if expected[key] is None:
                            assert actual[side][key] is None
                        else:
                            assert actual[side][key] == pytest.approx(expected[key], abs=1e-10)
                    if expected_f[key] is None or expected_s[key] is None:
                        assert actual["difference"][key] is None
                    else:
                        assert actual["difference"][key] == pytest.approx(expected_f[key] - expected_s[key], abs=1e-10)
                    if any(leave[key] is None for leave in leaves):
                        assert actual["error"][key] is None
                    else:
                        values = np.array([leave[key] for leave in leaves])
                        error = np.sqrt((len(blocks) - 1) / len(blocks) * np.sum((values - values.mean()) ** 2))
                        assert actual["error"][key] == pytest.approx(error, abs=1e-10)


def test_fixed_histogram_edges_and_96_cell_coordinate_order(saved_arrays):
    np.testing.assert_array_equal(cr.EDGES["energy"], np.arange(0., 111., 5.))
    np.testing.assert_array_equal(cr.EDGES["truth_ep"], np.linspace(0., 1.2, 25))
    np.testing.assert_array_equal(cr.EDGES["radial"], np.arange(-1400., 1401., 40.))
    np.testing.assert_array_equal(cr.EDGES["tangent"], np.arange(-1400., 1401., 40.))
    np.testing.assert_array_equal(cr.EDGES["time"], np.arange(-20., 21., 1.))
    result, arrays = cr.analyze(*saved_arrays)
    # Hand-evaluated bins at exact 50/80/95-MeV, .4/.7 cosine and 450/550/750/1000-mm boundaries.
    expected = np.array([[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [1, 2, 2, 1],
                         [0, 2, 2, 2], [1, 3, 2, 2], [0, 3, 2, 3], [1, 3, 0, 3]])
    np.testing.assert_array_equal(result["cell_coordinates"][arrays["cell_id"]], expected)


@pytest.mark.parametrize("field,value,message", [
    ("entry_energy", .1, "invalid incident momentum"),
    ("entry_energy", 111., "outside fixed diagnostic cells"),
    ("entry_disk", 2, "outside fixed diagnostic cells"),
    ("entry_cosine", 1.1, "outside fixed diagnostic cells"),
    ("entry_radius", 1001., "outside fixed diagnostic cells"),
    ("geometric_energy", -1., "invalid saved leading energy"),
    ("crystal_energy", np.nan, "invalid saved leading energy"),
])
def test_analyze_rejects_invalid_truth_or_saved_energy(saved_arrays, field, value, message):
    saved, diagnostic = saved_arrays
    saved[field][0 if field.startswith("entry_") else saved["query_event_rows"][0]] = value
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match=message):
        cr.analyze(saved, diagnostic)
