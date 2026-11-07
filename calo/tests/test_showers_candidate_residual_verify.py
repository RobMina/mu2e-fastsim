"""Independent saved-only verification; opt in with CALO_RESIDUAL_VERIFY_BASE.

No production helper imports, RNG, reconstruction, ROOT, or new analysis inputs.
The base contains production_candidate_trial and production_candidate_residual_final.
Run with pytest -s to print a compact diagnostic report; nothing is written.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

ARMS = ("control", "geometric", "crystal")
FIELDS = ("mean", "std", "covariance", "correlation")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def saved():
    base = os.environ.get("CALO_RESIDUAL_VERIFY_BASE")
    if not base:
        pytest.skip("set CALO_RESIDUAL_VERIFY_BASE for saved-only verification")
    base = Path(base)
    trial, residual = base / "production_candidate_trial", base / "production_candidate_residual_final"
    paths = [trial / n for n in ("summary.json", "numbers.json", "clusters.npz", "diagnostics.npz")]
    paths += [residual / n for n in ("numbers.json", "arrays.npz")]
    before = {p: digest(p) for p in paths}
    summary = json.loads((trial / "summary.json").read_text())
    result = json.loads((residual / "numbers.json").read_text())
    prior = json.loads((trial / "numbers.json").read_text())
    assert before[trial / "summary.json"] == "dee6f99faa4882a3ac98ded6a9a7d160aa011e1d739aab78235c443adc4eaf5a"
    for p in paths[:4]:
        assert before[p] == result["input_sha256"][str(p)]
        if p.name != "summary.json":
            assert before[p] == summary["output_sha256"][p.name]
    def read(path):
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}
    s, d, a = read(trial / "clusters.npz"), read(trial / "diagnostics.npz"), read(residual / "arrays.npz")
    print("\nHASHES", {str(p.relative_to(base)): h for p, h in before.items()})
    yield result, prior, s, d, a
    assert {p: digest(p) for p in paths} == before


@pytest.fixture(autouse=True)
def no_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved-only verification forbids RNG")
    for name in ("default_rng", "RandomState", "Generator", "SeedSequence", "random", "normal", "choice", "seed"):
        monkeypatch.setattr(np.random, name, forbidden)


def stats(y, x):
    # Independent numpy routines, including np.corrcoef (not production moments).
    return np.array([np.mean(y), np.std(y), np.mean((y - np.mean(y)) * (x - np.mean(x))), np.corrcoef(y, x)[0, 1]])


def test_css_stage_correlations_against_direct_numpy(saved):
    r, _, _, d, a = saved
    for arm in ("donor", "geometric", "crystal"):
        for name, coordinate in (("radial", "direction_radial"), ("tangent", "direction_tangent")):
            record = r["css_stage_correlations"]["arms"][arm][name]
            for field, source in (("fast", arm), ("stock", "original")):
                selected = d[f"css_{source}_centroid_energy"] > 0
                y = d[f"css_{source}_centroid_{name}"][selected]
                x = a[coordinate][selected]
                np.testing.assert_allclose([record[field][k] for k in FIELDS], stats(y, x), atol=1e-10)
                assert record[field]["n"] == int(selected.sum())
            print("CSS_STAGE", arm, name, "stock/fast", record["stock"]["correlation"], record["fast"]["correlation"])


def flow(value, edges):
    return np.r_[(value < edges[0]).sum(), np.histogram(value, edges)[0], (value > edges[-1]).sum()]


def test_rows_coordinates_observables_and_telescoping(saved):
    r, _, s, d, a = saved
    rows = a["event_row"]
    for other in (s["query_event_rows"], s["entry_event_row"], d["query_event_rows"], d["residual_event_row"]):
        np.testing.assert_array_equal(rows, other)
    for key in ("event_row", "full_ids"):
        np.testing.assert_array_equal(s[key], d[key])
    assert np.all(np.diff(rows) > 0) and len(rows) == r["queries"]
    assert len(s["event_row"]) == r["stored_events"]
    origin = np.column_stack([s["geometry_disk_origin_" + k] for k in "xy"])
    np.testing.assert_allclose(origin, np.broadcast_to(origin[0], origin.shape), atol=1e-6, rtol=0)
    for axis in "xyz":
        for component in "xyz":
            np.testing.assert_allclose(s[f"geometry_disk_axis_{axis}_{component}"], float(axis == component), atol=1e-6, rtol=0)
    local = s["entry_position"][:, :2] - origin[0]
    radius = np.linalg.norm(local, axis=1)
    np.testing.assert_allclose(radius, s["entry_radius"], atol=1e-6)
    radial = np.divide(local, radius[:, None], out=np.tile([1., 0.], (len(rows), 1)), where=radius[:, None] > 0)
    tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
    p = np.sqrt(s["entry_energy"] ** 2 - .51099895 ** 2)
    np.testing.assert_array_equal(a["momentum"], p)
    for name, unit in (("radial", radial), ("tangent", tangent)):
        np.testing.assert_allclose(a["direction_" + name], np.sum(s["entry_direction"][:, :2] * unit, axis=1), atol=1e-14)
    for arm in ("stockref", *ARMS):
        energy = s[arm + "_energy"][rows]
        np.testing.assert_array_equal(a[arm + "_energy"], energy)
        np.testing.assert_array_equal(a[arm + "_truth_ep"], energy / p)
        delta = np.column_stack((s[arm + "_x"][rows], s[arm + "_y"][rows])) - local
        for name, unit in (("radial", radial), ("tangent", tangent)):
            np.testing.assert_allclose(a[arm + "_" + name], np.sum(delta * unit, axis=1), equal_nan=True, atol=1e-12)
        np.testing.assert_allclose(a[arm + "_time"], s[arm + "_time"][rows] - d["residual_entry_folded_time"], equal_nan=True)
    for arm in ("geometric", "crystal"):
        terms = dict(donor=d["residual_donor"] - d["residual_original"],
                     spatial=-d[f"residual_{arm}_lost"],
                     timing=d["residual_original"] - d["residual_original_time_accepted"] - d[f"residual_{arm}_time_rejected"],
                     downstream=(a[arm + "_energy"] - d[f"residual_{arm}_time_accepted"]) - (a["control_energy"] - d["residual_original_time_accepted"]),
                     control=a["control_energy"] - a["stockref_energy"])
        for name, value in terms.items():
            np.testing.assert_allclose(value, a[f"{arm}_term_{name}"], atol=1e-12)
        target = a[arm + "_energy"] - a["stockref_energy"]
        np.testing.assert_allclose(sum(terms.values()), target, atol=1e-10)
        np.testing.assert_allclose(sum(v / p for v in terms.values()), target / p, atol=1e-12)
        print("TELESCOPING", arm, "max residual", np.max(np.abs(sum(terms.values()) - target)), "mean terms", {k: float(v.mean()) for k, v in terms.items()})
    print("ROWS", len(rows), "stored", len(s["event_row"]), "blocks", len(np.unique(rows // 100)), "transverse origin", origin.tolist())


def test_correlations_and_direct_block_deletion(saved):
    r, _, s, d, a = saved
    occupancy = np.bincount(a["cell_id"], minlength=96)
    top = np.argsort(-occupancy)[:4]
    print("OCCUPANCY", "empty", int((occupancy == 0).sum()), "under20", int(((occupancy > 0) & (occupancy < 20)).sum()), "top", [(int(c), int(occupancy[c]), r["cell_coordinates"][c]) for c in top])
    regions = [("all", np.ones(len(a["event_row"]), bool))] + [(f"cell_{c:02d}", a["cell_id"] == c) for c in top]
    for label, mask in regions:
        groups = a["event_row"][mask] // 100
        blocks = np.unique(groups)
        assert len(blocks) == r["regions"][label]["source_blocks"]
        for arm in ARMS:
            item = r["regions"][label]["arms"][arm]
            fs, ss = a[arm + "_energy"][mask] > 10, a["stockref_energy"][mask] > 10
            assert fs.sum() == item["selected"] and ss.sum() == item["stock_selected"]
            variables = ("radial", "tangent", "time") if label == "all" else ("radial",)
            for name in variables:
                x = (d["residual_entry_folded_time"] if name == "time" else a["direction_" + name])[mask]
                fast, stock = a[arm + "_" + name][mask], a["stockref_" + name][mask]
                for population, fm, sm in (("own", fs, ss), ("joint", fs & ss, fs & ss)):
                    actual = item["spatial"][name][population]
                    f, st = stats(fast[fm], x[fm]), stats(stock[sm], x[sm])
                    for side, expected in (("fast", f), ("stock", st), ("difference", f - st)):
                        np.testing.assert_allclose([actual[side][key] for key in FIELDS], expected, atol=3e-10, rtol=1e-10)
                    leaves = []
                    for block in blocks:
                        keep = groups != block
                        leaves.append(stats(fast[fm & keep], x[fm & keep]) - stats(stock[sm & keep], x[sm & keep]))
                    leaves = np.asarray(leaves)
                    error = np.sqrt((len(blocks) - 1) / len(blocks) * np.sum((leaves - leaves.mean(axis=0)) ** 2, axis=0))
                    np.testing.assert_allclose([actual["error"][key] for key in FIELDS], error, atol=3e-10, rtol=1e-9)
                    if name == "radial":
                        print("RADIAL", label, arm, population, "N", int(fm.sum()), int(sm.sum()), "r_f/s", *np.round([f[3], st[3]], 6), "delta/error", round(f[3] - st[3], 6), round(error[3], 6), "width_f/s", *np.round([f[1], st[1]], 3))


def test_fixed_cell_contributions_hist_errors_and_region_report(saved):
    r, prior, s, _, a = saved
    indexes = [s["entry_disk"].astype(int)]
    for name, edges in (("energy", [0, 50, 80, 95, 110]), ("cosine", [0, .4, .7, 1.000001]), ("radius", [0, 450, 550, 750, 1000])):
        indexes.append(np.clip(np.searchsorted(edges, s["entry_" + name], side="right") - 1, 0, len(edges) - 2))
    cell = np.ravel_multi_index(indexes, (2, 4, 3, 4))
    np.testing.assert_array_equal(cell, a["cell_id"])
    group = a["event_row"] // 100
    blocks = np.unique(group)
    for arm in ARMS:
        for name, title in (("energy", "Leading-cluster energy [MeV]"), ("truth_ep", "E/p at entry (truth diagnostic)")):
            h = r["global_histograms"][arm][name]
            old = prior["diagnostics"]["candidate"][arm]["observables"][title]
            for key in ("fast", "stock", "variance", "pull", "chi2", "ndf"):
                assert h[key] == old[key]
            edges = np.array(h["edges"])
            fast, stock = a[arm + "_" + name], a["stockref_" + name]
            counts = r["cell_histograms"][arm][name]
            for side, values in (("fast", fast), ("stock", stock)):
                expected = np.array([flow(values[cell == c], edges) for c in range(96)])
                np.testing.assert_array_equal(counts[side], expected)
                np.testing.assert_array_equal(expected.sum(axis=0), flow(values, edges))
                np.testing.assert_array_equal(expected.sum(axis=0)[1:-1], h[side])
            delta = np.asarray(counts["fast"]) - np.asarray(counts["stock"])
            np.testing.assert_array_equal(counts["difference"], delta)
            np.testing.assert_array_equal(delta.sum(axis=1), np.zeros(96))
            per_block = np.array([np.histogram(fast[group == b], edges)[0] - np.histogram(stock[group == b], edges)[0] for b in blocks])
            variance = len(blocks) * np.var(per_block, axis=0, ddof=1)
            np.testing.assert_allclose(np.array(h["variance"], float), variance, atol=1e-9)
            print("MARGINAL", arm, name, "chi2/ndf", h["chi2"], h["ndf"])
            if name == "truth_ep" and arm != "control":
                pull = np.array(h["pull"], float)
                for b in np.argsort(-np.nan_to_num(np.abs(pull)))[:3]:
                    cells = np.argsort(-np.abs(delta[:, b + 1]))[:3]
                    details = []
                    for c in cells:
                        # Error on the signed cell contribution in the SAME original block universe.
                        contrib = np.array([np.histogram(fast[(cell == c) & (group == block)], edges)[0][b] - np.histogram(stock[(cell == c) & (group == block)], edges)[0][b] for block in blocks])
                        details.append((int(c), int(delta[c, b + 1]), round(float(np.sqrt(len(blocks) * np.var(contrib, ddof=1))), 3)))
                    print("EP_BIN", arm, tuple(np.round(edges[b:b + 2], 4)), "delta/error/pull", int(delta[:, b + 1].sum()), round(float(np.sqrt(variance[b])), 3), round(float(pull[b]), 3), "dominant cell_delta_error", details)
    def rounded(value):
        return None if value is None else round(value, 5)
    for label in ("all", "disk0_radius_0_450", "disk0_radius_450_550", "disk0_radius_550_1000", "disk1_radius_0_450", "disk1_radius_450_550", "disk1_radius_550_1000", "problem109", "fallback"):
        region = r["regions"][label]
        print("REGION", label, "N", region["entries"], "blocks", region["source_blocks"], {arm: {name: (rounded(region["arms"][arm][name]["difference"]), rounded(region["arms"][arm][name]["error"])) for name in ("energy", "truth_ep")} for arm in ARMS})
    for arm in ARMS:
        print("EP_TAIL", arm, r["regions"]["all"]["arms"][arm]["ep_ge1"], "overflow", r["global_histograms"][arm]["truth_ep"]["overflow"])
        print("MOMENTUM_RELATION", arm, {key: r["regions"]["all"]["arms"][arm][key] for key in ("energy_momentum", "ep_momentum")})
