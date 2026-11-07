"""Opt-in independent SAVED-ONLY localization oracle.

CALO_DIRECTION_LOCALIZATION_VERIFY_BASE is the CAL2_20261105 directory (or its
production_direction_localization child). Only authenticated JSON/NPZ and the
receipt-listed source bytes are read. No production imports, RNG, ROOT, input
physics, placement, reconstruction, or report regeneration. Population moments
and uncertainty use direct NumPy operations on literally deleted row//100 blocks;
per-region caching avoids repeating identical arm/mask calculations.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

TRIAL_SHA = "4bf2eb82fa744fef6aac2d22932cabe2e3820de8039525a0de2758fc5cf5918e"
REPORT_SHA = "4cf44336ecb7085222454a2cdf8d2cb9da8a966eedbfd001f08520d796876009"
SUMMARY_SHA = "7b44aa8b0e95fa56118562159a75a5ab19d848428f5fd60d24933ea3594b620d"
OLD_SHA = "0385c0634055ec59cc97f88c9d1a4a10c5843bbbf0e4a99bd15908d8fcc5e403"
OUTPUT_PINS = {
    "execution.json": "34939428b0d373aa054b346b7b761d789bae04a5e8076a73fb20d4879df3e27b",
    "preparation.json": "4e4a2af1509b5fe8b2476aa80fc81e8c7f36ea815f78ea0de66d7a2764c845ba",
    "draw.npz": "736e39148804042402c2d9db7bf8ec7b467770eb1277a3010be365893d71df27",
    "pools.npz": "4f3e61ba5baaef864d374e59e24aa2ee728c2042c0bff6fac397e90327a88058",
    "clusters.npz": "471027260b7c5619b51e8ed0f2576bf8083fe2f291ecbeb776d7f79eb8294ed5",
    "numbers.json": "8d317aecda07c0cdb074911132fdbac83eb6d89a2c4dd44ba8c3d3b4133f3339",
    "diagnostics.npz": "f99aa82ee1aef0c8a405f6313a7131dbb4c90c8a1a5fc3fa63611c6e85b1a417",
    "conditional.json": "53a4cfda7bb7f2e7b32c64e411c6dddbb77875e0b81003b445a6416ac11a6edf",
    "conditional_arrays.npz": "8753c2d55737674767850d733e9cc04487933aef6c38f70fd462d7a7678075dd",
}
ARMS = ("geometric", "crystal")
TERMS = ("donor", "spatial", "timing", "downstream", "control")
FIELDS = ("mean", "std", "covariance", "correlation")
TAILS = {"radial": (100., 200., 400.), "tangent": (50., 100., 200.)}
EDGES = np.linspace(0., 1.2, 25)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(actual, expected, path="result"):
    """Require nulls exactly where the independent statistic is undefined."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            check(actual[key], expected[key], path + "/" + key)
    elif expected is None:
        assert actual is None, path
    elif isinstance(expected, str):
        assert actual == expected, path
    else:
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        assert a.shape == e.shape, path
        np.testing.assert_array_equal(np.isnan(a), np.isnan(e), err_msg=path)
        np.testing.assert_allclose(a, e, rtol=2e-9, atol=2e-9, equal_nan=True, err_msg=path)


def exact(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def region_masks(a, c):
    result = {"all": np.ones(len(a["event_row"]), dtype=bool)}
    for disk in range(2):
        result[f"disk{disk}"] = c["entry_disk"] == disk
    for name, edges in (("radius", (0, 450, 550, 1000, np.inf)),
                        ("energy", (0, 50, 80, 95, 110, np.inf)),
                        ("cosine", (0, .4, .7, 1.000001))):
        for low, high in zip(edges[:-1], edges[1:]):
            result[f"{name}_{low}_{high}"] = (c["entry_" + name] >= low) & (c["entry_" + name] < high)
    for coordinate in TAILS:
        result[coordinate + "_direction_negative"] = a["direction_" + coordinate] < 0
        result[coordinate + "_direction_nonnegative"] = a["direction_" + coordinate] >= 0
    result.update({name: a[name] for name in ("direction_used", "direction_fallback")})
    for disk in range(2):
        for label in tuple(result):
            if label.startswith("radius_"):
                result[f"disk{disk}_" + label] = result[f"disk{disk}"] & result[label]
    result.update({f"cell_{i:02d}": a["cell_id"] == i for i in range(96)})
    return result


@pytest.fixture(scope="module")
def saved():
    value = os.environ.get("CALO_DIRECTION_LOCALIZATION_VERIFY_BASE")
    if not value:
        pytest.skip("set CALO_DIRECTION_LOCALIZATION_VERIFY_BASE for bounded saved-only verification")
    base = Path(value).resolve()
    if base.name == "production_direction_localization":
        base = base.parent
    tracked = {}

    def read(path, expected):
        path = Path(path)
        assert digest(path) == expected, f"before-read hash: {path}"
        tracked[path] = expected
        try:
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == expected, f"read hash: {path}"
            if path.suffix == ".json":
                return json.loads(raw)
            if path.suffix == ".npz":
                with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                    return {key: archive[key] for key in archive.files}
            return raw
        finally:
            assert digest(path) == expected, f"after-read hash: {path}"

    try:
        summary = read(base / "production_direction_localization/summary.json", SUMMARY_SHA)
        report = read(base / "production_direction_localization/numbers.json", REPORT_SHA)
        trial = read(base / "production_direction_trial/summary.json", TRIAL_SHA)
        assert summary["completed"] is True and trial["completed"] is True
        assert trial["sampler_calls"] == 1 and trial["reconstruction_calls"] == 2
        assert summary["trial_sha256"] == TRIAL_SHA
        assert summary["output_sha256"] == {"numbers.json": REPORT_SHA}
        for field in ("sampler_calls", "placement_calls", "reconstruction_calls", "root_reads"):
            assert summary[field] == 0
        for name, pin in OUTPUT_PINS.items():
            assert trial["output_sha256"][name] == pin
        anchors = [v for k, v in trial["input_sha256"].items()
                   if k.endswith("/production_candidate_trial/diagnostics.npz")]
        assert anchors == [OLD_SHA]
        # Only the six completed saved inputs and five source artifacts in this
        # receipt; never traverse the old receipt's physics-input hash tree.
        inputs = summary["input_sha256"]
        assert len(inputs) == 11
        source_names = {"direction_localization.py", "direction_localization_run.py",
                        "conditional_stats.py", "compare.py", "candidate_residual.py"}
        assert {Path(k).name for k in inputs if k.endswith(".py")} == source_names
        data = {}
        for recorded_path, pin in inputs.items():
            p = Path(recorded_path)
            if p.suffix == ".py":
                read(p, pin)  # Authenticate bytes, deliberately never import.
            elif p.name == "summary.json":
                assert pin == TRIAL_SHA and p.parent.name == "production_direction_trial"
            elif p.parent.name == "production_candidate_trial":
                assert p.name == "diagnostics.npz" and pin == OLD_SHA
                data["old"] = read(base / p.parent.name / p.name, pin)
            else:
                assert p.parent.name == "production_direction_trial" and pin == OUTPUT_PINS[p.name]
                data[p.stem] = read(base / p.parent.name / p.name, pin)
        data["report"] = report
        data["regions"] = region_masks(data["conditional_arrays"], data["clusters"])
        yield data
    finally:
        # Teardown runs even on a failed numerical assertion: no saved input,
        # output, receipt, or receipt-listed production source may change.
        for path, pin in tracked.items():
            assert digest(path) == pin, f"source/artifact mutation: {path}"


def jk(deleted):
    deleted = np.asarray(deleted, dtype=float)
    if len(deleted) < 2:
        return np.full(deleted.shape[1:], np.nan)
    return np.sqrt((len(deleted) - 1) * np.mean((deleted - deleted.mean(axis=0)) ** 2, axis=0))


def mean_pair(f, s, blocks):
    f, s = np.asarray(f, float), np.asarray(s, float)
    if not len(f):
        return dict(fast=None, stock=None, difference=None, error=None)
    difference = f - s
    leave = [difference[blocks != b].mean() if np.any(blocks != b) else np.nan for b in np.unique(blocks)]
    return dict(fast=f.mean(), stock=s.mean(), difference=difference.mean(), error=jk(leave))


def test_saved_alignment_inclusive_energy_and_exact_terms(saved):
    a, c, d, old, p, r = [saved[k] for k in ("conditional_arrays", "clusters", "diagnostics", "old", "pools", "report")]
    rows = a["event_row"]
    assert r["schema"] == "direction-localization-v1"
    assert r["queries"] == len(rows) == 2396 and r["stored_events"] == 2813
    assert r["source_blocks"] == len(np.unique(rows // 100)) == 29
    assert np.all(np.diff(rows) > 0)
    assert len(np.unique(c["full_ids"])) == 2813
    exact(c["entry_event_row"], rows)
    for source in (c, d, old):
        exact(source["full_ids"], c["full_ids"])
        exact(source["event_row"], np.arange(2813))
        exact(source["query_event_rows"], rows)
    for source in (d, old):
        exact(np.flatnonzero(source["front"]), rows)
        for key in ("pool_event_row", "css_event_row", "residual_event_row"):
            exact(source[key], rows)
    exact(p["event_row"], rows)
    for name in ("event_row", "momentum", "direction_radial", "direction_tangent", "cell_id"):
        exact(a["production_" + name], a[name])
    check(a["momentum"], np.sqrt(c["entry_energy"] ** 2 - .51099895 ** 2))
    ebin = np.digitize(c["entry_energy"], [50, 80, 95])
    cosinebin = np.digitize(c["entry_cosine"], [.4, .7])
    rbin = np.digitize(c["entry_radius"], [450, 550, 750])
    exact(a["cell_id"], c["entry_disk"] * 48 + ebin * 12 + cosinebin * 4 + rbin)
    exact(r["cell_coordinates"], np.array([(d, e, co, ra) for d in range(2) for e in range(4) for co in range(3) for ra in range(4)]))
    check(r["tail_thresholds_mm"], TAILS)
    check(r["absolute_quantile_probabilities"], [.5, .9, .95, .99])
    check(r["variance_partition_cut_mm"], dict(radial=200., tangent=100.))
    for prefix, diagnostic in (("", d), ("production_", old)):
        for arm in ("stockref", "control", *ARMS):
            energy = a[prefix + arm + "_energy"]
            assert np.all(energy >= 0)
            check(a[prefix + arm + "_truth_ep"], energy / a["momentum"])
            exact(energy > 10, diagnostic["residual_" + arm + "_selected"])
            if not prefix:
                exact(energy, c[arm + "_energy"][rows])
            if arm in ("stockref", "control"):
                for field in ("energy", "truth_ep", "radial", "tangent", "time"):
                    exact(a["production_" + arm + "_" + field], a[arm + "_" + field])
        original, donor = diagnostic["residual_original"], diagnostic["residual_donor"]
        accepted = diagnostic["residual_original_time_accepted"]
        for arm in ARMS:
            lead = a[prefix + arm + "_energy"]
            lost, reject, kept = [diagnostic[f"residual_{arm}_" + field] for field in ("lost", "time_rejected", "time_accepted")]
            check(donor, lost + reject + kept)
            expected = (donor - original, -lost, original - accepted - reject,
                        lead - kept - a["control_energy"] + accepted,
                        a["control_energy"] - a["stockref_energy"])
            for term, value in zip(TERMS, expected):
                check(a[prefix + arm + "_term_" + term], value, prefix + arm + "/" + term)
            check(sum(expected), lead - a["stockref_energy"])
            check(sum(value / a["momentum"] for value in expected),
                  a[prefix + arm + "_truth_ep"] - a["stockref_truth_ep"])


def flow_hist(value):
    # np.histogram handles inclusive last edge; flows are counted explicitly.
    return np.r_[np.count_nonzero(value < EDGES[0]), np.histogram(value, EDGES)[0], np.count_nonzero(value > EDGES[-1])]


def test_all96_ep_signed_counts_full_original_blocks_and_migration(saved):
    a, c, r = [saved[k] for k in ("conditional_arrays", "clusters", "report")]
    ep = r["ep_counts"]
    exact(ep["edges"], EDGES)
    assert ep["flow_bins"] is True
    group = a["event_row"] // 100
    blocks = np.unique(group)
    partitions = {"cell": (a["cell_id"], 96), "disk": (c["entry_disk"], 2),
                  "radius": (np.digitize(c["entry_radius"], [450, 550, 1000]), 4),
                  "direction_flag": (a["direction_fallback"].astype(int), 2)}
    labels = {"control": "control", **{name + "_" + arm: prefix + arm for name, prefix in (("new", ""), ("production", "production_")) for arm in ARMS}}
    assert set(ep["arms"]) == set(labels)
    stock = a["stockref_truth_ep"]
    for label, prefix in labels.items():
        fast = a[prefix + "_truth_ep"]
        record = ep["arms"][label]
        assert set(record["partitions"]) == set(partitions)
        for pname, (ids, size) in {"global": (np.zeros(len(group), int), 1), **partitions}.items():
            fs, ss, ds, errors = [], [], [], []
            for index in range(size):
                mask = ids == index
                f, s = flow_hist(fast[mask]), flow_hist(stock[mask])
                by_block = np.array([flow_hist(fast[mask & (group == b)]) - flow_hist(stock[mask & (group == b)]) for b in blocks])
                fs.append(f); ss.append(s); ds.append(f - s)
                errors.append(np.sqrt(len(blocks) * np.var(by_block, axis=0, ddof=1)))
            expected = dict(fast=np.array(fs), stock=np.array(ss), difference=np.array(ds), error=np.array(errors), source_blocks=len(blocks))
            if pname == "global":
                expected = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in expected.items()}
                output = record["global_counts"]
            else:
                output = record["partitions"][pname]
                for key in ("fast", "stock", "difference"):
                    exact(np.sum(output[key], axis=0), record["global_counts"][key])
            check(output, expected, label + "/" + pname)
        # Literal per-query bin assignment, independent of production searchsorted.
        migration = np.zeros((26, 26), dtype=int)
        for s, f in zip(stock, fast):
            si = int(np.flatnonzero(flow_hist(np.array([s])))[0])
            fi = int(np.flatnonzero(flow_hist(np.array([f])))[0])
            migration[si, fi] += 1
        exact(record["stock_to_arm"], migration)
        exact(migration.sum(axis=0), record["global_counts"]["fast"])
        exact(migration.sum(axis=1), record["global_counts"]["stock"])
        assert migration.sum() == 2396


def test_fixed_stock_windows_literal_delete_block_terms(saved):
    a, r = saved["conditional_arrays"], saved["report"]
    stock = a["stockref_truth_ep"]
    windows = {f"{lo:.2f}_{hi:.2f}": (stock >= lo) & (stock < hi) for lo, hi in ((.8, .85), (.85, .9), (.9, .95), (.95, 1.))}
    windows.update(ge1=stock >= 1., gt1p2=stock > 1.2)
    assert set(windows) == set(r["stock_ep_windows"])
    labels = {"control": "control", **{name + "_" + arm: prefix + arm for name, prefix in (("new", ""), ("production", "production_")) for arm in ARMS}}
    for label, mask in windows.items():
        out = r["stock_ep_windows"][label]
        group = a["event_row"][mask] // 100
        assert out["entries"] == mask.sum() and out["source_blocks"] == len(np.unique(group))
        assert set(out["arms"]) == set(labels)
        for name, prefix in labels.items():
            expected = {field: mean_pair(a[prefix + "_" + field][mask], a["stockref_" + field][mask], group) for field in ("energy", "truth_ep")}
            if name != "control":
                for kind, divisor in (("energy", 1.), ("ep", a["momentum"][mask])):
                    expected[kind + "_terms"] = {term: mean_pair(a[prefix + "_term_" + term][mask] / divisor, np.zeros(mask.sum()), group) for term in TERMS}
            check(out["arms"][name], expected, label + "/" + name)


def direct_moments(y, x):
    if not len(y):
        return np.full(4, np.nan)
    covariance = np.mean((y - y.mean()) * (x - x.mean()))
    correlation = np.corrcoef(y, x)[0, 1] if len(y) > 1 and np.ptp(y) > 0 and np.ptp(x) > 0 else np.nan
    return np.array([np.mean(y), np.std(y), covariance, correlation])


class PopulationOracle:
    """One cache per fixed region/coordinate, not per contrast or population."""
    def __init__(self, x, group, thresholds=None):
        self.x, self.group, self.thresholds = x, group, thresholds
        self.blocks = np.unique(group)
        self.cache = {}

    def arm(self, key, values, selected):
        cache_key = (key, selected.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
        y, x = values[selected], self.x[selected]
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(x))
        full = direct_moments(y, x)
        deleted = np.array([direct_moments(values[selected & (self.group != b)], self.x[selected & (self.group != b)]) for b in self.blocks]).reshape(-1, 4)
        out = dict(moments=dict(n=len(y), **dict(zip(FIELDS, full))))
        tail_deleted = None
        if self.thresholds is not None:
            t = np.array(self.thresholds)
            hits = np.sum(np.abs(y[:, None]) > t, axis=0)
            fraction = hits / len(y) if len(y) else np.full(3, np.nan)
            leave = []
            for b in self.blocks:
                yy = values[selected & (self.group != b)]
                leave.append(np.mean(np.abs(yy[:, None]) > t, axis=0) if len(yy) else np.full(3, np.nan))
            tail_deleted = np.array(leave).reshape(-1, 3)
            out["tails"] = dict(n=len(y), counts=hits, fraction=fraction, error=jk(tail_deleted))
            if len(y):
                central = np.abs(y) <= t[1]
                squares = (y - y.mean()) ** 2
                out.update(abs_quantiles=np.quantile(np.abs(y), [.5, .9, .95, .99]), abs_max=np.max(np.abs(y)),
                           variance_parts=dict(central_n=central.sum(), tail_n=(~central).sum(),
                                               central=squares[central].sum() / len(y), tail=squares[~central].sum() / len(y), total=squares.mean()))
            else:
                out.update(abs_quantiles=None, abs_max=None,
                           variance_parts=dict(central_n=0, tail_n=0, central=None, tail=None, total=None))
        self.cache[cache_key] = out, full, deleted, tail_deleted
        return self.cache[cache_key]

    def verify(self, saved, values, masks, keys, path, reference="stock"):
        computed = {name: self.arm(keys[name], value, masks[name]) for name, value in values.items()}
        assert set(saved["arms"]) == set(values), path
        for name, (out, _, _, _) in computed.items():
            check(saved["arms"][name], out, path + "/" + name)
        pairs = [(name, reference) for name in values if name != reference] + [("new", "production")]
        assert set(saved["contrasts"]) == {f + "_minus_" + s for f, s in pairs}, path
        for fast, stock in pairs:
            f, s = computed[fast], computed[stock]
            difference = f[1] - s[1]
            error = jk(f[2] - s[2])
            error[~np.isfinite(difference)] = np.nan
            out = dict(moments=dict(difference=dict(zip(FIELDS, difference)), error=dict(zip(FIELDS, error))))
            if self.thresholds is not None:
                out["tails"] = dict(difference=f[0]["tails"]["fraction"] - s[0]["tails"]["fraction"], error=jk(f[3] - s[3]))
            check(saved["contrasts"][fast + "_minus_" + stock], out, path + "/" + fast + "_minus_" + stock)


def test_all_spatial_regions_own_threeway_fourway_moments_tails_variance(saved):
    a, r, regions = saved["conditional_arrays"], saved["report"], saved["regions"]
    expected_regions = {k for k in regions if not k.startswith(("energy_", "cosine_"))}
    assert set(r["spatial"]) == expected_regions and len(expected_regions) == 117
    missing_control = 0
    for label in sorted(expected_regions):
        region = regions[label]
        group = a["event_row"][region] // 100
        record = r["spatial"][label]
        assert record["entries"] == region.sum() and record["source_blocks"] == len(np.unique(group))
        assert set(record["arms"]) == set(ARMS)
        for coordinate in TAILS:
            oracle = PopulationOracle(a["direction_" + coordinate][region], group, TAILS[coordinate])
            for arm in ARMS:
                keys = dict(new=arm, production="production_" + arm, stock="stockref", control="control")
                values = {k: a[prefix + "_" + coordinate][region] for k, prefix in keys.items()}
                own = {k: a[prefix + "_energy"][region] > 10 for k, prefix in keys.items()}
                common = own["new"] & own["production"] & own["stock"]
                four = common & own["control"]
                output = record["arms"][arm][coordinate]
                path = "/".join((label, arm, coordinate))
                assert output["control_unselected_in_three_way"] == np.sum(common & ~own["control"])
                if label == "all":
                    exact(common, a[arm + "_common_three_way_selected"])
                    assert common.sum() == (1626 if arm == "geometric" else 1648)
                    missing_control += np.sum(common & ~own["control"])
                oracle.verify(output["own"], values, own, keys, path + "/own")
                three = {key: value for key, value in values.items() if key != "control"}
                oracle.verify(output["common_three_way"], three, {k: common for k in three}, keys, path + "/common_three_way")
                oracle.verify(output["common_four_way"], values, {k: four for k in values}, keys, path + "/common_four_way")
    assert missing_control > 0, "Regression guard: the primary three-way population must not silently acquire control selection"


def test_all_css_stages_regions_own_and_identical_positive_masks(saved):
    a, d, old, r, regions = [saved[k] for k in ("conditional_arrays", "diagnostics", "old", "report", "regions")]
    expected = {k for k in regions if not k.startswith(("cell_", "energy_", "cosine_", "disk0_radius", "disk1_radius"))}
    assert set(r["css_stages"]) == expected and len(expected) == 13
    for label in sorted(expected):
        region = regions[label]
        group = a["event_row"][region] // 100
        record = r["css_stages"][label]
        assert record["entries"] == region.sum() and record["source_blocks"] == len(np.unique(group))
        assert set(record["stages"]) == {"donor", *ARMS}
        for coordinate in TAILS:
            oracle = PopulationOracle(a["direction_" + coordinate][region], group)
            for stage in ("donor", *ARMS):
                sources = dict(new=(d, stage), production=(old, stage), original=(d, "original"))
                keys = {k: k + "_" + name for k, (_, name) in sources.items()}
                values = {k: source[f"css_{name}_centroid_{coordinate}"][region] for k, (source, name) in sources.items()}
                own = {k: source[f"css_{name}_centroid_energy"][region] > 0 for k, (source, name) in sources.items()}
                common = own["new"] & own["production"] & own["original"]
                output = record["stages"][stage][coordinate]
                path = f"CSS/{label}/{stage}/{coordinate}"
                oracle.verify(output["own"], values, own, keys, path + "/own", "original")
                oracle.verify(output["common_three_way"], values, {k: common for k in values}, keys, path + "/common", "original")


def test_fallback112_all_queries_pool_expectations_draw_sd_and_support(saved):
    a, c, d, old, p, r, regions = [saved[k] for k in ("conditional_arrays", "clusters", "diagnostics", "old", "pools", "report", "regions")]
    fallback, used = a["direction_fallback"], a["direction_used"]
    assert fallback.sum() == 112 and used.sum() == 2284
    assert np.all(used ^ fallback)
    for source in (p, d):
        exact(source["direction_fallback"], fallback)
        exact(source["direction_used"], used)
    exact(used, p["direction_count"] >= 8)
    exact(p["supported"], np.ones(2396, bool))
    exact(d["pool_donor"], c["sampling_donor"])
    exact(d["pool_donor"][fallback], old["pool_donor"][fallback])
    assert (a["direction_tangent"] < 0).sum() == 28
    assert np.all(fallback[a["direction_tangent"] < 0])
    out = r["fallback"]
    assert out["entries"] == 112 and out["negative_tangent_queries"] == 28 and out["capped_counts_available"] is True
    assert set(out["regions"]) == set(regions) and len(regions) == 125
    # Recompute conditional moments from saved finite-pool outcomes, without
    # opening the source library or drawing/replaying a single random number.
    for outcome in ("energy", "empty", "weak"):
        means, variances = [], []
        for eligible, size in zip(p["eligible"], p["size"]):
            members = eligible[:int(size)]
            assert np.all(members >= 0)
            values = p["outcome_" + outcome][members].astype(float)
            means.append(values.mean()); variances.append(values.var())
        check(d["pool_expected_" + outcome], means, "pool expected " + outcome)
        check(d["pool_variance_" + outcome], variances, "pool variance " + outcome)
        check(p["prob_" + outcome], means)
        actual = p["outcome_" + outcome][d["pool_donor"]]
        exact(d["pool_actual_" + outcome], actual)
        for kind in ("expected", "variance", "actual"):
            exact(d[f"pool_{kind}_{outcome}"][fallback], old[f"pool_{kind}_{outcome}"][fallback])
    queries = {"event_row": a["event_row"][fallback], "cell_id": a["cell_id"][fallback], "direction_count": p["direction_count"][fallback]}
    queries.update({k: c["entry_" + k][fallback] for k in ("disk", "energy", "cosine", "radius")})
    queries.update({k: a[k][fallback] for k in ("direction_radial", "direction_tangent")})
    queries.update({k: c["sampling_" + k][fallback] for k in ("radius_pool_size", "bin_pool_size")})
    queries.update({k: value[fallback] for k, value in a.items() if k.endswith(("_energy", "_truth_ep"))})
    for prefix, source in (("", d), ("production_", old)):
        queries.update({prefix + k: value[fallback] for k, value in source.items()
                        if k.startswith(("pool_expected_", "pool_actual_", "pool_variance_")) or k in ("residual_original", "residual_donor", "residual_original_steps", "residual_donor_steps")})
    check(out["queries"], queries, "fallback/queries")
    labels = {"stock": "stockref", "control": "control", **{name + "_" + arm: prefix + arm for name, prefix in (("new", ""), ("production", "production_")) for arm in ARMS}}
    for label, region in regions.items():
        mask = region & fallback
        group = a["event_row"][mask] // 100
        record = out["regions"][label]
        assert record["entries"] == mask.sum() and record["full_region_entries"] == region.sum()
        assert record["direction_used"] == (region & used).sum() and record["source_blocks"] == len(np.unique(group))
        exact(record["capped_count_histogram_0_to_7"], [np.sum(mask & (p["direction_count"] == i)) for i in range(8)])
        assert set(record["outcomes"]) == set(labels)
        for name, prefix in labels.items():
            e = a[prefix + "_energy"][mask]
            expected = dict(zero=np.sum(e == 0), weak_le10=np.sum(e <= 10), selected=np.sum(e > 10),
                            energy=mean_pair(e, a["stockref_energy"][mask], group),
                            truth_ep=mean_pair(a[prefix + "_truth_ep"][mask], a["stockref_truth_ep"][mask], group))
            check(record["outcomes"][name], expected, f"fallback/{label}/{name}")
        for name, source in (("new", d), ("production", old)):
            for outcome in ("energy", "empty", "weak"):
                actual, expected, variance = [source[f"pool_{kind}_{outcome}"][mask] for kind in ("actual", "expected", "variance")]
                original = source["residual_original"]
                if outcome == "empty":
                    original = source["residual_original_steps"] == 0
                elif outcome == "weak":
                    original = original <= 10
                original = original[mask]
                sd = np.sqrt(variance.sum())
                want = dict(actual_total=actual.sum(), expected_total=expected.sum(), original_total=original.sum(),
                            conditional_draw_sd_total=sd, conditional_draw_sd_mean=sd / len(actual) if len(actual) else None,
                            actual_minus_expected=mean_pair(actual, expected, group), actual_minus_original=mean_pair(actual, original, group),
                            expected_minus_original=mean_pair(expected, original, group))
                check(record["pool"][name][outcome], want, f"fallback/{label}/{name}/{outcome}")
    census = r["saved_capped_support"]
    assert census["categories"] == ["zero", "one_to_seven", "at_least_eight"]
    expected_regions = {"all", "direction_used", "direction_fallback", *[f"cell_{i:02d}" for i in range(96)]}
    assert set(census["regions"]) == expected_regions
    for label in expected_regions:
        mask = regions[label]
        count = p["direction_count"][mask]
        check(census["regions"][label], dict(entries=mask.sum(), used=(mask & used).sum(), fallback=(mask & fallback).sum(),
              count_categories=[np.sum(count == 0), np.sum((count > 0) & (count < 8)), np.sum(count >= 8)]), "support/" + label)
    allpool = out["regions"]["all"]["pool"]["new"]
    assert allpool["empty"]["actual_total"] == 9 and allpool["empty"]["expected_total"] == 4.875
    assert allpool["weak"]["actual_total"] == 24 and allpool["weak"]["expected_total"] == 23.625
