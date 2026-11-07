"""Opt-in saved-output audit, independent NumPy oracle (no production imports).

CALO_DIRECTION_VERIFY_BASE must name the existing CAL2_20261105 directory.
Reads saved NPZ/JSON and hashes declared source files only; writes no artifacts,
uses no RNG, and performs no placement, response, reconstruction or ROOT reads.
The <=25 mm / >=8 census is feasibility for a *future* proposal, not a sampler.
"""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest


BASE = os.environ.get("CALO_DIRECTION_VERIFY_BASE")
pytestmark = pytest.mark.skipif(not BASE, reason="set CALO_DIRECTION_VERIFY_BASE for saved-output verification")
PINS = {
    "production_candidate": "ed177ed1018e3bf515160c965d435b2bd4bd22b35b8d7bf6ed0db02c39a5d1f7",
    "production_candidate_support": "e9e9206104804c46f211e12093001756c359a01438ccdbc27fab872645a4936e",
    "production_candidate_trial": "dee6f99faa4882a3ac98ded6a9a7d160aa011e1d739aab78235c443adc4eaf5a",
}


def digest(path):
    path = Path(path)
    assert path.suffix in {".json", ".npz", ".py"}, path
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def archive(path, select=lambda key: True):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files if select(key)}


def equal(actual, expected, label=""):
    if np.asarray(expected).dtype.kind == "f":
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-9, err_msg=label)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=label)


def check_tree(actual, expected, label=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key in expected:
            check_tree(actual[key], expected[key], label + "/" + key)
    elif expected is None:
        assert actual is None, label
    else:
        equal(actual, expected, label)


def stats(values):
    values = np.asarray(values)
    x = values[np.isfinite(values)]
    out = {"n": len(x), "unavailable": values.size - len(x)}
    for key, fraction in zip(("min", "p50", "p90", "p95", "max"), (0, .5, .9, .95, 1)):
        out[key] = float(np.quantile(x, fraction)) if len(x) else None
    out["mean"] = float(np.mean(x)) if len(x) else None
    return out


def paired(fast, stock, group):
    if not len(fast):
        return dict(fast=None, stock=None, difference=None, error=None)
    difference = np.asarray(fast, float) - np.asarray(stock, float)
    groups = np.unique(group)
    leave = np.array([difference[group != g].mean() for g in groups]) if len(groups) > 1 else None
    error = None if leave is None else np.sqrt((len(groups) - 1) * np.mean((leave - leave.mean()) ** 2))
    return dict(fast=float(np.mean(fast)), stock=float(np.mean(stock)),
                difference=float(difference.mean()), error=error)


def cell_coordinates(table, lib):
    coordinates, outside = [], np.zeros(len(table["energy"]), bool)
    for name in ("energy", "cosine", "radius"):
        value, edges = table[name], lib["bins_" + name]
        # Count passed internal boundaries; inclusive upper edge remains inside.
        coordinates.append(np.sum(value[:, None] >= edges[None, 1:-1], axis=1))
        outside |= (value < edges[0]) | (value > edges[-1])
    return np.column_stack(coordinates), outside


def local(table, lib):
    disk_ids = list(lib["geometry_disk_disk"])
    answer, axis = [], []
    for position, direction, disk in zip(table["position"], table["direction"], table["disk"]):
        slot = disk_ids.index(disk)
        x = position[0] - lib["geometry_disk_origin_x"][slot]
        y = position[1] - lib["geometry_disk_origin_y"][slot]
        radius = np.hypot(x, y)
        ex, ey = (x / radius, y / radius) if radius else (1., 0.)
        ux, uy, uz = direction
        answer.append((ex * ux + ey * uy, ex * uy - ey * ux, uz))
        axis.append(radius == 0)
    return np.array(answer), np.array(axis)


def masks(query, cells, direction, fallback):
    n = len(query["energy"])
    out = dict(all=np.ones(n, bool), fallback=fallback)
    for disk in (0, 1):
        out[f"disk{disk}"] = query["disk"] == disk
    for key, edges in (("radius", (0, 450, 550, np.inf)), ("energy", (0, 50, 80, 95, np.inf)),
                       ("cosine", (0, .4, .7, np.inf))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            out[f"{key}_{lo}_{hi}"] = (query[key] >= lo) & (query[key] < hi)
    for j, key in enumerate(("radial", "tangent")):
        out[key + "_negative"] = direction[:, j] < 0
        out[key + "_nonnegative"] = direction[:, j] >= 0
    high = (query["disk"] == 1) & (query["radius"] < 450) & (query["energy"] >= 95)
    out.update(highEinnerdisk1=high, problem109=high & (query["cosine"] >= .7))
    flat = ((cells[:, 0] * 4 + cells[:, 1]) * 3 + cells[:, 2]) * 4 + cells[:, 3]
    for cell in range(96):
        out[f"cell_{cell:02d}"] = flat == cell
    return out


def test_saved_direction_support_independent_oracle(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-output verification must not draw")
    for module, names in ((np.random, ("default_rng", "RandomState", "random", "choice", "uniform", "seed")),
                          (random, ("Random", "SystemRandom", "random", "choice", "choices", "uniform", "seed"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    base = Path(BASE)
    summaries = {name: json.loads((base / name / "summary.json").read_text()) for name in PINS}
    watched = {}
    for name, pin in PINS.items():
        path = base / name / "summary.json"
        assert digest(path) == pin
        watched[path] = pin
    output = base / "production_direction_support"
    summary = json.loads((output / "summary.json").read_text())
    watched[output / "summary.json"] = digest(output / "summary.json")
    numbers = json.loads((output / "numbers.json").read_text())
    assert summary["input_sha256"] == numbers["input_sha256"]
    for path, sha in summary["input_sha256"].items():
        assert digest(path) == sha, path
        watched[Path(path)] = sha
    for name, sha in summary["output_sha256"].items():
        assert digest(output / name) == sha, name
        watched[output / name] = sha
    for directory, filename in (("production_candidate", "library.npz"),
                                ("production_candidate_support", "candidate_pools.npz"),
                                ("production_candidate_trial", "clusters.npz")):
        path = base / directory / filename
        assert digest(path) == summaries[directory]["output_sha256"][filename]
    lib = archive(base / "production_candidate/library.npz", lambda k: k.startswith(("entry_", "bins_", "geometry_disk_")) or k in ("offsets", "step_energy_birks", "step_disk"))
    pool = archive(base / "production_candidate_support/candidate_pools.npz")
    saved = archive(base / "production_candidate_trial/clusters.npz", lambda k: k.startswith("entry_") or k in ("full_ids", "event_row", "query_event_rows"))
    actual = archive(output / "arrays.npz")
    query = {k[6:]: v for k, v in saved.items() if k.startswith("entry_")}
    donor = {k[6:]: v for k, v in lib.items() if k.startswith("entry_")}
    n, nd = len(query["energy"]), len(donor["energy"])
    equal(saved["event_row"], np.arange(len(saved["full_ids"])))
    equal(query["event_row"], saved["query_event_rows"])
    equal(query["event_row"], pool["event_row"])
    assert len(np.unique(query["event_row"])) == n
    assert np.all((query["event_row"] >= 0) & (query["event_row"] < len(saved["full_ids"])))
    # Authenticate and compare full source identifiers without opening source ROOT.
    source_path = base / "parent_audit/production_bound_final/dts.mu2e.CeEndpoint.MDC2025ba.001430_00000000.art/bound/expected.npz"
    source_sha = summaries["production_candidate_support"]["input_sha256"][str(source_path)]
    assert digest(source_path) == source_sha
    watched[source_path] = source_sha
    source = archive(source_path)
    for key in ("run", "subrun", "event"):
        equal(saved["full_ids"][key], source[key])
        equal(actual["full_ids"][key][actual["event_row"]], source[key][query["event_row"]])
    assert len(np.unique(saved["full_ids"])) == len(saved["full_ids"])
    equal(lib["geometry_disk_disk"], [0, 1])
    equal(np.column_stack([lib["geometry_disk_axis_z_" + key] for key in "xyz"]), [[0., 0., 1.]] * 2)
    for table in (query, donor):
        np.testing.assert_allclose(np.linalg.norm(table["direction"], axis=1), 1., atol=1e-6, rtol=0)
    qc, outside = cell_coordinates(query, lib)
    dc, donor_outside = cell_coordinates(donor, lib)
    equal(dc, lib["entry_cell"])
    equal(donor_outside, lib["entry_outside_bins"])
    occupied, inverse, counts = np.unique(np.column_stack((donor["disk"], dc)), axis=0, return_inverse=True, return_counts=True)
    chosen = np.full(n, -1, int)
    for i in range(n):
        candidates = [j for j, label in enumerate(occupied) if label[0] == query["disk"][i]]
        if candidates:
            chosen[i] = min(candidates, key=lambda j: (sum(int(a - b) ** 2 for a, b in zip(qc[i], occupied[j, 1:])), j))
    supported = chosen >= 0
    full_size = np.array([counts[c] if c >= 0 else 0 for c in chosen])
    exact = np.array([c >= 0 and np.array_equal(qc[i], occupied[c, 1:]) for i, c in enumerate(chosen)])
    fallback = outside | ~exact
    cells = np.column_stack((query["disk"], qc))
    for key, value in dict(query_cell=cells, chosen_cell=chosen, occupied=occupied, library_cell=inverse,
                           counts=counts, full_size=full_size, exact_size=np.where(exact, full_size, 0),
                           supported=supported, fallback=fallback, outside=outside,
                           substituted=supported & ~exact).items():
        equal(pool[key], value, key)
    q, qa = local(query, lib)
    d, da = local(donor, lib)
    expected = dict(query_direction=q, donor_direction=d, axis_query=qa, axis_donor=da,
                    event_row=query["event_row"], full_ids=saved["full_ids"], query_cell=cells,
                    supported=supported, original_fallback=fallback)
    for kind in ("sign", "compatible"):
        for field in ("count", "baseline_count", "within25_count"):
            expected[kind + "_" + field] = np.zeros(n, int)
        for field in ("nearest_radius", "last_radius"):
            expected[kind + "_" + field] = np.full(n, np.nan)
    baseline, compatible = np.full((n, 8), -1, int), np.full((n, 8), -1, int)
    # Pair arrays are cell-vectorized; sorting is an independent scalar row oracle.
    for cell in np.unique(chosen[supported]):
        targets, rows = np.flatnonzero(chosen == cell), np.flatnonzero(inverse == cell)
        distance = np.abs(query["radius"][targets, None] - donor["radius"][rows])
        dots = np.einsum("ik,jk->ij", q[targets], d[rows])
        angle = np.degrees(np.arccos(np.clip(dots, -1., 1.)))
        sign = ((q[targets, None, :2] >= 0) == (d[rows, :2] >= 0)).all(axis=2)
        good = sign & (angle <= 10.)
        for j, i in enumerate(targets):
            ordered = sorted(range(len(rows)), key=lambda k: (distance[j, k], int(rows[k])))
            base_slots = ordered[:8]
            baseline[i, :len(base_slots)] = sorted(rows[base_slots])
            for kind, mask in (("sign", sign[j]), ("compatible", good[j])):
                slots = [k for k in ordered if mask[k]][:8]
                expected[kind + "_count"][i] = mask.sum()
                expected[kind + "_within25_count"][i] = np.count_nonzero(mask & (distance[j] <= 25.))
                expected[kind + "_baseline_count"][i] = np.count_nonzero(mask[base_slots])
                if slots:
                    expected[kind + "_nearest_radius"][i] = distance[j, slots[0]]
                    expected[kind + "_last_radius"][i] = distance[j, slots[-1]]
                if kind == "compatible":
                    compatible[i, :len(slots)] = sorted(rows[slots])
    direction_fallback = supported & (expected["compatible_count"] == 0)
    witness = compatible.copy()
    witness[direction_fallback] = baseline[direction_fallback]
    expected.update(baseline_eligible=baseline, baseline_size=(baseline >= 0).sum(axis=1),
                    compatible_eligible=compatible, witness_eligible=witness,
                    witness_size=(witness >= 0).sum(axis=1), direction_fallback=direction_fallback)
    equal(pool["eligible"], baseline, "original K8 independent radius ordering")
    equal(pool["size"], expected["baseline_size"])
    for arm, ids in (("baseline", baseline), ("witness", witness)):
        valid, safe = ids >= 0, np.maximum(ids, 0)
        for key in ("radius", "energy", "cosine"):
            expected[arm + "_" + key + "_delta"] = np.where(valid, donor[key][safe] - query[key][:, None], np.nan)
        for j, key in enumerate(("radial", "tangent")):
            expected[arm + "_" + key + "_delta"] = np.where(valid, d[safe, j] - q[:, None, j], np.nan)
        angle = np.degrees(np.arccos(np.clip(np.einsum("ik,ijk->ij", q, d[safe]), -1., 1.)))
        expected[arm + "_angle_deg"] = np.where(valid, angle, np.nan)
    offsets = lib["offsets"]
    assert offsets[0] == 0 and offsets[-1] == len(lib["step_energy_birks"])
    assert np.all(np.diff(offsets) >= 0)
    assert set(np.unique(lib["step_disk"])) == {0, 1}
    # Sum each donor's complete CSS slice, deliberately ignoring deposited disk.
    energy = np.array([lib["step_energy_birks"][lo:hi].sum(dtype=np.float64) for lo, hi in zip(offsets[:-1], offsets[1:])])
    outcomes = dict(energy=energy, empty=np.diff(offsets) == 0, weak=energy <= 10.)
    for key, value in outcomes.items():
        equal(pool["outcome_" + key], value, "bothdisk " + key)
        for arm, ids in (("baseline", baseline), ("witness", witness), ("compatible", compatible)):
            prob = np.array([np.mean(value[row[row >= 0]]) if np.any(row >= 0) else np.nan for row in ids])
            expected[arm + "_prob_" + key] = prob
            if arm == "baseline":
                equal(pool["prob_" + key], prob)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        equal(actual[key], value, key)
    region_masks = masks(query, cells, q, fallback)
    assert region_masks.keys() == numbers["regions"].keys()
    for region, mask in region_masks.items():
        selected = mask & supported
        report = dict(queries=int(mask.sum()), supported=int(selected.sum()), unsupported=int((mask & ~supported).sum()),
                      axis_queries=int((mask & qa).sum()), original_fallback=int((mask & fallback).sum()),
                      direction_fallback=int((mask & direction_fallback).sum()))
        maximum = {}
        for arm in ("baseline", "witness"):
            ids, size = expected[arm + "_eligible"][selected], expected[arm + "_size"][selected]
            valid = ids >= 0
            maximum[arm] = np.max(np.where(valid, np.abs(expected[arm + "_radius_delta"][selected]), -np.inf), axis=1)
            exposure = np.zeros(nd)
            for row in ids:
                members = row[row >= 0]
                exposure[members] += 1. / len(members)
            used = exposure > 0
            section = dict(size=stats(size), eligible_union=int(used.sum()), eligible_empty=int((used & outcomes["empty"]).sum()),
                           eligible_weak=int((used & outcomes["weak"]).sum()), max_expected_exposure=float(exposure.max()), outcomes={})
            for key in outcomes:
                prob = expected[arm + "_prob_" + key][selected]
                original = pool["original_" + key][selected]
                item = dict(expected_sum=float(prob.sum()), probability=stats(prob), original_sum=float(original.sum()),
                            expected_minus_original=paired(prob, original, query["event_row"][selected] // 100))
                if key != "energy":
                    item["conditional_count_sd"] = float(np.sqrt(np.sum(prob * (1 - prob))))
                section["outcomes"][key] = item
            for key in ("radius_delta", "angle_deg", "radial_delta", "tangent_delta", "energy_delta", "cosine_delta"):
                value = np.abs(expected[arm + "_" + key][selected])
                section[key + "_pool_mean_abs"] = stats(np.nansum(value, axis=1) / size)
                section[key + "_pool_max_abs"] = stats(np.max(np.where(valid, value, -np.inf), axis=1))
            report[arm] = section
        report["radius_expansion_mm"] = stats(maximum["witness"] - maximum["baseline"])
        report["compatible_only_outcomes"] = {key: stats(expected["compatible_prob_" + key][selected]) for key in outcomes}
        for kind in ("sign", "compatible"):
            count, within, initial = [expected[kind + "_" + field][selected] for field in ("count", "within25_count", "baseline_count")]
            last = expected[kind + "_last_radius"][selected]
            report[kind] = dict(zero=int((count == 0).sum()), fewer_than8=int((count < 8).sum()), full_count=stats(count),
                                within25_count=stats(within), within25_zero=int((within == 0).sum()), within25_atleast8=int((within >= 8).sum()),
                                baseline_count=stats(initial), baseline_zero=int((initial == 0).sum()), baseline_atleast8=int((initial >= 8).sum()),
                                nearest_radius=stats(expected[kind + "_nearest_radius"][selected]), last_radius=stats(last),
                                eighth_radius=stats(np.where(count >= 8, last, np.nan)))
        check_tree(numbers["regions"][region], report, region)
    assert summary["all"] == numbers["regions"]["all"]
    for doc in (summary, numbers):
        assert doc["sampling_performed"] is False and doc["reconstruction_performed"] is False
    for key, value in dict(angle_degrees=10., diagnostic_radius_mm=25., k=8, donors=nd,
                           empty_donors=int(outcomes["empty"].sum()), weak_donors=int(outcomes["weak"].sum()), axis_donors=int(da.sum())).items():
        equal(numbers[key], value, key)
    # Census only: this future locality-gated policy is NOT constructed or executed.
    local_feasible = expected["compatible_within25_count"] >= 8
    assert (n, nd, int(local_feasible.sum()), int((~local_feasible).sum())) == (2396, 24120, 2284, 112)
    for region in ("cell_41", "cell_89", "problem109"):
        assert np.all(local_feasible[region_masks[region]])
    assert region_masks["tangent_negative"].sum() == 28
    assert not np.any(local_feasible[region_masks["tangent_negative"]])
    for path, sha in watched.items():
        assert digest(path) == sha, "input/output changed during verification: " + str(path)
    all_report = numbers["regions"]["all"]
    print("Verified", n, "queries;", nd, "donors;", len(actual), "arrays;", len(region_masks), "regions;", len(watched), "hashes")
    print("compatible zero/fewer8/local>=8:", all_report["compatible"]["zero"], all_report["compatible"]["fewer_than8"], int(local_feasible.sum()))
    for arm in ("baseline", "witness"):
        print(arm, "union", all_report[arm]["eligible_union"], "expected empty/weak/energy",
              [all_report[arm]["outcomes"][key]["expected_sum"] for key in ("empty", "weak", "energy")])
