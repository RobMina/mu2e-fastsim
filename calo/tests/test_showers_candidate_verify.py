"""Independent read-only verifier of the ONE saved CAL2 candidate/support pair.

Optional local artifacts: CALO_CANDIDATE_VERIFY_BASE=/path/to/CAL2_20261105.
No builds, sampling, subprocesses, reconstruction, output writes or new physics.
Real-data oracle uses full-cell query-by-donor lexsort, never per-query loops;
production's bounded-window radius helper and pool enumerator are not used.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


CANDIDATE_SHA = "ed177ed1018e3bf515160c965d435b2bd4bd22b35b8d7bf6ed0db02c39a5d1f7"
LIBRARY_SHA = "cb411f7c7eaf3364071413ca4d871c7781950a52de63c90c0482936d891d3922"
COVERAGE_SHA = "106350f5bbf2611895094eca1ae3990ccadb567f19d29aa2943ed64f3d01b121"
PARENT_SHA = "52191d77bfa115bba8c2d9fe521ba7a3fe0e53f34e067d77034fd9682f7e419c"
# Snapshot reviewed alongside the externally supplied candidate anchors above.
SUPPORT_SHA = "e9e9206104804c46f211e12093001756c359a01438ccdbc27fab872645a4936e"
FEATURES = ("energy", "cosine", "radius")


def equal(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    np.testing.assert_array_equal(a, b)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def full_cell_oracle(packed, entries):
    """Independent clipped cells, same-disk lexicographic cell ties, full sort K8."""
    def labels(data):
        coordinates = []
        outside = np.zeros(len(data["disk"]), bool)
        for feature in FEATURES:
            edge, value = packed["bins_" + feature], data[feature]
            # Count crossed lower edges, rather than calling library.cells.
            cell = np.sum(value[:, None] >= edge[None, 1:], axis=1)
            coordinates.append(np.minimum(cell, len(edge) - 2))
            outside |= (value < edge[0]) | (value > edge[-1])
        return np.column_stack((data["disk"], *coordinates)).astype(np.int64), outside
    donor = {key: packed["entry_" + key] for key in ("disk", *FEATURES)}
    dl, _ = labels(donor)
    ql, outside = labels(entries)
    occupied, inverse, counts = np.unique(dl, axis=0, return_inverse=True, return_counts=True)
    n = len(ql)
    eligible = np.full((n, 8), -1, np.int64)
    chosen = np.full(n, -1, np.int64)
    full = np.zeros(n, np.int64)
    exact = np.zeros(n, np.int64)
    # Loops are over at most 96 cells, not over real-data queries.
    for label in np.unique(ql, axis=0):
        queries = np.flatnonzero(np.all(ql == label, axis=1))
        cells = np.flatnonzero(occupied[:, 0] == label[0])
        if not len(cells):
            continue
        distance = np.sum((occupied[cells, 1:] - label[1:]) ** 2, axis=1)
        cell = cells[np.argmin(distance)]
        rows = np.flatnonzero(inverse == cell)
        delta = np.abs(entries["radius"][queries, None] - donor["radius"][rows][None, :])
        ties = np.broadcast_to(rows, delta.shape)
        ranked = np.lexsort((ties, delta), axis=1)[:, :8]
        selected = np.sort(rows[ranked], axis=1)
        eligible[queries, :selected.shape[1]] = selected
        chosen[queries] = cell
        full[queries] = len(rows)
        if np.array_equal(occupied[cell], label):
            exact[queries] = len(rows)
    supported = chosen >= 0
    return dict(eligible=eligible, query_cell=ql, chosen_cell=chosen, occupied=occupied,
                library_cell=inverse, counts=counts, size=np.minimum(full, 8), full_size=full,
                exact_size=exact, supported=supported, outside=outside,
                fallback=outside | (exact == 0), substituted=supported & (exact == 0))


def test_full_cell_oracle_radius_ties_order_and_missing_disk():
    radii = np.array([110., 90., 110., 90., 110., 90., 110., 90., 110., 90.])
    packed = {"bins_energy": np.array([0., 50., 80., 95., 110.]),
              "bins_cosine": np.array([0., .4, .7, 1.000001]),
              "bins_radius": np.array([0., 450., 550., 750., 1000.]),
              "entry_radius": radii, "entry_energy": np.full(10, 100.),
              "entry_cosine": np.ones(10), "entry_disk": np.zeros(10, np.int64)}
    query = dict(radius=np.array([100., 500., 100., 1100.]), energy=np.full(4, 100.),
                 cosine=np.ones(4), disk=np.array([0, 0, 1, 0]))
    result = full_cell_oracle(packed, query)
    equal(result["eligible"][0], np.arange(8))
    equal(result["eligible"][1], np.array([0, 1, 2, 3, 4, 5, 6, 8]))
    equal(result["eligible"][2], np.full(8, -1))
    equal(result["fallback"], np.array([False, True, True, True]))
    equal(result["outside"], np.array([False, False, False, True]))


@pytest.fixture
def artifacts(monkeypatch):
    root = os.environ.get("CALO_CANDIDATE_VERIFY_BASE")
    if not root:
        pytest.skip("set CALO_CANDIDATE_VERIFY_BASE for read-only saved-artifact verification")
    from calofastsim.showers import library
    def forbidden(*args, **kwargs):
        pytest.fail("artifact verifier must never rebuild or sample")
    monkeypatch.setattr(library, "build", forbidden)
    monkeypatch.setattr(library, "sample", forbidden)
    for name in ("default_rng", "RandomState", "random", "choice"):
        monkeypatch.setattr(np.random, name, forbidden)
    base = Path(root)
    pins = {}
    def pin(path, sha):
        path = Path(path)
        assert digest(path) == sha, str(path)
        assert path not in pins or pins[path] == sha
        pins[path] = sha
        return path
    def relocate(path):
        path = str(path)
        if "/CAL2_20261105/" in path:
            return base / path.split("/CAL2_20261105/", 1)[1]
        if "/calofastsim/" in path:
            return Path(__file__).resolve().parents[1] / "calofastsim" / path.split("/calofastsim/", 1)[1]
        raise AssertionError("unrecognized nonportable receipt path: " + path)
    candidate = json.loads(pin(base / "production_candidate/summary.json", CANDIDATE_SHA).read_text())
    support = json.loads(pin(base / "production_candidate_support/summary.json", SUPPORT_SHA).read_text())
    for name, report in (("production_candidate", candidate), ("production_candidate_support", support)):
        for relative, sha in report["output_sha256"].items():
            assert Path(relative).name == relative
            pin(base / name / relative, sha)
        for path, sha in report["input_sha256"].items():
            if Path(path).is_absolute():
                pin(relocate(path), sha)
        for module, sha in report["source_sha256"].items():
            if module == "__main__":
                module = "calofastsim.showers." + ("production_candidate" if name == "production_candidate" else "candidate_support")
            assert module.startswith("calofastsim.")
            pin(Path(__file__).resolve().parents[1] / Path(*module.split(".")).with_suffix(".py"), sha)
    assert candidate["coverage_sha256"] == COVERAGE_SHA
    assert candidate["parent_summary_sha256"] == PARENT_SHA
    assert candidate["output_sha256"]["library.npz"] == LIBRARY_SHA
    assert candidate["training_only"] and not candidate["sampled"] and not candidate["legacy_donors_included"]
    assert not support["sampling_performed"] and not support["reconstruction_performed"]
    yield base, candidate, support, pin
    for path, sha in pins.items():
        assert digest(path) == sha, "input changed during verification: " + str(path)


def test_saved_candidate_every_entry_step_and_full_source_ledger(artifacts):
    from calofastsim.showers import export, library, source_events, source_split
    base, summary, _, pin = artifacts
    folder = base / "production_candidate"
    packed, metadata = library.load(folder / "library.npz")
    assert metadata == json.loads((folder / "metadata.json").read_text())
    assert all(summary[key] == value for key, value in metadata.items())
    ledger, expected = npz(folder / "training_ledger.npz"), npz(folder / "training_expected.npz")
    provenance = npz(folder / "donor_provenance.npz")
    assert len(ledger["event"]) == 28265
    assert len(packed["entry_energy"]) == 24120
    assert packed["offsets"][0] == 0 and packed["offsets"][-1] == 1034261
    for filename, actual in (("ledger.npz", ledger), ("expected.npz", expected)):
        original = npz(base / "parent_audit/overlap_final/training" / filename)
        assert actual.keys() == original.keys()
        for key in original:
            equal(actual[key], original[key])
    equal(provenance["donor_row"], np.arange(24120))
    row = packed["entry_event_row"]
    assert np.all(np.diff(row) > 0)
    equal(provenance["global_event_row"], row)
    for key in source_split.OUTPUT_KEY + source_split.PARENT_KEY:
        equal(provenance[key], ledger[key][row])
    assert np.all(ledger["parent_fid"] == "102778799")
    assert np.all(ledger["source_count"] == 1)
    assert np.all(ledger["source_present"]) and np.all(ledger["source_file_resolved"])
    for key in source_split.OUTPUT_KEY:
        equal(ledger[key], expected[key])
    parents = np.rec.fromarrays([ledger[key] for key in source_split.PARENT_KEY])
    assert len(np.unique(parents)) == 28265
    file_order = summary["file_order"]
    assert [int(f["item"].split("_")[-1].split(".")[0]) for f in file_order] == list(range(1, 902, 100))
    assert set(np.unique(provenance["output_fid"])) == {f["fid"] for f in file_order}
    event_offset = entry_offset = step_offset = 0
    for record in file_order:
        table = base / "training_coverage" / record["item"] / "export/calo_library_tables.root"
        e, s, geometry, census, _ = export.read(table, input_mode="primary")
        n, m = len(e["energy"]), len(s["entry"])
        selected = slice(entry_offset, entry_offset + n)
        events = slice(event_offset, event_offset + record["events"])
        assert census["events"] == record["events"]
        numeric, _ = source_events.read(table, {k: v[events] for k, v in expected.items()},
            source_tag="TargetStopResampler::Primary", source_branch="art::EventIDs_TargetStopResampler__Primary.")
        for key in export.EVENT + ("source_run", "source_subrun", "source_event", "source_count"):
            equal(numeric[key], ledger[key][events])
        for key, value in e.items():
            equal(packed["entry_" + key][selected], value + event_offset if key == "event_row" else value)
        equal(provenance["file_ordinal"][selected], np.full(n, record["ordinal"], np.int64))
        equal(provenance["file_event_row"][selected], e["event_row"])
        order = np.argsort(s["entry"], kind="stable")
        step_rows = s["entry"][order]
        expected_offsets = np.r_[0, np.cumsum(np.bincount(step_rows, minlength=n))] + step_offset
        equal(packed["offsets"][entry_offset:entry_offset + n + 1], expected_offsets)
        steps = slice(step_offset, step_offset + m)
        for key, value in s.items():
            if key not in ("entry", "position", "time"):
                equal(packed["step_" + key][steps], value[order])
        restored = e["position"][step_rows] + np.einsum("nij,nj->ni", library.frame(e["direction"])[step_rows],
                                                         packed["step_relative_position"][steps])
        np.testing.assert_allclose(restored, s["position"][order], rtol=0, atol=1e-9)
        np.testing.assert_allclose(e["time"][step_rows] + packed["step_relative_time"][steps],
                                   s["time"][order], rtol=0, atol=1e-9)
        for key in ("energy_g4", "energy_birks"):
            equal(np.bincount(step_rows, weights=packed["step_" + key][steps], minlength=n),
                  np.bincount(s["entry"], weights=s[key], minlength=n))
        for key, value in geometry.items():
            equal(packed["geometry_" + key], value)
        event_offset += record["events"]
        entry_offset += n
        step_offset += m
    assert (event_offset, entry_offset, step_offset) == (28265, 24120, 1034261)
    rows = np.repeat(np.arange(24120), np.diff(packed["offsets"]))
    totals = np.bincount(rows, weights=packed["step_energy_birks"], minlength=24120)
    assert np.sum(np.diff(packed["offsets"]) == 0) == 259
    assert np.sum(totals <= 10) == 835
    assert np.sum(packed["step_disk"] != packed["entry_disk"][rows]) == 115240
    print("CANDIDATE exact: stored=28265 donors=24120 CSS=1034261 empty=259 weak=835 other-disk-CSS=115240; all source rows/scalars/offsets/totals verified")


def test_saved_support_full_cell_oracle_and_historical_exactness(artifacts):
    from calofastsim.showers import export, library, source_events
    base, _, report, pin = artifacts
    query = pin(base / "source_event_validation/development_file0/calo_library_tables.root", report["input_sha256"]["query_tables"])
    e, css, _, census, _ = export.read(query, input_mode="primary")
    assert census["events"] == 2813 and len(e["energy"]) == 2396
    query_dir = base / "parent_audit/production_bound_final/dts.mu2e.CeEndpoint.MDC2025ba.001430_00000000.art/bound"
    expected, ledger = npz(query_dir / "expected.npz"), npz(query_dir / "ledger.npz")
    numeric, _ = source_events.read(query, expected, source_tag="TargetStopResampler::Primary",
                                    source_branch="art::EventIDs_TargetStopResampler__Primary.")
    for key in export.EVENT + ("source_run", "source_subrun", "source_event", "source_count"):
        equal(numeric[key], ledger[key])
    reports = {}
    for name, relative in (("candidate", "production_candidate/library.npz"), ("legacy", "ce_showers_dev.npz")):
        path = pin(base / relative, report["input_sha256"][name + "_library"])
        packed, _ = library.load(path)
        saved = npz(base / "production_candidate_support" / (name + "_pools.npz"))
        oracle = full_cell_oracle(packed, e)
        for key, value in oracle.items():
            equal(saved[key], value)
        equal(saved["event_row"], e["event_row"])
        lengths = np.diff(packed["offsets"])
        donor_rows = np.repeat(np.arange(len(lengths)), lengths)
        energy = np.bincount(donor_rows, weights=packed["step_energy_birks"], minlength=len(lengths))
        outcomes = dict(energy=energy, empty=lengths == 0, weak=energy <= 10)
        original_energy = np.bincount(css["entry"], weights=css["energy_birks"], minlength=2396)
        originals = dict(energy=original_energy, empty=np.bincount(css["entry"], minlength=2396) == 0,
                         weak=original_energy <= 10)
        ids = oracle["eligible"]
        valid = ids >= 0
        for key, value in outcomes.items():
            equal(saved["outcome_" + key], value)
            equal(saved["original_" + key], originals[key])
            gathered = np.zeros(ids.shape)
            gathered[valid] = value[ids[valid]]
            probability = np.divide(gathered.sum(axis=1), oracle["size"], out=np.full(len(ids), np.nan), where=oracle["supported"])
            equal(saved["prob_" + key], probability)
        queries, slots = np.nonzero(valid)
        donors = ids[queries, slots]
        for feature in FEATURES:
            values = np.full(ids.shape, np.nan)
            values[queries, slots] = packed["entry_" + feature][donors] - e[feature][queries]
            equal(saved["pool_" + feature + "_delta"], values)
        origin = np.column_stack([packed["geometry_disk_origin_" + a] for a in "xy"])[e["disk"][queries]]
        source, target = packed["entry_position"][donors, :2] - origin, e["position"][queries, :2] - origin
        phi = np.arctan2(target[:, 1], target[:, 0]) - np.arctan2(source[:, 1], source[:, 0])
        direction = packed["entry_direction"][donors].copy()
        x, y = direction[:, 0].copy(), direction[:, 1].copy()
        direction[:, 0] = np.cos(phi) * x - np.sin(phi) * y
        direction[:, 1] = np.sin(phi) * x + np.cos(phi) * y
        angle = np.full(ids.shape, np.nan)
        angle[queries, slots] = np.rad2deg(np.arccos(np.clip(np.sum(direction * e["direction"][queries], axis=1), -1, 1)))
        equal(saved["pool_angle_deg"], angle)
        if name == "legacy":
            historical = npz(base / "primary_file0_radius8_pools/reviewed/pools.npz")
            for key in report["legacy_pool_parity"]["fields"]:
                value = historical[key]
                if key in ("query_cell", "occupied"):
                    assert value.dtype == np.float64 and np.all(np.isfinite(value))
                    assert np.all((value >= 0) & (value <= [1, 3, 2, 3]))
                    np.testing.assert_array_equal(value, np.floor(value))
                    value = value.astype(np.int64)
                equal(saved[key], value)
        region = report[name]["regions"]["all"]
        reports[name] = {key: region[key] for key in ("queries", "supported", "unsupported", "fallback", "substituted", "unique_eligible_donors")}
        reports[name]["sizes"] = region["count_distributions"]["size"]
        reports[name]["eligible_slots_exact"] = int(valid.sum())
        reports[name]["expected_empty"] = float(saved["prob_empty"].sum())
        reports[name]["expected_weak"] = float(saved["prob_weak"].sum())
        assert region["fallback"] == int(oracle["fallback"].sum())
        assert region["supported"] == int(oracle["supported"].sum())
        assert region["unique_eligible_donors"] == len(np.unique(ids[valid]))
    # Independently intersect complete stored ledgers, including non-front rows.
    training = npz(base / "production_candidate/training_ledger.npz")
    for fields in (("output_fid", "run", "subrun", "event"),
                   ("parent_fid", "source_run", "source_subrun", "source_event")):
        left = np.rec.fromarrays([training[key] for key in fields])
        right = np.rec.fromarrays([ledger[key] for key in fields])
        assert not len(np.intersect1d(left, right))
    proof = json.loads((base / "source_event_validation/independent_check/summary.json").read_text())
    old_report = json.loads((base / "primary_file0_radius8_pools/reviewed/numbers.json").read_text())
    assert proof["input_sha256"]["tables"] == report["input_sha256"]["query_tables"]
    assert proof["input_sha256"]["baseline"] == old_report["inputs"]["tables"]["sha256"]
    assert old_report["inputs"]["library"]["sha256"] == report["input_sha256"]["legacy_library"]
    assert proof["events"] == 2813 and proof["ordered_event_census_exact"]
    assert proof["printer_source_ids_exact"] and proof["comparison"]["all_old_columns_exact"]
    assert report["candidate_query_overlap"]["shared_parent_events"] == 0
    assert report["candidate_query_overlap"]["shared_output_events"] == 0
    print("FULL-CELL ORACLE exact (both pools, probabilities, radius/angle): " + json.dumps(reports, sort_keys=True))
