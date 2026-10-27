"""Local synthetic ROOT checks for paired-shard uncertainty, with no event loops."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import uproot
from uproot.writing import identify

from valhist.paired_shapes import bootstrap_region, run


KEY = "Validation/CrvReco/hits"


def histogram(values, *, edges=None, title="fills", variances=None):
    data = np.asarray(values, dtype=np.float64)
    if edges is None:
        edges = [np.arange(s - 1, dtype=float) for s in data.shape]
    axes = [identify.to_TAxis("xyz"[i] + "axis", "", len(e) - 1, e[0], e[-1], np.asarray(e, float))
            for i, e in enumerate(edges)]
    args = dict(fName="h", fTitle=title, data=data.ravel(order="F"), fEntries=10,
                fTsumw=10, fTsumw2=10, fTsumwx=0, fTsumwx2=0,
                fSumw2=None if variances is None else np.asarray(variances, float).ravel(order="F"),
                fXaxis=axes[0])
    if data.ndim >= 2:
        args.update(fYaxis=axes[1], fTsumwy=0, fTsumwy2=0, fTsumwxy=0)
    if data.ndim >= 3:
        args.update(fZaxis=axes[2], fTsumwz=0, fTsumwz2=0, fTsumwxz=0, fTsumwyz=0)
    return getattr(identify, f"to_TH{data.ndim}x")(**args)


def named(title):
    obj = uproot.models.TNamed.Model_TNamed.empty()
    obj._deeply_writable = True
    obj._bases.append(uproot.models.TObject.Model_TObject.empty())
    obj._members.update(fName="binningVersion", fTitle=title)
    return obj


def pairs(tmp_path, ref=None, cand=None, key=KEY):
    ref = ref if ref is not None else [[0, 2, 4, 0], [0, 5, 3, 0]]
    cand = cand if cand is not None else ref
    rows = []
    for i, (a, b) in enumerate(zip(ref, cand), start=1):
        row = dict(shard_id=f"001400_{i:08d}")
        for arm, values in (("ref", a), ("cand", b)):
            path = tmp_path / f"{arm}.001400_{i:08d}.root"
            with uproot.recreate(path) as root:
                root[key] = histogram(values)
                root["crvRecoDQM/binningVersion"] = named("v1")
            row[arm] = path.name
        rows.append(row)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps(rows))
    return manifest, rows


def write_rows(path, rows):
    path.write_text(json.dumps(rows))


def audit(tmp_path, manifest, **kwargs):
    return run(manifest, tmp_path / "result.json", replicas=300, seed=71, batch_size=37, **kwargs)


def events(path, *, subrun=1, ids=(1, 2, 3)):
    ids = np.asarray(ids, np.int32)
    with uproot.update(path) as root:
        root.mktree("crvdump/events", {k: "int32" for k in ("run", "subrun", "event")})
        root["crvdump/events"].extend(dict(run=np.full(ids.size, 1400, np.int32),
                                          subrun=np.full(ids.size, subrun, np.int32), event=ids))


def test_normalization_is_not_shape():
    a = np.array([[1, 2, 6], [9, 2, 4], [3, 7, 2], [2, 1, 9]])
    r = bootstrap_region(a, 3 * a, replicas=400, seed=2)
    assert r["rate_ratio"]["value"] == 3
    assert r["rate_ratio"]["percentile_ci"] == [3, 3]
    assert r["shape"]["total_variation"] == 0
    assert r["shape"]["fraction_difference_bootstrap_rms"] == pytest.approx([0, 0, 0], abs=1e-15)
    assert r["shape"]["centered_tv_null"]["tail_probability_plus_one"] == 1
    b = a[:, ::-1]
    shape = bootstrap_region(a, b, replicas=400, seed=2)
    assert shape["rate_ratio"]["percentile_ci"] == [1, 1]
    assert shape["shape"]["total_variation"] > 0


def test_paired_correlation_and_direct_weight_calculation():
    a = np.array([[1, 10, 2], [90, 3, 5], [5, 8, 30]], float)
    b = a + [[2, 0, 1], [5, 0, 0], [0, 3, 0]]
    draws, seed = 503, 72
    w = np.random.default_rng(seed).multinomial(3, [1 / 3] * 3, size=draws)
    ar, br = w @ a, w @ b
    d = b.sum(axis=0) / b.sum() - a.sum(axis=0) / a.sum()
    ds = br / br.sum(axis=1, keepdims=True) - ar / ar.sum(axis=1, keepdims=True)
    null = np.abs(ds - d).sum(axis=1) / 2
    r = bootstrap_region(a, b, replicas=draws, seed=seed, batch_size=13)
    assert r["rate_ratio"]["percentile_ci"] == pytest.approx(np.quantile(br.sum(axis=1) / ar.sum(axis=1), [.025, .975]))
    assert r["shape"]["fraction_difference_bootstrap_rms"] == pytest.approx(np.sqrt(((ds - d) ** 2).mean(axis=0)))
    assert r["shape"]["centered_tv_null"]["quantiles"] == pytest.approx(np.quantile(null, [.5, .9, .95, .99]))
    exact = bootstrap_region(a, a, replicas=draws, seed=seed)
    broken = bootstrap_region(a, a[::-1], replicas=draws, seed=seed)
    assert exact["shape"]["fraction_difference_bootstrap_rms"] == [0, 0, 0]
    assert max(broken["shape"]["fraction_difference_bootstrap_rms"]) > 0.1
    assert exact["rate_ratio"]["percentile_ci"] == [1, 1]
    assert broken["rate_ratio"]["percentile_ci"][0] < 1 < broken["rate_ratio"]["percentile_ci"][1]


def test_batch_size_invariance():
    a = [[1, 3], [3, 2], [9, 4]]
    b = [[2, 4], [2, 2], [9, 1]]
    small = bootstrap_region(a, b, replicas=701, batch_size=3)
    large = bootstrap_region(a, b, replicas=701, batch_size=256)
    assert small["rate_ratio"] == large["rate_ratio"]
    assert small["shape"]["centered_tv_null"] == large["shape"]["centered_tv_null"]
    assert small["shape"]["fraction_difference_bootstrap_rms"] == pytest.approx(large["shape"]["fraction_difference_bootstrap_rms"])


def test_zero_denominator_replicas_are_not_silently_discarded():
    r = bootstrap_region([[0, 0], [1, 0]], [[0, 0], [2, 0]], replicas=1000)
    assert 150 < r["zero_total_replicas"]["both"] < 350
    assert r["rate_ratio"]["status"] == "conditional_on_defined_replicas"
    assert r["shape"]["valid_replicas"] + r["shape"]["undefined_zero_either_total_replicas"] == 1000
    assert r["rate_ratio"]["percentile_ci"] == [2, 2]
    zero = bootstrap_region([[0, 0], [0, 0]], [[1, 0], [1, 0]], replicas=30)
    assert zero["rate_ratio"]["value"] is None
    assert zero["shape"]["total_variation"] is None
    assert zero["rate_ratio"]["undefined_zero_reference_replicas"] == 30
    empty_candidate = bootstrap_region([[1, 0], [1, 0]], [[0, 0], [0, 0]], replicas=30)
    assert empty_candidate["rate_ratio"]["percentile_ci"] == [0, 0]
    assert empty_candidate["shape"]["status"] == "undefined_observed_zero_total"
    json.dumps(zero, allow_nan=False)


@pytest.mark.parametrize("a,b", [([[1, -1], [2, 3]], [[1, 1], [2, 3]]),
                                  ([[1, np.nan], [1, 2]], [[1, 2], [1, 2]]),
                                  ([[1, 2]], [[1, 2]]), ([[1, 2], [1, 2]], [[1], [2]]),
                                  ([[1, 2], [1, 2]], [[1, np.inf], [1, 2]]),
                                  ([[1e308, 1e308], [1e308, 1e308]], [[1, 1], [1, 1]])])
def test_invalid_arrays_rejected(a, b):
    with pytest.raises(ValueError):
        bootstrap_region(a, b, replicas=20)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_flows_axes_and_strict_json(tmp_path, ndim):
    a = np.ones((4,) * ndim)
    b = a.copy()
    b[(0,) * ndim] += 5
    manifest, rows = pairs(tmp_path, [a, a], [b, b])
    originals = {p: p.read_bytes() for p in tmp_path.glob("*.root")}
    report = audit(tmp_path, manifest)
    h = report["histograms"][KEY]
    assert h["regions"]["visible"]["shape"]["total_variation"] == 0
    assert h["regions"]["inclusive"]["shape"]["total_variation"] > 0
    assert h["flow_totals"]["cand"] - h["flow_totals"]["ref"] == 10
    assert h["regions"]["visible"]["bins"] == 2 ** ndim
    assert "event_pairing_and_completeness_unverified" in report["shards"][0]["pairing_evidence"]
    assert report["metadata"]["crvRecoDQM/binningVersion"]["title"] == "v1"
    assert report["multiplicity"]["adjustment"] == "none"
    assert "pass" not in report
    for path, content in originals.items():
        assert path.read_bytes() == content
    json.loads((tmp_path / "result.json").read_text(), parse_constant=lambda s: pytest.fail(s))
    with pytest.raises(FileExistsError):
        audit(tmp_path, manifest)


def test_exact_non_count_allowlist_and_missing_keys(tmp_path):
    key = "crvRecoDQM/ambiguous"
    manifest, rows = pairs(tmp_path, key=key)
    with pytest.raises(ValueError, match="non_count"):
        audit(tmp_path, manifest)
    with pytest.raises(ValueError, match="unused count-key"):
        audit(tmp_path, manifest, count_keys=[key, "not_present"])
    report = audit(tmp_path, manifest, count_keys=[key])
    assert report["explicit_count_keys"] == [key]
    (tmp_path / "result.json").unlink()
    with uproot.update(tmp_path / rows[1]["cand"]) as root:
        root[key + "_extra"] = histogram([0, 1, 2, 0])
    with pytest.raises(ValueError, match="missing/different"):
        audit(tmp_path, manifest, count_keys=[key])


def test_explicit_exclusion_of_missing_object(tmp_path):
    manifest, rows = pairs(tmp_path)
    extra = "Validation/CrvReco/derived_rate"
    with uproot.update(tmp_path / rows[0]["ref"]) as root:
        root[extra] = histogram([0, 1, 2, 0])
    report = audit(tmp_path, manifest, exclude_keys=[extra])
    assert report["excluded_keys"] == [extra]


@pytest.mark.parametrize("change,match", [("edges", "metadata mismatch"), ("class", "metadata mismatch"),
    ("title", "metadata mismatch"), ("tnamed", "TNamed metadata"), ("negative", "negative"),
    ("nan", "invalid source"), ("sumw2", "invalid source"), ("string", "unsupported")])
def test_invalid_booking_metadata_and_source_fail_closed(tmp_path, change, match):
    manifest, rows = pairs(tmp_path)
    with uproot.update(tmp_path / rows[1]["cand"]) as root:
        if change == "edges":
            root[KEY] = histogram([0, 5, 3, 0], edges=[np.array([0., 1., 2.0000000001])])
        elif change == "class":
            root[KEY] = histogram(np.ones((4, 4)))
        elif change == "title":
            root[KEY] = histogram([0, 5, 3, 0], title="different fills")
        elif change == "tnamed":
            root["crvRecoDQM/binningVersion"] = named("v2")
        elif change == "negative":
            root[KEY] = histogram([0, -1, 3, 0])
        elif change == "nan":
            root[KEY] = histogram([0, np.nan, 3, 0])
        elif change == "sumw2":
            root[KEY] = histogram([0, 5, 3, 0], variances=[0, -1, 3, 0])
        else:
            root[KEY] = "highest cycle is no longer a histogram"
    with pytest.raises(ValueError, match=match):
        audit(tmp_path, manifest)
    assert not (tmp_path / "result.json").exists()


def test_highest_histogram_cycles(tmp_path):
    manifest, rows = pairs(tmp_path)
    with uproot.update(tmp_path / rows[0]["cand"]) as root:
        root[KEY] = histogram([0, 20, 40, 0])
    report = audit(tmp_path, manifest)
    h = report["histograms"][KEY]
    assert h["sources"][1]["cycle"] == 2
    assert h["regions"]["inclusive"]["total_cand"] == 68


def test_permuted_filenames_not_paired_by_position(tmp_path):
    manifest, rows = pairs(tmp_path)
    rows[0]["cand"], rows[1]["cand"] = rows[1]["cand"], rows[0]["cand"]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="filename shard identity mismatch"):
        audit(tmp_path, manifest)


def test_stable_sort_and_declared_exposure_not_faked(tmp_path):
    manifest, rows = pairs(tmp_path)
    rows.reverse()
    for row in rows:
        row["expected_events"] = 50000
    write_rows(manifest, rows)
    report = audit(tmp_path, manifest)
    assert report["shards"][0]["shard_id"] == "001400_00000001"
    assert report["shards"][0]["ref"]["identity"]["expected_events_status"] == "unverified_histogram_only"


def test_event_census_identity_highest_cycle_and_generic_filenames(tmp_path):
    manifest, rows = pairs(tmp_path)
    for i, row in enumerate(rows, 1):
        row["expected_events"] = 3
        for arm in ("ref", "cand"):
            old = tmp_path / row[arm]
            new = tmp_path / f"{arm}-generic-{i}.root"
            old.rename(new)
            row[arm] = new.name
            events(new, subrun=i, ids=[3, 1, 2])
    # Old event cycle is stale, highest one is authoritative.
    events(tmp_path / rows[0]["cand"], subrun=1, ids=[2, 1, 3])
    write_rows(manifest, rows)
    report = audit(tmp_path, manifest)
    record = report["shards"][0]
    assert "matching_persisted_event_ids" in record["pairing_evidence"]
    assert record["cand"]["identity"]["event_census"]["cycle"] == 2
    assert record["ref"]["identity"]["expected_events_status"] == "verified"


@pytest.mark.parametrize("problem", ["wrong_shard", "different_events", "duplicates", "wrong_count", "unidentifiable"])
def test_identity_evidence_failures(tmp_path, problem):
    manifest, rows = pairs(tmp_path)
    if problem == "unidentifiable":
        old = tmp_path / rows[0]["ref"]
        old.rename(tmp_path / "anonymous.root")
        rows[0]["ref"] = "anonymous.root"
    else:
        for i, row in enumerate(rows, 1):
            events(tmp_path / row["ref"], subrun=i)
            events(tmp_path / row["cand"], subrun=(7 if problem == "wrong_shard" else i),
                   ids=([1, 2, 4] if problem == "different_events" else [1, 1, 3] if problem == "duplicates" else [1, 2, 3]))
        if problem == "wrong_count":
            rows[0]["expected_events"] = 4
    write_rows(manifest, rows)
    with pytest.raises(ValueError):
        audit(tmp_path, manifest)


@pytest.mark.parametrize("problem", ["one", "duplicate", "file_reuse", "unknown", "glob", "zero_events"])
def test_manifest_rejections(tmp_path, problem):
    manifest, rows = pairs(tmp_path)
    if problem == "one":
        rows = rows[:1]
    elif problem == "duplicate":
        rows[1]["shard_id"] = rows[0]["shard_id"]
    elif problem == "file_reuse":
        rows[1]["ref"] = rows[0]["ref"]
    elif problem == "unknown":
        rows[0]["expected_event"] = 10
    elif problem == "glob":
        rows[0]["ref"] = "*.root"
    else:
        rows[0]["expected_events"] = 0
    write_rows(manifest, rows)
    with pytest.raises(ValueError):
        audit(tmp_path, manifest)


def test_cli_new_output_and_error_exit(tmp_path):
    manifest, rows = pairs(tmp_path)
    out = tmp_path / "cli.json"
    command = [sys.executable, "-B", "-m", "valhist.paired_shapes", str(manifest), "--out", str(out),
               "--replicas", "30", "--seed", "7"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["shards"] == 2
    again = subprocess.run(command, capture_output=True, text=True, env=env)
    assert again.returncode == 2
    assert "output must be new" in again.stderr


def test_profiles_never_promoted_even_with_override(tmp_path):
    manifest, rows = pairs(tmp_path)
    p = identify.to_TProfile(fName="p", fTitle="profile", data=np.array([0., 2., 0.]),
        fEntries=2, fTsumw=2, fTsumw2=2, fTsumwx=1, fTsumwx2=1, fTsumwy=2,
        fTsumwy2=2, fSumw2=np.array([0., 2., 0.]), fBinEntries=np.array([0., 2., 0.]),
        fBinSumw2=np.array([0., 2., 0.]), fXaxis=identify.to_TAxis("xaxis", "", 1, 0, 1))
    with uproot.update(tmp_path / rows[0]["ref"]) as root:
        root[KEY] = p
    with pytest.raises(ValueError, match="profile"):
        audit(tmp_path, manifest, count_keys=[KEY])


def test_alias_collision_is_not_arbitrarily_selected(tmp_path):
    manifest, rows = pairs(tmp_path)
    with uproot.update(tmp_path / rows[0]["ref"]) as root:
        root["Validation/CrvReco_other/hits"] = histogram([0, 2, 4, 0])
    with pytest.raises(ValueError, match="alias collision"):
        audit(tmp_path, manifest, aliases=["CrvReco_other=CrvReco"])


def test_tnamed_validation_independent_of_histogram_selection(tmp_path):
    manifest, rows = pairs(tmp_path)
    with uproot.update(tmp_path / rows[1]["cand"]) as root:
        root["crvRecoDQM/binningVersion"] = named("different version")
    with pytest.raises(ValueError, match="TNamed metadata"):
        audit(tmp_path, manifest, select="^Validation/")


def test_missing_metadata_is_error(tmp_path):
    manifest, rows = pairs(tmp_path)
    # Recreate without the metadata: uproot's nested-directory deletion does
    # not reliably remove the key from the reopened file's directory listing.
    with uproot.recreate(tmp_path / rows[1]["cand"]) as root:
        root[KEY] = histogram([0, 5, 3, 0])
    with pytest.raises(ValueError, match="TNamed metadata"):
        audit(tmp_path, manifest)


@pytest.mark.parametrize("options", [{"replicas": 1}, {"replicas": True}, {"seed": -1},
                                     {"batch_size": 0}, {"confidence": 1}, {"confidence": np.nan}])
def test_invalid_bootstrap_options(options):
    with pytest.raises(ValueError):
        bootstrap_region([[1, 1], [1, 1]], [[1, 1], [1, 1]], **options)
