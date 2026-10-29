"""Synthetic saved JSON only; test pins override production constants explicitly."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from crvfastsim.veto import ce_joint_summary as S


def axis(edges, inclusive=False):
    return dict(edges=list(edges), bins=len(edges) + 1, underflow_index=0, overflow_index=len(edges),
                convention="left closed, right open; explicit underflow/overflow",
                last_finite_upper_inclusive=inclusive)


def packed(arrays):
    return dict(shape=list(arrays[0].shape), **{key: arr.tolist() for key, arr in
                zip((*S.COUNT_FIELDS, S.MULT, S.HIST), arrays)})


def fixture_report():
    axes = {name: axis(S.TC_EDGES if name == S.PROXIES[0] else S.HIT_EDGES) for name in S.PROXIES}
    axes.update({S.PB: axis(S.PB_EDGES), S.MC: axis([0, 1, 2, 3, 4, 5], True), S.TIME: axis(S.TIME_EDGES)})
    shape = (8, 7, 7)
    n, a, t = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    mult = np.zeros((*shape, 5), dtype=np.int64)
    hist = np.zeros((*shape, 22), dtype=np.int64)
    # Fixed synthetic cells: underflow, overflow, zero, one, two, >=4 tracks.
    cells = [((0, 0, 0), [80, 10, 10, 0, 0], 30, 0),
             ((0, 1, 1), [60, 40, 0, 0, 0], 40, 21),
             ((1, 0, 1), [50, 10, 0, 0, 0], 10, 1),
             ((7, 6, 6), [3, 0, 0, 0, 1], 4, 21),
             ((6, 5, 5), [9, 0, 0, 0, 0], 0, 7)]
    for index, m, tracks, time in cells:
        mult[index] = m
        n[index], a[index], t[index] = sum(m), sum(m[1:]), tracks
        hist[(*index, time)] = tracks
    three = (n, a, t, mult, hist)
    ntc_pb = S.reduce_grid(three, 1)
    grids = {S.PROXIES[0] + "__" + S.PB: packed(ntc_pb),
             S.MC + "__" + S.PB: packed(S.reduce_grid(three, 0)),
             "__".join((S.PROXIES[0], S.MC, S.PB)): packed(three)}
    for name in S.PROXIES[1:]:
        # Map original overflow to each proxy's own overflow; leave interior empty.
        arrays = []
        for arr in ntc_pb:
            other = np.zeros((11, *arr.shape[1:]), dtype=np.int64)
            other[:7] = arr[:7]
            other[10] = arr[7]
            arrays.append(other)
        grids[name + "__" + S.PB] = packed(arrays)
    sources = [dict(source_index=0, path="/pnfs/DO_NOT_OPEN.root", uuid="synthetic",
                    retained_events=int(n.sum()), events_with_tracks=int(a.sum()), selected_tracks=int(t.sum()),
                    multiplicity_0_1_2_3_ge4=mult.sum(axis=(0, 1, 2)).tolist(),
                    measured_pbtime_min=-250., measured_pbtime_max=-180.)]
    return dict(schema=S.INPUT_SCHEMA, axes=axes, grids=grids, sources=sources,
                counts=dict(retained_events=int(n.sum()), selected_events_matched=int(a.sum()),
                            selected_tracks_matched=int(t.sum()), retained_events_without_window_tracks=int(n.sum() - a.sum())))


@pytest.fixture
def saved(tmp_path, monkeypatch):
    (tmp_path / "input").mkdir()
    (tmp_path / "summary").mkdir()
    path, out = tmp_path / "input" / "report.json", tmp_path / "summary" / "new.json"
    data = json.dumps(fixture_report()).encode()
    path.write_bytes(data)
    pin = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(S, "REPORT_SHA256", pin)
    monkeypatch.setattr(S, "REPORT_BYTES", len(data))
    return path, out, pin


def test_frozen_contract():
    assert S.REPORT_SHA256 == "09c8831caca851de8ebb78cee91f467d9ff9f7c6b346ebdffe7ca5a2c70e3831"
    assert S.REPORT_BYTES == 138519
    assert S.INPUT_CAP == S.OUTPUT_CAP == 2 * 1024**2
    assert S.SOURCE_CAP == 64 * 1024
    assert S.INPUT_SCHEMA == "ce-joint-diagnostic-v1"


def test_conservation_rates_flows_and_sparse_nulls():
    report = fixture_report()
    result = S.summarize(report)
    assert result["axes"] == report["axes"]
    assert result["totals"] == dict(N=273, A=71, T=84)
    assert sum(result["all_selected_raw_time_histogram"]) == 84
    for row in result["marginals"].values():
        assert sum(row["N"]) == 273
        assert sum(row["A"]) == 71
        assert sum(row["T"]) == 84
        assert np.sum(row[S.HIST]) == 84
        assert np.asarray(row[S.MULT]).sum() == 273
        assert sum(v for v in row["all_event_bin_percent"] if v is not None) == pytest.approx(100)
        assert sum(v for v in row["selected_event_bin_percent"] if v is not None) == pytest.approx(100)
    ntc = result["marginals"]["nProtonTCs"]
    assert ntc["N"] == [200, 60, 0, 0, 0, 0, 9, 4]
    assert ntc["A_over_N"][0] == 0.3
    assert ntc["T_over_N"][0] == 0.35
    assert ntc["T_over_A"][0] == pytest.approx(70 / 60)
    assert ntc["A_over_N"][2] is None
    assert ntc["A_over_N"][6] == 0
    assert ntc["T_over_A"][6] is None
    assert ntc["T_over_A"][-1] == 4
    assert ntc["total_variation_vs_all_selected"][-1] is None
    assert ntc[S.HIST][0][0] == 30
    assert ntc[S.HIST][-1][-1] == 4
    assert result["sources"][0]["A_over_N"] == pytest.approx(71 / 273)
    assert result["F"] is result["weight_model"] is result["calibration"] is None


def test_known_total_variation_and_joint_support():
    result = S.summarize(fixture_report())
    row = result["conditional_on_measured_pbtime"]["nProtonTCs"]
    assert row["total_variation_vs_all_selected"][0][0] == pytest.approx(54 / 84)
    assert row["total_variation_vs_all_selected"][0][1] == pytest.approx(40 / 84)
    assert row["total_variation_vs_all_selected"][1][1] is None
    hist = np.zeros((3, 22), dtype=int)
    hist[0, 0], hist[1, 21] = 30, 30
    assert S.total_variation(hist, hist.sum(axis=0), np.array([True, False, True])) == [0.5, None, None]
    assert S.total_variation(hist, np.zeros(22, dtype=int), np.ones(3, dtype=bool)) == [None] * 3


def test_truth_and_phase_rate_ranges_and_row_percentages():
    result = S.summarize(fixture_report())
    truth = result["nProtonTCs_by_MC_truth_nprotons"]
    for spread in (truth["within_nProtonTCs_MC_truth_rate_range"], result["within_nProtonTCs_measured_pbtime_rate_range"]):
        assert spread["supported_cell_count"] == [2, 0, 0, 0, 0, 0, 0, 0]
        assert spread["min_A_over_N"][0] == 0.2
        assert spread["max_A_over_N"][0] == 0.4
        assert spread["max_minus_min_A_over_N"][0] == pytest.approx(0.2)
        assert spread["max_minus_min_A_over_N"][1] is None
        assert spread["supported_N"][0] == 200
        assert spread["supported_A"][0] == 60
        assert spread["supported_T"][0] == 70
    assert truth["row_all_event_bin_percent"][0][:2] == [50., 50.]
    assert truth["row_selected_event_bin_percent"][0][:2] == pytest.approx([100 / 3, 200 / 3])
    assert truth["row_selected_event_bin_percent"][6] == [None] * 7
    assert truth["row_all_event_bin_percent"][2] == [None] * 7


def test_one_supported_cell_range_is_zero():
    arrays = (np.array([[100, 99], [0, 0]]), np.array([[20, 40], [0, 0]]), np.array([[30, 40], [0, 0]]))
    row = S.row_range(arrays)
    assert row["supported_cell_count"] == [1, 0]
    assert row["max_minus_min_A_over_N"] == [0., None]


def test_permutation_equivariance_and_object_order():
    report = fixture_report()
    original = S.summarize(report)
    # Reorder saved object keys without any numerical effect.
    reordered = {key: value for key, value in reversed(list(report.items()))}
    reordered["grids"] = dict(reversed(list(report["grids"].items())))
    assert S.summarize(reordered) == original
    # Fixed-bin permutation of all NTC cells: output arrays permute, global counts do not.
    perm = np.array([7, 2, 0, 6, 1, 3, 5, 4])
    permuted = copy.deepcopy(report)
    for key in ("nProtonTCs__measured_pbtime", "nProtonTCs__MC_truth_nprotons__measured_pbtime"):
        for field in (*S.COUNT_FIELDS, S.MULT, S.HIST):
            permuted["grids"][key][field] = np.asarray(permuted["grids"][key][field])[perm].tolist()
    result = S.summarize(permuted)
    assert result["totals"] == original["totals"]
    assert result["marginals"][S.MC] == original["marginals"][S.MC]
    for key in ("N", "A", "T", "A_over_N", "total_variation_vs_all_selected"):
        assert result["marginals"]["nProtonTCs"][key] == np.asarray(original["marginals"]["nProtonTCs"][key], dtype=object)[perm].tolist()


@pytest.mark.parametrize("mutation", ["shape", "float", "negative", "A_gt_N", "T_lt_A", "hist", "mult", "overflow", "axis", "schema", "total", "sources", "missing_grid", "noninteger_total", "mult_T"])
def test_malformed_report_fails(mutation):
    report = fixture_report()
    grid = report["grids"]["nProtonTCs__measured_pbtime"]
    if mutation == "shape":
        grid[S.HIST][0][0].pop()
    elif mutation == "float":
        grid[S.COUNT_FIELDS[0]][0][0] = 100.0
    elif mutation == "negative":
        grid[S.COUNT_FIELDS[0]][0][0] = -1
    elif mutation == "A_gt_N":
        grid[S.COUNT_FIELDS[1]][0][0] = 101
    elif mutation == "T_lt_A":
        grid[S.COUNT_FIELDS[2]][0][0] = 19
    elif mutation == "hist":
        grid[S.HIST][0][0][0] += 1
    elif mutation == "mult":
        grid[S.MULT][0][0][0] += 1
    elif mutation == "overflow":
        grid[S.COUNT_FIELDS[0]][0][0] = 2**64
    elif mutation == "axis":
        report["axes"][S.TIME]["edges"][-1] += 1
    elif mutation == "schema":
        report["schema"] = "wrong"
    elif mutation == "total":
        report["counts"]["selected_tracks_matched"] += 1
    elif mutation == "sources":
        report["sources"][0]["selected_tracks"] += 1
    elif mutation == "missing_grid":
        report["grids"].pop("caloEnergy__measured_pbtime")
    elif mutation == "noninteger_total":
        report["counts"]["retained_events"] = True
    elif mutation == "mult_T":
        grid[S.MULT][0][0][1:3] = [20, 0]
    with pytest.raises(ValueError):
        S.summarize(report)


def test_inconsistent_joint_cross_marginal_fails():
    report = fixture_report()
    grid = report["grids"]["nProtonTCs__MC_truth_nprotons__measured_pbtime"]
    for field in (*S.COUNT_FIELDS, S.MULT, S.HIST):
        grid[field] = np.flip(np.asarray(grid[field]), axis=0).tolist()
    with pytest.raises(ValueError, match="cross-grid"):
        S.summarize(report)


def test_zero_all_selected_no_zero_fill():
    report = fixture_report()
    for grid in report["grids"].values():
        n = np.asarray(grid[S.COUNT_FIELDS[0]])
        grid[S.COUNT_FIELDS[1]] = np.zeros_like(n).tolist()
        grid[S.COUNT_FIELDS[2]] = np.zeros_like(n).tolist()
        mult = np.zeros((*n.shape, 5), dtype=int)
        mult[..., 0] = n
        grid[S.MULT] = mult.tolist()
        grid[S.HIST] = np.zeros((*n.shape, 22), dtype=int).tolist()
    report["counts"].update(selected_events_matched=0, selected_tracks_matched=0, retained_events_without_window_tracks=273)
    report["sources"][0].update(events_with_tracks=0, selected_tracks=0, multiplicity_0_1_2_3_ge4=[273, 0, 0, 0, 0])
    result = S.summarize(report)
    for marginal in result["marginals"].values():
        assert all(value is None for value in marginal["T_over_A"])
        assert all(value is None for value in marginal["total_variation_vs_all_selected"])
        assert all(value is None for value in marginal["selected_event_bin_percent"])


def test_run_source_snapshot_and_reproducibility(saved, monkeypatch):
    path, out, pin = saved
    opened = []
    real = S.bounded_bytes

    def tracked(path, cap):
        opened.append(Path(path))
        return real(path, cap)

    monkeypatch.setattr(S, "bounded_bytes", tracked)
    result = S.run(path, out, expected_report_sha256=pin)
    assert set(opened) == {path, Path(S.__file__)}
    assert len(opened) == 4
    assert result == json.loads(out.read_bytes())
    assert S.sha(result["source"]["snapshot"].encode()) == result["source"]["sha256"]
    assert result["report"]["sha256"] == pin
    S.run(path, out.with_name("second.json"), expected_report_sha256=pin)
    assert out.read_bytes() == out.with_name("second.json").read_bytes()
    with pytest.raises(ValueError, match="new output"):
        S.run(path, out, expected_report_sha256=pin)


@pytest.mark.parametrize("pin", ["0" * 64, "BAD", None, "A" * 64])
def test_external_pin_fails_before_reads(saved, monkeypatch, pin):
    path, out, _ = saved
    monkeypatch.setattr(S, "bounded_bytes", lambda *args: pytest.fail("no reads with bad pin"))
    with pytest.raises(ValueError, match="external pin"):
        S.run(path, out, expected_report_sha256=pin)
    assert not out.exists()


def test_report_digest_and_size(saved, monkeypatch):
    path, out, pin = saved
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        S.run(path, out, expected_report_sha256=pin)
    new_pin = S.sha(path.read_bytes())
    monkeypatch.setattr(S, "REPORT_SHA256", new_pin)
    with pytest.raises(ValueError, match="byte size"):
        S.run(path, out, expected_report_sha256=new_pin)
    assert not out.exists()


@pytest.mark.parametrize("which", ["input", "source"])
def test_recheck_rejects_changed_bytes(saved, monkeypatch, which):
    path, out, pin = saved
    original = S.bounded_bytes
    reads = {}

    def changed(p, cap):
        reads[p] = reads.get(p, 0) + 1
        data = original(p, cap)
        target = path if which == "input" else Path(S.__file__)
        return data + b" " if p == target and reads[p] == 2 else data

    monkeypatch.setattr(S, "bounded_bytes", changed)
    with pytest.raises(ValueError, match="changed before output"):
        S.run(path, out, expected_report_sha256=pin)
    assert not out.exists()


@pytest.mark.parametrize("kind", ["input", "code", "missing_parent", "existing", "symlink", "symlink_parent"])
def test_output_guards(saved, kind, tmp_path):
    path, out, pin = saved
    if kind == "input":
        out = path.parent / "new.json"
    elif kind == "code":
        out = Path(S.__file__).parent / "never-created-summary.json"
    elif kind == "missing_parent":
        out = tmp_path / "absent" / "new.json"
    elif kind == "existing":
        out.write_bytes(b"keep")
    elif kind == "symlink":
        out.symlink_to(path)
    elif kind == "symlink_parent":
        link = tmp_path / "linked"
        link.symlink_to(out.parent, target_is_directory=True)
        out = link / "new.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        S.run(path, out, expected_report_sha256=pin)
    if kind == "existing":
        assert out.read_bytes() == b"keep"


@pytest.mark.parametrize("value", ["root://host/report.json", "/pnfs/report.json", "/PNFS/report.json", "/x.ROOT/report.json", "//host/file", "a/../b"])
def test_local_path_guards(value):
    with pytest.raises(ValueError):
        S.local_path(value)


def test_input_symlink_and_nonregular(saved, tmp_path):
    path, out, pin = saved
    link = path.with_name("alias.json")
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        S.run(link, out, expected_report_sha256=pin)
    with pytest.raises(ValueError, match="regular file"):
        S.bounded_bytes(tmp_path, S.INPUT_CAP)


@pytest.mark.parametrize("cap", ["INPUT_CAP", "SOURCE_CAP", "OUTPUT_CAP"])
def test_byte_caps(saved, monkeypatch, cap):
    path, out, pin = saved
    monkeypatch.setattr(S, cap, 1)
    with pytest.raises(ValueError, match="byte cap"):
        S.run(path, out, expected_report_sha256=pin)
    assert not out.exists()


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'[]'])
def test_json_rejects_ambiguity(data):
    with pytest.raises(ValueError):
        S.json_object(data)


def test_cli(saved):
    path, out, pin = saved
    S.main([str(path), "--expected-report-sha256", pin, "--out", str(out)])
    assert json.loads(out.read_bytes())["schema"] == S.SCHEMA
    with pytest.raises(SystemExit):
        S.main([str(path), "--out", str(out)])
