"""Synthetic fixtures only; no old consumer execution or production data reads."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from crvfastsim.veto import ce_composition_diagnostic as C
from crvfastsim.veto import ce_joint_diagnostic as J


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden(*args, **kwargs):
    pytest.fail("old consumer or forbidden read")


class Saved:
    def __init__(self, root, mp):
        self.root, self.mp = root, mp
        for name in ("original", "rows", "selected", "ledger", "joint", "summary", "output"):
            (root / name).mkdir()
        self.sp, self.rp = root / "original/report.json", root / "rows/report.json"
        self.jp, self.up = root / "joint/report.json", root / "summary/report.json"
        self.npz, self.tracks, self.ledger = root / "rows/rows.npz", root / "selected/tracks.parquet", root / "ledger/ledger.parquet"
        self.out = root / "output/result.json"
        self.rows = dict(run=np.array([2**53 + 1] * 12, dtype=np.int64), subrun=np.array([3] * 11 + [4], dtype=np.int64),
            event=np.array([1, 2**32 + 1, 3, 4, 5, 6, 2**53 + 1, 8, 9, 10, 11, 1], dtype=np.int64),
            source_index=np.array([6] * 6 + [8] * 6, dtype=np.int16), measured_nprotons=np.zeros(12, dtype=np.int32),
            measured_pbtime=np.array([-241, -240, -230, -220, -210, -200, -190, -189, -235, -225, -215, -205], dtype=np.float32),
            measured_pbterr=np.full(12, .5, dtype=np.float32), nProtonTCs=np.array([0, 0, 1, 2, 3, 1, 0, 10, 1, 1, 0, 0], dtype=np.int32),
            nTrackerHits=np.arange(12, dtype=np.int32), nCaloHitsD0=np.arange(12, dtype=np.int32),
            nCaloHitsD1=np.arange(12, dtype=np.int32), caloEnergy=np.arange(12, dtype=np.int32), caphriIsValid=np.ones(12, dtype=bool))
        e = C.MC_EDGES
        self.ledger_arrays = {k: self.rows[k].copy() for k in J.IDS}
        self.ledger_arrays.update(nprotons=np.array([e[0]-1, *e, e[-1]+1, e[0], e[4], e[0], e[4]], dtype=np.int64),
            pbtime=np.arange(12, dtype=float), measured_nprotons=np.zeros(12), measured_pbtime=self.rows[C.PB].astype(float))
        indices = [0, 1, 2, 5, 6, 10]
        self.table = pa.table({**{k: pa.array([*self.rows[k][indices], 999], type=pa.int64()) for k in J.IDS},
            "p_mid": pa.array([[84., 85., 86., 200., 201.], [100., 100.], [100.] * 4, [], [86., 199.], [85.], [99.]]),
            "t_mid": pa.array([[-9., 13., -1., 15., 2001.], [0., 0.], [100., 1999., 2000., 2500.], [], [600., 1600.], [77.], [90.]]),
            "coincs": pa.array([[b"never decode"]] * 7)})
        self.original = dict(selection=dict(path=str(self.tracks)), exposure_ledger=dict(path=str(self.ledger)),
            momentum_window=dict(pmin=85, pmax=200, strict=True, selected_events=5, selected_tracks=10),
            diagnostic=dict(ntuple_events=12, selected_events_matched=4, selected_tracks_matched=9,
                selected_events_not_in_supplied_ntuples=1, pbi_quantile_strata=[]),
            ntuples=[dict(path=f"/mnt/pnfs/frozen_{i}.root", uuid=f"uuid-{i}", trees={"EventNtuple/ntuple": dict(entries=6)}) for i in (6, 8)])
        self.extracted = dict(schema="ce-proxy-rows-v1", availability_report=dict(sha256=J.AVAILABILITY_SHA256),
            files=[dict(index=i, path=r["path"], uuid=r["uuid"], entries=6) for i, r in zip((6, 8), self.original["ntuples"])])
        self.mult = np.array([1, 2, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0])
        self.te = np.array([0, 1, 1, 2, 2, 2, 2, 6, 6])
        self.time = J.bins(np.array([-1, 0, 0, 100, 1999, 2000, 2500, 600, 1600]), J.TIME_EDGES)
        mc = J.bins(self.ledger_arrays["nprotons"], e, True)
        for i in range(5):
            mask = mc == i + 1
            self.original["diagnostic"]["pbi_quantile_strata"].append(dict(pbi_lo=e[i], pbi_hi=e[i+1],
                upper_inclusive=i == 4, ntuple_events=int(mask.sum()), selected_events=int(np.count_nonzero(self.mult[mask])),
                selected_tracks=int(self.mult[mask].sum())))
        self.joint = self.completed_joint(mc)
        self.summary = self.completed_summary()
        mp.setattr(J, "EXPECTED_ENTRIES", 12)
        mp.setattr(J, "run", forbidden)
        mp.setattr(J, "diagnose", forbidden)
        self.repin()

    def completed_joint(self, mc):
        pb = J.bins(self.rows[C.PB], J.PB_EDGES)
        grids = {}
        axes = {C.PB: J.axis(J.PB_EDGES), C.MC: J.axis(C.MC_EDGES, True), "raw_stored_t_mid": J.axis(J.TIME_EDGES)}
        for name in J.PROXIES:
            edges = J.TC_EDGES if name == C.TC else J.HIT_EDGES
            axes[name] = J.axis(edges)
            grids[name + "__" + C.PB] = J.grid([J.bins(self.rows[name], edges), pb], (len(edges)+1, 7), self.mult, self.te, self.time)
        grids[C.MC + "__" + C.PB] = J.grid([mc, pb], (7, 7), self.mult, self.te, self.time)
        grids["__".join((C.TC, C.MC, C.PB))] = J.grid([J.bins(self.rows[C.TC], J.TC_EDGES), mc, pb], (8, 7, 7), self.mult, self.te, self.time)
        counts = dict(full_selected_rows=7, full_selected_tracks=15, momentum_window_events=5, momentum_window_tracks=10,
            exactly_85=2, exactly_200=1, below_85=1, above_200=1, zero_window_rows=2, retained_events=12,
            selected_rows_matched_including_empty=6, selected_rows_unmatched_including_empty=1,
            full_selected_tracks_matched_before_window=14, full_selected_tracks_unmatched_before_window=1,
            selected_events_matched=4, selected_tracks_matched=9, selected_events_unmatched=1, selected_tracks_unmatched=1,
            retained_events_without_selected_row=6, retained_events_without_window_tracks=8, proxy_not_in_ledger=0, ledger_not_in_proxy=0)
        source = []
        for i, record in zip((6, 8), self.original["ntuples"]):
            mask = self.rows["source_index"] == i
            source.append(dict(source_index=i, path=record["path"], uuid=record["uuid"], retained_events=6,
                events_with_tracks=int(np.count_nonzero(self.mult[mask])), selected_tracks=int(self.mult[mask].sum()),
                multiplicity_0_1_2_3_ge4=np.bincount(np.minimum(self.mult[mask], 4), minlength=5).tolist(),
                measured_pbtime_min=float(self.rows[C.PB][mask].min()), measured_pbtime_max=float(self.rows[C.PB][mask].max())))
        identity = np.array([[int(self.rows[k][event]) for k in J.IDS] + [pos]
            for event, pos in zip(self.te, [2, 0, 1, 0, 1, 2, 3, 0, 1])], dtype="<i8")
        identity = identity[np.lexsort(tuple(identity[:, i] for i in (3, 2, 1, 0)))]
        return dict(schema="ce-joint-diagnostic-v1", axes=axes, grids=grids, counts=counts, sources=source,
            matched_track_identity=dict(sha256=J.sha(identity.tobytes()), count=9,
                encoding="C bytes of lexicographically sorted run,subrun,event,original_list_position; four little-endian signed int64"),
            inputs=[dict(path="/forbidden/no-open.root")])

    def completed_summary(self):
        grid = self.joint["grids"]["__".join((C.TC, C.MC, C.PB))]
        values = [np.asarray(grid[k]).sum(axis=2) for k in C.RAW]
        return dict(schema="ce-joint-summary-v1", input_schema="ce-joint-diagnostic-v1", axes=self.joint["axes"],
            counts=self.joint["counts"], totals=dict(N=12, A=4, T=9),
            all_selected_raw_time_histogram=values[-1].sum(axis=(0, 1)).tolist(),
            nProtonTCs_by_MC_truth_nprotons={**{k: v.tolist() for k, v in zip(("N", "A", "T", C.MULT, C.HIST), values)},
                "A_over_N": J.divide(values[1], values[0])}, inputs=[dict(path="root://forbidden")])

    def repin(self):
        pq.write_table(self.table, self.tracks)
        pq.write_table(pa.table(self.ledger_arrays), self.ledger)
        np.savez(self.npz, **self.rows)
        for attribute, path in (("SELECTED_SHA256", self.tracks), ("LEDGER_SHA256", self.ledger), ("ROWS_SHA256", self.npz)):
            self.mp.setattr(C, attribute, digest(path))
        self.original["selection"]["sha256"] = digest(self.tracks)
        self.original["exposure_ledger"]["sha256"] = digest(self.ledger)
        self.sp.write_text(json.dumps(self.original))
        self.mp.setattr(C, "SELECTION_SHA256", digest(self.sp))
        self.extracted["selection_report"] = dict(sha256=digest(self.sp))
        self.extracted["rows"] = dict(path=str(self.npz), sha256=digest(self.npz), entries=12,
            columns={k: dict(dtype=v.dtype.str, shape=list(v.shape), sha256=J.sha(v.tobytes())) for k, v in self.rows.items()})
        self.rp.write_text(json.dumps(self.extracted))
        self.mp.setattr(C, "ROWS_REPORT_SHA256", digest(self.rp))
        self.jp.write_text(json.dumps(self.joint))
        self.mp.setattr(C, "JOINT_SHA256", digest(self.jp))
        self.summary["report"] = dict(sha256=digest(self.jp), path="/never/open/this.root")
        self.up.write_text(json.dumps(self.summary))
        self.mp.setattr(C, "SUMMARY_SHA256", digest(self.up))

    def run(self, **kwargs):
        if not kwargs:
            kwargs = dict(out=self.out)
        return C.run(self.sp, self.rp, self.jp, self.up, **kwargs)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    return Saved(tmp_path, monkeypatch)


def test_frozen_pins_and_source():
    assert C.CONTRACT_SHA256 == "ca69360f65abea85e12381601c299850636903fb57358804dd610711667655a1"
    assert C.HELPER_SHA256 == digest(Path(J.__file__)) == "1d9c4dafc450df68c72d365c82fd46116090a807838a0b07fdb055a410eab4b4"
    assert C.SELECTION_SHA256 == J.SELECTION_SHA256
    assert C.SELECTED_SHA256 == J.SELECTED_SHA256 and C.LEDGER_SHA256 == J.LEDGER_SHA256
    assert C.ROWS_REPORT_SHA256 == "af675694a4a997295d42d84a98bc8fb0aedac160aa83c2cc69002e3cb4e2b2e8"
    assert C.ROWS_SHA256 == "20f012be65a9487226923f0388a72f06eca8ee02c95bcdcadc822b8b68f45f3a"
    assert C.JOINT_SHA256 == "09c8831caca851de8ebb78cee91f467d9ff9f7c6b346ebdffe7ca5a2c70e3831"
    assert C.SUMMARY_SHA256 == "63fc77886c3febb00ba6120ee9dfd6140bb3fedf12260312544f3bb574b68e0a"
    assert C.MC_EDGES == [2412587, 11844973, 15682329, 19931318, 26354159, 93225507]
    assert C.OUTPUT_CAP == 2 * 1024**2 and C.SOURCE_CAP == 64 * 1024
    tree = ast.parse(Path(C.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            assert (node.func.value.id, node.func.attr) not in {("J", "run"), ("J", "diagnose")}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert not any(s in ast.unparse(node) for s in ("uproot", "ce_joint_summary", "ce_proxy_rows", "ROOT"))


def test_saved_integration(saved):
    result = saved.run()
    assert json.loads(saved.out.read_bytes()) == result
    assert saved.out.stat().st_size < C.OUTPUT_CAP
    assert result["counts"] == saved.joint["counts"]
    assert result["matched_track_identity"] == saved.joint["matched_track_identity"]
    assert result["axes"]["source_index"] == [6, 8]
    full = C.raw_grid(result["full_grid"], (2, 8, 7, 7))
    assert [int(v.sum()) for v in full[:3]] == [12, 4, 9]
    assert full[0][:, :, 0].sum() == full[0][:, :, 6].sum() == 1
    assert full[0][:, :, 2:5].sum() == 3
    assert full[0][..., 0].sum() == 1 and full[0][..., 6].sum() == 2
    assert full[4][..., 0].sum() == 1 and full[4][..., 21].sum() == 2
    assert full[4][..., 1].sum() == 2
    assert full[3][..., 4].sum() == 1
    for c in result["contrasts"].values():
        assert len(c["cells"]) == 14 and c["common_block_count"] == 0
        assert c["D_common_pp"] is c["D_standardized_pp"] is None
        assert all(cell["q"] is None for cell in c["cells"])
    assert all(result[k] is None for k in ("F", "calibration", "weight_model", "finite_library_uncertainty", "total_uncertainty"))
    assert result["source"]["sha256"] == J.sha(result["source"]["snapshot"].encode())
    assert result["helper_source"]["sha256"] == C.HELPER_SHA256


def test_read_guard(saved, monkeypatch):
    actual = J.pq.ParquetFile
    seen = []
    class Guard:
        def __init__(self, *args, **kwargs):
            self.inner = actual(*args, **kwargs)
        def __getattr__(self, key):
            return getattr(self.inner, key)
        def iter_batches(self, **kwargs):
            assert kwargs["columns"] in [list(J.SELECTED_KEYS), list(J.LEDGER_KEYS)]
            assert "coincs" not in kwargs["columns"] and kwargs["use_threads"] is False
            seen.append(kwargs["columns"])
            kwargs["batch_size"] = 1
            return self.inner.iter_batches(**kwargs)
    monkeypatch.setattr(J.pq, "ParquetFile", Guard)
    read = J.Evidence.read
    allowed = {saved.sp, saved.rp, saved.jp, saved.up, saved.npz, saved.tracks, saved.ledger, Path(C.__file__), Path(J.__file__)}
    def guard(self, path, *args, **kwargs):
        assert path in allowed
        return read(self, path, *args, **kwargs)
    monkeypatch.setattr(J.Evidence, "read", guard)
    assert saved.run()["counts"]["selected_tracks_matched"] == 9
    assert len(seen) == 2


def test_permutation_and_equal_track_entries(saved):
    first = saved.run()
    saved.out.unlink()
    order = np.array([11, 9, 5, 3, 1, 7, 0, 2, 4, 6, 8, 10])
    saved.rows = {k: v[order] for k, v in saved.rows.items()}
    saved.ledger_arrays = {k: v[::-1] for k, v in saved.ledger_arrays.items()}
    saved.table = saved.table.take(pa.array([6, 4, 2, 1, 0, 3, 5]))
    saved.extracted["files"].reverse()
    saved.repin()
    second = saved.run()
    for field in ("counts", "full_grid", "contrasts", "matched_track_identity", "sources"):
        assert first[field] == second[field]


@pytest.mark.parametrize("target", ["sp", "rp", "jp", "up", "npz", "tracks", "ledger"])
def test_input_pins(saved, target):
    p = getattr(saved, target)
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        saved.run()
    assert not saved.out.exists()


@pytest.mark.parametrize("target", ["sp", "rp", "jp", "up", "npz", "tracks", "ledger", "source", "helper"])
def test_before_after_recheck(saved, monkeypatch, target):
    if target in ("source", "helper"):
        module = C if target == "source" else J
        p = saved.root / "copy/pkg/subpkg" / (target + ".py")
        p.parent.mkdir(parents=True)
        p.write_bytes(Path(module.__file__).read_bytes())
        monkeypatch.setattr(module, "__file__", str(p))
    else:
        p = getattr(saved, target)
    recheck = J.Evidence.recheck
    def mutate(evidence):
        p.write_bytes(p.read_bytes() + b" ")
        recheck(evidence)
    monkeypatch.setattr(J.Evidence, "recheck", mutate)
    with pytest.raises(ValueError, match="changed"):
        saved.run()
    assert not saved.out.exists()


def test_helper_pin_before_decoding(saved, monkeypatch):
    path = saved.root / "helper" / "helper.py"
    path.parent.mkdir()
    path.write_text("changed helper")
    monkeypatch.setattr(J, "__file__", str(path))
    monkeypatch.setattr(J, "load_rows", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        saved.run()


@pytest.mark.parametrize("cap,module", [("OUTPUT_CAP", C), ("REPORT_CAP", C), ("SOURCE_CAP", C),
    ("ROWS_CAP", J), ("SELECTED_CAP", J), ("LEDGER_CAP", J), ("FOOTER_CAP", J), ("DECODED_CAP", J),
    ("MAX_ROWS", J), ("MAX_TRACKS", J), ("MAX_TRACKS_PER_ROW", J), ("MAX_LEDGER_ROWS", J)])
def test_caps_fail_closed(saved, monkeypatch, cap, module):
    monkeypatch.setattr(module, cap, 1)
    with pytest.raises(ValueError):
        saved.run()
    assert not saved.out.exists()


@pytest.mark.parametrize("kind", ["duplicate_rows", "duplicate_ledger", "duplicate_selected", "key", "measurement",
    "negative_proxy", "nonfinite", "uint_overflow", "object", "shape", "source", "source_uuid", "window", "MC_edges", "original_counts"])
def test_invalid_inputs(saved, kind):
    if kind in ("duplicate_rows", "duplicate_ledger"):
        arrays = saved.rows if kind == "duplicate_rows" else saved.ledger_arrays
        arrays["event"][1] = arrays["event"][0]
    elif kind == "duplicate_selected":
        event = saved.table["event"].to_pylist()
        event[3] = event[0]
        saved.table = saved.table.set_column(2, "event", pa.array(event))
    elif kind == "key":
        saved.ledger_arrays["event"][0] = 333
    elif kind == "measurement":
        saved.ledger_arrays["measured_pbtime"][0] += 1
    elif kind == "negative_proxy":
        saved.rows[C.TC][0] = -1
    elif kind == "nonfinite":
        saved.rows[C.PB][0] = np.nan
    elif kind == "uint_overflow":
        saved.rows["event"] = saved.rows["event"].astype(np.uint64)
        saved.rows["event"][0] = 2**63
    elif kind == "object":
        saved.rows["run"] = np.array([object()] * 12)
    elif kind == "shape":
        saved.rows["run"] = saved.rows["run"].reshape(6, 2)
    elif kind == "source":
        saved.rows["source_index"][0] = 7
    elif kind == "source_uuid":
        saved.extracted["files"][0]["uuid"] = "wrong"
    elif kind == "window":
        saved.original["momentum_window"]["strict"] = False
    elif kind == "MC_edges":
        saved.original["diagnostic"]["pbi_quantile_strata"][0]["pbi_lo"] -= 1
    else:
        saved.original["diagnostic"]["selected_tracks_matched"] += 1
    saved.repin()
    with pytest.raises(ValueError):
        saved.run()
    assert not saved.out.exists()


@pytest.mark.parametrize("kind", ["null_list", "null_element", "nonfinite_rejected_time", "nonfinite_momentum", "misaligned", "too_many"])
def test_invalid_selected_arrays(saved, kind):
    field = "p_mid" if kind == "nonfinite_momentum" else "t_mid"
    values = saved.table[field].to_pylist()
    if kind == "null_list":
        values[0] = None
    elif kind == "null_element":
        values[0][0] = None
    elif kind.startswith("nonfinite"):
        values[0][0] = float("inf")
    elif kind == "misaligned":
        values[0].append(1.)
    else:
        values[0] = [1.] * 65
    saved.table = saved.table.set_column(saved.table.schema.get_field_index(field), field, pa.array(values, type=pa.list_(pa.float64())))
    saved.repin()
    with pytest.raises(ValueError):
        saved.run()


@pytest.mark.parametrize("kind", ["counts", "identity", "source", "axis", "three", "proton", "mc", "other", "hist", "mult", "summary_counts", "summary_rates", "summary_hist", "summary_binding"])
def test_inherited_corruption(saved, kind):
    key = "__".join((C.TC, C.MC, C.PB))
    if kind == "counts":
        saved.joint["counts"]["full_selected_tracks"] += 1
    elif kind == "identity":
        saved.joint["matched_track_identity"]["sha256"] = "a" * 64
    elif kind == "source":
        saved.joint["sources"][0]["uuid"] = "wrong"
    elif kind == "axis":
        saved.joint["axes"][C.MC]["last_finite_upper_inclusive"] = False
    elif kind in ("three", "proton", "mc", "other", "hist", "mult"):
        name = key if kind in ("three", "hist", "mult") else (C.TC if kind == "proton" else C.MC if kind == "mc" else "nTrackerHits") + "__" + C.PB
        field = C.HIST if kind == "hist" else C.MULT if kind == "mult" else C.NAT[0]
        v = np.asarray(saved.joint["grids"][name][field])
        v.flat[0] += 1
        saved.joint["grids"][name][field] = v.tolist()
    elif kind == "summary_counts":
        saved.summary["nProtonTCs_by_MC_truth_nprotons"]["N"][1][1] += 1
    elif kind == "summary_rates":
        saved.summary["nProtonTCs_by_MC_truth_nprotons"]["A_over_N"][1][1] = .123
    elif kind == "summary_hist":
        saved.summary["all_selected_raw_time_histogram"][0] += 1
    saved.repin()
    if kind == "summary_binding":
        saved.summary["report"]["sha256"] = "a" * 64
        saved.up.write_text(json.dumps(saved.summary))
        saved.mp.setattr(C, "SUMMARY_SHA256", digest(saved.up))
    with pytest.raises(ValueError):
        saved.run()
    assert not saved.out.exists()


def synthetic_grid(cells, nsources=2):
    shape = (nsources, 8, 7, 7)
    n, a, t = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    mult, hist = np.zeros((*shape, 5), dtype=np.int64), np.zeros((*shape, 22), dtype=np.int64)
    for s, c, b, endpoint, nn, aa in cells:
        idx = (s, c + 1, 1 if endpoint == 0 else 5, b)
        n[idx], a[idx], t[idx] = nn, aa, aa
        mult[idx + (0,)], mult[idx + (1,)] = nn-aa, aa
        hist[idx + (7,)] = aa
    return n, a, t, mult, hist


def test_N_only_support_A_zero_and_count_specific_q():
    arrays = synthetic_grid([(0, 0, 0, 0, 100, 0), (0, 0, 0, 1, 200, 0),
        (1, 0, 6, 0, 300, 150), (1, 0, 6, 1, 100, 10),
        (0, 1, 0, 0, 99, 50), (0, 1, 0, 1, 1000, 500),
        (1, 1, 6, 0, 100, 20), (1, 1, 6, 1, 100, 30)])
    result = C.endpoint_contrasts(arrays, [6, 8])
    zero, one = result["0"], result["1"]
    assert zero["common_block_count"] == 2 and one["common_block_count"] == 1
    assert zero["cells"][0]["q"] == pytest.approx(3/7)
    assert zero["cells"][-1]["q"] == pytest.approx(4/7)
    assert zero["cells"][0]["low"]["support_A_ge20_and_T_ge30"] is False
    assert zero["cells"][0]["common_support"] is True
    assert zero["D_standardized_pp"] == pytest.approx(100 * 4/7 * .4)
    assert one["cells"][0]["q"] == 0 and one["cells"][-1]["q"] == 1
    assert one["D_standardized_pp"] == pytest.approx(-10)
    assert one["endpoints"]["low"]["excluded"]["N"] == 99
    assert one["endpoints"]["low"]["excluded_fractions"]["N"] == pytest.approx(99/199)


def test_support_restriction_and_composition_simpson():
    arrays = synthetic_grid([(0, 0, 1, 0, 900, 720), (0, 0, 1, 1, 100, 90),
        (1, 0, 1, 0, 100, 10), (1, 0, 1, 1, 900, 180),
        (0, 0, 2, 0, 50, 0), (0, 0, 2, 1, 50, 50)])
    result = C.endpoint_contrasts(arrays, [6, 8])["0"]
    assert result["D_all_pp"] == pytest.approx(100*(730/1050-320/1050))
    assert result["D_common_pp"] == pytest.approx(46)
    assert result["D_standardized_pp"] == pytest.approx(-10)
    assert result["D_all_pp"] != result["D_common_pp"]
    assert result["endpoints"]["low"]["excluded"]["N"] == 50
    assert result["endpoints"]["high"]["excluded_fractions"]["A"] == pytest.approx(50/320)


def test_cell_and_source_cancellation_visible():
    arrays = synthetic_grid([(0, 0, 1, 0, 100, 90), (0, 0, 1, 1, 100, 10),
        (1, 0, 2, 0, 100, 10), (1, 0, 2, 1, 100, 90)])
    result = C.endpoint_contrasts(arrays, [6, 8])["0"]
    assert result["D_standardized_pp"] == 0
    assert [s["standardized_contribution_pp"] for s in result["per_source"]] == [40, -40]
    assert [s["D_standardized_within_source_pp"] for s in result["per_source"]] == [80, -80]
    supported = [v for v in result["cells"] if v["common_support"]]
    assert [v["D_cell_pp"] for v in supported] == [80, -80]


def test_missing_endpoint_nulls():
    result = C.endpoint_contrasts(synthetic_grid([(0, 0, 1, 0, 200, 0)]), [6, 8])
    for c in result.values():
        assert c["D_all_pp"] is c["D_common_pp"] is c["D_standardized_pp"] is None
        assert all(v is None for v in c["endpoints"]["high"]["excluded_fractions"].values())
        assert c["endpoints"]["high"]["total"]["A_over_N"] is None
        assert len(c["cells"]) == 14
    assert result["0"]["cells"][1]["unsupported_reasons"] == ["high_N_zero"]
    assert result["0"]["endpoints"]["low"]["total"]["A_over_N"] == 0


def test_original_edges():
    e = np.array(C.MC_EDGES)
    assert J.bins(e, C.MC_EDGES, True).tolist() == [1, 2, 3, 4, 5, 5]
    assert J.bins(np.array([e[0]-1, e[-1]+1]), C.MC_EDGES, True).tolist() == [0, 6]
    assert J.bins(np.array([-241, -240, -230, -190, -189]), J.PB_EDGES).tolist() == [0, 1, 2, 6, 6]
    assert J.bins(np.array([-1, 0, 100, 2000, 2001]), J.TIME_EDGES).tolist() == [0, 1, 2, 21, 21]
    assert J.bins(np.arange(12), J.TC_EDGES).tolist() == [1, 2, 3, 4, 5, 5, 6, 6, 6, 6, 7, 7]


@pytest.mark.parametrize("directory", ["original", "rows", "joint", "summary", "selected", "ledger"])
def test_output_distinct(saved, directory):
    with pytest.raises(ValueError, match="output under"):
        saved.run(out=saved.root / directory / "new.json")


def test_output_exclusive_and_new_directory(saved, monkeypatch):
    target = saved.root / "new_directory"
    result = saved.run(outdir=target)
    assert json.loads((target / "report.json").read_bytes()) == result
    with pytest.raises(ValueError, match="new output"):
        saved.run(outdir=target)
    saved.out.write_bytes(b"keep")
    with pytest.raises(ValueError, match="new output"):
        saved.run()
    assert saved.out.read_bytes() == b"keep"
    saved.out.unlink()
    recheck = J.Evidence.recheck
    def race(evidence):
        recheck(evidence)
        saved.out.write_bytes(b"race")
    monkeypatch.setattr(J.Evidence, "recheck", race)
    with pytest.raises(ValueError, match="new output"):
        saved.run()
    assert saved.out.read_bytes() == b"race"


def test_outputcap_no_directory_creation(saved, monkeypatch):
    monkeypatch.setattr(C, "OUTPUT_CAP", 1)
    target = saved.root / "no_output"
    with pytest.raises(ValueError, match="no outcomes trimmed"):
        saved.run(outdir=target)
    assert not target.exists()


def test_cli(saved):
    C.main([str(p) for p in (saved.sp, saved.rp, saved.jp, saved.up)] + ["--out", str(saved.out)])
    assert saved.out.exists()
    for args in ([], ["a", "b", "c", "d"], ["a", "b", "c", "d", "--out", "e", "--outdir", "f"]):
        with pytest.raises(SystemExit):
            C.main(args)


def test_paths_symlinks_and_missing_parent(saved):
    for path in ("root://host/f", "/mnt/pnfs/x", "/PNFS/x", "/net/x", "/test.root", "//server/x"):
        with pytest.raises(ValueError):
            C.local_path(path)
    original = saved.npz.with_name("saved.npz")
    saved.npz.rename(original)
    saved.npz.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        saved.run()
    with pytest.raises((ValueError, FileNotFoundError)):
        saved.run(out=saved.root / "missing/never.json")


def test_full_scale_output_keeps_every_cell():
    arrays = synthetic_grid([], nsources=11)
    for v in arrays:
        v[...] = 100000
    result = C.endpoint_contrasts(arrays, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11])
    payload = dict(full_grid=C.pack(arrays), contrasts=result,
        source=Path(C.__file__).read_text(), helper_source=Path(J.__file__).read_text())
    assert len(json.dumps(payload, separators=(",", ":")).encode()) < C.OUTPUT_CAP
    assert len(result["0"]["cells"]) == len(result["1"]["cells"]) == 77


@pytest.mark.parametrize("gridname", [C.TC + "__" + C.PB, C.MC + "__" + C.PB,
    "__".join((C.TC, C.MC, C.PB))])
def test_conservation_preserving_marginal_corruption(saved, gridname):
    grid = saved.joint["grids"][gridname]
    shape = grid["shape"]
    arrays = C.raw_grid(grid, shape)
    for name, array in zip(C.RAW, arrays):
        grid[name] = np.roll(array, 1, axis=0).tolist()
    C.raw_grid(grid, shape)
    saved.repin()
    with pytest.raises(ValueError, match="marginal differs"):
        saved.run()
    assert not saved.out.exists()


@pytest.mark.parametrize("kind", ["schema", "selection", "availability", "rows_pin", "selected_pin", "ledger_pin", "rows_path"])
def test_report_binding_checks(saved, kind):
    if kind == "schema":
        saved.extracted["schema"] = "wrong"
    elif kind == "selection":
        saved.extracted["selection_report"]["sha256"] = "a" * 64
    elif kind == "availability":
        saved.extracted["availability_report"]["sha256"] = "a" * 64
    elif kind == "rows_pin":
        saved.extracted["rows"]["sha256"] = "a" * 64
    elif kind == "selected_pin":
        saved.original["selection"]["sha256"] = "a" * 64
    elif kind == "ledger_pin":
        saved.original["exposure_ledger"]["sha256"] = "a" * 64
    else:
        saved.extracted["rows"]["path"] = str(saved.root / "wrong.npz")
    saved.sp.write_text(json.dumps(saved.original))
    saved.mp.setattr(C, "SELECTION_SHA256", digest(saved.sp))
    if kind != "selection":
        saved.extracted["selection_report"]["sha256"] = digest(saved.sp)
    saved.rp.write_text(json.dumps(saved.extracted))
    saved.mp.setattr(C, "ROWS_REPORT_SHA256", digest(saved.rp))
    with pytest.raises(ValueError):
        saved.run()
    assert not saved.out.exists()


def test_no_selected_tracks_all_zero_fractions_null():
    arrays = synthetic_grid([(0, 0, 1, 0, 100, 0), (0, 0, 1, 1, 100, 0)])
    c = C.endpoint_contrasts(arrays, [6, 8])["0"]
    assert c["D_all_pp"] == c["D_common_pp"] == c["D_standardized_pp"] == 0
    assert c["common_block_count"] == 1
    assert c["endpoints"]["low"]["supported_fractions"] == dict(N=1., A=None, T=None)
    assert c["endpoints"]["low"]["total"]["T_over_A"] is None


def test_large_full_tuple_never_float_or_packed():
    ids = dict(run=np.array([1, 1, 1, 2**53+1]), subrun=np.array([2, 2, 3, 2]),
               event=np.array([1, 2**32+1, 1, 1]))
    keys = J.identities(ids)
    assert J.lookup(keys[::-1], keys).tolist() == [3, 2, 1, 0]
    binding = C.track_binding(ids, np.array([3, 3, 1]), np.array([0, 1, 4]))
    expected = np.array([[1, 2, 2**32+1, 4], [2**53+1, 2, 1, 0], [2**53+1, 2, 1, 1]], dtype="<i8")
    assert binding["sha256"] == J.sha(expected.tobytes())
    with pytest.raises(ValueError, match="differ"):
        J.exact_measurement_equal(np.array([2**53+1]), np.array([float(2**53)]), "measured_nprotons")


def test_relative_ledger_and_output_code_boundary(saved):
    saved.original["exposure_ledger"]["path"] = "../ledger/ledger.parquet"
    saved.repin()
    assert saved.run()["counts"]["selected_tracks_matched"] == 9
    with pytest.raises(ValueError, match="output under"):
        saved.run(out=Path(C.__file__).parents[2] / "never_composition.json")


def test_exclusive_open_race_preserves_competitor(saved, monkeypatch):
    outside = J.outside
    calls = 0
    def race(path, paths):
        nonlocal calls
        calls += 1
        outside(path, paths)
        if calls == 4:
            path.write_bytes(b"competing output")
    monkeypatch.setattr(J, "outside", race)
    with pytest.raises(FileExistsError):
        saved.run()
    assert saved.out.read_bytes() == b"competing output"
