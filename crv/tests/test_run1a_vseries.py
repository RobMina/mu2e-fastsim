"""Run 1A V-series paired statistics and the runchain --save-reco layout."""
import numpy as np
import pytest

from crvfastsim.analyses import runchain
from crvfastsim.validation import run1a_vseries as V


def test_paired_ratio_identical_and_shifted():
    r = np.array([100., 120., 90., 110.])
    same = V.paired_ratio(r, r)
    assert same["ratio"] == 1.0 and same["err"] == 0.0
    s = V.paired_ratio(1.1 * r, r)
    assert s["ratio"] == pytest.approx(1.1) and s["err"] == pytest.approx(0.0, abs=1e-12)
    n = V.paired_ratio(r + np.array([3., -3., 3., -3.]), r)
    assert n["ratio"] == pytest.approx(1.0) and n["err"] > 0


def test_paired_shape_pulls_and_chi2():
    rng = np.random.default_rng(1)
    R = rng.poisson(50, size=(10, 6)).astype(float)
    F = R + rng.normal(0, 3, size=R.shape)
    res = V.paired_shape(F, R)
    assert res["ndf"] == 6 and np.isfinite(res["chi2_ndf"])
    big = V.paired_shape(R + 10 + rng.normal(0, 1, size=R.shape), R)
    assert big["chi2_ndf"] > 100 and min(big["pull"]) > 5
    zero = V.paired_shape(R, R)
    assert zero["ndf"] == 0 and np.isnan(zero["chi2_ndf"])


def test_save_reco_round_trip(tmp_path):
    out = [{"cl_event": np.array([5, 5, 7]), "cl_type": np.array([1, 1, 1]), "cl_pe": np.array([10., 20., 30.]),
            "cl_start": np.zeros(3), "cl_end": np.ones(3), "cl_avgtime": np.ones(3),
            "mem_cl": np.array([0, 0, 1, 2]), "mem_idx": np.array([0, 1, 2, 0]),
            "pulses": dict(ev=np.array([5, 5, 5, 7]), idx=np.array([0, 1, 2, 0]), bar=np.array([0, 1, 2, 3]),
                           sipm=np.zeros(4, int), pe=np.array([5., 6., 7., 8.]), time=np.zeros(4))}]
    p = tmp_path / "r.npz"
    runchain._save_reco(p, out, 100, "run1a_v01")
    cl, mem, pulses = V._reco_frames_fast(p)
    assert cl.pe.tolist() == [10., 20., 30.] and mem.ev.tolist() == [5, 5, 5, 7]
    m = mem.merge(pulses, on=["ev", "idx"])
    assert m.bar.tolist() == [0, 1, 2, 3]


CL_KEYS = ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_avgtime")
PULSE_KEYS = ("ev", "idx", "bar", "sipm", "pe", "time")
EXPORT_KEYS = {*CL_KEYS, "mem_cl", "mem_idx", *(f"p_{k}" for k in PULSE_KEYS), "n_events", "run_config"}


def _reco_chunk(offset, int_dtype, float_dtype):
    return {
        "cl_event": np.array([5, 5, 7], dtype=int_dtype) + offset,
        "cl_type": np.array([1, 2, 2], dtype=int_dtype),
        "cl_pe": np.array([10., 20., 30.], dtype=float_dtype) + offset,
        "cl_start": np.array([1., 2., 3.], dtype=float_dtype) + offset,
        "cl_end": np.array([4., 5., 6.], dtype=float_dtype) + offset,
        "cl_avgtime": np.array([2., 3., 4.], dtype=float_dtype) + offset,
        "mem_cl": np.array([0, 0, 1, 2], dtype=int_dtype),
        "mem_idx": np.array([0, 1, 2, 1], dtype=int_dtype),
        "pulses": {
            "ev": np.array([5, 5, 5, 7, 7], dtype=int_dtype) + offset,
            "idx": np.array([0, 1, 2, 0, 1], dtype=int_dtype),
            "bar": np.array([10, 11, 12, 13, 14], dtype=int_dtype) + offset,
            "sipm": np.array([0, 1, 2, 3, 0], dtype=int_dtype),
            "pe": np.array([5., 6., 7., 8., 9.], dtype=float_dtype) + offset,
            "time": np.array([1., 2., 3., 4., 5.], dtype=float_dtype) + offset,
        },
    }


def _empty_reco_chunk(int_dtype=np.int32, float_dtype=np.float32):
    out = _reco_chunk(0, int_dtype, float_dtype)
    return {**{k: v[:0] for k, v in out.items() if k != "pulses"},
            "pulses": {k: v[:0] for k, v in out["pulses"].items()}}


@pytest.mark.parametrize("int_dtype,float_dtype", [(np.int32, np.float32), (np.int64, np.float64)])
@pytest.mark.parametrize("layout", ["two", "three", "empty-first", "empty-between", "untyped-empty", "empty-last", "pulse-only"])
def test_save_reco_keeps_every_chunk(tmp_path, int_dtype, float_dtype, layout):
    chunks = [_reco_chunk(offset, int_dtype, float_dtype) for offset in (0, 20)]
    cluster_offsets = [0, 20]
    if layout == "three":
        chunks.append(_reco_chunk(40, int_dtype, float_dtype))
        cluster_offsets.append(40)
    elif layout == "empty-first":
        chunks.insert(0, {})
    elif layout == "empty-between":
        chunks.insert(1, _empty_reco_chunk(int_dtype, float_dtype))
    elif layout == "untyped-empty":
        empty = _empty_reco_chunk()
        chunks.insert(1, {**{k: np.array([]) for k in empty if k != "pulses"},
                          "pulses": {k: np.array([]) for k in PULSE_KEYS}})
    elif layout == "empty-last":
        chunks.append({})
    elif layout == "pulse-only":
        empty = _empty_reco_chunk(int_dtype, float_dtype)
        empty["pulses"] = _reco_chunk(40, int_dtype, float_dtype)["pulses"]
        chunks.insert(1, empty)
    path = tmp_path / "r.npz"
    runchain._save_reco(path, chunks, 100, "run1a_v01")
    with np.load(path) as z:
        assert set(z.files) == EXPORT_KEYS
        assert z["n_events"].shape == () and z["n_events"].dtype == np.int64 and z["n_events"] == 100
        assert z["run_config"].shape == () and z["run_config"].dtype == np.asarray("run1a_v01").dtype
        assert z["run_config"] == "run1a_v01"
        for k in CL_KEYS:
            expected = np.concatenate([c[k] for c in chunks if k in c and len(c[k])])
            np.testing.assert_array_equal(z[k], expected, strict=True)
        for k in PULSE_KEYS:
            expected = np.concatenate([c["pulses"][k] for c in chunks if "pulses" in c and len(c["pulses"][k])])
            np.testing.assert_array_equal(z[f"p_{k}"], expected, strict=True)
        expected_cl = np.concatenate([np.array([0, 0, 1, 2], np.int64) + 3 * i for i in range(len(cluster_offsets))])
        np.testing.assert_array_equal(z["mem_cl"], expected_cl, strict=True)
        np.testing.assert_array_equal(z["mem_idx"], np.tile(np.array([0, 1, 2, 1], np.int64), len(cluster_offsets)),
                                      strict=True)
    cl, mem, pulses = V._reco_frames_fast(path)
    assert len(cl) == 3 * len(cluster_offsets)
    assert len(pulses) == 5 * (len(cluster_offsets) + (layout == "pulse-only"))
    m = mem.merge(pulses, on=["ev", "idx"], how="left", validate="many_to_one")
    assert len(m) == 4 * len(cluster_offsets) and not m.isna().any().any()
    assert m.bar.tolist() == [bar + offset for offset in cluster_offsets for bar in (10, 11, 12, 14)]
    assert cl.pe.to_numpy()[m.cl].tolist() == [pe + offset for offset in cluster_offsets for pe in (10., 10., 20., 30.)]


@pytest.mark.parametrize("n_events", [0, 100])
@pytest.mark.parametrize("layout", ["no-chunks", "empty-dict", "typed-empty", "multiple-empty"])
def test_save_reco_empty_export(tmp_path, n_events, layout):
    empty = _empty_reco_chunk()
    chunks = {"no-chunks": [], "empty-dict": [{}], "typed-empty": [empty], "multiple-empty": [empty, empty]}[layout]
    path = tmp_path / "empty.npz"
    runchain._save_reco(path, chunks, n_events, "run1a_v01")
    with np.load(path) as z:
        assert set(z.files) == EXPORT_KEYS
        assert z["n_events"].shape == () and z["n_events"].dtype == np.int64 and z["n_events"] == n_events
        assert z["run_config"].shape == () and z["run_config"].dtype == np.asarray("run1a_v01").dtype
        assert z["run_config"] == "run1a_v01"
        for k in EXPORT_KEYS - {"n_events", "run_config"}:
            assert z[k].shape == (0,)
            dtype = np.float64
            if layout in ("typed-empty", "multiple-empty"):
                dtype = empty["pulses"][k[2:]].dtype if k.startswith("p_") else empty[k].dtype
            if k in ("mem_cl", "mem_idx"):
                dtype = np.int64
            assert z[k].dtype == dtype
    cl, mem, pulses = V._reco_frames_fast(path)
    assert cl.empty and mem.empty and pulses.empty
