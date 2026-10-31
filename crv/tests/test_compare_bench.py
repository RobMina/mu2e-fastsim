"""The optional derived-property tolerance must never hide membership or pulse changes."""
import importlib.util
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1] / "scripts" / "compare_bench.py"
spec = importlib.util.spec_from_file_location("bench_compare", P)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_rounding_only_restricted_to_derived_cluster_properties():
    assert M.rounding_only([{"chunk_0000__cl_pos": {"max_abs_diff": 2e-9}}])
    assert not M.rounding_only([{"chunk_0000__mem_idx": {"max_abs_diff": 1}}])
    assert not M.rounding_only([{"chunk_0000__pulses__pe": {"max_abs_diff": 1e-14}}])
    assert not M.rounding_only([{"chunk_0000__cl_pe": {"shape_dtype_mismatch": []}}])
    assert not M.rounding_only([{"chunk_0000__cl_pos": {"max_abs_diff": 2e-8}}])
    assert not M.rounding_only([{"chunk_0000__cl_pos": {"max_abs_diff": float("nan")}}])


def test_diff_reports_members_and_float_properties(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, chunk_0000__cl_pe=np.array([2.]), chunk_0000__mem_idx=np.array([0]))
    np.savez(b, chunk_0000__cl_pe=np.array([2. + 1e-12]), chunk_0000__mem_idx=np.array([1]))
    diff = M.array_differences(a, b)
    assert diff["chunk_0000__mem_idx"]["different_values"] == 1
    assert not M.rounding_only([diff])
