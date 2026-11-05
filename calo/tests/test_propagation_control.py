"""Synthetic histogram and stock-output fixtures; no stock execution certificate."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_propagation", ROOT / "scripts/check_propagation.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class Axis:
    def __init__(self, n, low, high, edges=()):
        self.values = dict(fNbins=n, fXmin=low, fXmax=high, fXbins=edges)

    def member(self, name):
        return self.values[name]


def test_root_fixed_and_variable_axis_arithmetic():
    width, centers, metadata = checker.root_axis(Axis(3, 0, 1))
    assert width == np.float32(1 / 3)
    np.testing.assert_array_equal(centers, np.array([1/6, .5, 5/6], dtype=np.float32))
    assert not metadata["variable"]
    width, centers, metadata = checker.root_axis(Axis(3, -1, 4, [-1, 0, 1, 4]))
    assert width == 1
    np.testing.assert_array_equal(centers, [-.5, .5, 2.5])
    assert metadata["variable"]


@pytest.mark.parametrize("axis", [Axis(0, 0, 1), Axis(3, 1, 0), Axis(2, 0, 1, [0, 1]),
                                  Axis(2, 0, 1, [0, 1, 1]), Axis(2, 0, 1, [0, np.nan, 1])])
def test_bad_root_axes(axis):
    with pytest.raises(ValueError):
        checker.root_axis(axis)


def test_long_sequential_recurrence_and_threshold_neighbors():
    contents = np.array([[0, 1e-6, 1e8, 1, 0, 1e8, 2, 0], [0]*8], dtype=np.float32)
    cdf = checker.build_cdf(contents)
    np.testing.assert_array_equal(cdf, checker.source_cdf(contents))
    times = np.arange(8, dtype=np.float32)
    result = checker.compare(contents, cdf, times, np.float32(5))
    assert result["supplied_uniform_lookup_exact"]
    assert result["zero_content_depth_slices"] == 1
    assert result["stock_sampling_executed"] is False
    assert result["probe_count"] == 3 * contents.size + 6 * (3 * (len(cdf) + 1) + 2)


@pytest.mark.parametrize("backend,jit", [("numpy", False), ("jax", False), ("jax", True)])
def test_supplied_probe_backends(backend, jit):
    if backend == "jax":
        pytest.importorskip("jax")
    contents = np.array([[0, 2, 0, 3, 0], [0, 0, 0, 0, 0], [1, 5, 0, 0, 0]], dtype=np.float32)
    cdf = checker.build_cdf(contents)
    result = checker.compare(contents, cdf, np.arange(5, dtype=np.float32), np.float32(5), backend=backend, jit=jit)
    assert result["source_reconstructed_cdf_exact"]
    assert result["lookup_bit_mismatches"] == 0


def write_stock_fixture(path, *, mutation=None):
    contents = np.array([[0, 2, 3], [0, 0, 0]], dtype=np.float32)
    cdf, times, dz = checker.build_cdf(contents), np.array([1, 2, 5], dtype=np.float32), np.float32(5)
    depth, draws, rows = checker.probes(cdf, dz)
    observed = times[checker.expected_bins(cdf, rows, draws)].copy()
    before, after = np.arange(len(depth), dtype=np.uint64), np.arange(1, len(depth) + 1, dtype=np.uint64)
    iz, it = np.repeat(np.arange(2), 3), np.tile(np.arange(3), 2)
    filename = "wrong.root" if mutation == "file" else "test.root"
    histogram = "wrong_hist" if mutation == "histogram" else "h_zposTime"
    if mutation == "time":
        observed[0] += 1
    if mutation == "draw":
        after[0] += 1
    if mutation == "depth":
        depth[0] += 1
    if mutation == "uniform":
        draws[0] = .99
    if mutation == "registry":
        iz[0] = 1
    if mutation == "cdf":
        cdf = cdf.copy()
        cdf[0, 0] = .5
    with uproot.recreate(path) as output:
        output.mktree("propagation/metadata", {"nz": np.array([2]), "nt": np.array([3]),
                                               "dz": np.array([dz]), "refractive_index": np.array([1.8]),
                                               "file": [filename], "histogram": [histogram]})
        output.mktree("propagation/bins", {"iz": iz, "it": it, "content": contents.ravel(),
                                          "time": np.tile(times, 2), "probe_threshold": cdf.ravel()})
        output.mktree("propagation/probes", {"depth": depth, "uniform": draws, "time": observed,
                                            "draws_before": before, "draws_after": after})
    return contents, checker.build_cdf(contents), times, dz


def compare_fixture(path, table, **overrides):
    options = {"expected_sha256": checker.sha256(path), "expected_file": "test.root", **overrides}
    return checker.compare_stock(path, *table, **options)


def test_stock_fixture_exact_and_mismatch(tmp_path):
    path = tmp_path / "stock.root"
    table = write_stock_fixture(path)
    result = compare_fixture(path, table)
    assert result["stock_public_lookup_exact"]
    assert result["stock_cache_directly_observed"] is False
    assert result["stochastic_response_validated"] is False
    table = write_stock_fixture(path, mutation="time")
    assert compare_fixture(path, table)["stock_lookup_bit_mismatches"] == 1


@pytest.mark.parametrize("mutation", ["draw", "depth", "uniform", "registry", "cdf", "file", "histogram"])
def test_stock_fixture_bad_provenance_or_coverage(tmp_path, mutation):
    path = tmp_path / "stock.root"
    table = write_stock_fixture(path, mutation=mutation)
    with pytest.raises(ValueError):
        compare_fixture(path, table)


def test_stock_fixture_wrong_hash(tmp_path):
    path = tmp_path / "stock.root"
    table = write_stock_fixture(path)
    with pytest.raises(ValueError, match="SHA256"):
        compare_fixture(path, table, expected_sha256="0" * 64)


@pytest.mark.parametrize("arguments", [["--stock-control", "test.root"], ["--stock-control-sha256", "0" * 64],
                                        ["--stock-resolved-file", "test.root"]])
def test_cli_stock_pin_arguments_required_together(tmp_path, arguments):
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "scripts/check_propagation.py"), "unused.root",
                             "--input-sha256", "0" * 64, "--stock-source", "unused.cc",
                             "--stock-source-sha256", "0" * 64, "--out", str(tmp_path / "out.json"), *arguments],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert "required together" in result.stderr


def test_driver_and_stock_oracle_build_contract():
    driver = ROOT / "scripts/propagation_control.sh"
    assert subprocess.run(["bash", "-n", str(driver)]).returncode == 0
    text = driver.read_text()
    for required in ("CALO_PLUGIN_SHA256", "CALO_TABLE_SHA256", "CALO_STOCK_SOURCE_SHA256", "ldd -r", "-n 1",
                     "readlink -f", 'mkdir "$CALO_OUT"', "-nt", "sha256sum -c -"):
        assert required in text
    assert "muse build" not in text and "scons" not in text and "Mu2eG4" not in text
    source = (ROOT / "art/CaloFastSim/src/CaloPropagationDumper_module.cc").read_text()
    assert source.count("stock.buildTable();") == 1
    assert "stock.propTimeSimu(depth)" in source
    assert "std::lower_bound" not in source
    assert "after != before + 1" in source
    assert "ResetBranchAddresses" in source
    cmake = (ROOT / "art/CaloFastSim/CMakeLists.txt").read_text()
    scons = (ROOT / "art/CaloFastSim/src/SConscript").read_text()
    for library in ("CaloMC", "ConfigTools"):
        assert f"Offline::{library}" in cmake and f"'mu2e_{library}'" in scons
    fcl = (ROOT / "art/CaloFastSim/fcl/propagation_control.fcl").read_text()
    assert "EmptyEvent maxEvents : 1" in fcl and "CaloPropagationDumper" in fcl
    assert "OfflineData/ConditionsService/CsI-propag-2020-07-29.root" in fcl
    assert "CaloShowerROMaker" not in fcl and "g4run" not in fcl
