"""The XCOM attenuation table on the shared library core (mu2efastsim.library): bit-identical to the pre-migration
code (frozen copies below), for float32 and float64 energies, including the unsupported-energy accounting."""
import json
import shutil

import numpy as np
import pytest

from beamfastsim import flash_attenuation as FA
from mu2efastsim.library import hashing, store as LS


def _legacy_table():
    return np.loadtxt(FA._DATA, delimiter=",", comments="#", unpack=True)


def _legacy_mu_rho(energy):
    energy = np.asarray(energy, dtype=float)
    grid, mu = _legacy_table()
    supported = (np.isfinite(energy) & (energy >= grid[0]) & (energy <= grid[-1])
                 & ~((energy > FA._EDGE_LOW) & (energy < FA._EDGE_HIGH)))
    safe = np.where(supported, energy, grid[0])
    result = np.exp(np.interp(np.log(safe), np.log(grid), np.log(mu)))
    return np.where(supported, result, np.nan)


def _energies(dtype):
    rng = np.random.default_rng(5)
    e = np.exp(rng.uniform(np.log(1e-4), np.log(1e6), 50000))
    e[:6] = [np.nan, np.inf, 0.0015595, 0.0015596, 0.00155955, 1e-3]      # nonfinite, both K-edge ends, the gap, low end
    return e.astype(dtype)


def test_table_loads_bit_identical_with_its_manifest():
    new, old = FA._table(), _legacy_table()
    assert new.dtype == old.dtype and new.tobytes() == old.tobytes()
    m = json.loads(LS.sidecar_path(FA._DATA).read_text())
    assert m["name"] == FA.LIBRARY and m["axes"][0]["policy"] == "nan"
    assert m["content_hash"] == hashing.content_hash({"energy_mev": old[0], "mu_rho_cm2_g": old[1]})
    assert m["source"]["files"][0]["sha256"] == hashing.file_sha256(FA._DATA)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lookup_and_unsupported_accounting_are_bit_identical(dtype):
    e = _energies(dtype)
    new, old = FA.al_mu_rho(e), _legacy_mu_rho(e)
    assert new.dtype == old.dtype and new.tobytes() == old.tobytes()
    w = FA.uncollided_weight(e, np.full(e.shape, 3.0))
    ref_ok = np.isfinite(old)
    assert np.array_equal(w["supported"], ref_ok)
    assert int((~w["supported"]).sum()) == int((~ref_ok).sum()) > 0                     # unsupported photons
    assert float(e[~w["supported"] & np.isfinite(e)].astype(float).sum()) == float(
        e[~ref_ok & np.isfinite(e)].astype(float).sum())                                 # and their energy


def test_sidecar_is_reproducible_and_tampering_is_refused(tmp_path):
    copy = tmp_path / FA._DATA.name
    shutil.copyfile(FA._DATA, copy)
    FA.write_manifest(copy)
    got = json.loads(LS.sidecar_path(copy).read_text())
    assert got == json.loads(LS.sidecar_path(FA._DATA).read_text())
    text = copy.read_text().replace("0.001,1185.20263047458", "0.001,1185.20263047459")
    copy.write_text(text)
    with pytest.raises(LS.LibraryError, match="content hash"):
        FA.load_table(copy)
    LS.sidecar_path(copy).unlink()
    with pytest.raises(LS.LibraryError, match="no sidecar manifest"):
        FA.load_table(copy)
