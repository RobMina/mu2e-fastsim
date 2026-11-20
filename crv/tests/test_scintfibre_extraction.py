"""lookupgen.core is re-exported from scint-fibre-fastsim; old import paths stay identical."""

from __future__ import annotations

import numpy as np
import pytest


def test_core_modules_are_scintfibrefastsim():
    from crvfastsim.lookupgen import core as lg_core
    import scintfibrefastsim.core as sf_core

    for name in ("binning", "counter", "limits", "materials", "optics", "run", "transport"):
        assert getattr(lg_core, name) is getattr(sf_core, name), name


def test_optics_and_materials_symbols_match():
    from crvfastsim.lookupgen.core import materials as M
    from crvfastsim.lookupgen.core import optics as O
    import scintfibrefastsim.core.materials as sM
    import scintfibrefastsim.core.optics as sO

    assert O.G4Vector is sO.G4Vector
    assert O.DEFAULT_MODEL == sO.DEFAULT_MODEL == "current"
    assert M.CURRENT is sM.CURRENT


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_shimmed_generate_dtype_parity_smoke(dtype):
    """Both dtypes complete on the shimmed path with the same seed (flag-off smoke)."""
    pytest.importorskip("jax")
    from crvfastsim.lookupgen.core.binning import Binning
    from crvfastsim.lookupgen.core.counter import Counter
    from crvfastsim.lookupgen.core.run import generate

    c = Counter(
        length=200.0, thickness=20.0, width=50.0, fiber_separation=26.0, fiber_diameter=1.4,
        clad_fraction=0.03, hole_rx=1.0, hole_ry=1.0, coating=0.25, corner_radius=1.0,
        guide_bar=5.0, sipm_width=2.0, sipm_window=0.1, sipm_length=0.3, air_gap=0.001,
    )
    edges = {
        "x": np.linspace(0.0, c.hx, 3),
        "y": np.linspace(-c.hy, c.hy, 3),
        "z": np.linspace(-c.half_length, c.half_length, 3),
    }
    b = Binning(edges=edges, order={0: ["x", "y", "z"]})
    r1 = generate(c, b, 0, np.arange(b.n_bins(0)), 40, dtype=dtype, seed=3,
                  lanes=256, chunk_iters=400)
    r2 = generate(c, b, 0, np.arange(b.n_bins(0)), 40, dtype=dtype, seed=3,
                  lanes=256, chunk_iters=400)
    assert int(r1["gen"].sum()) > 0
    assert np.array_equal(r1["gen"], r2["gen"])
    assert np.array_equal(r1["ht"], r2["ht"])
    assert np.array_equal(r1["he"], r2["he"])


def test_sipm_bind_helper_importable():
    from scintfibrefastsim.backends import Mu2eSiPMBackend, bind_crvfastsim_sipm

    assert Mu2eSiPMBackend().sensor_kind == "sipm"
    assert callable(bind_crvfastsim_sipm)
