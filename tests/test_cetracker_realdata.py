"""Real-data check of the cechain -> fieldsteps hand-off (skipped unless CETRACKER_CHECK_DIR is set).

CETRACKER_CHECK_DIR: a `python3 -m beamfastsim.cetracker` output directory (states.root, fasttrk_field.root) of
Geant4 CeEndpoint dts tables that also holds the Geant4 arm of the same events (`python3 -m
trkfastsim.validation.cearms`: g4entries.root, g4prim_field.root). Fast and Geant4 are independent histories of the
same births, so their states agree in distribution, not per event; a wrong plane or tracker frame shows as a shift.
"""
import os

import numpy as np
import pytest

from mu2efastsim import tables

D = os.environ.get("CETRACKER_CHECK_DIR")
pytestmark = pytest.mark.skipif(not D, reason="CETRACKER_CHECK_DIR not set")
KEY = tables.KEY
Z_PLANE, AXIS = 8539.9, -3904.0


def _mean_err(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def _agree(a, b, nsig=4.0):
    (ma, ea), (mb, eb) = _mean_err(a), _mean_err(b)
    assert abs(ma - mb) < nsig * np.hypot(ea, eb), (ma, mb, ea, eb)


def test_handoff_plane_and_frame_agree_with_geant4():
    gen = tables.read(f"{D}/states.root", "gen")
    fast = tables.read(f"{D}/states.root", "entries")
    g4 = tables.read(f"{D}/g4entries.root", "entries")
    assert len(fast) > 0.99 * len(gen) and len(g4) > 0.99 * len(gen)
    np.testing.assert_allclose(fast.z, Z_PLANE, atol=1e-6)
    np.testing.assert_allclose(g4.z, Z_PLANE, atol=0.5)     # Run 1A (MDC2025ba) VD13 is at 8539.48 mm
    for df in (fast, g4):
        df["r"] = np.hypot(df.x - AXIS, df.y)
        df["p"] = np.sqrt(df.px ** 2 + df.py ** 2 + df.pz ** 2)
        df["cos"] = df.pz / df.p
    for c in ("r", "cos", "x", "y"):
        _agree(fast[c], g4[c])
    sf = tables.read(f"{D}/fasttrk_field.root", "sgsteps")
    sg = tables.read(f"{D}/g4prim_field.root", "sgsteps")
    ff = sf.sort_values("t").drop_duplicates(KEY)
    fg = sg.sort_values("t").drop_duplicates(KEY)
    for c in ("x0", "y0", "z0"):
        _agree(ff[c], fg[c])
    _agree(np.hypot(ff.x0, ff.y0), np.hypot(fg.x0, fg.y0))
