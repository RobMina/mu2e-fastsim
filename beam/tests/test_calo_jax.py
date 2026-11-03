"""Unit tests: calorimeter-acceptance proxy on the device (beamfastsim.calo_jax) against the NumPy proxy (cesob)."""
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from beamfastsim import calo_jax as CJ  # noqa: E402
from beamfastsim import cesob as S  # noqa: E402

from test_cesob import CALO, _helix_starts  # noqa: E402

FIELD = [0.0, 0.0, 1.0]


def _spread(n, seed=5, z=8539.9):
    """e- of 103 MeV at the entry plane with helix centres and pitches spread over the bore and the annulus."""
    rng = np.random.default_rng(seed)
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    xs, us = [], []
    for cos, rmax in zip(rng.uniform(0.45, 0.9, n), rng.uniform(250.0, 700.0, n)):
        R = math.sqrt(1 - cos ** 2) * p / 0.299792458
        x, u, _ = _helix_starts(rmax - R, R, n=1, cos=cos, z=z)
        ph = rng.uniform(0, 2 * np.pi)
        c, s = math.cos(ph), math.sin(ph)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        x = x.copy()
        x[:, 0] -= S.calo_from_config(CALO).axis_x
        x = x @ rot.T
        x[:, 0] += S.calo_from_config(CALO).axis_x
        xs.append(x), us.append(u @ rot.T)
    return np.concatenate(xs), np.concatenate(us), np.full(n, T)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("annulus", [False, True])
def test_vacuum_proxy_equals_the_numpy_proxy(dtype, annulus):
    c = S.calo_from_config(CALO)
    x, u, T = _spread(400)
    ref = S.calo_accept(FIELD, c, x, u, T, annulus=annulus)
    got = CJ.calo_accept(jax.random.PRNGKey(0), FIELD, c, x, u, T, annulus=annulus, dtype=dtype, min_bucket=64)
    assert 0.2 < ref.mean() < 0.9
    assert (got != ref).sum() <= (0 if dtype == np.float64 else 2)


def test_wall_proxy_agrees_with_the_numpy_proxy_and_keeps_its_limits():
    c = S.calo_from_config(CALO)
    w = S.calo_wall_from_config(CALO, c)
    T = 103.0
    p = math.sqrt(T * (T + 2 * 0.51099895))
    cos = 0.15
    R = math.sqrt(1 - cos ** 2) * p / 0.299792458
    x, u, _ = _helix_starts(0.0, R, n=8)                   # inside the bore: never meets the wall
    info = {}
    hit = CJ.calo_accept(jax.random.PRNGKey(1), FIELD, c, x, u, np.full(8, T), wall=w, info=info, min_bucket=64)
    assert not hit.any() and info["al"] == info["cf"] == 0
    x, u, _ = _helix_starts(520.0 - R, R, n=2, cos=cos, z=11895.0)
    x, u = np.repeat(x[1:], 8, 0), np.repeat(u[1:], 8, 0)
    info = {}
    hit = CJ.calo_accept(jax.random.PRNGKey(2), FIELD, c, x, u, np.full(8, T), wall=w, info=info, min_bucket=64)
    assert info["cf"] == 8 and hit.all()
    thick = S.CaloWall(w.case_lo, w.case_hi, 200.0, 356.0, 360.0, 200.0)
    info = {}
    hit = CJ.calo_accept(jax.random.PRNGKey(3), FIELD, c, x, u, np.full(8, T), wall=thick, info=info, min_bucket=64)
    assert info["al"] == 8 and not hit.any()
    # a spread sample: accepted fraction and crossing counts as the NumPy proxy within statistics
    x, u, T = _spread(1500, seed=11)
    ri, rj = {}, {}
    ref = S.calo_accept(FIELD, c, x, u, T, wall=w, rng=np.random.default_rng(4), info=ri)
    got = CJ.calo_accept(jax.random.PRNGKey(4), FIELD, c, x, u, T, wall=w, info=rj, dtype=np.float64, min_bucket=64)
    assert ri["cf"] > 50 and abs(rj["cf"] - ri["cf"]) <= 4 * math.sqrt(ri["cf"])
    assert abs(got.mean() - ref.mean()) < 4 * math.sqrt(ref.mean() * (1 - ref.mean()) / ref.size) + 0.01
