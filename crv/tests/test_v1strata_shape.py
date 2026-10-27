"""v1strata binned-ratio shapes: Poisson limit equals v1compare's ratio statistic; clustering inflates the variance."""
import numpy as np

from crvfastsim.analyses import v1compare as VC
from crvfastsim.analyses import v1strata as VS


def _arm(rng, n_events, per_event, shift=0.0):
    ev = np.repeat(np.arange(n_events), per_event)
    v = np.repeat(rng.normal(40.0 + shift, 8.0, n_events), per_event)   # every crossing of an event in one bin
    return v, ev


def _shape(va, eva, vb, evb, n):
    h_b = np.histogram(vb, VC.EDGES["len"])[0]
    group = {"len_ratio": VC._hist_ratio_band(np.histogram(va, VC.EDGES["len"])[0], h_b, VC.EDGES["len"])}
    edges, full = VS.shape_edges(group, "len")
    vec = VS._shape_vectors(va, vb, eva, evb, n, edges, full)
    return group["len_ratio"], VS.summarise_shape(VS._moments(vec), n, min_ref=1)


def test_poisson_limit_matches_v1compare():
    rng = np.random.default_rng(1)
    n = 40000
    va, eva = _arm(rng, n, 1)
    vb, evb = _arm(rng, n, 1)
    vc, sh = _shape(va, eva, vb, evb, n)
    assert sh["nbins_used"] == vc["nbins_used"] == 20
    np.testing.assert_allclose(sh["chi2_ndf_poisson"], vc["chi2_ndf"], rtol=1e-9)
    assert abs(sh["design_effect_median"] - 1) < 0.15


def test_clustered_variance_scales_with_crossings_per_event():
    rng = np.random.default_rng(2)
    n = 20000
    va, eva = _arm(rng, n, 4)
    vb, evb = _arm(rng, n, 4)
    vc, sh = _shape(va, eva, vb, evb, n)
    np.testing.assert_allclose(sh["chi2_ndf_poisson"], vc["chi2_ndf"], rtol=1e-9)
    assert 3.0 < sh["design_effect_median"] < 5.0
    assert sh["chi2_ndf_clustered"] < 0.5 * sh["chi2_ndf_poisson"]
