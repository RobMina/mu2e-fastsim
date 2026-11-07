"""Distributions drawn by the resident JAX sampler (sample_overlay), with fixed seeds.

Expectations are analytic: the beam-weighted (x f(x)) truncated lognormal intensity rounded to integers, Poisson
multiplicities conditional on that intensity, and stream covariance Cov(n_i, n_j) = r_i r_j Var(N) from the shared
intensity. Every tolerance is a fixed multiple of the statistic's own sampling error: |z| < Z_MAX for means of per-event
statistics, and p > P_MIN for Pearson chi-square tests.
"""
import itertools
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr, ndtri

jax = pytest.importorskip("jax")

from mixing.device_backend import Source
from mixing.device_source import Layout, pack_source, sample_overlay
from mixing.frames import FrameBatch
from mixing.sampling import BunchLaw, STREAMS, draw_multiplicities

Z_MAX = 4.0     # two-sided normal tail probability 6e-5 per statistic
P_MIN = 1e-4
BATCH, BATCHES = 20_000, 5
RATES = dict(zip(STREAMS, (2e-3, 1e-3, 5e-4, 3e-3)))
SKIP = dict(zip(STREAMS, (7, 3, 11, 5)))
POOL_SIZES = (53, 47, 61, 59)
LAWS = {"upper_cut": BunchLaw.from_sdf(1000., .6, 5000.),
        "both_cuts": BunchLaw.from_sdf(1000., .6, 5000., cut_min=500.)}


def frames(size, index):
    counts = (np.arange(size) % 3).astype(np.int64)
    ev = np.repeat(np.arange(size), counts)
    local = np.arange(counts.sum()) - np.r_[0, np.cumsum(counts)][ev]
    key = local * 10 + 3
    jagged = lambda columns: ak.unflatten(ak.zip(columns), counts)
    collections = {"particles": jagged({"key": key, "parent": np.where(local > 0, 3, -1), "genidx": local}),
                   "gen": jagged({"idx": local}),
                   "tracker": jagged({"sp": key, "sid": (ev + index) % 8, "t": ev * 1. + 400.}),
                   "crv": jagged({"sp": key, "bar": (ev + index) % 8, "t0": ev * 1. + 400.})}
    ids = np.column_stack([np.full(size, 7 + index, np.uint64), np.ones(size, np.uint64),
                           np.arange(size, dtype=np.uint64)])
    return FrameBatch(ids, collections)


def source(law):
    pools = [frames(BATCH, 0)] + [frames(size, index + 1) for index, size in enumerate(POOL_SIZES)]
    result = Source.__new__(Source)
    result._signal, result._pools = pools[0], dict(zip(STREAMS, pools[1:]))
    result._seed, result._max_selected_rows, result._ordinal, result._batch_index = 0, 10**8, 0, 0
    result._law, result._rates = law, RATES
    result._campaign = {"max_skip": SKIP, "mode": "sequential"}
    result.metadata = {}
    return result


class Replay:
    """Host RNG stand-in that returns the device's own uniforms."""

    def __init__(self, values):
        self.values = values

    def random(self, size):
        assert size == len(self.values)
        return self.values


@pytest.fixture(scope="module", params=sorted(LAWS))
def draws(request):
    import jax.numpy as jnp
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        law = LAWS[request.param]
        src = source(law)
        layout = Layout(BATCH, 48, dict.fromkeys(src.signal.collections, 400_000), 10**8)
        packed, constants = pack_source(src, layout)
        pools = jax.device_put(packed)
        step = jax.jit(lambda state, key, ordinal: sample_overlay(
            pools, state, key, ordinal, jnp.int64(BATCH), layout=layout, law=constants))
        state = {"first": np.array(True), "cursors": np.zeros(4, np.int64)}
        intensity, counts, mismatches = [], [], 0
        for index in range(BATCHES):
            key = jax.random.fold_in(jax.random.PRNGKey(20261107), index)
            _, receipt, state, errors = jax.device_get(step(state, key, jnp.uint64(index * BATCH)))
            assert not any(np.any(value) for value in errors.values()), errors
            intensity.append(receipt["intensity"])
            counts.append(receipt["multiplicities"])
            uniform = np.asarray(jax.random.uniform(jax.random.split(key, 3)[0], (BATCH,), dtype=jnp.float64))
            mismatches += int(np.sum(law.draw(Replay(uniform), BATCH, beam_primary=True) != receipt["intensity"]))
    finally:
        jax.config.update("jax_enable_x64", previous)
    return SimpleNamespace(law=law, N=np.concatenate(intensity), n=np.concatenate(counts), mismatches=mismatches)


def log_mu(law, beam=True):
    return np.log(law.extended_mean) + (.5 if beam else -.5) * law.sigma**2


def truncated_cdf(law, x):
    """CDF of the continuous truncated law before rounding."""
    mu = log_mu(law)
    phi = lambda v: ndtr((np.log(np.maximum(v, 1e-300)) - mu) / law.sigma)
    lower = phi(law.cut_min) if law.cut_min else 0.
    return np.clip((phi(np.asarray(x, float)) - lower) / (phi(law.cut_max) - lower), 0., 1.)


def truncated_moment(law, order, beam=True):
    """E[X^order] of the truncated lognormal."""
    mu, s = log_mu(law, beam), law.sigma
    mass = lambda shift: (ndtr((np.log(law.cut_max) - mu - shift) / s)
                          - (ndtr((np.log(law.cut_min) - mu - shift) / s) if law.cut_min else 0.))
    return np.exp(order * mu + .5 * (order * s) ** 2) * mass(order * s * s) / mass(0.)


def zscore(values):
    """Mean of per-event statistics in units of its standard error."""
    values = np.asarray(values, float)
    return values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))


def pearson_p(observed, expected, minimum=5.):
    """Pearson chi-square p-value after merging the upper tail until every bin expects >= minimum."""
    observed, expected = list(np.asarray(observed, float)), list(np.asarray(expected, float))
    while len(expected) > 2 and expected[-1] < minimum:
        expected[-2] += expected.pop()
        observed[-2] += observed.pop()
    observed, expected = np.array(observed), np.array(expected)
    assert np.all(expected >= minimum)
    return stats.chi2.sf(np.sum((observed - expected) ** 2 / expected), len(expected) - 1)


def test_intensity_is_the_exact_host_transform_of_the_device_uniforms(draws):
    assert draws.mismatches == 0


def test_intensity_follows_the_beam_weighted_truncated_law(draws):
    law, N = draws.law, draws.N
    assert N.min() >= law.cut_min and N.max() <= law.cut_max
    # 25 equal-probability bins on integer edges; rounding makes P(e_i <= N < e_j) = F(e_j - 1/2) - F(e_i - 1/2)
    mu, lower = log_mu(law), (ndtr((np.log(law.cut_min) - log_mu(law)) / law.sigma) if law.cut_min else 0.)
    upper = ndtr((np.log(law.cut_max) - mu) / law.sigma)
    quantiles = np.exp(mu + law.sigma * ndtri(lower + (upper - lower) * np.linspace(0, 1, 26)[1:-1]))
    edges = np.unique(np.rint(quantiles))
    observed = np.bincount(np.searchsorted(edges, N, side="right"), minlength=len(edges) + 1)
    cdf = np.r_[0., truncated_cdf(law, edges - .5), 1.]
    assert pearson_p(observed, len(N) * np.diff(cdf)) > P_MIN
    # first two moments; rounding adds 1/12 to the variance and a negligible bias to the mean
    mean, var = truncated_moment(law, 1), truncated_moment(law, 2) - truncated_moment(law, 1) ** 2 + 1 / 12
    assert abs(zscore(N - mean)) < Z_MAX
    centered = N - N.mean()
    z_var = (N.var(ddof=1) - var) / np.sqrt((np.mean(centered**4) - N.var() ** 2) / len(N))
    assert abs(z_var) < Z_MAX
    # the mean test resolves the beam weighting: the unweighted f(x) law is rejected
    assert abs(zscore(N - truncated_moment(law, 1, beam=False))) > 5 * Z_MAX


@pytest.mark.parametrize("column", range(4), ids=STREAMS)
def test_multiplicities_are_poisson_given_the_intensity(draws, column):
    lam = draws.N * RATES[STREAMS[column]]
    n = draws.n[:, column]
    residual = n - lam
    central = {1: np.zeros_like(lam), 2: lam, 3: lam, 4: lam + 3 * lam**2}
    for order, expected in central.items():
        z = zscore(residual**order - expected)
        assert abs(z) < Z_MAX, (order, z)
    top = int(n.max()) + 1
    expected = stats.poisson.pmf(np.arange(top)[None, :], lam[:, None]).sum(axis=0)
    expected[-1] += len(n) - expected.sum()
    assert pearson_p(np.bincount(n, minlength=top), expected) > P_MIN


@pytest.mark.parametrize("pair", list(itertools.combinations(range(4), 2)),
                         ids=["-".join(STREAMS[i] for i in pair) for pair in itertools.combinations(range(4), 2)])
def test_streams_share_the_intensity(draws, pair):
    i, j = pair
    N, n = draws.N.astype(float), draws.n.astype(float)
    ri, rj = RATES[STREAMS[i]], RATES[STREAMS[j]]
    # conditionally independent given N
    assert abs(zscore((n[:, i] - ri * N) * (n[:, j] - rj * N))) < Z_MAX
    # Cov(n_i, n_j) = r_i r_j Var(N); both per-event estimators carry the same (1 - 1/n) factor
    products = (n[:, i] - ri * N.mean()) * (n[:, j] - rj * N.mean())
    assert abs(zscore(products - ri * rj * (N - N.mean()) ** 2)) < Z_MAX
    # the covariance is resolved, so the check above has power against independent intensities
    assert zscore(products) > 5 * Z_MAX


def test_device_and_host_samplers_agree(draws):
    rng = np.random.default_rng(20261107)
    host_N = draws.law.draw(rng, len(draws.N), beam_primary=True)
    host_n = draw_multiplicities(rng, host_N, RATES)
    assert stats.ks_2samp(draws.N, host_N).pvalue > P_MIN
    for column in range(4):
        top = int(max(draws.n[:, column].max(), host_n[:, column].max())) + 1
        table = np.array([np.bincount(draws.n[:, column], minlength=top), np.bincount(host_n[:, column], minlength=top)])
        keep = table.sum(axis=0) >= 10
        assert stats.chi2_contingency(table[:, keep])[1] > P_MIN
