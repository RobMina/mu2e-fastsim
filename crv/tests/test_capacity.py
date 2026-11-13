"""Fixed device shapes (crvfastsim.capacity): the capacity-mode draws do not depend on the buffer sizes, the fixed
waveform synthesis gives the same digis for any capacity file, and the capacity file is derived and enforced."""
import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


def _sipm_args(n_photons, NB=None, seed=3):
    from crvfastsim.response import sipm, sipm_jax
    cfg = sipm.SiPMConfig()
    rng = np.random.default_rng(seed)
    pmap = np.zeros((40, 40))
    pmap[10:30, 10:30] = 1.0
    n_per = rng.integers(1, 60, n_photons // 30 + 1)
    slot = np.repeat(np.arange(len(n_per)), n_per)[:n_photons]
    t = rng.uniform(0, 300, len(slot))
    n = len(slot)
    NB, Ms, Ntot, q_ct = sipm_jax.device_buckets(n, cfg, NB=NB)
    dead = np.zeros(sipm_jax.NP2, bool)
    args = (jnp.asarray(np.pad(slot, (0, NB - n))), jnp.asarray(np.pad(t, (0, NB - n))), jnp.arange(NB) < n,
            jax.random.PRNGKey(11), jnp.asarray(np.cumsum(pmap.ravel()) / pmap.sum()), pmap.shape[1],
            jnp.asarray(dead), q_ct, cfg.overvoltage, cfg.time_constant, cfg.p1, cfg.p2, cfg.capacitance,
            NB, *Ms, Ntot)
    return sipm_jax, cfg, args, n


def _fired(r):
    f = np.asarray(r["fired"])
    return {k: np.asarray(r[k])[f] for k in ("slot", "time", "charge", "pe", "root")}


@pytest.mark.parametrize("truncate", [False, True])
def test_section_draws_do_not_depend_on_buffer_sizes(truncate):
    sj, cfg, a1, n = _sipm_args(3000)
    _, _, a2, _ = _sipm_args(3000, NB=3 * a1[13] + 17)          # larger photon and candidate buffers
    kw = dict(truncate=truncate, draws="section")
    r1 = sj.device_sipm(*a1, sj.STREAM_L, sj.LONG_SL, sj.LONG_LL, cfg.n_pixels_x, cfg.n_pixels_y, **kw)
    r2 = sj.device_sipm(*a2, sj.STREAM_L, sj.LONG_SL, sj.LONG_LL, cfg.n_pixels_x, cfg.n_pixels_y, **kw)
    assert not bool(r1["overflow"]) and not bool(r2["overflow"])
    f1, f2 = _fired(r1), _fired(r2)
    assert len(f1["time"]) > n // 5
    assert (f1["root"] < n).all() and (f2["root"] < n).all()
    for k in f1:
        assert np.array_equal(f1[k], f2[k]), k


def test_offset_draws_are_the_default_and_differ_from_section():
    sj, cfg, a1, _ = _sipm_args(3000)
    base = sj.device_sipm(*a1, sj.STREAM_L, sj.LONG_SL, sj.LONG_LL, cfg.n_pixels_x, cfg.n_pixels_y)
    off = sj.device_sipm(*a1, sj.STREAM_L, sj.LONG_SL, sj.LONG_LL, cfg.n_pixels_x, cfg.n_pixels_y, draws="offset")
    sec = sj.device_sipm(*a1, sj.STREAM_L, sj.LONG_SL, sj.LONG_LL, cfg.n_pixels_x, cfg.n_pixels_y, draws="section")
    for k in ("fired", "charge"):
        assert np.array_equal(np.asarray(base[k]), np.asarray(off[k])), k
    assert not np.array_equal(np.asarray(base["fired"]), np.asarray(sec["fired"]))
    with pytest.raises(ValueError, match="draws"):
        sj.device_sipm(*a1, sj.STREAM_L, sj.LONG_SL, sj.LONG_LL, cfg.n_pixels_x, cfg.n_pixels_y, draws="other")


def test_device_buckets_fixed_photons():
    from crvfastsim.response import sipm, sipm_jax
    cfg = sipm.SiPMConfig()
    NB, Ms, Ntot, _ = sipm_jax.device_buckets(1000, cfg, NB=5000)
    assert NB == 5000 and Ntot >= NB + sum(Ms)
    assert sipm_jax.device_buckets(1000, cfg)[0] == 4096
    with pytest.raises(ValueError, match="capacity"):
        sipm_jax.device_buckets(6000, cfg, NB=5000)


def test_per_cluster_noise_keys_are_prefix_stable():
    from crvfastsim.reco import waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    tmpl = jnp.asarray(np.r_[np.linspace(0, 1, 50), np.linspace(1, 0, 150)])
    rng = np.random.default_rng(2)
    C, Q = 5, 4
    T = rng.uniform(0, 40, (C, Q))
    W = rng.uniform(0.5, 2, (C, Q))
    n = np.full(C, Q)
    ts = np.zeros(C)
    keep = np.ones(C, bool)
    keys = jax.vmap(jax.random.fold_in, (None, 0))(jax.random.PRNGKey(5), jnp.arange(C, dtype=jnp.uint32))

    def run(Cp, L):
        pad = lambda x, v: np.concatenate([x, np.full((Cp - C,) + x.shape[1:], v, x.dtype)])
        k = jnp.concatenate([keys, jnp.zeros((Cp - C, 2), keys.dtype)])
        wave, adc, st, last = WJ.synth_dense(pad(T, 0.0), pad(W, 0.0), pad(n, 0), pad(ts, 0.0), pad(keep, False),
                                             tmpl, k, L, cfg.samples_zs, 0.2, 0.5, cfg.min_voltage,
                                             cfg.adc_conversion, cfg.pedestal, cfg.min_adc, cfg.max_adc)
        return np.asarray(wave)[:C], np.asarray(last)[:C]

    w1, l1 = run(C, 40)
    w2, l2 = run(3 * C, 64)
    assert np.array_equal(l1, l2)
    assert np.array_equal(w1, w2[:, :40])


def _charges(n_events=300, seed=4):
    from crvfastsim.reco import repro
    row, t, q, tdc0 = repro._charges(n_events, seed)
    return {"row": row, "time": t, "charge": q}, tdc0


def test_fixed_synthesis_does_not_depend_on_capacities():
    from crvfastsim.capacity import FORMAT, Capacities
    from crvfastsim.reco import cli, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    Q, tdc0 = _charges()
    rec = Capacities(record=True)
    a = cli._synth_jax_fixed(WJ, Q, tdc0, cfg, rec, noise=cfg.noise, seed=3)
    use = rec.report()["usage"]
    qs = sorted({k.split("/")[1] for k in use if k.startswith("synth_samples/")}, key=int)
    tight = Capacities({"format": FORMAT, "synth_samples": {q: use[f"synth_samples/{q}"] for q in qs},
                        "synth_rows": {q: max(1, use[f"synth_rows/{q}"] // 3) for q in qs}})
    wide = Capacities({"format": FORMAT, "synth_samples": {q: use[f"synth_samples/{q}"] + 21 for q in qs},
                       "synth_rows": {q: 2 * use[f"synth_rows/{q}"] for q in qs}})
    short = Capacities({"format": FORMAT, "synth_samples": {q: max(1, use[f"synth_samples/{q}"] - 3) for q in qs}})
    b = cli._synth_jax_fixed(WJ, Q, tdc0, cfg, tight, noise=cfg.noise, seed=3, budget_gib=0.01)
    c = cli._synth_jax_fixed(WJ, Q, tdc0, cfg, wide, noise=cfg.noise, seed=3)
    d = cli._synth_jax_fixed(WJ, Q, tdc0, cfg, short, noise=cfg.noise, seed=3)
    assert len(a["adc"]) > 500
    for other in (b, c, d):
        for k in ("chan", "start", "starttdc", "adc", "volt"):
            assert np.array_equal(a[k], other[k]), k
    assert tight.report()["overflow"] and not wide.report()["overflow"]
    assert any(k.startswith("synth_samples/") for k in short.report()["overflow"])


def test_fixed_synthesis_without_noise_equals_bucketed():
    from crvfastsim.capacity import Capacities
    from crvfastsim.reco import cli, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    Q, tdc0 = _charges(200, 7)
    a = cli._synth_jax_bucketed(WJ, Q, tdc0, cfg)
    b = cli._synth_jax_fixed(WJ, Q, tdc0, cfg, Capacities(record=True))
    for k in ("chan", "start", "starttdc", "adc"):
        assert np.array_equal(a[k], b[k]), k
    assert np.max(np.abs(a["volt"] - b["volt"])) < 1e-12


def test_capacity_file_derive_and_enforce(tmp_path):
    from crvfastsim import capacity
    usage = {"steps": 1000, "photon_rows": 80, "refit_samples": 33, "synth_rows/64": 40, "synth_samples/64": 41,
             "sipm_photons": 999, "photon_substeps": 1 << 22}
    caps = capacity.derive(usage, 1.25, fixed={"sipm_photons": 4096})
    assert caps["steps"] == 1250 and caps["photon_rows"] == 100 and caps["refit_samples"] == 48
    assert caps["photon_substeps"] == 1 << 22                         # clamped to the kernel's chunk limit
    assert caps["synth_rows"] == {"64": 50} and caps["synth_samples"] == {"64": 56} and caps["sipm_photons"] == 4096
    path = tmp_path / "caps.json"
    path.write_text(json.dumps(caps))
    c = capacity.Capacities.load(path, record=True)
    assert c.size("photon_rows", 90, 128) == 100
    assert c.size("photon_rows", 101, 128) == 128                     # overflow: the bucketed shape, counted
    assert c.size("photon_arrivals", 7, 8) == 8                       # no capacity for this buffer
    assert c.cap("synth_samples", 64) == 56 and c.cap("synth_samples", 128) is None
    rep = c.report()
    assert rep["overflow"] == {"photon_rows": 1} and rep["usage"]["photon_rows"] == 101
    with pytest.raises(ValueError, match="unknown"):
        capacity.Capacities({"format": capacity.FORMAT, "photon_rowz": 3})
    with pytest.raises(ValueError, match="format"):
        capacity.Capacities({"format": "other"})
    with pytest.raises(ValueError, match="margin"):
        capacity.derive(usage, 0.5)


def test_slab_crossings_padding_is_exact():
    from mu2efastsim import crossings
    rng = np.random.default_rng(1)
    o = rng.normal(size=(1000, 3)) * 100
    d = rng.normal(size=(1000, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    lo = rng.normal(size=(7, 3)) * 50
    hi = lo + rng.uniform(1, 30, (7, 3))
    a = crossings.slab_crossings(o, d, lo, hi, backend="jax")
    b = crossings.slab_crossings(o, d, lo, hi, backend="numpy")
    assert a[0].shape == (1000, 7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_packaged_run1a_file_loads_by_name():
    from crvfastsim import capacity
    c = capacity.Capacities.load("run1a_v01")
    assert c.cap("sipm_photons") == 1 << 22 and c.cap("photon_substeps") == 1 << 22
    assert c.cap("synth_samples", 64) and c.cap("steps")


def test_packaged_run1a_coinc_hits_cover_every_tier():
    # every coincidence tier has a fixed hit buffer, never above the most hits a batch can hold (batch x H)
    from crvfastsim import capacity
    from crvfastsim.reco import coinc_jax as J
    c = capacity.Capacities.load("run1a_v01")
    for H, _, batch in J.TIERS:
        assert 0 < c.cap("coinc_hits", H) <= batch * H, H


def test_device_sorted_charges_equal_host_concatenation():
    from crvfastsim.capacity import Capacities
    from crvfastsim.response import resident
    rng = np.random.default_rng(5)
    pieces, host = [], []
    for width, n in ((500, 310), (500, 0), (500, 499), (500, 120)):
        chan = rng.integers(0, 40, width)
        time = np.round(rng.uniform(0, 50, width), 1)            # rounded: many equal (chan, time) pairs
        charge, pe, photon = rng.uniform(size=width), rng.uniform(size=width), rng.integers(0, 10**6, width)
        pieces.append(((jnp.asarray(chan), jnp.asarray(time), jnp.asarray(charge), jnp.asarray(pe),
                        jnp.asarray(photon)), n))
        host.append([x[:n] for x in (chan, time, charge, pe, photon)])
    ref = [np.concatenate([h[i] for h in host]) for i in range(5)]
    o = np.lexsort((ref[1], ref[0]))
    for caps in (Capacities(record=True), Capacities({"format": "crvfastsim-capacities-1", "charges": 2000})):
        out = resident._sorted_charges(pieces, caps)
        for i, k in enumerate(("chan", "time", "charge", "pe", "photon")):
            assert np.array_equal(out[k], ref[i][o]), k


def test_host_workers_take_over_futures_submitted_ahead():
    from crvfastsim import host_workers

    class Pool:
        def __init__(self):
            self.calls = []

        def submit(self, fn, *args):
            self.calls.append(args)
            return ("future", args)

    p = Pool()
    host_workers.submit_ahead(p, ["a.root", "b.root"], {"a.root": "GEN_A"}, 8, None)
    assert p.calls == [(0, "a.root", "GEN_A", 8, None), (1, "b.root", None, 8, None)]
    assert host_workers.submit(p, 0, "a.root", "GEN_A", 8, None) == ("future", (0, "a.root", "GEN_A", 8, None))
    assert len(p.calls) == 2                                          # taken over, not submitted again
    host_workers.submit(p, 1, "b.root", None, 9, None)                # another seed: a new task
    assert p.calls[-1] == (1, "b.root", None, 9, None)
    host_workers._AHEAD.clear()
