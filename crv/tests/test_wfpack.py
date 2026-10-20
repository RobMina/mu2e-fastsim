"""Device waveform packer (response/wfpack.py) equals reco.waveform_jax.pack on the same charges."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_device_pack_matches_host_pack():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    try:
        from crvfastsim.reco import waveform_jax as WJ
        from crvfastsim.reco.waveform import WaveformConfig
        cfg = WaveformConfig()
        tmpl = WJ.pack(np.array([1]), np.array([0.0]), np.array([1e-13]), np.zeros(2), cfg)["tmpl"]
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"single-PE template not available: {e}")
    from crvfastsim.response import wfpack
    rng = np.random.default_rng(1)
    lab = np.repeat(np.arange(40), rng.integers(1, 30, 40))
    t = np.sort(rng.uniform(0, 400, len(lab))) + lab * 1000.0
    o = np.lexsort((t, lab)); lab, t = lab[o], t[o]
    q = rng.uniform(0.5, 1.0, len(lab)) * 2.652e-13
    tdc0 = rng.uniform(0, 12.5, 40)
    H = WJ.pack(lab, t, q, tdc0, cfg)
    D = wfpack.pack(jnp.asarray(lab), jnp.asarray(t), jnp.asarray(q), jnp.ones(len(lab), bool), jnp.asarray(tdc0[lab]),
                    float(tmpl.max()), cfg.single_pe_reference_charge, cfg.single_pe_max_time, cfg.min_voltage,
                    C=128, Q=64)
    C = len(H["n"])
    assert not bool(D["overflow"]) and int(D["nclus"]) == C
    for k in ("n", "t_start", "keep", "chan", "est"):
        assert np.allclose(np.asarray(D[k])[:C], np.asarray(H[k])), k
    Qh = H["T"].shape[1]
    assert np.allclose(np.asarray(D["T"])[:C, :Qh], H["T"]) and np.allclose(np.asarray(D["W"])[:C, :Qh], H["W"])
