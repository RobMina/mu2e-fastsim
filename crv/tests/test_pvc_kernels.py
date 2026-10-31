"""Opt-in PVC kernels retain Hessian decisions and exact component roots."""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
from crvfastsim.reco import pulsefit_jax as P, coinc_jax as C


def test_jacobi_eigenvalues_psd_and_condition_boundary():
    rng = np.random.default_rng(61)
    q, _ = np.linalg.qr(rng.normal(size=(1000, 3, 3)))
    eig = np.column_stack((np.geomspace(1e-14, 1e-10, 1000), np.full(1000, .1), np.ones(1000)))
    m = (q * eig[:, None, :]) @ q.swapaxes(-1, -2)
    out = np.asarray(jax.jit(P._eigvalsh3)(m))
    ref = np.linalg.eigvalsh(m)
    np.testing.assert_allclose(out, ref, atol=3e-15, rtol=1e-12)
    assert np.array_equal(out[:, 0] > 1e-12 * out[:, -1], ref[:, 0] > 1e-12 * ref[:, -1])


def test_jacobi_degenerate_zero_and_scaled_matrices():
    matrices = np.stack([np.zeros((3, 3)), np.eye(3), np.ones((3, 3)),
                         np.diag([1e-200, 2e-200, 3e-200]), np.diag([1e200, 2e200, 3e200])])
    got = np.asarray(jax.jit(P._eigvalsh3)(matrices))
    np.testing.assert_allclose(got, np.linalg.eigvalsh(matrices), atol=2e-15, rtol=1e-14)


def test_pvc_batch_cap_only_on_opt_in_oneapi(monkeypatch):
    from crvfastsim.reco import chain_jax as CH, pulsefit as PF
    count = 4100
    W = dict(adc=np.zeros((count, 12), np.int64), n=np.full(count, 12),
             start=np.zeros(count, np.int64), ped=np.zeros(count), area=np.ones(count), height=np.ones(count))
    seen = []

    def fake(adc, *args, **kwargs):
        seen.append(adc.shape[0])
        return {k: np.zeros((adc.shape[0], 3), bool) if k == 'found' else np.zeros(adc.shape[0], bool)
                for k in CH.PULSE_KEYS if k != 'input_truncated'}

    monkeypatch.setattr(P, 'reconstruct_dense', fake)
    for opt, oneapi, expected in (('0', True, [8192]), ('1', False, [8192]), ('1', True, [4096, 4096])):
        seen.clear()
        monkeypatch.setenv('CRVFASTSIM_PVC_RECO', opt)
        monkeypatch.setattr(C, 'is_oneapi', lambda: oneapi)
        pending = CH._pulse_dispatch(W, np.arange(count), PF.RecoConfig(), 8192, 12)
        assert seen == expected
        assert sum(n for _, n in pending) == count


def test_label_propagation_equals_closure(monkeypatch):
    rng = np.random.default_rng(3)
    for h in (8, 16, 32, 64):
        a = rng.random((8, h, h)) < .04
        a |= a.swapaxes(-1, -2) | np.eye(h, dtype=bool)
        valid = np.ones((8, h), bool)
        monkeypatch.setenv('CRVFASTSIM_PVC_RECO', '0')
        ref = np.asarray(C._labels(a, valid))
        monkeypatch.setenv('CRVFASTSIM_PVC_RECO', '1')
        out = np.asarray(C._labels(a, valid))
        np.testing.assert_array_equal(out, ref)


def test_label_propagation_worst_case_chain_and_invalid_rows(monkeypatch):
    monkeypatch.setenv('CRVFASTSIM_PVC_RECO', '1')
    h = 64
    a = np.eye(h, dtype=bool) | np.eye(h, k=1, dtype=bool) | np.eye(h, k=-1, dtype=bool)
    valid = np.ones((2, h), bool)
    valid[1] = False
    adj = np.broadcast_to(a, (2, h, h)).copy()
    adj[1] = False
    out = np.asarray(C._labels(adj, valid))
    np.testing.assert_array_equal(out[0], np.zeros(h, int))
    np.testing.assert_array_equal(out[1], np.full(h, -1))
