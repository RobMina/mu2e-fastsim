"""Diagnostic copy of the Gumbel LM arithmetic with fail-closed batch solver errors."""
import numpy as np

from ..reco.pulsefit import _to_ext, _to_int


def fit_gumbel(t, y, mask, p0, lo, hi, iterations=40):
    """Same numerical equations; a numerical failure aborts instead of altering companion rows."""
    if iterations != 40:
        raise ValueError("diagnostic fit requires 40 LM iterations")
    if (any(not np.isfinite(a).all() for a in (t, y, p0, lo, hi))
            or np.any(hi <= lo) or np.any(p0 < lo) or np.any(p0 > hi)):
        raise ValueError("invalid diagnostic fit inputs/bounds")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = _fit(t, y, mask, p0, lo, hi, iterations)
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError("diagnostic fit numerical failure; batch incomplete") from exc
    if any(not np.isfinite(a).all() for a in result):
        raise ValueError("nonfinite diagnostic fit result; batch incomplete")
    return result


def _solve(a, b):
    value = np.linalg.solve(a, b)
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite diagnostic solve")
    return value


def _fit(t, y, mask, p0, lo, hi, iterations):
    size = t.shape[0]
    u = _to_int(p0, lo, hi)
    lam = np.full(size, 1e-3)
    w = mask.astype(float)

    def resid_jac(u):
        p = _to_ext(u, lo, hi)
        A, mu, b = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        z = (t - mu) / b
        ez = np.exp(-z)
        g = np.exp(-z - ez)
        f = A * g
        r = (y - f) * w
        dfdA = g
        dfdz = f * (ez - 1.0)
        dfdmu = dfdz * (-1.0 / b)
        dfdb = dfdz * (-z / b)
        dpdu = (hi - lo) * 0.5 * np.cos(u)
        J = np.stack([dfdA * dpdu[:, 0:1], dfdmu * dpdu[:, 1:2], dfdb * dpdu[:, 2:3]], axis=-1) * w[..., None]
        return r, J

    r, J = resid_jac(u)
    chi2 = (r * r).sum(1)
    for _ in range(iterations):
        JTJ = np.einsum("kpi,kpj->kij", J, J)
        JTr = np.einsum("kpi,kp->ki", J, r)
        D = np.einsum("kii->ki", JTJ)
        Aug = JTJ + (lam[:, None, None] * (D[:, :, None] * np.eye(3) + 1e-12 * np.eye(3)))
        du = _solve(Aug, JTr[..., None])[..., 0]
        un = u + du
        rn, Jn = resid_jac(un)
        chi2n = (rn * rn).sum(1)
        better = chi2n < chi2
        u = np.where(better[:, None], un, u)
        r = np.where(better[:, None], rn, r)
        J = np.where(better[:, None, None], Jn, J)
        chi2 = np.where(better, chi2n, chi2)
        lam = np.clip(np.where(better, lam * 0.3, lam * 10.0), 1e-9, 1e9)
    for _ in range(8):
        JTJ = np.einsum("kpi,kpj->kij", J, J)
        JTr = np.einsum("kpi,kp->ki", J, r)
        du = _solve(JTJ + 1e-12 * np.eye(3), JTr[..., None])[..., 0]
        for scale in (1.0, 0.5, 0.25, 0.125):
            un = u + scale * du
            rn, Jn = resid_jac(un)
            chi2n = (rn * rn).sum(1)
            better = chi2n < chi2
            u = np.where(better[:, None], un, u)
            r = np.where(better[:, None], rn, r)
            J = np.where(better[:, None, None], Jn, J)
            chi2 = np.where(better, chi2n, chi2)
            du = np.where(better[:, None], 0.0, du)
    JTJ = np.einsum("kpi,kpj->kij", J, J)
    JTr = np.einsum("kpi,kp->ki", J, r)
    step = _solve(JTJ + 1e-12 * np.eye(3), JTr[..., None])[..., 0]
    edm = 0.5 * np.einsum("ki,ki->k", JTr, step)
    eig = np.linalg.eigvalsh(JTJ)
    hess_ok = eig[:, 0] > 1e-12 * np.maximum(eig[:, -1], 1e-300)
    converged = np.isfinite(edm) & (edm <= 1e-3 * np.maximum(chi2, 1.0))
    return _to_ext(u, lo, hi), chi2, converged, hess_ok
