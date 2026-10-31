"""CPU normal-equation cache retaining every fit iteration and arithmetic order."""
import numpy as np


def fit_cached(u, r, J, chi2, lo, hi, iterations, residual):
    """Reuse normal equations only while their input residual and Jacobian are unchanged."""
    from .pulsefit import _to_ext
    lam = np.full(len(u), 1e-3)
    eye = np.eye(3)
    JTJ = np.einsum("kpi,kpj->kij", J, J)
    JTr = np.einsum("kpi,kp->ki", J, r)
    for _ in range(iterations):
        D = np.einsum("kii->ki", JTJ)
        Aug = JTJ + (lam[:, None, None] * (D[:, :, None] * eye + 1e-12 * eye))
        try:
            du = np.linalg.solve(Aug, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            du = np.zeros_like(u)
        un = u + du
        rn, Jn = residual(un)
        chi2n = (rn * rn).sum(1)
        better = chi2n < chi2
        u = np.where(better[:, None], un, u)
        r = np.where(better[:, None], rn, r)
        J = np.where(better[:, None, None], Jn, J)
        chi2 = np.where(better, chi2n, chi2)
        lam = np.clip(np.where(better, lam * 0.3, lam * 10.0), 1e-9, 1e9)
        JTJ[better] = np.einsum("kpi,kpj->kij", J[better], J[better])
        JTr[better] = np.einsum("kpi,kp->ki", J[better], r[better])
    for _ in range(8):
        try:
            du = np.linalg.solve(JTJ + 1e-12 * eye, JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        du = np.where(np.isfinite(du), du, 0.0)
        for scale in (1.0, 0.5, 0.25, 0.125):
            un = u + scale * du
            rn, Jn = residual(un)
            chi2n = (rn * rn).sum(1)
            better = chi2n < chi2
            u = np.where(better[:, None], un, u)
            r = np.where(better[:, None], rn, r)
            J = np.where(better[:, None, None], Jn, J)
            chi2 = np.where(better, chi2n, chi2)
            du = np.where(better[:, None], 0.0, du)
            JTJ[better] = np.einsum("kpi,kpj->kij", J[better], J[better])
            JTr[better] = np.einsum("kpi,kp->ki", J[better], r[better])
    try:
        step = np.linalg.solve(JTJ + 1e-12 * eye, JTr[..., None])[..., 0]
        edm = 0.5 * np.einsum("ki,ki->k", JTr, step)
        eig = np.linalg.eigvalsh(JTJ)
        hess_ok = eig[:, 0] > 1e-12 * np.maximum(eig[:, -1], 1e-300)
    except np.linalg.LinAlgError:
        edm = np.full(len(u), np.inf)
        hess_ok = np.zeros(len(u), bool)
    converged = np.isfinite(edm) & (edm <= 1e-3 * np.maximum(chi2, 1.0))
    return _to_ext(u, lo, hi), chi2, converged, hess_ok
