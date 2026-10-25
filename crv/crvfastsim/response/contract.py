"""Supported response settings and lossless-output requirements for production chains."""
import numpy as np


def validate_sipm(cfg):
    """Dark noise and afterpulsing require a complete chain implementation before use."""
    unsupported = [k for k in ('thermal_rate', 'trap0_prob', 'trap1_prob') if getattr(cfg, k) != 0]
    if unsupported:
        raise ValueError(f'CRV response: unsupported SiPM processes: {unsupported}')


def validate_window(window, fused=False):
    """Reject invalid windows and requests beyond the fused photon-time support."""
    w = np.asarray(window, dtype=float)
    if w.shape != (2,) or not np.isfinite(w).all() or w[1] <= w[0]:
        raise ValueError('CRV response: window must contain finite, increasing bounds')
    if fused and w[1] > 1e5:
        raise ValueError('CRV fused response: window upper bound exceeds 100000 ns photon support')


def require_complete(result):
    """Never consume residual photon or SiPM overflow as a physical charge sample."""
    flags = ('overflow', 'overflow_photons', 'overflow_sipm')
    failed = [k for k in flags if bool(np.asarray(result.get(k, False)).any())]
    if failed:
        raise RuntimeError(f'CRV response: residual overflow ({", ".join(failed)}); refusing truncated output')
