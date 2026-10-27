"""Versioned PBI laws; the historical floating unconditional draw remains the default."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

LEGACY = "legacy_unconditional_v1"
CE_CONDITIONED = "ce_conditioned_v1"
MODES = (LEGACY, CE_CONDITIONED)
PBI = dict(extended_mean=1.58e7, sdf=0.8, cut_max=9.48e7)


def require_legacy_response(manifest):
    """Keep legacy response/checking paths closed to unbound new exposure schemas."""
    if manifest.get("schema") != "crv-mixed-replay-v1" or "exposure_contract" in manifest:
        raise ValueError("legacy response/checker requires replay-v1; CE replay-v2 exposure binding is not implemented here")


def llrint_pbi(raw):
    """Match stock llrint under FE_TONEAREST (nearest integer, ties to even)."""
    raw = np.asarray(raw, dtype=np.float64)
    rounded = np.rint(raw)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0) or np.any(rounded >= float(2**63)):
        raise ValueError("PBI must be finite, nonnegative and round within signed int64")
    return rounded.astype(np.int64)


@dataclass(frozen=True)
class ExposureContract:
    mode: str = LEGACY
    extended_mean: float = PBI["extended_mean"]
    sdf: float = PBI["sdf"]
    cut_max: float = PBI["cut_max"]
    intensity_scale: float = 1.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown exposure mode: {self.mode}")
        if not all(math.isfinite(v) for v in (self.extended_mean, self.sdf, self.cut_max, self.intensity_scale)):
            raise ValueError("PBI parameters must be finite")
        if self.extended_mean <= 0 or not 0 < self.sdf <= 1 or self.cut_max <= 0 or self.intensity_scale <= 0:
            raise ValueError("require extended_mean, cut_max, intensity_scale > 0 and 0 < sdf <= 1")
        if not math.isfinite(self.cut_max * self.intensity_scale):
            raise ValueError("scaled PBI cutoff must be finite")
        if self.mode == CE_CONDITIONED and self.cut_max * self.intensity_scale >= float(2**63):
            raise ValueError("scaled PBI cutoff must fit signed int64")
        if self.sigma == 0:
            if self.extended_mean >= self.cut_max or math.exp(self.logmean) >= self.cut_max:
                raise ValueError("degenerate PBI has no support below cut_max")
        elif 0.5 * math.erfc((self.logmean - math.log(self.cut_max)) / (self.sigma * math.sqrt(2))) < 1e-6:
            raise ValueError("PBI rejection acceptance below supported minimum 1e-6")

    @property
    def sigma(self):
        return math.sqrt(max(0.0, -math.log(self.sdf)))

    @property
    def logmean(self):
        s = self.sigma
        mu = math.log(self.extended_mean) - 0.5 * s * s
        return mu + s * s if self.mode == CE_CONDITIONED else mu

    def draw_raw(self, rng, n):
        """Reject above the strict upper cutoff with the historical batch/RNG ordering."""
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("sample size must be a nonnegative integer")
        out = np.empty(0)
        batches = 0
        while len(out) < n:
            x = rng.lognormal(self.logmean, self.sigma, 2 * (n - len(out)) + 16)
            out = np.r_[out, x[x < self.cut_max]]
            batches += 1
            if batches >= 10000 and len(out) < n:
                raise RuntimeError("PBI rejection sampler exceeded 10000 batches")
        return out[:n] * self.intensity_scale

    def record(self):
        """JSON-safe mathematical contract, separate from library/response certification."""
        ce = self.mode == CE_CONDITIONED
        return {
            "schema": "crv-pbi-exposure-v1", "mode": self.mode,
            "parameters": {"extended_mean": self.extended_mean, "sdf": self.sdf,
                           "cut_max": self.cut_max, "intensity_scale": self.intensity_scale},
            "sigma": self.sigma, "logmean": self.logmean,
            "base_law": "lognormal(mu=ln(extended_mean)-sigma^2/2, sigma=sqrt(-ln(sdf)))",
            "density": "x*f_LN(x) / integral_0^cut_max(t*f_LN(t) dt)" if ce else "f_LN(x) / integral_0^cut_max(f_LN(t) dt)",
            "support": "0 < x < cut_max before scaling; sdf=1 is a point mass",
            "sampling": "normal logmean shifted by sigma^2" if ce else "unshifted logmean; historical rejection batches",
            "scale_order": "truncate, multiply by intensity_scale, then round if CE-conditioned",
            "rounding": "llrint FE_TONEAREST: nearest integer, ties to even (numpy.rint), int64" if ce else "none; floating PBI used directly",
            "poisson_mean": "intensity * product(SimEfficiencies) * product(meanEventsPerPOTFactors)",
            "target": "beam-based mu2eCeMinusLeadingLog primary" if ce else "historical unconditional diagnostic; not CE-conditioned",
            "classification_basis": "Offline v13_39_00/v13_41_00 ProcessCode and retained production configuration",
            "rng": "numpy Generator; no bitwise equivalence to Offline RNG claimed",
            "sampler_limits": {"minimum_acceptance": 1e-6, "maximum_batches": 10000},
        }
