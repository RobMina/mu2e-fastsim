"""Stock bunch-intensity and conditional-Poisson laws, with explicit donor ordering."""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from scipy.special import ndtr, ndtri


STREAMS = ("MuStopPileup", "EleBeamFlash", "MuBeamFlash", "NeutralsFlash")
CHAINS = {
    "MuStopPileup": ("MuBeamCat", "MuminusStopsCat", "MuStopPileupCat"),
    "EleBeamFlash": ("EleBeamCat", "EleBeamFlashCat"),
    "MuBeamFlash": ("MuBeamCat", "MuBeamFlashCat"),
    "NeutralsFlash": ("NeutralsCat", "NeutralsFlashCat"),
}


def _nonnegative(value, name):
    value = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _integers(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in "iu" or np.any(value < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    if np.any(value > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64")
    return value.astype(np.int64, copy=False)


@dataclass(frozen=True)
class BunchLaw:
    """Uncut mean and log sigma, truncated before nearest-integer POT rounding."""

    extended_mean: float
    sigma: float
    cut_max: float
    cut_min: float = 0.0

    def __post_init__(self):
        values = _nonnegative(
            [self.extended_mean, self.sigma, self.cut_min, self.cut_max], "bunch law"
        )
        if values[0] == 0 or values[1] == 0 or self.cut_max <= self.cut_min:
            raise ValueError("positive mean/sigma and cut_max > cut_min required")
        if self.cut_max >= np.iinfo(np.int64).max:
            raise ValueError("cut_max exceeds supported POT integer range")

    @classmethod
    def from_sdf(cls, extended_mean, sdf, cut_max, cut_min=0.0):
        if not 0 < sdf < 1:
            raise ValueError("SDF must be between zero and one")
        return cls(extended_mean, np.sqrt(-np.log(sdf)), cut_max, cut_min)

    def draw(self, rng, size, *, beam_primary):
        """Sample x*f(x) for beam primaries, f(x) otherwise; not stock RNG bytes."""
        size = int(_integers(size, "size"))
        beam = np.broadcast_to(np.asarray(beam_primary, dtype=bool), (size,))
        mu = np.log(self.extended_mean) - 0.5 * self.sigma**2 + beam * self.sigma**2
        lower = ndtr((np.log(self.cut_min) - mu) / self.sigma) if self.cut_min else 0.0
        upper = ndtr((np.log(self.cut_max) - mu) / self.sigma)
        if np.any(upper - lower < 1e-6):
            raise ValueError("requested truncation has less than 1e-6 probability")
        probability = lower + (upper - lower) * rng.random(size)
        probability = np.clip(probability, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        values = np.exp(mu + self.sigma * ndtri(probability))
        values = np.clip(values, self.cut_min, np.nextafter(self.cut_max, self.cut_min))
        return np.rint(values).astype(np.int64)


def stock_rates(efficiencies: Mapping, *, stop_prescale, early_prescales=None,
                early_reduction=0.01):
    """Multiply SimBookkeeper efficiencies and epilog factors, never catalog counts.

    Supply the values from the campaign's conditions. Early samples require all
    three flash prescales; their reduction also multiplies the stop stream.
    """
    stop_prescale = float(_nonnegative(stop_prescale, "stop_prescale"))
    reduction = float(_nonnegative(early_reduction, "early_reduction"))
    if early_prescales is not None and set(early_prescales) != set(STREAMS[1:]):
        raise ValueError("early_prescales must name all three flash streams")
    rates = {}
    for stream, tags in CHAINS.items():
        if early_prescales is not None and stream != "MuStopPileup":
            tags = (*tags[:-1], "Early" + tags[-1])
        factors = [efficiencies[tag] for tag in tags]
        factors += [stop_prescale] if stream == "MuStopPileup" else []
        if early_prescales is not None:
            factors += [reduction]
            if stream != "MuStopPileup":
                factors += [early_prescales[stream]]
        rates[stream] = float(np.prod(_nonnegative(factors, f"{stream} factors")))
        if not np.isfinite(rates[stream]):
            raise ValueError(f"{stream} efficiency product overflow")
    return rates


def draw_multiplicities(rng, intensity, rates):
    """Independent Poisson draws conditional on the SAME primary-event intensity."""
    intensity = _nonnegative(intensity, "intensity")
    if intensity.ndim != 1:
        raise ValueError("intensity must be a vector")
    if set(rates) != set(STREAMS):
        raise ValueError("rates must name the four stock streams")
    means = intensity[:, None] * _nonnegative([rates[s] for s in STREAMS], "rates")
    return rng.poisson(means)


@dataclass(frozen=True)
class FrameSelection:
    """One row per donor occurrence, including donors whose detector arrays are empty."""

    counts: np.ndarray
    donors: np.ndarray

    def __post_init__(self):
        counts = _integers(self.counts, "counts")
        donors = _integers(self.donors, "donors")
        if counts.ndim != 1 or donors.ndim != 1 or counts.sum() != donors.size:
            raise ValueError("one-dimensional counts must sum to the donor count")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "donors", donors)

    @property
    def event_index(self):
        return np.repeat(np.arange(self.counts.size), self.counts)


def select_frames(rng, counts, pool_size, *, mode="sequential", cursor=0, wrap=True,
                  max_events_to_skip=0):
    """Return (selection, next_cursor); cursor=None draws the first inclusive skip.

    Sequential is the stock production default. Pass next_cursor into the next
    batch, without another skip draw. randomReplace is explicit, not the default.
    """
    counts = _integers(counts, "counts")
    if counts.ndim != 1:
        raise ValueError("counts must be a vector")
    pool_size = int(_integers(pool_size, "pool_size"))
    skip = int(_integers(max_events_to_skip, "max_events_to_skip"))
    total = int(counts.sum())
    if mode not in ("sequential", "randomReplace"):
        raise ValueError("supported modes: sequential, randomReplace")
    if pool_size == 0 and total:
        raise ValueError("cannot draw from an empty frame pool")
    if cursor is None:
        cursor = int(rng.integers(skip + 1)) if mode == "sequential" else 0
    cursor = int(_integers(cursor, "cursor"))
    if mode == "randomReplace":
        donors = rng.integers(pool_size, size=total) if total else np.empty(0, dtype=np.int64)
        return FrameSelection(counts, donors), cursor
    if not wrap and total and cursor + total > pool_size:
        raise ValueError("sequential frame pool exhausted with wrap=False")
    donors = np.arange(cursor, cursor + total, dtype=np.int64)
    if total and wrap:
        donors %= pool_size
    next_cursor = (cursor + total) % pool_size if wrap and pool_size else cursor + total
    return FrameSelection(counts, donors), next_cursor
