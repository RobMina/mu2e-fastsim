"""Physical SiPM presence (Offline CrvSiPMChargeGenerator: hasCMB(SiPM % 2)), independent of calibration status and
optical reflectors. The runchain and fasttables loaders apply it to every configuration."""
from collections.abc import Mapping
from dataclasses import replace

import numpy as np


def sensor_mask(geom, counters_cfg):
    """Dense bool mask: channel 4 * bar + sipm is present iff its sector has side sipm % 2."""
    names = getattr(geom, "sector_names", None)
    if (not isinstance(names, (list, tuple, np.ndarray))
            or (isinstance(names, np.ndarray) and names.ndim != 1) or len(names) == 0):
        raise ValueError("readout geometry requires nonempty sector_names")
    names = list(names)
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("readout geometry sector_names must be unique nonempty strings")
    sector = np.asarray(getattr(geom, "sector", None))
    if sector.ndim != 1 or sector.dtype.kind not in "iu":
        raise ValueError("readout geometry sector must be a one-dimensional integer array")
    nbars = getattr(geom, "nbars", None)
    if (not isinstance(nbars, (int, np.integer)) or isinstance(nbars, (bool, np.bool_))
            or nbars < 0 or sector.shape != (nbars,)):
        raise ValueError("readout geometry sector size must equal nbars")
    if np.any(sector < 0) or np.any(sector >= len(names)):
        raise ValueError("readout geometry contains an unknown sector index")
    if not isinstance(counters_cfg, Mapping):
        raise ValueError("readout counters_cfg must be a mapping")
    if "crs.sectorNames" in counters_cfg:
        configured = counters_cfg["crs.sectorNames"]
        if (not isinstance(configured, (list, tuple, np.ndarray))
                or (isinstance(configured, np.ndarray) and configured.ndim != 1) or list(configured) != names):
            raise ValueError("readout geometry sector_names disagree with crs.sectorNames")
    sides = np.empty((len(names), 2), dtype=bool)
    for i, name in enumerate(names):
        for side in (0, 1):
            key = f"crs.sipmsAtSide{side}{name}"
            if key not in counters_cfg:
                raise ValueError(f"readout counters missing {key}")
            value = counters_cfg[key]
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"readout counters {key} must be boolean, got {value!r}")
            sides[i, side] = value
    return sides[sector][:, [0, 1, 0, 1]].reshape(4 * nbars)


def apply(cond, geom, counters_cfg):
    """Return unmutated conditions with the physical mask; channels beyond the bars stay present."""
    mask = sensor_mask(geom, counters_cfg)
    n = len(np.asarray(cond.status))
    if n < len(mask):
        raise ValueError(f"readout: conditions cover {n} channels, the geometry has {len(mask)}")
    masked = replace(cond, readout_mask=np.concatenate([mask, np.ones(n - len(mask), bool)]))
    effective_status(masked)
    return masked


def effective_status(cond):
    """Return original status when disabled; otherwise copy and OR bit 0 for absent sensors."""
    mask = cond.readout_mask
    if mask is None:
        return cond.status
    status = np.asarray(cond.status)
    if status.ndim != 1 or status.dtype.kind not in "iu":
        raise ValueError("readout status must be a one-dimensional integer array")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(bool):
        raise ValueError("readout_mask must be a boolean numpy array")
    if mask.ndim != 1 or mask.shape != status.shape:
        raise ValueError("readout_mask must be one-dimensional with exactly one entry per status channel")
    effective = status.copy()
    effective[~mask] |= 1
    return effective
