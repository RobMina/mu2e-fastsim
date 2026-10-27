"""Opt-in physical SiPM presence, independent of calibration status and optical reflectors."""
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json

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
    """Return unmutated conditions with a physical mask and JSON-safe execution-policy evidence."""
    mask = sensor_mask(geom, counters_cfg)
    masked = replace(cond, readout_mask=mask)
    effective_status(masked)
    digest = hashlib.sha256(json.dumps(mask.tolist(), separators=(",", ":")).encode()).hexdigest()
    return masked, {"policy": "opt_in_counters_sipmsAtSide", "present": int(mask.sum()),
                    "absent": int((~mask).sum()), "mask_sha256": digest}


def validate_merge(records):
    """Require identical explicit readout policy on every shard; legacy all-unmarked stays unmarked."""
    if not records or not any("geometry_readout" in r for r in records):
        return {}
    expected = None
    for record in records:
        evidence = record.get("geometry_readout")
        if not isinstance(evidence, Mapping) or set(evidence) != {"policy", "present", "absent", "mask_sha256"}:
            raise ValueError("runchain merge: missing or malformed geometry_readout policy")
        digest = evidence["mask_sha256"]
        counts = (evidence["present"], evidence["absent"])
        if (evidence["policy"] != "opt_in_counters_sipmsAtSide"
                or any(type(n) is not int or n < 0 for n in counts) or sum(counts) % 4
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("runchain merge: malformed geometry_readout policy")
        if expected is not None and evidence != expected:
            raise ValueError("runchain merge: incompatible geometry_readout policy")
        expected = dict(evidence)
    return {"geometry_readout": expected}


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
