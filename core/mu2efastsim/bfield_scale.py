"""Per-map magnetic-field scale factors, as Offline's ``bfield.mapScale.<map>`` (Mu2e/Offline#2079).

Opt-in and off by default. One process-wide setting, the environment variable ``MU2EFS_BFIELD_MAPSCALE``, lists
``<map>=<factor>`` entries separated by ``:`` (or ``,``), e.g. ``DSMap=0.9:DSExtension=0.9:TSdMap=1.05``. The map
name is the G4BL file's basename without ``.header``, the same key suffix Offline uses. The two field-map readers
(``mu2efastsim.ehelix.read_g4bl`` and ``beamfastsim.fieldmap.load``) multiply a scaled map's grid by its factor as
they read it, so every chain built on them (beam transport, CE device chain, tracker field steps, the TS bank) sees
the scaled field; a whole-bank scale such as ``tsfield.FieldBank(scale=)`` multiplies on top, as Offline's
``bfield.scaleFactor`` does.

Unset or empty: no map is touched and the readers' code path is the unscaled one. An environment variable is used
(not a CLI flag) so worker processes inherit it. A factor must lie in [0.5, 1.5]; an unknown map name is an error.
Each scaled map is reported once per process on stderr: ``bfield_scale: map DSMap scaled by 0.9``.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict

ENV = "MU2EFS_BFIELD_MAPSCALE"
LIMITS = (0.5, 1.5)
# Maps of the Mu2e field configurations (Mau13 and later; bfgeom_*.txt inner and outer maps). A name outside this
# set is a typo, refused rather than silently ignored.
KNOWN = frozenset({"PSMap", "PSAreaMap", "PStoDumpAreaMap", "ProtonBeamDumpMap", "TSuMap", "TSdMap", "DSMap",
                   "DSExtension", "PSMap_fix", "TSuMap_fix", "TSdMap_fix", "DSMap_fix"})
# Task parameters of the beam kit tasks -> the maps they scale (as Mu2eBO's col5_bfield studies set them).
TASK_PARAMS = {"ds_bscale": ("DSMap", "DSExtension"), "tsd_bscale": ("TSdMap",)}

_parsed: Dict[str, Dict[str, float]] = {}
_reported: set = set()


def parse(text: str) -> Dict[str, float]:
    """{map: factor} of a setting string; ValueError on a bad entry, an unknown map or a factor out of range."""
    out: Dict[str, float] = {}
    for entry in filter(None, (e.strip() for e in re.split(r"[:,]", text or ""))):
        m = re.fullmatch(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)", entry)
        if not m:
            raise ValueError(f"{ENV}: bad entry {entry!r} (want <map>=<factor>)")
        name, value = m.group(1), float(m.group(2))
        if name not in KNOWN:
            raise ValueError(f"{ENV}: unknown field map {name!r}; known: {sorted(KNOWN)}")
        if not LIMITS[0] <= value <= LIMITS[1]:
            raise ValueError(f"{ENV}: {name} factor {value} outside {list(LIMITS)}")
        if name in out and out[name] != value:
            raise ValueError(f"{ENV}: {name} given twice")
        out[name] = value
    return out


def scales() -> Dict[str, float]:
    """The current setting, parsed (cached per value of the variable)."""
    text = os.environ.get(ENV, "")
    if text not in _parsed:
        _parsed[text] = parse(text)
    return _parsed[text]


def map_name(header_path: str) -> str:
    base = os.path.basename(str(header_path))
    return base[: -len(".header")] if base.endswith(".header") else base


def factor(header_path: str) -> float:
    """The factor for the map at header_path (1.0 when unscaled); reports an applied one once per process."""
    s = scales()
    if not s:
        return 1.0
    name = map_name(header_path)
    f = s.get(name, 1.0)
    if f != 1.0 and name not in _reported:
        _reported.add(name)
        print(f"bfield_scale: map {name} scaled by {f:g}", file=sys.stderr, flush=True)
    return f


def setting(params: Dict[str, float]) -> str:
    """The variable's value for task parameters {ds_bscale, tsd_bscale}; '' when every factor is 1 (unscaled)."""
    entries = []
    for key, maps in TASK_PARAMS.items():
        v = float(params.get(key, 1.0))
        if v != 1.0:
            entries += [f"{m}={v!r}" for m in maps]
    text = ":".join(entries)
    parse(text)
    return text


def record() -> Dict[str, float]:
    """The applied setting for run records: {} when unscaled."""
    return dict(scales())
