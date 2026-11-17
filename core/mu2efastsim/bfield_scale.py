"""Opt-in uniform scale factors for named G4BL field maps, as Offline's ``bfield.mapScale.<map>``.

``MU2EFS_BFIELD_MAPSCALE="DSMap=0.9,DSExtension=0.9,TSdMap=1.0"`` multiplies every grid value of each named map when
it is read (``ehelix.read_g4bl``, ``beamfastsim.fieldmap.load``). The key is the map's header basename. Unset or empty:
no map is touched. The environment variable is the single switch, so worker processes inherit it.
"""
import os
import shlex
import sys
from pathlib import Path

ENV = "MU2EFS_BFIELD_MAPSCALE"
MAPS = ("PSMap", "TSuMap", "TSdMap", "DSMap", "DSExtension", "PStoDumpAreaMap", "ProtonBeamDumpMap")
LIMITS = (0.5, 1.5)
KIT_PARAMS = {"ds_bscale": ("DSMap", "DSExtension"), "tsd_bscale": ("TSdMap",)}

_reported = {}


def parse(text):
    """{map: factor} from the variable's text; an unknown map, a bad number or a factor outside LIMITS raises."""
    out = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in MAPS:
            raise ValueError(f"{ENV}: {item!r} is not <map>=<factor> with a map in {list(MAPS)}")
        if name in out:
            raise ValueError(f"{ENV}: map {name} given twice")
        try:
            f = float(value)
        except ValueError:
            raise ValueError(f"{ENV}: {name} factor {value!r} is not a number") from None
        if not LIMITS[0] <= f <= LIMITS[1]:
            raise ValueError(f"{ENV}: {name} factor {f} is outside [{LIMITS[0]}, {LIMITS[1]}]")
        out[name] = f
    return out


def factors():
    """The configured {map: factor} (empty when the variable is unset)."""
    return parse(os.environ.get(ENV, ""))


def map_name(header_path):
    """Offline's map key for a header path: its basename without ``.header``."""
    name = Path(os.fspath(header_path)).name
    return name[: -len(".header")] if name.endswith(".header") else name


def scale_for(header_path):
    """The factor for this map (1.0 when not configured); the first use of a factor is printed, as Offline does."""
    name = map_name(header_path)
    f = factors().get(name, 1.0)
    if f != 1.0 and _reported.get(name) != f:
        print(f"map {name} scaled by {f}", file=sys.stderr, flush=True)
        _reported[name] = f
    return f


def provenance():
    """{"bfield_mapscale": {map: factor}} for a run record when any map is scaled, else {}."""
    f = {k: v for k, v in factors().items() if v != 1.0}
    return {"bfield_mapscale": f} if f else {}


def kit_value(params):
    """The variable's value for kit params ds_bscale / tsd_bscale (factors other than 1 only; "" when none)."""
    items = []
    for key, maps in KIT_PARAMS.items():
        f = float(params.get(key, 1.0))
        if f != 1.0:
            items += [f"{m}={f!r}" for m in maps]
    value = ",".join(items)
    parse(value)
    return value


def kit_exports(params):
    """Shell lines that set the variable in a kit job (also inside an apptainer --cleanenv container); "" at 1."""
    value = kit_value(params)
    if not value:
        return ""
    q = shlex.quote(value)
    return f"export {ENV}={q} APPTAINERENV_{ENV}={q}\n"
