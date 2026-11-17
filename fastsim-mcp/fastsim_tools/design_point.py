"""One stopping-target design point: the overlay, the kit requests per dtype, and what comes back.

Pure functions (no MCP, no network, no kit): ``render_target`` turns a target description into the SimpleConfig
overlay the kit accepts as a geometry FileRef; ``requests`` names one kit job per dtype; ``summarise`` turns the kit's
results into the S/sqrt(B)-ready outputs and the analysis-server call that scores them.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

BASE = "Offline/Mu2eG4/geom/geom_common.txt"          # the one base include the kit admits
FORMAT = "offline_simpleconfig"
RENDERER = "fastsim_tools.design_point.render_target/1"
RENDERER_SHA256 = hashlib.sha256(RENDERER.encode()).hexdigest()

# overlay keys: short name -> (SimpleConfig key, vector?)
TARGET_KEYS: Dict[str, Tuple[str, bool]] = {
    "radii": ("stoppingTarget.radii", True),
    "halfThicknesses": ("stoppingTarget.halfThicknesses", True),
    "zVars": ("stoppingTarget.zVars", True),
    "holeRadii": ("stoppingTarget.holeRadii", True),
    "deltaZ": ("stoppingTarget.deltaZ", False),
    "z0InMu2e": ("stoppingTarget.z0InMu2e", False),
    "holeRadius": ("stoppingTarget.holeRadius", False),
    "distFromTargetEnd": ("protonabsorber.distFromTargetEnd", False),
}
FULL_KEYS = {full: (short, vec) for short, (full, vec) in TARGET_KEYS.items()}
ABSORBER_OFF = ("hasTSdA",)

# stage -> (kit task, takes dtype, electron limits); CPU tasks only (this server never asks for a GPU)
STAGES: Dict[str, Tuple[str, bool, Tuple[int, int]]] = {
    # one CPU core per dtype; 240 min job limit: about 53 ms per electron (float64) plus ~7 min of compilation
    "edep": ("beam_edep_cpu", True, (1000, 50_000)),
    # one 32-core CPU job (cesob: stops, CE chain, calorimeter wall proxy); no dtype switch
    "sob": ("beam_sob", False, (1000, 200_000)),
}
DTYPES = ("float64", "float32")
EDEP_FILE = "nts.mu2efastsim.edep.root"
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,63}")
GEOM_NAME = re.compile(r"[A-Za-z0-9_.-]+\.txt")

# the parameters the stopping-target design studies pass to approx_ce_sensitivity (besides stops_per_pot)
STUDY_PARAMETERS = {"selection": "event_calo_edep_vis > 10", "npot": 1e18, "cosmic_rate_per_s_per_mev": 1 / 550}


def _number(key: str, v: Any) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"target {key}: finite numbers only, got {v!r}")
    return repr(float(v)) if isinstance(v, float) else str(int(v))


def render_target(target: Mapping[str, Any]) -> str:
    """The SimpleConfig overlay of a target: ``#include`` of the base geometry, then one declaration per key.

    Keys: radii, halfThicknesses, zVars, holeRadii (lists, one entry per foil); deltaZ, z0InMu2e, holeRadius,
    distFromTargetEnd (numbers); hasTSdA (only false). Full SimpleConfig names (stoppingTarget.radii, ...) work too.
    Lists must have one length. The kit validates the text again before it binds it to a job."""
    if not isinstance(target, Mapping) or not target:
        raise ValueError("target: give at least one key, e.g. {\"radii\": [75.0, ...], \"zVars\": [...]}")
    lines = [f'#include "{BASE}"']
    lengths = {}
    for key in sorted(target):
        v = target[key]
        if key in ABSORBER_OFF:
            if v is not False:
                raise ValueError(f"target {key}: only false is allowed (turns that absorber off)")
            lines.append(f"bool {key} = false;")
            continue
        if key in TARGET_KEYS:
            full, vec = TARGET_KEYS[key]
        elif key in FULL_KEYS:
            full, vec = key, FULL_KEYS[key][1]
        else:
            raise ValueError(f"target: unknown key {key!r}; known: {', '.join([*TARGET_KEYS, *ABSORBER_OFF])}")
        if vec:
            if not isinstance(v, Sequence) or isinstance(v, (str, bytes)) or not v:
                raise ValueError(f"target {key}: a nonempty list of numbers")
            lengths[key] = len(v)
            lines.append(f"vector<double> {full} = {{{', '.join(_number(key, x) for x in v)}}};")
        else:
            lines.append(f"double {full} = {_number(key, v)};")
    if len(set(lengths.values())) > 1:
        raise ValueError(f"target: per-foil lists differ in length: {lengths}")
    short = {FULL_KEYS.get(k, (k,))[0] for k in target}
    if {"holeRadius", "holeRadii"} <= short:
        raise ValueError("target: give holeRadius or holeRadii, not both")
    for k in ("radii", "halfThicknesses", "deltaZ"):
        if k in target and min(_as_list(target[k])) <= 0:
            raise ValueError(f"target {k}: must be positive")
    for k in ("holeRadius", "holeRadii"):
        if k in target and min(_as_list(target[k])) < 0:
            raise ValueError(f"target {k}: must be nonnegative")
    return "\n".join(lines) + "\n"


def _as_list(v):
    return list(v) if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) else [v]


def file_ref(uri: str, text: str) -> Dict[str, Any]:
    """The kit's geometry FileRef for an overlay written at ``uri`` (file:// on the kit host)."""
    return {"name": "geom", "kind": "geom", "uri": uri,
            "sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
            "metadata": {"format": FORMAT, "schema_version": 1, "renderer": FORMAT,
                         "renderer_sha256": RENDERER_SHA256}}


def check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError(f"name {name!r}: letters, digits, _ and -, at most 64 characters, not starting with -")
    return name


def check_geometry(geometry: str) -> str:
    if geometry == "nominal" or GEOM_NAME.fullmatch(geometry or ""):
        return geometry
    raise ValueError(f"geometry {geometry!r}: 'nominal' or a .txt file name under beam/geom; use target for "
                     "a new design")


def requests(name: str, stage: str, dtypes: Iterable[str], electrons: int, muons: int, seed: int,
             geometry: Optional[str]) -> List[Dict[str, Any]]:
    """One kit request per job: {"handle", "dtype", "params"}. ``geometry`` None: the caller sends a FileRef."""
    check_name(name)
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r}: choose from {', '.join(STAGES)}")
    task, has_dtype, (lo, hi) = STAGES[stage]
    if not lo <= electrons <= hi:
        raise ValueError(f"electrons {electrons}: stage {stage} takes {lo:,}-{hi:,}")
    if not 0 <= muons <= 10 ** 7:
        raise ValueError("muons: 0 (the whole beam table) to 10^7")
    if not 0 <= seed < 2 ** 31:
        raise ValueError("seed: 0 to 2^31-1")
    dts = list(dict.fromkeys(dtypes))
    if not dts or any(d not in DTYPES for d in dts):
        raise ValueError(f"dtypes {dtypes!r}: one or both of {', '.join(DTYPES)}")
    base = {"task": task, "muons": muons, "electrons": electrons, "seed": seed}
    if geometry is not None:
        base["geom"] = check_geometry(geometry)
    if not has_dtype:
        return [{"handle": f"{name}.{stage}", "dtype": None, "params": base}]
    return [{"handle": f"{name}.{stage}.{d}", "dtype": d, "params": {**base, "dtype": d}} for d in dts]


def expected_minutes(stage: str, electrons: int, dtype: Optional[str]) -> float:
    """Rough wall time once the job runs (beam/docs/edepana.md, docs/performance.md), queue not included."""
    if stage == "edep":
        per_e = 0.053 if dtype != "float32" else 0.067
        return round(9 + electrons * per_e / 60, 1)
    return round(4 + electrons / 117 / 60, 1)


def summarise(stage: str, dtype: Optional[str], result: Mapping[str, Any]) -> Dict[str, Any]:
    """The S/sqrt(B)-ready view of one kit result: metrics, the output files and, for edep, the scoring call."""
    metrics = dict(result.get("metrics") or {})
    files = list(result.get("files") or [])
    meta = dict(result.get("metadata") or {})
    out: Dict[str, Any] = {"dtype": dtype, "metrics": metrics, "job_id": meta.get("job_id"),
                           "files": files, "unavailable_metrics": meta.get("unavailable_metrics") or []}
    if stage == "sob":
        out["sensitivity"] = metrics.get("sensitivity")
        out["note"] = ("beam_sob's built-in approximate sensitivity (calorimeter wall proxy); for the analysis "
                       "server's approx_ce_sensitivity run stage 'edep'")
        return out
    edep = next((f for f in files if f.get("name") == EDEP_FILE), None)
    if edep is None:
        out["note"] = f"no {EDEP_FILE} among the results"
        return out
    path = file_path(edep["uri"])
    out["edepana_file"] = path
    out["edep_run_record"] = path + ".edep.json"
    out["edep_summary"] = path.rsplit("/", 1)[0] + "/edep_summary.txt"
    params = {"stops_per_pot": metrics["stops_per_pot"]} if "stops_per_pot" in metrics else {}
    params.update(STUDY_PARAMETERS)
    out["score_with"] = {
        "server": "Mu2e analysis-mcp", "tool": "run_analysis",
        "arguments": {"analysis": "approx_ce_sensitivity", "data_file": path,
                      "output_dir": path.rsplit("/", 1)[0] + "/approx_ce_sensitivity", "parameters": params},
        "note": ("stops_per_pot is the fast W1 value of this point (full fast); for a file-level comparison with "
                 "a stock board pass the board's Geant4 stops_per_pot instead. The other parameters are the "
                 "design studies' values; the analysis' own defaults apply to anything left out."),
    }
    return out


def file_path(uri: str) -> str:
    from urllib.parse import unquote, urlsplit
    parts = urlsplit(uri)
    if parts.scheme != "file":
        raise ValueError(f"not a file URI: {uri!r}")
    return unquote(parts.path)


def overall_state(states: Iterable[str]) -> str:
    """One state for a point from its jobs' kit states (working, completed, failed, cancelled)."""
    s = list(states)
    if not s:
        return "unknown"
    if all(x == "completed" for x in s):
        return "completed"
    if any(x == "working" for x in s):
        return "working"
    if any(x == "failed" for x in s):
        return "failed"
    if any(x == "cancelled" for x in s):
        return "cancelled"
    return "unknown"
