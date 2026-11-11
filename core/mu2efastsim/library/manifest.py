"""The library manifest: what a library is, what it was built from, and how it may be used (docs/libraries.md).

A manifest is a JSON object with the fields in FIELDS. Every field must be present. A field whose value is not known
(a library built before this scheme) is ``null`` and is listed in ``unrecorded`` with the reason; a manifest with no
unrecorded field is *complete*. Only a complete manifest is admissible for a new library; migrated libraries carry
their gaps openly instead of inventing provenance.

:func:`create` builds and checks one, :func:`validate` checks a loaded one (raising :class:`ManifestError` with every
problem listed), and :func:`manifest_hash` is the digest of its canonical JSON. Standard library and NumPy only.
"""
import copy
import json
from pathlib import Path

from .hashing import canonical_json, digest, is_hex

SCHEMA = 1
SUBSYSTEMS = ("core", "crv", "calo", "tracker", "flash", "secondaries", "beam", "trigger", "mixing")
KINDS = ("binned", "pool", "parametric", "interpolation", "replay")
STATUSES = ("bit-identical", "exact", "statistically equivalent", "tested, not closed", "not validated")
COMPUTE_DTYPES = ("float32", "float64")

# field -> (types allowed when not null, description)
FIELDS = {
    "schema": (int, "manifest schema (SCHEMA)"),
    "name": (str, "library name, '<subsystem>.<what>.<configuration>' (e.g. 'tracker.nrow.run1a')"),
    "subsystem": (str, "one of SUBSYSTEMS"),
    "version": (int, "positive integer, raised on every change of content"),
    "kind": (str, "one of KINDS"),
    "description": (str, "one paragraph: what the library answers"),
    "axes": (list, "the keys: axes.Axis.describe() entries (name, unit, frame, array, scale, policy, nbins, lo, hi)"),
    "outputs": (list, "what a lookup returns: [{name, unit, dtype, description}]"),
    "arrays": (dict, "every stored array: {name: {dtype, shape}}"),
    "content_hash": (str, "hashing.content_hash of the arrays (64 hex)"),
    "source": (dict, "source samples: {samples: [{dataset, selection, events}], description}"),
    "generator": (dict, "what produced the source samples: {name, release, physics_list, configuration}"),
    "release": (dict, "code that built the library: {repository, commit, entry_point}"),
    "seeds": (dict, "every seed of the source samples and of the build: {name: value}"),
    "geometry": (dict, "geometry or configuration identity: {tag, hash}"),
    "config_hash": (str, "hashing.digest of the build configuration, when one exists"),
    "split": (dict, "training and held-out samples and the rule that keeps them disjoint: {training, held_out, rule}"),
    "dtype": (dict, "{storage: {array: dtype}, compute: ['float32', 'float64'], notes}"),
    "rng": (dict, "{host: 'numpy.random.Generator', device: 'jax.random key', streams: {...}, draws_per_row}"),
    "domain": (dict, "{coverage: text, fallbacks: [{name, levels, description}]} (axis policies live in axes)"),
    "validation": (dict, "{status: one of STATUSES, record: path or name of the validation record}"),
    "unrecorded": (dict, "{field: reason} for every null field"),
}
OPTIONAL = {"legacy": (dict, "the format the library had before this manifest: {schema, format}"),
            "interpolation": (dict, "how a lookup combines cells: {method: 'nearest' | 'linear', axes: [...]}"),
            "assumptions": (list, "modelling assumptions beyond the keys: [{name, description, tested}]"),
            "created_utc": (str, "build time, ISO 8601"),
            "notes": (str, "anything else")}


class ManifestError(ValueError):
    """A manifest that does not meet the schema; ``problems`` lists every failure."""

    def __init__(self, name, problems):
        self.problems = list(problems)
        super().__init__(f"library manifest {name!r}: " + "; ".join(self.problems))


def _problems(m):
    out = []
    if not isinstance(m, dict):
        return ["the manifest is not a JSON object"]
    unknown = set(m) - set(FIELDS) - set(OPTIONAL)
    if unknown:
        out.append(f"unknown fields {sorted(unknown)}")
    missing = [k for k in FIELDS if k not in m]
    if missing:
        out.append(f"missing fields {missing}")
    unrecorded = m.get("unrecorded") or {}
    if not isinstance(unrecorded, dict):
        out.append("unrecorded must be an object {field: reason}")
        unrecorded = {}
    for k, (types, _) in {**FIELDS, **OPTIONAL}.items():
        if k not in m:
            continue
        v = m[k]
        if v is None:
            if k in ("schema", "name", "subsystem", "version", "kind", "axes", "arrays", "content_hash",
                     "dtype", "validation", "unrecorded"):
                out.append(f"{k} may not be null")
            elif k in FIELDS and not str(unrecorded.get(k, "")).strip():
                out.append(f"{k} is null but has no reason in unrecorded")
            continue
        if not isinstance(v, types) or (types is int and isinstance(v, bool)):
            out.append(f"{k} must be {types.__name__}")
    for k in unrecorded:
        if k not in FIELDS:
            out.append(f"unrecorded names an unknown field {k!r}")
        elif m.get(k) is not None:
            out.append(f"{k} is listed in unrecorded but has a value")
    if out:
        return out
    if m["schema"] != SCHEMA:
        out.append(f"schema {m['schema']}, expected {SCHEMA}")
    if m["subsystem"] not in SUBSYSTEMS:
        out.append(f"subsystem {m['subsystem']!r} is not one of {SUBSYSTEMS}")
    if not m["name"].startswith(m["subsystem"] + "."):
        out.append(f"name {m['name']!r} must start with '{m['subsystem']}.'")
    if m["version"] < 1:
        out.append("version must be a positive integer")
    if m["kind"] not in KINDS:
        out.append(f"kind {m['kind']!r} is not one of {KINDS}")
    if not is_hex(m["content_hash"]):
        out.append("content_hash must be 64 lower-case hex digits")
    for a in m["axes"]:
        if not isinstance(a, dict) or not {"name", "unit", "array", "policy", "nbins", "lo", "hi"} <= set(a):
            out.append(f"axis entry {a!r} needs name, unit, array, policy, nbins, lo, hi")
        elif a["array"] not in m["arrays"]:
            out.append(f"axis {a['name']!r}: edges array {a['array']!r} is not among the arrays")
    for name, spec in m["arrays"].items():
        if not isinstance(spec, dict) or set(spec) != {"dtype", "shape"}:
            out.append(f"array {name!r} needs exactly dtype and shape")
    for o in m["outputs"] or []:
        if not isinstance(o, dict) or not {"name", "unit", "dtype"} <= set(o):
            out.append(f"output entry {o!r} needs name, unit, dtype")
    compute = (m["dtype"] or {}).get("compute")
    if not isinstance(compute, list) or not compute or not set(compute) <= set(COMPUTE_DTYPES):
        out.append(f"dtype.compute must list the compute dtypes it supports, from {COMPUTE_DTYPES}")
    if m["validation"].get("status") not in STATUSES:
        out.append(f"validation.status must be one of {STATUSES}")
    if m.get("interpolation") is not None and m["interpolation"].get("method") not in ("nearest", "linear"):
        out.append("interpolation.method must be 'nearest' or 'linear'")
    for a in m.get("assumptions") or []:
        if not isinstance(a, dict) or not {"name", "description", "tested"} <= set(a):
            out.append(f"assumption {a!r} needs name, description, tested")
    if m.get("config_hash") is not None and not is_hex(m["config_hash"], len(m["config_hash"])):
        out.append("config_hash must be hex")
    return out


def validate(m):
    """Raise ManifestError listing every problem; return the manifest."""
    problems = _problems(m)
    if problems:
        raise ManifestError(m.get("name", "?") if isinstance(m, dict) else "?", problems)
    return m


def complete(m):
    """True when no field is unrecorded."""
    return not (m.get("unrecorded") or {})


def array_specs(arrays):
    """The manifest ``arrays`` entry of a mapping name -> array."""
    import numpy as np
    return {k: {"dtype": np.asarray(v).dtype.str, "shape": list(np.asarray(v).shape)} for k, v in sorted(arrays.items())}


def create(**fields):
    """A checked manifest from keyword fields. Missing FIELDS become null and must be explained in ``unrecorded``;
    ``schema`` defaults to SCHEMA."""
    m = {k: None for k in FIELDS}
    m["schema"] = SCHEMA
    m["unrecorded"] = {}
    m.update(copy.deepcopy(fields))
    return validate(m)


def manifest_hash(m, length=None):
    """Digest of the manifest's canonical JSON (all 64 hex digits by default)."""
    return digest(m, length)


def dumps(m):
    """Canonical text of a manifest (the form stored in a library file)."""
    return canonical_json(m)


def read(path):
    """A manifest from a JSON file, checked."""
    return validate(json.loads(Path(path).read_text()))


def write(path, m):
    """Write a checked manifest as indented JSON (sorted keys), for sidecars and review."""
    validate(m)
    Path(path).write_text(json.dumps(m, sort_keys=True, indent=1, allow_nan=False) + "\n")
