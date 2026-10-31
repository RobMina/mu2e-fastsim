"""Tier-1b asset sets: the library, its placement sidecars and the sampling options that go with them.

The callers (tier1b-apply, runchain, e2e-bench, the endpoint kits) pick one set by name; crvfastsim.tier1b itself has
no default library. docs/tier1b-default.md has the provenance and the evidence behind the default.

  strict_v4  default for extracted_v04: strict_v3 with whole-muon draws of hard library muons (summed descendant
             energy above 30 MeV, keyed by momentum class and the group and layer of each crossing; tier1b.attach_hard),
             which restores the correlation of a muon's descendants across its crossings.
  strict_v3  strict_v2 with the v3 template sidecar (library chords, the larger near-miss census) and chord placement of
             the same-bar descendant steps (the default 2026-10-29 to 2026-10-30).
  strict_v2  association-certified library (20 donors) with crossing context, endpoint containment, point
             placement, topology placement and the never-visited-layer / near-miss templates (the default
             2026-10-28 to 2026-10-29).
  legacy     the tier1b_v7 production library with no placement options (the default before 2026-10-28), and the
             only set for the named run configurations (run1a_v01, run1b_v01, run2_v10): no library has been built
             on their geometry, so they map their sectors onto this extracted library's groups (runconfig.lib_group).

    spec = resolve(run_config="extracted_v04")            # the configuration's default set under $CRVFS_DATA
    spec = resolve("legacy"); spec = resolve("strict_v4", directory="/dev/shm/x")   # files staged elsewhere
    spec = resolve(library="lib.root")                     # a custom library, no placement options
    lib = load(spec, geom); tier1b.sample(..., lib, geom, rng, **spec.options)
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT = "strict_v4"
LEGACY = "legacy"
EXTRACTED = "extracted_v04"


@dataclass(frozen=True)
class AssetSet:
    directory: str                     # under $CRVFS_DATA
    library: str
    topology: str | None = None        # valhist.descendant_topology sidecar (+ .json)
    templates: str | None = None       # valhist.descendant_templates sidecar (+ .json)
    library_sha256: str | None = None  # pinned content of the library the sidecars were made for
    options: dict = field(default_factory=dict)   # tier1b.sample keywords
    run_configs: tuple = (EXTRACTED,)

    def files(self):
        """Every file of the set, relative to its directory (sidecars with their metadata)."""
        side = [s for p in (self.topology, self.templates) if p for s in (p, p + ".json")]
        return [self.library] + side


_STRICT = dict(context=True, contain=True, point=True, topology=True, unvisited=True, nearmiss=True)
_STRICT_LIB = "ebb015a76d56edcf96c5bc533daa96729e3851ee3e59185abf0be5fd469257c7"

SETS = {
    "strict_v4": AssetSet(
        "tier1b_strict_v2", "library.root", "topology_sidecar_strict_v2_v2.npz", "templates_v3.npz", _STRICT_LIB,
        dict(_STRICT, chord=True, hard=True)),
    "strict_v3": AssetSet(
        "tier1b_strict_v2", "library.root", "topology_sidecar_strict_v2_v2.npz", "templates_v3.npz", _STRICT_LIB,
        dict(_STRICT, chord=True)),
    "strict_v2": AssetSet(
        "tier1b_strict_v2", "library.root", "topology_sidecar_strict_v2_v2.npz", "templates_v2.npz",
        _STRICT_LIB, dict(_STRICT)),
    "legacy": AssetSet("tier1b_v7", "lib_prod_as_20files.root",
                       run_configs=(EXTRACTED, "run1a_v01", "run1b_v01", "run2_v10")),
}


@dataclass(frozen=True)
class Tier1bSpec:
    """A resolved choice: name (None for a custom library), absolute paths and the sample options."""
    name: str | None
    library: str
    topology: str | None = None
    templates: str | None = None
    library_sha256: str | None = None
    options: dict = field(default_factory=dict)

    def paths(self):
        """Every file the choice reads (library, sidecars and their metadata)."""
        side = [s for p in (self.topology, self.templates) if p for s in (p, p + ".json")]
        return [self.library] + side

    def describe(self):
        return {"set": self.name, "paths": self.paths(), "options": dict(self.options),
                "library_sha256": self.library_sha256}


def default_for(run_config=None):
    """The default set of a run configuration: strict_v4 for extracted_v04, legacy for the named ones."""
    return DEFAULT if (run_config or EXTRACTED) == EXTRACTED else LEGACY


def resolve(tier1b=None, library=None, directory=None, run_config=None, data=None):
    """Tier1bSpec from a set name or a custom library path (not both). Neither: the run configuration's default set.
    directory: where the set's files are (default $CRVFS_DATA/<set directory>)."""
    rc = run_config or EXTRACTED
    if library and (tier1b or directory):
        raise ValueError("pass either a custom --library or --tier1b [--tier1b-dir], not both")
    if library:
        return Tier1bSpec(None, str(library))
    name = tier1b or default_for(rc)
    if name not in SETS:
        raise ValueError(f"unknown tier-1b set {name!r}; known: {sorted(SETS)}")
    s = SETS[name]
    if rc not in s.run_configs:
        raise ValueError(f"tier-1b set {name!r} has no library for run configuration {rc!r} "
                         f"(use {LEGACY!r}: {sorted(n for n, x in SETS.items() if rc in x.run_configs)})")
    if directory is None:
        data = data or os.environ.get("CRVFS_DATA")
        if not data:
            raise ValueError(f"tier-1b set {name!r}: set CRVFS_DATA (source site.env), pass --tier1b-dir or --library")
        directory = Path(data) / s.directory
    d = Path(directory)
    p = lambda f: str(d / f) if f else None
    return Tier1bSpec(name, p(s.library), p(s.topology), p(s.templates), s.library_sha256, dict(s.options))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(spec, geom):
    """Load checked sidecars normally, or opt into a content-bound attached-library cache."""
    directory = os.environ.get("CRVFS_PREPARED_LIBRARY_CACHE")
    if directory:
        from .analyses.library_cache import load as cached_load
        return cached_load(spec, geom, directory, _load_uncached)
    return _load_uncached(spec, geom)


def _load_uncached(spec, geom):
    """Attach checked topology and template sidecars in their reference order."""
    from . import tier1b
    from .analyses.tier1b import read_template_sidecar, read_topology_sidecar
    lib = tier1b.read_library(spec.library)
    if not (spec.topology or spec.templates):
        return lib
    digest = sha256(spec.library)
    if spec.library_sha256 and digest != spec.library_sha256:
        raise ValueError(f"tier-1b set {spec.name!r}: {spec.library} is not the pinned library")
    if spec.topology:
        lib = tier1b.attach_topology(lib, *read_topology_sidecar(spec.topology, library_sha256=digest), geom)
    if spec.templates:
        lib = tier1b.attach_templates(lib, read_template_sidecar(spec.templates, library_sha256=digest), geom)
    if spec.options.get("hard"):
        lib = tier1b.attach_hard(lib, geom)
    return lib
