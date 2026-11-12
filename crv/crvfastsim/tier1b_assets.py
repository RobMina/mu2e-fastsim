"""Tier-1b asset sets: the library, its placement sidecars and the sampling options that go with them.

The callers (tier1b-apply, runchain, e2e-bench, the endpoint kits) pick one set by name; crvfastsim.tier1b itself has
no default library. docs/tier1b-default.md has the provenance and the evidence behind the default.

  strict_v5  default, extracted_v04: the strict library rebuilt with step-tree roots (the same 20 donors, bins and
             showers) with crossing context, endpoint containment, point and topology placement, the never-visited-layer
             / near-miss templates (v3 sidecar), chord placement and root-coherent draws: a delta ray and its cascade
             stay together along the fast muon within a module (tier1b.attach_anchors, sample(anchor)).
  The previous default strict_v4 (hard-muon draws) is on branch park/tier1b-strict-v4.
  run1a_v01  opt-in, run1a_v01 only: the same builder and options on the Run 1A geometry, per primary muon
             (tier1b-build --run-config), from the development half of the Run 1A Geant4 reference (odd subruns
             101-119, 500k generated); non-muon showers on (the library's own bar positions), near-miss per muon.

Other option combinations run with a custom library and explicit placement flags (tier1b-apply --library ...
--crossing-context ...; docs/tier1b-default.md). run1b_v01 and run2_v10 have no set: they need --library.

    spec = resolve(run_config="extracted_v04")            # the configuration's default set under $CRVFS_DATA
    spec = resolve("strict_v5", directory="/dev/shm/x")   # the default set, files staged elsewhere
    spec = resolve(library="lib.root")                     # a custom library, no placement options
    lib = load(spec, geom); tier1b.sample(..., lib, geom, rng, **spec.options)

Library pinning (mu2efastsim.library, core/docs/libraries.md "Pinning in run configurations"): a named set is pinned
by the sha256 of its library file (library_sha256), checked on every load (cached or not); a custom --library is
allowed and recorded as unpinned. Every load is entered in mu2efastsim.library.provenance. domain_counts(info) turns
a tier1b.sample info record (dropped, clamped and other-bar descendant placements) into a DomainCounter.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT = "strict_v5"
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
_ROOTS_LIB = "fc6e1518a90632217f948e43cefde7c0a4f8919e6e4ba0e8ffa2d1d939bcc5f4"

SETS = {
    "strict_v5": AssetSet(
        "tier1b_strict_v5", "library.root", "topology.npz", "templates_v3.npz", _ROOTS_LIB,
        dict(_STRICT, chord=True, anchor=True)),
    "run1a_v01": AssetSet(
        "tier1b_run1a_v01", "library.root", "topology.npz", "templates_v3.npz",
        "75da97efbdc593bbed4831d7c427199d3842e487702b272138e0a97e19cc83cd",
        dict(_STRICT, chord=True, anchor=True, per_primary=True, showers=True), ("run1a_v01",)),
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
    """The default set of a run configuration: strict_v5 for extracted_v04, None for the named ones (no library)."""
    return DEFAULT if (run_config or EXTRACTED) == EXTRACTED else None


def resolve(tier1b=None, library=None, directory=None, run_config=None, data=None):
    """Tier1bSpec from a set name or a custom library path (not both). Neither: the run configuration's default set.
    directory: where the set's files are (default $CRVFS_DATA/<set directory>)."""
    rc = run_config or EXTRACTED
    if library and (tier1b or directory):
        raise ValueError("pass either a custom --library or --tier1b [--tier1b-dir], not both")
    if library:
        return Tier1bSpec(None, str(library))
    name = tier1b or default_for(rc)
    if name is None:
        raise ValueError(f"run configuration {rc!r} has no tier-1b asset set (no library has been built on its "
                         f"geometry): pass a custom --library")
    if name not in SETS:
        raise ValueError(f"unknown tier-1b set {name!r}; known: {sorted(SETS)}")
    s = SETS[name]
    if rc not in s.run_configs:
        raise ValueError(f"tier-1b set {name!r} has no library for run configuration {rc!r}: pass a custom --library")
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


def _record(spec, digest):
    """Enter the load in mu2efastsim.library.provenance: pinned (a named set, its library sha256) or unpinned."""
    from mu2efastsim.library import provenance
    pinned = bool(spec.name and spec.library_sha256)
    return provenance.record(dict(name=f"crv.tier1b.{spec.name or 'custom'}", file=os.path.basename(spec.library),
                                  content_hash=digest, hash="file_sha256", status="pinned" if pinned else "unpinned",
                                  pin_set=spec.name if pinned else None, options=dict(spec.options)))


def load(spec, geom):
    """Load checked sidecars normally, or opt into a content-bound attached-library cache. A named set's library must
    have its pinned sha256 (refused otherwise); the load is recorded pinned or unpinned (custom library)."""
    digest = sha256(spec.library)
    if spec.library_sha256 and digest != spec.library_sha256:
        raise ValueError(f"tier-1b set {spec.name!r}: {spec.library} is not the pinned library")
    directory = os.environ.get("CRVFS_PREPARED_LIBRARY_CACHE")
    if directory:
        from .analyses.library_cache import load as cached_load
        lib = cached_load(spec, geom, directory, _load_uncached)
    else:
        lib = _load_uncached(spec, geom)
    _record(spec, digest)
    return lib


def domain_counts(info, counter=None, library="crv.tier1b"):
    """DomainCounter from a tier1b.sample info record: descendant steps placed (lookups), served by a fallback
    placement (clamped back inside a bar: level 1; moved to another bar: level 2) and dropped (refused)."""
    from mu2efastsim.library.domain import DomainCounter
    counter = counter or DomainCounter(library)
    placed, dropped, clamped = int(info["desc_steps"]), int(info["desc_dropped"]), int(info["desc_clamped"])
    other = int(info.get("other_bar", {}).get("other", 0))
    counter.add_lookups(placed + dropped)
    counter.add_refused(dropped)
    d = counter.levels.setdefault("placement", {})
    for level, n in (("0", placed - clamped - other), ("1", clamped), ("2", other)):
        d[level] = d.get(level, 0) + n
    return counter


def _load_uncached(spec, geom):
    """Attach checked topology and template sidecars in their reference order, then anchors."""
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
    if spec.options.get("anchor"):
        lib = tier1b.attach_anchors(lib, geom)
    return lib
