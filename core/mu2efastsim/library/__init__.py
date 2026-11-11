"""Shared core of every response and interaction library (spec: core/docs/libraries.md).

Subsystem libraries (CRV photon and secondary tables, calorimeter response, tracker response and row pools, flash
response, geometry-portable interaction banks, beam tables) stay in their packages and build on these pieces:

    hashing     canonical JSON, file sha256, array content hash, short digests
    axes        Axis (key, unit, frame, edges, out-of-domain policy), Grid (row-major cells), NumPy/JAX bin lookup
    domain      OutOfDomain (loud refusal), DomainCounter (declared fallbacks and clips, counted)
    manifest    the manifest schema (provenance, split, seeds, dtype and RNG conventions, validation status)
    store       .npz library files with an embedded or sidecar manifest; load() checks everything it can
    registry    shipped names, registry JSON, release assets and the download cache (sha256 checked each time)
    dtypes      the float32/float64 compute policy and device copies
    fidelity    the validation vocabulary: status levels, metrics, criteria declared before the held-out read,
                validation records, and the bench performance entry

NumPy and the standard library only; the ``*_jax`` helpers import JAX when they are called.
"""
from . import axes, domain, dtypes, fidelity, hashing, manifest, registry, store  # noqa: F401
from .axes import Axis, Grid  # noqa: F401
from .domain import DomainCounter, OutOfDomain  # noqa: F401
from .registry import Registry  # noqa: F401
from .store import Library, LibraryError, load, save  # noqa: F401
