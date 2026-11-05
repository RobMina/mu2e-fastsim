"""Audit parent forests and label descendants of explicitly designated incidents."""
from __future__ import annotations

import numpy as np

from .identities import EVENT_FIELDS, contains, keys


SIM_FIELDS = EVENT_FIELDS + ("sim_product_id", "sim_key")


def _references(table, prefix):
    return keys({**table, "sim_product_id": table[prefix + "_product_id"],
                 "sim_key": table[prefix + "_key"]}, SIM_FIELDS)


def forest(particles):
    """Return parent row indices, rejecting unresolved links, duplicate IDs and cycles.

    Process one independently identified source artifact and completed transport
    only. Natural roots point to their own row internally; persistent identities
    always include the event and ProductID, not just the particle key.
    """
    identities = keys(particles, SIM_FIELDS)
    n = len(identities)
    if len(np.unique(identities)) != n or np.any(identities["sim_product_id"] == 0):
        raise ValueError("duplicate or zero SimParticle product identity")
    valid = np.asarray(particles["parent_valid"])
    if valid.shape != (n,) or valid.dtype.kind != "b":
        raise ValueError("parent presence flag must be a one-dimensional boolean column")
    references = _references(particles, "parent")
    if len(references) != n:
        raise ValueError("parent columns must match particles")
    order = np.argsort(identities)
    if np.any(valid & ~contains(identities[order], references)):
        raise ValueError("unresolved parent identity")
    parent = np.arange(n, dtype=np.int64)
    parent[valid] = order[np.searchsorted(identities[order], references[valid])]
    resolved = ~valid
    jump = parent.copy()
    for _ in range(n.bit_length() + 1):
        resolved = resolved | resolved[jump]
        jump = jump[jump]
        if np.all(resolved):
            break
    if not np.all(resolved):
        raise ValueError("cyclic SimParticle ancestry")
    return parent


def assign(particles, incidents, references):
    """Label supplied references by exact ancestry to one supplied root per event.

    Incidents use run/subrun/event/product_id/particle_key. References use
    run/subrun/event/sim_product_id/sim_key and must all belong to a designated
    root's descendant subtree. The returned index is the original incident row;
    neither particles nor references are reordered. Empty responses are allowed.
    This tests identity ancestry only: entrance-state selection, causal compression,
    material-boundary coverage, source independence and zero census are external.
    """
    parent = forest(particles)
    identities = keys(particles, SIM_FIELDS)
    root_ids = keys({**incidents, "sim_product_id": incidents["product_id"],
                     "sim_key": incidents["particle_key"]}, SIM_FIELDS)
    event_ids = keys(incidents, EVENT_FIELDS)
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("one designated incident per event required")
    order = np.argsort(identities)
    if not np.all(contains(identities[order], root_ids)):
        raise ValueError("designated incident identity missing from particles")
    roots = order[np.searchsorted(identities[order], root_ids)]
    label = np.full(len(identities), -1, dtype=np.int64)
    label[roots] = np.arange(len(roots), dtype=np.int64)
    jump = parent.copy()
    jump[roots] = roots
    for _ in range(len(identities).bit_length() + 1):
        inherited = label[jump]
        label = np.where(label >= 0, label, inherited)
        jump = jump[jump]
    ref_ids = keys(references, SIM_FIELDS)
    if not np.all(contains(identities[order], ref_ids)):
        raise ValueError("reference identity missing from particles")
    result = label[order[np.searchsorted(identities[order], ref_ids)]]
    if np.any(result < 0):
        raise ValueError("reference is not a descendant of a designated incident")
    return result.copy()
