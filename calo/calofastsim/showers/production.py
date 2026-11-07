"""Exact-identity reader for the explicitly selected production CE export mode."""
from __future__ import annotations

import numpy as np

from .export import EVENT, IDENTITY, columns, join, keys


def _unique(table, fields, label):
    identity = keys(table, fields)
    if len(np.unique(identity)) != len(identity):
        raise ValueError(f"duplicate {label} identity")
    return identity


def _ancestors(parent):
    ancestor = parent.copy()
    for _ in range(max(1, len(parent).bit_length())):
        advanced = ancestor[ancestor]
        if np.array_equal(advanced, ancestor):
            break
        ancestor = advanced
    return ancestor


def primary_identity(source, events):
    """Return the primary and its physical continuations, auditing all saved links.

    Ordinary descendants contribute to the whole-event shower, but only selfParent
    edges can extend the direct incident electron. The CE's generator link may be
    null: modern CeEndpoint is produced from a stopped-muon StageParticle.
    """
    event_ids = _unique(events, EVENT, "event")
    flags = columns(source, "events", EVENT + (
        "input_mode", "transport_product_id", "transport_particles_present",
        "primary_present", "primary_product_id", "n_primaries", "n_particles",
        "n_steps", "showers_present", "unresolved_links"))
    order = join(_unique(flags, EVENT, "event"), event_ids)
    flags = {k: v[order] for k, v in flags.items()}
    if np.any(flags["input_mode"] != 1):
        raise ValueError("primary reader requires an explicit production export")
    if not np.all(flags["primary_present"] & flags["transport_particles_present"] & flags["showers_present"]):
        raise ValueError("production export has missing required products")
    if np.any(flags["unresolved_links"] != 0):
        raise ValueError("production export has unresolved persistent links")

    def event_rows(table, expected, label):
        rows = join(event_ids, keys(table, EVENT))
        if not np.array_equal(np.bincount(rows, minlength=len(event_ids)), expected):
            raise ValueError(f"{label} row counts disagree with events")
        return rows

    particles = columns(source, "particles", IDENTITY + (
        "sim_valid", "pdg", "parent_valid", "parent_product_id", "parent_key",
        "generator_valid", "generator_product_id", "generator_key", "self_parent", "is_ce_endpoint"))
    particle_ids = _unique(particles, IDENTITY, "SimParticle")
    event_rows(particles, flags["n_particles"], "SimParticle")
    if not np.all(particles["sim_valid"]):
        raise ValueError("null exported SimParticle identity")
    index = np.arange(len(particle_ids))
    has_parent = particles["parent_valid"].astype(bool)
    parent = index.copy()
    target = {k: v[has_parent] for k, v in particles.items()}
    target.update(sim_product_id=target["parent_product_id"], sim_key=target["parent_key"])
    parent[has_parent] = join(particle_ids, keys(target, IDENTITY))
    resolved = ~has_parent
    jump = parent.copy()
    for _ in range(max(1, len(parent).bit_length())):
        resolved = resolved | resolved[jump]
        jump = jump[jump]
    if not np.all(resolved):
        raise ValueError("cyclic SimParticle ancestry")

    gen = columns(source, "genparticles", EVENT + ("product_id", "index"))
    gen_ids = _unique(gen, EVENT + ("product_id", "index"), "GenParticle")
    join(event_ids, keys(gen, EVENT))
    linked = particles["generator_valid"].astype(bool)
    gen_target = {k: v[linked] for k, v in particles.items()}
    gen_target.update(product_id=gen_target["generator_product_id"], index=gen_target["generator_key"])
    join(gen_ids, keys(gen_target, EVENT + ("product_id", "index")))

    primaries = columns(source, "primaries", IDENTITY + ("product_id", "index", "sim_valid"))
    _unique(primaries, EVENT + ("product_id", "index"), "PrimaryParticle row")
    _unique(primaries, IDENTITY, "primary pointer")
    primary_events = event_rows(primaries, flags["n_primaries"], "PrimaryParticle")
    if np.any(flags["n_primaries"] != 1):
        raise ValueError("expected exactly one production CE primary per stored event")
    if not np.all(primaries["sim_valid"]):
        raise ValueError("null production primary pointer")
    if np.any(primaries["index"] != 0) or not np.array_equal(
            primaries["product_id"], flags["primary_product_id"][primary_events]):
        raise ValueError("PrimaryParticle product or vector index mismatch")
    if not np.array_equal(primaries["sim_product_id"], flags["transport_product_id"][primary_events]):
        raise ValueError("primary does not reference the configured transport product")
    primary_rows = join(particle_ids, keys(primaries, IDENTITY))
    if not np.all((particles["pdg"][primary_rows] == 11) & particles["is_ce_endpoint"][primary_rows]):
        raise ValueError("primary is not a mu2eCeMinusEndpoint electron")
    continuation = particles["self_parent"].astype(bool)
    if np.any(continuation & ~has_parent) or np.any(continuation[primary_rows]):
        raise ValueError("invalid physical-continuation marker")
    if np.any(particles["pdg"][continuation] != particles["pdg"][parent[continuation]]):
        raise ValueError("physical continuation changes particle species")
    physical = _ancestors(np.where(continuation, parent, index))
    direct = np.isin(physical, primary_rows)
    ancestry = parent.copy()
    ancestry[primary_rows] = primary_rows
    descendant = np.isin(_ancestors(ancestry), primary_rows)

    step_ids = columns(source, "steps", IDENTITY + ("sim_valid",))
    event_rows(step_ids, flags["n_steps"], "step")
    shower_ids = columns(source, "showers", IDENTITY + ("sim_valid",))
    event_rows(shower_ids, events["n_showers"], "shower")
    for table in (step_ids, shower_ids):
        if not np.all(table["sim_valid"]):
            raise ValueError("null step or shower particle pointer")
        join(particle_ids, keys(table, IDENTITY))
    shower_particles = join(particle_ids, keys(shower_ids, IDENTITY))
    audit = dict(direct_ce_particles=int(np.sum(direct)),
                 ce_descendant_css_rows=int(np.sum(descendant[shower_particles])),
                 other_ancestry_css_rows=int(np.sum(~descendant[shower_particles])))
    return particle_ids[direct], audit
