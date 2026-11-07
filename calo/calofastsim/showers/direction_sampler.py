"""Opt-in capped-direction K8 draws; no default changes or outcome selection."""
from __future__ import annotations

import numpy as np

from . import support_audit as sa
from .direction_support import ANGLE_DEG, K, RADIUS_MM, compatible, local_direction, nearest
from .library import frame


def enumerate_pools(library, entries):
    """Return the full support-audit schema plus direction_count/used/fallback.

    Within the original chosen cell and entry disk, require matching radial
    and tangent signs (zero nonnegative), angle <= 10 degrees and radius
    distance <= 25 mm. At least eight qualifying rows enable nearest-eight
    selection, with radius/library-row ties and final library-row ordering.
    Otherwise keep original K8, including its unsupported and fallback flags.
    This function consumes no randomness and never examines shower outcomes.
    """
    pools = sa.enumerate_pools(library, entries)
    entries = sa._entries(entries)
    query = local_direction(library, entries)
    donor = local_direction(library, {k[6:]: v for k, v in library.items() if k.startswith("entry_")})
    count = np.zeros(len(query), np.int64)
    used = np.zeros(len(query), bool)
    for cell in np.unique(pools["chosen_cell"][pools["supported"]]):
        target = np.flatnonzero(pools["chosen_cell"] == cell)
        rows = np.flatnonzero(pools["library_cell"] == cell)
        distance = np.abs(entries["radius"][target, None] - library["entry_radius"][rows][None, :])
        _, good, _ = compatible(query[target, None, :], donor[None, rows, :])
        good &= distance <= RADIUS_MM
        count[target] = good.sum(axis=1)
        take = count[target] >= K
        used[target] = take
        if np.any(take):
            pools["eligible"][target[take]] = nearest(rows, distance[take], good[take])
            pools["size"][target[take]] = K
    pools.update(direction_count=count, direction_used=used,
                 direction_fallback=pools["supported"] & ~used)
    return pools


def sample(library, entries, rng):
    """Draw whole both-disk events with the original per-disk uniform schedule.

    The base output schema and disk-axis azimuth arithmetic match
    library.sample(radius_neighbors=8, orientation="disk"); three additional
    per-query fields are direction_count, direction_used, direction_fallback.
    Empty/weak donors, relative times and both deposited energies are copied
    without filtering, rescaling or incident-frame query rotation.
    """
    entries = {k: np.asarray(v) for k, v in entries.items()}
    pools = enumerate_pools(library, entries)
    n = len(entries["energy"])
    donors = np.full(n, -1, np.int64)
    for disk in (0, 1):
        where = np.flatnonzero((entries["disk"] == disk) & pools["supported"])
        if not len(where):
            continue
        uniform = rng.random(len(where))
        slot = (uniform * pools["size"][where]).astype(np.int64)
        donors[where] = pools["eligible"][where, slot]
    supported = pools["supported"]
    length = np.zeros(n, dtype=np.int64)
    length[supported] = np.diff(library["offsets"])[donors[supported]]
    offsets = np.r_[0, np.cumsum(length)]
    row = np.repeat(np.arange(n), length)
    source = library["offsets"][donors[row]] + np.arange(offsets[-1]) - np.repeat(offsets[:-1], length)
    result = {k.removeprefix("step_"): v[source] for k, v in library.items() if k.startswith("step_")}
    relative = result.pop("relative_position")
    # Preserve the original disk-mode arithmetic, including the donor-frame unpacking.
    delta = np.einsum("nij,nj->ni", frame(library["entry_direction"])[donors[row]], relative)
    origin = np.column_stack([library["geometry_disk_origin_" + a] for a in "xyz"])
    slot = np.searchsorted(library["geometry_disk_disk"], np.asarray(entries["disk"])[row])
    source_xy = library["entry_position"][donors[row], :2] - origin[slot, :2]
    target_xy = entries["position"][row, :2] - origin[slot, :2]
    angle = np.arctan2(target_xy[:, 1], target_xy[:, 0]) - np.arctan2(source_xy[:, 1], source_xy[:, 0])
    c, s = np.cos(angle), np.sin(angle)
    x, y = delta[:, 0].copy(), delta[:, 1].copy()
    delta[:, 0], delta[:, 1] = c * x - s * y, s * x + c * y
    result["position"] = entries["position"][row] + delta
    result["time"] = entries["time"][row] + result.pop("relative_time")
    result.update(entry=row, offsets=offsets, donor=donors, supported=supported, fallback=pools["fallback"],
                  radius_pool_size=pools["size"], bin_pool_size=pools["full_size"],
                  **{key: pools[key] for key in ("direction_count", "direction_used", "direction_fallback")})
    return result
