"""Selected vacuum-mother/leaf point locations, not native or swept navigation.

Strict interiors retain exact exported material and placement identities. Zero
clearances, overlapping roots/leaves and out-of-scope points stay unresolved.
Vacuum leaves coalesce into their mother. Positive local clearance is a
conservative ball only under the supplied complete-cell/1-Lipschitz contracts;
no crossing, absorption, material physics or whole-world coverage is inferred.
"""

from dataclasses import dataclass

import numpy as np

from . import tsnav

OUTSIDE_SCOPE = 0
VACUUM = 1
MATERIAL = 2
BOUNDARY = 3
OVERLAP = 4
STATUS = {OUTSIDE_SCOPE: "outside_scope", VACUUM: "vacuum", MATERIAL: "material",
          BOUNDARY: "boundary_unresolved", OVERLAP: "overlap_unresolved"}


@dataclass(frozen=True)
class Volume:
    solid: object
    material: str
    physical: str | None
    root: str
    vacuum: bool
    instance: int | None = None
    logical: str | None = None
    instance_path: tuple = ()

    def __post_init__(self):
        tsnav._child(self.solid)
        for key in ("material", "root"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty exact reference")
        for key in ("physical", "logical"):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{key} must be a nonempty name or None")
        if not isinstance(self.vacuum, bool):
            raise ValueError("vacuum must be boolean")
        path = tuple(self.instance_path)
        if any(type(i) is not int or i <= 0 for i in path):
            raise ValueError("instance path requires positive integer identifiers")
        if self.instance is not None:
            if type(self.instance) is not int or self.instance <= 0:
                raise ValueError("instance requires a positive integer identifier")
            if not path or path[-1] != self.instance:
                raise ValueError("instance must end its complete source path")
        elif path:
            raise ValueError("instance path requires an instance")
        object.__setattr__(self, "instance_path", path)


@dataclass(frozen=True)
class Cell:
    mother: Volume
    daughters: tuple = ()

    def __post_init__(self):
        if not isinstance(self.mother, Volume) or not self.mother.vacuum:
            raise ValueError("cell requires a vacuum mother")
        daughters = tuple(self.daughters)
        if any(not isinstance(v, Volume) or v.root != self.mother.root for v in daughters):
            raise ValueError("leaf volumes must belong to their mother root")
        object.__setattr__(self, "daughters", daughters)


@dataclass(frozen=True)
class Location:
    status: np.ndarray
    volume_index: np.ndarray
    boundary_index: np.ndarray
    clearance: np.ndarray
    root_count: np.ndarray
    material_count: np.ndarray
    boundary_count: np.ndarray


@dataclass(frozen=True)
class Partition:
    cells: tuple

    def __post_init__(self):
        cells = tuple(self.cells)
        if not cells or any(not isinstance(c, Cell) for c in cells):
            raise ValueError("nonempty explicit cells required")
        if len({c.mother.root for c in cells}) != len(cells):
            raise ValueError("duplicate cell root")
        object.__setattr__(self, "cells", cells)

    @property
    def volumes(self):
        return tuple(v for cell in self.cells for v in (cell.mother, *cell.daughters))

    def locate(self, points):
        """Classify strict point interiors; indices refer to ``volumes`` order.

        A unique zero-valued root/material leaf gets boundary_index, not an owner.
        Local clearance is zero for every unresolved result, never a crossing step.
        Geometry loops are over fixed volumes; all particle queries are batched.
        """
        p = tsnav._real(points, "points")
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("points must have shape (N,3)")
        n = len(p)
        roots = np.zeros(n, dtype=np.int64)
        materials = np.zeros(n, dtype=np.int64)
        boundaries = np.zeros(n, dtype=np.int64)
        owner = np.full(n, -1, dtype=np.int64)
        boundary = np.full(n, -1, dtype=np.int64)
        local = np.zeros(n)
        root_guard = np.full(n, np.inf)
        offset = 0
        for cell in self.cells:
            m = tsnav._query(cell.mother.solid, p)
            root_guard = np.minimum(root_guard, np.abs(m))
            interior, represented = m > 0, m >= 0
            roots += interior
            at = m == 0
            boundaries += at
            boundary[at] = offset
            obstacles = [(offset + j + 1, tsnav._query(v.solid, p))
                         for j, v in enumerate(cell.daughters) if not v.vacuum]
            vacuum_clearance = m.copy()
            for index, d in obstacles:
                vacuum_clearance = np.minimum(vacuum_clearance, -d)
                inside = represented & (d > 0)
                materials += inside
                owner[inside] = index
                at = represented & (d == 0)
                boundaries += at
                boundary[at] = index
            free = interior & (vacuum_clearance > 0)
            owner[free] = offset
            local[free] = vacuum_clearance[free]
            for index, d in obstacles:
                inside = interior & (d > 0)
                clearance = np.minimum(m, d)
                for other_index, other in obstacles:
                    if index != other_index:
                        clearance = np.minimum(clearance, -other)
                local[inside] = clearance[inside]
            offset += 1 + len(cell.daughters)
        status = np.full(n, OUTSIDE_SCOPE, dtype=np.int8)
        status[roots == 1] = VACUUM
        status[(roots == 1) & (materials == 1)] = MATERIAL
        status[boundaries > 0] = BOUNDARY
        status[(roots > 1) | (materials > 1)] = OVERLAP
        resolved = (status == VACUUM) | (status == MATERIAL)
        local = np.minimum(local, root_guard)
        owner[~resolved] = -1
        boundary[boundaries != 1] = -1
        local[~resolved] = 0
        if np.any(resolved & ((owner < 0) | (local <= 0))):
            raise ValueError("contradictory selected-cell clearance classification")
        return Location(status, owner, boundary, local, roots, materials, boundaries)
