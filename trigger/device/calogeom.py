"""Calorimeter geometry for the calo trigger stage, built as Offline builds it (no table export).

Offline v13_41_01, geometry run1_a (`calorimeter_CsI_v2.txt`):

- `Disk::fillCrystals`: the crystal map is a `SquareShiftMapper` with cell size 2 (crystalXYLength / 2 + wrapper),
  up to `int(1.5 diskCrystalROut / cellSize)` rings. Map slots are visited in index order; a slot takes the first
  crystal of `crystalPos.txt` (this disk, file order) within half a cell of the slot centre in x and y. Crystal ids
  follow that slot order; disk 1 ids are offset by disk 0's count.
- `localPosition` = (x, y, 0) from the file (read as float) + (0, 0, disp); `position` (Mu2e frame) =
  disk origin + localPosition (identity disk rotation).
- `neighbors()` / `nextNeighbors()`: `SquareShiftMapper::neighbors(slot, 1 or 2)`, in that order, keeping valid
  slots.
- `mu2eToDiskFF(p)` = (p - origin) - (0, 0, disp).

`data/calo_crystal_pos.txt` is Offline's `CalorimeterGeom/data/crystalPos.txt` at that release.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

CRYSTAL_FILE = Path(__file__).with_name("data") / "calo_crystal_pos.txt"

# calorimeter_CsI_v2.txt and geom_run1_a.txt (mu2e.solenoidOffset); every length in mm.
CALO_GEOM = dict(
    solenoidOffset=3904.0, caloMotherZ0=11842.0, caloMotherZ1=13220.0, vdThickness=0.02,
    diskCrystalROut=660.0, diskZMotherShift=(0.0, 700.0),
    crystalXYLength=34.0, crystalZLength=200.0, crystalCapZLength=2.0, wrapperThickness=0.150,
    FPCarbonZLength=3.0, FPFoamZLength=21.75, FPCoolPipeRadius=16.85, pipeRadius=4.75,
    BPHoleZLength=15.0, FEEZLength=58.0, FEEBoxThickness=0.7, BPPipeZOffset=43.0, BPPipeRadiusHigh=13.45,
    FEBToDiskZOffset=35.0, caphri=(582, 609, 610, 637),
)
N_PER_DISK = 674

_STEP = ((1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0))


def _lk(index):
    """SquareShiftMapper::lk."""
    if index == 0:
        return 0, 0
    ring = int(0.5 + math.sqrt(0.25 + (index - 1) / 3.0))
    seg = (index - 3 * ring * (ring - 1) - 1) // ring
    pos = (index - 3 * ring * (ring - 1) - 1) % ring
    if seg >= len(_STEP):
        return 0, 0
    l, k = pos * _STEP[seg][0], -ring + pos * _STEP[seg][1]
    for i in range(seg):
        l += ring * _STEP[i][0]
        k += ring * _STEP[i][1]
    return l, k


def _ring(l, k):
    if l * k > 0:
        return max(abs(l), abs(k))
    return abs(l - k)


def _index(l, k):
    """SquareShiftMapper::index."""
    if not l and not k:
        return 0
    r = _ring(l, k)
    pos = 3 * r * (r - 1) + 1
    if k == -r and l == 0:
        return pos
    if l == r:
        return pos + (r + k)
    if l == -r:
        return pos + (4 * r + abs(k))
    if k == r:
        return pos + (3 * r - l)
    if k == -r:
        return pos + (6 * r - abs(l))
    if l > k:
        return pos + l
    return pos + 3 * r + abs(l)


def _neighbor_slots(index, level):
    l, k = _lk(index)
    k -= level
    out = []
    for dl, dk in _STEP:
        for _ in range(level):
            l, k = l + dl, k + dk
            out.append(_index(l, k))
    return out


def _disk_z(g):
    """Disk origin z (Mu2e) and disp (disk origin to crystal origin), with DiskCalorimeterMaker's operations."""
    crystal_half_z = g["crystalZLength"] / 2.0
    cap_half_z = g["crystalCapZLength"] / 2.0
    fp_half_z = (g["FPCarbonZLength"] + g["FPFoamZLength"] - g["pipeRadius"] + g["FPCoolPipeRadius"]) / 2.0
    bp_half_z = (g["BPHoleZLength"] / 2.0 + g["FEEZLength"] / 2.0 + 2.0 * g["FEEBoxThickness"]
                 + g["BPPipeZOffset"] / 2.0 + g["BPPipeRadiusHigh"])
    case_half_z = crystal_half_z + cap_half_z
    disk_half_z = fp_half_z + case_half_z + bp_half_z + g["vdThickness"]
    crate_to_disk = g["FEBToDiskZOffset"] + g["vdThickness"]
    disp = (-disk_half_z + g["vdThickness"] + 2 * fp_half_z + case_half_z - crystal_half_z + cap_half_z)
    mother_half_z = (g["caloMotherZ1"] - g["caloMotherZ0"]) / 2.0
    calo_z = 0.5 * (g["caloMotherZ0"] + g["caloMotherZ1"])
    origin_z = [calo_z + (-mother_half_z + disk_half_z + sep + crate_to_disk) for sep in g["diskZMotherShift"]]
    return origin_z, disp


@lru_cache(maxsize=None)
def build(crystal_file=str(CRYSTAL_FILE), **overrides):
    """Arrays per crystal id (both disks): local x/y, disk, Mu2e position, FF position, neighbours (-1 padded)."""
    g = {**CALO_GEOM, **overrides}
    cell = 2.0 * (g["crystalXYLength"] / 2.0 + 2.0 * (g["wrapperThickness"] / 2.0))
    rings = int(1.5 * g["diskCrystalROut"] / cell)
    nmap = 3 * rings * (rings + 1) + 1
    rows = [line.split() for line in Path(crystal_file).read_text().splitlines() if line.strip()]
    origin_z, disp = _disk_z(g)
    out = {k: [] for k in ("local_x", "local_y", "disk", "slot")}
    to_id = []
    for disk in (0, 1):
        pos = [(np.float32(r[2]), np.float32(r[3])) for r in rows if int(r[0]) == disk]
        if [int(r[1]) for r in rows if int(r[0]) == disk] != list(range(len(pos))):
            raise ValueError("crystal file indices out of order")
        px = np.array([float(p[0]) for p in pos])
        py = np.array([float(p[1]) for p in pos])
        slot_to = np.full(nmap, -1, np.int64)
        n = 0
        for slot in range(nmap):
            l, k = _lk(slot)
            ix, iy = cell * ((l + k) / 2.0), cell * float(l - k)
            hit = np.flatnonzero((np.abs(px - ix) < 0.5 * cell) & (np.abs(py - iy) < 0.5 * cell))
            if hit.size == 0:
                continue
            j = hit[0]
            slot_to[slot] = n
            out["local_x"].append(px[j])
            out["local_y"].append(py[j])
            out["disk"].append(disk)
            out["slot"].append(slot)
            n += 1
        if n != N_PER_DISK:
            raise ValueError(f"disk {disk}: {n} crystals, expected {N_PER_DISK}")
        to_id.append(slot_to)
    geo = {k: np.asarray(v) for k, v in out.items()}
    n_all = geo["disk"].size
    offset = np.where(geo["disk"] == 0, 0, N_PER_DISK)
    nb = np.full((n_all, 6), -1, np.int64)
    nn = np.full((n_all, 12), -1, np.int64)
    for c in range(n_all):
        st = to_id[geo["disk"][c]]
        for arr, level in ((nb, 1), (nn, 2)):
            ids = [st[s] for s in _neighbor_slots(int(geo["slot"][c]), level) if s < nmap and st[s] >= 0]
            arr[c, :len(ids)] = np.asarray(ids, np.int64) + offset[c]
    origin = np.array([[-g["solenoidOffset"], 0.0, z] for z in origin_z])
    o = origin[geo["disk"]]
    pos = np.stack([o[:, 0] + geo["local_x"], o[:, 1] + geo["local_y"], o[:, 2] + disp], axis=1)
    ff = np.stack([pos[:, 0] - o[:, 0], pos[:, 1] - o[:, 1], (pos[:, 2] - o[:, 2]) - disp], axis=1)
    geo.update(position=pos, ff=ff, neighbors=nb, next_neighbors=nn, origin=origin, disp=disp,
               front_face=origin + np.array([0.0, 0.0, disp]),
               caphri=np.isin(np.arange(n_all), g["caphri"]),
               crystal_size=g["crystalXYLength"] + 2.0 * g["wrapperThickness"])
    for v in geo.values():
        if isinstance(v, np.ndarray):
            v.setflags(write=False)
    return geo
