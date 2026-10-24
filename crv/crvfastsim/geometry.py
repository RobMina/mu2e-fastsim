"""CRV bar geometry for the fast simulation, built from Offline's own geometry inputs.

Two independent sources, compared by the V0 check (crvfastsim v0check):
  * `from_offline_config`: re-implements GeometryService/src/CosmicRayShieldMaker.cc (makeCRVBars, makeSingleSector)
    on the Offline `crv_counters_*.txt` parameters, so a module move is a parameter edit;
  * `from_gdml`: global bar placements read from a Geant4 GDML export of the same world.
Bar index order is CosmicRayShieldMaker's creation order: sector -> module -> layer -> counter.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

OFFLINE_ROOTS = ("/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v13_39_00",
                 os.path.expanduser("~/mu2e/remote/off"))


def find_counters(path):
    """Resolve an Offline-relative counters file (e.g. 'Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt').

    Absolute or existing paths are returned as is; otherwise MU2E_SEARCH_PATH, then the pinned CVMFS Offline release
    and the laptop Offline mirror are searched.
    """
    if os.path.exists(path):
        return path
    rel = path[len("Offline/"):] if path.startswith("Offline/") else path
    roots = [p for p in os.environ.get("MU2E_SEARCH_PATH", "").split(":") if p] + list(OFFLINE_ROOTS)
    for r in roots:
        for cand in (os.path.join(r, path), os.path.join(r, "Offline", rel)):
            if os.path.exists(cand):
                return cand
    got = _fetch_public(rel)
    if got:
        return got
    raise FileNotFoundError(f"counters file {path} not found under MU2E_SEARCH_PATH or {OFFLINE_ROOTS}")


OFFLINE_TAG = "v13_39_00"


def _fetch_public(rel):
    """Sites without a CVMFS mount (UVA, Polaris jobs: the image holds only the extracted counters file): fetch a public
    Offline geometry text file at OFFLINE_TAG into $CRVFS_OFFLINE_CACHE (default ~/.cache/crvfastsim/offline), from
    the CVMFS Stratum-1 over HTTP (mu2efastsim.cvmfsget, the route the jobs already use for the tables), else from
    GitHub; None if it is not a Mu2eG4/geom text file or both fetches fail."""
    if not (rel.startswith("Mu2eG4/geom/") and rel.endswith(".txt")) or os.environ.get("CRVFS_NO_FETCH"):
        return None
    root = os.environ.get("CRVFS_OFFLINE_CACHE", os.path.expanduser("~/.cache/crvfastsim/offline"))
    dest = os.path.join(root, OFFLINE_TAG, "Offline", rel)
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        from mu2efastsim import cvmfsget
        cvmfsget.Repo("mu2e.opensciencegrid.org").fetch(f"/Musings/Offline/{OFFLINE_TAG}/Offline/{rel}", dest)
        return dest
    except Exception:   # noqa: BLE001  (any failure: try GitHub)
        pass
    import urllib.request
    url = f"https://raw.githubusercontent.com/Mu2e/Offline/{OFFLINE_TAG}/{rel}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except OSError:
        return None
    tmp = f"{dest}.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return dest


_DECL = re.compile(r"^\s*(string|int|double|bool|vector<\s*(?:string|int|double)\s*>)\s+([\w.]+)\s*=\s*(.*?);", re.S)


def parse_simpleconfig(text: str) -> dict:
    """Parse Mu2e SimpleConfig declarations (no #include handling: the counters file is self-contained)."""
    text = re.sub(r"//[^\n]*", "", text)
    out = {}
    for stmt in text.split(";"):
        m = _DECL.match(stmt + ";")
        if not m:
            continue
        typ, key, val = m.group(1), m.group(2), m.group(3).strip()
        if typ.startswith("vector"):
            items = [v.strip().strip('"') for v in val.strip("{}").split(",") if v.strip()]
            out[key] = [float(v) for v in items] if "string" not in typ else items
        elif typ == "string":
            out[key] = val.strip('"')
        elif typ == "bool":
            out[key] = val.lower() == "true"
        else:
            out[key] = float(val)
    return out


@dataclass
class CRVGeometry:
    sector_names: list
    sector: np.ndarray      # [nbars] int
    module: np.ndarray
    layer: np.ndarray
    counter: np.ndarray
    center: np.ndarray      # [nbars, 3] Mu2e coordinates, mm
    halflen: np.ndarray     # [nbars, 3] half lengths along Mu2e x, y, z, mm
    local_to_world: np.ndarray  # [nbars, 3]: Mu2e axis of (thickness, width, length)

    @property
    def nbars(self) -> int:
        return len(self.sector)

    def bar_index(self, sector, module, layer, counter):
        m = (self.sector == sector) & (self.module == module) & (self.layer == layer) & (self.counter == counter)
        return int(np.flatnonzero(m)[0])


def from_offline_config(path: str) -> CRVGeometry:
    c = parse_simpleconfig(open(path).read())
    names = c["crs.sectorNames"]
    n_layers = int(c["crs.nLayers"])
    thick, width = c["crs.scintillatorBarThickness"], c["crs.scintillatorBarWidth"]
    gap_small, gap_large, gap_mod = c["crs.gapSmall"], c["crs.gapLarge"], c["crs.gapBetweenModules"]
    gaps_layers = c["crs.gapBetweenLayers"]
    offset = c["crs.layerOffset"]
    rows = {k: [] for k in ("sector", "module", "layer", "counter", "center", "halflen", "l2w")}
    for isec, name in enumerate(names):
        layer_dir = np.array(c[f"crs.layerDirection{name}"])
        gap_dir = np.array(c[f"crs.gapDirection{name}"])
        off_dir = np.array(c[f"crs.offsetDirection{name}"])
        counters_only = bool(c.get(f"crs.countersOnly{name}", False))
        length = c[f"crs.scintillatorBarLength{name}"]
        nmod, ncnt = int(c[f"crs.nModules{name}"]), int(c[f"crs.nCountersPerModule{name}"])
        first = np.array(c[f"crs.firstCounter{name}"])
        dl = dw = dt = 0
        for d in range(3):
            if gap_dir[d] == 0 and layer_dir[d] == 0:
                dl = d
            if gap_dir[d] != 0:
                dw = d
            if layer_dir[d] != 0:
                dt = d
        half = np.zeros(3)
        half[dl], half[dt], half[dw] = length / 2, thick / 2, width / 2
        layer_off = [np.zeros(3)]
        for j in range(1, n_layers):
            v = layer_off[-1] + thick * layer_dir
            if not counters_only:
                v = v + gaps_layers[j - 1] * layer_dir
            layer_off.append(v + offset * off_dir)
        v_small = (width + gap_small) * gap_dir
        v_large = (width + gap_large) * gap_dir
        v_mod = (width + gap_mod) * gap_dir
        for im in range(nmod):
            for il in range(n_layers):
                for ic in range(ncnt):
                    pos = first + layer_off[il]
                    pos = pos + im * ((ncnt // 2 - 1) * v_large + (ncnt // 2) * v_small) + im * v_mod
                    pos = pos + (ic // 2) * v_large + ((ic + 1) // 2) * v_small
                    for k, v in (("sector", isec), ("module", im), ("layer", il), ("counter", ic),
                                 ("center", pos), ("halflen", half), ("l2w", (dt, dw, dl))):
                        rows[k].append(v)
    return CRVGeometry(names, *(np.array(rows[k]) for k in ("sector", "module", "layer", "counter")),
                       np.array(rows["center"], float), np.array(rows["halflen"], float), np.array(rows["l2w"], int))


_BAR = re.compile(r"^CRSScintillatorBar_(\d+)_(\d+)_(\d+)_(\d+)(?:0x[0-9a-f]+)?$")


def _strip_ptr(name: str) -> str:
    return re.sub(r"0x[0-9a-f]+$", "", name)


def gdml_global_placements(path: str):
    """Walk a GDML file from the world volume; yield (physvol name, logical volume name, global position [mm]).

    Only translations are composed: the Mu2e CRV and the extracted building have no rotated placements on the path to
    a bar (checked: any rotation on that path raises).
    """
    root = ET.parse(path).getroot()
    structure = root.find("structure")
    vols = {v.get("name"): v for v in structure.findall("volume")}
    world = root.find("setup/world").get("ref")
    defines = {p.get("name"): p for p in root.findall("define/position")}

    def pos_of(pv):
        p = pv.find("position")
        if p is None:
            pr = pv.find("positionref")
            p = defines.get(pr.get("ref")) if pr is not None else None
        if p is None:
            return np.zeros(3)
        scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}[p.get("unit", "mm")]
        return scale * np.array([float(p.get(a, 0)) for a in "xyz"])

    def has_rotation(pv):
        r = pv.find("rotation")
        return pv.find("rotationref") is not None or (r is not None and any(float(r.get(a, 0)) for a in "xyz"))

    stack = [(world, np.zeros(3), False)]
    while stack:
        lv, origin, rotated = stack.pop()
        for pv in vols[lv].findall("physvol"):
            child = pv.find("volumeref").get("ref")
            gpos = origin + pos_of(pv)
            rot = rotated or has_rotation(pv)
            yield _strip_ptr(pv.get("name")), child, gpos, rot
            if child in vols:
                stack.append((child, gpos, rot))


def gdml_box_halflengths(path: str) -> dict:
    root = ET.parse(path).getroot()
    out = {}
    for b in root.findall("solids/box"):
        scale = {"mm": 1.0, "cm": 10.0, "m": 1000.0}[b.get("lunit", "mm")]
        out[b.get("name")] = 0.5 * scale * np.array([float(b.get(a)) for a in ("x", "y", "z")])
    return out


def from_gdml(path: str, sector_names: list) -> tuple[CRVGeometry, int]:
    """Bars from a GDML export, in Geant4 world coordinates. Returns (geometry, number of rotated bars)."""
    root = ET.parse(path).getroot()
    lv_solid = {v.get("name"): v.find("solidref").get("ref") for v in root.find("structure").findall("volume")}
    boxes = gdml_box_halflengths(path)
    rows = []
    nrot = 0
    for name, lv, gpos, rot in gdml_global_placements(path):
        m = _BAR.match(name)
        if not m:
            continue
        nrot += int(rot)
        s, mo, la, co = map(int, m.groups())
        rows.append((s, mo, la, co, gpos, boxes[lv_solid[lv]]))
    rows.sort(key=lambda r: r[:4])
    a = list(zip(*rows))
    half = np.array(a[5])
    l2w = np.stack([np.argmin(half, 1), np.zeros(len(half), int), np.argmax(half, 1)], 1)
    l2w[:, 1] = 3 - l2w[:, 0] - l2w[:, 2]
    return CRVGeometry(sector_names, *(np.array(x) for x in a[:4]), np.array(a[4]), half, l2w), nrot


def parse_g4_printout(log_path: str) -> dict:
    """constructCRV.cc verbosity-5 lines: 'CRSScintillatorBar_s_m_l_c (x,y,z)' in Mu2e coordinates, ~6 digits."""
    pat = re.compile(r"^CRSScintillatorBar_(\d+)_(\d+)_(\d+)_(\d+) \(([^,]+),([^,]+),([^)]+)\)")
    out = {}
    for line in open(log_path):
        m = pat.match(line)
        if m:
            out[tuple(map(int, m.groups()[:4]))] = np.array([float(v) for v in m.groups()[4:]])
    return out
