"""Fast conversion-electron files in the EdepAna format (opt-in): `EDepAna/tree` as Offline's EdepAna writes it.

Offline ``Analyses/src/EdepAna_module.cc`` (v13_39_00 and later) writes one TTree row per stored event of a CeEndpoint
sample; the design studies' ``edep`` and ``approx_ce_sensitivity`` analyses read it (the event selection, the signal
shape at the tracker front, the CE acceptance per generated event from ``ngen``). This module writes the same tree
from the fast chain, so those analyses run unchanged on fast output:

1. births: conversion electrons (104.97 MeV/c, isotropic, CeEndpoint decay time) resampled from the mu- stops of a
   stop table (``beamfastsim point --stops-out``; or ``point`` here runs the beam stage first);
2. tracker: ``cetracker.DeviceChain`` on one device (cechain_jax to the tracker entry plane, fieldsteps_jax
   StrawGasSteps with the downstream albedo; delta rays opt-in, ``deltas``);
3. calorimeter (``calo="library+edge"``, the default): the electron's first calorimeter VD crossing from the tracker
   exit, in vacuum (``first_contact``): a disk's front plane inside its annulus (r 336-719 mm), or its inner-edge
   cylinder (r = 335 mm) along the disk. A front crossing draws a whole-event Geant4 shower pattern of the calorimeter
   shower library there (both disks' CaloShowerSteps, G4 and Birks-visible energies; ``radius_neighbors=8``,
   ``calofastsim.showers``); an inner-edge crossing draws the event energies of a Geant4 electron that entered the
   same disk's inner edge at the same depth and helix maximum radius (``EdgeEnergies``: the inner-edge table); a
   contact whose helix maximum radius stays below 400 mm (it cannot reach the crystals, r >= 374 mm; no such electron
   passes the storage filter in Geant4) deposits nothing. Options:
   ``"library+wall"`` (the cesob wall proxy for electrons that can reach the inner wall, front-entry patterns at its
   first crystal contact) and ``"library"`` (the vacuum disk-front crossing of the device hand-off only);
4. storage: the CeEndpoint output filter (``DetectorStepFilter`` of Production ``JobConfig/primary/prolog.fcl``): an
   event is stored when its electron makes at least 12 StrawGasSteps with ionising energy > 1 eV and |p| > 50 MeV/c,
   or its visible calorimeter energy reaches 45 MeV;
5. one tree row per stored event, with the EdepAna branch names, types (float32, int32, int64 ``ngen``), units (mm,
   MeV, MeV/c; Mu2e frame) and the running generated-event count; one subrun per batch.

Beside the ROOT file: ``<file>.edep.json`` (the run record the ``edep`` analysis writes: one full run, ``max_events``
null), ``edep_summary.txt`` (the block EdepAna prints at endJob, the same arithmetic, so the ``edep`` summary parser
reads it) and ``<file>.fastsim.json`` (provenance; also stored in the file as ``EDepAna/fastsim_provenance``).
The file is a TTree (uproot), not an RNTuple: EdepAna writes a TTree and the readers use the TTree interface.

  python3 -m beamfastsim.edepana ce --stops STOPS.npz [--geom G] --electrons N --seed S --dtype float64 \\
      --library LIB.npz --edge-library EDGE.npz --gdml GDML [--offline-root R --maps DSMap TSdMap] --out DIR
  python3 -m beamfastsim.edepana point --g4 TABLE --z-up 3900 [--geom G] --muons 0 --electrons N ... --out DIR

Not modelled (they change event_trk_edep and the event_calo_edep shapes, not the start or front branches): delta
rays and conversions in the tracker unless ``deltas`` is set; inner-edge entries carry the event energies of the
Geant4 table (nominal target, EM option 4), not a shower of their own kinematics; the tracker transport is first
passage plus the albedo library.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

TREE = "EDepAna/tree"
PROVENANCE = "EDepAna/fastsim_provenance"
RUN_RECORD_SUFFIX = ".edep.json"
FILE_NAME = "nts.mu2efastsim.edep.root"
SUMMARY_NAME = "edep_summary.txt"
ME = 0.51099895                       # electron mass [MeV]
PDG_E = 11
CALO_MODES = ("library+edge", "library+wall", "library")
CALO_MODE = "library+edge"
EDGE_R = 335.0                        # inner-edge surface of the disks (the calorimeter VDs' cylinder) [mm]
FRONT_RMIN, FRONT_RMAX = 336.0, 719.0 # disk front-face annulus of the calorimeter VDs [mm]
DISK_LENGTH = 384.59                  # front VD to back VD of a disk [mm]
EDGE_BIN_MM = 25.0                    # depth bins of the inner-edge energy table
EDGE_RMAX_EDGES = (450.0, 520.0)      # helix maximum-radius bins of the inner-edge table (below, between, above) [mm]
EDGE_MIN_ROWS = 20                    # a sparser bin borrows its nearest depth bin of the same disk and radius bin
RMAX_MIN = 400.0                      # contacts of helices whose maximum radius (1 T) stays below this deposit nothing
B_REF = 1.0                           # [T] the uniform field of the maximum-radius variable (binning, not transport)
RADIUS_NEIGHBORS = 8                  # the W7 production setting of the shower-library draw
WALL_MARGIN_MM = 10.0                 # helices reaching within this of the wall's inner radius go through the proxy
FRONT_CONTACT_MM = 3.0                # a proxy contact this close behind a crystal front face is a front entry

# DetectorStepFilter (Production JobConfig/primary/prolog.fcl PrimaryFilter, used by CeEndpoint.fcl)
FILTER_MIN_TRK_STEPS = 12
FILTER_MIN_TRK_STEP_E = 1.0e-6        # MeV
FILTER_MIN_PART_MOM = 50.0            # MeV/c
FILTER_MIN_CALO_E = 45.0              # MeV, Birks-visible

# EdepAna's branches in its booking order: (name, dtype, per primary)
BRANCHES = (
    ("nprimaries", np.int32, False),
    ("primary_start_x", np.float32, True), ("primary_start_y", np.float32, True),
    ("primary_start_z", np.float32, True), ("primary_start_px", np.float32, True),
    ("primary_start_py", np.float32, True), ("primary_start_pz", np.float32, True),
    ("primary_start_e", np.float32, True), ("primary_start_m", np.float32, True),
    ("primary_start_pdg", np.int32, True), ("primary_calo_edep", np.float32, True),
    ("primary_calo_edep_vis", np.float32, True), ("primary_trk_front_p", np.float32, True),
    ("primary_trk_front_energy", np.float32, True),
    ("event_calo_edep", np.float32, False), ("event_calo_edep_vis", np.float32, False),
    ("event_trk_edep", np.float32, False), ("run", np.int32, False), ("subrun", np.int32, False),
    ("event", np.int32, False), ("weight", np.float32, False), ("ngen", np.int64, False),
)
PRIMARY = tuple(name for name, _, per in BRANCHES if per)
EVENT = tuple(name for name, _, per in BRANCHES if not per and name != "nprimaries")


# --- the file ----------------------------------------------------------------------------------------------------------

def detector_filter(trk_steps, calo_vis):
    """CeEndpoint's stored-event decision: StrawGasSteps of the electron with ionising energy > 1 eV and |p| > 50
    MeV/c (counted by the caller) at least 12, or the visible calorimeter energy at least 45 MeV."""
    return (np.asarray(trk_steps) >= FILTER_MIN_TRK_STEPS) | (np.asarray(calo_vis) >= FILTER_MIN_CALO_E)


def _check_rows(rows):
    n = None
    for name, _, _ in BRANCHES:
        if name == "nprimaries":
            continue
        if name not in rows:
            raise ValueError(f"edepana: missing column {name}")
        a = np.asarray(rows[name])
        if a.ndim != 1 or (n is not None and a.size != n):
            raise ValueError(f"edepana: column {name} is not one value per event")
        n = a.size
    ngen = np.asarray(rows["ngen"])
    if ngen.size > 1 and (np.diff(ngen) < 0).any():
        raise ValueError("edepana: ngen (the running generated-event count) must not fall")
    return n


def write_tree(path, rows, provenance=None):
    """Write EDepAna/tree (one primary per event: the conversion electron) to a new ROOT file. rows: one array per
    branch name of BRANCHES except nprimaries, one value per stored event (float64 or float32; cast to the EdepAna
    types)."""
    import awkward as ak
    import uproot
    n = _check_rows(rows)
    counts = np.ones(n, np.int64)
    prim = ak.zip({k: ak.unflatten(np.asarray(rows[k]).astype(dt), counts) for k, dt, per in BRANCHES if per})
    data = {"primary": prim}
    data.update({k: np.asarray(rows[k]).astype(dt) for k, dt, per in BRANCHES if not per and k != "nprimaries"})
    path = Path(path)
    with uproot.recreate(path) as f:
        f.mktree(TREE, {k: (v.type if k == "primary" else v.dtype) for k, v in data.items()},
                 title="Energy deposition data", counter_name=lambda counted: "nprimaries",
                 field_name=lambda outer, inner: inner)
        if n:
            f[TREE].extend(data)
        if provenance is not None:
            f[PROVENANCE] = json.dumps(provenance, sort_keys=True)
    return path


def summary(rows):
    """The numbers EdepAna prints at endJob, with its arithmetic (weighted sums in double; per gen event by the
    running ngen; the > 50 MeV count on the visible energy)."""
    w = np.asarray(rows["weight"], np.float64)
    ngen = int(np.max(rows["ngen"])) if np.size(rows["ngen"]) else 0
    total = float(w.sum())
    calo = float((w * np.asarray(rows["event_calo_edep"], np.float64)).sum())
    trk = float((w * np.asarray(rows["event_trk_edep"], np.float64)).sum())
    above = float(w[np.asarray(rows["event_calo_edep_vis"], np.float64) > 50.0].sum())
    return dict(n_events=total, n_gen_events=ngen, event_rate=total / ngen if ngen > 0 else -1.0,
                avg_calo_edep_per_event_mev=calo / total if total > 0 else -1.0,
                avg_calo_edep_per_gen_event_mev=calo / ngen if ngen > 0 else -1.0,
                n_events_calo_edep_above_50mev=above,
                avg_trk_edep_per_event_mev=trk / total if total > 0 else -1.0,
                avg_trk_edep_per_gen_event_mev=trk / ngen if ngen > 0 else -1.0)


def _g(v):
    """A double as std::cout prints it by default (6 significant digits, %g)."""
    return "%.6g" % v


def summary_text(m):
    """EdepAna's endJob summary block, formatted as the module prints it."""
    return ("EdepAna summary:\n"
            f"  Saw {_g(m['n_events'])} events ({int(m['n_gen_events'])} gen events) --> output rate = "
            f"{_g(m['event_rate'])} events / gen event\n"
            f"  Average calo energy deposition per event: {_g(m['avg_calo_edep_per_event_mev'])} MeV\n"
            f"  Average calo energy deposition per gen event: {_g(m['avg_calo_edep_per_gen_event_mev'])} MeV\n"
            f"  Events with calo Edep > 50 MeV: {_g(m['n_events_calo_edep_above_50mev'])}\n"
            f"  Average tracker energy deposition per event: {_g(m['avg_trk_edep_per_event_mev'])} MeV\n"
            f"  Average tracker energy deposition per gen event: {_g(m['avg_trk_edep_per_gen_event_mev'])} MeV\n")


def write_outputs(outdir, rows, provenance, name=FILE_NAME, max_events=None):
    """The file, its edep run record, the summary block and the provenance sidecar in outdir. Returns the paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = write_tree(outdir / name, rows, provenance)
    record = path.with_name(path.name + RUN_RECORD_SUFFIX)
    record.write_text(json.dumps({"max_events": max_events}) + "\n", encoding="utf-8")
    m = summary(rows)
    text = outdir / SUMMARY_NAME
    text.write_text(summary_text(m), encoding="utf-8")
    side = path.with_name(path.name + ".fastsim.json")
    side.write_text(json.dumps(dict(provenance, summary=m), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return dict(file=path, run_record=record, summary=text, provenance=side)


def read_tree(path):
    """The tree back as flat arrays (first primary), for checks."""
    import uproot
    with uproot.open(path) as f:
        t = f[TREE]
        a = t.arrays([k for k, _, _ in BRANCHES], library="np")
    out = {k: a[k] for k in EVENT}
    out["nprimaries"] = a["nprimaries"]
    for k in PRIMARY:
        out[k] = np.array([e[0] if len(e) else np.nan for e in a[k]])
    return out


# --- calorimeter ---------------------------------------------------------------------------------------------------------

class ShowerEnergies:
    """Whole-event Geant4 shower patterns of the calorimeter shower library (calofastsim.showers.library): the draw of
    one pattern per query (library.sample, orientation disk, radius_neighbors 8) and its summed G4 and Birks energies."""

    def __init__(self, library):
        from calofastsim.showers.library import load
        if isinstance(library, (str, os.PathLike)):
            library, _ = load(Path(library))
        self.library = library
        off = np.asarray(library["offsets"])
        e_g4 = np.r_[0.0, np.cumsum(library["step_energy_g4"], dtype=np.float64)]
        e_vis = np.r_[0.0, np.cumsum(library["step_energy_birks"], dtype=np.float64)]
        self.e_g4 = e_g4[off[1:]] - e_g4[off[:-1]]
        self.e_vis = e_vis[off[1:]] - e_vis[off[:-1]]

    def geometry(self, dtype=np.float64):
        from calofastsim.showers import handoff_device as HD
        return HD.disk_geometry(self.library, dtype)

    def draw(self, rng, entries):
        """(G4 energy, visible energy, donor row, fallback) per query; entries: energy (total), cosine, radius, disk,
        time, position, direction as library.sample takes them."""
        from calofastsim.showers.library import sample
        n = len(entries["energy"])
        if n == 0:
            z = np.zeros(0)
            return z, z, np.zeros(0, np.int64), np.zeros(0, bool)
        s = sample(self.library, entries, rng, orientation="disk", radius_neighbors=RADIUS_NEIGHBORS)
        donor = s["donor"]
        ok = donor >= 0
        e_g4, e_vis = np.zeros(n), np.zeros(n)
        e_g4[ok], e_vis[ok] = self.e_g4[donor[ok]], self.e_vis[donor[ok]]
        return e_g4, e_vis, donor, s["fallback"]


def _entries(energy, x, u, disk, time_ns, origin):
    """library.sample queries from states at the calorimeter (total energy, position, direction, disk index)."""
    d = np.asarray(disk, np.int64)
    r = np.hypot(x[:, 0] - origin[d, 0], x[:, 1] - origin[d, 1])
    return dict(energy=np.asarray(energy, float), cosine=np.clip(u[:, 2], 0.0, 1.0), radius=r, disk=d,
                time=np.asarray(time_ns, float), position=np.asarray(x, float), direction=np.asarray(u, float))


def rmax_1t(x, y, px, py, axis_x=-3904.0):
    """Largest radius about the DS axis of the e- helix through (x, y) [mm] with transverse momentum (px, py) [MeV/c]
    in a uniform axial field B_REF: the maximum-radius variable of the inner-edge table and of the RMAX_MIN cut (the
    same formula is applied to the Geant4 crossings the table was made from)."""
    pt = np.hypot(px, py)
    R = pt / (0.299792458 * B_REF)
    ux, uy = px / np.maximum(pt, 1e-12), py / np.maximum(pt, 1e-12)
    return np.hypot(x - axis_x - R * uy, y + R * ux) + R


class EdgeEnergies:
    """Geant4 inner-edge calorimeter entries: the event's visible (Birks) and G4 calorimeter energy of electrons whose
    first calorimeter VD crossing is the r = 335 mm inner-edge cylinder of a disk, binned by disk, helix maximum radius
    (rmax_1t at the crossing; bins split at EDGE_RMAX_EDGES) and crossing depth (z behind the disk's front VD, 25 mm
    bins). A draw picks one Geant4 entry of the query's bin (when the bin holds fewer than 20 entries, the nearest
    depth bin of the same disk and radius bin that does; then the nearest radius bin). npz columns: disk, dz,
    rmax (optional: one radius bin without it), e_vis, e_g4."""

    def __init__(self, table):
        t = dict(np.load(table)) if isinstance(table, (str, os.PathLike)) else dict(table)
        self.disk = np.asarray(t["disk"], np.int64)
        self.bin = np.clip((np.asarray(t["dz"], float) // EDGE_BIN_MM).astype(np.int64), 0, None)
        self.has_rmax = "rmax" in t
        self.rbin = (np.searchsorted(EDGE_RMAX_EDGES, np.asarray(t["rmax"], float), side="right")
                     if self.has_rmax else np.zeros(self.disk.size, np.int64))
        self.nr = len(EDGE_RMAX_EDGES) + 1 if self.has_rmax else 1
        self.e_vis = np.asarray(t["e_vis"], float)
        self.e_g4 = np.asarray(t["e_g4"], float)
        self.nbin = int(self.bin.max()) + 1 if self.bin.size else 1
        self.pools = {}
        for d in np.unique(self.disk):
            grid = [[np.flatnonzero((self.disk == d) & (self.rbin == r) & (self.bin == b)) for b in range(self.nbin)]
                    for r in range(self.nr)]
            full = [(r, b) for r in range(self.nr) for b in range(self.nbin) if grid[r][b].size >= EDGE_MIN_ROWS]
            if not full:
                full = [max(((r, b) for r in range(self.nr) for b in range(self.nbin)), key=lambda rb: grid[rb[0]][rb[1]].size)]
            self.pools[int(d)] = [[grid[r][b] if grid[r][b].size >= EDGE_MIN_ROWS else
                                   grid[min(full, key=lambda f: (abs(f[0] - r), abs(f[1] - b), f))[0]]
                                       [min(full, key=lambda f: (abs(f[0] - r), abs(f[1] - b), f))[1]]
                                   for b in range(self.nbin)] for r in range(self.nr)]

    def draw(self, rng, disk, dz, rmax=None):
        """(G4 energy, visible energy) per query (disk index, depth behind the front VD [mm], helix maximum radius)."""
        n = len(disk)
        e_g4, e_vis = np.zeros(n), np.zeros(n)
        b = np.clip((np.asarray(dz, float) // EDGE_BIN_MM).astype(np.int64), 0, self.nbin - 1)
        r = (np.searchsorted(EDGE_RMAX_EDGES, np.asarray(rmax, float), side="right") if self.has_rmax and rmax is not None
             else np.zeros(n, np.int64))
        u = rng.random(n)
        for i in range(n):
            pools = self.pools.get(int(disk[i]))
            if pools is None:
                continue
            pool = pools[r[i]][b[i]]
            j = pool[int(u[i] * pool.size)]
            e_g4[i], e_vis[i] = self.e_g4[j], self.e_vis[j]
        return e_g4, e_vis


FRONT, EDGE = 1, 2


def first_contact(field, x, u, T, front_z, axis_x, *, h=10.0, sub=5, s_max=5000.0, r_wall=1000.0):
    """First calorimeter VD crossing of e- helices from (x, u, kinetic energy T) in vacuum (midpoint-field helix steps
    in the field maps, `sub` samples per step): a disk's front plane (z = front_z[d]) inside the annulus
    [FRONT_RMIN, FRONT_RMAX], or its inner-edge cylinder (r = EDGE_R about the calorimeter axis) between the front plane
    and DISK_LENGTH behind it. Returns kind (0 none, FRONT, EDGE), disk, and position, direction and depth behind the
    front plane at the crossing sample, and the helix maximum radius there (rmax_1t)."""
    from mu2efastsim import ehelix
    x, u, T = np.array(x, float), np.array(u, float), np.asarray(T, float)
    n = len(T)
    fz = np.asarray(front_z, float)
    bz = fz + DISK_LENGTH
    p = np.sqrt(T * (T + 2 * ME))
    kind, disk = np.zeros(n, np.int8), np.full(n, -1, np.int64)
    xc, uc = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    s = np.zeros(n)
    z_lo = x[:, 2] - 200.0 if n else np.zeros(0)
    live = np.ones(n, bool)
    while live.any():
        i = np.flatnonzero(live)
        xi, ui, pi = x[i], u[i], p[i]
        B0 = ehelix.field_at(field, xi)
        Bm = ehelix.field_at(field, ehelix.helix(xi, ui, pi, -1.0, B0, 0.5 * h)[0])
        xa = xi
        r_a = np.hypot(xa[:, 0] - axis_x, xa[:, 1])
        hit = np.zeros(i.size, bool)
        for k in range(1, sub + 1):
            xk, uk = ehelix.helix(xi, ui, pi, -1.0, Bm, h * k / sub)
            r_k = np.hypot(xk[:, 0] - axis_x, xk[:, 1])
            for d in range(fz.size):
                fr = ~hit & (xa[:, 2] < fz[d]) & (xk[:, 2] >= fz[d]) & (r_k >= FRONT_RMIN) & (r_k <= FRONT_RMAX)
                ed = ~hit & ~fr & (r_a < EDGE_R) & (r_k >= EDGE_R) & (xk[:, 2] >= fz[d]) & (xk[:, 2] <= bz[d])
                for m, kd in ((fr, FRONT), (ed, EDGE)):
                    if m.any():
                        j = i[m]
                        kind[j], disk[j], xc[j], uc[j] = kd, d, xk[m], uk[m]
                        hit |= m
            xa, r_a = xk, r_k
        xn, un = ehelix.helix(xi, ui, pi, -1.0, Bm, h)
        x[i], u[i] = xn, un
        s[i] += h
        rn = np.hypot(xn[:, 0] - axis_x, xn[:, 1])
        done = hit | (xn[:, 2] > bz.max()) | (xn[:, 2] < z_lo[i]) | (rn > r_wall) | (s[i] > s_max)
        live[i[done]] = False
    dz = np.where(kind > 0, xc[:, 2] - fz[np.maximum(disk, 0)], np.nan)
    rmax = np.where(kind > 0, rmax_1t(xc[:, 0], xc[:, 1], p * uc[:, 0], p * uc[:, 1], axis_x), np.nan)
    return dict(kind=kind, disk=disk, x=xc, u=uc, dz=dz, rmax=rmax)


class SideProxy:
    """Electrons that can reach the disks' inner wall: the cesob wall proxy (inner Al and carbon-fibre rings of the
    disk cases, then >= 17.8 mm inside crystals) from the tracker exit state, on the host; it gives the first crystal
    contact state (front face or inner edge) and the acceptance. procs > 1: chunks over a spawned process pool (the
    proxy is NumPy, one core each)."""

    def __init__(self, geom_file, offline_root, maps, procs=1):
        from . import cechain, cesob, simpleconfig
        self.args = (geom_file, offline_root, list(maps))
        path = geom_file if os.path.isabs(geom_file) else f"{offline_root}/{geom_file}"
        cfg = simpleconfig.read(path, offline_root)
        self.calo = cesob.calo_from_config(cfg)
        self.wall = cesob.calo_wall_from_config(cfg, self.calo)
        self.field = cechain.field_maps(maps)
        self.procs = int(procs)
        self._pool = None

    def can_reach_wall(self, x, u, T):
        """True where the helix from (x, u, T) in the local axial field reaches the disks' inner-wall radius (minus
        WALL_MARGIN_MM) about the calorimeter axis: the electrons the wall can change."""
        from . import cechain, cesob
        p = np.sqrt(T * (T + 2 * ME))
        bz = self.field(x)[:, 2]
        bz = np.where(np.abs(bz) > 1e-3, bz, 1e-3)
        rmax, _ = cechain.helix_rmax(cesob._Axis(self.calo.axis_x), x, u, p, bz)
        return rmax >= self.wall.al_rin - WALL_MARGIN_MM

    def _one(self, seed, x, u, T):
        from . import cesob
        contact = {}
        acc = cesob.calo_accept(self.field, self.calo, x, u, T, wall=self.wall,
                                rng=np.random.default_rng(np.random.SeedSequence(seed)), contact=contact)
        return acc, contact

    def __call__(self, seed, x, u, T):
        """(accepted, contact dict, disk index of the contact) for exit states x, u, kinetic energy T. seed: a
        sequence of ints (the random stream is per chunk of 2048 electrons, so it does not depend on procs)."""
        n = len(T)
        bounds = [(a, min(a + 2048, n)) for a in range(0, n, 2048)]
        jobs = [(list(seed) + [k], x[a:b], u[a:b], T[a:b]) for k, (a, b) in enumerate(bounds)]
        if self.procs > 1 and len(jobs) > 1:
            if self._pool is None:
                import multiprocessing as mp
                self._pool = mp.get_context("spawn").Pool(self.procs, initializer=_side_worker_init,
                                                          initargs=(self.args,))
            parts = self._pool.map(_side_worker, jobs, chunksize=1)
        else:
            parts = [self._one(*j) for j in jobs]
        acc = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, bool)
        contact = {k: np.concatenate([p[1][k] for p in parts]) for k in ("found", "x", "u", "T", "s")} if parts else \
            dict(found=np.zeros(0, bool), x=np.zeros((0, 3)), u=np.zeros((0, 3)), T=np.zeros(0), s=np.zeros(0))
        z = contact["z"] = contact["x"][:, 2]
        zf = np.asarray(self.wall.case_lo)[None]
        zb = np.asarray(self.wall.case_hi)[None]
        disk = np.argmax((z[:, None] >= zf) & (z[:, None] <= zb), axis=1)
        return acc & contact["found"], contact, disk

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


_SIDE = {}


def _side_worker_init(args):
    from .cesob import _single_thread
    _single_thread()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    _SIDE["proxy"] = SideProxy(*args)


def _side_worker(job):
    return _SIDE["proxy"]._one(*job)


# --- the fast chain --------------------------------------------------------------------------------------------------

def _mu_minus_stops(stops):
    st = dict(np.load(stops)) if isinstance(stops, (str, os.PathLike)) else dict(stops)
    keep = np.ones(np.size(st["t"]), bool)
    if "q" in st and (np.asarray(st["q"]) < 0).any():
        keep = np.asarray(st["q"]) < 0
    x = np.asarray(st["x"], float)
    if x.ndim == 1:                    # x, y, z columns (stop tables from Geant4 stop files) instead of x [n, 3]
        x = np.column_stack([x, np.asarray(st["y"], float), np.asarray(st["z"], float)])
    x = x[keep]
    t = np.asarray(st["t"], float)[keep]
    if t.size == 0:
        raise ValueError("edepana: no mu- stops")
    return dict(x=x[:, 0], y=x[:, 1], z=x[:, 2], t=t)


class FastEdep:
    """The fast chain to EdepAna rows, built once: DeviceChain (no digis, no reco response), shower library, side
    proxy. Each call is one batch = one subrun of births."""

    def __init__(self, offline_root, maps, gdml, library, geom=None, dtype="float64", calo=CALO_MODE, deltas=None,
                 albedo="default", lanes=None, procs=1, edge_library=None):
        from .cetracker import DeviceChain
        if calo not in CALO_MODES:
            raise ValueError(f"edepana: calo must be one of {CALO_MODES}")
        from .__main__ import GEOM_FILE
        self.geom = geom or GEOM_FILE          # cesob's default (the target geometry of the beam_sob stand-in)
        self._chain_args = (DeviceChain, (offline_root, maps, gdml),
                            dict(geom=self.geom, response=None, digi_cond=None, dtype=dtype, deltas=deltas,
                                 albedo=albedo, lanes=lanes))
        self._chain = None
        self.dtype = dtype
        self.calo_mode = calo
        self.showers = ShowerEnergies(library)
        self.geo = self.showers.geometry(np.dtype(dtype))
        self.origin = np.asarray(self.geo["origin"], float)
        self.side = SideProxy(self.geom, offline_root, maps, procs) if calo == "library+wall" else None
        self.edge = None
        if calo == "library+edge":
            if edge_library is None:
                raise ValueError("edepana: calo library+edge needs the inner-edge energy table (edge_library)")
            from . import cechain
            self.edge = EdgeEnergies(edge_library)
            self.field = cechain.field_maps(maps)
            self.front_z = np.asarray(self.geo["front"], float)
            self.axis_x = float(self.origin[0, 0])
        self.config = dict(dtype=dtype, calo=calo, deltas=deltas, albedo=albedo, geom=self.geom,
                           radius_neighbors=RADIUS_NEIGHBORS)
        if self.edge is not None:
            self.config.update(edge_r_mm=EDGE_R, front_annulus_mm=[FRONT_RMIN, FRONT_RMAX], edge_bin_mm=EDGE_BIN_MM,
                               edge_rmax_edges_mm=list(EDGE_RMAX_EDGES), rmax_min_mm=RMAX_MIN, b_ref_t=B_REF)

    @property
    def chain(self):
        """The DeviceChain, built on first use (the host-only calo() needs no device)."""
        if self._chain is None:
            cls, args, kw = self._chain_args
            self._chain = cls(*args, **kw)
        return self._chain

    def device(self, births, seed, pad_to=None):
        """The device part of one batch: transport, tracker step sums, tracker exit and disk-front states, as host
        arrays (one row per birth). pad_to: transport this many electrons (the births repeated), so every batch has
        one compiled shape; the padding is dropped."""
        import jax
        import jax.numpy as jnp
        from calofastsim.showers import handoff_device as HD
        from trkfastsim.sim import fieldsteps_jax as FJ
        x0, u0, p0, t0 = (np.asarray(v, float) for v in births)
        n = t0.size
        if pad_to is not None and pad_to > n:
            take = np.arange(pad_to) % n
            x0, u0, p0, t0 = x0[take], u0[take], p0[take], t0[take]
        tic = time.perf_counter()
        res = self.chain.transport((x0, u0, p0, t0), seed)
        st = res["tracker"].steps
        valid = st["valid"]
        pmag = jnp.sqrt(st["px"] ** 2 + st["py"] ** 2 + st["pz"] ** 2)
        eion = jnp.where(valid, st["eion"], 0.0)
        trk_e = jnp.sum(eion, axis=1)
        trk_n = jnp.sum(valid & (st["eion"] > FILTER_MIN_TRK_STEP_E) & (pmag > FILTER_MIN_PART_MOM), axis=1)
        fin = res["tracker"].final
        xe = jnp.stack([fin["x"], fin["y"], fin["z"]], axis=1)
        pv = jnp.stack([fin["px"], fin["py"], fin["pz"]], axis=1)
        pe = jnp.linalg.norm(pv, axis=1)
        Te = jnp.sqrt(pe * pe + ME ** 2) - ME
        out_trk = res["has_entry"] & (fin["status"] == FJ.LEFT) & (pe > 0)
        ue = pv / jnp.where(pe > 0, pe, 1.0)[:, None]
        front = HD.disk_front(self.chain.maps, self.geo, xe, ue, Te, fin["t"], out_trk)
        h = jax.device_get(dict(has=res["has_entry"], pf=res["p"], trk_e=trk_e, trk_n=trk_n, out=out_trk,
                                found=front["found"], disk=front["disk"], pos=front["position"],
                                dir=front["direction"], energy=front["energy"], time=front["time"],
                                xe=xe, ue=ue, Te=Te, te=fin["t"]))
        h = {k: np.asarray(v)[:n] for k, v in h.items()}
        h.update(x0=x0[:n], u0=u0[:n], p0=p0[:n], t0=t0[:n])
        timing = dict(device_s=round(time.perf_counter() - tic, 3), **{f"chain_{k}": v for k, v in res["timing"].items()})
        return h, timing

    def batch(self, births, seed, subrun, pad_to=None):
        """Per-electron columns of one batch (all births, before the storage filter) and counters."""
        h, timing = self.device(births, seed, pad_to)
        cols, stats = self.calo(h, seed, subrun)
        stats.update(timing)
        return cols, stats

    @staticmethod
    def _columns(h, x0, u0, p0, t0, e_g4, e_vis, front):
        has, p_front, e_front = _front_columns(h)
        return dict(x=x0, u=u0, p=p0, t=t0, trk_e=h["trk_e"].astype(np.float64), trk_n=h["trk_n"].astype(np.int64),
                    has_entry=has, p_front=p_front, e_front=e_front, calo_g4=e_g4, calo_vis=e_vis,
                    disk_front=front, out_tracker=h["out"].astype(bool))

    def calo(self, h, seed, subrun):
        """The host part of one batch: calorimeter energies (shower library, inner-edge table or wall proxy) and the per-electron
        columns, from device() output."""
        x0, u0, p0, t0 = h["x0"], h["u0"], h["p0"], h["t0"]
        n = t0.size
        rng = np.random.default_rng(np.random.SeedSequence([seed, subrun, 7]))
        e_g4, e_vis = np.zeros(n), np.zeros(n)
        f = h["found"]
        if self.edge is not None:            # library+edge: the first VD crossing decides front or inner edge
            tic = time.perf_counter()
            k = np.flatnonzero(h["out"])
            ue = h["ue"][k].astype(float)
            c = first_contact(self.field, h["xe"][k].astype(float), ue / np.linalg.norm(ue, axis=1, keepdims=True),
                              h["Te"][k].astype(float), self.front_z, self.axis_x)
            low = (c["kind"] > 0) & (c["rmax"] < RMAX_MIN)      # cannot reach the crystals: no energy
            fr, ed = (c["kind"] == FRONT) & ~low, (c["kind"] == EDGE) & ~low
            q = _entries(h["Te"][k][fr].astype(float) + ME, c["x"][fr], c["u"][fr], c["disk"][fr],
                         h["te"][k][fr].astype(float), self.origin)
            g, v, _, fb = self.showers.draw(rng, q)
            e_g4[k[fr]], e_vis[k[fr]] = g, v
            g, v = self.edge.draw(rng, c["disk"][ed], c["dz"][ed], c["rmax"][ed])
            e_g4[k[ed]], e_vis[k[ed]] = g, v
            f = np.zeros(n, bool)
            f[k[fr]] = True
            cols = self._columns(h, x0, u0, p0, t0, e_g4, e_vis, f)
            stats = dict(births=n, tracker_entry=int(cols["has_entry"].sum()), tracker_exit=int(h["out"].sum()),
                         disk_front=int(fr.sum()), side_candidates=0, side_accepted=0, wall_front=0,
                         edge_entry=int(ed.sum()), below_rmax=int(low.sum()), library_fallback=int(fb.sum()),
                         side_s=round(time.perf_counter() - tic, 3))
            return cols, stats
        # library+wall: electrons whose helix from the tracker exit can reach the disks' inner wall go through the wall
        # proxy for their first contact; the others (and all in "library") meet the calorimeter, if at all, at the
        # vacuum disk-front crossing of the device hand-off
        wall_ok = np.zeros(n, bool)
        if self.side is not None:
            wall_ok = h["out"] & self.side.can_reach_wall(h["xe"].astype(float), h["ue"].astype(float),
                                                          h["Te"].astype(float))
        front = f & ~wall_ok
        q = _entries(h["energy"][front], h["pos"][front].astype(float), h["dir"][front].astype(float),
                     h["disk"][front], h["time"][front], self.origin)
        g, v, _, fb_front = self.showers.draw(rng, q)
        e_g4[front], e_vis[front] = g, v
        n_side = n_side_acc = n_wall_front = 0
        fb_side = np.zeros(0, bool)
        tic = time.perf_counter()
        if self.side is not None:
            cand = np.flatnonzero(wall_ok)
            n_side = cand.size
            if cand.size:
                ue = h["ue"][cand].astype(float)
                acc, c, disk = self.side((seed, subrun, 11), h["xe"][cand].astype(float),
                                         ue / np.linalg.norm(ue, axis=1, keepdims=True), h["Te"][cand].astype(float))
                at_front = c["found"] & (c["z"] - np.asarray(self.side.calo.z_front)[disk] <= FRONT_CONTACT_MM)
                draw = at_front | (acc & ~at_front)   # a front-face contact always draws; an inner-edge one if accepted
                k = cand[draw]
                n_side_acc = int((draw & ~at_front).sum())
                n_wall_front = int(at_front.sum())
                ux = c["u"][draw] / np.linalg.norm(c["u"][draw], axis=1, keepdims=True)
                q = _entries(c["T"][draw] + ME, c["x"][draw], ux, disk[draw], h["te"][k].astype(float), self.origin)
                g, v, _, fb_side = self.showers.draw(rng, q)
                e_g4[k], e_vis[k] = g, v
        t_side = time.perf_counter() - tic
        cols = self._columns(h, x0, u0, p0, t0, e_g4, e_vis, f)
        has = cols["has_entry"]
        stats = dict(births=n, tracker_entry=int(has.sum()), tracker_exit=int(h["out"].sum()), disk_front=int(f.sum()),
                     side_candidates=int(n_side), side_accepted=int(n_side_acc), wall_front=int(n_wall_front),
                     edge_entry=0, below_rmax=0,
                     library_fallback=int(fb_front.sum() + fb_side.sum()), side_s=round(t_side, 3))
        return cols, stats


def _front_columns(h):
    pf = np.linalg.norm(h["pf"].astype(np.float64), axis=1)
    has = h["has"].astype(bool)
    return has, np.where(has, pf, 0.0), np.where(has, np.sqrt(pf * pf + ME * ME), 0.0)


def rows_from_columns(cols, run, subrun, ngen_running, event0=1):
    """EdepAna rows of the stored events of one batch (CeEndpoint filter applied); event numbers from event0 in birth
    order (unstored births keep their numbers, as in a filtered art stream)."""
    keep = detector_filter(cols["trk_n"], cols["calo_vis"])
    idx = np.flatnonzero(keep)
    p = cols["p"][idx]
    pv = cols["u"][idx] * p[:, None]
    x = cols["x"][idx]
    e_start = np.sqrt(p * p + ME * ME)
    n = idx.size
    return dict(primary_start_x=x[:, 0], primary_start_y=x[:, 1], primary_start_z=x[:, 2],
                primary_start_px=pv[:, 0], primary_start_py=pv[:, 1], primary_start_pz=pv[:, 2],
                primary_start_e=e_start, primary_start_m=np.full(n, ME), primary_start_pdg=np.full(n, PDG_E),
                primary_calo_edep=cols["calo_g4"][idx], primary_calo_edep_vis=cols["calo_vis"][idx],
                primary_trk_front_p=cols["p_front"][idx], primary_trk_front_energy=cols["e_front"][idx],
                event_calo_edep=cols["calo_g4"][idx], event_calo_edep_vis=cols["calo_vis"][idx],
                event_trk_edep=cols["trk_e"][idx], run=np.full(n, run), subrun=np.full(n, subrun),
                event=idx + event0, weight=np.ones(n), ngen=np.full(n, ngen_running, np.int64))


def concat_rows(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def provenance(**kw):
    """What made the file: code commit (M2FS_CODE_REF / git), chain configuration, inputs, counts."""
    import subprocess
    here = Path(__file__).resolve().parents[2]
    commit = os.environ.get("M2FS_CODE_COMMIT")
    if not commit:
        try:
            commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                                    timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    return dict(producer="mu2efastsim beamfastsim.edepana", format="Offline EdepAna EDepAna/tree (v13_39_00)",
                code_commit=commit, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


DEVICE_COLUMNS = "edep_device_columns.npz"


def run(stops, electrons, seed, out, *, offline_root, maps, gdml, library, geom=None, dtype="float64",
        calo=CALO_MODE, deltas=None, batch=65536, run_number=1, name=FILE_NAME, extra_provenance=None,
        keep_columns=False, procs=1, stage="all", columns=None, edge_library=None):
    """Births from the mu- stops -> EdepAna rows (one subrun per batch) -> the file and its sidecars in `out`.
    Every batch transports `batch` electrons (the last one padded and the padding dropped), so the device compiles
    one shape. stage: "all"; "device" (the device part only: per-batch columns to out/edep_device_columns.npz, no
    file); "calo" (the host part from `columns`, a device-stage npz: same births, same file as "all").
    Returns (paths, counters)."""
    from .cetracker import DeviceChain
    if stage not in ("all", "device", "calo"):
        raise ValueError("edepana: stage must be all, device or calo")
    st = _mu_minus_stops(stops) if stage != "calo" else None
    fe = FastEdep(offline_root, maps, gdml, library, geom=geom, dtype=dtype, calo=calo, deltas=deltas, procs=procs,
                  edge_library=edge_library)
    parts, stats, cols_all, dev = [], [], [], {}
    ngen = 0
    tic = time.perf_counter()
    if stage == "calo":
        src = np.load(columns)
        meta = json.loads(str(src["meta"]))
        if (meta["electrons"], meta["seed"], meta["batch"], meta["dtype"]) != (electrons, seed, batch, dtype):
            raise ValueError(f"edepana: {columns} was made with {meta}, not this run")
        nsub = meta["subruns"]
        st = dict(t=np.zeros(meta["mu_minus_stops"]))
    else:
        nsub = (electrons + batch - 1) // batch
    for sub in range(nsub):
        nb = min(batch, electrons - sub * batch)
        bseed = seed * 1000 + sub
        if stage == "calo":
            h = {k.split("__", 1)[1]: src[k] for k in src.files if k.startswith(f"b{sub}__")}
            timing = json.loads(str(src[f"timing__{sub}"]))
        else:
            h, timing = fe.device(DeviceChain.births(st, nb, bseed), bseed, pad_to=batch)
        ngen += nb
        if stage == "device":
            dev.update({f"b{sub}__{k}": v for k, v in h.items()})
            dev[f"timing__{sub}"] = json.dumps(timing)
            print(json.dumps(dict(subrun=sub, births=nb, **timing)), flush=True)
            continue
        cols, s = fe.calo(h, bseed, sub)
        s.update(timing)
        parts.append(rows_from_columns(cols, run_number, sub, ngen))
        s.update(subrun=sub, stored=int(parts[-1]["event"].size))
        stats.append(s)
        if keep_columns:
            cols_all.append({k: v for k, v in cols.items() if np.ndim(v) == 1})
        print(json.dumps(s), flush=True)
    wall = time.perf_counter() - tic
    if stage == "device":
        Path(out).mkdir(parents=True, exist_ok=True)
        meta = dict(electrons=int(electrons), seed=int(seed), batch=int(batch), dtype=dtype, subruns=nsub,
                    mu_minus_stops=int(st["t"].size), stops=str(stops), geom=fe.geom, wall_s=round(wall, 2))
        np.savez(Path(out) / DEVICE_COLUMNS, meta=json.dumps(meta), **dev)
        return dict(columns=Path(out) / DEVICE_COLUMNS), dict(wall_s=round(wall, 2), subruns=nsub)
    if fe.side is not None:
        fe.side.close()
    rows = concat_rows(parts)
    tot = {k: int(sum(s[k] for s in stats)) for k in ("births", "tracker_entry", "tracker_exit", "disk_front",
                                                     "side_candidates", "side_accepted", "wall_front",
                                                     "edge_entry", "below_rmax", "library_fallback", "stored")}
    lib_path = library if isinstance(library, (str, os.PathLike)) else None
    prov = provenance(chain=fe.config, electrons=int(electrons), seed=int(seed), batch=int(batch),
                      run=int(run_number), subruns=len(stats), generated=int(ngen), counts=tot,
                      stops=dict(path=str(stops) if isinstance(stops, (str, os.PathLike)) else None,
                                 mu_minus=int(st["t"].size)),
                      library_sha256=_sha256(lib_path) if lib_path else None,
                      edge_library_sha256=_sha256(edge_library) if isinstance(edge_library, (str, os.PathLike)) else None,
                      filter=dict(min_trk_steps=FILTER_MIN_TRK_STEPS, min_trk_step_e_mev=FILTER_MIN_TRK_STEP_E,
                                  min_part_mom=FILTER_MIN_PART_MOM, min_calo_vis_mev=FILTER_MIN_CALO_E),
                      stage=stage, device_s=round(sum(s.get("device_s", 0.0) for s in stats), 2),
                      side_s=round(sum(s.get("side_s", 0.0) for s in stats), 2), **(extra_provenance or {}))
    paths = write_outputs(out, rows, prov, name=name)
    if keep_columns:
        np.savez_compressed(Path(out) / "edep_columns.npz", **concat_rows(cols_all))
    tot.update(wall_s=round(wall, 2), electrons_per_s=round(electrons / wall, 2) if wall > 0 else 0.0)
    return paths, tot


# --- CLI -------------------------------------------------------------------------------------------------------------

def _paths(a):
    from .__main__ import MAPS, OFFLINE_ROOT
    root = a.offline_root or OFFLINE_ROOT
    maps = a.maps or [f"{MAPS}/DSMap.header", f"{MAPS}/TSdMap.header"]
    return root, maps


def _metrics(out, paths, tot, extra=None):
    """metrics.json for the kit: the edep summary, the default sob selection's count, chain counters."""
    m = json.loads(Path(paths["provenance"]).read_text())["summary"]
    rows = read_tree(paths["file"])
    sel = rows["event_calo_edep_vis"] > 10.0
    m.update(n_events_vis_gt10=float(sel.sum()), ce_acceptance=float(sel.sum()) / m["n_gen_events"],
             **{k: float(v) for k, v in tot.items()}, **(extra or {}))
    Path(out, "metrics.json").write_text(json.dumps({k: float(v) for k, v in m.items()}) + "\n")
    return m


def cmd_ce(a):
    root, maps = _paths(a)
    paths, tot = run(a.stops, a.electrons, a.seed, a.out, offline_root=root, maps=maps, gdml=a.gdml,
                     library=a.library, geom=a.geom, dtype=a.dtype, calo=a.calo,
                     deltas="default" if a.deltas else None, batch=a.batch, keep_columns=a.keep,
                     procs=a.procs, stage=a.stage, columns=a.columns, edge_library=a.edge_library)
    if a.stage == "device":
        print(json.dumps({k: str(v) for k, v in {**paths, **tot}.items()}))
        return
    print(json.dumps(_metrics(a.out, paths, tot)))


def cmd_point(a):
    """beam stage (stops of the point's target, as cesob point) then ce; metrics add stops_per_pot."""
    from .__main__ import main as beam_main
    from . import cesob
    os.makedirs(a.out, exist_ok=True)
    stops = os.path.join(a.out, "stops.npz")
    t0 = time.time()
    argv = ["point", "--g4", a.g4, "--out", a.out, "--n", str(a.muons), "--seed", str(a.seed), "--repeat", "0",
            "--batch", "131072", "--chunk", "400", "--stops-out", stops]
    if a.geom:
        argv += ["--geom", a.geom]
    if a.z_up is not None:
        argv += ["--z-up", str(a.z_up)]
    beam_main(argv)
    bm = json.load(open(os.path.join(a.out, "metrics.json")))
    g4 = np.load(a.g4)
    events = g4.get("events")
    if events is None or np.ndim(events) != 0 or int(events) <= 0:
        raise ValueError(f"{a.g4}: the start table records no generated events (g4table --events)")
    stops_per_gen = bm["stop_frac"] * int(g4["evt"].size) / float(events)
    t_beam = time.time() - t0
    root, maps = _paths(a)
    paths, tot = run(stops, a.electrons, a.seed, a.out, offline_root=root, maps=maps, gdml=a.gdml, library=a.library,
                     geom=a.geom, dtype=a.dtype, calo=a.calo, deltas="default" if a.deltas else None, batch=a.batch,
                     procs=a.procs, edge_library=a.edge_library, extra_provenance=dict(stops_per_gen_event=stops_per_gen, upstream_eff=cesob.UPSTREAM_EFF))
    extra = dict(stops_per_gen_event=stops_per_gen, stops_per_pot=stops_per_gen * cesob.UPSTREAM_EFF,
                 stop_frac=bm["stop_frac"], stop_frac_err=bm.get("stop_frac_err", math.nan), beam_s=t_beam)
    print(json.dumps(_metrics(a.out, paths, tot, extra)))
    if not a.keep:
        os.remove(stops)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.edepana", description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--geom", default=None, help="Offline geometry file (absolute, or under the Offline root)")
        s.add_argument("--electrons", type=int, default=200_000)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--dtype", choices=("float64", "float32"), default="float64", help="device chain precision")
        s.add_argument("--library", required=True, help="calorimeter shower library (.npz)")
        s.add_argument("--gdml", required=True, help="Run 1A GDML for the tracker transport")
        s.add_argument("--offline-root", dest="offline_root", default=None)
        s.add_argument("--maps", nargs="+", default=None, help="G4BL headers, DS map first")
        s.add_argument("--calo", choices=CALO_MODES, default=CALO_MODE)
        s.add_argument("--edge-library", dest="edge_library", default=None,
                       help="Geant4 inner-edge energy table (.npz: disk, dz, e_vis, e_g4), for calo library+edge")
        s.add_argument("--deltas", action="store_true", help="delta-ray straw steps (trkfastsim.sim.deltas library)")
        s.add_argument("--batch", type=int, default=65536, help="births per batch (one subrun each)")
        s.add_argument("--keep", action="store_true", help="keep per-electron columns (and the stops of point)")
        s.add_argument("--procs", type=int, default=1, help="processes for the side-entry wall proxy (host)")
        s.add_argument("--out", required=True)
    s = sp.add_parser("ce", help="EdepAna file from a stop table")
    s.add_argument("--stops", required=True)
    s.add_argument("--stage", choices=("all", "device", "calo"), default="all",
                   help="device: only the device part (columns npz); calo: the host part from --columns")
    s.add_argument("--columns", default=None, help="device-stage columns (stage calo)")
    common(s)
    s = sp.add_parser("point", help="beam stage then the EdepAna file (one target geometry)")
    s.add_argument("--g4", required=True, help="start table with its generated events (g4table --events)")
    s.add_argument("--muons", type=int, default=0)
    s.add_argument("--z-up", dest="z_up", type=float, default=None)
    common(s)
    a = ap.parse_args(argv)
    {"ce": cmd_ce, "point": cmd_point}[a.cmd](a)


if __name__ == "__main__":
    main()
