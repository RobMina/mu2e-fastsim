"""Named CRV run configurations: the counters file, GDML world, response, coincidence cuts and digitisation window of one
stock Offline configuration, so every stage of the fast chain (transport, tier 1b, photons, SiPM, reco) takes one name.

    rc = runconfig.get("run2_v10");  geom = rc.geometry();  cfg = rc.chain_config(jax=True)

Sources (Offline v13_39_00):
  extracted_v04  crv_counters_extracted_v04, CRVResponse epilog_extracted_v05 (Production digitize/Extracted.fcl),
                 CRVReco epilog_extracted_v04 (data operating point), offspill window 0..100 us.
  run1a_v01      crv_counters_run1a_v01 (T1 17 + T2 3 modules on the hatch blocks), CRVResponse/CRVReco
                 epilog_run1a_v01 (Production digitize/epilog.fcl), on-spill digitisation window 200..1500 ns.
  run1b_v01      Run 1A's CRV (geom_run1_b_v01 includes geom_run1_a and does not override the CRV); own world (relocated
                 tracker, field off).
  run2_v10       crv_counters_v10 (23 sectors), CRVResponse/CRVReco prolog_v12 (per-sector tables, yields, cuts).
Tier 1b: the library was built on extracted v04 G4 (EX/T1/T2/taggers). Other configurations map every sector to the
library group of the same counter kind (lib_group); the non-muon showers keep absolute extracted bar positions, so they
are off outside extracted_v04 (showers=False; ~15 % of the CRV visible energy, mostly events of their own).
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import numpy as np

from .reco import coinc as C
from .response import photons as PH

GEOM_DIR = "geometry"        # under $CRVFS_DATA: the GDML world exports (geometry/README.md)
ONSPILL_WINDOW = (200.0, 1500.0)
OFFSPILL_WINDOW = (0.0, 1e5)


def _cuts(pe, dt_adj, dt, smin, smax, sdiff, layers, min_pe, d0, d1=0.0):
    return C.SectorCuts(pe, dt_adj, dt, smin, smax, sdiff, layers, min_pe, d0, d1)


# CRVReco prolog_v12 sectorConfig (Run 2), in sector order
RUN2_CUTS = {
    "R1": _cuts(12, 10, 10, -2, 5, 3, 3, 200, 250), "R2": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 250),
    "R3": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 250), "R4": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 250),
    "R5": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 250), "R6": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 250),
    "L1": _cuts(12, 10, 10, -5, 3, 3, 3, 100, 250), "L2": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 250),
    "L3": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 250),
    "T1": _cuts(15, 10, 20, -5, 2, 3, 4, 60, 250), "T2": _cuts(15, 10, 20, -5, 2, 3, 3, 60, 250),
    "T3": _cuts(8, 20, 20, -4, 3, 3, 3, 0, 250), "T4": _cuts(8, 20, 20, -6, 6, 3, 3, 0, 250),
    "T5": _cuts(8, 20, 20, -6, 6, 3, 3, 0, 250),
    "E1": _cuts(20, 10, 20, 0, 3, 3, 4, 200, 100), "E2": _cuts(20, 10, 20, 0, 3, 3, 4, 200, 100),
    "U": _cuts(8, 10, 30, -2, 0.01, 3, 4, 60, 100),
    "D1": _cuts(8, 10, 20, -6, 6, 3, 4, 0, 150), "D2": _cuts(8, 10, 20, -6, 6, 3, 4, 0, 150),
    "D3": _cuts(8, 10, 20, -6, 6, 3, 4, 0, 150), "D4": _cuts(8, 10, 20, -6, 6, 3, 4, 0, 150),
    "C1": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 150), "C2": _cuts(8, 10, 10, -6, 6, 3, 3, 0, 150),
}
# CRVResponse prolog_v12 CrvPhotons: (table, reflector, scintillation yield) per sector
_R2 = dict(zip(
    ["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3", "T1", "T2", "T3", "T4", "T5", "E1", "E2", "U",
     "D1", "D2", "D3", "D4", "C1", "C2"],
    zip(["4550_0", "4550_0", "1045_1", "3040_0", "4550_0", "3200_0", "4550_0", "4550_0", "3200_0", "6000_1", "6000_1",
         "6000_0", "6000_0", "6000_0", "5000_1", "5000_1", "6900_1", "5700_0", "2370_1", "2370_1", "5700_0", "2100_1",
         "1550_1"],
        [0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1, 1, 0, 1, -1, 0, -1, -1],
        [28610, 28610, 30482, 28610, 28610, 28610, 28610, 28610, 28610, 30482, 30482, 39794, 39794, 30482, 23841,
         23841, 23841, 28610, 28610, 28610, 28610, 28610, 28610])))
RUN2_SECTORS = {n: PH.SectorConfig(t, r, float(y)) for n, (t, r, y) in _R2.items()}

RUN1A_SECTORS = {"T1": PH.SectorConfig("6000_0", 0, 39794.0), "T2": PH.SectorConfig("6000_0", 0, 30482.0)}
RUN1A_CUTS = {n: _cuts(8, 20, 20, -6, 6, 3, 3, 0, 250) for n in ("T1", "T2")}


@dataclass(frozen=True)
class RunConfig:
    name: str
    counters: str                     # Offline-relative counters file
    gdml: str                         # GDML world export, relative to $CRVFS_DATA
    sectors: dict                     # sector -> photons.SectorConfig
    photon_yield_scale: float
    cuts: dict                        # sector -> coinc.SectorCuts
    ignore_secondary_pulses: bool
    window: tuple                     # digitisation window [ns]
    lib_group: dict = field(default_factory=dict)   # sector -> tier-1b library group (EX, T1, T2, M); {} = census
    tagger_sectors: tuple = ()        # sectors with the tagger delta cut (no CRSmother region cut in the reference)
    showers: bool = True              # tier-1b non-muon showers (absolute extracted bar positions)
    delta_cut_mev: float = 0.0        # delta production threshold in every non-tagger bar [MeV]; 0 = extracted (87.2 keV,
                                      # the 0.1 mm CRSmother region cut). Run 1A / Run 2 cosmic jobs: 1 mm everywhere,
                                      # lowest muon-made delta in the bars 0.361 MeV (G4 test 9112, 19,491 deltas)
    below_world: bool = False         # trace the world material between the MSC plane and every bar (side sectors)
    below_margin: float = 2000.0      # mm: below_world material is read inside the CRV box grown by this; Run 2: the
                                      # hall dirt and walls that shield the side walls from near-horizontal muons
    below_msc: bool = False           # with below_world: multiple scattering below the plane, bar to bar
                                      # (transport.scattered_crossings; opt-in, worse V1 than straight lines)
    geom_text: str = ""               # Offline geometry file of the world the GDML was made from
    add_opposite_side_pulses: bool = True

    def counters_path(self):
        from . import geometry
        return geometry.find_counters(self.counters)

    def geometry(self):
        """CRVGeometry with the tier-1b sector -> library group map attached (geom.lib_group, read by tier1b)."""
        from . import geometry
        g = geometry.from_offline_config(self.counters_path())
        if self.lib_group:
            g.lib_group = self.group_of_sector(g)
        return g

    def counters_cfg(self):
        from . import geometry
        return geometry.parse_simpleconfig(open(self.counters_path()).read())

    def gdml_path(self, data=None):
        data = data or os.environ.get("CRVFS_DATA")
        if not data:
            raise SystemExit(f"run config {self.name}: pass --gdml or set CRVFS_DATA (source site.env)")
        return os.path.join(data, self.gdml)

    def response(self):
        return dataclasses.replace(PH.ResponseConfig(), sectors=dict(self.sectors),
                                   photon_yield_scale=self.photon_yield_scale)

    def finder(self):
        return dataclasses.replace(C.FinderConfig(), cuts=dict(self.cuts),
                                   ignore_secondary_pulses=self.ignore_secondary_pulses,
                                   add_opposite_side_pulses=self.add_opposite_side_pulses)

    def chain_config(self, **kw):
        from . import chain
        return chain.ChainConfig(response=self.response(), finder=self.finder(), window=self.window, **kw)

    def bar_cut(self, geom):
        """[nbars] delta production threshold [MeV] for the transport, or None for the extracted default."""
        if not self.delta_cut_mev:
            return None
        from mu2efastsim import materials
        return np.where(self.tagger_bar(geom), materials.POLYSTYRENE.cuts_e_MeV["tagger"], self.delta_cut_mev)

    def tagger_bar(self, geom):
        """[nbars] bool: bars with the tagger (10 mm) delta production cut."""
        names = np.array(geom.sector_names, dtype=object)
        return np.isin(names[geom.sector], list(self.tagger_sectors))

    def group_of_sector(self, geom):
        """[nsectors] library group name per sector."""
        return np.array([self.lib_group.get(n, n) for n in geom.sector_names], dtype=object)


EXTRACTED_V04 = RunConfig(
    "extracted_v04", "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt", f"{GEOM_DIR}/mu2e_extracted_v04.gdml",
    dict(PH.EXTRACTED_V05), 0.94, dict(C.EXTRACTED_V04), True, OFFSPILL_WINDOW,
    tagger_sectors=tuple(f"M{i}" for i in range(1, 9)),
    geom_text="Offline/Mu2eG4/geom/geom_common_extracted_v04.txt")

# every non-tagger sector of Run 1A / Run 2 is a CRV module with 0.1 mm-cut-like descendants: library group EX (6 m
# CRV-T bars); the library cut (87.2 keV) is the transport cut, so restricted loss + library deltas add up
RUN1A_V01 = RunConfig(
    "run1a_v01", "Offline/Mu2eG4/geom/crv_counters_run1a_v01.txt", f"{GEOM_DIR}/mu2e_run1a_v01.gdml",
    RUN1A_SECTORS, 0.90, RUN1A_CUTS, False, ONSPILL_WINDOW, lib_group={"T1": "EX", "T2": "EX"}, showers=False,
    delta_cut_mev=0.361, below_world=True,
    geom_text="Offline/Mu2eG4/geom/geom_run1_a.txt")
# Run 1B: the CRV and everything above it (building, hatch blocks) are Run 1A's; the relocated tracker and DS2 are inside
# the CRV and never seen by the transport, so the Run 1A GDML serves (the stock cosmic job does not build the Run 1B
# world: "Failed to retrieve ... MECOStyleProtonAbsorber", the PA is disabled in geom_run1_b_v01)
RUN1B_V01 = dataclasses.replace(RUN1A_V01, name="run1b_v01", geom_text="Offline/Mu2eG4/geom/geom_run1_b_v01.txt")
RUN2_V10 = RunConfig(
    "run2_v10", "Offline/Mu2eG4/geom/crv_counters_v10.txt", f"{GEOM_DIR}/mu2e_run2_v10.gdml",
    RUN2_SECTORS, 0.84, RUN2_CUTS, False, ONSPILL_WINDOW, lib_group={n: "EX" for n in RUN2_SECTORS}, showers=False,
    delta_cut_mev=0.361, below_world=True, below_margin=20000.0,
    geom_text="Offline/Mu2eG4/geom/geom_run2.txt")

REGISTRY = {rc.name: rc for rc in (EXTRACTED_V04, RUN1A_V01, RUN1B_V01, RUN2_V10)}
DEFAULT = "extracted_v04"


def get(name=None, window=None):
    """Run configuration by name (None = extracted_v04); window: "offspill" / "onspill" / (t0, t1) overrides it."""
    rc = REGISTRY[name or DEFAULT]
    if window is not None:
        w = {"offspill": OFFSPILL_WINDOW, "onspill": ONSPILL_WINDOW}.get(window, window)
        rc = dataclasses.replace(rc, window=tuple(float(x) for x in w))
    return rc


def names():
    return sorted(REGISTRY)


def nominal_conditions(geom, pedestal=100.0, pulsearea=394.6, pulseheight=11.4):
    """chain.Conditions for a configuration without a dumped conditions table: the CRVCalib nominal constants of the
    extracted Sim_best table (pedestal 100, area 394.6, height 11.4; no time offsets), every channel live, no photon-
    yield deviations, and FEBs as consecutive 64-channel blocks (CRVId::nChanPerFEB; 24 FEBs per ROC). The FEB map only
    groups channels for the per-(event, FEB) TDC0 phase; the stock channel map differs in which channels share a FEB."""
    from . import chain
    n = 4 * geom.nbars
    ch = np.arange(n)
    feb = ch // 64
    return chain.Conditions(np.full(n, pedestal), np.full(n, pulsearea), np.full(n, pulseheight), np.zeros(n),
                            np.zeros(n, np.int64), (feb // 24).astype(np.int64), (feb % 24).astype(np.int64), np.zeros(n))


def sector_type_names(geom, counters_cfg):
    """{sectorType: name} from the counters file: the sector names of each type joined (Run 2: 1 R1-R6, 3 T1-T5, ...)."""
    out = {}
    for n in geom.sector_names:
        out.setdefault(int(counters_cfg[f"crs.sectorType{n}"]), []).append(n)
    return {t: (v[0] if len(v) == 1 else f"{v[0]}-{v[-1]}") for t, v in out.items()}
