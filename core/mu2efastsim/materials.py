"""Mu2e materials of the kernels (hep-simkernels): the materials of the Mu2e geometry, with density, I and composition
as in the v04 GDML export (`<material>` blocks, read 2026-10-16 and 2026-10-23), registered into
hepsimkernels.materials, plus the Mu2e production cuts of G4_POLYSTYRENE (CRV and tagger regions). The Mu2e fcl
BirksConsts override is unset (Mu2eG4/fcl/prolog.fcl), so Geant4's Birks constants apply.

Every name of the former module stays: the NIST materials, Material, ELEMENTS, N_A, BY_NAME (the shared registry, in
the former order) and get() are hepsimkernels.materials' own.
"""
from hepsimkernels.materials import *  # noqa: F401,F403
from hepsimkernels.materials import BY_NAME, ELEMENTS, N_A, Material, add_cuts, get, register  # noqa: F401
from hepsimkernels.materials import (AIR, ALUMINIUM, CESIUM_IODIDE, COPPER, GOLD, LEAD, MYLAR, POLYETHYLENE,  # noqa: F401
                                     POLYSTYRENE, SILICA, SILICON, TUNGSTEN)

_CABLE_FIBER = {"C": 0.34093, "O": 0.28578, "Al": 0.28265, "H": 0.05676, "Si": 0.03172, "F": 0.00216}
CONCRETE_MARS = Material("CONCRETE_MARS", 2.35, 125.203942404978,
                         {"H": 0.006, "C": 0.03, "O": 0.5, "Na": 0.01, "Al": 0.03, "Si": 0.2, "K": 0.01, "Ca": 0.2,
                          "Fe": 0.014})
MB_OVERBURDEN = Material("MBOverburden", 2.15, 125.453468818982,
                         {"O": 0.518324115382827, "Si": 0.279959130320566, "Al": 0.201716754296607})
MILD_STEEL = Material("MildSteel", 7.86, 284.511116716604, {"Mn": 0.01, "Si": 0.0025, "C": 0.0025, "Fe": 0.985})
SHIELDING_CONCRETE = Material("ShieldingConcrete", 2.5, 129.122501214703,
                              {"O": 0.52, "Si": 0.325, "Ca": 0.06, "Na": 0.015, "Fe": 0.04, "Al": 0.04})
ELECTRONICS2 = Material("Electronics2", 4.52, 229.423798310015,
                        {"Si": 0.0393766268325182, "O": 0.0937139019072929, "Ca": 0.021183442718492,
                         "Al": 0.0107331999698117, "Mg": 0.00235184116252291, "B": 0.0036336834583261,
                         "Ti": 0.000747978346220884, "Na": 0.00115729823161608, "K": 0.00129503062544011,
                         "Fe": 0.000545552317870526, "F": 0.001092, "C": 0.0715765662980504, "H": 0.0089116569351508,
                         "N": 0.00368122119668749, "Cu": 0.74})
CUW1090 = Material("CuW1090", 17.3, 663.689239365592, {"W": 0.9, "Cu": 0.1})
ALUMINIUM_6061 = Material("Al6061", 2.70, 166.28607227249, {"Al": 0.98, "Mg": 0.01, "Si": 0.006, "Fe": 0.004})
ALUMINIUM_7075 = Material("Al7075", 2.81, 173.678276646971, {"Al": 0.903, "Zn": 0.056, "Mg": 0.025, "Cu": 0.016})
BRONZE_C938 = Material("BronzeC938", 9.4, 374.512803514625,
                       {"Cu": 0.76, "Pb": 0.145, "Sn": 0.069, "Ni": 0.01, "Zn": 0.008, "Sb": 0.008})
TRACKER_MANIFOLD = Material("TrackerManifold", 1.95, 140.792157928869,
                            {"C": 0.136446618831317, "H": 0.0171758962147145, "Cl": 0.201377484953969, "Al": 0.645})
G10 = Material("G10", 1.70, 96.1961250410254,
               {"Si": 0.280460305074916, "O": 0.395811062502435, "C": 0.275294485761732, "H": 0.0342756035967338,
                "N": 0.0141585430641826})
ELECTRONICS = Material("Electronics", 0.58, 226.40105018703,
                       {"Si": 0.103, "Al": 0.034, "Au": 0.034, "Cu": 0.691, "C": 0.118166876291217,
                        "H": 0.0198331237087827})
ALUMINIUM_HONEYCOMB = Material("AluminumHoneycomb", 0.03, 166.0, {"Al": 1.0})
CARBON_FIBER = Material("CarbonFiber", 1.422, 77.023718420056,
                        {"C": 0.794002107481052, "H": 0.0280294500649256, "O": 0.177968442454022})
CAL_CABLE_RUN_OUTER = Material("CalCableRunOuter", 6.844, 336.137388288734,
                               {"Ag": 0.52603, "Cu": 0.34825, "F": 0.09553, "C": 0.03019})
CAL_CABLE_RUN_FIBER = Material("CalCableRunFiber", 1.6575, 89.439662534022, _CABLE_FIBER)
TRK_CABLE_RUN_FIBER = Material("TrkCableRunFiber", 1.6575, 89.439662534022, _CABLE_FIBER)
TRK_CABLE_RUN_OUTER = Material("TrkCableRunOuter", 4.6254, 212.602169355846,
                               {"Cu": 0.71458, "C": 0.10259, "H": 0.02099, "N": 0.00197, "O": 0.06161, "Si": 0.09826})
POLYETHYLENE_092 = Material("Polyethylene092", 0.92, 56.6323346252475, {"H": 0.1428, "C": 0.8572})
PEEK = Material("Polyetheretherketone", 1.32, 35.719636793795, {"C": 0.513514, "H": 0.405405, "O": 0.081081})  # GDML fractions as given
LYSO = Material("Lyso_01", 7.4, 399.385743651319, {"Lu": 0.708935, "Si": 0.069895, "O": 0.17973, "Y": 0.03994, "Ce": 0.0015})
STAINLESS_STEEL = Material("StainlessSteel", 8.02, 280.927103569437,
                           {"Mn": 0.02, "Si": 0.01, "Cr": 0.19, "Ni": 0.1, "Fe": 0.68})

register(CONCRETE_MARS, MB_OVERBURDEN, MILD_STEEL, SHIELDING_CONCRETE, ELECTRONICS2, CUW1090, ALUMINIUM_6061, ALUMINIUM_7075, BRONZE_C938, TRACKER_MANIFOLD, G10, ELECTRONICS, ALUMINIUM_HONEYCOMB, CARBON_FIBER, CAL_CABLE_RUN_OUTER, CAL_CABLE_RUN_FIBER, TRK_CABLE_RUN_FIBER, TRK_CABLE_RUN_OUTER, POLYETHYLENE_092, PEEK, LYSO, STAINLESS_STEEL)
add_cuts("G4_POLYSTYRENE", crv=0.0872, tagger=2.2296)

# the registry in the order of the former module (iteration order is part of the outputs of some tables)
_ORDER = (POLYSTYRENE, ALUMINIUM, AIR, CONCRETE_MARS, MB_OVERBURDEN, MILD_STEEL, SHIELDING_CONCRETE, COPPER, SILICON, ELECTRONICS2, CUW1090, ALUMINIUM_6061, ALUMINIUM_7075, BRONZE_C938, TRACKER_MANIFOLD, G10, ELECTRONICS, POLYETHYLENE, SILICA, MYLAR, TUNGSTEN, GOLD, LEAD, ALUMINIUM_HONEYCOMB, CARBON_FIBER, CAL_CABLE_RUN_OUTER, CAL_CABLE_RUN_FIBER, TRK_CABLE_RUN_FIBER, TRK_CABLE_RUN_OUTER, CESIUM_IODIDE, POLYETHYLENE_092, PEEK, LYSO, STAINLESS_STEEL)
_rest = {k: v for k, v in BY_NAME.items() if k not in {m.name for m in _ORDER}}
BY_NAME.clear()
BY_NAME.update({m.name: m for m in _ORDER})
BY_NAME.update(_rest)
