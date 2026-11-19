"""crvfastsim.lookupgen: an on-device generator of CRV photon lookup tables (version-6 format).

A JAX port of the table mode of Mu2e/CRVStandalone (the program that makes the released CRVConditions lookup tables):
photons generated per bin as WLSPrimaryGeneratorAction does, transported with the Geant4 10.7 optical physics of the
stock counter model (TiO2 coating, fiber holes, double-clad Y11 fiber, guide bars, SiPM windows or reflector), and the
SiPM-0 arrivals histogrammed and written as the stock merged table. Counter dimensions that are compile-time in the
stock program are runtime parameters here (geometry.Counter, geometry.design).

Modules: materials (optical tables, Geant4 vector semantics), geometry (counter, bin edges), transport (device
kernel), writer (v6 files), compare (bin-by-bin comparison with a reference table). CLI: python -m crvfastsim.lookupgen.
"""
