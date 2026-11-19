"""crvfastsim.lookupgen: an on-device generator of CRV photon lookup tables (version-6 format).

A JAX port of the table mode of CRVStandalone (the program that makes the released CRVConditions lookup tables):
photons generated per bin as WLSPrimaryGeneratorAction does, transported with the Geant4 10.7 optical physics of the
counter model (TiO2 coating, fiber holes, double-clad Y11 fiber, guide bars, SiPM windows or reflector), and the
SiPM-0 arrivals histogrammed and written as the stock merged table. Counter dimensions that are compile-time in the
stock program are runtime parameters here.

Two layers (docs/lookupgen.md, "Layout"):

  core   experiment-neutral and extractable: strip geometry (counter), bins (binning), Geant4 optical building blocks
         and the model registry (optics), the literature-based ``current`` optical model with its sources
         (materials), the device transport (transport) and the host driver (run). Imports only the standard
         library, NumPy and JAX.
  mu2e   the Mu2e adapter: CRV counters and the stock bin-edge rule (sections), the test-beam-tuned ``v6`` model (v6),
         the version-6 file writer (writer), comparisons with released tables (compare) and library-core manifests
         (library).

CLI (Mu2e): python -m crvfastsim.lookupgen.
"""
