"""crvfastsim -- Geant4-free, GPU-accelerated simulation of the Mu2e KPP (extracted) CRV cosmic sample.

One package + CLI, following the crvextrap conventions (mu2e_track_crv_mcresponse). Definitions live once in the
shared modules, documented in ``README.md``:

    geometry   CRV bar geometry from Offline's counters file (CosmicRayShieldMaker re-implementation) + GDML readers
    crossings  vectorised ray-box crossings of straight tracks with the bars (JAX, CPU or GPU)
    tables     the CrvStep-table interchange format (CrvStepTableDumper trees; fast-sim output uses the same columns)
    stats      robust statistics shared by every comparison (median, core sigma, MPV, binned ratio band)
    plots      shared plotting helpers + write_json

Run as:  .venv/bin/python3 -m crvfastsim <subcommand> <inputs> [...]   (from the repository root)
"""
__version__ = "0.1.0"
