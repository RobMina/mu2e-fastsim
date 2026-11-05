"""Geant4-free muon beam transport from the DS entrance (TS5 exit) to the stopping target, batched in JAX.

Modules: simpleconfig (Offline geometry text files), fieldmap (G4BL grid maps, trilinear on device), geometry (target
foils, TSdA, frame), transport (RK4 in the field map + foil energy loss, scattering, range-out, decay in flight),
g4ref (per-muon reference table from the stock MuBeamResampler job), compare (fast vs G4 metrics and plots).
"""
