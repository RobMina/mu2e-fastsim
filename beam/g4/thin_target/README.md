# thin_target: Geant4 single-interaction bank generator

Generates the geometry-free thin-target interaction bank used by `beamfastsim.ptgt_device` (opt-in): per projectile
species, material and log10(kinetic energy) bin, the macroscopic inelastic and elastic cross sections, stopping power
and radiation length (`xs` mode), a pool of single inelastic interactions with their secondaries in the projectile
frame (`inel` mode) and the elastic scattering angle (`el` mode). See the header of `thin_target.cc`.

Build against any Geant4 11 installation (`geant4-config` on the path):

    cmake -S . -B build && cmake --build build -j

`python3 -m beamfastsim.ptgt_bank` converts the binary outputs into the bank library file.
