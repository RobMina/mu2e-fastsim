# CRV photon lookup-table generator (`crvfastsim.lookupgen`)

The CRV response (`MakeCrvPhotons` in Offline, and `crvfastsim.response` here) turns energy deposits into photons at
the SiPMs through lookup tables: per bin of position (and particle speed / direction for Cerenkov light), the
probability that a photon reaches SiPM 0, its arrival-time distribution and its number of fiber emissions. The
released tables (CRVConditions v6) were made by the table mode of the standalone optical simulation
(Mu2e/CRVStandalone, `wls -c`): Geant4 tracks 10^5 photons per bin (10^4 for Cerenkov light in the fiber), about 3,000
CPU jobs per table. Its counter cross-section and fiber are compile-time constants.

`crvfastsim.lookupgen` is a JAX port of that table mode that runs on a GPU (or CPU) and takes the counter dimensions
at run time. It writes the same version-6 file.

## Two optical models

The released v6 tables were not made by today's CRVStandalone. The v6 format dates from December 2018, and the tables
were made by the standalone code then in Offline (January 2019, Geant4 10.3). CRVStandalone was updated in July 2021.
Both versions are here (`--model`):

| | `v6` (default: made the released tables) | `current` (CRVStandalone since 2021) |
|---|---|---|
| TiO2 reflectivity | 1 - 0.9 (1 - R_measured) | 1 - 0.75 (1 - R_measured) |
| Y11 attenuation, decay time | measured x 0.80, 8.7 ns | measured x 0.85, 10 ns |
| PPO / POPOP emission spectra | 2019 tables | updated tables |
| PS+PPO / POPOP quantum yield, decay times | 0.83 / 0.86, 1.68 / 1.5 ns | 0.94 / 0.97, 1.36 / 1.12 ns |
| fiber end | SiPM (R = 0.30) or reflector (R = 0.90) disc on the fiber end | 1 um air, 2 x 2 mm SiPM with epoxy window (R = 0.20), reflector R = 0.83 |
| fiber / air interfaces | polished | unified, ground (sigma_alpha 0.02) |
| table writer | probability over all arrivals, histograms normalised to all arrivals | only arrivals inside both histograms (< 150 ns, < 10 emissions) |

Ported from the current model alone, the generator gives 16 % more light than the released tables (3200 mm counter,
scintillation, 58 bins: ratio 1.164 +- 0.018) with the same time and emission distributions. The `v6` model is the
default, so that tables for other counters are made with the optics of the released ones that Offline uses.

## Agreement with the released tables (`v6` model)

Bin by bin at stock statistics (`compare`: arrival probability with binomial errors on both sides, means of the
time and emission distributions); ratio = generated / released arrivals, summed over the compared bins.

| table part | counter | bins | float32 | float64 |
|---|---|---|---|---|
| scintillation | 3200_0 | 294 | 1.008 +- 0.009 | 1.009 +- 0.009 |
| scintillation | 2370_1 | 60 | 0.988 +- 0.023, 1.003 +- 0.021 | |
| Cerenkov in the scintillator | 3200_0 | 61 / 38 | 0.999 +- 0.038, 0.993 +- 0.034 | 0.90 +- 0.08, 0.96 +- 0.04 |
| Cerenkov in the fiber | 3200_0 | 296 / 125 | 1.22 +- 0.065 | 1.21 +- 0.13 |

Time and emission pulls average within +-0.4 in every set. The fiber-Cerenkov excess is not understood yet: it
comes entirely from the bins that start their photons in the hole air (the third quirk below) and is largest for the
fastest particles (1/beta < 1.24).

## What is ported

| stock (standalone optical simulation, Geant4 10.3 / 10.7) | here |
|---|---|
| bins and sources: `WLSPrimaryGeneratorAction` (scintillation, Cerenkov in the scintillator, Cerenkov in the fiber) | same sampling; scintillation time drawn as the exact sum of the decay and rise exponentials |
| geometry: `WLSDetectorConstruction` (TiO2 coating, rounded corners, elliptical holes, double-clad fiber, guide bars, SiPM / reflector) | analytic ray tracing on the same solids |
| materials and surfaces: `WLSMaterials`, the surface tables | the same tables with Geant4's property-vector semantics (insertion sorting, clamped linear interpolation, `GROUPVEL` from `RINDEX`) |
| `G4OpBoundaryProcess` (dielectric-dielectric Fresnel with polarization, unified-model facets, dielectric-metal) | ported line by line, including the reflection-type constants that a surface without a table inherits from the last unified surface hit |
| `G4OpWLSY11`, `G4OpWLSPSPPO`, `G4OpWLSPOPOP` | same (quantum yield, emission below the primary energy within 100 tries, exponential delay) |
| `WLSEventAction` table writing | byte-compatible encoding (float32 probability, uint8 histograms) |

Three properties of the stock tables are reproduced on purpose:

- the Y11 decay times are inside the stored time delays, and the header's fiber decay time is 0,
  because the table mode sets an unused property (`WLSTIMECONSTANT`) to zero;
- the fiber-Cerenkov table starts its photons in the fiber at +y but scores SiPM 0, which reads the fiber at -y;
- the fiber-Cerenkov start position is x = r_min + r (the sampled radius is added to the bin's lower edge), so the
  outer radius bins start outside the core, in the claddings or the hole air; each photon starts in the volume that
  contains its start point, as Geant4 locates it.

### Exact shortcut

A ray in the fiber core whose next hit on the core/cladding interface is a total internal reflection hits it at the
same angle every time (concentric polished cylinders). It is moved to its Y11 absorption point or to the fiber end in
one step along the reflected path: same path length, time, position, direction and polarization as bounce by bounce.
The same holds for a ray confined to the inner cladding annulus. `Config(helix=False)` turns it off; the slow test
compares both.

### Departures (small, counted)

- `current` model only: the SiPM window's side faces are treated as silicon; the gap between fiber end and SiPM is
  closed at the SiPM edges (a photon leaving it sideways is dropped). Stock lets the 1 um gap open into the world.
- `v6` model with a fiber other than 1.4 mm: the SiPM / reflector disc has the fiber radius (stock: 0.7 mm, the
  1.4 mm fiber), so it covers the whole fiber end.
- Rejection loops that Geant4 runs without limit have a limit: facet sampling (24 sub-iterations of 8 candidates,
  `forced` counts lanes left over), Cerenkov energy (25,600 candidates, `badsrc`), chains longer than 2^19 steps
  (`capped`; about 1 photon in 3 million). The step limit is set by the random-number counter (19 bits of step
  index per photon). `len_end` and `len_arr` in the run statistics histogram the chain length (bin k: 2^k to
  2^(k+1) - 1 steps) of every chain and of every SiPM-0 arrival, to measure what a lower limit would drop.
  `--max-steps` sets a lower limit. Measured on a complete 24.71 x 25 mm, 3200 mm table part: of 2.7 million
  scintillation arrivals, 4 come from chains of 2^16 steps or more (1.5e-6) and 143 from 2^15 or more; of 200,000
  fiber-Cerenkov arrivals, none from 2^16 or more. `--max-steps 65536` cuts the per-job tail about 8 times.
- Random numbers come from a counter-based generator keyed by bin and photon, so results are reproducible and
  independent of how bins are split into jobs, but not the same random sequence as Geant4.

## Counters other than stock

`geometry.design(thickness, width, length, reflector, fiber_diameter)` builds a two-fiber counter with the stock fiber
separation ratio (26 / 51.34 of the width), holes with the stock clearance around the fiber (semi-axes r + 1.3 mm
across the thickness, r + 0.3 mm across the width) and cladding layers of 3 % of the fiber diameter each. The stock x / y bin edges are hard-coded for the 19.78 x 51.34 mm counter; other counters get the same pattern
moved to their dimensions (`geometry.general_xy_edges`). z, beta, theta, phi and r bins follow the stock formulas.

The tables are normalised as the released ones: probability per photon generated in the bin. A fiber other than
1.4 mm therefore enters through the table, and the response uses the 1.4 mm-fiber scintillation yield.

## Running

```bash
python -m crvfastsim.lookupgen run --length 3200 --reflector 0 --simtype 0 --bins 0:2296 --dtype float64 --out s0.npz
python -m crvfastsim.lookupgen compare s0.npz LookupTable_3200_0          # bin by bin against a released table
python -m crvfastsim.lookupgen assemble --length 3200 --reflector 0 --out LookupTable_3200_0 s0.npz s1.npz s2.npz
```

`--section 24.71x25` (thickness x width, mm) selects a design counter. `--dtype float32` runs the transport in single
precision. The released tables are Mu2e-internal and are not part of this repository.

`assemble` refuses pieces with incomplete bins or made for another counter or model, and writes
`<table>.provenance.json` next to the table (the version-6 file has no metadata field): counter, optical model,
the reproduced stock quirks, per table part the pieces with photons per bin, dtype, step limit and capped chains
(also of the run a piece was resumed from), and `--note` lines.

`--checkpoint` rewrites `--out` while running (`--checkpoint-interval` seconds apart) with the bins that are complete;
`--resume` keeps those and runs only the others. Random numbers are keyed by bin and photon, so a resumed or split run
gives the same counts as one run.

### Cost

On an A100 (40 GB), scintillation bins of the 3200 mm counter run at about 330,000 photons/s in float32 and 180,000 in
float64 (4 CPU cores: 700 and 450). Photons trapped in the fiber by total reflection at the outer cladding (skew
rays that cross it nearly transversely) bounce until the Y11 dye absorbs them, up to a few 10^5 steps, as in Geant4:
about 1 photon in 10^4 takes more than 7 x 10^4 steps. The last of them set a fixed tail of about 20 minutes per job
on the A100, so jobs should cover a whole table part.
