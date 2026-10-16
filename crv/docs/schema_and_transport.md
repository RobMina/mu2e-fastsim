# CrvStep table schema and muon-transport design (draft, 2026-10-16)

The implemented table format and column meanings are in `crvfastsim/README.md` section 2 (authoritative); this page
keeps the design reasoning.

## 1. Interchange tables (one ROOT file or Parquet set per shard)

The reference side writes them with the art module `CrvStepTableDumper` (`art/CrvFastSim`). The fast sim writes the
same columns, so one comparison harness reads both.

| table | one row per | columns |
|---|---|---|
| `gen` | CRY primary, every generated event | run, subrun, event, idx, pdg, genid, x, y, z, px, py, pz, e, t (after `cosmicTimeOffset`) |
| `steps` | CrvStep | run, subrun, event, nstep, bar, sp (SimParticle key), edep (visible, MeV), t0, t1 (ns), x0..z1 (mm, Mu2e), px0..pz0 (MeV/c), p1 (\|p\| at end), len (mm, true path) |
| `particles` | SimParticle in the accepted event | run, subrun, event, key, parent, pdg, ccode, scode, isprim, genidx, start/end position, 4-momentum, time |
| `subruns` | subrun | run, subrun, livetime (s), primaries |

- **Event identity.** The (run, subrun, event) triple is shared with the G4 reference: the fast sim reads `gen` and
  keeps the triple, so every comparison is event by event, matched on truth-particle identity.
- **Primary muon.** The particle with `isprim == 1` and \|pdg\| == 13. Its CrvSteps are the V1 sample.
- **Why a custom dumper.** uproot cannot read the art products themselves:
  - `CrvStep` fails its strided-size check (174 bytes per record on disk, the model expects fewer);
  - `SimParticle` and `GenParticle` use memberwise streaming, which uproot does not support.

  The raw `CrvStep` layout was decoded by hand: a 16-byte vector header, then members column-wise. It would work,
  but a 50-line art module is the maintainable route.

## 2. What a cosmic crosses before the CRV (from the v04 GDML)

- **Where primaries start.** CRY primaries start at y of about 15.4 m (the world top). The extracted CRV sits at
  y = 4.24-4.71 m.
- **What lies above it:** the backfill/overburden (`MBOverburden`, `backfillPsHatch*`, up to y of about 8.3 m), the
  building slabs and walls (`CONCRETE_MARS`), the DS-area hatch blocks and G-blocks (y of about 3.6-4.1 m, i.e.
  mostly *below* EX: they matter for upward-scattered secondaries and the T1 sector), and the CRV's own aluminium
  sheets and strongbacks.
- **Solid types.** The building is ~400 `xtru` prisms (rotated placements, extruded along y), plus boxes,
  subtractions and unions. The CRV is boxes only (V0: no rotations on any path to a bar).

## 3. Transport design (tier 1a: muons)

1. **Material map, not a live G4 geometry.** Pre-compute once, from the GDML, a 2D map over the CRY generation plane
   and incidence direction.
   - Preferred form: for each (x, z) cell on a plane just above the CRV, and each direction bin, the integrated
     column density per material class along the ray up to the world top.
   - Alternative: a voxelised material grid (5-10 cm voxels over the overburden column) that a JAX kernel
     ray-marches.
   - Both need correct handling of rotated `xtru` and of daughter volumes displacing their mothers. Use the deepest
     containing volume, decided by point-in-solid tests on a voxel grid, done once on a GPVM.
2. **Per muon** (vectorised over muons): a straight line from the CRY start point.
   - Mean energy loss with Bethe-Bloch per material class (Mu2eKinKal/KinKal-style constants; PDG tables for
     concrete and dirt), sampled with Landau/Vavilov fluctuations in thin slices.
   - Highland multiple scattering applied as a random walk per slice, which shifts position and direction at the
     CRV entry plane.
3. **Inside the CRV:**
   - ray-box crossings of the 576 bars from `crvfastsim.geometry`;
   - per crossing, the true path length; visible energy from a Landau/Vavilov in polystyrene with Birks saturation
     (G4 default kB for `G4_POLYSTYRENE`, to be matched to the reference);
   - delta rays above a cut kept in the same bar, as G4 mostly does;
   - aluminium sheets and strongbacks as slabs between layers;
   - times from the line, \|p\| from the running energy.
4. **Validation order.**
   - V1 needs exactly this: per crossing, path length and visible energy against the G4 primary-muon CrvSteps.
   - Then the muon-descendant energy (delta rays), then tier 1b for the non-muon component.

## 4. Open design choices

- Material classes: concrete (`CONCRETE_MARS`, `ShieldingConcrete`), overburden (`MBOverburden`), air, aluminium,
  steel. Collapse the rest into these by density and Z/A, or keep exact compositions from the GDML `<material>`
  blocks? Exact is cheap once the map is precomputed.
- Whether the material map needs direction bins at all, or whether a voxel ray-march on the GPU is fast enough to
  replace it. Decide after measuring both on one A100 against P1 (2 ms per accepted event on one CPU core).
