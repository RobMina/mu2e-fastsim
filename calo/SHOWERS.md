# CE whole-shower library (opt-in)

This module samples deposited patterns from saved conversion-electron `CaloShowerStep`s. It does not run Geant4,
change a chain default, or replace `beamfastsim.cesob`'s acceptance proxy. Entry energy/direction/radius are
**simulated incident-state inputs**, not truth-conditioned corrections to measured reconstruction.

## Build

The reader accepts the numeric `CaloLibraryDumper` export of a single-CE StageParticle sample: `dump/events`,
`generated`, `particles`, `products`, `steps`, `showers`, `crystals`, `disks`, and `virtualdetectors`. It selects the
first forward **direct-electron** disk-front VD crossing (73 or 75), matched by persistent particle identity.
The StageParticle upstream parent is reseated into the raw g4run product under the same key. Other input formats
need their own adapter; matching by proximity is not used.

```bash
PYTHONPATH=calo:core python3 -m calofastsim.showers TABLES.root \
  --out ce_showers_dev.npz --split-row 1000 \
  --source-commit EXPORTER_COMMIT --code-commit LIBRARY_COMMIT
```

The first 1,000 source event rows are development; the remaining rows are held out. The event split is chosen
before examining shapes, and is **not independent production-shard validation**. The builder writes an NPZ
without pickle plus a short JSON description. Nothing is renormalized to total incident energy: leakage is already
part of the sampled deposits. The whole event's CSS, both disks and empty patterns remain correlated.

Stored per-shower features are total entry energy, forward cosine, radius and disk. Default bin edges are
`[0,50,80,95,110] MeV`, `[0,.4,.7,1.000001]`, and `[0,450,550,750,1000] mm`. Each CSS retains its G4/Birks
energies and time/position relative to the entry time and momentum frame. Donors are uniform within a bin.
An empty bin uses the nearest occupied bin in bin-index distance **within the same disk**, with a returned
`fallback` flag. A disk without donors is explicitly unsupported, not a genuine zero-response draw.

## Apply and response hand-off

`sample(library, entries, rng, orientation="disk")` rotates a whole donor pattern about the disk axis and translates
it to the query entry. It preserves longitudinal deposit positions relative to the entry, so the second disk does
not rotate out of the detector. Exact incident-direction rotation is also available as `orientation="incident"`;
it is unsuitable for transporting a two-disk pattern in a fixed detector. Neither placement rescales energy.

`place` associates the transformed deposits with crystal boxes using their saved geometry, and returns the crystal
ID, crystal-local position, global position and an explicit `mapped` mask. It checks up to four nearby boxes and
never snaps an outside deposit into a crystal. Unmapped deposits are a model loss, reported separately. Crystal
size comes from the exported geometry-service cell dimensions; wrapper/active-volume differences are not resolved.

`cluster_sum` is a temporary **one-cluster-per-entry energy-weighted sum**, not a fitted CaloCluster. It accepts
calorimeter corrected energies and time-acceptance masks. The saved comparison imports the calorimeter `response.py` explicitly,
uses the saved per-event timing and per-crystal LRU, and applies the same deterministic response to both arms:

```bash
PYTHONPATH=calo:core python3 -m calofastsim.showers.compare \
  --library ce_showers_dev.npz --input TABLES.root \
  --response-input RESPONSE.root --response-source calo/calofastsim/response.py \
  --out comparison --arm heldout --seed 20261105 \
  --digitization-start 450 --digitization-end 1795 --digitization-buffer 10
```

Those timing values belong to this saved control; supply the actual values for another input. The script produces
an energy overlay with paired pulls, `numbers.json`, and numeric cluster arrays. Errors use paired jackknife over
100-source-event blocks; the shape chi-square uses diagonal paired block variance and no fitted normalization.
The shared library's training uncertainty is not included. The E/p entry uses **true disk-front momentum**, and is
labelled a truth diagnostic, not reconstructed E/p. Package dependencies are NumPy, SciPy, uproot and matplotlib;
the hand-off additionally needs `mu2efastsim`.

## Pilot result and current limit

A 2,000-event stock pilot supplies 1,042 direct-electron front entries: 505 development (20,336 CSS, 17 empty
showers) and 537 held out. Of 1,195 shower-positive events, 189 have no direct-electron front entry; they remain in
the census but are outside this library's scope.

On the held-out front-entry population, sampled vs original CSS through the same deterministic response and simple
cluster sum gives:

| Quantity | Result |
|---|---|
| Acceptance above 10 MeV | ratio **0.9896 ± 0.0201**; difference **−0.745 ± 1.435 percentage points** |
| Mean energy difference | **−4.67 ± 1.46 MeV** |
| Energy shape, 5 MeV bins | **χ²/ndf = 52.1/21** |
| Empty-bin fallback | 16/537 entries; no unsupported disk |
| Visible energy outside mapped crystals | **10.0%** |

The energy shape is not adequate for proxy replacement. Independent production-shard confirmation and
side/photon entries remain open. Do not tune on these held-out results; any revised model needs a new confirmation
sample. The complete CE-chain measurement is a separate follow-up using this same frozen model, not a retune.

Tests: `PYTHONPATH=calo:core python3 -m pytest -q calo/tests/test_showers*.py`.
