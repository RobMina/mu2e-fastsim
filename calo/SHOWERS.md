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

## Status and scope

A 2,000-event stock pilot supplies 1,042 direct-electron front entries: 505 development (20,336 CSS, 17 empty
showers) and 537 held out. Of 1,195 shower-positive events, 189 have no direct-electron front entry; they remain in
the census but are outside this library's scope.

This is a development-only library builder, not a validated replacement for the calorimeter acceptance proxy.
The response application and complete CE-chain comparison are separate follow-ups; no default is changed.
The development library and all raw results stay outside the repository.

Dependencies: NumPy and uproot. From the umbrella checkout, use `PYTHONPATH=calo:core`;
the showers subpackage also works as a namespace package before the calorimeter package scaffolding is merged.

Tests: `PYTHONPATH=calo:core python3 -m pytest -q calo/tests/test_showers.py calo/tests/test_showers_export.py`.
