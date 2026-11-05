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

## cechain / cesob adapter

```python
from calofastsim.showers.handoff import from_cechain
out = from_cechain(result, field, library, rng)
```

`result` is cechain's Result including absolute times. The adapter retains its generated-event denominator,
propagates tracker-entry states with the same midpoint-field helix stepping used by cesob, selects the first forward
disk-front entry, and returns query states plus sampled, placed CSS for the calorimeter response. Existing cesob NPZs omit times and
are not silently assigned fabricated times. No beam file is modified.

**This is a diagnostic adapter only:** it does not transport tracker/passive material or side-entry interactions.
Full generated-event comparisons and timing are below; their denominators include non-front-entry events.
The proxy remains unchanged.

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
sample. The following full-chain measurement uses the same frozen model, not a post-confirmation retune.

## Full generated-event measurement

`prepare_chain` extracts the exact generated electron starts, full-event stock visible/response energies,
per-event timing and (optionally) stock leading clusters. `measure_chain` runs CE primary transport, the disk-front
adapter, frozen showers and response; it separately evaluates the cesob inner-wall proxy and its vacuum variant
on those same tracker-entry states. `reconstruct` connects the sampled CSS to the calorimeter digi/template-fit/
hit/cluster chain. No source Geant4, digitization or reconstruction reference is regenerated.

For the **1,000 held-out generated CE events**, double-precision device transport followed by the host disk-front
adapter and the frozen library gives:

| Selection | Fast | Stock | Paired fast/stock ratio |
|---|---:|---:|---:|
| Visible deposited energy >10 MeV | 543 | 566 | **0.9594 ± 0.0222** |
| Timed deterministic response sum >10 MeV | 422 | 441 | **0.9569 ± 0.0204** |
| cesob inner-wall proxy vs visible-deposit selection | 566 | 566 | **1.0000 ± 0.0139** |
| cesob vacuum proxy vs visible-deposit selection | 576 | 566 | **1.0177 ± 0.0157** |
| Reconstructed leading-cluster energy >10 MeV | 390 | 410 | **0.9512 ± 0.0224** |

The proxy targets visible-deposit acceptance and **must not be compared directly with the timed cluster selection**.
The new shower visible acceptance is 2.30 ± 0.78 percentage points below the wall proxy on identical fast states.
For reconstructed clusters, the all-event mean energy difference is −1.47 ± 0.82 MeV and the energy shape χ²/ndf
is 28.96/19. On the 356 events where both arms pass, the time difference is +0.899 ± 0.332 ns and the mean paired
transverse position separation is 251 mm. The large positional disagreement is not a good resolution or validation
result. There are 34 fast-only and 54 stock-only selected events. Independent fast/stock stochastic response and
one finite development library are used; the errors exclude library-training uncertainty.

**Measured complete-route throughput:** 1,000-event batches, one cold and two identical-seed warm calls, from
CE birth through reconstructed clusters. CPU and GPU use the same double-precision JAX transport and retain the
same acceptance counts. Device synchronization, host propagation, shower placement, response, digitization,
fitting and clustering are included; input loading and first-call compilation are separate.

| Allocation | First call | Median warm call | Generated CE/s | Stock grid-slot rates (23.9 CE/s) |
|---|---:|---:|---:|---:|
| One CPU core, JAX CPU | 134.44 s | **97.672 s** | **10.24** | **0.428** |
| RTX PRO 6000 1g.24GB MIG + four allocated host cores | 35.68 s | **14.162 s** | **70.61** | **2.954** |

Warm CPU transport/downstream times are 46.43/51.24 s; MIG transport/downstream 10.84/3.32 s. The wall proxy is a
separate comparison and not included in the new shower-chain timing. The fast route is primary-only, omits
tracker/passive material and fails the shower validity requirements; the stock timing covers the full production
CE simulation stage, not a matched reconstruction workload. These are rate comparisons, not validated replacement
claims or measurements of large-batch steady-state device saturation. Legacy NumPy transport did not finish its
first measurement before it was stopped; no rate is inferred from that partial run.

Reconstruction uses the calorimeter source at `656bd9b` with its saved timing snapshot; transport uses umbrella
`7bb64cd` and core `d041bb3`. The shower implementation is published in this PR series; the measurements used its frozen pre-publication snapshot.
No defaults changed. Package/CLI naming and test-file organization do not change the measured physics.

```bash
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.prepare_chain \
  --tables TABLES.root --response RESPONSE.root --response-source calo/calofastsim/response.py \
  --sim-crystal Sim_crystal.txt --geometry geometry.resolved.txt --stock-ntuple STOCK.root \
  --out inputs --digitization-start 450 --digitization-end 1795 --digitization-buffer 10
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.measure_chain \
  --inputs inputs/chain_inputs.npz --library ce_showers_dev.npz --geometry inputs/geometry.txt \
  --maps DSMap.header TSdMap.header --response-source calo/calofastsim/response.py \
  --out chain --arm heldout --backend device --device gpu --repeat 3 \
  --digitization-start 450 --digitization-end 1795 --digitization-buffer 10 \
  --umbrella-commit UMBRELLA_SHA --core-commit CORE_SHA
PYTHONPATH=calo:beam:core python3 -m calofastsim.showers.reconstruct \
  --chain chain/chain.npz --chain-report chain/numbers.json --inputs inputs/chain_inputs.npz \
  --library ce_showers_dev.npz --maps DSMap.header TSdMap.header \
  --reco-code calo/calofastsim --reco-inputs RECO_INPUTS --reco-commit RECO_SHA \
  --live-transport device --transport-geometry inputs/geometry.txt --out complete --device gpu
```

The last command repeats the same transport/draws to time the full route and checks exact saved-state equality.
It does not draw a new held-out sample or choose a candidate. Omit `--live-transport` to reuse the saved states;
that mode reports a **sum of separately measured stages**, not a single-call end-to-end wall time.
`chain_report` renders saved summaries without re-executing held-out physics.

Tests: `PYTHONPATH=calo:core python3 -m pytest -q calo/tests/test_showers*.py`.
