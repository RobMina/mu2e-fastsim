# Native stock-frame overlay and occupancy validation

Companion to the [sampler/readers](README.md); requires already-built tracker
and CRV analyzer plugins and a matching released Offline runtime.

`run_native_diagnostic.sh CE.art DONOR.art` uses one donor frame per primary via
stock `ResamplingMixer`, then stock tracker and CRV response/reconstruction.
This is an input-wiring diagnostic, **not** the four-stream multiplicity law.
Its runtime paths come from environment variables; use only already-built
compatible plugins and a released Offline backing. It does not build software.

## Native-art overlay preparation

`python3 -m mixing.native donors.json --parent-fcl campaign.fcl --bunch-fcl OneBB.fcl
--out mix.fcl` writes input bindings for the stock `MixBackgroundFrames` modules.
The parent must wrap stock `Mix.fcl` with the reference campaign's geometry,
conditions and primary-specific filters. All four donor streams are required;
missing streams are not silently assigned zero rates. The generated FCL leaves
the stock intensity law and SimBookkeeper rates in the parent unchanged and adds
no timing offset. Donor identity sequences are saved for paired checks.

`--signal-tag fastSteps` redirects tracker and CRV step consumers to an existing
fast-step product while leaving stock calorimeter inputs alone. That producer
must be scheduled by the parent, with valid event-local particle references.
Native mixers retain donor GenParticles and SimParticles; the Python flat-table
producer must not be substituted without importing its corresponding generators.
These bindings are source-tested but not yet runtime-validated through digitisation.

## Native response and saved-digi comparison

`run_response.sh CONFIG.fcl SIGNAL.art [STOP.art ELE.art MU.art NEUTRAL.art]`
uses the same environment variables as the diagnostic driver. With
`fcl/four_stream_response.fcl`, it runs all four native mixers with database
SimBookkeeper efficiencies, the OneBB bunch law, common timing and stock
tracker/CRV response. The FCL pins the selected campaign's Run1A geometry and
`Sim_best/v1_3`; it is not a general campaign default. It does not run calorimeter
response or trigger selection. The default initial skip bounds are from the
reference job; donor files are supplied separately. The ten-event wiring test
used zero initial skip and is distinct from the development comparison.

`fcl/four_stream_occupancy.fcl` includes the same response configuration, disables
only the diagnostic CRV charge and waveform-MC output tables (`crvdump.mc: false`),
and enables stock `TimeTracker` module summaries. All response producers, physics
settings, donor ordering and random seeds remain unchanged. Use a development
input for performance work; compare against its saved counts with `--stock`.
The report includes changed-event counts per detector as well as paired moments,
so equal totals cannot hide compensating event-level changes. This option does
not rerun held-out validation. On the complete 2,741-event development input,
all event IDs and plane/sector count arrays were unchanged. ROOT output fell
from 129.0 to 59.7 MB. Art elapsed time was 422.77 s versus the saved 436.47 s;
the runs used different nodes, so this small difference is not a controlled
speedup. The profile attributes 95.8 ms/event to `makeSD`, 25.8 ms to the four
mixers and 20.4 ms to CRV photons, out of 150.1 ms/event. Tracker digitisation,
not diagnostic output, is the main target for a substantial improvement.

With `fcl/selected_digi_dump.fcl` and no donors, the driver instead reconstructs
CRV pulses/clusters on **saved digis**, retaining the saved tracker digis. It does
not mix or digitize again. `mc: false` disables unsaved CRV charge truth inputs,
not the simulation calibration. An absent optional MC table is summarized as
`null`, never as a measured zero.

`python3 -m mixing.counts response.root --geometry mixing/run1a_geometry.json
--selection 'reference-selected events' --out counts.npz` writes complete IDs
and per-event plane/sector counts. `--stock reference.npz` aligns the exact ID
intersection and reports unmatched events plus paired errors. All 36 straw
planes and every mapped CRV sector retain empty bins and empty events.
`run1a_geometry.json` explicitly contains **only T1/T2**; it cannot validate side
sectors. A trigger-selected reference supports only a selected-event diagnostic,
not an unconditional occupancy closure or a claim of complete CRV coverage.

## Occupancy comparison helpers

`mixing.compare.occupancy` bins jagged straw-digi or CRV-pulse channel IDs into an
explicit sample-matched geometry map, retaining empty events and bins.
`paired_counts` takes already event-aligned count matrices and reports mean
counts, differences and ratios with paired delta-method errors and per-bin
pulls. Its reported chi-square is **diagonal only**, not a full correlated-shape
test; the caller must state that limitation or aggregate independent shards.
These helpers are synthetic-tested and used in the selected-event comparison.
Errors are conditional on finite donor pools reused sequentially with wrapping;
they do not include donor-library uncertainty.

## Measured selected-CE check

After a development shard, fixed settings were applied once to a held-out shard:
2,720 finished candidate events, of which 2,089 match saved selected stock IDs.
The native candidate uses stock primary steps, four stock streams and database
multiplicity factors, not fast transport. Overlay/stock totals are
`0.9915 +/- 0.0165` for straw digis and `1.0030 +/- 0.0202` for CRV pulses.
Diagonal chi-square/ndf is `26.69/36` across planes and `7.88/2` across T1/T2.
T2 alone is `0.7611 +/- 0.0740`, paired pull `-2.78`; this discrepancy is retained,
not tuned away. No lateral sectors exist in this geometry. Independent pileup
and response on stock-selected IDs are not a replay of the stock trigger.
The measured mixer+tracker/CRV-response+flat-output boundary is 6.37 events/s;
it excludes environment startup, calorimeter and trigger. No matched stock-grid
speed comparison or end-to-end fast-chain speedup is established.

`python3 -m mixing.plot_counts counts.json --out occupancy.png` renders the
measured overlay, ratio and pull panels with the selection/coverage caveats.
