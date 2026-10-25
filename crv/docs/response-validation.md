# Explicit-input response validation

`python3 -m crvfastsim.objective.validation` runs validation outside the BO board, with no new scheduler.
Use `inventory --inputs FILE... --exclude PILOT_FILE... --asset calib=CSV deviation=CSV reference=JSON
[background=NPZ time_shift=JSON] --out MANIFEST.json` first. It reads the highest-cycle `dump/gen` and `dump/steps`
event keys, checks that step events exist in gen, rejects input/excluded overlap by full run/subrun/event identity,
and hashes the inputs/assets. No input files are copied. Inventory refuses to overwrite an existing manifest.

`run --manifest MANIFEST.json --out NEW_DIRECTORY --seed SEED [--jax] [--calibration] [--knob KEY=VALUE ...]`
verifies the input hashes, requires clean tracked package checkouts, records code SHAs, resolved geometry/LUT/map/
waveform hashes and Python/package versions, and runs exactly one complete pass. Finished response exposure and
step counts must equal the inventory. Output directories must not exist; partial failures remain for diagnosis.
Calibration mode requires the background, time shift and reference assets and uses the strict 32-term score.
Closure mode keeps background/time shifts off, compares to a stock accumulator with identical exposure and reports
partial diagnostic terms; it is not an automatic pass/fail physics gate. All sector terms must be reviewed.

Use disjoint, full held-out files for separate nominal/best pairs and a fresh seed per pair; within each pair keep
inputs, seed and fixed settings identical. Compare paired changes rather than independent quadrature of histogram
sigmas. Two pairs are a consistency check, not a precision variance/convergence result. Keep the old BO board intact.

The first stock closure reuses `fiberprop_v1/score/stock_emit/nominal/001400_00000990.json` and production table
`prod_as_tables/crvsteps.001400_00000990.root`; confirm the saved stock accumulator against the highest-cycle stock
ntuple before claiming full acceptance. For held-out work the five pilot subruns are 990,1023,1056,1089,1122.
Remaining production tables are candidates only after event-key overlap and provenance checks.

`stock-count --ntuple ROOT --accumulator JSON --out RECEIPT.json` verifies the highest EventNtuple tree cycle's
entry count against the saved reference exposure, without rerunning the stock reconstruction.

October 25 first execution (GPVM04 CPU, CRV `8d42b1b`, core `a554571`, seed 20261025): the complete subrun 990
finished in 247.84 s, one pass, 22,573 step events + 2,603 empty events = 25,176 exposure events, 30,145 clusters.
The existing stock file's highest tree `EventNtuple/ntuple;3` has exactly 25,176 entries. Diagnostic objective
1.55786, conditional candidate/reference sigmas 0.86197/0.87459 over 32 terms. The largest term is sparse
`oot_pe_TAG` (90 candidate/84 reference entries; 38.66 +/- 26.77 conditional candidate sigma). This is a first
CPU closure observation, not GPU or formal stock-compatibility acceptance. Outputs and hashes:
`$CRVFS_DATA/thread_c_validation_20261025/{closure_manifest.json,stock_count.json,closure_nominal_cpu/}`.

Held-out inventories also completed: A = subruns 1155/1188 (44,494 step / 49,730 generated events), B = 1221/1254
(44,668 / 49,829). Full event-key checks exclude the five pilot files from both and A from B. Manifests
`heldout_a.json` and `heldout_b.json` carry fixed P4 conditions/background/reference hashes. Nominal/best replicas
have not yet run; do not interpret the census as a calibration result.

Run from an isolated committed checkout, with pinned core and CRV, on a free GPVM for CPU or an unoccupied approved
GPU endpoint. Do not reuse/stop another thread's EAF server blindly. This runner does not submit jobs, build software,
copy data off-site, update checkouts, or infer runtime from a short test.
