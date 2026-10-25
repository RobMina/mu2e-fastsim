# Thread B: corrected-transport A1/E2E pilot and histogram coverage

## Executed scope, 2026-10-25

One complete `g4ref_v1` shard (11), **50,000 generated events**, through both stock-histogram arms on
mu2egpvm06. This is the first recorded post-aluminium A1/E2E pilot, **not** the full twenty-shard validation and
**not an aluminium-only A/B**. It includes the subsequent world-energy bookkeeping corrections.

Transport was reused from Thread A, not regenerated:
`$CRVFS_DATA/thread_a_ledger_20261025_c3895aae/extracted20/tables/fastC.001400_00000011.root`.
Its retained audit certifies 50,000 persisted generated event IDs. Source transport CRV `3895aae`; tier-1b,
response/reco and pilot driver CRV `7d4642b` (isolated worktree based on `15c99fe`); isolated core `dcde9f0`.
Tier-1b library `$CRVFS_DATA/tier1b_v7/lib_prod_as_20files.root`; seed 11; NumPy response/reco for the E2E table.
Frozen stock area `/exp/mu2e/app/users/ram2aq/crvval`, backing
`51a5125d_19cf240c_2241f345`, DQM `e27eeee`. No source/build change in that area; plugins postdate their source.
No builds, grid submissions, baseline rewrites or new G4 simulation.

Output: `$CRVFS_DATA/valhist/thread_b_dqm_20261025_s11/` (38 MB at first audit). `manifest.json` records hashes of
transport/library/conditions/FHiCL/CRV source/plugins; `completion.json` requires both successful art exits,
highest-cycle event counts and exact sorted event-ID hashes matching transport. Both arms pass, with no recorded
source/input hash changes during the job. The initial manifest did not hash LUT/template/core source; this is a
provenance limitation. LUT/source checksums were recorded post-run in `postrun_assets.sha256`, not retroactively
claimed as pre-run verification. The driver now records those assets before future runs too.

First launch failed before creating outputs because an inherited `MU2E` skipped shell function setup; fixed in
`7d4642b` and rerun after confirming no job/output existed. Successful driver:
`$CRVFS_DATA/thread_b_dqm_20261025_s11_driver_v2.log`.

## Persisted products and rates

| Arm | Generated event rows | Digis | Pulses | Clusters |
|---|---:|---:|---:|---:|
| A1 fast steps + stock response/reco | 50,000 | 55,860 | 70,260 | 2,868 |
| E2E fast steps + fast response/reco | 50,000 | 55,857 | 70,389 | 2,877 |

All product/event trees use cycle 1. 2,178 events have digis/pulses in either arm; clusters occur in 1,947 / 1,946
events. Complete empty-event population is retained in the `crvdump/events` trees. Relative cluster totals:

- A1 versus frozen A0g's 2,918: **-1.7135%**.
- E2E versus A0g: **-1.4051%**.
- E2E versus A1: **+0.3138% +/- 0.2491%**, paired-event delta-method standard error.
- Against pre-aluminium all-muon saved outputs: A1 2,890 -> 2,868 (-0.7612%); E2E 2,894 -> 2,877 (-0.5874%).

`paired_e2e_vs_a1.json` uses all 50,000 exactly matched event IDs. The paired error assumes independent generated
events and is conditional on the fixed secondary library and conditions; finite-library/systematic uncertainty is
not included. No paired A0g error is claimed from its aggregate histograms. Sector-type E2E-minus-A1 clusters:
EX 0, T1 +8, T2 +1, taggers 0. Small pilot counts do not certify sector efficiency.

Measured stock processing: A1 69.97 CPU / 96.25 real seconds; E2E injection 17.75 CPU / 42.79 real seconds.
These timings exclude fast-table generation and are not a campaign runtime projection.

## Coverage, flows and shape attribution

`valhist.coverage` reads original unmerged shard files directly. Legacy `compare.merge` wrote only visible bins,
dropping flow and source Sumw2; legacy merged files must not be used for flow certification. Legacy code/output
remains untouched. New reports preserve array contents/variance evidence and classify derived maps conservatively.

| Comparison vs A0g | TH1 | TH2 | Binning metadata | Missing histograms |
|---|---:|---:|---:|---|
| A1 | 155 | 9 | 2 TNamed, equal | 0 |
| E2E | 133 | 9 | 2 TNamed, equal | 14 step + 8 DigiMC |

No profiles were booked in this selection; profiles are explicitly unsupported by the audit, not silently
skipped. The histogram union is audited; this is not yet an independent check for a catalogue entry absent from
**both** files. E2E does not inject steps or DigiMC and its 22 omissions remain declared coverage gaps.
Original source objects do not supply Sumw2 for these compared histograms: no invented independent-Poisson
variance, calibrated p-value, shape significance or blanket pass is reported.

**Flow finding:** legacy cluster Y/Z and Validation cluster Y/Z have all entries in overflow, not zero activity.
For A0g/A1/E2E respectively the overflow carries 2,918 / 2,868 / 2,877 clusters. Validation PE has 86 / 73 / 75
flow entries. This is why visible-only totals must not be called full yields.

Normalized visible-bin total-variation distances (descriptive only):

| Histogram | A1/A0g | E2E/A0g | E2E/A1 | Frozen stock reseed A0'/A0 |
|---|---:|---:|---:|---:|
| CRVRecoDQM PEc | 0.1035 | 0.1045 | 0.0944 | 0.0808 |
| X_extracted | 0.0908 | 0.0923 | 0.0929 | 0.0914 |
| Y_extracted | 0.0504 | 0.0452 | 0.0237 | 0.0174 |
| NPc | 0.0481 | 0.0509 | 0.0421 | 0.0251 |

Each histogram is normalized separately: these distances do not absorb the total-rate difference. The sizable
reseed floor illustrates why a single shard/draw cannot calibrate shape significance. Only source-reviewed
cluster histograms are promoted to count semantics with explicit `--count-key`; MPV/rate maps are not.
The known float-position round-trip limitation is reproduced on the frozen A0-dump/RT shard: 156 common
histograms including nine TH2s, only the three cluster-X histograms differ, and all common flow bins agree.

Evidence: `coverage_{a1,e2e}_v2`, `coverage_e2e_vs_a1`, `coverage_{a1,e2e}_vs_frozen`, `coverage_floor`,
`coverage_frozen_roundtrip`, each with `summary.json`, `histograms.json`, `report.tsv`. Initial `coverage_a1/e2e`
reports are retained historical artifacts from before explicit count overrides and metadata handling.

## Reproduction and next gates

- Drivers: `valhist/run_pilot.sh`, `valhist/audit_pilot.sh`; contracts and limitations: `docs/valhist-coverage.md`.
- Local tests: **25 passed** across `test_valhist_{coverage,pilot,paired_rates}.py`, before the final manifest
  hardening (rerun recorded with publication).
- Extend new candidate arms to the remaining shards using retained Thread A transport and unique tags; never
  rerun A0/A0'/A0g merely to get a fresh timestamp.
- Add independent catalogue coverage, event/shard-paired shape covariance, uncertainty calibration and
  finite-library treatment. Current event-paired rate errors do not close the shape-uncertainty programme.
- Keep derived maps/profile semantics, expected omissions and stock float-round-trip differences explicit.
  This pilot is executed and audited, **not a full DQM pass**.
