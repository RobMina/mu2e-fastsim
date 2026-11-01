# CRV: limitations

What the fast CRV chain does not do, and what it does but has not been validated. The validated scope is in
[`validation.md`](validation.md).

## Open discrepancies in the default chain

On the 5 M held-out events, 21 of 155 CRVDQM histograms are flagged for fast steps + stock response (A1) and 13 of
133 end to end (E2E); fast response and reco on fast steps agree with stock on the same steps (0 of 133), so the
flags come from the steps (transport + tier-1b library), not from the response or reco. The two largest known causes:

- **Coincidence-cluster PE shape**: A1 chi2/ndf 3.10 against a flag level of 2.73.
- **Adjacent-layer correlation of muon descendants**: the energy that the primary muon's descendants deposit in
  neighbouring layers is less correlated than in Geant4 (correlation 0.12 fast vs 0.38 Geant4). This is a property of
  the tier-1b library sampling.

Cluster, pulse and digi counts agree within 0.25 %.

## Configurations

- Only the **extracted (KPP) configuration** (`extracted_v04`) is supported end to end. The named configurations
  (`run1a_v01`, `run1b_v01` = alias of Run 1A, `run2_v10`) have transport, geometry and conditions code, but their
  only tier-1b set (`legacy`) was built from a library file that no longer exists, so a named chain cannot run until a
  library is rebuilt for it. The `strict_v*` sets describe the extracted layer structure and are refused for named
  configurations.
- **Run 2 V1 fails** (primary-muon crossings vs Geant4, 1 M events): the CRV that wraps the DS needs below-plane
  scatter-in, which is an experimental opt-in ([below-scatter-in.md](below-scatter-in.md)). Run 1A V1 passes.
- No magnetic field: the transport is straight-line plus multiple scattering, valid for the extracted configuration.

## Not implemented

- Veto inefficiency: main keeps the C4 rule evaluation on a cached Run 1A selection (`crvfastsim.veto`). The full
  frame-level veto closure is on branch `park/veto-closure`; it came out about 21 % low and is not maintained.
- Device tier 1b is a speed option only: its independent-stream histograms fail the deck rule for five step shapes
  (bar, t, t2, lt, posx); see [`device.md`](device.md). The chain is hybrid (host tables between stages), not fully
  device resident, and there is no memory-capped mode yet.
- Device transport: single-scattering tails, sliced MSC and below-plane variants raise.
- The out-of-time background is resampled from KPP data (`crvfastsim.reco.background`); it is not simulated.
- Native and device CRY generation are opt-in; the default reads cached CRY primary tables (gen tables).
- Fast reco on Intel PVC (Aurora) has opt-in kernels (`CRVFASTSIM_PVC_RECO`); not validated against real inputs.

## Not validated

- **Float32 device precision** (default): the 0.1 % pulse-PE offset is bounded (<= 0.3 %), not removed; float32 output
  repeats on one GPU model but discrete counts differ slightly between models (float64 is bit-identical across models).
- The opt-in CPU-speed switches (`CRVFS_CPU_*`, `CRVFS_CONTEXT_HASH_WORKERS`, `CRVFS_PREPARED_LIBRARY_CACHE`) and the
  compact reco dispatch (`CRVFASTSIM_COMPACT_DISPATCH`, default in the tests only) are checked for identical output
  on test samples, not on a full campaign.
- P1 (about 1.9 ms per accepted event on one core) was measured on a shared host and has to be re-measured on an idle
  one.
- Named-configuration response and conditions bundles ([named-response.md](named-response.md),
  [named-conditions.md](named-conditions.md)) are admission contracts, not physics validation.
- The fast chain is validated on MC only. KPP data comparisons need the measured data offsets and are outside this
  repository.

## Inputs

The default chain reads Mu2e-internal inputs (the tier-1b library and sidecars, the GDML export, simulation
conditions, the photon lookup tables, Geant4 gen tables) that are not in this repository and have no published
collaborator release yet. See the umbrella's [`docs/running.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/docs/running.md).
