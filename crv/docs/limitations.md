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
- Device tier 1b is opt-in; NumPy stays the default and the reference. Its independent-stream step shapes pass the deck
  rule at float32 and float64 (0 of 21 flagged with a reseeded NumPy floor; bar passes through the floor), but
  downstream of it there is only a 500 k diagnostic; see [`device.md`](device.md#validation-status).
- The chain is hybrid (host tables between stages), not fully device resident: the tier-1b output and the response and
  SiPM preparation still go through the host, and full-residency throughput has not been measured.
- Memory caps cover the NumPy chain only. With `CRVFS_PREPARED_LIBRARY_CACHE` set it completes under a 4 GB host cap
  (Slurm `--mem=4G`), identical per seed at every stage to the uncapped run on ten held-out shards, not on a full
  campaign. The device chain fits a 10.66 GB preallocated XLA pool (peak 7.64 GB, float32, three passes of one shard
  on an RTX PRO 6000 MIG slice), but its host memory is not capped: RSS reaches 16 GB, partly a host copy of the
  library in device mode (+2.3 GB).
- Device transport: single-scattering tails, sliced MSC and below-plane variants raise.
- The out-of-time background is resampled from KPP data (`crvfastsim.reco.background`); it is not simulated.
- Native and device CRY generation are opt-in; the default reads cached CRY primary tables (gen tables).
- Fast reco on Intel PVC (Aurora) has opt-in kernels (`CRVFASTSIM_PVC_RECO`); not validated against real inputs.

## Not validated

- **Float32 device precision** (default): the 0.1 % pulse-PE offset is bounded (<= 0.3 %), not removed. Output repeats
  run to run on one GPU kind (with the default `MU2EFS_DEVICE_DETERMINISTIC`), but float32 discrete counts differ
  slightly between GPU architectures ([device-precision.md](device-precision.md#reproducibility)).
- The speed switches are checked for identical output on ten held-out shards at two seeds
  ([switch table](validation.md#speed-switches)), not on a full campaign.
- **Float32 cluster properties depend on the kernel batch shape.** `find_dense` sums cluster moments with `einsum`, whose
  addition order on the GPU follows the batch shape. With hit times and positions centred per group, batch shapes that
  differ by padding alone move `cl_avgtime` by up to 0.0005 ns, `cl_pe` by 0.001 PE, `cl_pos` by up to 0.07 mm and
  `cl_slope` by up to 1.5e-5 with the same clusters and members (A100, ten shards; float64: 6e-11 ns, 5e-9 mm).
- **Run-to-run identity on the GPU needs XLA autotuning off**, which is the default (`MU2EFS_DEVICE_DETERMINISTIC`).
  With `MU2EFS_DEVICE_DETERMINISTIC=0`, separate processes can pick different kernel configurations: float32 output then
  differs between runs on every card tested (MIG slice, A100, RTX PRO 6000), float64 in digi voltages. A shared warm
  compilation cache hides this only for the jobs that load from it.
- The prepared library cache has no eviction: its key binds every source file, so each code version writes a new 3.1 GiB
  entry under `CRVFS_PREPARED_LIBRARY_CACHE`.
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
