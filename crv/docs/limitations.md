# CRV: limitations

What the fast CRV chain does not do, and what it does but has not been validated. The validated scope is in
[`validation.md`](validation.md).

## Open discrepancies in the default chain

On the 5 M held-out events with the default tier 1b (`strict_v5`, root-coherent draws), fast steps + stock response
(A1) against Geant4 steps + stock response (A0g), paired per-shard statistic: fast response and reco on fast steps
agree with stock on the same steps, so what remains comes from the steps (transport + tier-1b library), not from the
response or reco.

- **Coincidence clusters per event (NPc)**: paired chi2/ndf about 2 with every tier-1b default so far; not moved by
  the root-coherent draws. Part of the step multiplicity (NStep 1.3-1.6 against 2.9 before) is steps per donor, not
  placement.
- **Wide-range digi and pulse multiplicities** (NDigis2, NPulse2): worse than with `strict_v4` (3.04 and 2.10 against
  1.28 and 1.99); partly seed noise of the tier-1b draws (two seeds on the dev half: 1.65 and 0.58 end to end). Events
  with more than 100 digis are 0.83 +- 0.05 of Geant4 (`strict_v4` 0.90 +- 0.06), more than 150 about 0.6 with both
  sets; not attributed yet.
- **Descendant steps at the bar ends**: the first and last bins of the step position along the bars are 13-16 % above
  Geant4 (`strict_v4` 9-13 %), all muon-descendant steps. A descendant keeps its library offset along the bar, and one
  that the offset carries past the bar end is moved back inside (`fit_inside`) instead of being lost.
- **Tier-1b draw noise in the yields**: one tier-1b seed moves the digi, pulse and cluster yields by about 0.2 % (two
  seeds differ by -0.22 +- 0.20 % on the dev half and -0.20 +- 0.17 % on the confirmation half). With one seed the
  confirmation half showed +0.29 +- 0.16 % above `strict_v4` with a flat per-SiPM ratio, which flags the occupancy
  histograms; the second seed gives +0.05 +- 0.15 %. Over all 5 M events (first seed): digis +0.33 +- 0.14 %, pulses
  +0.37 %, clusters +0.23 % against Geant4.
- **Finite library**: about 3,100 effective donor trees carry the forward energy above 10 MeV on 2.5 M events
  (repeat rate 0.12); the library's finite size is not propagated into the errors.
- The device tier 1b implements the previous default `strict_v4` only.

Fixed by the root-coherent draws: the coincidence-cluster PE shape (A1 paired chi2/ndf 3.28 -> 1.09) and the cluster
PE tail above 2000 PE (0.923 +- 0.014 -> 0.991 +- 0.014), and the adjacent-layer correlation of muon descendants
(0.24 -> 0.39-0.43, Geant4 0.40).

## Configurations

- Only the **extracted (KPP) configuration** (`extracted_v04`) is supported end to end. The named configurations
  (`run1a_v01`, `run1b_v01` = alias of Run 1A, `run2_v10`) have transport, geometry and conditions code, but their
  only tier-1b set (`legacy`) was built from a library file that no longer exists, so a named chain cannot run until a
  library is rebuilt for it. The `strict_v*` sets describe the extracted layer structure and are refused for named
  configurations.
- **Run 2 V1 fails** (primary-muon crossings vs Geant4, 1 M events): the CRV that wraps the DS needs below-plane
  scatter-in; the experimental marcher is on branch `park/named-run2`. Run 1A V1 passes.
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
- Named-configuration V2 ([named-response.md](named-response.md)) groups the metrics by sector type; its overall status
  stays unsupported until both arms carry verified configuration provenance.
- The fast chain is validated on MC only. KPP data comparisons need the measured data offsets and are outside this
  repository.

## Kept off main

- Intel PVC (Aurora) reco kernels (`CRVFASTSIM_PVC_RECO`: Jacobi Hessian eigenvalues, label propagation, batch cap):
  branch `park/pvc-reco`. Not validated on PVC hardware; identical output and no gain on an A100.

- Below-plane scatter-in for the CRV that wraps the DS (`--below-scatter-in`, `transport.scatter_in_crossings`,
  Run 2 algorithm development; not converged, V1 Run 2 still fails): branch `park/named-run2`.

- Named-run provenance and diagnostics: strict conditions bundles (`named_conditions`, `--conditions-manifest`),
  fasttable receipts and the V2 receipt audit (`response_receipt`, `--write-receipt`, `--candidate-receipt`), the
  fasttable named mode (`--run-config`, `--event-reference`), the stock shared-yield emulation
  (`--stock-shared-yield-fcl`) and the response-execution trace (`--trace-response`): branch `park/named-run2`.

## Inputs

The default chain reads Mu2e-internal inputs (the tier-1b library and sidecars, the GDML export, simulation
conditions, the photon lookup tables, Geant4 gen tables) that are not in this repository and have no published
collaborator release yet. See the umbrella's [`docs/running.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/docs/running.md).
