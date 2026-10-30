# Run 2 scatter-in convergence and a V1 variance proposal

`crvfastsim.analyses.scatter_convergence` is an opt-in study driver, not a transport default or an acceptance-rule
change. It reuses the certified 20-shard Run 2 reference, verifies each generator/reference file against its
original campaign SHA256, and rejects duplicate subruns or a geometry hash mismatch. It reads the highest tree
cycles through the standard table reader. Generated-event counts must be 50,000 per shard; persisted reference
accepted counts must match the original manifest. Candidate generator rows must equal the input generator table.
Hashes establish artifact consistency, not independent producer attestation.

```bash
python3 -m crvfastsim.analyses.scatter_convergence \
  --reference-root "$REFERENCE_ROOT" --gdml "$REFERENCE_ROOT/geometry/mu2e_run2_v10.gdml" \
  --outdir NEW_OUTPUT --subruns 101 --step-mm 12.5 --envelope-mm 1500
```

The reference root preserves the relative paths of `thread_a_named1M_20261026/manifest.json`, the three original
dump partitions and `geometry/mu2e_run2_v10.gdml`. Use a staged Offline geometry tree on `MU2E_SEARCH_PATH` and an
explicit core/CRV `PYTHONPATH`. The driver does not fetch reference data, run art, build software or submit jobs.

Use `--candidate-root SAVED` to analyze existing `SAVED/SUBRUN/tables/fastC.001400_SUBRUN.root` candidates instead
of regenerating them. Transport controls recorded by a saved `.scatter_in.json` sidecar take precedence as evidence
of that old production; the current invocation's requested controls do not retroactively describe it. A reused
straight candidate has no scatter sidecar and is not upgraded to a prospectively certified producer.

## Statistical unit and proposal

The original `v1compare` summary and **all** its validity and non-shape checks are retained unchanged. Per-event
vectors use its exact `(run,subrun,event,bar)` primary-muon aggregation, not a second count definition preserving
individual `genidx`. Every generated event, including zeros, is a cluster. Within-event cross products and paired
arm covariance are kept, and independent shards merge additive sufficient statistics. There are no per-event
Python loops. Every individual sector is retained in JSON in addition to the seven V1 sector-type groups.

For each 20-bin V1 reference range the clustered log-ratio covariance is

```text
r_j = (A_j/A)/(B_j/B)
z_ij = a_ij/A_j - a_i/A - b_ij/B_j + b_i/B
Cov(log r) = N/(N-1) * sum_i z_i z_i^T
```

The diagonal proposal uses the existing normalized-ratio residual and chi2/ndf threshold 1.5, replacing only the
independent-crossing variance by the corresponding event-clustered diagonal variance. Bin eligibility and original
shape-validity guards remain required. The driver asserts that its independent-crossing statistic reproduces
V1's statistic to `rtol=1e-9`. Missing support stays unsupported; there is no variance floor or added two-SE mean/rate
allowance. The optional full covariance/log-ratio statistic is diagnostic, not the proposed replacement gate.

`summary.json` retains unchanged V1, the explicit proposal checks, all-sector mean/count uncertainties, shape
statistics, input/output/geometry/source fingerprints and available scatter completion diagnostics. `table.md`
lists the two verdicts side by side. A numerical FAIL is a completed study, not a process error. The output root
must be new. To render already saved evidence without recomputing transport:

```bash
python3 -m crvfastsim.analyses.scatter_convergence --render OUTPUT/summary.json
```

## Convergence axes and interpretation

Refine the maximum segment 50, 25, 12.5, 6.25 mm independently of admission/escape envelope (1500, 3000, 6000 mm)
and loaded passive-material margin (20 m versus 100 m). No cap hit is allowed. Compare all sectors, recovered
initial misses, stops, maximum segment x/X0 and scattering sigma, and the angle/momentum strata from
`crvfastsim.analyses.v1strata`. Different controls change the RNG draw order; common seed is not common scattering.
A same-generator paired comparison includes event covariance but does not remove stochastic transport variance.

Keep the stock Run 2 delta production cut **0.361 MeV** fixed. Changing the in-bar cut or stopping algorithm would
mix a physics change into a numerical convergence study. Momentum/incidence selections are truth diagnostics,
not alternate production cuts or a new V1 event population. Large worst-segment angles remain a model-validity
warning even when a group rate happens to agree. The finite envelope also omits upward or initially preselected-out
tracks; numerical convergence is not proof of G4 closure.

## Named V2 is a separate gate

A fixed-stock Run 1A replay is not a Run 2 reference. A Run 2 independent-seed floor requires at least eight fresh
stock response seeds and corresponding fast seeds on the same declared Run 2 input/configuration, with spectra
and prospective producer/optical/runtime bindings for both arms. Replaying fast response against a single old
stock sample measures conditional fast-seed spread only. Historical metadata is not upgraded into a receipt.
The transport/statistics driver does not relax named V2's unsupported state or infer fixed-stock provenance.

`python3 -m crvfastsim.analyses.scatter_scan_report --summaries SUMMARY... --outdir NEW --strata` collects
completed, uncapped probes, adds truth momentum/incidence strata and adjacent-probe event-paired count/mean
contrasts. Order the summaries to define the intended contrast sequence. Same generators are required; the seed
is not treated as common scattering. Input summaries/tables remain unchanged and files are processed one at a time.
