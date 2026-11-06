# Selected-muon history diagnostic (opt-in)

`transport(..., trace=observer)` copies selected source rows into `begin`,
`material` and `end` callbacks. The observer supplies distinct integer `rows`;
no RNG is supplied. `begin` records accepted field/boundary proposals, `material`
records actual sampler inputs/results, and `end` records updated states/fates.
Cold-at-start rows have terminal output but no accepted steps. Synthetic tests
cover callback mutation isolation, active-lane compaction and material stops.
The default `trace=None` leaves this diagnostic inactive.

`python3 -m beamfastsim.tstrace --runs <existing-baseline-jsons> --out <fresh-dir>`
replays only development seeds 20261111/12/13, with the complete original positive
cohort. It saves two selected muons' histories only after every saved result and
exposure array matches (runtime excluded). `tscrossings` extracts their sparse
stock VD/DS records, and `tshistory --stock <npz> --traces <npzs> --out <json>`
compares recorded states and applied material episodes. All output paths must be
fresh; these commands describe the completed diagnostic, not a request to rerun it.

## Completed development comparison (2026-11-06)

The bounded comparison traces mu+ `1430:920075:4321:1155` (A) and
`1430:920071:2260:4478` (B). `tscrossings` reads only their existing development
ntvd/nttvd records: each has 12 VD samples plus DS; exact VD1/DS states match the
saved reference. Both stock momenta change between VD4 and VD5, from 34.148017 to
31.546330 MeV/c (A) and 24.712124 to 15.826283 MeV/c (B). These sparse pre-step
records are not material histories and do not measure stock material columns.

`tstrace` replayed existing baseline seeds 20261111/12/13 with all 12,832 positive
inputs in original order. Every saved terminal and exposure array is exactly
reproduced except runtime. Before the absorber, fast chord intersections at VD4's
recorded x plane differ from stock by 0.00823 mm (A) and 0.01257 mm (B). After
Ti6Al4V, A survives seed11 but encounters polyethylene and stops in seeds12/13;
B encounters A1100 and stops in seeds11/12 but survives seed13.

All four losing histories leave titanium with momentum above the corresponding
stock VD5 value. B's survivor has the largest titanium column and lowest exit
momentum of its three replays. A simple excessive-titanium-loss explanation is
not supported; post-absorber direction/acceptance is implicated in these fast
losses, but the cause of their frequency relative to stock remains unresolved.

## Interpretation limits

- The two targets are deliberately fragile stock survivors, not a population
  sample. Their six traced histories cannot estimate a failure rate.
- `tshistory` summarizes sampler-call episodes and all directional intersections
  of the recorded VD3..6 x planes. Those intersections are not validated VD hits.
- Applied material order is dictionary order, not geometric within-step order;
  episode positions bracket accepted steps rather than exact material boundaries.
- The extractor retains repeated DS samples, but this bounded report requires one
  DS sample and unique VD3..6 samples per target; the selected records satisfy this.
- No new rate sample, seed scan, settings change, stock simulation or held-out
  access is involved. The histories do not establish convergence or justify tuning.

Regression tests: `test_tstrace_transport.py`, `test_tstrace.py`,
`test_tscrossings.py` and `test_tshistory.py`; all use synthetic inputs.
