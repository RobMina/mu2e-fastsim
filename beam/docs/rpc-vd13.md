# Internal RPC electron-marginal study at VD13

This opt-in study uses `rpcspec` pair starts with the existing `cechain --ref`
interface. It does not modify CE transport or claim a combined electron/positron
response. `rpcstudy starts` and `rpcstudy compare` reproduce the study using
saved input tables and transport outputs.

## Samples and comparisons

Reference: four `RPCInternalPhysical.MDC2025af` dts files, 25,000 generated
captures each. There are 21,315 saved primary electrons after the production
DetectorStepFilter. The matching `PhysicalPionStops.MDC2025ac` files supply the
same production-window occurrences (25,000 per shard, including repeated wraps).
One million fast pairs are ten independent kinematic draws per occurrence;
only the primary electron is transported. Source and transport seeds are
20261105 and 20261106. The saved-start transport seed is 20261105.

Two comparisons are kept separate:

1. **Same saved starts:** transport the electron start states of the saved dts.
   This is paired by primary, conditional on the stock detector filter.
2. **Same pion windows:** independent pair draws from those stop occurrences.
   Compare the saved-stock VD13 count per 25,000 generated captures with the
   fast count per 250,000, paired per source shard. The stock sample is filtered;
   this is not an inclusive Geant4 acceptance measurement.

No tuning is applied. Shards 0–1 are development, 2–3 held out. A held-out sample
from an earlier study is not new independent production; the split is specific
to this implementation. Only two shards per split means the between-shard
errors are poorly determined and can be accidentally small.

## Cuts and statistics

Both arms use the Run 1A TT_FrontHollow radius 850.1 mm around the DS axis. With
`protonabsorber.isShorterCone=true`, the stock VD is a disk. Report all VD
crossings, p > 60 MeV/c, and the established CE-study fiducial p > 60 MeV/c with
local-helix maximum radius > 450 mm. The last cut does **not** prove that an
RPC pair would pass the stock detector filter. No filter-implication assumption
or corrective scale factor is used.

Yield ratios and differences use paired per-shard residual errors. Empty selected
arms remain in the exposure. Histograms include flow bins and use fixed bins
(2 MeV/c momentum, 0.05 cos theta, 25 mm radius). Shapes are equal-area; their
per-primary variance includes normalisation and, for saved starts, same-primary
bin covariance. Generated-source errors use sum-of-squared weights. The chi2 is
a sum of marginal pulls; bin covariance is not included in that descriptive
statistic. Between-shard bin errors are retained separately in the JSON, not
used as the plotted denominator because two shards do not estimate each bin's
variance reliably. No p-value is claimed for the shape chi2.

## Commands

```bash
python3 -m beamfastsim.rpcstudy starts --window pion_windows.npz \
  --repeats 10 --seed 20261105 --out rpc_1m.npz
python3 -m beamfastsim.cechain --ref rpc_1m.npz \
  --offline-root "$MU2E_OFFLINE_ROOT" --geom Offline/Mu2eG4/geom/geom_run1_a.txt \
  --maps DSMap.header TSdMap.header --backend device --device-float float32 \
  --lanes 131072 --shrink 32 --unroll --seed 20261106 --out rpc_plane.npz
python3 -m beamfastsim.rpcstudy compare --ref rpc_stock.npz --starts rpc_1m.npz \
  --fast rpc_plane.npz --shards 2 3 --offline-root "$MU2E_OFFLINE_ROOT" \
  --maps DSMap.header TSdMap.header --out rpc_heldout.json --png rpc_heldout.png
```

`pion_windows.npz` contains `xstop,ystop,zstop,tstop,pdg,at_rest,foil,codestop`,
`production_ifile`, and `rpc_event_occurrence`: one selected physical pion per
capture occurrence. Repeated parent events must not be collapsed. `rpc_stock`
is the `cereference.read(..., code=179)` layout with `ifile` and `gencount`.
For saved-start pairing, run cechain with `--ref rpc_stock.npz` and omit
`--starts` in the comparison. Every start must have a final output row, including
stopped, absorbed, upstream and path-capped particles.

## Measured result

Held-out shards 2–3: 50,000 stock generated captures, 10,569 retained stock
primary electrons, and 500,000 fast generated electrons. At the standard
p > 60 MeV/c and helix rmax > 450 mm fiducial:

| Comparison | Stock count / denominator | Fast count / denominator | Fast / stock |
|---|---|---|---|
| Same pion windows | 3,815 / 50,000 | 38,016 / 500,000 | 0.99649 ± 0.02208 |
| Same saved starts (conditional) | 3,815 / 10,569 | 3,663 / 10,569 | 0.96016 ± 0.00056 |

The second uncertainty is the observed scatter of just two paired shards, not
a robust precision estimate. The conditional deficit is not corrected away.
For the generated-source comparison, shape chi2/ndf is 23.49/26 (momentum),
6.12/6 (cos theta), and 41.41/25 (radius); the corresponding saved-start values
are 20.32/26, 3.98/6 and 33.90/25. These do not establish inclusive acceptance
or pair-response closure. With no helix-radius cut the generated-source yield
above 60 MeV/c is 2.6066 times the saved-stock yield, illustrating the effect of
the stock output selection. That inclusive-looking ratio is not a validation
metric. The held-out histograms were reprocessed only to correct a two-shard
bin-variance estimator; no physics, cuts, samples or yields were changed.

One million fast electrons take **108.24 s (9,239/s)** for a single device call
on one RTX PRO 6000 MIG 1g.24gb slice with four host CPUs, including map loading
and JIT but excluding source generation and file I/O. Pair generation takes
1.56 s. All one million rows are present: 701,791 plane, 26,872 stopped,
139,203 upstream, 21,115 wall, 109,960 absorbed, 1,059 path-capped.
A stock production log gives **23.107 generated captures/s per grid slot** for
the whole art job, or **30.697/s** for its Geant4 module. The measured rate ratios
are **400** and **301**, respectively, with the different stage boundaries
stated below. This is not a warmed steady-state or CPU benchmark.

`rpcsummary` bundles the saved comparisons, source report, raw fates and measured
stock log timing into `numbers.json` and copies the VD13 figure. Re-run it from
saved files; it launches no transport:

```bash
python3 -m beamfastsim.rpcsummary --results comparisons --source source_numbers.json \
  --plane rpc_plane.npz --starts rpc_1m.npz --stock-log stock_rpc.log --outdir incoming
```

## Limits

No new Geant4, full-pair transport, reconstruction, external photon conversion,
absolute physics rate or default promotion. The stock generation rate describes
both particles plus all detector transport and secondary production; fast rates
describe the primary electron to the tracker-front plane. Their rate ratio is a
stage-boundary comparison, not a validated full-chain replacement factor.
