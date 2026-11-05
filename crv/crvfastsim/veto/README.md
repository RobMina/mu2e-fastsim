# Cached C4 comparisons

The cached track selection stays fixed. `python3 -m crvfastsim.veto` reproduces the stock veto rule;
`python3 -m crvfastsim.veto.paired` additionally compares a replacement coincidence collection:

```bash
python3 -m crvfastsim.veto.paired selection.parquet --card veto_stats.json \
  --fast fast.parquet --label 'Primary-only placeholder' --sectors 1 2 --outdir comparison
```

Both parquet files carry `run`, `subrun`, `event`; the reference also contains jagged `p_mid`, `t_mid` and `coincs`.
The candidate contains `coincs` with `time`, `timeStart`, `timeEnd`, `PEs`, `nLayers`, `sectorType`.
Every cached event must be present, including events with an empty fast response. Event rows may be reordered.
A `shard` column in the reference defines the paired error clusters; otherwise `(run, subrun)` is used.

Outputs are `numbers.json` and a momentum-binned inefficiency overlay with a paired pull panel, `crv_veto.png`.
Vetoed fraction and inefficiency are separately named: inefficiency is **one minus the vetoed fraction**.
Ratios and differences use paired cluster-robust delta-method errors. One shard cannot estimate a shard error;
zero reference counts have an undefined ratio. Neither case is replaced with a zero uncertainty.
The shape chi-square is the sum of squared paired bin pulls, without off-diagonal covariance or a fitted normalisation.

The denominator is the stock-selected, stock-gate-passing population. Fast gate migrations are reported separately,
not used to silently remove selected tracks. Sector counts are exclusive attribution to the highest-PE vetoing
coincidence, matching the existing veto CLI. They are **not** sector-conditioned inefficiencies. Supply every sector
type through `--sectors` to display sectors with zero response in both arms.

## Primary-only placeholder

`crvfastsim.veto.primaries` reads saved original-track-index-aligned `trkmcsim` genealogy and the same source
EventNtuple's true and reconstructed bunch times. Generator-associated muons (`valid`, `gen == 38`, `abs(pdg) == 13`)
are deduplicated by event and SimParticle ID. Their stored start states, not a proximity match, provide the inputs.
The detector-frame origin in Mu2e coordinates is an explicit argument; it must match the source geometry.
`--files` with `--cached` and `--mca` recovers all source members using the original analysis selection.

```bash
python3 -m crvfastsim.veto.primaries --lineage selected_lineage.parquet --ntuple source.root \
  --detector-origin -3904 0 10171 --outdir primary_inputs
python3 -m crvfastsim.veto.primary_chain --selection primary_inputs/selection.parquet \
  --primaries primary_inputs/primaries.parquet --gdml run1a.gdml \
  --calib calib.csv --deviation photon.csv --outdir primary_response
```

The placeholder composes the existing Run 1A material transport, photon, SiPM, waveform and coincidence stages,
without tier-1b sampling. It converts step times to the true bunch frame, folds photon arrival times by the nominal
1695 ns microbunch period, digitises in 200–1500 ns, and subtracts the reconstructed bunch time from reconstructed
coincidence times. The period can be supplied with `--period`.

Add `--library-dir DIRECTORY` to use the `run1a_v01` root-coherent library and its topology/template sidecars.
This opt-in route needs the per-primary sampler. The default remains primary-only. `--split dev` uses source-file
shards 0–21 and `--split heldout` uses 22 onward; `--split-at` changes that boundary before comparing results.
The output also includes the exact selection subset. Choose settings on development files, then report held-out once.

This is a diagnostic: represented genealogy muons are not an all-primary census; primary-only has no secondary showers,
no magnetic-field deflection, and no production-era response validation. The existing SiPM pre-window approximation
also differs from the stock start margin. Zero represented-muon events remain in the response file. Do not use its
throughput per selected event as a rate per generated cosmic or as a stock-grid speedup.
