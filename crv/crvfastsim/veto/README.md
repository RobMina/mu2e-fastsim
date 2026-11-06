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

## Retained-generator inputs (opt-in)

Use independently extracted upstream generator collections when the track genealogy omits other particles in the
same event. Establish the source-file ancestry and event/state conventions before preparing inputs; never construct
these tables from vetoing-cluster truth or by matching particles to reconstructed coincidences.

Input parquet schemas:

- Selection: the cached selection above, including integer `run`, `subrun`, `event`, `shard` and bunch times.
- Generators: integer `run`, `subrun`, `event`, `gen_product_id`, `idx`, `pdg`, `genid`; numerical
  `x`, `y`, `z` (mm), `px`, `py`, `pz` (MeV/c), `e` (total MeV), `t` (global ns), in upstream Mu2e coordinates.
- Event census: integer `run`, `subrun`, `event`, `n_gen` for every selected event, including empty collections.
  If present, `gen_product_id` identifies the single collection and must match the payload. Without it, `n_gen`
  counts the explicitly supplied collection union; ProductID/index pairs distinguish particles within each event.

Keys must have integer storage, be nonnegative and fit signed 64-bit; particle PDG/genid may be signed. Floating
identity columns are rejected, not rounded. ProductIDs must be positive. Sparse source indices are retained;
identical states with different identities are not deduplicated. Missing events/rows, duplicate identities,
nonfinite states and energy below momentum magnitude are errors. Counts establish retained-collection coverage,
not completeness of the original generated stream or upstream physics selection.

```bash
python3 -m crvfastsim.veto.generators --selection selection.parquet \
  --generators upstream_generators.parquet --events upstream_events.parquet --outdir retained_dev
python3 -m crvfastsim.veto.primary_chain --selection retained_dev/selection.parquet \
  --primaries retained_dev/primaries.parquet --input-kind retained \
  --generator-events retained_dev/generator_events.parquet --split dev \
  --gdml run1a.gdml --calib calib.csv --deviation photon.csv --library-dir library --outdir response
```

Preparation selects development shards <22 by default (`--split-at` changes the boundary). The response chain
preserves every selected event, including empty responses. It retains source states/total energies and assigns
unique dense internal event/particle keys; `generators.parquet` and `generator_map.parquet` preserve the original
`source_run/subrun/event`, `source_idx` and `gen_product_id`. These GenParticle identities are not SimParticle IDs.
The existing muon transport still calculates its kinetic quantities from momentum and the muon mass; preserving
input `e` does not change that transport model. The represented-muon default retains its existing energy calculation.

All non-muons are passed to the existing library sampler; non-muon input without a library is rejected. The report
includes sampler counts and `inputs.shower_support`: counts by signed PDG/KE bin of supported, missing-cell,
zero-denominator, zero-shower and unknown-mass inputs, plus out-of-grid counts. This reporting makes library gaps
explicit without changing RNG consumption or the response model. A missing cell or zero observed donors is not
proof of a physical zero response. The existing sampler clips to endpoint energy bins and does not condition
non-muon response on supplied position/direction or charge sign; these remain diagnostic predictions, not closure.

The Slurm wrapper takes `C4_INPUT_KIND=retained` and `C4_INPUTS=<prepared directory>`; invalid input-kind values
are rejected by the Python CLI. No simulation, library rebuilding or response tuning is performed by preparation.
Keep held-out results frozen while developing this route: changing input count/order and stored precision changes
random-number consumption, so a changed veto decision cannot be attributed solely to an added particle.
