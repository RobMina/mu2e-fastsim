# Tracker trigger paths on the device

Opt-in JAX evaluation of the tracker trigger paths (tpr, apr, mpr and cst families, plus the prescale, digi-count
and lumi-stream paths) of the production menu, from StrawDigis to per-path decisions, on a GPU or CPU. Nothing in the stock replay (`trigger/replay.py`, `compare.py`) changes.

| Stage | Stock module | Device | Agreement with stock on stock digis |
|---|---|---|---|
| straw hits | `StrawHitReco` (TrigMakeSH, FitType 5) | port with the stock arithmetic (`hits.straw_hits`) | bit for bit |
| panel hits | `CombineStrawHits` (TrigMakePH) | port with the stock arithmetic (`hits.panel_hits`) | bit for bit |
| background flags | `DeltaFinder` (TrigFlagPH) | port with the stock arithmetic and orders (`deltafinder.py`); opt-in (off: no hit flagged) | bit for bit; see [Background flags](#background-flags-deltafinder) |
| time clusters | `TimeClusterFinder` | port (`timecluster.py`), incl. the TMVA hit MLP | see the README table of the PR / record |
| time clusters (apr) | `TZClusterFinder` (TrigTZClusterFinder) | port with the stock arithmetic (`tzcluster.py`); opt-in (off: the TimeClusterFinder port with TZ's `clusterThresh`) | on: equal to stock in every event; off: approximation; see [apr time clusters](#time-clusters-of-the-apr-paths-tzclusterfinder) |
| helices | `RobustHelixFinder` (tpr), `AgnosticHelixFinder` (apr) | compact fit with the stock per-face hit selection and chi2 definitions (`helix.py`); the hit ratio over `HelixTool`'s nominal face z | approximation |
| helices (tpr) | `RobustHelixFinder` + `MergeHelices` (TrigHelixFinder(Ue), TrigHelixMergerDe / Ue) | port with the stock arithmetic (`rhelix.py`); opt-in (`Chain(tpr_helix=True)`, off: the compact fit) | on: equal to stock given the stock time clusters; see [tpr helices](#helices-of-the-tpr-paths-robusthelixfinder) |
| helix cuts | `HelixFilter` | same cuts, incl. the D0-phi prescale counter on the host | exact given the helix |
| track fit + cuts | `KalSeedFit` + `KalSeedFilter` | response layer: the filter's cuts on helix quantities and a pass probability per path and process from development shards (shipped, `data/`) | statistical |
| prescale | `PrescaleEvent` | event number modulo the factor; a mode the module does not configure rejects (`apr_proton`) | exact |
| straw-digi count | `DigiFilter` (`minBias_SDCount`) | straw digis of the event | exact |
| lumi stream | `LumiStreamFilter` | event counter per subrun on the host, in event order | exact |
| off-spill hits, time clusters (`cst_TimeCluster`) | off-spill `StrawHitReco` / `CombineStrawHits`, `SimpleTimeCluster` | the hit ports with the off-spill settings, `simpletc.py` | exact on the held-out shards |
| multi-helix finder (`mpr_*`) | `RobustMultiHelixFinder` + `MergeHelices` | port with the stock arithmetic (`mpr.py`); opt-in | see [Multi-helix paths](#multi-helix-paths-mpr) |
| cpr helices | `CalHelixFinder` + `MergeHelices` (on the `CalTimePeakFinder` peaks) | port with the stock arithmetic (`calhelix.py`); with the cpr paths (opt-in) | see [Calorimeter-seeded helices](#calorimeter-seeded-helices-cpr) |
| calo hits, clusters | `CaloHitMakerFast`, `CaloClusterFast` | port with the stock arithmetic and orders (`calo.py`), from CaloDigis | bit for bit; see [Calorimeter paths](#calorimeter-paths) |
| calo filters | `CaloClusterCounterFilter`, `CaloLikelihood`, `CaloNNEval` + `FilterEcalNNTrigger`, `CaloCosmicCalib`, `DigiFilter` (calo digis) | ports (`calo.py`, `calopaths.py`) | exact on the development and held-out files |

Not evaluated on the device: `cst_CosmicTrackSeed` (`LineFinder` + `CosmicSeedFilter`), paths with
`MultiHelixFilter` or `minNHelices > 1`. `apr_TC_calo` needs the TZClusterFinder port (`Chain(tz=True)`); without it the
path is always false.

### Background flags (DeltaFinder)

`deltafinder.py` ports DeltaFinder (Offline v13_41_01, CalPatRec: `DeltaFinderAlg`, `DeltaSeed`, `DeltaCandidate`,
`ProtonCandidate`), the menu's `TrigFlagPH`. It sets `Background` (StrawHitFlag bit 6) on the panel hits of delta
electrons, sets `EnergySelection` on every hit and clears it on the hits of proton candidates; every hit is written,
so only the flags change. It is off by default and runs with `Chain(deltafinder=True)`, `decide --deltafinder` or
`TRIGGER_DEVICE_DELTAFINDER=1` (bench); with it off, no hit carries `Background` (as before). With it on, the stages
that read the flags see them: `CalTimePeakFinder` (cpr, `BkgSelBits`) and the helix fits (`HitBackgroundBits`).
`TimeClusterFinder` runs with `TestFlag: false` and reads no flag, so the tpr time clusters do not change, nor do the
apr ones unless the TZClusterFinder port is on (`tz=True`, which needs the flags).

The port follows the stock containers and orders: hits time-ordered by the radix sort's key (the early-end time,
`ComboHit::time()`, which the panel hits now carry as `etime`), the per-face time bins with the out-of-range reads of
an OnSpill job, seeds in creation order per station, the delta-candidate vector, the proton candidates sorted with
libstdc++'s `std::sort`; float members in float32, double members and promotions in float64, products not fused
(`exact.py`). DeltaFinder reads the nominal tracker (`GeomHandle<Tracker>`), not the aligned one the hits are made with:
the straw-0/1 positions of each panel are shipped in `data/tracker_nominal_panels.json`. An event where stock reads
outside its arrays, or a table overflows, gets error bits (`TrigFlagPH:err` in the aux arrays; none on the
production files). `tests/deltafinder_reference.py` is a literal transcription; the unit test requires the port to
match it bit for bit.

On the production CE files (Offline v13_41, OnSpill, all secondaries; development 14,052 events, held-out 13,992),
the TrigFlagPH flags from the stored StrawDigis equal the stock ones on every hit of every event, in both precisions
(stock flags `Background` on about 1,500 hits per 1,000 events, in 8 % of them). With the port on:

| time-cluster stage, events discordant with stock (dev / held-out) | off | on |
|---|---|---|
| cpr (`CalTimePeakFinder` peak counts and `TimeClusterFilter`, 11 paths) | 8 (De paths), 44 (`cpr_HlxUe_40`) on file 1 | 0 / 0 |
| tpr (`TimeClusterFinder`) | 57-205 / 67-173 per path | unchanged |
| apr (`TZClusterFinder`, device: `TimeClusterFinder`) | 124 / 132 | unchanged |

The tpr discordance comes from the calo-seeded time clusters (`UseCaloCluster: true`), which the device does not make:
against a stock replay with `UseCaloCluster: false` it is 0 on every tpr path (2 held-out events on
`tpr_HlxUe_50m30p`). The apr one comes from the time-clustering approximation; with `tz=True` it is 0 (see
[apr time clusters](#time-clusters-of-the-apr-paths-tzclusterfinder)). The port costs about 0.15-0.2 s per
event on four CPU cores (compiles included); its GPU rate is not measured.

### Multi-helix paths (mpr)

The mpr paths are off by default: they run with `Chain(mpr=True)`, `decide --mpr` or `TRIGGER_DEVICE_MPR=1` (bench), and
are otherwise listed as skipped. With them off, no other path changes.

`mpr.py` ports `RobustMultiHelixFinder` (Compton flagging, triplet circle search with the per-panel stub selection,
the dz/dphi scan and fits, the z-phi and z-t filters, the chi2 circle refit with hit removal, duplicate removal) and
`MergeHelices` (overlap, hit-count and chi2 rules; the merged seed's `LsqSums4` circle chi2). The helix cuts read the
`HelixTool` quantities of the merged seeds. The port follows the stock arithmetic, so that the discrete choices of the
finder come out as in stock:

- hit order: each time cluster's hits in the stock `TimeCluster` order (assigned hits in index order, recovered hits in
  the order they are added; `timecluster.py` carries this order key when an mpr path is on), and the per-panel sort with
  the libstdc++ `std::sort` (introsort, not stable above 16 hits);
- float32 values with the double-precision steps of the C++ (`1.0 - c2`, `1.0/res2`, the loop counts, the Compton
  binning, `polyAtan2`'s `+ M_PI`, `HelixTool`'s loop count, `LsqSums4`), and the float `cos` / `sin` where stock calls
  them;
- no fused multiply-add: XLA contracts `a*b + c` on CPU and no flag turns that off, so each product passes through a
  bitwise guard that the compiler cannot fold.

These hold in both precisions; `float32` / `float64` change only the inputs. `tests/mpr_reference.py` is a literal
transcription of `findHelicesInTC` with these semantics, and the unit test requires the port to match it bit for bit.
Each time cluster's hits are gathered into 64 slots (`<path>:overflow` counts clusters with more). Where the sort would
reach its heap-sort fallback, the port uses a stable sort instead (`<path>:fallback` counts these events). The seed fit
and its cuts use the response layer: one scalar KalSeed factor from the development shards (`ks` = stock KS/HS over
device KS/HS = 0.9909), in both shipped response files.

Device on stock digis, against the stock replay (development: 19,543 events; held-out: 19,627):

| | stock | float32 | float64 |
|---|---|---|---|
| development: time-cluster / helix filter passes | 17,949 / 17,452 | 17,949 / 17,452 | 17,949 / 17,452 |
| held-out: time-cluster / helix filter passes | 18,056 / 17,562 | 18,056 / 17,562 | 18,056 / 17,562 |
| held-out path acceptance, paired difference | 0.8831 | -0.33 +- 0.10 pp | -0.16 +- 0.10 pp |

The panel hits are those of stock (bit for bit) and the hit ratio uses `HelixTool`'s nominal face z, so the finder
sees stock's inputs. All helix quantities agree with the stock printout in about 98.6 % of the events with a time
cluster (both precisions give the same helices); the hit ratio differs in 0.7 % of the helices. Clusters with more
than 64 hits (28 development and 19 held-out events) and 2 development events with the sort fallback account for part
of the rest. Adding the mpr paths changes no decision of the other paths. The port is not tuned for speed: a decide
job with the mpr paths on takes 0.6-1.5 s per event on four CPU cores (the other paths together take about 3 ms), and
its GPU rate is not measured.

### Calorimeter-seeded helices (cpr)

With `Chain(cpr=True)` the cpr paths run the calo stage, `CalTimePeakFinder` (`cpr.py`) and then `calhelix.py`: the
port of `CalHelixFinder` (with `CalHelixFinderAlg`) per (event, time peak), both helicities, and `MergeHelices` per
event over the finder's seeds in time-peak order. `HelixFilter` reads the merged seeds through `HelixTool` (non-outlier
hits, the nominal face z), with MergeHelices' recomputed chi2dXY / chi2dZPhi. The port equals the literal
transcriptions `tests/cal_helix_reference.py` and `tests/merge_helices_reference.py` bit for bit, as the mpr port does
(float32 values, the double steps in float64, no fused multiply-add, chained divisions kept as written, libstdc++
`std::sort` for the face-ordered hits, the walk order of every face/panel/hit loop for the sums), including the stock
state that crosses calls (the triplet search's shared slope, the rescue trials' add/remove rounding and counters).

Geometry inputs (`calhelix.geometry`): face z and panel azimuths from the nominal tracker (`data/tracker_nominal_panels.json`,
shared with the DeltaFinder port), the calorimeter offset, disk origins and tracker centre as Offline builds them; the
field at the tracker origin (1 T) is provisional. A peak keeps at most `CHF_HITS` selected hits (more: `:hxoverflow`); stock behaviour the port
flags instead of reproducing (a findDfDz throw, a non-terminating phi-z cleanup) is in `:hxerr`. The calo-cluster
slots of the peaks are fixed (`CPR_CLUSTERS`, more: `:overflow`). `strawart.py` reads the stored StrawDigis, path
status and HelixSeeds of the stock files; `cpr.calo_batch` adds the event's CaloDigis to a block (with `digi_slots`,
one digi axis per file, so fewer shapes compile).

`KalSeedFilter` (CalSeedFit) is the response layer, as for the tpr and mpr paths: its cuts on the merged seeds'
helix quantities, times a pass factor per path from the development files (both shipped responses, draw keys 600+).

Validation on the production CE files (Offline v13_41, OnSpill), with `deltafinder=True`; development files 1-5
(14,052 events) and held-out files 6-10 (13,992), the split and the rule (time-cluster and helix stages equal to stock,
every path within 2 % / 2 sigma) declared before the held-out read:

| | development | held-out |
|---|---|---|
| TimeClusterFilter and HelixFilter stages, 11 paths: events that differ from stock | 0 | 0 |
| `cpr_HlxDe_50`, `cpr_HlxUe_40`: discordant events | 0 | 0 |
| 9 `cpr_TrkDe_*` paths (response factors from development): paired difference | factors 0.992-0.997 | -0.16 to +0.13 pp, within 1.9 sigma |

Both precisions. Against the stored stock HelixSeeds, the reference finder and merger reproduce every seed
bit for bit on file 1 events 0-299 (De 170, Ue 109), and the port equals the reference in both precisions. In events
where every time peak has fewer straw hits than the paths' TimeClusterFilter minimum, stock never runs the finder
(the filter stops the path first), so no seeds are stored; the decisions do not depend on it.

### Helices of the tpr paths (RobustHelixFinder)

With `Chain(tpr_helix=True)` (which needs `deltafinder=True`; the one-call path only) the tpr paths take their helices
from `rhelix.py`, the port of `RobustHelixFinder` (Offline v13_41_01, TrkPatRec, with TrkReco `RobustHelixFit`), the
menu's `TrigHelixFinder` and `TrigHelixFinderUe`, and of `MergeHelices` (`TrigHelixMergerDe` / `Ue`). Per time
cluster it follows the stock module: the face-ordered hits (libstdc++ `std::sort` by panel), the hit prefilter (boost
P² medians), the median circle fit over hit triplets (MedianCalculator's weighted medians), the phi-z init and fit,
both helicities, the hit clean-up, `updateT0` (boost's weighted mean and variance, the calo cluster's time first) and
`pickBestHelix`. Float where stock computes in float, double where it promotes, no fused multiply-add, and the float64
divisions by a scalar made elementwise (XLA compiles them as reciprocal products, 1 ulp off).
`tests/robust_helix_reference.py` is the literal transcription; the port equals it in both precisions.
`MergeHelices` keeps a seed whose time cluster has a calorimeter cluster over an overlapping one without (`UseCalo`).

Fixed capacities, flagged in `<path>:rhoverflow` (bits in `rhelix.py`): 96 selected hits per time cluster, 16 seeds
per event.

Against the stored stock seeds (1,500 CE and 1,500 cosmic events, stock time clusters as input; every field and hit
list equal): the transcription equals stock on every finder seed (CE De 1,051, Ue 1,282; cosmic De 1,147, Ue
1,225), the port equals the transcription in both precisions, and the merged seeds equal stock (CE 1,039 / 1,032,
cosmic 1,083 / 1,106).

In the chain the time clusters come from the `TimeClusterFinder` port, which differs from stock in two ways: it does
not make the calo-seeded clusters (stock runs the tpr finders with `UseCaloCluster: true`), and its hit membership is
not exact (`atan2` where stock uses `polyAtan2`, and the order of the MLP sums), so in some events a cluster has a
hit more or less than stock's. The chain is therefore compared with stock replays of the v13_41 menu with
`UseCaloCluster: false` on both tpr finders (CE files 1-5 development, 6-10 held-out; cosmic file 1 development, 2-3
held-out), and the rule (the tpr HelixFilter stage equal to stock in every event whose device time clusters equal
the stored stock ones, hit lists in order; the other events counted) declared before the held-out read, both
precisions:

| | CE development (1-5) | CE held-out (6-10) | cosmic development (1) | cosmic held-out (2-3) |
|---|---|---|---|---|
| events | 14,052 | 13,992 | 8,269 | 16,567 |
| tpr HelixFilter stage (6 paths), events with equal time clusters that differ from stock | 0 | 0 | 0 | 0 |
| events whose device time clusters differ, De / Ue finder | 86 / 218 | 100 / 230 | 1,760 / 1,704-1,705 | 3,644 / 3,452 |
| tpr TimeClusterFilter stage, events that differ from stock (per path) | 0 | 0-2 | 66-96 | 123-193 |
| events with `rhoverflow` | 0 | 0 | 2 | 2 |

`tpr_HlxDe_30p_IPAPhi` is left out of the HelixFilter-stage row: its HelixFilter applies the D0-phi prescale on the
host, after the stage the chain reports, so only its final decision can be compared. IPAPHI_NOTE

With the stock time clusters as input the finder reproduces the stock seeds, so the remaining differences are those of
the time clusters: an exact tpr helix stage needs an exact `TimeClusterFinder` port and its calo seeding. A decide
job with `tpr_helix=True` (and DeltaFinder) takes 0.13-0.30 s per CE event and 0.40-0.57 s per cosmic event on four
CPU cores, compiles included (the finder alone: 14-17 ms per CE event and 60-92 ms per cosmic event per finder
instance); the GPU rate is not measured. The KalSeedFilter stage stays the response layer, whose tpr factors were
measured with the compact helices.

### Time clusters of the apr paths (TZClusterFinder)

With `Chain(tz=True)` (which needs `deltafinder=True` and the event's CaloDigis in the batch, as for cpr) the apr
paths take their time clusters from `tzcluster.py`, the port of `TZClusterFinder` (Offline v13_41_01, CalPatRec), the
menu's `TrigTZClusterFinder`. It reads the TrigFlagPH panel hits (it skips energy-selected hits flagged Noisy, Dead or
Background) and the `CaloClusterFast` clusters of the calo stage. The stages run in the stock order: per-plane hit
lists in descending time, chunks, combined chunks, recovered hits (LsqSums2 line fits, in float64), combined lines,
proton counting, then the calo step, which attaches matching calo clusters (a second match copies the chunk) and
recovers clusters around unmatched ones. The port equals the literal transcription `tests/tz_cluster_reference.py`
(float32 where stock computes in float, libstdc++ `std::sort` with its ties, no fused multiply-add), which equals the
stored stock TimeClusters, hit list in order, t0, nsh and calo cluster, in every event checked (1,500 CE and 1,500
cosmic events, from the stored TrigFlagPH flags). `strawart.decode_time_clusters` reads the stored TimeClusters.

`TimeClusterFilter` reads `nStrawHits` and, with `requireCaloCluster`, the cluster's calo cluster (`apr_TC_calo`:
minimum 15 straw hits, with a calo cluster). The other apr paths' time-cluster stage is the same filter without the
calo requirement, so with `tz=True` it is exact too; their helix and track stages stay the approximation and the
response layer.

Two things are flagged instead of reproduced (aux arrays `<path>:tzoverflow` and `<path>:tzindet`):
- fixed capacities: 64 chunks from the chunk step, 16 chunks appended by the calo step, 16 output clusters (the
  production files reach 29, 3 and 8);
- stale state: the calo recovery reuses the module's chunk record without resetting its energy selection, so a
  recovered cluster counts as an electron cluster according to the last chunk stored before it, possibly in an earlier
  event. `tzcluster.clusters` resolves that within a call whose events are consecutive in file order (`nrg_in`,
  `nrg_out` carry it across calls). The chain's blocks may be sorted by hit count, so it calls the port with
  `carry=False`: a recovery that needs an earlier event's value is flagged. On the production files every recovered
  cluster was decided by its own event.

Validation on the v13_41 menu replays of [Calorimeter paths](#calorimeter-paths) (CE files 1-5 development, 6-10 held-out;
cosmic file 1 development, 2-3 held-out), the rule (the apr time-cluster stage equal to stock in every event, no
discordant `apr_TC_calo` event) declared before the held-out read, both precisions:

| | CE development | CE held-out | cosmic development | cosmic held-out |
|---|---|---|---|---|
| events | 14,052 | 13,992 | 8,269 | 16,567 |
| apr time-cluster stage (14 paths): events that differ from stock | 0 | 0 | 0 | 0 |
| `apr_TC_calo` passes, stock / device | 7,676 / 7,676 | 7,709 / 7,709 | 3,790 / 3,790 | 7,558 / 7,558 |
| `apr_TC_calo` discordant events | 0 | 0 | 0 | 0 |
| flagged events (`tzoverflow`, `tzindet`) | 0 | 0 | 0 | 0 |

On one cosmic event the DeltaFinder port flags 3 hits differently from stock; the TZ clusters of that event differ,
but no filter decision does. A decide job with `tz=True` (and the cpr calo stage) takes 0.07-0.24 s per CE event and
0.3-0.9 s per cosmic event on four CPU cores, compiles included; the GPU rate is not measured.

The apr response factors (`data/response_ce_dio.json`, `data/response_rpc.json`) were measured with the approximate
time clusters (`tz=False`); with `tz=True` they have not been measured again.

### Calorimeter paths

`calo.py` ports `CaloHitMakerFast` and `CaloClusterFast` (Offline v13_41: CaloDigi times in ticks, `CalCalib` from the
configuration) and the calo filters; `calopaths.py` reads the paths from the menu (`PrescaleEvent`, the calo stage,
the filter); `calorun.py` decides them on stock art files and compares them with the decisions and clusters stored
there. The geometry is built as Offline builds it (`calogeom.py`, from Offline's crystal file in `data/`). The port keeps
the stock orders where they decide a result:

- digis in (SiPM id, t0) order, hits per crystal in pulse creation order, float pulse averaging;
- the cluster seeds sorted with libstdc++'s `std::sort` on stock's comparator itself (times within 1 ps compare by
  crystal id, so it is not a strict weak order), the seed window found with stock's iterator scan, which never reaches
  back to the first hit, and stock's used-hit sentinel (a hit whose index equals the number of selected hits counts as
  used);
- clusters grown breadth first in stock's neighbour order (first ring, then second), members sorted by energy with
  `std::sort`, every sum in the stock order;
- `CaloLikelihood`'s template projections and bin lookup as stock computes them, `CaloNNEval`'s MLP with sequential float
  sums, `ClusterUtils`' moments.

On the production CE files (Offline v13_41, OnSpill; development 14,052 events, held-out 13,992), the device clusters
equal the stored stock clusters in every field and the 11 paths have no discordant event, in both precisions.
`calo_cosmic` and the two prescaled digi-count paths do not fire on these files. They are validated against the
v13_41 menu replayed in stock on other digis, with the device deciding the same files:
- **cosmic digis** (`CosmicSignalOnSpill`, MDC2025be, 3 files, 24,836 events; their digitiser and calo trigger modules
  are those of v13_41, except `CaloClusterFast`, which the replay runs in its v13_41 form): one development file, two
  held-out. Clusters equal in every event and no discordant event on the 11 paths, in both precisions; `calo_cosmic`
  passes 739 / 1,469 events (development / held-out), `minBias_CDCount` 14 / 44, `calo_N0Source` 2,668 / 5,536.
  `CaloCosmicCalib` calls `abs` on floats: the float overload is the one stock takes (the int one would change 30
  development and 44 held-out decisions).
- **the CE files with the two prescales lowered** (`CaloN0SourcePS` 3, `MinBiasCDCountPS` 1), development 1-5 and
  held-out 6-10: no discordant event; `calo_N0Source` passes 4,693 / 4,670 events, `minBias_CDCount` none (every CE
  event has fewer than its 200 calo digis).

**Inputs from the calorimeter fast simulation.** The stage reads CaloDigis as the v13_41 `CaloDigiMaker` writes them
(SiPM id, `t0` in 5 ns ticks, waveform, peak position). It does not read the clusters of the calofastsim device chain
(`calofastsim.device_chain`): those come from the offline reconstruction (template fit, `CaloClusterMaker`), which the
trigger does not run. To run on simulated events, the stage takes the chain's digis:
- only the chain's `"waveform"` digi response makes digis; the default `"parametric"` response draws hits and makes
  none;
- the chain's digis follow the earlier `CaloDigiMaker` (`t0` in ns), and the device chain's digi dictionary has no
  peak position (the host `digi.extract_digis` has it). The stage needs `t0` in ticks and the peak position.

The per-entrant shower library (`calofastsim.showers.entrants`) replaces the chain's shower draw; digitisation and
everything after it stay the same. When it becomes the device default, the calo stage does not change, but its
inputs do: the library adds the inner-edge entrants (its inner-edge face), so the clusters near the inner radius, and
with them the calo path rates, are validated again against stock on the chain's output.

## Use

The response layer is chosen per process with `--response` (bench: `TRIGGER_DEVICE_RESPONSE`; `beamfastsim.centuple
trigger --response`): the named responses of `response.py`, shipped in `data/`, or a response JSON file.

| `--response` | file | binned rows | built from |
|---|---|---|---|
| `ce` (default), `dio` | `data/response_ce_dio.json` | the 12 `apr_(Trk\|Hlx)` paths | CE + DIO development shards |
| `rpc` | `data/response_rpc.json` | the 19 `(tpr\|apr)_(Trk\|Hlx)` paths | RPC development shards |
| `none` | - | no response layer (probability 1) | - |

Paths without a binned row keep the scalar hs/ks factors from CE development shards (`response`) in both files.
`ce` is the default of `decide` and the bench. `beamfastsim.centuple trigger` defaults to `none`: its `assemble` step
applies the on-spill trigger table (trigcal) by default, which replaces the response layer; pass `--response ce` for
decisions used without the table.
Development runs (inputs of `response`, `response-binned` and trigcal) use `--response none`; `response` and
`response-binned` refuse a decide output with a response layer.

```bash
# development shards: device decisions, then the response layer from the stock replay of the same shards
python3 -m trigger.device.run decide --dump dev/*/trgdump.root --menu replay.fcl --tc-mva TimeCluster.weights.xml \
    --response none --logs dev/*/replay.log --prepared PREPARED --output dev.json
python3 -m trigger.device.run response --logs dev/*/replay.log --device dev.json --menu replay.fcl --output response.json
# held-out shards: paired per-path comparison with the stock replay
python3 -m trigger.device.run decide --dump hold/*/trgdump.root --menu replay.fcl --tc-mva TimeCluster.weights.xml \
    --response response.json --logs hold/*/replay.log --prepared PREPARED --output hold.json --check-hits
```

Binned response for the apr paths with a helix stage (AgnosticHelixFinder is not ported, and one scalar factor
from CE does not carry to other samples): decide with `--response none` on the development shards of each sample, then

```bash
python3 -m trigger.device.run response-binned --device ce_dev.json dio_dev.json \
    --logs-sets "ce_dev/*/replay.log" "dio_dev/*/replay.log" --prepared PREPARED --base response.json \
    --output response_binned.json
```

Each `apr_(Trk|Hlx)` path (`--paths`) gets one pass probability, binned in the straw hits of the event's best time
cluster (`--edges`) and the momentum of the best device helix (`--edges-p`); it replaces that path's hs/ks factors.
Every other path keeps its row from `--base`, so its decisions are unchanged. This is how `data/response_ce_dio.json`
was built (the `ce` and `dio` response). Held-out shards, device on stock digis, paired per event with the stock replay
(device - stock):

| | DIO float32 | DIO float64 | CE float32 | CE float64 | CE, scalar response (float32 / float64) |
|---|---|---|---|---|---|
| apr_TrkDe/TrkUe (11 paths) | -0.73..+0.94 pp | -0.76..+0.05 pp | -0.86..+0.48 pp | -1.00..+0.42 pp | -0.09..+0.84 / +0.30..+1.04 pp |
| apr_Hlx (2) | -0.22..-0.10 pp | -0.08..+0.09 pp | -0.68..-0.62 pp | -0.81..-0.69 pp | +0.04..+0.16 / -0.09..+0.11 pp |

On DIO the scalar response was +3.5..+7 pp. On CE the binned response is no better than the scalar one: it moves single
apr paths by up to 1.9 pp and apr_Hlx to -0.7 pp, and on OnSpill CE (stock digis) its apr paths are -1.13..+0.15 pp,
worse than the scalar response there. The scalar-only file is not shipped; build it with `response` and pass it as a
file to compare.

**Per-process response (RPC).** The response from CE shards does not carry over to RPC events. In RPC events the stock
helix and seed-fit stages lose 9–19 % of the events whose device helix passes, against 2–3 % in CE events, at
the same helix momentum, hit count and time-cluster time, and at the helix stage for e- and e+ alike. The cause is not yet
found (not the DeltaFinder flags, not MergeHelices). Build an RPC response from RPC development shards instead, with
every tpr/apr track and helix path binned and momentum rows for the RPC spectrum (`--response rpc`, built as below,
for both precisions); use it only when simulating RPC:

```bash
python3 -m trigger.device.run response-binned --device rpc_dev.json --logs-sets "rpc_dev/*/replay.log" \
    --prepared PREPARED --paths "^(tpr|apr)_(Trk|Hlx)" --edges 0 15 20 25 30 35 40 50 60 80 1000 \
    --edges-p 0 70 80 90 100 110 1e6 --min-count 100 --base response.json --output response_rpc.json
```

Validation on Run 1A production RPC files (5 development, 5 held-out with 31,177 events; stock digis, paired with
the stock replay): within 2 % or 2 sigma on 17/19 tested paths in float64 and 18/19 in float32, against 4/19 and 3/19
with the CE response (misses: `apr_Hlx_30` -4 %, and in float64 `tpr_TrkDe_50_D0200` +3 %). On fast e- e+ pair digis
per generated capture (250,000 captures, the stock files' count), the tpr and apr track paths are 0.98–1.00 of the
stock replay in float64 and 0.96–0.99 in float32 (±0.027), against 1.04–1.30 with the CE response; `apr_Hlx_*`
0.94–0.96. The union over paths stays about 5 % low under either response: on the device the apr paths use the tpr helix
fit, so they fire on the same events as tpr; in stock they partly fire on others.

Inputs: `trgdump.root` from `StrawHitTableDumper` (tracker package) run in the replay job on the trigger's own
collections (`StrawHitTag: TrigMakeSH`, `PanelHitTag: TrigMakePH`, tree `trigph` (`--tree` for another name)), which also gives the stock hits for
`--check-hits`; `replay.fcl`, `manifest.json` and `inventory.json` from `trigger/replay.py`; the `TimeCluster.weights.xml`
MLP from Offline `TrkPatRec/data`. Timing: `trigger/device/bench.py` with `mu2efastsim.bench`.

Device digis (`trkfastsim.digi.digi_jax.digitize` output: flat `ev`, `sid`, `tdc`, `tot`, `pmp`, ordered by event and
StrawId) go to `Chain.decide` through `digis.blocks`, which packs them into the dense event blocks without a
per-event loop; events without digis keep their place.

`--precision float32` (bench: `TRIGGER_DEVICE_PRECISION=float32`) runs the device arrays in 32-bit, except the stock
double-precision steps of the hit reconstruction and of the mpr finder: those trace in float64 in both precisions
(`exact.py`), so the hits are the same in both. The default `float64` keeps the stock types throughout.
JAX's x64 flag is process-wide, so a process runs one precision. float32 also sets the process-wide
`jax_default_matmul_precision` to `highest`, so the helix matmuls do not run on TF32 tensor cores. The time-cluster MVA
is float32 in both precisions, as stock, and asks for `precision="highest"` itself.

### Batching and CPU settings

`trigger/device/bench.py` batches events by digi count (`TRIGGER_DEVICE_ORDER=hits`, the default; `input` keeps the
dumps' order in each batch): every block pads its hit axis only to its own largest event, and the per-event results
are put back in event order before the prescale counter, so the decisions are the same bit for bit. The hit-axis
floor is `TRIGGER_DEVICE_HMIN` (default 64, as `hits.pack_events`).

`run.py decide` batches the same way (`--order hits`, the default; `--order input` keeps the dumps' order).

On a CPU backend (one core per process, as a grid slot) both use small batches and the compact cluster path by default
(`run.CPU_BATCHING`: batch 16, hit-axis floor 32, compact on; override with `TRIGGER_DEVICE_BATCH`, `TRIGGER_DEVICE_HMIN`,
`TRIGGER_DEVICE_COMPACT=0|1`, or `decide --batch`, `--hmin`, `--no-compact`). The compact path
refines time clusters and fits helices only on the occupied (event, cluster) pairs (about a quarter of the
`NTC` slots), in per-producer device calls with a host read of the counts between them; pair capacities are 1/4, 1/2 or
all of the batch's cluster slots, and the bench compiles them all in its first pass (setup), so later passes compile
nothing. Same per-cluster computation and the same decisions as the one-call path; on a GPU keep the one-call path
(the host reads would stall the device).

In the compact path's helix fits the XY circle is fitted once per cluster for both helicities, and the lambda coherence
scan (98 lambdas) sums over the XY-selected hits only. Those are at most one per face (72), and the other hits carry
zero weight. The scan runs at the narrowest of 16, 32 or 72 slots that holds the launch's largest selected-hit count,
or over every hit when the hit axis is narrower. The helix outputs are bit-identical to the full scan's in float32 and
float64. One Xeon Gold 6248 core, 6,000 events per pass, same node:

| | float32 (events/s) | float64 (events/s) |
|---|---|---|
| scan over every hit | 781 / 782 | 464 / 458 |
| selected hits only | 838 / 832 (x1.07 / x1.06) | 592 / 595 (x1.28 / x1.30) |

(The second number of each pair is from the run with the two trees on swapped cores.) The three scan widths add
30-55 s of compile at setup. Process memory at these settings is 3.9 GB (float32) / 4.9 GB (float64).

### One-call path on a GPU

The one-call path (the GPU default) keeps the number of sequential loop iterations per batch low. With device loops
off, each iteration is a host round trip; W8 runs with device loops on (`mu2efastsim.devloops`), which keeps the loops
on the GPU, and the fewer iterations still help there:

- Panel hits combine over the run position of the hits in their panel, not hit by hit.
- Time-cluster recovery jumps straight to the next hit that passes the time and phi window, and evaluates the hit MVA
  there.
- The peak scan stops at the last candidate.
- Producers that differ only in pitch (`TrigTimeClusterFinder`, `TrigTimeClusterFinderUe`), and helix finders with the
  same settings, run in one batched call.
- The helix scans are unrolled.
- Cluster refinement and helix fits run only on the gathered (producer, event, slot) pairs with a candidate or a valid
  cluster (pairs, the default; `TRIGGER_DEVICE_PAIRS=0` or `decide --no-pairs` keeps the full batched path). A batch
  with more pairs than one per event and producer takes the full batched path in the same call.

All of these give the same decisions bit for bit. Warm rate on an NVIDIA A100-SXM4-80GB, 13,000 fresh events per pass,
batch 1024, median of 3 runs interleaved in one job:

| | float32 (events/s) | float64 (events/s) |
|---|---|---|
| before these changes | 8,731 | 8,019 |
| fewer loop iterations | 23,179 | 20,553 |
| plus pairs (default) | 29,185 | 27,852 |

Setup with a warm compile cache is 13 s (float32) / 14 s (float64) with pairs, and 9 s / 8 s without.

The results of a batch are then stacked on the device into a few arrays (`Chain.decide_packed`), so they reach the
host in 4 copies instead of about 170, and W8 turns device loops on. Second job, same settings, its own control:

| | float32 (events/s) | float64 (events/s) |
|---|---|---|
| before these changes | 8,740 | 7,994 |
| fewer loop iterations and pairs, device loops on | 31,241 | 28,948 |
| plus packed results, device loops off | 39,594 | 35,383 |
| plus packed results, device loops on (default) | 44,148 | 39,137 |

Device loops on and off give the same decisions on the GPU, bit for bit, and the same as the CPU: 39,000 events x 21
paths in each dtype.
