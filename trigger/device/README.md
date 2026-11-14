# Tracker trigger paths on the device

Opt-in JAX evaluation of the tracker trigger paths (tpr and apr families) of the production menu, from StrawDigis
to per-path decisions, on a GPU or CPU. Nothing in the stock replay (`trigger/replay.py`, `compare.py`) changes.

| Stage | Stock module | Device | Agreement with stock on stock digis |
|---|---|---|---|
| straw hits | `StrawHitReco` (TrigMakeSH, FitType 5) | port (`hits.straw_hits`) | counts, straw ids, flags exact; floats to float32 rounding |
| panel hits | `CombineStrawHits` (TrigMakePH) | port (`hits.panel_hits`) | counts, ids, flags, combo sizes exact; floats to float32 rounding |
| background flags | `DeltaFinder` (TrigFlagPH) | not run: no hit flagged | exact on primary-only CE digis (stock flags none); not with pileup |
| time clusters | `TimeClusterFinder` | port (`timecluster.py`), incl. the TMVA hit MLP | see the README table of the PR / record |
| time clusters (apr) | `TZClusterFinder` | the TimeClusterFinder port with TZ's `clusterThresh` | approximation |
| helices | `RobustHelixFinder` (tpr), `AgnosticHelixFinder` (apr) | compact fit with the stock per-face hit selection and chi2 definitions (`helix.py`) | approximation |
| helix cuts | `HelixFilter` | same cuts, incl. the D0-phi prescale counter on the host | exact given the helix |
| track fit + cuts | `KalSeedFit` + `KalSeedFilter` | response layer: the filter's cuts on helix quantities and a pass probability per path and process from development shards (shipped, `data/`) | statistical |

Not evaluated on the device: calorimeter (`calo_*`, `cpr_*`) and cosmic (`cst_*`) paths, paths with
`MultiHelixFilter` or `minNHelices > 1`, `mpr_*` (multi-helix finder), `apr_proton`.

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

`--precision float32` (bench: `TRIGGER_DEVICE_PRECISION=float32`) runs every device array in 32-bit, including
the double-precision intermediates of the stock hit reconstruction; the default `float64` keeps the stock types.
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
(the host reads would stall the device). Process memory at these settings is 2.7 GB (float32) / 3.5 GB (float64).

### One-call path on a GPU

The one-call path (the GPU default) keeps the number of sequential loop iterations per batch low, because each
iteration of a device loop is a host round trip when device loops are off (the W8 setting):

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
