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
| track fit + cuts | `KalSeedFit` + `KalSeedFilter` | response layer: the filter's cuts on helix quantities and a pass probability per path from development shards | statistical |

Not evaluated on the device: calorimeter (`calo_*`, `cpr_*`) and cosmic (`cst_*`) paths, paths with
`MultiHelixFilter` or `minNHelices > 1`, `mpr_*` (multi-helix finder), `apr_proton`.

## Use

```bash
# development shards: device decisions, then the response layer from the stock replay of the same shards
python3 -m trigger.device.run decide --dump dev/*/trgdump.root --menu replay.fcl --tc-mva TimeCluster.weights.xml \
    --logs dev/*/replay.log --prepared PREPARED --output dev.json
python3 -m trigger.device.run response --logs dev/*/replay.log --device dev.json --menu replay.fcl --output response.json
# held-out shards: paired per-path comparison with the stock replay
python3 -m trigger.device.run decide --dump hold/*/trgdump.root --menu replay.fcl --tc-mva TimeCluster.weights.xml \
    --response response.json --logs hold/*/replay.log --prepared PREPARED --output hold.json --check-hits
```

Opt-in binned response for the apr paths with a helix stage (AgnosticHelixFinder is not ported, and one scalar factor
from CE does not carry to other samples): decide without `--response` on the development shards of each sample, then

```bash
python3 -m trigger.device.run response-binned --device ce_dev.json dio_dev.json \
    --logs-sets "ce_dev/*/replay.log" "dio_dev/*/replay.log" --prepared PREPARED --base response.json \
    --output response_binned.json
```

Each `apr_(Trk|Hlx)` path (`--paths`) gets one pass probability, binned in the straw hits of the event's best time
cluster (`--edges`) and the momentum of the best device helix (`--edges-p`); it replaces that path's hs/ks factors.
Every other path keeps its row from `--base`, so its decisions are unchanged. Without the file the default is unchanged.

Inputs: `trgdump.root` from `StrawHitTableDumper` (tracker package) run in the replay job on the trigger's own
collections (`StrawHitTag: TrigMakeSH`, `PanelHitTag: TrigMakePH`, tree `trigph` (`--tree` for another name)), which also gives the stock hits for
`--check-hits`; `replay.fcl`, `manifest.json` and `inventory.json` from `trigger/replay.py`; the `TimeCluster.weights.xml`
MLP from Offline `TrkPatRec/data`. Timing: `trigger/device/bench.py` with `mu2efastsim.bench`.

Device digis (`trkfastsim.digi.digi_jax.digitize` output: flat `ev`, `sid`, `tdc`, `tot`, `pmp`, ordered by event and
StrawId) go to `Chain.decide` through `digis.blocks`, which packs them into the dense event blocks without a
per-event loop; events without digis keep their place.

`--precision float32` (bench: `TRIGGER_DEVICE_PRECISION=float32`) runs every device array in 32-bit, including
the double-precision intermediates of the stock hit reconstruction; the default `float64` keeps the stock types.
JAX's x64 flag is process-wide, so a process runs one precision.
