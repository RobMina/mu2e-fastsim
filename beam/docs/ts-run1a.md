# Run1A raw reference and development census (opt-in)

The stock MDC2025ac raw-output FCL retains every event, all species and complete
raw ancestry; it removes downstream output selection, not stock physics cuts.
The compact ReadVirtualDetector diagnostic retains mu/pi/proton records and
all-species DS Beam StepPoints. Protons preserve empty-source bookkeeping.
Simulation submissions always require explicit authorization; these scripts do
not build Offline or a custom recorder.

The campaign launcher fixes 200x5,000POT,32concurrent,1CPU/12GB/24h per shard,
with indices0–99 development and100–199 held out. Seed spacing1000 exceeds the
stock100-engine block. Requested counts are not finished exposure. Smokes are
excluded, retries require fresh output directories, and no automatic requeue runs.
Both generation launchers force `OFFLINE_WARM_REPEAT=0`, including when inherited
as 1, so the runner cannot repeat the same-seed raw-art job for timing.
Freeze FCL/script/existing `offline_uva.sh` under `$TS_CAMPAIGN_ROOT/config/` and
create its `logs/` and `shards/` before approved submission:

```bash
sbatch --parsable -A "$SBATCH_ACCOUNT" -p "$SBATCH_PARTITION" \
  -o "$TS_CAMPAIGN_ROOT/logs/%A_%a.log" \
  "$TS_CAMPAIGN_ROOT/config/tsrun1a_campaign.sbatch"
python3 -m beamfastsim.tscampaign --root "$TS_CAMPAIGN_ROOT" \
  --generation-job "$TS_GENERATION_JOB" --index 0 --out census-000.json
python3 -m beamfastsim.tscampaign --combine census-*.json --out summary.json
```

Set `CVMFSEXEC_DIST` to the existing EL8 client and record the returned generation
array ID as `TS_GENERATION_JOB`. The census and `tsstatus.sbatch` use that explicit
ID for completed logs, never a built-in job ID. The latter additionally requires
`TS_STATUS_OUT` (fresh directory) and `TS_STATUS_FCL`; it runs only the stock
StatusG4Analyzer on completed development inputs. Undecodable uproot streamers
are reported explicitly, not interpreted as bad physics. Histogram flow bins count.

```bash
python3 -m beamfastsim.tsrawreference --ntuple SHARD/tsrun1a_raw.root \
  --gdml RUN1A.gdml --out reference-000.npz
python3 -m beamfastsim.tsrawreference --combine reference-*.npz \
  --root-lineages --out development.npz
python3 -m beamfastsim.tsfield field-run1a --release v10_47_00 \
  --field-config Offline/Mu2eG4/geom/bfgeom_no_ds_v01.txt
```

The same-stage adapter uses first forward VD1 mu/pi states and actual DS2Vacuum
Beam StepPoints, not terminal SimParticles or Run1B restart semantics. Exact
parent-key joins retain all endpoints, events and zero-output inputs. Nested
sources require explicit known-root projection; nested states remain saved.
Unresolved DS ancestry or multiple endpoints raise instead of silently dropping
rows. Compact upstream omissions can hide additional nesting: this is not proof
of full primary-root closure. Raw art remains the backing for that question.

Completed development:500,000countedEvents,25,080recordedentrances reduced to
25,078known roots, all1,118stockDSlinks resolved;2,696upstream ancestry gaps remain.
Held-out physics stays unexamined. Raw bulk outputs stay at the production site.
Tests: `test_tssmoke.py`, `test_tscampaign.py`, `test_tsrawreference.py` and
`test_tsrun1a_campaign.py`; launcher tests use fake runners, not Slurm/art.
