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
Held-out access is explicit and requires a frozen configuration and authorization.
Default census/reference merges still reject held-out indices. After authorization,
`tscampaign --cohort heldout` and `tsrawreference --combine ... --cohort heldout`
accept indices100..199/subruns920100..920199 only; never mix cohorts. The
`tsholdout_extract.sbatch` reader runs census and compact reference extraction,
not simulation. Set `TS_ANALYSIS_BIND` explicitly to container bind paths covering
the campaign, output and beam code; no site storage path is built in.
Set `TS_STATUS_COHORT=heldout` for the analyzer-only status read;
its default remains development, and warm repeats are disabled.

`tsholdout prepare` checks all100 explicitly named shard outputs,500,000 finished
Events, exact event/source disjointness from development, status/geometry coverage,
and known-root ancestry before saving the reference. `tsholdout run` uses the frozen
2.5mm/0.25mm/5%,seed20261105 baseline once, retains all losses and parent exposure,
and writes charge-separated held-out comparisons. Existing output paths are refused;
there is no held-out seed/settings scan. The selected geometry/field must be the
same saved Run1A inputs used for development. Raw bulk outputs stay at the production site.
Completed held-out:500,000countedEvents,25,339recordedentrances ->25,334knownroots,
all1,171DSlinks resolved;2,665upstreamancestrygaps,5nestedentrances retained. Together
with development this independently counts all1,000,000productionEvents. Frozen
comparison is reported in the TS transport guide; no post-held-out retuning.
Tests: `test_tssmoke.py`, `test_tscampaign.py`, `test_tsrawreference.py`,
`test_tsrun1a_campaign.py`, `test_tsholdout.py`, `test_tsholdout_extract.py`;
launcher tests use fake runners, not Slurm/art.
