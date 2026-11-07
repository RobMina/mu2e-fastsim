# Run1A raw reference and cohort census (opt-in)

The stock MDC2025ac raw-output FCL retains every event, all species and complete
raw ancestry; it removes downstream output selection, not stock physics cuts.
The compact ReadVirtualDetector diagnostic retains mu/pi/proton records and
all-species DS Beam StepPoints. Protons preserve empty-source bookkeeping.
The scripts do not build Offline or a custom recorder.

The campaign launcher uses 200 × 5,000 POT, at most 32 concurrent jobs, and
1 CPU / 12 GB / 24 h per shard. Indices 0–99 are development and 100–199 held out.
Seed spacing of 1,000 exceeds the stock 100-engine block. Requested counts are not
finished exposure. Smokes are excluded, retries require fresh output directories,
and no automatic requeue runs. Both generation launchers force
`OFFLINE_WARM_REPEAT=0`, including when inherited as 1, so the runner cannot
repeat the same-seed raw-art job for timing. Save the FCL, batch script and
existing `offline_uva.sh` under `$TS_CAMPAIGN_ROOT/config/`, and create its `logs/`
and `shards/` directories before submission:

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
ID for completed logs, never a built-in job ID. The status wrapper also requires
`TS_STATUS_OUT` (fresh directory) and `TS_STATUS_FCL`; it runs only the stock
StatusG4Analyzer on completed inputs, not simulation. Its default cohort is
development; set `TS_STATUS_COHORT=heldout` for held-out inputs. Warm repeats are
disabled. Undecodable uproot streamers are reported explicitly, not interpreted
as bad physics. Histogram flow bins count.

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
The geometry and field must match the saved Run1A stock inputs, not Run1B DSOff.

## Explicit cohort selection and completed exposure

Census and reference merges default to development and reject held-out indices.
`tscampaign --cohort heldout` and `tsrawreference --combine ... --cohort heldout`
accept indices 100–199 / subruns 920100–920199 only; never mix cohorts. Raw bulk
outputs remain at the production site.

| Cohort | Counted Events | Recorded entrances | Known-root inputs | Resolved stock DS links | Upstream ancestry gaps | Nested entrances |
|---|---:|---:|---:|---:|---:|---:|
| Development | 500,000 | 25,080 | 25,078 | 1,118 | 2,696 | 2 |
| Held-out | 500,000 | 25,339 | 25,334 | 1,171 | 2,665 | 5 |

Together these independently count all 1,000,000 production Events. The held-out
event/source sets are disjoint from development; all held-out stock statuses are
zero. The completed held-out comparison used the frozen 2.5 mm / 0.25 mm / 5%
caps and seed 20261105, with no post-held-out retuning. It is consumed, not a fresh
test. See the [transport guide](ts-transport.md) for results and limitations,
the [history guide](ts-history.md) for reusable diagnostics, and the
[sensitivity guide](ts-sensitivity.md) for comparisons of saved outputs.

Tests: `test_tssmoke.py`, `test_tscampaign.py`, `test_tsrawreference.py` and
`test_tsrun1a_campaign.py`; launcher tests use fake runners, not Slurm/art.
