# Stock trigger replay

Replays the production trigger menu that is embedded in a stock digi file on that file's stored digis, then compares
the replayed decisions with the stored ones path by path. Nothing here changes simulation or trigger defaults. It
needs the matching released Offline runtime and stock digis; no C++ build, new Geant4 or grid job.

| File | Does |
|---|---|
| `replay.py` | parses the `config_dumper -P` output, writes `inventory.json` (every scheduled path, its modules, prescales, output selection, geometry, conditions), `replay.fcl` and `manifest.json` |
| `replay.sbatch` | runs `replay.fcl` through `container/offline/offline_uva.sh`, then `compare.py` (or `timing.py` with `TRIGGER_MODE=timing`) |
| `compare.py` | per-path comparison of the stored (`TriggerResults::Digitize`) and replayed decisions printed by the job, keyed by run/subrun/event |
| `menu_diff.py` | module-setting differences between two inventories (e.g. CE and cosmic eras) |
| `timing.py` | trigger-module wall time per event from an art TimeTracker table |

## Use

```bash
config_dumper -P "$INPUT" > "$OUT/input.config.txt"
python3 trigger/replay.py "$OUT/input.config.txt" --output "$OUT/prepared"
sbatch -p <partition> --export=ALL,TRIGGER_CODE=$PWD,TRIGGER_PREPARED=$OUT/prepared,\
TRIGGER_INPUT=$INPUT,TRIGGER_OUTPUT=$OUT/run,TRIGGER_AREA=$RELEASE,TRIGGER_PYTHON=python3,OFFLINE_EVENTS=$N,\
CVMFSEXEC_DIST=$CVMFSEXEC \
  trigger/replay.sbatch
```

The account comes from `-A` or Slurm's `SBATCH_ACCOUNT` (see `docs/running.md`). `$RELEASE` is the released Offline
area on CVMFS (or a muse area) that made the input file. `OFFLINE_EVENTS=0` replays the whole file and takes the
expected count from `TRIGGER_EVENT_LIST` (`file_info_dumper --event-list` output). `compare.py` exits 0 when every
non-stateful path matches; `--strict` requires all paths. `TRIGGER_FORCE=1` lets a run reuse an output directory.
`--quiet` prepares a timing configuration without the per-event decision printer. `--straw-digi-tag` and
`--calo-digi-tag` rebind the trigger to other digi collections in the same events (fast digis).

What `replay.py` changes in the production configuration: it removes `DigitizePath` modules from the trigger paths
(the stored digis are used, not remade), drops the truth-triggerable paths and `SelectRecoMC`, replaces
`StrawDigisFromArtdaqFragments` with `NullProducer`, drops the stored reconstructed collections, and uses a new
process name. Services, conditions, geometry and every filter threshold are the production ones. The replay
process must be the release that made the file: the CE and cosmic productions differ (13_41 vs 13_33, 17 module
settings, calorimeter digi time units).

## Results (2026-11-04)

| Sample | Events | Paths exact | Differences |
|---|---|---|---|
| CE `dig` MDC2025ba55, stored digis | 21,078 | 50/52 | 4 of 1,096,056 event-path decisions, all replay-only: lumiStream 3, `tpr_HlxDe_30p_IPAPhi` 1 |
| CE MDC2025ba55 + ba64, complete digitization history | 28,044 + 28,148 | 52/52 | 0 of 2,921,984 |

The 50/52 run replays the stored (trigger-selected) events only. Two filters keep counters that advance on events
the stored file does not contain, so they cannot match exactly on selected digis:

- **`LumiStreamFilter`** (`lumiStream`) counts processed events, flushes every `eventFreq` events and can pass the
  first event of a subrun.
- **`HelixFilter`** with `prescaleUsingD0Phi: true` (`tpr_HlxDe_30p_IPAPhi`) prescales on its own invocation count
  `_nevt % evalIPAPresc(phi)`, not on the art event number.

`manifest.json` lists both as `stateful_paths`; `compare.py` reports `nonstateful_mismatches` separately but still
counts them in `mismatches` and `equal`. The 52/52 rows replayed every event the digitization job processed (regenerated from its
dts parents with the production seed), so both counters evolve as in production; the regeneration tooling is not
in this package.

**Stock cost.** Production trigger modules take 8.17-8.32 ms per processed dts event for CE and 9.55-10.24 ms for
cosmics (two production logs each, TimeTracker). A quiet CE replay on UVA: 9.54 ms of trigger work per stored
event, 11.22 ms per event with I/O.

The comparison uses the accepted/rejected bit per path. A decision line in any other state (not run, exception) is
an error, not a rejection.
