# Production trigger replay

**Paused by user — 2026-11-05 06:10 EST.** No R7 jobs are running. Only the first
C response-control parent is complete; the remaining nine and full-history C trigger
have not launched. See the self-contained [pause/resume handoff](PAUSE_RESUME.md)
for saved results, assignments, admission gates and authorization required to resume.

This opt-in diagnostic inventories the **embedded production configuration**, not a guessed current menu. It does
not change simulation or trigger defaults. It requires an existing compatible Offline release and stock digis;
no C++ build, new simulation or grid production is needed.

```bash
config_dumper -P "$INPUT" > "$PRIVATE_OUT/input.config.txt"
python3 trigger/replay.py "$PRIVATE_OUT/input.config.txt" --output "$PRIVATE_OUT/prepared"
# Set up the matching Offline release, then run from a private output directory.
mu2e -c "$PRIVATE_OUT/prepared/replay.fcl" -s "$INPUT" -n 100 > "$PRIVATE_OUT/replay.log" 2>&1
python3 trigger/compare.py "$PRIVATE_OUT/replay.log" \
  --manifest "$PRIVATE_OUT/prepared/manifest.json" --expected-events 100 \
  --output "$PRIVATE_OUT/comparison.json"
```

Use `container/offline/offline_uva.sh` for the supported Slurm container runtime. Account, partition, paths,
release and event limit come from the caller. Never run art on a login node. Generated configurations, raw logs,
input manifests and results belong in private data storage, not in this repository. The preparation and
comparison tools require Python 3.10+; `replay.sbatch` requires explicit `TRIGGER_PYTHON` and can use
`TRIGGER_ANALYSIS_IMAGE` with the existing Python project image. Never assume a Slurm node's system Python is
new enough. Failed post-processing does not justify rerunning completed art: compare its saved log first.

## Complete-history quiet timing companions

`quiet_replay.py` admits a completed full-stock or full-hybrid verbose reference, removes only its decision
printer and existing ArtReport limits, and preserves the original 52 paths, ordered source and decision output.
Preflight rechecks source bytes, baseline art/comparison/configuration and complete hybrid history. Mount aliases
are accepted only with same-inode identity and exact size/SHA, never unrelated identical-byte copies.
Postflight checks the executed configuration/runtime, highest-cycle count, independent input/ROOT event order,
both registries and **exact original plus replay packed words** against the saved verbose reference. It reads
ROOT metadata directly and does not require the removed per-event log records.

`quiet_pair.sbatch` runs the two quiet histories sequentially on one CPU allocation, failing before the next arm
if admission, art or postflight fails. This is a separate timing experiment, not a retry of completed reference
art. Every invocation has a fresh private CVMFS cache; stock then fast is an explicit order, not randomized
replication or controlled warm-cache timing. The two source-art layouts also differ. Report weighted module work
and full-event time separately from source/output/startup. This is trigger-replay timing with retained stock
calorimeter and primary-only tracker limitations, **not full fast-chain/JAX throughput or a production-grid multiplier**.
Production-log hardware/population differences remain explicit.

## Saved-truth partition and runtime binding

`split_source_tables.py` partitions event trees by exact `(run, subrun, event)` and the explicitly named
`subruns` metadata tree by `(run, subrun)`. Matching metadata may accompany multiple intersecting parents;
it is not additive exposure. Tree/schema differences and foreign identities fail before writing outputs.
Use a fresh directory after failure; retain partial files and logs. The split receipt records per-tree rows,
input/output hashes and parent event counts. Array tasks clear inherited whole-export inputs before transport.

For a multi-parent continuation, select a private retained analysis image and set
`TRIGGER_ANALYSIS_SHA256` for the split and transport wrappers; each checks the image before its payload.
The optional split runtime preflight records interpreter/dependency versions and imported frozen transport
module paths/hashes. It tests imports, not physics; it does not replace the production Offline worker image.

## Inventory and input contract

The configuration records the scheduled bit, path, ordered modules, prescale module/configuration, module
parameters, output selectors, geometry and database purpose/version. `trigger_paths` is authoritative for
scheduling; defining a path or marking it enabled in JSON is not proof that a particular production ran it.

The physics menu comprises tracker-pattern, calorimeter-seeded, agnostic and merged-pattern track/helix paths;
calorimeter photon/NN/cosmic/RMC/energy paths; cosmic/time-cluster calibration paths; occupancy/minimum-bias and
luminosity paths. Path definitions and thresholds must come from the particular file's production provenance.

The digi-input configuration uses straw digis (`makeSD`), calorimeter digis (`CaloDigiMaker`), proton-bunch time
and spill marker (`EWMProducer`). Tracker paths can also run calorimeter reconstruction; a tracker-only product
file is insufficient. Firmware PMP hit reconstruction (`FitType:5`) does not read ADC waveforms, but another
configuration may. Preserve the full straw digi payload, original IDs and timing; do not silently replace an
unavailable calorimeter or timing product with zeros. PrescaleEvent in current Offline uses event-number modulo
and the spill marker, not an independently sampled random efficiency. Verify the production release's implementation.

## Replay transformations and limits

- Remove the original `DigitizePath` modules from trigger paths; never redigitize existing digis or replace EWM.
- Drop truth-triggerable paths and do not introduce their gate. All **stored** events are replayed.
- Remove `SelectRecoMC` truth decoration from paths; retain original parameters for decision-making modules.
- Replace fragment unpacking `TrigMakeSD` with `NullProducer`; retain embedded digi-input hit-maker settings.
- Drop old reconstructed collections to avoid unqualified-input ambiguity; retain both TriggerResults products.
- Use a distinct process name and retain the original services, conditions, geometry and filter thresholds.
- Print both result products and save them in `decisions.art` for diagnosis. No output SelectEvents filter is added.

**Validation is not complete merely because preparation or expansion succeeds.** Check release compatibility,
all product tags and auxiliary files, absence of remade digis/timing, finished event count and job exit status.
The current generator is deliberately limited to literal expanded FHiCL; unresolved references and duplicate
keys are errors. It is not a replacement for `fhicl-dump`.

The comparator checks every manifest path's bit ID and accepted boolean for every run/subrun/event; missing
products/paths, duplicate events/paths, unexpected replay paths and wrong counts fail closed. It does **not**
compare full art path states or last-module indices; rejected/not-run/exception are not distinguished by the
stock printer. The separate retained-metadata reader below compares those fields without changing this boolean comparator. It understands the stock fixed-width event header, rejects malformed decision-shaped records,
and requires a successful art completion marker. Use `--input-event-list` with `file_info_dumper --event-list`
output to check identities/order against the input independently, not just the two processes in the same job.
Removed non-decision modules change last-module indices, so this is not serialization identity.
Retained art products allow further diagnosis. Physics efficiency needs paired per-shard differences and an
unbiased denominator; production outputs retained by trigger OR truth selection do not supply that by themselves.
`LumiStreamFilter` is different from PrescaleEvent: it counts processed events, flushes every `eventFreq` events
and can pass the first event of each subrun. Selected digis omit events that advanced its original counter.
The manifest marks these stateful paths, but the comparator does not excuse their mismatches: they still make
`equal:false` and exit 1. `HelixFilter` with `prescaleUsingD0Phi:true` is also stateful: its phi-dependent
prescale uses the filter invocation count, which advances only on events that reach that filter, not the art
event number. The manifest records both mechanisms. Full closure needs the original event/filter-call history
or preserved counter state; independent event-ID modulo is not an equivalent replacement.
Timing comparisons need separate quiet runs without PrintModule per-event I/O, and matching production timing
logs. No validity, throughput improvement, or fast-trigger emulation is claimed by these preparation tools.

## Stock timing and paired summaries

```bash
python3 trigger/timing.py "$PRODUCTION_LOG" --inventory "$PRIVATE_OUT/prepared/inventory.json" \
  --output "$PRIVATE_OUT/stock-timing.json" --top 10
python3 trigger/paired.py "$PRIVATE_OUT/shard-a.json" "$PRIVATE_OUT/shard-b.json" \
  --output "$PRIVATE_OUT/paired-summary.json"
```

Production logs may be on a different storage class from art outputs; use catalog-approved location tools,
not guessed paths. Check the actual job stage: a digi-named log can describe offline reconstruction.
`timing.py` requires a complete successful art job, TimeTracker table and matching event totals. It excludes
original `DigitizePath` modules, MC truth decoration, source and output from trigger work. Shared labels' costs
are `sum(average * executed events)` across disjoint first-execution paths. Path totals are attribution, **not**
independent path latency, because modules shared by several paths execute once. Input dts events processed,
stored output events and generated-particle counts are different denominators. No universal hardware rate or
uncertainty can be inferred from two production jobs.

`paired.py` requires at least two independently selected, nonoverlapping shards and matching path bits and
input-verified decisions. It computes equal-shard paired efficiency differences with their standard errors,
not independent-binomial errors. The caller establishes shard independence and the development/holdout split.
[`HANDOFF.md`](HANDOFF.md) specifies the combined art inputs and unchanged selection needed for stock,
tracker-only, calorimeter-only and combined fast-digi arms.

## Validation status

The complete-history stock CE control now reproduces all **52 physics paths over 28,044 events**:
**1,458,288 event-path comparisons**, with zero accepted-bit, state, `wasrun`, or mapped-endpoint differences.
All retained original first-pass words and independent ROOT event identities/source traversal order are preserved.
The luminosity path passes ten events and the phi-prescaled helix path five, with exact event-key decisions.
All 1,458,288 raw indices/packed words still differ after deliberate path shortening: this is **not full-art
serialization identity**, cosmic-menu validation, or fast-digi validation.

A frozen independent CE production job subsequently passes all 52 paths on **28,148 events**
(**1,463,696 comparisons**) with the same zero accepted/state/`wasrun`/mapped-endpoint differences. Its complete
21,140-event original selected overlap matches all six serialized products and 56 original packed statuses;
the original selector reproduces exactly that selected set. Original words, ROOT IDs/order and both embedded
schedule bindings pass. All replay raw indices remain different. No candidate tuning used held-out outcomes.
Together the two disjoint jobs cover **56,192 preselection events and 2,921,984 event-path comparisons**.
This establishes independent CE stock-control confirmation, not generated-event efficiency or fast/cosmic closure.

The prerequisites were checked over the full source history: 28,044 regenerated events in ten original-parent
boundary files, and all 21,078 original selected events matched by independently decoded identities. On that
complete original overlap, all six serialized digi/waveform/timing products and all 56 original packed status
words match exactly. Parameter-set IDs agree; the original and replay schedule IDs also independently resolve
to their exact ordered bit:path lists in the saved replay's embedded SQLite registry.
A postprocessing-only identity-representation correction reused completed art; no reconstruction rerun was needed.

The earlier selected-stream control below remains useful evidence of why full history is required.
A stock CE digi replay has processed all 21,078 stored events with identities/order checked against the original
event list and the highest-cycle art `Events` count. Across 52 paths there are four accepted-bit differences:
three in luminosity buffering and one in the phi-dependent helix prescale. The other 50 paths match exactly;
the original production output selection has zero differences. A prior 100-event smoke matched all 52 paths.
Both differing modules use history-dependent counters; their original histories cannot generally be recovered
from selected digis. This is **not bit-for-bit all-menu closure**, full-state identity, unbiased efficiency or a
fast-digi validation. Original conditions/geometry and released runtime were retained; no trigger default changes.

## Era-specific menus and quiet replay

```bash
python3 trigger/menu_diff.py "$CE_INVENTORY" "$COSMIC_INVENTORY" --output "$PRIVATE_OUT/menu-diff.json"
python3 trigger/replay.py "$PRIVATE_OUT/input.config.txt" --quiet --output "$PRIVATE_OUT/quiet"
```

Identical path names/bits/prescales do not establish identical trigger behavior: release-specific hit conversion,
fit settings, conditions, geometry and buffering schemas can differ. `menu_diff.py` compares module settings once
per shared label and identifies all affected paths; full differences belong in private output. Exit 0 means the
comparison ran, not that the menus are equal.

For timing use `TRIGGER_MODE=timing` with the `--quiet` preparation and `replay.sbatch`; its parser uses
`replay_inventory.json` (not the original fragment-unpacker module type). The quiet config keeps the same trigger
modules and order and saves both decision products, but removes decision-print and per-event ArtReport traffic.
No setup/compile/cache time is silently subtracted; report cold job wall and per-event module attribution separately.

Finished-event verification is independent of printed progress counters:

```bash
python3 trigger/event_count.py "$PRIVATE_OUT/decisions.art" --expected-events 100 \
  --output "$PRIVATE_OUT/art-entry-count.json"
```

It reads only the highest-cycle `Events` TTree entry count with uproot; it is not an art physics-product reader.
The saved stock-control smoke has 100 finished `Events` entries as well as 100 matching input event keys.

## Retained path-state metadata

```bash
python3 trigger/art_schema.py "$PRIVATE_OUT/decisions.art" --output "$PRIVATE_OUT/art-schema.json"
python3 trigger/path_state.py "$PRIVATE_OUT/decisions.art" \
  --config "$PRIVATE_OUT/input.config.txt" --prepared "$PRIVATE_OUT/prepared" \
  --log "$PRIVATE_OUT/replay.log" --input-event-list "$PRIVATE_OUT/input-event-list.txt" \
  --expected-events 21078 --output "$PRIVATE_OUT/path-states.json"
```

These tools inspect framework trigger/event metadata only, not detector physics products. `art_schema.py` also
inventories matching luminosity/header/filter-fraction branches in `Events`, `SubRuns` and `Runs` without reading
their payload. Its optional `--probe-status` records ordinary uproot's support or failure on one status entry.

`path_state.py` uses a file-local custom uproot model for the observed memberwise `vector<art::HLTPathStatus>`
layout; it does not patch the installed library. It requires the exact recorded member schemas/versions,
byte-count framing, product presence, constant valid parameter-set IDs, event count and path counts. It decodes
the low two bits as Ready/Pass/Fail/Exception and the remaining bits as the stored index. Acceptance follows
canvas semantics: **Ready also reports accepted**, so accepted-bit agreement alone cannot certify state agreement.
Every compared manifest-path accepted bit is crosschecked against the successful event-key-validated PrintModule
log by entry, and the source configuration hash must match the manifest. A separate schema/framing-checked
`event_keys.py` decoder reads the observed fixed EventAuxiliary layout directly from serialized baskets; all
ROOT IDs/order must independently match the input event list. Unsupported layouts and duplicate IDs fail closed.
The retained parameter-set IDs are resolved against `RootFileDB`'s embedded SQLite `ParameterSets` table,
using an in-memory, query-only copy. Each ID must identify exactly the ordered bit:path list in its supplied
configuration. This binds the trigger schedule, not recursively every module/service configuration.

The full 21,078-event stock CE control has four state differences, zero `wasrun` differences and four mapped
endpoint differences, on the same two counter-dependent paths as the boolean comparison. No Ready or Exception
states occurred. **All 1,096,056 raw indices/packed words differ** after removing digitization and truth decoration.
The mapped endpoint diagnostic uses art 3.15's executed-worker count for successful/rejected paths, maps rejects
to module labels and passes to end-of-path; it is not raw serialization identity. Exception endpoints are not
mapped. The CLI exits 1 unless raw packed words are equal; structural differences are never silently waived.
No art rerun, build, fast-digi validation or independent confirmation is implied by this metadata check.

## Same-event fast-input admission

[`HYBRID_READINESS.md`](HYBRID_READINESS.md) records the concrete owner handoff needed for the first tracker-only
arm with stock calorimeter. `hybrid_input.py` compares corresponding original-parent-boundary art files using
independent event IDs and exact original stock-product/TriggerResults bytes, and requires present correctly typed
injected digi collections under explicit distinct tags. It includes zero-response events in the population but
does not decode injected digi physics, certify ADC alignment, establish source completeness or waive era matching.
A keyed table importer operating inside each original event can be validated; concatenating separate art files
with matching IDs does not join them. The bounded historical-era unfiltered pilot remains a possible adapter
control, not proof of the production-era handoff.

```bash
python3 trigger/hybrid_input.py --reference "$STOCK" --candidate "$COMBINED" \
  --config "$EMBEDDED_CONFIG" --expected-events "$COMPLETE_EVENT_COUNT" \
  --straw-digi-tag FastStraw::Injected --output "$PRIVATE_OUT/handoff.json"
```

### Released-runtime adapter and primary-only diagnostic

`adapter_probe.py` / `adapter_probe.sbatch` round-trip original StrawDigis and aligned ADCs through an existing
`StrawDigisFromTable` plugin inside the original events. The C++ dumper's `compareTo` checks every field/sample;
`hybrid_input.py` separately checks retained stock bytes, decisions and identities. A successful adapter does not
validate fast physics. `plugin_runtime.py` makes a task-private runner and frozen hash-checked plugin bundle while
leaving the released Offline runtime unchanged; it never builds or changes a backing area. Runtime compatibility
is an observed gate, not inferred from a shared environment label.

`step_handoff.py` prepares a read-only truth export or a distinct-label primary-step injection using the original
tracker digitizer and explicitly pinned original EWM/PBTMC. It never uses extracted filtering or shadows original
calorimeter/step labels. `step_dump.sbatch` captures the actual released geometry sources for the CE transport;
`ce_transport.sbatch` calls a frozen owner transport implementation. `step_table_audit.py` binds all source births,
including no-handoff/zero-step events, to integer identities and the single-primary key contract. It checks finite
step payloads but does not establish fast-physics closure or complete material/secondary/return coverage.

`step_inject.sbatch` verifies the executed FCL/table/source/configuration binding before stock digitization.
`hybrid_replay.py` / `hybrid_replay.sbatch` measure every scheduled path, retaining both migration directions,
full statuses, reference words and schedule bindings. Fast/stock equality is not an acceptance gate: migrations
are the measured result. The initial one-parent diagnostic explicitly does not certify complete production
history or supply independent-shard errors. Full production claims still require all original parents in one
trigger process and the matching stock control; a prefix is never promoted to that result.

## Full-preselection recovery prerequisites

```bash
python3 trigger/source_audit.py "$PRIVATE_OUT/input.config.txt" --output "$PRIVATE_OUT/source-audit.json"
# Optional after the standard catalog ops setup; uses only metacat get_file calls.
python3 trigger/source_audit.py "$PRIVATE_OUT/input.config.txt" --catalog \
  --output "$PRIVATE_OUT/source-audit-catalog.json"
```

This private inventory records the original parent URLs, digitizer labels/types, complete configured module set,
SeedService and random-engine settings. Configured URL order is not certified runtime order. It never launches
simulation, emits a regeneration FCL, transfers data or modifies the catalog. Catalog sizes/counts do not certify
availability or finished entries. Module construction/engine registration matters for `autoIncrement` seeds;
retaining only a subset of digitizers is not a reproducible shortcut. Complete source coverage and original
random state, followed by digi/timing payload equality on stored events, are prerequisites before a regenerated
full-population control can establish production closure. Existing random-state snapshot products/files may
help only when authentically recorded; a seed alone does not recover omitted stochastic draws.

`regenerate.py` prepares a distinct full-parent control after verifying the matching original production log's
opened-file order, base seed and successful event summary. It unwraps only Digitize (never upstream Primary),
preserves all physics/services exactly, reconciles the framework-generated schedule metadata, substitutes staged
identical-byte parent paths in order, and removes only the existing Output selection/replaces its filename.
It rejects configured Geant4 modules and non-single-thread/schedule controls. `regenerate.sbatch` verifies every
parent's size, catalog Adler-32 and streaming SHA256 receipt before the existing Offline runtime runs. Supply an
explicit bounded event count, immutable code/preparation and a new output directory. Do not use a pilot's RNG
end state or extrapolate its runtime to the complete source.

The exact parent staging helper streams a catalogued read-only archive and verifies its receipt at destination.
It is for separate unsynced scratch storage only, not an alternative deployment route into mirrored worktrees.
When production source names are worker-local, `parent_locations.py` resolves exact DIDs with read-only
`mdh print-url`, records the storage class and preserves the original configured history separately. A derived
storage URL is not proof of actual availability: the wrong tape/disk class can yield nonexistent paths.
A failed stream has no final source SHA receipt and must not pass the ordinary staging gate.
`catalog_bytes.py` can check a cached prefix against catalog size/Adler-32 and measure destination SHA, but marks
source SHA receipt verification false. `regenerate_prefix.sbatch` is a separately labelled bounded startup probe
using such a cached first parent; it does not establish full-source, production-overlap or all-menu closure.
Parent output census uses actual independently decoded event IDs, not a nominal filename subrun: resampled
cosmic parents can contain multiple subruns.

`serialized.py` compares the **raw per-event serialized bytes** of the six value-only digi/waveform/timing
products on an explicitly sized event-key intersection, with product presence and type checks. This includes
persisted flags, peak position and floating-point timing bits that stock PrintModule omits or rounds. It does not
deserialize detector physics, normalize fields, reorder collection elements or ignore framing; byte inequality
can reflect serialization differences rather than physics. `regenerated_compare.py` adds the complete original
TriggerResults packed words on the same event intersection and requires equal schedule parameter-set IDs. This
is a regeneration diagnostic, not yet second-pass trigger closure or fast-digi validation. Complete-source
entry/ID checks and authentic state coverage are still required beyond any selected-event overlap.

For complete-history replay, use `regenerate.py --split-input-files`: the original Output module switches at
input-file boundaries with a numbered filename. RootInput's `noEventSort` still sorts run/subrun groups; it does
**not** preserve original cross-subrun history in a combined file. The census verifies split outputs in original
parent order against independently decoded parent event IDs. A pilot-prefix census does not assert complete
history. After complete production-overlap validation, `full_replay.py` prepares the second pass and
`full_replay.sbatch` checks count/configuration/ordered input checksums before execution. `full_validate.py`
requires retained first-pass words, event identities and parent ordering to remain unchanged. Its accepted/state/
mapped-endpoint result is separate from raw packed identity, which remains false after path shortening.

The initial complete-parent digitization pilot finished 100 events. On its 85-event intersection with original
selected production, all six serialized digi/waveform/timing products and all 56 original packed path words
match exactly. That is prefix reproducibility only, not full-history, fast-digi or independent closure.

## Saved migration endpoint diagnostic

`migration_endpoints.py` reads completed decision art with its bound config, replay FCL and comparison receipt.
It groups lost/gained passes and changed both-fail endpoints for all 52 paths, independently cross-checking the
loss/gain counts. This locates the last executed filter in each arm; it does not establish where a discrepancy
originated or remove primary-only, electronics-stream or stock-calorimeter-history limitations. It reruns neither
art nor transport and makes no threshold adjustment or independent-shard uncertainty claim.

## Saved parent-partition migration diagnostic

`migration_partitions.py` reads a completed all-52-path comparison and its bound saved decision art. It decodes
candidate input IDs, preserves their supplied parent order (numeric traversal only within each file), checks the
saved decision order and both registries, and reports every path's migrations/states/endpoints per contiguous
parent boundary. Every additive count is checked against the aggregate and existing per-path comparison.
Partitions are slices of one continuously evolving filter history, not new trigger runs or independent replicas.
Common-seed parent resets do not authorize independent error bars or a shard SEM. The generic bounded
`migration_partitions.sbatch` wrapper reads artifacts only; it never reruns art or transport.

## JAX BO-selection prototype

`jax_kalseed.py` compiles the two supported TPR final-track selections from the exact embedded configuration
and applies them to prederived measured track quantities. See [JAX_SELECTION.md](JAX_SELECTION.md) for the
release-source binding, required feature export, per-path measured-validation gates and cost limits.

```bash
JAX_ENABLE_X64=1 python3 trigger/jax_kalseed.py --config "$EMBEDDED_CONFIG" \
  --path tpr_TrkDe_80m70p_D0200 --output "$PRIVATE_OUT/tpr-selection-contract.json"
```

Enable JAX x64 explicitly, call `validate_features` outside JIT, then evaluate eager or with a static contract
closure. The float comparison helper retains IEEE float64 subnormal/NaN comparison semantics on flush-to-zero
backends. This does not establish the production C++ process's floating-point mode; measured parity remains a
gate. Synthetic boundary/eager/JIT tests are **not measured hybrid validation**. The separately completed
feature export supplies actual fast-replay tracks for all 28,044 events: both final selections have zero
eager/JIT measured track/count/filter/composed-bit differences, with actual fast reach retained. Decision
art alone lacks these features; original stock KalSeeds cannot substitute for them. Strong parity/provenance
flags remain false. These filters cost about 0.15% of quiet trigger work: this is a fixed-reconstruction BO
cut-scan component, not dominant reconstruction emulation or an end-to-end speedup.

## Complete-history tracker-fast campaign

`injection_campaign.py` maps each original numeric parent ordinal to its isolated transport, preparation and
injection directory. Parent 1's completed artifact is reused; `injection_array.sbatch` prepares only parents 2–10
and invokes the existing stock-digitizer wrapper once per parent. Electronics restart from the common explicit
base seed at each parent: these correlated streams are neither independent replicas nor original production draws.
Every candidate has its own `unfiltered_1.art`; its basename is not the original parent ordinal.

`hybrid_history.py` requires the full original processed count, passing original-selector event-set equality,
unchanged original-reference bytes and a freshly decoded complete parent census in original order. A passing
selected overlap or a prefix alone cannot authorize the complete-history flag. `hybrid_campaign.py` additionally
binds completed injection outputs, execution/table/seed receipts, frozen plugin identities and expanded production
services, then re-audits all six retained stock products and original statuses across every paired boundary.
FHiCL numeric spelling is compared with exact decimal values, not a floating tolerance.

`hybrid_campaign.sbatch` invokes the stock trigger exactly once over the ordered combined files, with all 52 paths
and an explicitly checked total count. Prepare, runtime preflight and postflight recheck evidence bytes. No global
subrun sorting, per-parent trigger restart, copied stock prescale state or forced fast/stock equality is allowed.
This extends stored-history diagnostic coverage only; primary-only steps, absent secondaries/returns, stock-calorimeter
history and separate electronics randomness still limit the physics interpretation. Synthetic wrapper tests are not
an actual full-history result. Failed saved-artifact bookkeeping is repaired without rerunning successful art.

## Separately declared measured-selection experiment

`feature_replay.py` admits a completed full-history hybrid and inserts one measured
`KalSeedSelectionFeatures` analyzer into the existing unconditional endpath, without changing the 52
physics paths, source history or output decisions. The user-built satellite and exact precision/schema
are documented in [`art/README.md`](art/README.md).

`feature_runtime.py` inspects the explicit released build, source/build/library hashes, backing/profile,
ELF dependencies and relocations, then freezes one plugin and a private adapted runtime launcher.
The transferred bundle is admitted with independently pinned library and manifest hashes; worker-side
`ldd -r` and source-set/library/runner receipts must pass before art. This is not reproducible-build or
cryptographic execution attestation. `feature_replay.sbatch` is a separate experiment, not a retry of
completed reference art or a timing benchmark; it refuses existing outputs and runs the full history once.

`feature_table.py` uses vectorized awkward/numpy over uproot TTrees, checks exact persistent dtypes/schema,
highest cycles, event/source order, all empty events, measured track and accepted-key identities. It does
not guess raw status bit definitions or infer reach from product absence. `feature_validate.py` additionally
requires exact executed FCL and all-52 both-process packed words against the saved baseline, authenticates
the shared fit/filter input and producer-only suffix, then compares eager/JIT selected tracks/counts and
composed bits using the actual fast reach. The pure-array comparator and source-only preparation retain
false provenance/readiness flags; zero mock mismatches cannot certify measured parity. Dominant upstream
pattern recognition/fits, independent confirmation and full emulator performance remain separate work.

For a second-layer check, supply `feature_validate.py --status-source-directory` with the exact released
source copies listed in `feature_status.SOURCE_SHA256`. The optional gate rehashes those bytes, compares
all exported raw statuses to SeedOK/supported-fit predicates (including unreached events), and requires
ascending accepted keys for the single shared collection. It reports source-bound consistency separately;
structural-loader metadata and measured-parity flags remain unpromoted. Saved successful art/table outputs
can be rechecked without simulation or trigger reruns. This does not independently authenticate loaded
plugins, ProductID provenance, derived kinematics or C++ numerical-boundary behavior.

## Saved common-filter stage flow

`stage_flow.py` reads only completed decision art and its exact configuration/FCL/manifest/comparison bytes.
It rechecks highest-cycle counts, saved event identities and both schedule registries, retaining all 52 paths;
relocated evidence copies are admitted by ordered exact size/SHA rather than original site-path spelling.
No historical input recensus or reconstruction runs. For filters configured in both arms it counts actual reach,
pass/fail return (including the rejecting endpoint), both/one-arm reach and paired returns among both-reached
events. Producers, including shared fits, are not classified as filters. Final path states and migrations must
reconcile with the completed comparison. Unconditional fractions use every saved event; reached and both-reached
fractions have explicit selected denominators and null when empty. These descriptive stage counts are not
isolated causal filter effects, physics closure or independent-shard uncertainties; evolving counters remain
those of the saved full-history run.

`feature_saved.py` provides a bounded local check of copied completed feature outputs. The original full
postflight has also completed successfully; this saved-output check remains a separate, narrower admission.
It binds exact manifest/config/FCL/baseline-art/comparison bytes, independently decodes baseline ROOT event order, requires both original/replay packed matrices and registries unchanged, rechecks actual
feature schema/status sources, then performs eager/JIT per-track/filter/composed-bit comparisons. It does not
reopen original parents or rederive preparation. Successful saved comparison is measured-array evidence,
not runtime, executed-configuration, source-history or complete measured-parity admission. All such flags
remain false; independent confirmation and numerical-boundary evidence are separate. The CLI saves a
mismatch report before exit 1 and refuses overwrite. No successful art rerun is needed.

## Fixed-reconstruction selection microbenchmark

`selection_benchmark.py` accepts the same saved-art/table/config/baseline/source-directory arguments as
`feature_saved.py`, plus `--repeats` (default 30), `--warmup` (default 3), and a new `--output`. It first
requires a nonvacuous measured-array comparison, then times both unchanged BO selections on every padded
input event with actual fast reach. It checks timed output digests against measured targets. Every output
leaf is synchronized; device placement, lowering, compile and first execution are separate from warm repeats.
Recorded p95 is a batch percentile, not a per-event tail; amortized values divide by all input events. Source
and artifact hashes are rechecked and mismatch reports are preserved before exit 1. It is CPU-only and does
not claim a core-pinned rate, C++ speedup, full-trigger/reconstruction throughput, GPU performance or SU cost.
See [the measured result and scope](JAX_SELECTION.md#saved-measured-cpu-batch-benchmark).

## Saved original-step population diagnostic

`stock_step_census.py` binds original SGS tables by exact saved receipts, reuses the completed all-52
stage-flow admission and joins `(event, SimParticle key)` exactly. It retains original generated-event zero
coverage, checks SGS counts/payload/StrawId and particle associations, and reports both MC-primary and
root-primary definitions rather than guessing they coincide. Original species/other-particle step populations
are grouped by saved stock/fast migration and fixed count bins. These are truth-conditioned **diagnostics**,
never response/correction models or causal secondary-loss estimates. Zero exported rows do not certify
source-collection presence; the dumper does not persist that distinction. Repeated grouped SGS rows do not
count physical passages. See [the response-control design and findings](RESPONSE_CONTROL.md), which
proposes no threshold change or automatic launch. No transport, digitization or trigger rerun is performed.

### Full-stock imported-SGS response control

The user-authorized [separate response control](RESPONSE_CONTROL.md) reuses completed S/F references.
`response_table.py` admits all original species and source rows without weakening the primary-only gate;
`response_presence.py` independently checks persisted SGS presence per event. `response_control.py` binds
original table receipts, completed F response machinery and the saved all-52 decision history before fresh
preparation. `response_roundtrip.py` tests exact re-exported SGS numerics and encounter order. Imported
SimParticle identity/state remain non-lossless: this is not an electronics-only or pure-secondary control.

At the 2026-11-05 06:10 EST pause, only the first C parent (ordinal 1 of ten) has completed: exit 0,
about 10 min 32 s, 2,811 events, 138,045 SGS rows and 11 zero-SGS events. Ordered SGS numerics and
generator data round-trip exactly; all six original products, 56 status words and retained original SGS
serialized bytes are preserved. This is not physics closure or electronics-only isolation. The remaining
nine parents and complete-history C trigger are unlaunched; no R7 jobs are running. Resume only through
[the explicit handoff gates](PAUSE_RESUME.md).
