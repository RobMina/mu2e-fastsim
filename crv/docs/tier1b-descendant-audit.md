# Descendant support, placement trace and retained-step DQM pilot

## Recommendation

Keep context/containment/point placement opt-in. Prefer the strict association-certified library over its
misassociated predecessor as the identity-correct input. Independent 10M-event additive shower-energy means now
agree with G4 at the percent level; that does not certify nonlinear response or justify switching production assets. No rate correction or response tuning is introduced. The only sampler change is an optional
`trace` dictionary; tests show identical output/RNG state when enabled. Existing library files remain immutable.

## Truth-diagnostic partition

`python3 -m valhist.descendant_audit` partitions muon-descendant steps by full event/ancestor identity:
`own_bar`, `own_other_bar`, `no_own_crossing_other_muon`, `no_event_crossing`. It rejects duplicate/missing
identities, broken/cyclic step ancestry, invalid energy and geometry indices. No event loops or packed-key joins.
All sectors, including taggers, are retained. This is an ancestry diagnostic, not a measured-variable response model.

For the retained strict CCP replay on ten held-out shards (500,000 generated events), other-bar energy ratios
conditional on the ancestor having a CRV crossing are EX **0.936 +/- 0.058**, T1 **0.718 +/- 0.076**,
T2 **0.721 +/- 0.104**, M **0.448 +/- 0.081**. Additional G4 energy with no primary-muon crossing in the event
is EX 832.95 +/- 166.36, T1 133.84 +/- 74.19, T2 207.78 +/- 85.60, M 27.46 +/- 14.25 MeV; fast is structurally
zero there. Errors use paired event Poisson-bootstrap moments with a fixed library, excluding donor/denominator
uncertainty. A structural zero's zero ratio error is not a physical upper limit.

All 20 strict production donors have **66,141** descendant steps in no-primary-crossing events, exactly matching
`desc_steps_unattached` in their library manifest. Event/bar grouping collapses only one crossing; 108 descendant
steps lack their own ancestor's crossing but have another muon's crossing. Thus the known multi-muon limitation is
not the leading explanation by these counts. Missing no-crossing support is real but cannot explain the remaining
conditional T1/T2/M deficit.

A separate check on reference shards 1-10 gives conditional other-bar energy ratios EX **0.858 +/- 0.085**,
T1 **0.632 +/- 0.076**, T2 **0.666 +/- 0.092**, M **0.616 +/- 0.125**. This uses retained pre-repair CCP files
because the repair left all descendant library trees unchanged. These are additional events for the descendant
check only: they were used as the shower reference-training/denominator population, so they are **not independent
held-out shower validation**. No fresh larger independent G4 sample is claimed.

## Placement migration, not dropped output

`python3 -m valhist.descendant_placement` replays original shard seeds, traces sampled donor labels through CCP,
and compares the entire emitted descendant physics payload exactly with retained output. All ten shard checks
pass; zero sampled rows are dropped. Donor other-bar steps may land on another bar crossed by the receiving muon:
within EX 3,858 steps / 1,366.59 MeV, T1 1,289 / 460.03 MeV, T2 770 / 250.62 MeV, M 107 / 64.73 MeV. Full
cross-sector and layer migrations are recorded. The donor `samebar` label means the attachment anchor, not an
arbitrary nearby primary bar; interpret transitions accordingly. No fitted relocation is justified by these counts.

## Current-code DQM pilot

The retained strict CCP shard 11 was replayed through current Python response/reco with opposite-side attachment,
and through unchanged stock response/reco in the existing frozen area. Preparation verifies the old successful
pilot certificate and exact area binary/source/FHiCL hashes. Completion requires both art exits zero, 50,000 unique
highest-cycle event rows with the original generated-identity hash, and unchanged frozen inputs/source/assets.
No C++ build or step-dumper invocation was needed. A shell setup failure before output was corrected by clearing
an inherited `MU2E` when shell functions were absent; process/output state was checked before retrying.

| Observable | Strict CCP + stock response (A1) | Strict CCP + current fast response (E2E) |
|---|---:|---:|
| Events | 50,000 | 50,000 |
| Digis | 56,005 | 56,101 |
| Pulses | 70,567 | 70,637 |
| Clusters | 2,858 | 2,864 |

E2E/A1 cluster difference is **+0.210 +/- 0.232%** (paired fixed-event delta-method SE). Frozen A0g has 2,918
clusters, so E2E/A0g is **-1.851%**, versus old E2E -1.405%; no paired A0g error is claimed. The comparison uses
the presentation's unchanged floor rule, first-bin cuts, clustered time errors and per-bin position variances.
Own cluster tables supply A1/E2E variances; A0g and the reseed floor borrow A0 variance as explicitly labelled.

| Shard-11 comparison | Old placement/library, attachment on | Strict CCP, attachment on |
|---|---:|---:|
| E2E versus A0g flags | 17/133 | 15/133 |
| E2E versus its matching A1 flags | 0/133 | 0/133 |

Cluster-Y score improves 1.226 -> 0.853 on this shard; neither is flagged. Occupancy/ADC/BarId/SiPM flags remain.
The removed flags are two NDigis2 histograms, not proof of closure. These are one-shard numbers, not the prior
20-shard comparison. Library and placement change together: their separate causal effects are not identified.
Original histograms are read directly and flow counts retained. E2E still lacks injected step/DigiMC histograms.

## Denominator sizing proposal, no submission

`python3 -m valhist.denominator_sizing` reads the certified denominator audit. For 44 nominal shower-populated
cells, 1,000 whole-file draws and a 1% union-bound budget for any zero-support draw, **16 distinct supporting files
per cell** suffice since `(1-m/n)^n <= exp(-m)`. This controls raw empirical zero support only, not integer
normalization rounding, saturation or unseen tails. A new bank with 50,000 events per independent file, unchanged Poisson event rates, and a 5% modeled failure
budget to acquire support would need **239 files / 11.95 million events** at observed point rates. The singleton
photon cell makes that optimistic: simultaneous 95% Bonferroni Poisson lower rates yield **203,147 files /
10.15735 billion events**. These deliberately contrasting scenarios show that a reliable fixed-size guarantee is
not supported by the ten-file census. Confidence in rates and support-acquisition probability are distinct.

Prefer an approved staged independent generator census, preserving the complete population and recording all
attempts; rerun nominal/evaluated-tail support and full ensemble probability gates on actual data. Do not infer
runtime, submit the point estimate as a certified campaign, redraw failed bootstrap replicas, clip probabilities,
merge bins or normalize on accepted events. Named-run libraries await compatible conditions and strict donors.

## Continuation: exact expectation, placement replicas and factorial control

The donor-feature cache reconstructs all 2,655,560 ordered crossing bins/contexts/directions from strict donors.
Zero-descendant crossings remain in the empirical distribution. `descendant_expectation` computes exact conditional
pre-placement moments for other-bar count/energy by original destination sector/layer. Context/default energy
ratios to held-out G4 are EX 1.0612/1.0613, T1 1.0534/1.0532, T2 0.9148/0.9098, M 1.0597/1.0562. These are
truth-destination diagnostics, not placed predictions. Selection of context alone barely changes these totals.

Sixteen independent explicit-key replicas of the actual CCP placer, over the same ten shards, give post-placement
other-bar energy/G4 means EX **0.9322 +/- 0.0090**, T1 **0.7532 +/- 0.0060**, T2 **0.6815 +/- 0.0175**,
M **0.5269 +/- 0.0209**. These errors are numerical MC errors of the means, not physics errors. Single-draw
sampling SDs divided by G4 are 0.0360/0.0239/0.0700/0.0838. No sampled row is dropped; pre/post covariance is
retained. The large conditional T1/M deficit is a persistent placement/membership effect, not just the original
single draw. T2 additionally has a pre-placement shortfall. Do not correct this by multiplying energy/rates.

In the retained trace, the fractions of sampled donor other-bar energy landing in each sector but reclassified as
own-muon-bar are EX 17.05%, T1 22.57%, T2 16.09%, M 26.96%; cross-sector migrations also contribute. The
pre-placement source label is relative to the donor anchor, whereas the final other-bar observable excludes all
bars crossed by the receiving ancestor. They are deliberately distinct observables, not lost energy.

The completed four-arm shard-11 control isolates library identity and placement factors. Muon-family payloads
are exact across library repair at fixed placement; shower payloads are exact across placement at fixed library.
Only strict/default steps were newly generated; legacy/default and both CCP step arms were reused.

| Library / placement | E2E/A0g flags | E2E/A1 flags | E2E clusters | E2E/A1 relative cluster difference |
|---|---:|---:|---:|---:|
| Legacy / default | 17/133 | 0/133 | 2877 | retained baseline |
| Strict / default | 12/133 | 0/133 | 2865 | +0.035 +/- 0.234% |
| Legacy / CCP | 17/133 | 0/133 | 2858 | -0.591 +/- 0.242% |
| Strict / CCP | 15/133 | 0/133 | 2864 | +0.210 +/- 0.232% |

All new stock/fast arms contain 50k exact generated identities. Placement at fixed strict library changes E2E
clusters by **-0.035 +/- 0.259%** (paired events). Cluster-Y scores versus A0g are respectively 1.226, 1.099,
0.649 and 0.853. Position improves without a monotonic improvement in total flags; no placement promotion follows.
Common seed does not mean aligned random streams when shower multiplicities change. No full-sample causal claim.

## Existing independent larger sample and candidate denominator bank

Reuse existing data before new production. The additional G4 reference shards 21-220 provide **10M generated
identities**, independent of donors and the reference-source/denominator training. Exact additive conditional
shower-energy expectation on this sample is:

| Sector | Fixed strict library / G4 | Paired fixed-library event SE | Candidate 94M-bank sensitivity / G4 |
|---|---:|---:|---:|
| EX | 1.00303 | 0.00572 | 1.00159 |
| T1 | 1.00541 | 0.01509 | 0.99927 |
| T2 | 0.99262 | 0.01034 | 0.99159 |
| M | 1.00458 | 0.01863 | 1.00555 |

The historical 5-7% energy deficit **does not persist** in this twenty-times-larger independent event sample.
It is not evidence for tuning shower energy. No donor/denominator error is included, no equivalence test or
nonlinear response closure claimed. Fifty-two primaries outside the nominal cell inventory occur in events with
zero nonmuon G4 energy. A separate certificate checks complete event ranges 1-50000 in every shard, no overlaps,
valid ancestry, exact accepted/full generator payload pairing and immutable hashes. Legacy-table producer role
and schema-v2 full-event certification remain admission limits. Wall times: 236.54 s evaluation, 139.32 s certificate.

The existing **94M-event generator-only bank** contains support in at least 31 files per inventoried cell, with
photon KE-bin 12 supported by 92 files. At unchanged 10M donor exposure it has zero nominal missing/saturated
cells and **0/1000 invalid whole-file draws** (conditional Wilson 95% upper bound 0.383%). All 94 highest cycles
and row identities were checked; the first pilot exposed that the legacy subrun `primaries` field equals 1, not
N events, so it is recorded but never used as the exposure denominator. The old ten-file bank remains unchanged.
No new generator production is needed just to seek empirical support. Strict full-stream census, compatible
producer/configuration and missing-job selection review, evaluated-tail gates and a versioned library are still
required before propagation. The candidate-bank arm above is a sensitivity diagnostic only.

## Existing-bank strict admission: bounded capture and provenance

Two full files, subruns **2001 and 2003**, have now been captured from their original art inputs with the
user-built merged dumper: **2,000,000 complete event identities / 2,011,493 generator rows**. Schema-v2 full-stream
role, ProductIDs, census closure, configured generator tag, exact event ranges, legacy gen/subrun values and row
order all pass. The merged aliases satisfy `ngen=n_gen`, `nsteps=-1`. Actual loader traces identify the approved
plugin. Measured art times are **60.95 and 58.21 s**, not a projection for the remaining bank. Only these two files
are certified; no whole-bank or downstream authorization follows.

Original art-stored configurations and successful generation logs for both files were compared to original
reference shard 2. The observed generator/counter/time-offset blocks, geometry/global-constants/RNG-service
blocks, runtime build banner and geometry/field/global-constant fingerprints, CRY settings, reference point,
target box and world agree. Each candidate log records all 1M events through the generator path and zero G4
transport visits. The comparator parses the expanded configuration strictly, binds the input filename to the
configured output, and checks the completed module census. This establishes the stated pairwise observations,
not all historical external-file contents or all 94 producer configurations.

### Seed assignment is a separate limitation

Zero-event initialization of the historical stock Musing/configuration gives:

| Probe | Time-offset MixMaxRng | G4Engine | CRY MixMaxRng |
|---|---:|---:|---:|
| crygen subrun 2001 | 20263000 | 20263001 | 20263002 |
| crygen subrun 2003 | 20263002 | 20263003 | 20263004 |
| reference subrun 2 | 20261017 | 20261018 | 20261019 |

Thus the observed initialization reuses seed **20263002** across CRY generation in one file and time offset in
another. Assuming the same offsets across the retained 94-file inventory gives **87** such cross-module pairs.
That count is an inference from the observed assignment rule, not 94 captured historical RNG states. CRY generator
seeds themselves remain distinct under that rule. This is **not** evidence of duplicated showers or biased
species/KE counts: the denominator does not use time. It does prevent an unqualified full-stream independence
claim; use-specific uncertainty needs explicit review. G4Engine has a different engine kind and is not counted
as a same-MixMaxRng collision.

### Original scheduler evidence recovered

The original 100-job submission and retrieved worker archive now account for all outcomes: **94 successful
worker logs** match the candidate file inventory exactly, and scheduler final ads identify all six omitted
subruns as one-start runtime-limit holds followed by user removal. All six ran on **one host**, with
**1092–1754 CPU seconds over 10853–10854 wall seconds** (10.1–16.2% CPU/wall). The original submit file requests
100 jobs with the recorded million-event exposure and seed/subrun rules. This directly supports the host-slowness
explanation; it is stronger than survivor spectrum agreement. Only 65 final scheduler ads remain available,
and the failed workers' progress logs are absent from the fetched archive. Therefore this is not a complete
attempt history or a proof that censoring was independent of generated content. The hash-bound
`valhist.generator_history` report distinguishes those statements and remains nonauthorizing.

### Denominator-only RNG scope

The native `_reach` implementation reads only event identity, PDG and energy; its generated exposure comes from
the complete event census. The uncertainty cell inventory likewise groups by absolute PDG and native KE.
Synthetic tests explicitly change/permutate all generator times and obtain **exactly unchanged cell counts and
rounded reach denominators**, including an empty-generator census event.

Writing file i's species/KE count vector as `C_i=f(U_i)`, the observed seed reuse relates `U_i` to another file's
time-offset stream, not its CRY generator stream. Under the usual assumption that distinct CRY seed streams are
independent, `Cov(C_i,C_j)=0` for distinct files before selection. No extra cross-file count covariance is implied
merely by the time-offset reuse. This is a conditional model statement, not an empirical RNG-independence proof.
Runtime selection may still depend on `U_i`, and joint timing/energy or nonlinear response statistics are outside
this denominator-only argument. **Time-offset reuse alone is therefore not a blocker to a species/KE-only bank**;
the full-stream warning remains. No old denominator or uncertainty result is relabelled as admitted.

The conditional species/KE-only scope and remaining strict full-event/producer captures have been approved.
`valhist.generator_bank_campaign` now prepares a read-only, launch-digest-pinned manifest with explicit host-related
missingness and distinct-CRY-stream assumptions, reuses the two completed captures, and assigns every remaining
partition exactly once. Workers use one sequential queue per host, verify their reservation and actual executing
code closure, pin the approved dumper/FHiCL hashes, and retain a quota reserve. Original stored configurations and
generation logs are compared against the fixed reference for every partition. A failed attempt stops its queue;
no file is silently skipped. Aggregation rechecks 94 complete per-partition censuses without collecting 94M event
keys in memory, verifies exact plan membership and producer evidence, and remains nonauthorizing for propagation.

The complete campaign now certifies **94M events / 94,537,488 generators**, with exact legacy payloads and all 94
producer comparisons passing. The subsequent full-cell audit does not pass all evaluated-tail gates (details below).
Neither a new denominator library nor uncertainty propagation was started. No fresh generated events, G4 transport,
build, grid submission, baseline overwrite or default change occurred. The manifest carries the source/seed
evidence's transitive hashes; a capture receipt is not an unconditional missingness or RNG-independence certificate.

Capture tooling pins the approved dumper and its entire FHiCL include closure before execution; current read-only
verification rechecks both completed captures without repeating art. Zero-event probe certificates store parsed
numeric seed assignments. Old reports retain their original code hashes and are not rewritten.

## Input contracts and validation

Duplicate shard lists, overlapping run/subrun populations, donor/evaluation overlap, reference events outside
transport gen, mismatched report/cache/library/transport bindings and pilot-versus-requested-shard identities now
fail closed. Historical external assets must match their old hashes, not be re-baselined. New replay manifests
hash generated FCL and the shell driver; prior outputs keep their original limitation. A read-only recertification
checks all three completed pilots' original external assets/populations without rerunning art. Guarded v2
expectation/replica/budget and DQM reports reproduce v1 numbers exactly.

## Reproduction

With this checkout and matching core on `PYTHONPATH`, use module `--help` for explicit input/output arguments:

- `valhist.descendant_audit --reference PATTERN --candidate PATTERN --shards 11-20 --out NEW.json`
- `valhist.descendant_audit --donor-manifest LIBRARY.json --out NEW.json`
- `valhist.descendant_placement --library LIBRARY.root --transport PATTERN --retained PATTERN --out NEW.json`
- `valhist.tier1b_replay prepare ...`; `scripts/run_tier1b_replay.sh NEW_OUT FROZEN_AREA`; then
  `valhist.tier1b_dqm_compare --data DATA_ROOT --new NEW_OUT --shard 11 --outdir NEW_COMPARISON`.
- `valhist.denominator_sizing SUPPORT_AUDIT.json --out NEW.json`.

The optional DQM comparator imports the presentation-statistics package read-only and does not edit it. The shell
replay requires `M2FS_CORE` and `CRVFS_DATA`; preparation refuses an existing output and mismatched area hashes.
Continuation full CRV regression suite on integrated main, including `--runslow`: **2,456 passed, 3 skipped**;
remote selected suite: **153 passed**. Shell syntax and whitespace checks pass. T1-y integrated total-position
residual, full-sample nonlinear DQM, admitted denominator propagation and named-configuration libraries remain open.
The independent larger additive shower-energy check is completed with the legacy-table qualifications above.

Additional modules: `descendant_expectation cache|evaluate`, `descendant_replicas`, `descendant_migration_report`,
`tier1b_controls`, `generator_bank_candidate`, `shower_heldout`, `heldout_certificate`, `replay_certificate`.
All accept explicit paths and refuse existing outputs; use `--help` for their contracts. The DQM comparator accepts
repeatable `--control NAME=COMPLETED_PILOT` to compare factorial arms on exactly the same generated population.

Existing-bank admission tooling (all outputs must be new paths):

```bash
M2FS_CORE=CORE bash scripts/run_generator_bank_capture.sh BUILT_DUMPER_AREA \
  --candidate CANDIDATE.json --subrun 2001 --outdir NEW_CAPTURE
python3 -m valhist.generator_capture_verify --certificates CAPTURE_2001/certificate.json \
  CAPTURE_2003/certificate.json --out NEW_VERIFICATION.json
python3 -m valhist.generator_bank_provenance --crygen-config SAVED_CRYGEN.fcl \
  --crygen-log ORIGINAL_CRYGEN.log --g4ref-config SAVED_REFERENCE.fcl \
  --g4ref-log ORIGINAL_REFERENCE.log --output NEW_PROVENANCE.json
M2FS_CORE=CORE bash scripts/run_generator_seed_probe.sh --mode crygen --subrun 2001 --outdir NEW_PROBE
python3 -m valhist.generator_seed_report --candidate CANDIDATE.json \
  --probes PROBE_2001/certificate.json PROBE_2003/certificate.json PROBE_REFERENCE/certificate.json \
  --out NEW_SEED_REPORT.json
```

`genassoc_fullgen.compare_payload` accepts an explicit expected event count for this capture; its historical
50k-event default and original twenty-file campaign are unchanged. After current-main verification, the admission
continuation passed **273 local tests** in 12 isolated modules and **134 remote selected tests** (7.38 s), plus
shell syntax and whitespace checks. This selected result is not a new full-suite claim.

Historical campaign consistency can be reproduced with:

```bash
python3 -m valhist.generator_history --history-file SAVED_CLUSTER_HISTORY.json \
  --archive RETRIEVED_CAMPAIGN_DIRECTORY --candidate CANDIDATE.json --out NEW_HISTORY_REPORT.json
```

The history/scope continuation passed **140 local selected tests** in seven isolated modules and **13 remote tests**
(1.79 s). No physics implementation or seed configuration changed.

Conditional full-bank capture orchestration (never a library publication):

```bash
M2FS_CORE=CORE bash scripts/run_generator_bank_campaign.sh BUILT_DUMPER_AREA prepare \
  --candidate CANDIDATE.json --history HISTORY.json --seed-report SEEDS.json \
  --certificates CAPTURE_2001/certificate.json CAPTURE_2003/certificate.json \
  --area BUILT_DUMPER_AREA --hosts HOST1 HOST2 --outdir NEW_CAMPAIGN
M2FS_CORE=CORE bash scripts/run_generator_bank_campaign.sh BUILT_DUMPER_AREA run \
  --manifest NEW_CAMPAIGN/manifest.json --manifest-sha256 PRINTED_SHA256 --host HOST1
M2FS_CORE=CORE bash scripts/run_generator_bank_campaign.sh BUILT_DUMPER_AREA aggregate \
  --manifest NEW_CAMPAIGN/manifest.json --manifest-sha256 PRINTED_SHA256
```

Hosts must be sorted distinct allowed GPVM names with matching exclusive reservations; run each assigned host
once, not two queues on a host. Do not edit the pinned code or manifest while workers run. The aggregate command
requires every worker's successful completion record. New campaign controls have **32 synthetic tests**;
the remote isolated selection passes **179 tests / eight modules**, including existing strict-capture, provenance,
seed, history and full-generator validators. No new full-suite claim is made.

After successful aggregation, a separate read-only support/tail diagnostic is available:

```bash
python3 -m valhist.generator_bank_support \
  --campaign-report COMPLETED_CAMPAIGN/aggregate/report.json \
  --cache ORIGINAL_uncertainty_v2/production \
  --evaluation-certificate EXISTING_HELDOUT_CERTIFICATE.json --out NEW_SUPPORT.json
```

It checks strict full-stream exposure and original candidate-bin counts, unions **all** bank/shower/evaluation
cells, retains every whole-file bootstrap draw and distinguishes absent-bank from present-but-rounded-zero cells.
For evaluated cells it reports identity/ancestry-based whole-event G4 step/energy bounds in all four sectors;
no-template cells remain a separate limitation. Categories overlap and their per-cell event bounds must not be
summed as disjoint contributions. No physical-zero or downstream-admission claim follows from a zero observed
bound. The old fixed donor cache is reused by its payload/projection/completion hashes; its retired historical
dumper paths are recorded as provenance, not reinterpreted as the currently executing dumper. No cache or library
is rewritten. In-progress campaign receipts are refused before bank reads. Synthetic support tests: **48 passed**;
a separate explicitly enabled read-only retained-cache preflight passes on the GPVM. Full-bank execution is complete;
its measured wall time was 863.57 s, including fresh strict counts and all 200 held-out shards.

### Completed conditional support result

The union of bank/shower/evaluation contains **58 native cells**. Every existing shower template has valid nominal
support, with no saturation; **0/1000** whole-file draws invalidate shower-populated cells (conditional Wilson 95%
upper failure fraction 0.383%). However, the broader evaluated-tail gate **fails**, which is not a contradiction:

- No evaluated primary is absent from the entire 94M bank.
- Photon cell `(22,15)` has 3 bank primaries and neutron cell `(2112,16)` has 4; both denominators round to zero
  at the unchanged 10M donor normalization. Each contains one evaluated primary and no template. Both events
  have zero observed nonmuon G4 activity, not a certified physical zero.
- **930/1000** draws have at least one evaluated zero denominator. Photon bin 15: 45 absent + 779 rounded-zero
  draws; neutron bin 16: 16 absent + 611 rounded-zero draws. The union, not their sum, is relevant.
- **78 evaluated primaries / 78 events** have no template. Their whole-event nonmuon G4 upper bound is **12 steps /
  16.170240 MeV in EX**, zero in T1/T2/M. The nonzero bound occurs among events containing pion cell `(211,16)`;
  it is not attributed to that primary without an identity match. Per-cell bounds may overlap and are not additive.

Capture/observed-producer evidence is complete under the accepted assumptions, but generic all-tail propagation
remains **blocked**. No replacement library was written. The old ten-file bank's 562/1000 failures remain unchanged.
A future observed-template-only diagnostic must explicitly exclude unknown tail response; it cannot silently turn
these failures into a full coverage claim or discard affected bootstrap draws.

After the physics run, this work was brought onto CRV main `c3423f2` (named configurations) unchanged. Local isolated
validation: **357 passed, 1 skipped / 20 modules**. Remote: **311 passed, 1 skipped, 4 failed / 13 modules**; the
four newly merged JAX/fused test branches cannot import JAX in the shared `pyenv ana` environment and pass locally.
No installation or test weakening was performed. The skip is the explicitly enabled retained-cache preflight,
which passed separately. These are selected regression checks, not a new full-suite result or retroactive relabeling
of historical physics-code provenance.
