# CE exposure continuation: audits and timing validation plan

These are preparation/read-only diagnostics, not production promotion. Historical
mixtures, response defaults, selections and fixed-stock outputs remain unchanged.
See `veto-exposure.md` for the opt-in PBI contract and `veto-cat-v2.md` for the
census-bound input format; `veto-cat-primary-policy.md` covers the user-built
staged-Cat policy and bounded runtime validation.

## Source identity and exposure diagnostics

`python3 -m crvfastsim.veto.source_provenance` binds retained subset manifests and
explicit original extraction-list paths, hashes actual selected compact tables,
reports full-tuple step IDs, and records matching retained worker-log lines.
`--prior ... --art-names ...` inspects only the named original art files' ROOT
headers. Unsplit art auxiliary objects require an art event census; there is no
custom object-streamer workaround. `--prior ... --sam-dir ...` summarizes saved
exact-file SAM metadata, parent-set hashes and catalog/table count differences.
No namespace enumeration, deduplication or yield renormalization is performed.
Repeated log downloads are not independent extraction attempts. File-name subruns
are locators, not promises about contained event subruns. Equal compact payloads
and equal parent sets establish overlap, not the intended production weight.

`python3 -m crvfastsim.veto.selection_exposure --selection ... --ntuple-list ...
--tree EventNtuple/ntuple --out NEW.json` joins retained selection rows to explicit
source EventNtuple paths by full `(run,subrun,event)`. Duplicate tuples fail closed;
source-qualified selection IDs are required if duplicates occur. The strict
momentum window defaults to `85 < p < 200`, matching the frozen coverage contract.
All supplied ntuple events remain in the denominator, including those with no
selected track. A companion exposure parquet and input hashes/ROOT UUIDs are
retained for follow-up without rereading physics branches. Only event scalars are
read from source ntuples; the existing selected track times are reused.

MC `nprotons`/`pbtime` correlations and quantile strata are **truth diagnostics**.
They are not measured-variable corrections, generated-CE efficiencies, statistical
independence proofs or factorisation certificates. The tool also checks whether
nominally measured exposure branches are constant or exactly copy MC. A correction
model must use validated measured quantities. Trigger/output selection may already
condition the ntuple denominator. For selected-track veto loss the desired joint
law is `p(PBI, bunch phase, track time | selected track)`, not merely the primary's
size-biased PBI marginal times an independently sampled track-time marginal.

## Full-sample photon-arrival timing plan

The read-only `step_timing_audit` census (`veto-step-timing-audit.md`) binds every
raw replay row and empty frame, reports stock endpoint eligibility versus historical
start-shift folding, and retains all-sector/phase ledgers. It does not generate
photons or establish the optical/response gates below. The bounded
`optical_timing_probe` (`veto-optical-timing-probe.md`) follows the numerically
exceptional frames with new common-stream raw/folded optical draws and four
separate operation arms. It preserves candidate photon identities for downstream
reuse but is not yet subset/chunk-invariant keyed generation or full-sample closure.
The separate diagnostic `keyed_optics` kernel (`veto-keyed-optics.md`) now supplies
full input/event/step/channel/substep/type/ordinal keys and passes synthetic
subset/permutation/partition tests. It does not replace that completed probe or
constitute full-sample closure. The bounded `keyed_optical_smoke`
(`veto-keyed-optical-smoke.md`) binds real inputs/assets and exercises two explicit
response seeds with exact partition/subset checks, a collision-checked downstream
root bridge and saved-only verification. Full retained-sample execution and
response-seed uncertainty remain separate gates. The full retained resource plan
and frozen source-qualified frame registry (`veto-retained-plan.md`) now provide
whole-frame batching without assuming contiguous source rows. The diagnostic
qualified response adapter restores full identities; exact candidate/charge/ADC
subset checks do not certify fitted-pulse partition invariance. That numerical
fit gate, pre-allocation guards and output-byte admission still block full execution.
Separate prerequisites now provide an indexed per-input cache
(`veto-indexed-replay.md`), width-bucketed diagnostic fits and pre-allocation guards
(`veto-diagnostic-fit.md`), and bounded atomic artifacts (`veto-batch-artifacts.md`).
The separate bounded downstream executor (`veto-bounded-response.md`) now integrates
explicit stochastic budgets and validates all-live response on the existing saved
four-frame photons, including exact physical-field partition checks through the
new diagnostic fits. It does not alter the frozen adapter or certify a full
retained run. Real indexed-cache admission and every planned raw-row/frame
selection now pass the complete retained-input audit (`veto-retained-cache-audit.md`),
with measured process RSS rather than payload-size extrapolation. This closes the
tested cache-selection coverage gate, not full streaming physics coverage. A
combined bounded pilot (`veto-streaming-pilot.md`) now passes on nine new frames
in three existing batches, two seed pairs and four arms, with complete source,
all-live and arm/seed ledgers. It preserves previous evidence and stops before
coincidences. Full-registry campaign persistence/output admission and coincidence/
track gates remain required before a full retained-sample run or F claim.
The separate saved-pulse coincidence/native-membership and frozen-grid diagnostic
(`veto-pilot-membership.md`) now completes on retained saved pulses, including
receipt-bound admission, explicit ignored-channel policy, complete frame/arm/seed
ledgers and saved-output verification. Its real coincidence collections and
coverage are all empty: this validates the empty-path integration but cannot
establish nonempty native membership or timing closure. Nonzero synthetic tests
remain separately labelled. No completed pilot is regenerated or full campaign
admitted.

The separate saved-only nonempty control (`veto-nonempty-saved-control.md`) now
exercises real nonzero historical-grid coverage and an event-gate failure using
existing legacy stock and recovered-fast native memberships. It executes no
finder. The old pulses lack waveform/start-TDC/original-relative-time provenance;
the table-import path also retains ignored-channel members. Neither missing
identity nor the different channel policy is silently repaired. This closes only
the selected saved-native/coverage check, not the new adapter's nonempty gate,
full timing closure or stock equivalence under the new pulse policy.

The retained-pulse inventory (`veto-pulse-inventory.md`) identifies another path
without a new response draw: existing three-event stock-ADC refits preserve saved
waveform links, original relative/reco times and compatible channel flags. A local
identity/time inspection has completed. The separately admitted finder-only check
(`veto-refit-membership.md`) now binds the complete registry and calibration and
produces three real coincidences/17 native links from all 87 saved refit pulses.
This exercises the nonempty adapter path; it performs no refitting or response draw.
Saved-output verification (`veto-refit-verification.md`) also passes: exact source
bridge, native link/geometry/side-count consistency and complete frame accounting,
without finder or prior-verifier execution. Floating cluster physics remains a
hash-bound observation, not independently recalculated.
Refits are not numerically identical stock pulses. A subsequent saved-only
stock/refit comparison (`veto-stock-refit-comparison.md`) establishes exact unordered
member sets for these three events: 8/4/5 common members, none unique to either
collection. Legacy v4 has one cluster per event, making parent assignment
unambiguous despite missing ordinals. Native order and missing scalar properties
remain untested; observed average-time/PE differences are retained explicitly.
The separate all-live/extreme inventory remains implemented/local-tested but has
not inspected real remote products. The separate
saved-grid coverage consumer (`veto-refit-coverage.md`) now completes, with 208
passing local and remote targeted tests. All three frames have card coverage one
because each fails the unchanged event gate; window-only diagnostic coverages are
0.0041084864319399445, 0.0037394638080543332 and zero. Both integer-grid parity
checks pass. No selected-frame mean or F is calculated.
It additionally requires the original refit completion-receipt pin: the earlier
summary/report pins alone do not anchor changed cluster files against resealed
receipts. Inherited grid artifact pins are checked against the original refit
summary, without opening selected tracks or rerunning any earlier verifier.
Timing, exposure/library uncertainty and full-campaign gates remain open.

1. Freeze code, conditions, lookup, photon deviation, raw mixture, selected tracks,
   card and coverage grid before execution. Keep the completed fixed-stock sample;
   do not rebuild/rerun its baseline. Evaluate the retained sample as a diagnostic
   of its historical exposure, not as the new CE target.
2. Enumerate every raw step in every retained frame, including empty frames. Use
   event-specific `w=-float32(pbtime)` and the reviewed endpoint-rejection predicate.
   Generate optics from raw endpoints; use stable `(frame,step,end,draw)` keys for
   shared optical draws. Cover all sectors, both ends, every bunch phase, late and
   multiperiod steps, and NaNs according to the pinned stock behavior.
3. Keep separately labelled arms for historical step folding, arrival wrapping
   only, stock step rejection only, and both stock operations. Evaluate the union
   of channels reached by any arm, not only previously successful stock channels.
   Preserve candidate photon identities when an arm rejects them so RNG consumption
   cannot confound the timing contrast.
4. Compare accepted arrival counts/times, wrapped boundary/ULP classifications,
   charges, zero-suppression decisions, full waveform tails, fitted pulses, native
   membership and frozen-grid frame coverage. Use keyed shared downstream random
   draws. Do not use time/distance as truth matching, widen gates, retune yields,
   or silently replace independent stock photons by fast photons.
5. Report paired frame differences, gate flips and response-seed variation within
   original frames. Source-provenance resampling is a separate uncertainty level;
   overlapping parent groups cannot supply independent block errors. Existing
   three-event controls constrain those events only.
6. Only after reviewing those results consider a versioned production timing path.
   New CE exposure requires new matched stock/fast inputs and explicit exposure
   binding throughout consumers; historical outputs cannot validate new frames.

The current `replay_fast` and `replay_check` paths explicitly reject replay-v2
rather than silently discard its exposure contract. A future v2 consumer must
validate raw/rounded PBI and per-species counts, bind the full exposure/library
manifest in response/check reports, and reject pooling across unequal target,
library-weighting or timing contracts. This deliberate gate keeps preparatory
export support distinct from a production-ready validation chain.

## Before a broader sample

The saved metadata inventory (`veto-exposure-metadata.md`) now preserves the full
catalog fields omitted from the compact provenance summary. All twelve Neutrals
records declare `dh.gencount=6700000000`, but count semantics, additivity, physical
reuse weights and deeper independence are not established. The saved exposure
ledger has seven event fields and no source-file or track-time column; the prior
measured-intensity zero observation remains inherited, not a correction model.
The documentary follow-up (`veto-historical-production.md`) now pins the three
registered resampler/Cat/CE job-definition tarballs and their declared release
sources. Flash specifies 40 auxiliary files, sequential reuse, 1,340,000 draws
and a maximum first skip of 1,335,651. Exact worker evidence now resolves the
Flash 13147 discrepancy as a parent-list generation defect: its FCL uses 32644,
while the catalog records 32205 (`veto-flash-parent-metadata.md`). A bounded full
stream of the Cat job-parameter member measures 5,000,000 filename entries; all
twelve retained 5,000-name blocks match saved SAM parent hashes
(`veto-cat-jobdef-audit.md`). This binds declared blocks to catalog sets, not
worker execution or deeper physical ancestry. The declared CE recipe names EventNtuple 6.13.2 via
AnalysisMDC2025/v02_01_00: measured nprotons is unfilled there, while luminosity
proxy fields exist but their usable values/calibration are unverified. The
concrete printJson.sh rule writes the GenEventCountReader total to dh.gencount;
its deployment on the historical workers is not yet bound. Next obtain
file-level execution/assignment records and measured-observable availability—not
another three-frame control or an exposure correction from catalog counts.

Resolve original production parent overlap and exposure weights; verify a complete
all-event census against source counts; check empty-generator and packed-key
collision cases; freeze conditions/efficiency IOV; establish selected-exposure
correlations with measured-variable feasibility; then propose sample sizes and
resource requirements. The user builds the dumper and separately approves any grid
campaign. No automatic broad extraction, timing campaign or stock rerun is launched.

## Flash parent metadata follow-up

Exact worker evidence for subrun 13147 now shows input 32644 in the generated
FCL, while the catalog says 32205. The worker's prodtools v1.3.7 used inconsistent
auxiliary-input selection for FCL generation versus parent-list generation; the
latter reproduces 32205. See `veto-flash-parent-metadata.md`. Deep catalog ancestry
must not be treated as physical ancestry without auditing this distinction.
No catalog or physics product was changed.

## Next critical-path increment

The separately authorized measured-observable check is implemented in
`ce_proxy_audit` (`veto-ce-proxy-audit.md`): exact retained identities, stored
version and allowlisted scalar fields only, not calibration or selection. Its
first NFS read was interrupted while storage-blocked before the first header;
no availability finding follows. The bounded explicit XRootD transport passes
144 targeted tests locally/remotely, but its first real open failed authentication.
After user token refresh and a reviewed structural-parent metadata correction,
the bounded audit succeeds on all eleven pinned files / 91,226 unique events.
The correction proves empty serialized basket collections without decoding them;
328 targeted tests pass locally/remotely and rejected attempts remain preserved.
All files store version 6.13.2. Measured nprotons is zero throughout, but all six
scalar luminosity observables are present without reset sentinels or invalid
values: nProtonTCs spans 0–14 (35,037 zeros), and tracker/calo hit counts and integer
caloEnergy are positive throughout. pbtime is finite, pbterr is uniformly 0.5 ns,
and caphriIsValid is always true. See the saved availability report/contract;
reuse this result, not another availability pass. No input substitution, field
aliases or cap changes were used. Calibration and the selected-track joint law
remain open; availability is not a CE correction or a new F.

The next authorized semantic trace binds the registered upstream digitization
recipe to SimJob/MDC2025au / Offline v13_33_00. `TrigFlagPH` is DeltaFinder's
>=15-combo-hit proton-candidate counter, not a beam-PBI measurement; inclusive
tracker/calo activity can contain CE signal. The recipe's reconstructed bunch
time is truth smeared by 0.5 ns. See `veto-ce-proxy-semantics.md`; source/recipe
evidence still does not certify every historical worker or a calibration.

A separate slim keyed extraction matches every scalar-array digest to the completed
availability audit. The new saved-products joint diagnostic then reconciles
42,319 selected events/tracks with 91,226 retained events and preserves all
48,907 zero-window-track events. It retains frozen proxy/time and truth-PBI/time
tables, source-file summaries and sparse cells; no cuts or successful controls
were rerun. See `veto-ce-joint-diagnostic.md`. This is descriptive conditional
selection dependence, not calibrated target weights, generated-CE efficiency or F.
The saved-table reduction finds common proton-count marginal selection rates near
46%, but substantial descriptive MC-PBI dependence remains within fixed count:
zero candidates give 48.382% versus 44.341% in the lowest/highest PBI quintiles;
one candidate gives 48.579% versus 44.080%. No uncertainty/significance claim is
made. A one-variable proton-count replacement for the exposure law is not admitted;
reuse the saved rows/tables for any separately specified calibration validation.

The next **design-only** contract is now in
[veto-ce-calibration-validation.md](veto-ce-calibration-validation.md). It separates
retained A/N, generated-CE efficiency and the selected-track target F; requires an
independent calibration reference, deployable measured inputs, provenance and a
genuinely unexamined ancestry-separated validation population. All eleven viewed
sources remain development material. Event IDs, filenames and response replicas
are not automatically independent uncertainty units; library covariance is still
unestimated. No fit, split, new physics read or response run has been performed.

The subsequently authorized saved-row source/stored-PBT composition diagnostic
is now complete; see [veto-ce-composition-diagnostic.md](veto-ce-composition-diagnostic.md).
All inherited counts/identities/marginals reconcile. Count-0 contrasts are
4.042 points overall, 4.263 after common-support restriction and 4.247 after
common-composition weighting; count-1 values are 4.499, 4.376 and 4.337 points.
Support excludes 26.7–28.1% of count-0 endpoint events and 15.4–16.6% of count-1
endpoint events, retained explicitly in the report. These are descriptive
selection-fraction differences, not changes in F, significance or target correction
weights. No one-dimensional proton-count calibration is admitted. Reuse this
successful report; it cannot replace external calibration/validation evidence.

The subsequent read-only discovery is recorded in
[veto-ce-reference-evidence.md](veto-ce-reference-evidence.md). Two catalog-hash-
verified worker logs show why index matching is unsafe: worker0 writes output33;
the saved four-input group12 identifies worker12, whose actual log writes retained
dig output0 with 8212 output events and art status0. Its CE inputs are
150,0,185,136; runtime mixing-wrap warnings explicitly demonstrate reuse. This
closes one upstream digitization execution binding, not all eleven files or the
reco/ntuple/selection/fit-time chain, and provides no fresh physics-payload hash.
Local IHEP relative-PBI, straw-digi timing, STM energy, KPP relative timing and
FlatPBI recipe leads do not yet supply an independent target-domain reference.
KB access returned HTTP401; no authenticated literature search occurred. Exact
SAM definition lookup succeeds despite earlier empty wildcard-name queries, so
no dataset-absence claim is made. No fit, new holdout or F correction is admitted.

The documented on-host KB client subsequently authenticated successfully; the
previous401 was unauthenticated, not a continuing access blocker. See
[veto-ce-kb-evidence.md](veto-ce-kb-evidence.md): six successful searches and eight
full-text DocDB retrievals, with one separate timed-out search preserved. The
references identify a named pure-background MC study (53310), luminosity-stream
precision/systematics work (53761), and explicit pulse-history/minimum-bias/CRV
normalization requirements (34184/43581). Simulation provenance points to
Werkema32620/33344; tracker proxy/timing references point to31700/35621. These
are concrete next leads, not an independently measured or ancestry-separated CE
validation sample. No coefficients, uncertainty values or target weights imported;
all previous physics results remain frozen. Credentials stayed on their host.

The runtime-identified 32644 ancestor's metadata and upstream registered
20,000-input / 500-per-block NeutralsCat recipe are now captured. This advances
provenance but does not certify deeper physical histories. See
`veto-physical-reuse-gates.md` for the remaining population/reuse contract.

`veto-full-timing-admission.md` identifies the actual full-campaign implementation
and storage gates. Reuse successful controls and fixed stock; first validate new
persistence/import/aggregation using saved products. Current per-child caps do
not fit available space at full scale. No broad campaign is launched, and no
new F, CE correction or library uncertainty is claimed.
