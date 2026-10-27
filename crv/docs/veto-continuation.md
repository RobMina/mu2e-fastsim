# CE exposure continuation: audits and timing validation plan

These are preparation/read-only diagnostics, not production promotion. Historical
mixtures, response defaults, selections and fixed-stock outputs remain unchanged.
See `veto-exposure.md` for the opt-in PBI contract and `veto-cat-v2.md` for the
census-bound input format and unbuilt dumper addition.

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

Resolve original production parent overlap and exposure weights; verify a complete
all-event census against source counts; check empty-generator and packed-key
collision cases; freeze conditions/efficiency IOV; establish selected-exposure
correlations with measured-variable feasibility; then propose sample sizes and
resource requirements. The user builds the dumper and separately approves any grid
campaign. No automatic broad extraction, timing campaign or stock rerun is launched.
