# Physical source reuse: evidence and remaining gates

## Established evidence is not a physical weighting model

The twelve retained NeutralsFlashCat occurrences have seven measured payload /
immediate-parent groups. Their occurrence-weighted draw law is already documented
in `veto-library-weights.md`. The registered FlashCat recipe's twelve declared
5,000-file blocks now match saved SAM parent sets (`veto-cat-jobdef-audit.md`).
Neither observation establishes independent upstream histories.

Flash 13147's exact worker log uses NeutralsCat 32644, while its parent metadata
says 32205. The named prodtools v1.3.7 parent-list helper reproduces the latter
because it ignores sequential auxiliary selection; FCL generation honors it.
Do not use that defective catalog edge as physical ancestry or silently rewrite
all files using a presumed modulo rule (`veto-flash-parent-metadata.md`).

## Next ancestor, identified from runtime FCL

A new exact-file catalog query for
`sim.mu2e.NeutralsCat.MDC2025ab.001430_00032644.art` gives:

- file ID 102314010;
- event count 1,335,711 and `dh.gencount = 2500000`;
- declared physics-file SHA-256
  `a13526ef7df44dd61f118e4e7d8ffc2d3f49f50bf3282abb5ad3f9afe7bdbe4b`.

The complete 40,186-byte catalog reply has SHA-256
`59169d149a160a3a195dbd8696bc2a5dcd0bd9849c5a2fddd0b95d7e8c7b76cd`.
This is metadata for the filename identified by the worker FCL, not a fresh
physics-file hash or proof of that file's own deeper parentage.

The registered upstream concatenation archive is
`cnf.mu2e.NeutralsCat.MDC2025ab.0.tar`, catalog ID 102310740, 60,499 bytes,
SHA-256 `22dc4c2e314dfb9da041c25856cb361dffa02c38f969ddf901c6b56f67749bd3`.
Its complete 1,340,500-byte job-parameter member has SHA-256
`7b85e82681849c71c73be6998c669f60db718598c95779ee6fbd43d0dfde71ff` and specifies:

- `SimJob/MDC2025ab/setup.sh`;
- 20,000 named Neutrals files, grouped 500 per concatenation;
- maximum events 2,147,483,647 and empty subrun key;
- no explicit `sequencer_from_index` flag.

Thus there are forty declared input blocks, **not forty certified independent
source populations**. The 197-byte FCL includes `common/artcat.fcl`, the NeutralsCat
output pattern, and a `TriggeredOutput` filename override. That override is
preserved, not assumed to be executed or relevant to this output. This recipe
has not yet been tied to every concatenation worker's effective input assignment.
Its output sequencer must not be treated as a zero-based block index by analogy
with the later FlashCat recipe.

`scripts/capture_veto_neutrals_ancestor.sh` captures only these small documentary
objects from exact known paths, with byte checks, archive before/after hashes,
catalog/source-copy checks and a fresh output directory. Bash syntax checks and
manifest verification pass; the single capture took 0.53 s, 17,024 KiB peak RSS.
This is not a physics test, a full configuration expansion or execution proof.

## Minimal path to a physical reuse contract

1. **Bind assignments before grouping.** Obtain exact worker FCL/index/version
   evidence for the relevant resampling and concatenation paths, including any
   recovery jobs. Separately record catalog and runtime assignments; catalog bugs
   are findings, not permission to tune a mapping until hashes agree.
2. **Identify upstream histories.** Follow runtime-bound inputs to original
   generation identity and common ancestry across species. Seeds and disjoint
   immediate filenames alone do not establish independent shower populations.
3. **Define the target population and intended reuse.** Distinguish original
   incident exposure, surviving filtered events, resampling draws, retained source
   occurrences and selected-track exposure. The production reuse contract must
   say which population a weight represents. Repeated draws or concatenated counts
   cannot be called unique POT merely because sums are arithmetically consistent.
4. **Retain source identity in mixtures.** Library resampling must happen upstream
   of nonlinear mixing and response, respecting shared ancestry. Save selections,
   probabilities and original histories; no final-frame or filename bootstrap may
   masquerade as library covariance.
5. **Separate uncertainty levels.** Common raw mixtures pair stock and fast;
   response replicas are nested within original frames. Library replicas require
   a separately justified physical population. If this cannot be established,
   report historical F conditional on the retained library and leave the library
   uncertainty explicitly unestimated.

The latest evidence advances the first gate only. No deeper full ancestry census,
cross-species covariance, exposure renormalization, corrected mixture or new F is
claimed. Existing successful controls remain unchanged; C4 stays stock.
