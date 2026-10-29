# Retained source-occurrence accounting

`python3 -m crvfastsim.veto.library_weights` audits the empirical Cat sampling law
using **saved JSON only**. It neither reads physics payloads nor generates a new
mixture. It leaves all library, response, replay and BO defaults unchanged.

```bash
python3 -m crvfastsim.veto.library_weights \
  --manifests "$BLOCK0_MANIFEST" "$BLOCK1_MANIFEST" "$BLOCK2_MANIFEST" \
  --provenance "$SOURCE_PROVENANCE_SUMMARY" --out "$NEW_REPORT"
```

Inputs are historical replay-v1 manifests and a saved `source_provenance`
`summarize_prior` report. Input and tool hashes are retained; existing reports are
never overwritten. An output hashes the supplied summaries, **not a fresh
validation of the source art/parquet files**. Retain the original audit chain.

## What the weights mean

The historical mixer chooses uniformly from source-frame **occurrences**, with a
binomial thinning for stepless frames. Within block `b`, let `m_bg` be the number
of aliases of source group `g`, and `N_g` its retained table-frame denominator.
The probability of a source draw landing in that group is

```
w_bg = m_bg N_g / sum_h(m_bh N_h).
```

Duplicating a source changes this probability even though the configured Poisson
mean of total source draws is unchanged. It does not generate independent showers.
The report includes the current group weights and a separately labelled
**one-copy-per-group diagnostic**, `N_g / sum_h N_h`, within each block. Their total
variation measures a change of the empirical source sampling law, **not** a veto
loss, correction to F, or decision to deduplicate. The actual production reuse
weights remain a separate external contract.

The table also reports the probability of a drawn source frame having steps,
under current occurrence weights and the one-copy diagnostic. Another diagnostic
changes only the denominator to the catalog event count, holding the stored stepped
count fixed. It is not a substitute for all-event/identity validation: packed-key
collisions may also alter step grouping, and catalog counts alone cannot repair it.

Across blocks, the pooled source-draw concentration uses expected species draws
conditional on each saved mean PBI: `events * mean_intensity * efficiency * factor`.
This is not the frame weighting used for pooling F, nor an observed source-draw
count: the historical per-event species counts were not retained.

`sum_g w_bg*w_cg`, its normalized cosine, and `1/sum_g w_g**2` describe group-weight
overlap/concentration only. They do **not** measure the covariance of nonlinear
response or give an effective count of independent showers. In particular,
coincidences can combine several sources and a mixed frame has no unique donor
group. Do not bootstrap final mixed frames and call it finite-library uncertainty.
The report keeps `finite_library_uncertainty: null` explicitly.

## Fail-closed provenance requirements

- Every selected table binds to the saved exact source basename, payload hash,
  dataset and frame/stepped counts.
- Every pair of sources has exactly one parent-overlap record. Parent sets must be
  disjoint or identical; partial overlap requires a separate grouping model.
- Identical parent sets must have identical compact payload and count records to
  be treated as aliases here; identical payloads with different parent groups fail.
- Repeated input replay hashes, repeated source locators within a block, missing
  source bindings, invalid counts/hashes and unrepresented provenance sources fail.
- Even disjoint immediate-parent sets do not certify independent upstream ancestry.

## Remaining uncertainty and promotion gates

A library error requires a reviewed population and reuse weighting, all-event
census-complete source inputs, and a resampling design that changes the **upstream
library** before drawing mixtures and both response arms. Common raw mixtures pair
fast and stock; response replicas are nested within those original frames. Save
source selections/weights and separate mixing, response and library randomness.
Independent library replicas must respect production ancestry across species too.
No such production design, new simulation, stock rerun, CE-factorisation
certification or promotion is supplied by this accounting tool.

Tests:

```bash
python3 -m pytest -q tests/test_veto_library_weights.py
```
