# Tier-1b same-bar descendants: chord placement, a follow draw and a larger near-miss census

## Decision

Two new **opt-in** options on top of the templates arm (`descendant-template-closure.md`), with a new template
sidecar (v3) that adds the library crossings' chords and a census eleven times larger:

- `chord` (`tier1b-apply --chord-placement`): muon-descendant steps in the muon's own bar are placed along the fast
  crossing's chord at their fraction of the library crossing's chord. Before, they were placed at their distance
  along the library muon's direction.
- `follow` (`tier1b-apply --follow-muon`): along each fast muon, a crossing takes the next crossing of the previous
  crossing's library muon when that crossing lies in its own draw cell.

On 5M held-out events (even g4ref_v2 shards 22-220):

- `chord` halves the step y residuals: all/y 3.23 -> 1.73, T1/y 4.53 -> 1.60, T2/y 5.42 -> 2.84; muon descendants
  alone T1/y 13.3 -> 6.3, T2/y 10.4 -> 5.8. It leaves every other histogram within noise.
- `follow` brings NStep 4.67 -> 3.03, all/z 2.36 -> 2.04, EX/z 2.17 -> 1.87 and bar 19.7 -> 17.8. It costs y
  (T2/y +0.5 on top of chord), lE (1.72 -> 2.18, floor 1.4-2.7) and pathLen (1.69 -> 1.83).

Recommendation: `chord`, with the templates arm, is the robust gain. `follow` is a trade (NStep and z against y and
lE) and stays a separate option. **The step-level closure target (deck <= 2 everywhere) is still not met.** The
remaining failures are T1/z (3.1), T2/y (2.8-3.3), NStep (3.0-4.7), all/z (2.0-2.3), and bar/posx (at or near their
Poisson floor).

The larger census moves the near-miss reach in the "ray enters the box" cells by about +30 % (T2 0.281 -> 0.365,
M 0.300 -> 0.385). It changes no step histogram beyond noise. Defaults, CCP, topology and the templates arm are
unchanged, bit for bit.

## Diagnosis (development shards g4ref_v1 11-20, G4-only statistics on 1-20)

Muon descendants in the muon's own bar are 86 % of descendant steps. Three explanations were ruled out first:

- **Segmentation.** CrvSteps are one per (track, bar) in both the reference and the library donors, and a primary
  muon has exactly one step per bar, as tier 1a does. Library rows are not merged or split differently.
- **Library physics.** Donors and reference agree, crossing by crossing, to three digits in same-bar step length,
  energy, path fraction, distance from the path and thickness coordinate.
  - Process mix per crossing (donors / reference): muIoni 2.248 / 2.238, eIoni 0.356 / 0.353, compt 0.141 / 0.135,
    phot 0.115 / 0.113. Decay electrons: 0.0007 / 0.0006.
  - Same-bar multiplicity per library bin, G4 / library: EX 0.995, T1 1.007, T2 1.004, M 0.982 (chi2 22-35 over 28
    bins; M 46).
- **Rows changing bar.** Other-bar rows landing in a bar the muon crosses, and same-bar rows leaving it, are below
  0.1 %.

Two mechanisms remain.

**A. Position along the path (step y).**

- What happens: a same-bar row keeps its offset from the library entry, rotated from the library muon's direction to
  the fast one. Within a path-length bin the library chord is longer or shorter than the fast one.
  - With a shorter library chord, the steps crowd toward the entry face.
  - With a longer one, they overshoot the fast exit, leave the bar and fall back to their library position across
    the bar, which is unrelated to where the fast muon crosses it.
- Signatures (own-bar steps, fast/G4):
  - In-bar thickness distribution by quarter, bottom to top: EX 1.014 / 1.010 / 1.012 / 1.044, T1 0.974 / 0.954 /
    0.989 / 1.040, T2 1.023 / 1.013 / 1.003 / 1.033.
  - Chord fraction (0.9, 1]: 0.75x; beyond the exit: 1.4-1.5x.
  - Steps more than 10 mm off the path: 1.7-1.9x.
- Size: own-bar steps alone give exact y chi2/ndf EX 1.24, T1 1.74, T2 1.49 on 500k events.

**B. Correlation between the crossings of one muon (NStep).**

- In G4, the own-bar counts of a muon's crossings are correlated: rho = 0.14 over all pairs (0.22 for adjacent
  crossings), after the mean of the (bin, context) cell.
- The correlation is in the EM-cascade products, not in the direct deltas:

  | Own-bar steps per crossing | Mean | Variance | rho between crossings |
  |---|---:|---:|---:|
  | Direct muIoni deltas made in the bar | 2.11 | 2.14 (Poisson) | 0.003 |
  | Cascade products made in the bar (compt, phot, eIoni) | 0.61 | 2.97 | 0.21 (adjacent 0.34) |
  | Imported from another bar | 0.14 | - | 0.10 |

- Path length and momentum within the bin explain none of it (rho 0.138 -> 0.134).
- Effect: independent draws per crossing under-disperse the per-event descendant count. With 8 crossings the
  variance is 78 against G4's 107; with 12 or more, 292 against 456.

## Design

**`chord`** (needs the library chords of a v3 template sidecar). For a same-bar row with start offset `d` from the
library entry, library chord `c_L` (length `l_L`) and fast chord `c_F` (length `l_F`):

- Split `d` into `a` along `c_L` and a perpendicular part.
- The along-chord coordinate becomes `a * l_F / l_L` inside the chord (`0 <= a <= l_L`). Before the entry it keeps
  `a`; beyond the exit it becomes `l_F + a - l_L`.
- The perpendicular part, the step vector and the momentum are rotated from `c_L` to `c_F`.
- The existing same-bar fallback (G4 thickness and width when the step leaves the bar) and `contain` then apply
  unchanged.
- No random number is used.
- Tried and rejected: replacing only the out-of-bar axes in that fallback (development desc all/y 0.98 -> 1.15,
  T1/y 1.38 -> 1.67).

**`follow`** (needs the next crossing of each library crossing: the v3 sidecar).

- The crossings of each fast muon are taken in time order.
- Crossing k takes the library muon's next crossing after crossing k-1's pick, but only when that crossing lies in
  crossing k's own draw cell: the (bin, context) cell, or the bin fallback.
- Otherwise crossing k keeps its own uniform pick.
- The same uniform numbers are drawn, so every later component (showers, unvisited rows, near misses) is unchanged.
- On the development shards 39 % of crossings follow.

**Template sidecar v3** (`valhist.descendant_templates sidecar`):

- Per library crossing, it adds the exit point, path length and next crossing in time (`xing_qx/qy/qz/len/nxt`).
- The census options can be repeated, to take several table sets.
- With the v2 census (`g4ref_v1` 1-10) it gives the v2 counts exactly (2,044 near-miss templates, 66,141 steps,
  350,179 census muons, 117,637 unvisited rows). Its templates arm equals the v2 arm bin for bin.

## Census and the evaluation split

- **No new training tables exist.** `g4ref_v2` holds shards 21-220 only, and the library donors have no full
  generator stream, which the reach denominator needs.
- **Split.** The census therefore uses g4ref_v1 shards 1-10 (as before) plus the **odd** g4ref_v2 shards 21-219:
  5.5M events and 3,849,382 no-crossing muons, against 500k / 350,179 before. Every held-out number below uses the
  **even** shards 22-220 only (5M events).
- **Not used:** g4ref_v1 shards 11-20 (the deck's held-out DQM set) were not used for the census.
- **Reach in the "ray enters the box" cells:**
  - T2: 13 -> 110 census muons, probability 0.281 -> 0.365.
  - M: 7 -> 60 census muons, probability 0.300 -> 0.385.

  Both shifts lie within the old +-28-38 % Poisson error. The EX cells move by at most 0.006. No template sits in a
  cell without census muons.

## Results (held-out even shards 22-220, 5M events)

chi2/ndf vs G4. "templates" is the templates arm with the v2 sidecar; `chord`, `follow` and chord+follow use the v3
sidecar.

- Statistic: the deck statistic with Poisson variance, exact clustered value in brackets. The y/z rows use the exact
  clustered variance (the deck rule), so for them deck = exact.
- Seeds: single-seed columns use the `tier1b-apply` seed rule; "3 seeds mean" averages the vs-G4 scores of three
  independent seeds.
- Floor: the range over the three pairs of independent seeds of the same arm.

| Histogram | templates | chord | follow | chord+follow | templates 3 seeds mean | chord+follow 3 seeds mean | floor (templates seeds) | floor (chord+follow seeds) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E | 1.31 [0.93] | 1.32 [0.94] | 1.35 [0.91] | 1.35 [0.91] | 1.59 | 1.71 | 1.07-1.29 | 1.03-1.46 |
| lE | 1.72 [0.68] | 1.74 [0.69] | 2.18 [0.81] | 2.18 [0.81] | 2.31 | 2.79 | 1.43-2.39 | 1.65-2.72 |
| pathLen | 1.69 [0.66] | 1.70 [0.67] | 1.83 [0.61] | 1.83 [0.61] | 2.23 | 2.64 | 0.55-0.83 | 0.46-0.95 |
| bar | 19.68 [0.71] | 19.18 [0.69] | 17.76 [0.60] | 17.76 [0.60] | 26.42 | 29.17 | 3.80-15.20 | 5.92-19.13 |
| NStep | 4.67 | 4.68 | 3.03 | 3.03 | 4.73 | 3.20 | 0.71-1.08 | 0.80-1.08 |
| SMom | 1.32 [1.18] | 1.32 [1.18] | 1.38 [1.24] | 1.39 [1.24] | 1.40 | 1.40 | 0.96-1.06 | 0.97-1.13 |
| SMom2 | 0.45 [0.05] | 0.45 [0.05] | 0.45 [0.05] | 0.45 [0.05] | 0.46 | 0.46 | 0.00-0.01 | 0.00-0.01 |
| posx | 37.70 [1.61] | 37.68 [1.60] | 37.06 [1.54] | 36.93 [1.54] | 34.09 | 33.95 | 15.60-18.65 | 15.11-17.70 |
| all y / z | 3.23 / 2.36 | 1.73 / 2.34 | 3.60 / 2.04 | 1.97 / 2.02 | 3.38 / 2.30 | 2.36 / 2.12 | 0.59-0.93 / 0.48-0.59 | 0.65-1.19 / 0.46-0.54 |
| EX y / z | 1.66 / 2.17 | 1.57 / 2.15 | 1.52 / 1.87 | 1.50 / 1.86 | 2.13 / 2.08 | 2.41 / 1.90 | 0.38-0.52 / 0.45-0.60 | 0.43-0.94 / 0.45-0.56 |
| T1 y / z | 4.53 / 3.17 | 1.60 / 3.13 | 5.24 / 3.10 | 2.07 / 3.06 | 4.28 / 3.52 | 2.22 / 3.46 | 0.65-0.90 / 0.57-0.76 | 0.63-1.33 / 0.48-0.90 |
| T2 y / z | 5.42 / 0.93 | 2.84 / 0.93 | 6.25 / 0.96 | 3.33 / 0.96 | 5.61 / 1.06 | 3.58 / 1.10 | 0.49-0.72 / 0.47-0.59 | 0.33-0.81 / 0.43-0.67 |
| M y / z | 0.77 / 1.05 | 0.78 / 1.04 | 0.73 / 1.08 | 0.73 / 1.08 | 1.02 / 1.57 | 1.02 / 1.53 | 0.79-1.84 / 0.87-1.01 | 0.58-2.19 / 0.84-0.93 |

Histograms passing deck <= 2 (seed 0):

| Arm | Passing | Count |
|---|---|---:|
| templates | E, lE, pathLen, SMom, SMom2, EX/y, T2/z, M/y, M/z | 9 |
| chord | the templates set plus all/y and T1/y | 11 |
| chord+follow | the chord set without lE and T1/y, plus EX/z | 10 |

**Muon descendants only** (exact clustered chi2/ndf, templates -> chord -> chord+follow; seed 0):

| Arm | all/y | all/z | EX/y | EX/z | T1/y | T1/z | T2/y | T2/z | M/y | M/z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| templates | 7.73 | 3.24 | 2.90 | 2.95 | 13.28 | 4.05 | 10.41 | 1.38 | 3.05 | 2.31 |
| chord | 4.56 | 3.19 | 2.58 | 2.90 | 6.32 | 4.01 | 5.79 | 1.36 | 3.11 | 2.28 |
| chord+follow | 4.68 | 2.73 | 2.33 | 2.41 | 6.07 | 3.88 | 6.73 | 1.46 | 3.16 | 2.24 |

Bar: 0.98 / 0.99 / 0.82. Three-seed means of chord+follow: all/y 4.52, T1/y 5.91, T2/y 6.46. Primary-muon and
non-muon steps are unchanged, at <= 2.2.

**Step yields vs G4** (relative difference, clustered independent SE). They are the same within 0.3 % for every
arm.

| Sector | all steps | muon descendants (templates / chord+follow) |
|---|---:|---:|
| all | +0.29 +- 0.49 % (paired 0.40) | -0.02 / +0.07 +- 0.47 % |
| EX | +0.36 +- 0.44 % | +0.22 / +0.18 +- 0.43 % |
| T1 | +0.85 +- 1.06 % | -1.25 / -0.79 +- 0.98 % |
| T2 | -0.40 +- 0.77 % | +0.28 / +0.60 +- 0.77 % |
| M | -0.13 +- 1.62 % | -2.43 / -3.72 +- 2.55 % |

**Truth census, energy fast/G4.**

- Own bar is unchanged by the options: EX 1.002, T1 0.999, T2 1.011, M 1.010 (chord+follow 0.978).
- Never-visited layers are unchanged: EX 0.69, T1 0.85, T2 0.89, M 0.87.
- No-crossing descendants, v2 -> v3 census: EX 0.72 -> 0.75, T1 0.30 -> 0.35, T2 0.69 -> 0.77, M 0.71 -> 0.81
  (SE 0.09-0.22). The near-miss deficit is not a census effect.

**Cluster level** (NumPy chain on the step tables, shards 21-30, deck diagonal score with paired in brackets). The odd
shards 21-29 fed the v3 census, through the near-miss reach only.

| Arm | EX/y | T1/y | T2/y | M/y | z (all sectors) | cluster counts vs G4 |
|---|---:|---:|---:|---:|---|---|
| templates | 0.93 (0.98) | 1.19 (1.41) | 1.23 (1.28) | 0.43 (0.44) | <= 0.52 (T1/z paired 2.43, ndf 1) | EX -0.06, T1 -1.37 +- 0.87, T2 -0.64, M -0.54 % |
| chord+follow | 0.70 (0.75) | 0.94 (1.12) | 1.31 (1.37) | 0.98 (1.05) | <= 0.47 (T1/z paired 0.94) | EX -0.06, T1 -0.84 +- 0.87, T2 -0.45, M -0.60 % |

No cluster score is flagged. The templates row reproduces that study's cluster run (counts identical, PE sums to
1e-9).

## Bit identity

All checks are exact frame equality.

- **Sidecar v3 attached, new options off** (`placement_closure run --retained`, held-out shards 21-40, all 20
  shards): default, CCP and topology equal the retained tables of `descendant-placement-closure.md`.
- **Templates arm with the v2-census v3 sidecar:** equals the retained templates tables of
  `descendant-template-closure.md`.
- **`tier1b-apply` on shard 21:**
  - templates (v2 sidecar) equals that retained table (79,164 steps, 101,217 particles).
  - `--chord-placement --follow-muon` (v3 sidecar) equals the closure run's table (79,602 steps, 101,639
    particles).

## What changed (all opt-in)

- `crvfastsim.tier1b`:
  - `_chord_map`, `_follow_picks`;
  - the `chord` and `follow` options of `_desc_for_crossings` and `sample`;
  - `attach_templates` reads library chords and next crossings (`lib["xgeo"]`) when the sidecar has them;
  - placement counters `chord` and `follow`.
- `valhist.descendant_templates`: per-crossing exit, length and next crossing; `next_crossing`; repeated census
  sources.
- `valhist.placement_closure`: the options `chord` and `follow`.
- `tier1b-apply`: the flags `--chord-placement` and `--follow-muon`.
- Tests: `tests/test_tier1b_samebar.py`.

## Reproduction

Paths are examples. Code is this checkout plus a matching core on `PYTHONPATH`, and every output goes to a new path.
The transport, reference and topology sidecar are those of `descendant-placement-closure.md`.

```bash
# template sidecar v3: library chords + census on g4ref_v1 1-10 and the odd g4ref_v2 shards (about 1 min, 6.4 GB)
python3 -m valhist.descendant_templates sidecar --manifest LIBRARY.json \
    --census-steps 'G4V1/crvsteps.{s}.root' --census-gen 'G4V1/gen.{s}.root' --census-shards 1-10 \
    --census-steps 'G4V2/crvsteps.{s}.root' --census-gen 'G4V2/gen.{s}.root' --census-shards 21,23,...,219 --out T3.npz
# held-out: even shards, in chunks; join the v2 and v3 runs of a chunk with '+' in report
python3 -m valhist.placement_closure run --library LIBRARY.root --sidecar SIDE.npz --templates T3.npz \
    --transport 'TRANSPORT/fastC.{s}.root' --reference 'G4V2/crvsteps.{s}.root' --shards 22,24,...,40 \
    --arm chord=context,contain,point,topology,unvisited,nearmiss,chord \
    --arm sb=context,contain,point,topology,unvisited,nearmiss,chord,follow \
    --arm sb2=context,contain,point,topology,unvisited,nearmiss,chord,follow@2 --out RUN.json
# the same arm as a table
python3 -m crvfastsim tier1b-apply fastC.SUBRUN.root --library LIBRARY.root --outdir DIR --seed K \
    --crossing-context --contain-steps --point-placement --topology-sidecar SIDE.npz \
    --template-sidecar T3.npz --unvisited-layers --near-miss --chord-placement [--follow-muon]
```

Measured on a 10-core laptop: the v3 sidecar takes 57 s (6.4 GB); six arms take 16 s per shard (7.7 GB per 10-shard
process). The v3 sidecar is 130 MB.

## Limitations

- **Conditional statistics.** They are conditional on the fixed library, transport, reference and census. The
  diagonal scores are not calibrated p-values. Single-seed scores move by about +-1 between seeds on these y
  histograms, so the three-seed means are the fairer comparison.
- **Smaller evaluation set.** Held-out numbers come from 5M events (half of the earlier 10M), because the odd shards
  went into the census. A systematic chi2 excess halves with the sample size.
- **Open residuals:**
  - T1/z (3.1) and T2/y (2.8-3.3) remain.
  - NStep remains 3.0 against a floor of about 1, because `follow` catches only crossings whose library successor
    lies in their cell (39 %). The per-event variance at 12 or more crossings is still 303 against G4's 456.
  - T2 own-bar energy stays 1.1 % high (library multiplicity, 1.8 SE).
  - No-crossing descendant energy is 0.35-0.81 of G4.
- **Chord mapping** keeps the library row's step vector and length; the along-chord stretch moves only the start
  point.
