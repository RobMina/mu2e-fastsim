# Common-support same-bar/end pulse timing

`python3 -m crvfastsim.objective.joint_pairs --directory BLOCK --data DATA --previous TIMING.json --out NEW.json`
continues the retained-array [timing diagnostic](response-joint-timing.md), not simulation or calibration.
It reuses full four-arm configuration/exposure/replay/identity gates, checks all capture NPZ hashes against the
previous completed timing report (exact capture-file-set and data-exposure equality), and independently reproduces
its same-PE-bin pair counts, distinct groups and
difference moments. Source and inputs are rehashed at the end; output is exclusively created. Original captures,
reports and calibration/reference assets are never rewritten. The MC and data capture loaders retain their
existing audits. No pulses are matched across arms, no fitted lines are recomputed, and no RNG is called.

## Pair population and algorithm

Retain complete finite selected event/slot spread groups with >=3 member hits and both measured line projections,
as in `joint_timing.prepare`. Within each group, pair pulses on the same bar and readout end with **distinct SiPM
indices**. Orientation is lower-index sensor minus higher-index sensor: 0−2 at side 0, 1−3 at side 1. It is not
PE ordering. If one sensor has multiple retained pulses, every cross-sensor combination contributes; same-channel
pairs never do. Retain all sectors, all 16 bodies and both sides. Empty bodies/support are explicit.

Shared along, path, distance, layer and group-count values are verified per event/bar/end. Same-bar/end distance
cancels; the difference of centered corrected residuals is the raw measured-time difference. The algorithm
reduces first and centered second moments per sensor/PE-bin, then takes the fixed 5×5 cross-bin combinations.
For sensors A/B: count=nA*nB, mean=meanA−meanB, second moment=varA+varB+(meanA−meanB)^2. No per-event Python loop
or full pulse-pair product is materialized. Cell summaries retain distinct event/slot counts and per-group mean
pair moments, including cases with multiple bars contributing to the same cell.

## Coordinates, support and weights

Base cell keys are **body × readout side × along bin × path bin × PE(lower SiPM) bin × PE(higher SiPM) bin**.
Edges are inherited unchanged: along [-1,-.6,-.2,.2,.6,1], path [1,1.25,1.5,2,3], PE [0,20,40,60,100,200].
All last upper edges are inclusive. Modes add no multiplicity, N bands (3–4, 5–8, 9–12, 13–16, 17–24, >=25),
exact N, or layer plus N band. N is the original **full event/slot spread-member count**, not pair count,
per-bar multiplicity, or newly defined per-body count.

A common cell requires >=30 pairs and >=10 distinct event/slot groups in all four arms and data. The data's
pair fractions on that support are fixed weights for every sample. Reports include marginal moments, each
sample's own-fraction moment on common support, standardized signed mean/second moment/variance and both PE
means, plus a within-cell group-equal second moment weighted by the same data pair cell fractions. This last
statistic is not a globally event-weighted moment: one event can occupy several cells. `group_cell_incidences`
is explicitly not an independent event count. Report pair and hit flows and common/in-range coverage; reject
unavailable support rather than lowering thresholds. Within-bin measured compositions can still differ.
Exclusive pair losses are PE exclusions followed by coordinate exclusions among PE-valid pairs; the separate
coordinate-excluded marginal overlaps PE exclusions and must not be added to them. Variance is accumulated via
within/between-cell centered moments, not a large second-moment-minus-mean-squared subtraction.

`diagonal_pe` uses only pairs whose two PEs occupy the same specified bin, but otherwise the same common-cell
gate. Off-diagonal PE cells remain in `all_pe`. Signed sensor differences may reveal asymmetry but are not
calibration offsets. Half a difference second moment is not an intrinsic sensor variance without independence,
equal variance, zero offset and appropriate selection assumptions.

## Fixed-support multiplicity contrast

The usual modes select different support and data targets; their differences cannot isolate multiplicity.
`fixed_support_multiplicity` instead starts **only** from exact-N cells common to every sample. Within that
identical retained pair population, compare:

1. Fixed data fractions across exact-N cells.
2. The same data weights across the coarser six-coordinate cells, but each sample's own N fractions inside each.

No pair rows are added or removed between the two statistics. Their difference measures a reweighting of the
observed N mixture within the coarse cells on this restricted support, not a causal multiplicity effect. Counts,
coarse-cell counts and the exact-support coverage remain available. No extrapolation to unsupported cells.

## Map provenance boundary

Readout side and sensor index are offline identities, not a claim of verified historical physical cabling. The
only FEB map used by the preparation gate remains the frozen MC CSV. The locally retained P4 reconstructed-art
config dump records `crvOrdinal.useDb:false`, `filePath:Offline/CRVConditions/data`, and `useROC4asROC2:true`,
but selects a moving geometry alias and does not retain the resolved map bytes. Current local `extracted_v04.txt`
matching the MC CSV is candidate-map agreement, not certification of the historical job or wiring. Do not
replace that boundary with inference from current source. Pulse/arrival/waveform/phase upstream observables
remain absent; calibration/background/response/GP noise are unchanged.

## Completed October 26 follow-up

`pairs_{a,b}_20261026_v1.json` in `diagnosis_saved_v2/` hold the complete four-arm/data results. T1 exact-N
mean pair difference² remains MC 12.83–15.35 versus data 4.93–5.30 ns², with only 19.2–20.5% data pair coverage.
The fixed-support multiplicity reweighting changes moments by at most 0.047 ns². Only 6–10 coarse cells have
multiple supported N values (33.6–47.8% of the already-retained data target weight); this is limited leverage,
not a global exclusion of multiplicity. All sectors and sides are included, with explicit sparse support.

`python3 -m crvfastsim.objective.joint_pairs_report --a A.json --b B.json --outdir NEW` creates the checked summary
and plots; final output is `pair_summary_20261026_v2/`. It checks exclusive pair flow, moment/variance identity,
fixed-target weight normalization, zero data multiplicity contrast, and exact A/B data-summary equality.
The same pair population is not identical to the earlier single-pulse residual or unrestricted pair population;
compare the named quantities/support, never their numeric scales as if interchangeable.

`objective.joint_map_audit` records local candidate-map agreement, frozen-CSV hashes and retained expanded-config
values. `map_audit_20261026_v1.json` finds all 1,920 instrumented current-v04 channels equal to the frozen MC CSV
(384 additional CSV channels uninstrumented). The alternate `extracted.txt` differs on 64 shared assignments
and lacks 128 v04 channels. The report explicitly sets historical-map and physical-cabling certification false.
The saved reports retain detailed results, provenance and the unresolved upstream-evidence boundaries.
