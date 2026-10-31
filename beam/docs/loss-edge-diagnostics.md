# Restricted-soft / explicit-hard edge diagnostics

## Status

The split sampler remains diagnostic and opt-in. No production-default change is
supported by these results. Matching a mean or stop fraction does not certify the
exit-energy distribution. The reference configuration limitations below are
independent of numerical agreement.

## Evidence tiers

Three kinds of saved inputs must not be interchanged:

1. Original grid `t1056` scan: 100,000 mu− events, original hash-bound effective
   config, log, macro, geometry, gun, receipt and audited reduced arrays. SimJob
   MDC2025ac / Geant4 11.3.2, ShieldingM, G4_Al, 0.01-mm range cut, zero proton
   production cut, lateral displacement enabled. No process-disable or stepping
   commands in its macro. ROOT was deliberately removed with its identity
   retained. Run-specific electron energy-cut export is absent.
2. Monoenergetic mu− cells: 5,000 events per mode with surviving reduced arrays,
   canonical summary and slab ASCII loss tables. Nine refined arrays retain
   manifest hashes; original 18 arrays lack their generation manifest but match
   recorded census and moments. Original effective configs/logs/receipts/ROOT
   are unavailable, so their semantic configuration qualification is unverified.
3. Moved-target pilot: original 2,000-muon whole-event input block and historical
   outputs retain binding to the full G4 input hash and all selected columns.
   An older full fast baseline instead has counts/metadata but no input hash or
   row IDs; its paired analysis is conditional on recorded row ordering.

Current checks cannot upgrade absent historical evidence. No fake receipt or
relaxed audited-loader condition is used to make these inputs equivalent.

## Repeatability and edge variance

New fast execution with the original seeds reproduces the reported mu− edge
variances and endpoints. Independent seeds with 8,000 entries preserve the
failures against the available **stock full**, not converged full, G4 reference:

| Cell | Fast transformed variance (MeV²) | Fast − G4 ± independent SE (MeV²) |
|---|---:|---:|
| 2.02 MeV, 0.2 mm, normal | 0.01634723 | +0.00319604 ± 0.00041578 |
| 2.3 MeV, 0.2 mm, cosine 0.8 | 0.05977803 | +0.00891551 ± 0.00108775 |

Both fail the provisional pointwise 5%-variance band at the 3-SE criterion.
Loss-only controls against finest loss give −0.00070942 ± 0.00042459 and
−0.00069347 ± 0.00056421 MeV² respectively: indeterminate, not equivalence.
The raw-residual excesses persist (+0.00342583 and +0.00906799 MeV²), so the
explicit `T < 0.1 MeV -> 0` transform does not create the discrepancy. Positive
floor endpoints remain unresolved and must not be called physical stops.

## What contributes to the excess

For `q = P(T >= 0.1 MeV)`, conditional mean `m` and conditional variance `s²`,
all-entry population variance is `V = q*s² + q*(1-q)*m²`. The exact symmetric
algebraic decomposition separates occupancy, conditional mean-squared and
conditional width terms. Independent-8k contributions are:

| Cell | Occupancy | Conditional mean-squared | Conditional width |
|---|---:|---:|---:|
| normal | −0.00073576 ± 0.00022607 | +0.00314930 ± 0.00028409 | +0.00078309 ± 0.00011332 |
| oblique | −0.00106971 ± 0.00030946 | +0.00771531 ± 0.00087911 | +0.00227260 ± 0.00041139 |

Units MeV²; pointwise delta-method SEs are correlated. This is outcome-selected
bookkeeping, not causal attribution. Fewer fast above-threshold survivors reduce
variance; their excessive energy and width dominate in the opposite direction.
Calling it an occupancy-only failure would be wrong.

The no-hard-emission outcome cohorts retain the width excess. Direct hard-applied
variance plus covariance is small compared with the full excess, but the applied
soft channel also contains path/stop selection. No fitted soft or hard scale, or
removal of hard transfers, follows from these diagnostics.

## Numerical controls and model scope

Halving max step, energy fraction, angular fraction, or all three together does
not remove either variance failure. For these samples the half-angle control
leaves threshold-energy arrays identical; this does not certify exit angles.
Keyed-identity response differences are not Brownian-bridge coupled trajectories.

The table relation `lambda1 = energy² / LambdaMod0` agrees with Geant4 11.3.2
`G4VMscModel::GetTransportMeanFreePath`. Its WentzelVI transport also implements
finite-collision/single-scattering handling, true/geometric path conversion,
non-Gaussian sampling and lateral displacement. Gaussian tangent diffusion is
not this model. The empirical evidence points to the coupled path/angle/energy
response, but surviving arrays have neither exit directions nor true step lengths
to identify a unique cause. Chord sums are not true G4 paths.

## Remaining decisions

A new reference must preserve complete configuration, process/cut/export and
highest-cycle reduced-array evidence. At 5 MeV a 0.02 CDF band with 0.01 distance
headroom requires 43,811 independent G4 entries with 1M fast, under the existing
pointwise 99% DKW rule. Fifty thousand per separately labelled mode is a scoped
precision proposal, not an expected PASS or authorization. Refined full and
material-qualified mu+ evidence are separate requirements. No full-chain model
promotion is justified by a small pilot recovering first-foil counts.

## Independent refined-full controls

A later bounded G4 test supplies 5k independent mu− histories per cell/mode,
with retained ROOT, effective configuration, native table exports and runtime
provenance. Both modes use 168 bins/decade and StepFunctionMuHad 0.005 / 0.00005 mm;
only muon/hadron lateral displacement differs. Exported range/DEDX/lambda nodes
exactly equal the package used by the already-completed fast samples. These new
receipts do not repair the absent historical monoenergetic configuration bundles.

| All-entry threshold variance, MeV² | Normal | Oblique |
|---|---:|---:|
| Historical stock full | 0.01315118 | 0.05086252 |
| New refined full | 0.01842124 | 0.05854984 |
| New refined full, lateral off | 0.01895671 | 0.05907849 |
| Unchanged independent 8k fast diffuse | 0.01634723 | 0.05977803 |

Refinement changes variance by +0.00527006 ± 0.00063359 normal and
+0.00768733 ± 0.00124141 oblique (independent pointwise SE). Both exceed the
historical 5% variance band at 3 SE. This is combined table/step-setting sensitivity,
not a single-control convergence proof. The unchanged fast variance ratios versus
new refined G4 are 0.8874 / 1.0210; both variance gates are **indeterminate**, not
passed. Thus the historical positive excess is not sufficient evidence to assign
its sole cause to the fast angular/path approximation.

Matched refined lateral-off shifts are +0.00053547 ± 0.00087914 and
+0.00052865 ± 0.00136985 MeV², indeterminate rather than equivalence. The point
shifts are much smaller than refinement sensitivity. Fast/G4 two-sample DKW
radius is 0.04383 > 0.02; CDF acceptance remains statistically impossible here.

First-exit momentum rows provide directions without a new dumper. Above-threshold
median incident deflection is 0.3161 / 0.3162 rad fast versus 0.2819 / 0.2860 G4
(normal/oblique). These are each arm's own exit-conditioned populations, not an
all-primary angular gate or causal test. Nonexits retain undefined directions;
true G4 step lengths remain absent. No physics parameter, default or gate is
changed. Further G4 convergence and precision work is a separate decision.

## Bulk, exit selection and local angular law

Read-only follow-up of the retained four-control ROOT files establishes:

- Two normal-incidence high-energy back exits add 0.00060138 MeV² to the
  all-entry centered variance. They are not the entire normal discrepancy:
  fixed 0.5-MeV capping of **all** entries leaves fast−refined-G4 variance
  −0.00147326 ± 0.00046541 MeV². Refined−historical-stock remains
  +0.00466236 ± 0.00046238 under the same diagnostic cap. Caps are not
  replacement acceptance gates; original all-entry gates stay indeterminate.
- Forward above-threshold mean deflection excess is +0.02314 ± 0.00564 rad
  normal and +0.02126 ± 0.00475 oblique. Exact symmetric standardization
  on fixed exit-energy bins [0.1,0.2,0.3,0.5,infinity] assigns only
  +0.00159 / +0.00015 rad to between-bin composition. Within-bin shape
  contributions are +0.02155 ± 0.00551 / +0.02110 ± 0.00472 rad.
  These are correlated, outcome-conditioned diagnostics; coarse bins do not
  remove within-bin composition or establish causation.
- The lateral toggle has a visible geometric signature: cumulative adjacent
  chord components transverse to pre-step momentum average 1.282 / 0.463 µm
  on/off normally and 1.489 / 0.442 µm obliquely in forward-above exits.
  This is neither total lateral displacement nor true path. Zeroing the toggle
  does not remove all internal single-scattering displacement.
- For above-threshold adjacent recorded steps across **all** primaries,
  sum(1−u·u_next)/sum(chord/lambda1(T_pre)) is 1.0033 ± 0.0158 normal,
  1.0219 ± 0.0169 oblique. Per-primary cluster errors are used, not iid-step
  errors. The transport normalization is approximately consistent, while
  fourth moments are strongly non-Gaussian: sum(theta^4)/sum(8q²) is
  243.5 ± 41.8 / 265.4 ± 42.0. Jumps above 0.6 rad carry about 20% of
  the first moment. At q<=0.001 the fourth-moment ratios remain
  458.8 ± 94.1 / 557.6 ± 99.3, so large-q Gaussian truncation is not
  the explanation of that local evidence.
- Selecting only forward-above exits changes the first-moment ratios to
  0.7934 / 0.8822. Those outcome-selected samples must not be used to
  calibrate the transport coefficient. Large deflections can occur within a
  step ended by muIoni; an ending-process label is not an exclusive scattering
  classification.

The local audit uses pre-step energy and geometric chords, not true Geant4
lengths; pre/post energies are required >=0.1 MeV. It compares a small-q
Gaussian increment law at those recorded budgets, not fast legs with different
numerical sizes. No angular rescale or fitted tail correction is justified.
Further validation should isolate the angular law from entry selection and
boundary/energy coupling, while separately checking reference convergence.

Aggregating retained fine G4 chords into fixed 0.01-mm chord blocks leaves
all-primary transport ratios 0.9927 ± 0.0165 / 1.0145 ± 0.0176 and fourth-moment
ratios 7.34 ± 1.14 / 7.65 ± 1.12 (normal/oblique); 0.005-mm blocks give 10.91 /
11.48. Thus the non-Gaussian evidence is not confined to the finest readback
increments. Blocks retain their actual summed chord/lambda budget and actual
endpoints, with no interpolated trajectory, and uncertainties cluster by primary.
They are not the fast integrator's adaptive legs and do not establish how much
of the all-entry energy discrepancy a new angular sampler would correct.

## Isolating released-runtime step and table-grid settings

A bounded two-level factorial retains lateral displacement and all other
geometry/material/runtime settings. At each edge, three independent controls
(stock-step/7-bin, stock-step/168-bin, refined-step/7-bin) complement the already
retained refined-step/168-bin arm. "Stock step" denotes the inherited released
runtime rather than a recovered historical signature. Original artifacts and
native exported grids distinguish requested settings from executed settings.

Independent-primary contrasts separately estimate step changes at each grid,
grid changes at each step setting, average main effects and the difference of
differences. Shared-arm covariance is retained. This can identify sensitivity
within the current released setup; it cannot establish the old missing
configuration, a numerical convergence limit, or acceptance of the original
CDF gate with only 5k references. No result or model correction is asserted
before those controls are complete and audited.

### Measured two-level factorial result

All six new 5k references retain 5,000 resolved primaries, original ROOT and
actual native-grid verification; binary and recorded Offline-source hashes
match the reused refined/168 arm. Threshold-transformed sample variances are:

| Incidence | Stock step, 7 bins | Stock step, 168 bins | Refined step, 7 bins | Refined step, 168 bins |
|---|---:|---:|---:|---:|
| normal | 0.01332453 | 0.01317323 | 0.02036973 | 0.01842124 |
| oblique | 0.04816685 | 0.04725983 | 0.06125357 | 0.05854984 |

Units are MeV²; the explicit T<0.1-MeV transform retains every primary. Median
recorded foil steps are 3 for stock, about 470 normal / 503 oblique for refined.
Refined−stock variance changes at fixed 168 bins are **+0.00524801 ± 0.00063423**
normal and **+0.01129001 ± 0.00122862** oblique; at fixed 7 bins they are
+0.00704519 ± 0.00092726 / +0.01308672 ± 0.00179860. All four exceed the
unchanged 5% variance band after pointwise 3SE. Step sensitivity is therefore
identified without relying on the missing historical configuration bundles.

The 168−7 bin effects and step×bin interactions remain unresolved at pointwise
3SE, **not established equivalent or absent**. Interactions are
−0.00179718 ± 0.00112342 normal / −0.00179671 ± 0.00217818 oblique. Two step
levels do not prove a convergence limit. Related contrasts share arms and
retain covariance; these are not simultaneous confidence statements.

Unchanged independent fast variance ratios versus newly audited stock/7 are
1.22685 / 1.24106, both FAIL. Refined/168 ratios remain 0.88741 / 1.02098,
INDETERMINATE. Normal refined/7 instead gives 0.80253 and a variance FAIL,
plus threshold difference +5.9475 percentage points (conservative 99% interval
+2.5565 to +9.3408 pp), also FAIL. No reference setting is selected to obtain
acceptance. The CDF radius remains 0.04383 > 0.02. Historical stock versus
current stock remains statistically indeterminate and cannot repair absent
historical signatures. No model correction or production default is promoted.

The step effect is not confined to rare high-energy back exits. Applying a
fixed 0.5-MeV cap to **every** primary leaves refined−stock variance changes
+0.00569342 ± 0.00046558 / +0.00464101 ± 0.00046327 normally (7/168 bins),
and +0.00408768 ± 0.00062706 / +0.00441693 ± 0.00061955 obliquely. The
0.3-MeV cap also leaves positive pointwise-3SE differences in all four cases.
These all-entry diagnostic caps never replace the original gate.

At 168 bins the exact symmetric population-variance decomposition gives normal
occupancy +0.00044422, conditional-mean-squared +0.00352541 and conditional
width +0.00127732 MeV²; oblique −0.00043975, +0.00915189 and +0.00257562.
Survivor energy/width, not occupancy alone, therefore carries most of the
algebraic change. Conditional mean energies rise 0.22100→0.25393 MeV normal
and 0.37623→0.42220 oblique, despite opposite survivor-fraction shifts.
Components retain their covariance and outcome-selection caveats; they do not
identify an exclusive microscopic interaction or validate a correction.
