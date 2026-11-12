# Parametric flash estimate (ranking surrogate)

`beamfastsim.flash_param` estimates the stock flash objective (tracker ionising
deposit per generated beam electron) from the stopping-target foil arrays alone:

```
flash = c0 + c1 * sum_i t_i * [exp(-rin_i^2 / 2 sigma^2) - exp(-rout_i^2 / 2 sigma^2)] * exp((z_i - z_ref) / lam)
```

The sum is the aluminium thickness seen by a Gaussian beam footprint of width
`sigma`, with material nearer the tracker weighted up by the e-folding length
`lam`. `c0` holds the part of the deposit that does not scale with target
material (photons born outside the target, non-photon contributions). The four
constants are global: nothing is fitted per design.

It is an opt-in surrogate for ranking designs and screening a flash budget. It
does no transport and is **not** the physics-validated flash replacement. The
building blocks in [flash-source.md](flash-source.md),
[flash-flux.md](flash-flux.md) and [flash-inclusive.md](flash-inclusive.md) are
the route towards that. Nothing in the kit uses it by default.

## Why this form

Geant4 ancestry of the stock objective attributes most of the deposit, and most
of its design dependence, to photons born in the target foils, which reach the
tracker and make Compton and photoelectrons in the straws. Photon production is
roughly proportional to the material the beam e+- cross, and the tracker
acceptance grows as the emission point moves towards it. The two shape
parameters encode those two facts, and the constant term encodes the rest.

## Supported domain

Coaxial, untilted aluminium annular foils, read with the checks of
`flash_geometry.from_offline`, on a fixed external geometry (absorbers, field,
detectors). Changes outside the foil arrays are outside the domain even if the
formula returns a number. Calibrate on stock evaluations of the same external
geometry and release. Report the out-of-fold ranking test with the constants.

## Use

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.flash_param \
  --table designs.tsv --offline-root /path/to/geometry --kfold 5 --out fit.json
PYTHONPATH=core:beam python3 -m beamfastsim.flash_param \
  --table new.tsv --offline-root /path/to/geometry --params params.json --out pred.json
```

`designs.tsv` has the columns `name`, `geom` (overlay path relative to the
geometry root) and, for calibration or comparison, `flash`. With a `flash`
column and no `--params`, the constants are calibrated on the table (grid over
`sigma` and `lam`, relative least squares for `c0` and `c1`), and `--kfold`
reports out-of-fold Spearman, Pearson, relative bias and RMS, and agreement of
the budget decision. With `--params` (the `params` object of an earlier output)
it predicts and, when `flash` is present, compares. `--float32` computes the
feature in float32; the two-column solve is float64 in both modes.
