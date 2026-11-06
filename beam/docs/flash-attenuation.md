# Aluminium: original-photon uncollided attenuation

`beamfastsim.flash_attenuation` supplies an **untuned survival probability for
no interaction**, not the total outgoing photon flux. Every interaction removes
the original photon from this component, including coherent scattering. There is
no scattering redistribution, fluorescence, secondary production, buildup,
electron transport, or deposited-energy calculation.

## API and units

```python
from beamfastsim.flash_attenuation import al_mu_rho, uncollided_weight
mu = al_mu_rho(1.0)                 # MeV -> cm²/g (zero-dimensional ndarray)
r = uncollided_weight(1.0, 10.0)    # 10 mm = 1 cm, density defaults to 2.699 g/cm³
# r['weight'] ~ 0.847, r['supported'] == True
```

`al_mu_rho(energy)` returns an ndarray with the input shape. It interpolates
**log(mu/rho) against log(energy)** between tabulated nodes, rather than
reimplementing XCOM's process-specific interpolation. The table covers
**0.001–100000 MeV (1 keV–100 GeV), inclusive**. There is no extrapolation or
clamping. Nonfinite or out-of-range energies yield NaN. In particular, the
current bremsstrahlung cutoff of **0.00099 MeV is unsupported**, not mapped to
1 keV.

The aluminium K-edge is represented by separate nodes at **1.5595 keV**
(pre-edge) and **1.5596 keV** (post-edge). Both endpoints are supported, but the
open interval between them returns NaN. This conservative explicit gap avoids
inventing a smooth transition across the absorption discontinuity.

`uncollided_weight(energy, distance_mm, density_g_cm3=2.699)` broadcasts its
inputs and returns a dictionary of ndarrays, `weight` and boolean `supported`:

\[
w = \exp[-(\mu/\rho)(E)\,\rho\,d_{\rm mm}/10].
\]

Lengths and densities must be finite and nonnegative; invalid values produce
NaN/False. Zero length or zero density gives unity **only for supported energy**.
Underflow to zero at very large optical depth is valid. Use the supported mask
explicitly; do not replace unsupported values with transparent propagation or
silently omit their photon population. Distance is the supplied path through
homogeneous aluminium, not a projected thickness; this module performs no
geometry navigation. Other materials and variable-density paths are not modeled.

## Data provenance and deterministic regeneration

The checked-in `beamfastsim/data/flash_al_xcom.csv` has 82 nodes derived from
NIST XCOM v3.1 `MDATX3.013`. Its header identifies the source archive. The
tracked parser sums the **coherent + incoherent + photoelectric + nuclear-field
pair + electron-field pair** cross sections in barns/atom, then converts with
**0.60221367 / 26.981541** to cm²/g, exactly the XCOM conversion convention.
Source energies in eV are divided by 1e6. Trailing subshell interpolation blocks
are not needed for the tabulated-node totals. No compilation or fitted parameters
are involved; the supplied local source is read without network access.

From the `beam` directory, with NumPy available:

```sh
PYTHONPATH=. python3 -m beamfastsim.flash_attenuation \
  --xcom-data /path/to/XCOM/MDATX3.013 \
  --out beamfastsim/data/flash_al_xcom.csv
```

The table is loaded relative to the module's filesystem path, so source-tree
usage does not require package installation. Distributions must include the CSV.
Tests cover every table node, the K-edge gap, log-log interpolation, mm/cm
conversion, invalid/unsupported inputs, zero paths, broadcasting, and parser
reproducibility. Independent public-table anchors at 0.1, 1, and 20 MeV are
0.1704, 0.06146, and 0.02168 cm²/g; the relative tolerance is 3e-4 to allow
published precision and channel rounding. These are **total attenuation**
coefficients, not the separate mass energy-absorption coefficients.

### References

- Berger, Hubbell et al., *XCOM: Photon Cross Sections Database*, NIST Standard
  Reference Database 8: <https://physics.nist.gov/PhysRefData/Xcom/Text/intro.html>.
- NIST source archive (Readme.txt, MDATX3.013, XCOM.f):
  <https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz>. XCOM.f lines 130–165
  define the input blocks; lines 240–255 apply the atomic-weight conversion;
  `AVOG = 0.60221367` is defined at line 36.
- NIST public aluminium mass attenuation table:
  <https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html>.
