# Flash photon response library

`beamfastsim.flash_response_library` is the flash response library of the shared core
(`mu2efastsim.library`, [core/docs/libraries.md](../../core/docs/libraries.md) section 8). It tabulates the
expected tracker ionising deposit per photon that starts inside a source volume around the stopping target. It is
built from stock Mu2e simulation: a photon gun in that volume, the standard detector transport, and the stock
`EdepAna` analyzer. `EdepAna` records each photon's start position and momentum and the summed StrawGasStep
ionising energy, with no time cut.

## Keys, outputs and domain

| Axis | Unit | Definition | Out of domain |
|---|---|---|---|
| `species` | 1 | categorical, photon only | refused |
| `energy` | MeV | photon energy, log bins | below the lowest edge: 0 (declared fallback, counted); above: refused |
| `z`, `r` | mm | start position in the source volume, Mu2e frame, r about the DS axis (x = -3904 mm) | refused |
| `cos_theta` | 1 | direction cosine to +z | clipped into [-1, 1) |
| `u_r` | 1 | radial direction cosine about the DS axis | clipped into [-1, 1) |

Outputs per cell are `mean` (MeV per photon), `variance` (per photon), `n` (photons) and `n_nonzero` (photons
with a nonzero deposit). Deposits are rare, so a cell's `n_nonzero` says how far its mean can be trusted. Lookups
use the nearest cell. The manifest records two assumptions, both untested: azimuthal symmetry about the DS axis, and
the target configuration inside the source volume.

## Use

```python
from beamfastsim import flash_response_library as F
lib = F.load("flash_response.npz")
mean = F.lookup_numpy(lib, energy, z, r, cos_theta, u_r)       # host, float64; refuses out-of-domain rows
counts = F.domain_counts(lib, energy, z, r, cos_theta, u_r)    # DomainCounter: fallbacks and refusals
dl = F.to_device(lib, np.float32)                               # or np.float64
mean_dev = F.lookup(dl, energy, z, r, cos_theta, u_r)           # jit-able; refused rows NaN
```

`python -m beamfastsim.flash_response_library build --config campaign.json --out flash_response.npz` builds a
library from a campaign configuration with `store.save`. The configuration names the EdepAna files and gives, for
each, its direction class, job index, seed and event count. It also holds the bin edges, the sampling density per
direction class, and the generator, geometry and configuration hashes. Each file must hold exactly its declared
number of events. The even/odd job split is recorded for the realization spread.
