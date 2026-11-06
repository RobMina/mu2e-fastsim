# Opt-in slab bremsstrahlung source

`beamfastsim.flash_source_jax.slab_source` collects the core
`lane_step_photons` records into a fixed per-parent buffer on the device. It
requires a core checkout providing that API. It does not change the CE or beam
transport defaults.

```python
import jax
import numpy as np
from mu2efastsim import eslab_jax as slab, materials
from beamfastsim.flash_source_jax import slab_source, validate_complete

n = 16
tables = slab.lane_tables(
    [(materials.BY_NAME["G4_W"], 5.0, 0.0814923, 0.00770222, 0.5)], msc=False)
state = slab.lane_state(np.full(n, 105.0), np.tile([0., 0., 1.], (n, 1)),
                        np.zeros(n), np.full(n, 5.0))
result = slab_source(jax.random.PRNGKey(2), tables, np.zeros(n, np.int32),
                     state, capacity=64, max_steps=5000, msc=False)
validate_complete(result)
```

`records` contains energy, position, direction, path and validity arrays with
leading dimensions `(parents, capacity)`. Positions and directions use the slab
frame; energies are MeV and lengths mm. `state` is the terminal charged-particle
state; `count` includes emissions beyond capacity. `overflow` counts dropped
records and `capped` marks lanes still active when `max_steps` is exhausted.
`validate_complete` raises on either condition, rather than treating truncation
as zero response. Zero-event arms should be counted by the calling analysis;
the low-level function requires at least one parent lane.

This is a **bremsstrahlung birth source**, not target-exit photon flux. It lacks
finite-foil geometry, photon propagation/attenuation, secondary cascades and
annihilation photons. Positron transport requires positron tables and
`positron=True`; its annihilation photons are not part of this record.

## Timing and response boundaries

The production pileup detector-step selection uses the strict condition
`fmod(step_time, T) > 250 ns or fmod(step_time, T) < 10 ns`, with the configured
nominal DR period (1695 ns). This is an event selection through detector steps,
not a cut on DS-entrance source times. Do not reject late source particles or
replace `fmod` with a different folding convention. This slab primitive has no
time-selection or bunch-offset operation.

A response derived from retained detector-selected photon ancestry is conditional
on that selection. Dividing its deposit by its retained photon count does not
supply an unconditional response per incident photon. Such a conditional diagnostic
must not be reported as a validated absolute flash prediction.
