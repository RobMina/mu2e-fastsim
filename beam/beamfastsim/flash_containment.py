"""Flash-only host classification in admitted nonoverlapping finite foil annuli.

Shared by charged-source initialization and photon postprocessing, not CE. This
is containment only; it does not add charged lateral foil-boundary navigation.
"""
import numpy as np


def containing_foil(beam, position):
    """Return the containing foil index, or -1 outside all closed annuli.

    Geometry must have nonempty, nonoverlapping foils, as required by the flash
    geometry loader. Unequal thicknesses are supported; centre proximity is not
    a containment test. Faces, the outer rim and the hole boundary are included,
    matching ``foil_exit`` without rounding positions or expanding material.
    Operations are vectorized over positions and foils, including empty inputs.
    """
    xyz = np.asarray(position, dtype=float)
    radius = np.hypot(xyz[..., 0] - beam.axis_x, xyz[..., 1])
    inside = ((np.abs(xyz[..., 2, None] - beam.foil_z) <= beam.foil_ht)
              & (radius[..., None] >= beam.foil_rin)
              & (radius[..., None] <= beam.foil_rout))
    return np.where(np.any(inside, axis=-1), np.argmax(inside, axis=-1), -1)
