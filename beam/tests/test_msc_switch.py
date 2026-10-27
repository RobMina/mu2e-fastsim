"""Disabling scattering also disables its opt-in effective layer-path correction."""
import numpy as np

from beamfastsim import foilfast, transport


def test_no_msc_has_geometric_path_and_identical_response():
    common = dict(physics="g4", msc=False, table="slab")
    a = foilfast.respond([1.4], [0.8], .1056, transport.params(**common, msc_path=True),
                         replicas=256, seed=3, raw=True)
    b = foilfast.respond([1.4], [0.8], .1056, transport.params(**common, msc_path=False),
                         replicas=256, seed=3, raw=True)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert np.all(a[2] == .1056 / .8)
