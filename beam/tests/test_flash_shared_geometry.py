"""Both target navigators consume the same geometry implementations."""
import pytest

pytest.importorskip("jax")

from beamfastsim import cechain_jax as CE
from beamfastsim import flash_navigator_jax as FL
from beamfastsim import target_geometry_jax as TG


def test_both_navigators_use_identical_helpers():
    for public, shared in [('layer_types', 'layer_types'), ('device_maps', 'device_maps'),
                           ('_field', 'field'), ('_mom', 'momentum'), ('_beta', 'beta'),
                           ('_dot', 'dot'), ('gather', 'gather')]:
        assert getattr(CE, public) is getattr(FL, public) is getattr(TG, shared)
    assert CE.NCOUNT == FL.NCOUNT == TG.NCOUNT == 4
