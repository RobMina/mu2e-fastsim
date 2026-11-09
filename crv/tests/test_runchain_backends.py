"""runchain's auto backends: device transport and tier 1b with --jax on the extracted configuration, NumPy otherwise."""
from types import SimpleNamespace

from crvfastsim import runconfig
from crvfastsim.analyses.runchain import resolve_backends

EXT = SimpleNamespace(name=runconfig.DEFAULT)
V5, CUSTOM = SimpleNamespace(name="strict_v5"), SimpleNamespace(name=None)


def test_auto_is_device_with_jax_on_extracted():
    assert resolve_backends(True, "auto", "auto", EXT, V5, None, {}) == ("jax", "jax")
    assert resolve_backends(False, "auto", "auto", EXT, V5, None, {}) == ("numpy", "numpy")


def test_auto_falls_back_where_the_device_twin_does_not_apply():
    assert resolve_backends(True, "auto", "auto", SimpleNamespace(name="run1a_v01"), CUSTOM, None, {}) == ("numpy", "numpy")
    assert resolve_backends(True, "auto", "auto", EXT, CUSTOM, None, {}) == ("jax", "numpy")
    assert resolve_backends(True, "auto", "auto", EXT, V5, object(), {}) == ("numpy", "jax")
    assert resolve_backends(True, "auto", "auto", EXT, V5, None, {"prefetch": True}) == ("jax", "numpy")


def test_explicit_choices_are_kept():
    assert resolve_backends(True, "numpy", "numpy", EXT, V5, None, {}) == ("numpy", "numpy")
    assert resolve_backends(False, "jax", "jax", EXT, V5, None, {}) == ("jax", "jax")
