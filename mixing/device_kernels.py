"""Lift closed-over JAX condition arrays into explicit resident kernel operands."""
from types import SimpleNamespace


def lift(kernel, *arguments, **keywords):
    """Trace a shared kernel once at setup; call its unchanged JAXPR with asset inputs.

    Tracing the unwrapped function exposes captured arrays as JAXPR constants.
    Leaving those as GPU closures makes older JAX lowerers download them while
    compiling the outer graph. Supplying them as operands avoids that transfer
    without copying or patching the shared physics implementation.
    """
    import jax
    import numpy as np

    function = getattr(kernel, "__wrapped__", kernel)
    if getattr(kernel, "jitted", None) is not None:
        # Core float32 precision-aware jit: keep its float64->float32 argument cast and creator scope when tracing
        # the unwrapped function, so the lifted program computes exactly as the wrapped kernel does.
        from mu2efastsim import precision
        raw = function

        def function(*args, **kwargs):
            precision._SCOPE.depth = getattr(precision._SCOPE, "depth", 0) + 1
            try:
                return raw(*precision.dev(args), **precision.dev(kwargs))
            finally:
                precision._SCOPE.depth -= 1
    program, shape = jax.make_jaxpr(function, return_shape=True)(*arguments, **keywords)
    _, inputs = jax.tree_util.tree_flatten((arguments, keywords))
    _, outputs = jax.tree_util.tree_flatten(shape)
    constants = tuple(np.array(value, copy=True) for value in jax.device_get(program.consts))
    for value in constants:
        value.setflags(write=False)
    jaxpr = program.jaxpr

    def apply(assets, *args, **kwargs):
        values, structure = jax.tree_util.tree_flatten((args, kwargs))
        if structure != inputs:
            raise ValueError("lifted kernel argument structure differs from setup trace")
        result = jax.core.eval_jaxpr(jaxpr, assets, *values)
        return jax.tree_util.tree_unflatten(outputs, result)

    return SimpleNamespace(assets=constants, apply=apply)
