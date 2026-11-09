"""Opt-in JAX persistent compilation cache for the CRV chain drivers (runchain, veto.primary_chain).

A fresh process compiles every device kernel again (the photon, SiPM, waveform, pulse and coincidence kernels, one
executable per padded bucket shape). With a cache directory, XLA executables are written there on first use and read
back by later processes with the same JAX/jaxlib, device model, kernel and shapes, so only the first job of a campaign
pays the compile. The executables are the same, so outputs are unchanged.

JAX caches only compiles slower than one second by default; most chain kernels compile faster than that, so this
lowers the threshold to zero and caches entries of any size. Nothing changes unless a directory is given
(``--compile-cache DIR`` on the drivers, or ``enable(DIR)``).
"""
from pathlib import Path


def enable(directory):
    """Turn on the persistent compilation cache in ``directory`` (created if missing). Returns the resolved path,
    or None (and changes nothing) for an empty argument. Call before the first kernel compiles."""
    if not directory:
        return None
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    import jax
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    return str(path)


def describe():
    """The cache settings in effect (provenance for the stats JSON)."""
    import jax
    return {"dir": jax.config.jax_compilation_cache_dir,
            "min_compile_time_s": jax.config.jax_persistent_cache_min_compile_time_secs,
            "min_entry_size_bytes": jax.config.jax_persistent_cache_min_entry_size_bytes}
