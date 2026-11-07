"""crvfastsim.compile_cache: off by default; with a directory a second process reads the first one's executables."""
import json
import subprocess
import sys

import pytest

pytest.importorskip("jax")

from crvfastsim import compile_cache

PROBE = """
import json, sys
import jax, jax.numpy as jnp
from crvfastsim import compile_cache
hits = []
jax.monitoring.register_event_listener(lambda event, *a, **kw: hits.append(event))
path = compile_cache.enable(sys.argv[1])
y = jax.jit(lambda x: jnp.sin(x) * 3.0 + x.sum())(jnp.arange(17.0)).block_until_ready()
print(json.dumps({"path": path, "hit": "/jax/compilation_cache/cache_hits" in hits, "y": float(y[3])}))
"""


def test_no_directory_changes_nothing():
    import jax
    before = compile_cache.describe()
    assert compile_cache.enable(None) is None and compile_cache.enable("") is None
    assert compile_cache.describe() == before
    assert jax.config.jax_compilation_cache_dir == before["dir"]


def test_second_process_hits_the_cache(tmp_path):
    cache = tmp_path / "cache"
    run = lambda: json.loads(subprocess.run([sys.executable, "-c", PROBE, str(cache)], check=True,
                                            capture_output=True, text=True).stdout.strip().splitlines()[-1])
    first, second = run(), run()
    assert first["path"] == str(cache.resolve()) and any(cache.iterdir())
    assert not first["hit"] and second["hit"]
    assert first["y"] == second["y"]
