"""Importing the digitiser modules creates no JAX backend (their backend-dependent sizes resolve on first use), so a
workflow's mu2efastsim.devloops.select made after those imports still takes effect; the sizes keep their values."""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("jax")

CODE = """
import json
from mu2efastsim.determinism import _backend_started
from trkfastsim.digi import digi_jax as DJ, digi_clusters_jax as DC, digi_resident as DR
before = _backend_started()
from mu2efastsim import devloops
devloops.select("W2")
late = devloops.state()["late"]
sizes = [DJ.GPU, DJ.BATCH, DJ.WAVEFORM_CLUSTER_LANES, DJ.SATURATION_TILE, DJ.CHARGE_BLOCKS, DC.BATCH_SIZE]
print(json.dumps({"before": before, "late": late, "after": _backend_started(), "sizes": sizes}))
"""


def test_digitiser_import_starts_no_backend():
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["before"] is False and out["late"] is False and out["after"] is True
    assert out["sizes"] == [False, 32, 2048, 16, 1, 128]          # the CPU values


def test_unknown_attribute_still_raises():
    from trkfastsim.digi import digi_jax as DJ, digi_clusters_jax as DC
    with pytest.raises(AttributeError):
        DJ.NOT_A_SIZE
    with pytest.raises(AttributeError):
        DC.NOT_A_SIZE
