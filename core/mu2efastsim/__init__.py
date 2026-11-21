"""mu2efastsim -- detector-independent core of the Mu2e Geant4-free fast simulation (extracted / KPP configuration).

Shared by the CRV (crvfastsim) and tracker (trkfastsim) stages, which sit beside it in the mu2e-fastsim
repository. Definitions live once in these modules, documented in ``README.md``:

    eloss      muon energy loss in thin layers matched to Geant4 11.3.2 (restricted Bethe, Urban fluctuations, Birks,
               delta rays, radiation length, Highland angle)
    materials  material constants matched to what Geant4 uses (GDML composition, density-effect, fluctuation model)
    world      passive world material from a Geant4 GDML export, ray-traced as convex pieces
    crossings  vectorised ray-box crossings of straight tracks (JAX on CPU/GPU, else numpy)
    transport  detector-independent transport pieces: correlated multiple scattering to a plane, sliced slab loss
    tables     flat-table I/O in the art dumper layout (TFileService directory dump/), keyed by (run, subrun, event)
    stats      robust statistics shared by every comparison (median, core sigma, MPV, binned ratio band)
    plots      shared plotting helpers + write_json

Importing the package applies the GPU determinism setting (determinism.py, MU2EFS_DEVICE_DETERMINISTIC) and the
command-buffer settings (devloops.py: device loops, MU2EFS_DEVICE_LOOPS, and concurrent graph regions,
MU2EFS_GRAPH_CONCURRENCY; only forced values at import, and a workflow's entry point then picks the faster setting of
each with devloops.select) before any JAX backend exists.
"""
import os as _os

# the physics kernels live in hep-simkernels (mu2efastsim.eloss, .materials, .precision, ... are its modules): keep the
# MU2EFS_ environment variable names (MU2EFS_DEVICE_FLOAT, MU2EFS_DEVICE_DETERMINISTIC, MU2EFS_SLAB_KERNEL, ...)
_os.environ.setdefault("HEPSIMKERNELS_ENV_PREFIX", "MU2EFS")

from . import determinism as _determinism  # noqa: E402
from . import devloops as _devloops  # noqa: E402

__version__ = "0.1.0"
_determinism.apply()
_devloops.apply()


def _install_kernel_hooks():
    """Mu2e's pieces of the kernels, through hepsimkernels.settings (standard library only: nothing here imports NumPy
    or fixes the device float): the Mu2e GDML materials (mu2efastsim.materials, imported with
    hepsimkernels.materials), CONCRETE_MARS as the material of world pieces without constants (transport.world_loss),
    the per-map field scale of read_g4bl (bfield_scale.factor) and the device-loop settings in precision.provenance()."""
    import hepsimkernels.settings as kernel_settings

    from . import bfield_scale

    if "mu2efastsim.materials" not in kernel_settings.MATERIAL_MODULES:
        kernel_settings.MATERIAL_MODULES.append("mu2efastsim.materials")
    if kernel_settings.FALLBACK_MATERIAL is None:
        kernel_settings.FALLBACK_MATERIAL = "CONCRETE_MARS"
    kernel_settings.MAP_SCALE = bfield_scale.factor
    if _devloops_provenance not in kernel_settings.PROVENANCE_HOOKS:
        kernel_settings.PROVENANCE_HOOKS.append(_devloops_provenance)


def _devloops_provenance():
    return {"device_loops": _devloops.state(), "graph_concurrency": _devloops.concurrency_state()}


_install_kernel_hooks()
