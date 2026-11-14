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
from . import determinism as _determinism
from . import devloops as _devloops

__version__ = "0.1.0"
_determinism.apply()
_devloops.apply()
