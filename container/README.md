# mu2e-fastsim container (R4)

Two variants of one Apptainer definition (`VARIANT=gpu|cpu container/build.sh`): Python 3.13, JAX 0.11.2 (the same
JAX as the Polaris venv); `gpu` with the pip CUDA 12 wheels minus the ones no kernel loads (NCCL, NVSHMEM: `TRIM`,
override with `TRIM=""`; cuDNN and cuFFT stay, the JAX CUDA plugin will not start without them), `cpu` without CUDA (FermiGrid,
local, CI, CPU endpoints; 234 MB); the three packages installed from git-tracked sources, and the two public Offline files the code reads
(`singlePEWaveform_v3.txt`, `crv_counters_extracted_v04.txt`, Offline v13_39_00) under `/opt/offline_data`
(`MU2E_SEARCH_PATH`). No CVMFS. No Mu2e-internal data: inputs and outputs are bind-mounted at run time.

```bash
container/build.sh /path/to/mu2efastsim.sif          # on a machine with apptainer + user namespaces (Mu2e GPVM)
apptainer exec --cleanenv mu2efastsim.sif python -m pytest -q /opt/mu2e-fastsim/crv/tests
apptainer exec --nv mu2efastsim.sif crvfastsim reco-bench synthetic --stage chain --max-events 200000   # GPU node
```

- `build.sh` exports the umbrella and submodules with `git archive` at their checked-out commits (recorded in
  `/opt/mu2e-fastsim/VERSION`), so nothing untracked (site.env, work areas, build outputs, data) can enter the image.
- `fetch_offline_data.sh <dest>` is also usable on its own on a site without CVMFS.
- Polaris compute nodes: `module use /soft/spack/base/0.8.1/install/modulefiles/Core; module load apptainer`
  (1.3.2; the 1.4.1 module under spack-pe-base 0.10.x does not put apptainer on PATH). `core/iri/scripts/sif_check.sh`
  builds the variants on the node (~2 min each, fakeroot works), times a cold start and runs the GPU tests and the
  chain bench inside each image.
- Size: the untrimmed GPU image is 3.5 GB (5.6 GB unpacked, 4.8 GB of it CUDA wheels; cuDNN alone 1.2 GB). R4 asks for
  "the fast-sim image + < 0.5 GB": the reco adds ~1 MB of code and data to the shared JAX/CUDA stack.

## Celeritas image (`celeritas.def`, `build_celeritas.sh`)

A second, separate image for the GPU electromagnetic transport feasibility study of the Mu2eBO `elebeam_flash` stage:
Celeritas 0.7.0 (CUDA, sm_80), Geant4 11.3.2, VecGeom 2.2.0 (+gdml, CUDA), G4VG and covfie, built with spack on
`nvidia/cuda:12.6.3-devel-rockylinux9` along the lines of Celeritas' own `scripts/docker/dev/rocky-cuda12.yaml`. It
provides `celer-g4` (Celeritas' standalone Geant4 application) to run the Mu2e GDML export with the DS field map.
Built by the user on a UVA node (1-3 h, needs `--fakeroot` and outbound internet):

```bash
module load apptainer
JOBS=32 container/build_celeritas.sh /scratch/$USER/mu2efastsim/mu2e-celeritas.sif
```

