# mu2e-fastsim container (R4)

One Apptainer image for CPU and NVIDIA GPU: Python 3.13, JAX 0.11.2 with the pip CUDA 12 wheels (the same JAX as the
Polaris venv), the three packages installed from git-tracked sources, and the two public Offline files the code reads
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
- Polaris: `crv_reco_bench.sh` runs from the Eagle venv; the image runs under `apptainer` on compute nodes (module
  `spack-pe-base apptainer` or `/usr/bin/apptainer`, to be confirmed by `core/iri/scripts/sif_check.sh`).
