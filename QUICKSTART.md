# Quickstart

From the repository URL to your own job in about 10 minutes, on an endpoint you have: a Fermilab **EAF** server, a
Mu2e **GPVM**, or your **own Slurm cluster**. Pick one path below and copy its blocks in order.

**Before you start.** You need a Fermilab account in the Mu2e group and access to the input release
`mu2e-fastsim-inputs-v1.2` ([setup/README.md](setup/README.md) says how to request it). The access reply gives two
values; every path starts by exporting them with your output area:

```bash
export M2FS_INPUTS=<release root from the access reply>
export MANIFEST=<manifest path from the access reply>
export M2FS_DATA=<your Mu2e data area>/fastsim        # writable and with quota; not the app area, not $HOME
```

Timings below are from a clean-clone test (see the end of this page); `pip` dominates the first run.

## Path A: EAF notebook (Fermilab Elastic Analysis Facility)

Start a server at <https://analytics-hub.fnal.gov> (any profile; a GPU slice runs JAX on the GPU), open a terminal
(File > New > Terminal) and paste the three exports above, then:

```bash
mkdir -p "$M2FS_DATA" && cd "$M2FS_DATA/.."
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
source site/eaf.env && python3 site/check.py eaf
/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python -m venv --system-site-packages "$M2FS_VENV"
"$M2FS_VENV/bin/pip" install -q "jax==0.11.2" -e core -e crv -e tracker -e beam
source site/eaf.env                                   # again: puts the venv first on PATH
python3 -m mu2efastsim.inputs check-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" | grep -c '"ok"'
python3 -m mu2efastsim.demo
```

The clone and the venv live on Fermilab storage, so they outlive the server: in a new server only the three exports and
`source site/eaf.env` are needed. The same steps as a notebook: [notebooks/quickstart_eaf.ipynb](notebooks/quickstart_eaf.ipynb).
Then [run your own job](#run-your-own-job) (EAF column).

## Path B: Mu2e GPVM

`pyenv ana` has no JAX, which the beam stage needs, so the GPVM path uses a venv on top of the same Mu2e Python. Log
in to any `mu2egpvm0N`, paste the three exports, then:

```bash
mkdir -p "$M2FS_DATA" && cd "$M2FS_DATA/.."
git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
source site/gpvm.env && python3 site/check.py gpvm
/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python -m venv --system-site-packages "$M2FS_VENV"
"$M2FS_VENV/bin/pip" install -q "jax==0.11.2" -e core -e crv -e tracker -e beam
source site/gpvm.env                                  # again: puts the venv first on PATH
python3 -m mu2efastsim.inputs check-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" | grep -c '"ok"'
python3 -m mu2efastsim.demo
```

The venv is the one Path A makes, so a clone on Fermilab storage serves both. Keep to one job per GPVM. Then
[run your own job](#run-your-own-job) (GPVM column): on the GPVM itself, or on EAF driven from the GPVM.

## Path C: your own Slurm cluster

The kit runs one job per point on the cluster through `slurm_ssh` (hep-endpoints) and the project image; the job
streams its inputs from a Mu2e GPVM, so the inputs never stay on the cluster. Use a kit host with Python 3.10 or newer, git
and ssh access to both the cluster login host and a Mu2e GPVM (your laptop, with a Fermilab Kerberos ticket).

1. **The image**, once: on a machine with Apptainer and user namespaces (a cluster node, or a Mu2e GPVM), build it
   from this repository and copy it to `$FASTSIMKIT_SIF` on the cluster. It is not on CVMFS and not downloadable.

   ```bash
   git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
   container/build.sh "$PWD/mu2efastsim.sif"       # GPU variant, 3.5 GB; VARIANT=cpu for CPU nodes (container/README.md)
   ```

2. **The kit host**: clone, venv with the kit dependencies, the preset with your cluster account:

   ```bash
   git clone --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
   python3 -m venv .venv && .venv/bin/pip install -q -e core -e crv -e tracker -e beam
   export FASTSIMKIT_FNAL_HOST=mu2egpvm01 FASTSIMKIT_SLURM_HOST=<login host> \
          FASTSIMKIT_SLURM_ACCOUNT=<account> FASTSIMKIT_SCRATCH=<your work area on the cluster>
   source site/slurm.env && .venv/bin/python site/check.py slurm
   ```

   `M2FS_INPUTS` and `MANIFEST` are the paths on the Fermilab side (as the access reply gives them); `M2FS_DATA` is a
   writable directory on the kit host. Put the image at `$FASTSIMKIT_SIF` on the cluster.

3. Then [run your own job](#run-your-own-job) (Slurm column). The demo needs the release mounted, so on this path it
   runs on a GPVM or EAF.

## The demo

`python3 -m mu2efastsim.demo` runs three short chains on the CPU and prints one line per chain, the headline number
next to its reference value and tolerance (`--quick`: a quarter of each sample; `--only beam,ce,crv`). On a GPVM
(4-core AMD EPYC VM, JAX on the CPU) the three take about 6 minutes, wall time per chain including Python start-up,
JAX compilation and input reads: beam 108 s, CE 140 s, CRV 88 s. Most of it is fixed cost (the beam transport
compiles for about a minute, the CRV chain loads its library and tables for about a minute), so `--quick` saves less
than its sample sizes suggest. On a GPU server set `JAX_PLATFORMS=cuda` first.

| Chain | Sample | Headline | Reference |
|---|---|---|---|
| beam | 20,000 DS-entrance muons (MDC2025ac table of the release) | target stop fraction, nominal target | Geant4 0.3592 on the full 105,350 muons ([docs/validation.md](docs/validation.md)) |
| CE | 2,000 conversion electrons from that table's target stops | fraction reaching the tracker entry plane | fast chain on 40,000 electrons from the same stops, another seed: 0.7260 |
| CRV | 1,000 generated cosmic events, default chain (shard 11 of `g4ref_v1`) | coincidence clusters per generated event | stock chain (Geant4 + digi + reco) on all 50,000 events of that shard |

A line passes within 3 sigma of the combined counting errors. It also writes `$M2FS_DATA/demo/demo_summary.json` and a
log per chain. Exit 0: every line passes. It is a check of your setup, not a validation.

## Run your own job

One kit task with one changed parameter: `beam_point`, the stopping-target metrics of a target geometry, with the
49-foil design `beam/geom/move_extras12.txt` instead of the nominal 37 foils. The stop fraction should rise from about
0.36 to about 0.67 (docs/validation.md: 0.666 vs Geant4 0.669).

**EAF or GPVM, in place** (CPU; the command the `beam_point` task runs):

```bash
python3 -m beamfastsim point --g4 "$M2FS_INPUTS/beam_fastsim/g4ref_full/g4ref.npz" \
  --geom "$M2FS_CODE/beam/geom/move_extras12.txt" --n 20000 --seed 1 --out "$M2FS_DATA/my_point"
cat "$M2FS_DATA/my_point/metrics.json"
```

**Through the kit** (GPVM to EAF with `EAF_TOKEN_FILE` set to your own mode-600 hub token, made at
<https://analytics-hub.fnal.gov/hub/token>; or the Slurm path, where the kit picks a GPU and streams the muon table in):

```bash
python3 - <<'EOF'
import time
from mu2efastsim.endpoints.kit import FastsimKit
kit = FastsimKit()                                   # FASTSIMKIT_ENDPOINT from the preset: eaf (GPVM) or uva (Slurm)
params = {"task": "beam_point", "geom": "move_extras12.txt", "muons": 20000, "seed": 1, "repeat": 0}
print(kit.check("my_point", params, [], []))
h = kit.submit("my_point", params, [], [])["handle"]
while (s := kit.status(h))["state"] not in ("completed", "failed", "cancelled"):
    print(s["state"], s.get("message", "")); time.sleep(30)
r = kit.results(h)
print(r["metrics"]["stop_frac"], "+-", r["metrics"]["stop_frac_err"])
EOF
```

**Reading the output.** `metrics.json` (or `results()["metrics"]`) holds `stop_frac` and its binomial error
`stop_frac_err`, the `mu-` stop fraction, stops per foil (`stops_foil_mean`), the mean stop radius and the fraction of
stops in the upstream half; the kit also returns the job's log files. Other tasks and their parameters:
`python3 -c "from mu2efastsim.endpoints.kit import FastsimKit; import json; print(json.dumps(FastsimKit().describe(), indent=1))"`,
and [core/docs/running.md](core/docs/running.md).

## Clean-clone timings

To be filled by the clean-clone test.
