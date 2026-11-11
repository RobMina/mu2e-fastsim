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

A clean-clone test took 7.6 minutes on a GPVM, most of it the demo, and about 10 on an A100 Slurm node, most of it the
CE line on the GPU and the CPU (timings at the end of this page and under [The demo](#the-demo)).

## Path A: EAF notebook (Fermilab Elastic Analysis Facility)

Start a server at <https://analytics-hub.fnal.gov> (any profile; on a GPU slice the demo's CE line also runs on the
GPU), open a terminal (File > New > Terminal) and paste the three exports above, then:

```bash
mkdir -p "$M2FS_DATA" && cd "$M2FS_DATA/.."
git clone https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
source site/eaf.env && python3 site/check.py eaf
/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python -m venv --system-site-packages "$M2FS_VENV"
"$M2FS_VENV/bin/pip" install -q "jax[cuda12]==0.11.2" -e core -e crv -e tracker -e beam   # CUDA wheels for a GPU slice
source site/eaf.env                                   # again: puts the venv first on PATH
python3 -m mu2efastsim.inputs check-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" | grep -c '"ok"'
python3 -m mu2efastsim.demo
```

The clone and the venv live on Fermilab storage, so they outlive the server: in a new server only the three exports and
`source site/eaf.env` are needed. `jax[cuda12]` adds JAX's CUDA wheels (a few GB in the venv); on a CPU server, or a
GPVM using the same venv, JAX falls back to the CPU. The same steps as a notebook: [notebooks/quickstart_eaf.ipynb](notebooks/quickstart_eaf.ipynb).
Then [run your own job](#run-your-own-job).

## Path B: Mu2e GPVM

`pyenv ana` has no JAX, which the beam stage needs, so the GPVM path uses a venv on top of the same Mu2e Python. Log
in to any `mu2egpvm0N`, paste the three exports, then:

```bash
mkdir -p "$M2FS_DATA" && cd "$M2FS_DATA/.."
git clone https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
source site/gpvm.env && python3 site/check.py gpvm
/cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/python -m venv --system-site-packages "$M2FS_VENV"
"$M2FS_VENV/bin/pip" install -q "jax==0.11.2" -e core -e crv -e tracker -e beam
source site/gpvm.env                                  # again: puts the venv first on PATH
python3 -m mu2efastsim.inputs check-inputs --manifest "$MANIFEST" --root "$M2FS_INPUTS" | grep -c '"ok"'
python3 -m mu2efastsim.demo
```

The venv is the one Path A makes, so a clone on Fermilab storage serves both. Keep to one job per GPVM. Then
[run your own job](#run-your-own-job): on the GPVM itself, or on EAF driven from the GPVM.

## Path C: your own Slurm cluster

The kit runs one job per point on the cluster through `slurm_ssh` (hep-endpoints) and the project image; the job
streams its inputs from a Mu2e GPVM, so the inputs never stay on the cluster. Use a kit host with Python 3.10 or newer, git
and ssh access to both the cluster login host and a Mu2e GPVM (a workstation with a Fermilab Kerberos ticket). Both
ssh names must log in without a prompt, for the whole job: `ssh -o BatchMode=yes YOUR_GPVM true` and the same for
the login host should return at once (an ssh `ControlMaster` connection to the GPVM is the easy way).

1. **The image**, once: on a machine with Apptainer and user namespaces (a cluster node, or a Mu2e GPVM), build it
   from this repository and copy it to `$FASTSIMKIT_SIF` on the cluster. It is not on CVMFS and not downloadable.

   ```bash
   git clone https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
   container/build.sh "$PWD/mu2efastsim.sif"       # GPU variant, 3.5 GB; VARIANT=cpu for CPU nodes (container/README.md)
   ```

2. **The kit host**: clone, venv with the kit dependencies, the preset with your cluster account:

   ```bash
   git clone https://github.com/RobMina/mu2e-fastsim.git && cd mu2e-fastsim
   python3 -m venv .venv && source .venv/bin/activate     # python3 is now the venv's, as the kit block below needs
   pip install -q -e core -e crv -e tracker -e beam
   export FASTSIMKIT_FNAL_HOST=YOUR_GPVM FASTSIMKIT_SLURM_HOST=YOUR_LOGIN_HOST \
          FASTSIMKIT_SLURM_ACCOUNT=YOUR_ACCOUNT FASTSIMKIT_SCRATCH=YOUR_CLUSTER_WORK_AREA
   source site/slurm.env && python3 site/check.py slurm
   ```

   `M2FS_INPUTS` and `MANIFEST` are the paths on the Fermilab side (as the access reply gives them); `M2FS_DATA` is a
   writable directory on the kit host. Put the image at `$FASTSIMKIT_SIF` on the cluster. Job records, results and
   logs land under `$FASTSIMKIT_STATE_DIR` on the kit host (default `~/.mu2efastsim/kit`).

3. Then [run your own job](#run-your-own-job), the "through the kit" block, from this shell. The demo needs the release
   mounted, so on this path it runs on a GPVM or EAF.

Tested end to end from a workstation (kit host) to a Slurm cluster, with the 49-foil `beam_point` of the next section:
check and submit 18 s (code snapshot on the cluster, image and input checksums); queue 212 s (a maintenance
reservation; yours will differ); job 139 s on an A100-SXM4-80GB, of which 15 s for the muon table to stream in from the
GPVM and 88 s for the transport of 20,000 muons including JAX compilation. It prints `completed` and
`0.66685 +- 0.0033`. On a fresh kit host, the venv and pip take about 3 minutes.

## The demo

`python3 -m mu2efastsim.demo` runs three chains and prints one line per chain run, the headline number next to its
reference value and tolerance (`--quick`: a quarter of each sample; `--only beam,ce,crv`). The **CE line** runs the
conversion-electron device chain (`beamfastsim.cechain --backend device`, the whole chain in one JAX loop) and shows
what a GPU does:

- when JAX sees a GPU, 2 x 1,048,576 electrons there (a power of two less on a GPU with fewer streaming
  multiprocessors than an A100, such as a MIG slice), then 2 x 4,096 on the CPU; without one (a GPVM), the CPU part
  only;
- the first batch pays the compilation, so each part prints its setup (compilation and maps: first batch minus
  second) and its warm rate (electrons per second of the second batch), and the demo ends with the GPU / CPU ratio
  of the warm rates;
- float64 by default, `--float32` for the CE chain; `--device cpu` skips the GPU part, `--device gpu` requires it.

The beam and CRV lines are short and run on the CPU; most of their time is fixed cost (the beam transport compiles for
about a minute, the CRV chain loads its library and tables for about a minute), so a GPU would not shorten them and
`--quick` saves less than its sample sizes suggest.

Clean clone, cold compilation cache, float64 unless noted; the CPU part runs on the same node, with 4 CPU threads:

| Where | CE warm rate on the GPU (setup) | on the CPU (setup) | GPU / CPU | CE line | whole demo |
|---|---:|---:|---:|---:|---:|
| A100-SXM4-80GB | 18,540 electrons/s (120 s) | 97/s (74 s) | **190x** | 254 + 162 s | 592 s |
| A100-SXM4-80GB, `--only ce --float32` | 27,980/s (95 s) | 143/s (73 s) | **196x** | 197 + 135 s | 340 s |
| RTX A6000 (float64 at 1/32 of the float32 rate) | 4,520/s (143 s) | 64/s (96 s) | 70x | 393 + 229 s | 844 s |
| GPVM (4-core AMD EPYC VM), no GPU | - | 96/s (58 s) | - | 146 s | 336 s |
| CPU node (Xeon Gold 6248), no GPU | - | 68/s (88 s) | - | 215 s | 430 s |

On an A100 the CE line carries 2.1 M electrons in about four minutes, compilation included; the CPU part would need
about six hours for the same sample, and a stock CE production job (Geant4 CeEndpoint) runs about 24 electrons per
second per grid slot ([docs/performance.md](docs/performance.md)). Peak memory per chain on a CPU node: beam 1.3 GB,
CE 2.2 GB, CRV 5.1 GB. On a busy GPVM with less than about 5 GB free the CRV line is killed (`crv FAILED (exit -9)`);
that happened to the first GPVM attempt, and the GPVM row is the rerun in the same clone. Use a quieter GPVM, or
`--only beam,ce`.

| Chain | Sample | Headline | Reference |
|---|---|---|---|
| beam | 20,000 DS-entrance muons (MDC2025ac table of the release) | target stop fraction, nominal target | Geant4 0.3604 +- 0.0015 on the full 105,350 muons, clean stock rerun ([docs/validation.md](docs/validation.md)) |
| CE | conversion electrons from that table's target stops, device chain: 2,097,152 on an A100, 8,192 on the CPU | fraction reaching the tracker entry plane | host chain on 40,000 electrons from the same stops, another seed: 0.7260 +- 0.0022 (the device chain matches the host chain: yield ratio 0.9999 +- 0.0007) |
| CRV | 1,000 generated cosmic events, default chain (shard 11 of `g4ref_v1`) | coincidence clusters per generated event | stock chain (Geant4 + digi + reco) on all 50,000 events of that shard: 0.0582 +- 0.0011 |

A line passes within 3 sigma of the combined counting errors. It also writes `$M2FS_DATA/demo/demo_summary.json` (with
the setup, warm rate and device of each CE part, and the ratio) and a log per run. Exit 0: every line passes. It is a
check of your setup, not a validation.

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
<https://analytics-hub.fnal.gov/hub/token>; or the Slurm path, where the kit picks a GPU and streams the muon table in;
run it where the preset is sourced and `python3` is the venv's):

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

Each path was followed verbatim from a fresh clone in a new shell (GPVM: `env -i` plus a login shell; EAF: a newly
started server, default profile). Wall times:

| Step | GPVM (4-core VM) | EAF (default server, 2 CPUs) |
|---|---:|---:|
| server start | - | 73 s |
| clone | 2 s | 4 s |
| `site/check.py` | 0 s | 0 s |
| venv on the Mu2e ana Python | 4 s | 19 s |
| `pip install` (JAX and the four packages) | 19 s | 39 s (CPU `jax`) |
| `check-inputs` (78 files) | 17 s | 7 s |
| demo (all lines ok; the GPVM run of the table under [The demo](#the-demo)) | 336 s | 434 s (the earlier demo, three CPU chains) |
| your own job (`beam_point`, 49 foils: stop fraction 0.6669 +- 0.0033) | 76 s | 144 s |
| **total** | **7.6 min** | **10.8 min** (12 min with the server start) |

The demo is most of it, and most of the demo is fixed cost (JAX compilation, table loading); skip it, or use `--quick`,
when you only need your own job. The EAF column and the GPVM "your own job" row predate the GPU demo and the
`jax[cuda12]` install; on a GPU slice the CE line runs with a batch scaled to the slice's share of an A100 (a 1g.10gb
slice: 2 x 131,072 electrons). On the Slurm path the image build from a fresh clone took 87 s (CPU variant); there the
demo runs in the image (GPU variant), 592 s on an A100 node.
