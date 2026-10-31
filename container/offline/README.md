# Offline on a Slurm CPU node without host CVMFS

`offline_uva.sh AREA_OR_TARBALL FCL [INPUT ...]` is an opt-in runner for an **already-built** Mu2e Offline area.
It mounts the signed OSG CVMFS repositories with cvmfsexec, uses a private node-local cache and removes it at exit.
By default it mounts the stock FNAL worker image directory from the `singularity` CVMFS repository and starts
unprivileged Apptainer inside that namespace; no image is built or copied. An explicit local EL9 SIF uses the
reverse nesting (Apptainer, then cvmfsexec). It never builds Offline, makes muse tarballs or installs system packages.
The Python fast-simulation image is not an Offline runtime. The runtime image must supply EL9 external libraries.

## Requirements

- A Slurm CPU allocation, Apptainer, unprivileged namespaces, `/dev/fuse`, and outbound HTTP to CVMFS Stratum-1s.
- The stock `/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-el9:latest` image directory (default),
  fetched on demand by CVMFS. Pin its `.images/sha256:...` target for reproducible campaigns. A local `OFFLINE_IMAGE`
  may be used only if it supplies the full EL9 runtime: the existing Celeritas Rocky9 SIF lacks libfreetype and
  is **not** a working Offline image. Rocky9 additionally needs the al9 Spack/UPS selection that the runner sets.
- A prepared cvmfsexec tree (`CVMFSEXEC_DIST`, with `dist/`); make it once on a compute node with host tools.
  For the default outer mount, the client must match the **host** (UVA standard hosts are EL8), not the EL9 payload:

  ```bash
  git clone https://github.com/cvmfs/cvmfsexec.git cvmfsexec
  cd cvmfsexec
  git rev-parse HEAD > ../cvmfsexec.commit
  ./makedist osg > ../makedist.log 2>&1
  ```

  For the alternate local EL9 SIF route, select `./makedist -m rhel9-x86_64 osg` because the client runs inside it.

  This downloads and unpacks client RPMs without installing them. `curl`, `rpm2cpio` and `cpio` are required.
  Do not put caches in the saved distribution. The runner copies it to local scratch and replaces its config.
- An existing muse area or a **trusted** `muse tarball` containing `Code/setup.sh`. Its backing remains on CVMFS.
- Staged input files and any non-CVMFS FCL/text conditions/table dependencies. The runner does not contact dCache,
  SAM or Data Dispatcher, and does not transfer credentials. A conditions DB configuration needs site QE egress;
  text conditions are an alternative, not a silent fallback.

## Example

Submit from the site login node with explicit account and partition. Set the paths for your site; no home or
personal data-area defaults are embedded in the runner.

```bash
# Default image is the stock CVMFS FNAL worker; optionally pin its .images/sha256 target.
unset OFFLINE_IMAGE
export CVMFSEXEC_DIST=/path/to/cvmfsexec
export OFFLINE_OUTPUT_FILE="$PWD/result.root"  # art TFileService output (-T)
export OFFLINE_BINDS=/path/to/tables          # additional same-path binds, comma-separated
sbatch -A YOUR_ACCOUNT -p standard --cpus-per-task=1 --mem=12G --time=00:30:00 \
  --wrap='module load apptainer; bash /path/to/offline_uva.sh /path/to/Code.tar.bz2 /path/to/job.fcl /path/to/input.art'
```

The working directory is the output directory. Inputs, FCL, the area and working directory are bound at their
original absolute paths. Include paths use `OFFLINE_FCL_PATH`; arbitrary table paths referenced by FCL need
`OFFLINE_BINDS`. The backing's `build/$MUSE_STUB` is appended to `FHICL_FILE_PATH` for generated trigger menus.
The runner checks `MUSE_WORK_DIR`, expands FCL before running, records `mu2e --version`, and propagates errors.

| Variable | Default | Purpose |
|---|---|---|
| `OFFLINE_NODE_TMP` | `/tmp` | Must resolve under `/tmp`, `/local` or `/localscratch`; private cache per invocation |
| `OFFLINE_CACHE_MB` | `6000` | CVMFS cache quota in MB; allocate memory too if local storage is tmpfs |
| `OFFLINE_CVMFS_PROXY` | `DIRECT` | Site squid if available; never assume a FNAL-only squid is reachable |
| `OFFLINE_CVMFS_SERVERS` | FNAL, then BNL | Signed CVMFS content from these Stratum-1s; domain-local setting |
| `OFFLINE_EVENTS` | all | Explicit event limit for smoke tests |
| `OFFLINE_LOG_PREFIX` | `offline` | Output filename prefix; `.expanded.fcl`, `.art.time`, `.total.time`, `.cache.txt` |
| `OFFLINE_WARM_REPEAT` | `0` | Set `1` to repeat art in the same mounted environment/cache, with a separate `.warm.root` and `.warm.art.time` |
| `OFFLINE_SEARCH_PATH` | unset | Directory prepended to `MU2E_SEARCH_PATH` (e.g. a private geometry or conditions copy); also bind it with `OFFLINE_BINDS` |

One invocation has one cache manager. Parallel invocations **must not share a cache**; the wrapper allocates a
unique local directory for each. Two concurrent art jobs on two cores are validated below; shared pilot/prewarming
and whole-node scaling are not certified. Use one core per art job unless a measured configuration explicitly uses
art multithreading.

## Probes and exact bridge comparison

`probe_r1.sbatch` prepares the client with UVA host tools and runs `probe_r1.sh` in an existing EL9 image. Supply
`OFFLINE_ROOT` as an output/snapshot directory and `OFFLINE_IMAGE`; an optional `CVMFSEXEC_DIST` reuses the distribution.
The probe mounts config-osg, mu2e and fermilab, sets up released SimJob MDC2025ba, runs version and FCL checks twice
against the same local cache, and reports cold/warm time. The release is an explicit test reference, not a default
for physics production.

`bridge_reference.sbatch` runs the fixed seed-20261023 A0 and A0g configurations on staged reference inputs.
Supply `OFFLINE_INPUT` (containing `gen.art` and `dts.art`), `OFFLINE_AREA` (the existing area/tarball),
`OFFLINE_G4_TABLE` and the same runtime variables. `compare_reference.sbatch` also takes `OFFLINE_BRIDGE` and
`OFFLINE_REFERENCE` (with `<arm>/valhist.<shard>.root`; `OFFLINE_SHARD` defaults to `001400_00000022`).
`parallel_reference.sbatch` runs two independent-cache A0 invocations in a two-core allocation. These are explicit
validation fixtures, not general data-discovery or production-submission tools.

```bash
python3 container/offline/compare_root.py reference.root candidate.root --output comparison.json
```

Compare histogram bins (including flow), variances, axes and scalar statistics, and all tree branches at the highest
ROOT cycle with zero numerical tolerance. ROOT UUIDs, compression and serialization timestamps are intentionally not
compared. Unsupported objects fail closed. Check the recorded event-tree counts against each arm's input selection,
not against the number of generator events in another arm.

## Validation status

On UVA standard CPU nodes the stock FNAL worker-image route reproduces a stock shard-22 reference exactly:
A0 (stock digi/reco, seed 20261023) has 2,475 event rows and 519 equal objects; A0g (injected G4 steps) has 50,000 event
rows and 280 equal objects. All histogram contents/statistics, binning labels and tree branches agree at zero tolerance;
ROOT serialization metadata is excluded. The unchanged DbService Sim_best v1_5 setup works from the compute node.
The runtime is SimJob MDC2025ba / Offline v13_41_01 with art 3.15.00, ROOT 6.38.00, KinKal 3.9.00.

Measured cold A0: 403.92 s total, 131.62 s art, 697 MB max art-process RSS. Cold A0g: 410.26 s total, 145.67 s art,
647 MB max art-process RSS, 990 MB cache. A same-environment/cache A0 repeat takes 78.98 s versus 113.22 s for its
cold art invocation (698 MB warm RSS, 1,005 MB final cache). Separate released version/FCL-only checks take 236.06 s
cold versus 19.03 s warm. These are one-shard observations, not production-runtime projections.
Two independent-cache A0 invocations pass simultaneously on one 2-core/16-GB allocation; aggregate Slurm max RSS
is 5.1 GB. This proves small-node concurrency, not full-node scaling. One core per art job is sufficient here.

CVMFS cache and tarball extraction are job-local and deleted at exit. User areas still need their existing builds;
inputs and non-CVMFS payloads still need staging; other conditions purposes/runs and scaled G4 production are not
certified by this bridge. The launcher and comparison tests run in the existing Python project image, not on login
cores. No simulation or reconstruction default changes.
