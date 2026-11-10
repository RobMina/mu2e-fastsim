"""Demo: three fast chains on the input release, each headline number next to its reference value, and the CE device
chain timed on the GPU against the CPU.

    python3 -m mu2efastsim.demo [--quick] [--out DIR] [--only beam,ce,crv] [--device auto|gpu|cpu] [--float32]

Needs the site variables of a preset (site/*.env in the umbrella): M2FS_INPUTS (the input release), M2FS_DATA
(outputs; default DIR is $M2FS_DATA/demo), and core, crv and beam on PYTHONPATH. Each chain is the existing entry point,
run as a subprocess:

  beam  python3 -m beamfastsim point     20,000 DS-entrance muons (MDC2025ac) -> target stop fraction (CPU)
  ce    python3 -m beamfastsim.cechain   --backend device: conversion electrons from that table's target stops ->
                                         fraction that reaches the tracker entry plane; on the GPU 3 x 262,144
                                         electrons when JAX sees one, and always on the CPU 3 x 1,024
  crv   python3 -m crvfastsim runchain   1,000 generated cosmic events (default chain) -> coincidence clusters per
                                         generated event (CPU)

The beam and CRV chains run on the CPU (JAX_PLATFORMS=cpu unless set): they are short and their time is mostly
compilation and table loading. The CE chain runs in three equal batches in one process; the first batch pays the
compilation, so setup = map loading + first batch - mean later batch, and the warm rate is electrons/s over the later
batches. With both arms the CE line ends with the GPU/CPU ratio of the warm rates. --device auto (default) runs the GPU
arm when `jax.default_backend()` is a GPU (not when JAX_PLATFORMS=cpu); gpu requires one; cpu skips it. --float32: the
CE chain in float32 (default float64).

--quick uses a quarter of each sample. A line passes when |value - reference| <= TOLERANCE_SIGMA * sqrt(stat^2 +
ref^2), stat being the value's counting error. The field maps and the beam geometry come from CVMFS (or
$BEAMFS_OFFLINE_ROOT / $BEAMFS_MAPS, as `python3 -m beamfastsim.stage` prints them). Writes <out>/demo_summary.json.
Exit 0 every line passes, 1 a line is outside its tolerance, 2 inputs or settings missing (nothing run), 3 a chain
failed. A smoke test of the setup, not a validation: the references are listed in the umbrella's QUICKSTART.md,
section "The demo", and measured in its docs/validation.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import bench

CVMFS = "/cvmfs/mu2e.opensciencegrid.org"
TOLERANCE_SIGMA = 3.0
BEAM_TABLE = "beam_fastsim/g4ref_full/g4ref.npz"
CRV_GEN = "crv_fastsim/g4ref_v1_tables/gen.001400_00000011.root"
CRV_FILES = ("crv_fastsim/tier1b_strict_v5/library.root", "crv_fastsim/geometry/mu2e_extracted_v04.gdml",
             "crv_fastsim/geometry/crvcalib_sim_best_v1_5_run1400.csv",
             "crv_fastsim/geometry/crvphoton_sim_best_v1_5_run1400.csv", "crv_fastsim/lookup_v6_0/photonMap.root")


@dataclass(frozen=True)
class Reference:
    value: float
    error: float
    source: str


# reference values (umbrella QUICKSTART.md, "The demo"): sample, how measured, and the error that enters the tolerance
REFERENCES = {
    "beam": Reference(0.36038, 0.00148, "Geant4 DS stage on all 105,350 muons of the table, clean stock rerun "
                      "(docs/validation.md; the first reference job, which stopped at event 105,100, gave 0.35919); "
                      "the fast chain gives 0.35834"),
    "ce": Reference(0.72598, 0.00223, "host chain (NumPy with JAX slab layers) on 40,000 electrons from the same "
                    "stops, seed 2: 29,039 reach the plane; the device chain matches the host chain (yield per start "
                    "0.9999 +- 0.0007 float64, 1.0010 +- 0.0007 float32) and the CE chain agrees with Geant4 CeEndpoint "
                    "(docs/validation.md)"),
    "crv": Reference(0.05822, 0.00108, "stock chain (Geant4 + digi + reco) on all 50,000 events of the same shard, "
                     "2,911 clusters (crv_fastsim/v2_fastC_v1/ref.001400_00000011.root); the fast chain gives 0.0576"),
}
CHAINS = ("beam", "ce", "crv")
SAMPLES = {"beam": 20000, "crv": 1000}                    # muons, generated events
CE_PASSES = 3
CE_BATCH = {"gpu": 262144, "cpu": 1024}                  # electrons per batch of each arm
CE_GPU_OPTIONS = ["--lanes", "131072", "--shrink", "32", "--unroll"]   # GPU recipe of beam/README.md


class Missing(Exception):
    """An input or setting the demo needs is absent; nothing was run."""


def _offline_and_maps():
    off = os.environ.get("BEAMFS_OFFLINE_ROOT", f"{CVMFS}/Musings/Offline/v10_47_00")
    maps = os.environ.get("BEAMFS_MAPS", f"{CVMFS}/DataFiles/BFieldMaps/Mau13")
    return off, maps


def _cpu_only_platforms():
    p = {s.strip().lower() for s in os.environ.get("JAX_PLATFORMS", "").split(",") if s.strip()}
    return bool(p) and p <= {"cpu"}


def gpu_visible() -> bool:
    """True when JAX in this Python sees a GPU (a subprocess; never when JAX_PLATFORMS allows only the CPU)."""
    if _cpu_only_platforms():
        return False
    try:
        p = subprocess.run([sys.executable, "-c", "import jax; print(jax.default_backend())"], capture_output=True,
                           text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    last = p.stdout.strip().splitlines()[-1:] if p.returncode == 0 else []
    return last in (["gpu"], ["cuda"], ["rocm"])


def ce_command(inputs: str, out: Path, arm: str, batch: int, dtype: str) -> list:
    """cechain --backend device in CE_PASSES batches of `batch` electrons on one arm (gpu or cpu)."""
    off, maps = _offline_and_maps()
    return [sys.executable, "-m", "beamfastsim.cechain", "--offline-root", off, "--maps", f"{maps}/DSMap.header",
            f"{maps}/TSdMap.header", "--stops", f"{inputs}/{BEAM_TABLE}", "--n", str(CE_PASSES * batch),
            "--batch", str(batch), "--seed", "1", "--backend", "device", "--device-float", dtype,
            *(CE_GPU_OPTIONS if arm == "gpu" else []), "--out", str(out / f"ce_{arm}.npz")]


def commands(inputs: str, out: Path, n: dict, dtype: str = "float64") -> dict:
    """Command line per run (lists for subprocess), all from the existing entry points; n: samples per chain, and
    the batch of each CE arm under ce_gpu / ce_cpu."""
    g = f"{inputs}/crv_fastsim/geometry"
    py = sys.executable
    return {
        "beam": [py, "-m", "beamfastsim", "point", "--g4", f"{inputs}/{BEAM_TABLE}", "--n", str(n["beam"]),
                 "--seed", "1", "--out", str(out / "beam")],
        "ce_gpu": ce_command(inputs, out, "gpu", n["ce_gpu"], dtype),
        "ce_cpu": ce_command(inputs, out, "cpu", n["ce_cpu"], dtype),
        "crv": [py, "-m", "crvfastsim", "runchain", f"{inputs}/{CRV_GEN}",
                "--tier1b-dir", f"{inputs}/crv_fastsim/tier1b_strict_v5", "--gdml", f"{g}/mu2e_extracted_v04.gdml",
                "--calib", f"{g}/crvcalib_sim_best_v1_5_run1400.csv",
                "--deviation", f"{g}/crvphoton_sim_best_v1_5_run1400.csv", "--max-events", str(n["crv"]),
                "--seed", "11", "--outdir", str(out / "crv")],
    }


def needed_files(inputs: str, chains=CHAINS) -> list:
    """Release and CVMFS files the selected chains read."""
    off, maps = _offline_and_maps()
    files = []
    if {"beam", "ce"} & set(chains):
        files += [f"{inputs}/{BEAM_TABLE}", f"{maps}/DSMap.header", f"{maps}/TSdMap.header",
                  f"{off}/Offline/Mu2eG4/geom/geom_common.txt"]
    if "crv" in chains:
        files += [f"{inputs}/{CRV_GEN}"] + [f"{inputs}/{f}" for f in CRV_FILES]
    return files


def last_json_object(text: str, key: str) -> dict:
    """The last JSON object in `text` that starts a line and holds `key` (the entry point may log other lines first)."""
    dec, found, pos = json.JSONDecoder(), None, 0
    for line in text.splitlines(keepends=True):
        body = line.lstrip()
        if body.startswith("{"):
            try:
                obj, _ = dec.raw_decode(text, pos + len(line) - len(body))
            except ValueError:
                obj = None
            if isinstance(obj, dict) and key in obj:
                found = obj
        pos += len(line)
    if found is None:
        raise ValueError(f"no JSON object with {key!r} in the output")
    return found


def headline(chain: str, out: Path, stdout: str) -> tuple:
    """(value, counting error, sample size) of a finished chain from the files and output it writes."""
    if chain == "beam":
        m = json.loads((out / "beam" / "metrics.json").read_text())
        return float(m["stop_frac"]), float(m["stop_frac_err"]), int(m["muons"])
    if chain == "ce":
        s = last_json_object(stdout, "fates")
        n, k = int(s["n"]), int(s["fates"]["plane"])
        f = k / n
        return f, math.sqrt(f * (1 - f) / n), n
    if chain == "crv":
        s = json.loads((out / "crv" / "runchain_stats.json").read_text())
        n = int(s["generated_events"])
        c = sum(s["clusters_per_generated"].values()) * n
        return c / n, math.sqrt(c) / n, n
    raise ValueError(chain)


def ce_timing(stdout: str) -> dict:
    """Setup and warm rate of a batched device-chain run: setup = map loading + first batch - mean later batch, warm =
    electrons/s over the later batches."""
    s = last_json_object(stdout, "fates")
    b = s.get("batches") or []
    if len(b) < 2:
        raise ValueError("the CE device run needs at least two batches to separate setup from the warm rate")
    later_n, later_s = sum(x["n"] for x in b[1:]), sum(x["wall_s"] for x in b[1:])
    warm = later_n / later_s if later_s > 0 else float("inf")
    setup = float(s.get("maps_s", 0.0)) + b[0]["wall_s"] - later_s / (len(b) - 1)
    return dict(setup_s=round(max(setup, 0.0), 1), warm_rate=round(warm, 1), dtype=s.get("dtype"),
                batch_s=[x["wall_s"] for x in b])


def run_chain(cmd: list, log: Path, arm: str | None = None) -> tuple:
    """Run one entry point; (exit code, stdout, seconds). arm None: JAX_PLATFORMS=cpu unless set; "cpu": the CPU;
    "gpu": JAX's own choice."""
    env = dict(os.environ)
    if arm == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    elif arm is None:
        env.setdefault("JAX_PLATFORMS", "cpu")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log.write_text(f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p.returncode, p.stdout, time.time() - t0


def main(argv=None, runner: Callable = run_chain, probe: Callable = gpu_visible, label: Callable = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m mu2efastsim.demo", description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="a quarter of each sample")
    ap.add_argument("--out", help="output directory (default $M2FS_DATA/demo)")
    ap.add_argument("--only", default="beam,ce,crv", help="comma list of chains (beam, ce, crv)")
    ap.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                    help="CE chain: also on the GPU when JAX sees one (auto), require it (gpu), CPU only (cpu)")
    ap.add_argument("--float32", action="store_true", help="CE device chain in float32 (default float64)")
    a = ap.parse_args(argv)
    label = label or bench.device_label
    chains = [c for c in a.only.split(",") if c]
    if set(chains) - set(CHAINS):
        ap.error(f"--only takes {', '.join(CHAINS)}")
    q = (lambda v: max(1, v // 4)) if a.quick else (lambda v: v)
    n = {"beam": q(SAMPLES["beam"]), "crv": q(SAMPLES["crv"]), "ce_gpu": q(CE_BATCH["gpu"]),
         "ce_cpu": q(CE_BATCH["cpu"])}
    dtype = "float32" if a.float32 else "float64"
    try:
        inputs = os.environ.get("M2FS_INPUTS")
        if not inputs:
            raise Missing("M2FS_INPUTS is not set: source a site preset (site/*.env) after your access request")
        missing = [f for f in needed_files(inputs, chains) if not os.path.isfile(f)]
        if missing:
            raise Missing("missing or unreadable: " + ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")
                          + " (input release, or CVMFS; see QUICKSTART.md)")
        out = Path(a.out or (os.environ.get("M2FS_DATA") or "") + "/demo")
        if not a.out and not os.environ.get("M2FS_DATA"):
            raise Missing("M2FS_DATA is not set and no --out given")
        gpu = "ce" in chains and a.device != "cpu" and probe()
        if a.device == "gpu" and "ce" in chains and not gpu:
            raise Missing("--device gpu, but JAX sees no GPU (JAX_PLATFORMS=cpu, or a JAX without CUDA; "
                          "QUICKSTART.md, 'The demo')")
    except Missing as e:
        print(f"demo skipped: {e}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    cmds = commands(inputs, out, n, dtype)
    arms = ("gpu", "cpu") if gpu else ("cpu",)
    runs = [r for c in chains for r in ([(f"ce_{arm}", c, arm) for arm in arms] if c == "ce" else [(c, c, None)])]
    if "ce" in chains and not gpu:
        print(f"ce    no GPU visible to JAX: the CE device chain runs on the CPU only ({n['ce_cpu'] * CE_PASSES:,} "
              f"electrons; on a GPU also {n['ce_gpu'] * CE_PASSES:,})")
    rows, worst, warm = [], 0, {}
    for key, c, arm in runs:
        rc, stdout, sec = runner(cmds[key], out / f"{key}.log", arm)
        ref = REFERENCES[c]
        if rc != 0:
            rows.append(dict(chain=c, arm=arm, exit_code=rc, seconds=round(sec, 1), log=str(out / f"{key}.log")))
            print(f"{c:5s} FAILED (exit {rc}) after {sec:.0f} s; log {out / f'{key}.log'}")
            worst = 3
            continue
        v, e, size = headline(c, out, stdout)
        tol = TOLERANCE_SIGMA * math.hypot(e, ref.error)
        ok = abs(v - ref.value) <= tol
        worst = max(worst, 0 if ok else 1)
        row = dict(chain=c, value=v, stat_error=e, sample=size, reference=ref.value, reference_error=ref.error,
                   tolerance=tol, passed=ok, seconds=round(sec, 1), reference_source=ref.source)
        where = ""
        if arm:
            t = ce_timing(stdout)
            row.update(arm=arm, device=label(arm), **t)
            warm[arm] = t["warm_rate"]
            where = f" on the {arm.upper()}"
        rows.append(row)
        print(f"{c:5s} {LABELS[c]:44s} {v:.4f} +- {e:.4f}  reference {ref.value:.4f}  tolerance +- {tol:.4f}  "
              f"{'ok' if ok else 'OUTSIDE'}  ({size:,} {UNITS[c]}{where}, {sec:.0f} s)")
        if arm:
            print(f"      {row['device']}, {t['dtype']}: setup {t['setup_s']:.0f} s (compilation, maps), "
                  f"warm {t['warm_rate']:,.0f} electrons/s")
    speedup = round(warm["gpu"] / warm["cpu"], 1) if {"gpu", "cpu"} <= set(warm) and warm["cpu"] > 0 else None
    if speedup:
        print(f"speed CE device chain ({dtype}), warm rate GPU / CPU: {speedup:,.0f}x")
    summary = {"quick": a.quick, "tolerance_sigma": TOLERANCE_SIGMA, "inputs": inputs,
               "release_manifest": os.environ.get("MANIFEST"), "ce_dtype": dtype, "ce_gpu_over_cpu": speedup,
               "chains": rows, "passed": worst == 0}
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"summary {out / 'demo_summary.json'}")
    return worst


LABELS = {"beam": "target stop fraction", "ce": "CE reaching the tracker entry, per electron",
          "crv": "CRV coincidence clusters per generated event"}
UNITS = {"beam": "muons", "ce": "electrons", "crv": "generated events"}


if __name__ == "__main__":
    raise SystemExit(main())
