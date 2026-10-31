# PVC reconstruction profiling

`python3 -m crvfastsim.reco.pvc_profile --out results` measures pulse fitting,
small-matrix linear algebra and coincidence finding with fixed synthetic inputs.
Run on the intended compute node, not on a login node. The explicit oneAPI
backend selector is `JAX_PLATFORMS=oneapi`; `gpu` may expand to CUDA/ROCm only.

The diagnostic writes environment metadata, optimized HLO, compiler cost and
memory analyses, resident execution samples, transfer times and output hashes.
`--trace` captures JAX traces when available; `--summarize-trace results` reports
the duration events present. A trace without device events is not a device-kernel
profile. Compilation is measured separately from execution and transfer.

Use the same `--rows`, `--groups`, `--hits`, `--slots`, `--rounds` and seed for
comparisons. `--against other_results` compares every output by bytes, including
NaN payloads and signed zeros. A failed byte check must not be relabelled as an
identity certificate because a tolerance or histogram check passes. Reconstruction
on different backends can differ in the last floating-point bits before any
optimization; no rounding or quantization is applied to hide that.

The site drivers in `scripts/pvc_profile_{aurora,uva}.sh` compare one device with
a CPU control. They use synthetic inputs, an already installed environment and
read-only geometry; `PVC_RUN_DIR` names the writable run directory (it holds the code snapshot
under `code/`). Adjust the site root/account for another installation.
The tests driver runs the selected diagnostic tests plus the reconstruction
regressions on a CPU compute node. It does not require a new image.

`--compare-arrays results --against reference` compares saved arrays without
rerunning a kernel. Its selected-output section distinguishes found pulses and
accepted cluster roots from unreported padded slots. `--summarize-node results`
reports the slowest-worker bottleneck rate for simultaneous runs, with explicit
waveform/s or synthetic-event/s units; neither is an end-to-end accepted-event
rate.

## Bottleneck isolation

The pulse Hessian check calls `jnp.linalg.eigvalsh` on a batch of 3x3 matrices.
The oneAPI 0.11.2 lowering uses `oneapisolver_syevd_ffi`. In the implementation
introduced by [jax#40543](https://github.com/jax-ml/jax/pull/40543), that call
loops over matrices on the host, issuing a separate oneMKL eigensolve per matrix.
Small-matrix batches therefore need a resident execution profile, not a hardware
FP64 peak comparison. The LM/GN scans have fixed iteration counts; a `while` in
optimized HLO is not evidence of data-dependent fit convergence.

## Experimental reconstruction path

`CRVFASTSIM_PVC_RECO=1` selects a scaled cyclic-Jacobi 3x3 eigensolver for the
Hessian condition test, and minimum-label propagation to convergence for finder
connected components. The Hessian eigenvalue-ratio test, fit iterations, all
thresholds, hit selection and recovery semantics remain present. On oneAPI only,
the chain caps pulse batches at4096 and retains original finder row capacities
even when compact dispatch is requested. Other backends keep their caller batch
sizes. The mode must be set before first JAX tracing and remain fixed during a
process; changing it does not invalidate already compiled kernels.

This path is opt-in pending the complete identity/validation certificate. Changing
the eigenvalue implementation preserves its mathematical test but does not by
itself prove identical floating-point decisions at the condition boundary. The
saved-output comparisons test that explicitly on each sample. No fit-output
rounding, quantization or removal of the Hessian check is used. Backend-specific
transcendental evaluation, reductions and linear algebra can already differ
before any optimization; CPU/PVC literal equality is a separate gate.

`python3 -m crvfastsim.reco.pvc_parity --out saved --batch 4096` saves canonical
synthetic pulses and cluster properties for comparisons against an unchanged
process. Use `pvc_profile --compare-arrays saved --against reference` for byte
comparison, not only histogram agreement.

## Compiler reproducer

`python3 -m crvfastsim.reco.pvc_finddense_repro --hits 32 --slots 8` directly
calls the finder at the historically problematic shape. It uses no detector
geometry or conditions. Run in its own process: affected compilers abort rather
than raising a Python exception. `--hits 16` is a control. The production oneAPI
skip remains unchanged.

The upstream compiler family has a relevant fix:
[openxla/xla#49948](https://github.com/openxla/xla/pull/49948), merged 2026-10-02,
disables LLVM vector-combine and horizontal SLP transformations that can emit
unsupported SPIR-V vector widths. Presence in a source commit does not establish
its presence in an installed wheel or compatibility with the site's driver.
No compiler/runtime or reconstruction default is changed by these diagnostics.
