# Bounded retained CE writer/proxy availability audit

## Scope

`crvfastsim.veto.ce_proxy_audit` asks whether the eleven already retained CE
ntuples persist a version observation and usable-looking scalar luminosity
observables. It does **not** estimate calibration, selection dependence, a joint
exposure law, correction weights, efficiency or veto loss.

The mandatory external pin for `selection_window_v2.json` is
`f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3`.
Only its exact eleven campaign paths, UUIDs and highest-cycle tree entry counts
are admitted: 91,226 entries total. No namespace scan or alternate input is used.
The selected-track path is inherited provenance only; neither it nor the old
exposure ledger is opened, inspected or hashed.

## Reads and interpretation

Allowed event leaves are full run/subrun/event identity; measured `nprotons`,
`pbtime`, `pbterr`; and scalar luminosity fields `nProtonTCs`, `nTrackerHits`,
`nCaloHitsD0`, `nCaloHitsD1`, integer `caloEnergy`, and `caphriIsValid`.
No MC, track, CRV, vector CAPHRI energies/IDs or other physics collections are
read. Missing optional leaves are reported, with no alias substitution. Full
identities must be unique within and across files.

Counts and energy retain the documented -1 reset sentinel separately from zero
and positive values. Signed bunch time, including -1 ns, is valid: there is no
source evidence for a -1 time sentinel. Negative bunch-time error is invalid,
not silently treated as missing. `caphriIsValid` is a boolean validity observation.
Persisted zero is not itself proof of a filled/calibrated observable: EventInfo
fields can be reset/unfilled at zero.

The exact `EventNtuple/version` key is read only if it is a small TH1I. Values,
flow, labels, cycles and a three-component interpretation are recorded; a version
different from the declared recipe's 6.13.2 is a finding, not a selection failure.
A stored version is not proof of the complete writer configuration or package
closure.

## Resource and integrity contract

- One file at a time; scalar summaries are vectorized NumPy operations.
- Exact-path `os.open/read/lseek`, no PNFS stat/fstat/lstat or discovery.
- Cumulative reads bounded to 128 MiB/file and 1 GiB total, counting reopens;
  single reads at most 32 MiB. These caps do not impose an NFS I/O timeout.
- Uproot caches disabled; trivial decompression/interpretation executors.
- Scalar dtype, branch class, entry count and basket byte/count bounds checked
  before selected array reads; 1 MiB/branch and 16 MiB selected payload/file.
- Scoped uproot metadata guards bound directory keys and decompressed metadata;
  embedded-basket/metadata-triggered fallback reads fail closed. This uses uproot
  internals and is tested in the deployed environment before execution.
- Version object at most 64 KiB/16 bins, local pinned JSON at most 8 MiB, source
  at most 64 KiB and output at most 2 MiB. These are not whole-process RSS caps.
- Source snapshot and per-array digest/dtype/shape retained. Header reopen checks
  UUID, tree count/cycles and initial bytes without rereading arrays.
- Original report and source rechecked before exclusive new-file output, outside
  old evidence/code/selection directories; no overwrite, atomic or durability claim.

Neither UUID/header checks nor consumed-array digests are full-file hashes.
Unchanged headers cannot exclude changed baskets. Per-file valid quantiles and
exact aggregate counts are reported; no fabricated aggregate quantiles or
intensity calibration is provided. F, weight model and finite-library uncertainty
remain null.

## Validation and execution status

The reviewed implementation has **66 targeted tests passing locally (1.95 s)**
and **remotely (6.02 s)**, using synthetic real ROOT files and adversarial guards.
Parent review corrected integer calorimeter-energy admission and removed an
unsupported -1 sentinel interpretation for time/error/boolean fields before any
real read. No production pin or expected outcome was altered.

The first NFS invocation produced no report. Its caller timed out while the
still-live process waited in storage I/O on the first file, before the initial
100-byte header read advanced. Exact-file locality reported ONLINE_AND_NEARLINE,
so tape absence was not established as the cause. The exact owned process was
terminated with SIGTERM and verified absent after 6:46.03 wall, with 0.25 s user
CPU and 90,496 KiB peak RSS. The signal-termination line is authoritative; the
`time` footer's exit-status-zero field is not an audit-success receipt. Source,
tests and logs were preserved; no duplicate invocation was launched.

Opt-in `--transport xrootd` now provides bounded positional reads for the same
pinned inputs with 30-second open/read/close timeouts, no stat calls, and no
application retry, fallback or input substitution. Zero-length reads never reach
the client (its size-zero convention is unbounded); short reads fail. Transport,
resolved URLs and client version are reported. Default NFS behavior is unchanged.
The complete suite passes **144 tests locally and remotely** (remote 8.15 s).

The second real attempt exited one in 0.78 s at XRootD open with
`[FATAL] Auth failed: No protocols left to try`; close reported the same failure.
Peak RSS was 97,156 KiB. No report or physics payload was obtained. Both failures
are preserved. The user confirmed that Kerberos is active and refreshed the grid
token; the initial suggestion to renew Kerberos was not supported by this error.
The alternative host's data mount then failed with `Transport endpoint is not
connected`, before reservation. Once the original host became idle, the next
XRootD invocation authenticated and reached tree metadata. It failed closed in
2.29 s (96,532 KiB peak RSS), before any scalar array, with
`embedded baskets forbidden during metadata`.

A bounded error-context-only change, with **145 targeted tests passing locally
and remotely**, identified the split `evtinfo` parent: 8,212 entries, zero normal
baskets, zero own total/compressed bytes, six children and one leaf. This second
metadata rejection took 0.56 s (96,956 KiB peak RSS). Entry-coverage mismatch on a
structural parent is not itself proof of an embedded basket. Both attempts and
source versions are preserved; neither produced a report. A safe correction must
prove the serialized basket collection empty before admitting such parents and
must still forbid metadata fallback/payload reads. No input pin, field alias,
expected value or resource cap is changed. No CE availability/version finding
existed at that checkpoint. Client timeouts are per operation, not a whole-run
deadline.

## Completed retained-file availability result

The structural-parent correction now proves the serialized basket collection
empty/null-only from resident bounded metadata for every branch. Only zero-own-
payload parents with children can bypass the entry-coverage equality; entries
and offsets are never altered. Unsupported serialization, non-null basket tags,
metadata fallback and lazy basket decoding still fail closed. Supported envelopes
are byte-counted TObjArray v3, plain nonreferenced TObject v1, bounded names/counts,
zero lower bound and literal-null slots. Hooks remain single-threaded and internal
to uproot. **328 targeted tests pass locally (2.92 s) and remotely (9.02 s)**,
including serialized base/derived parents and adversarial basket envelopes.

The subsequent single successful audit covered all **11 pinned files / 91,226
events**, with unique full run/subrun/event identities. Highest tree cycles
(15 or 16) match the saved entry counts; all UUID/header/cycle rechecks pass.
It read **3,467,882 bytes**, including reopens, in **17.95 s wall** with
**125,912 KiB peak RSS**. The 323,463-byte report has SHA-256
`2b59602667126391fe1b32bb6ce4051b83c88268f5f60c580922d1cfeef18325`.

Every file stores version **6.13.2**, with labels major/minor/patch, zero flow
bins and version-key cycle 1. All nine optional scalar fields are present in all
files; no count/energy -1 sentinel, nonfinite or domain-invalid values were found.
Ranges below are exact minima/maxima across the saved per-file summaries, not
pooled quantiles.

| Field | Observed range / counts across 91,226 events |
|---|---|
| `evtinfo/nprotons` | Zero in every event |
| `evtinfo/pbtime` | -226.62432861328125 to -198.27413940429688 ns; all finite |
| `evtinfo/pbterr` | 0.5 ns in every event |
| `lumistream.nProtonTCs` | 0–14; 35,037 zero and 56,189 positive |
| `lumistream.nTrackerHits` | 14–3,100 |
| `lumistream.nCaloHitsD0` | 14–849 |
| `lumistream.nCaloHitsD1` | 4–463 |
| `lumistream.caloEnergy` | Integer 49–3,232; no unit/calibration inferred here |
| `lumistream.caphriIsValid` | True in every event |

This closes **stored version/scalar availability only**. Measured nprotons still
cannot supply a varying intensity observable. The luminosity fields are candidates
for a separately justified measured proxy, not calibrated PBI or proof of its
joint distribution with bunch phase and selected-track time. No selected tracks,
old exposure-ledger rows, MC or CRV arrays were read. No weights, F or finite-library
uncertainty were calculated; those result fields remain null. Reuse this completed
result rather than rereading the retained files just to establish availability.

Successful source SHA-256:
`3765b8114f4128d7bae438d3b943cc6f8b61c1c4f393e0963fb4ede6494cc776`;
test SHA-256:
`88886eda4b16e909e319cce4999f3f4db3b3dd4e2e18499f8aea94de7232cad3`.

```bash
python3 -m crvfastsim.veto.ce_proxy_audit /data/SAVED/selection_window_v2.json \
  --expected-selection-sha256 f8029e1afc3db72b09549ac9e3ae92c5f5263f60f4db38fcda161166077731f3 \
  --out /data/SEPARATE_OUTPUT/NEW_REPORT.json
```

Reuse any eventual successful result. Do not rerun old selection/metadata audits
or turn proxy availability into a calibrated CE exposure correction. C4 remains
stock; historical F is unchanged.
