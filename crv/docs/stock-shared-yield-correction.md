# Stock shared-lookup yield correction and validation proposal

**Status: proposal only. No Offline source patch, C++ test or build has been performed.**
The fast shared-yield option remains a diagnostic of existing stock behavior, not
an acceptable replacement for the declared sector calibration. See
[named-response-diagnosis.md](named-response-diagnosis.md) for the measured effect.

## Evidence boundary

The archived actual CVMFS Offline `v13_39_00`
`CRVResponse/src/CrvPhotonGenerator_module.cc` has SHA256
`1b5aef320f6f344d607fbf3c18e92b075b9d4daed84532d61ef78938dccd826e`.
Its relevant behavior is:

- Lines 195–223 share the complete `MakeCrvPhotons` helper by lookup filename;
  only the first allocation calls `SetScintillationYield`.
- Lines 343–362 select the sector helper and refresh four channel deviations,
  but do not refresh the sector yield before `MakePhotons`.
- Line 205 prints the configured sector yield, not the helper's effective yield.

The inspected local module is byte-identical to that release copy. Helper-source
review was made separately in Offline checkout
`acc62589e19d122ba5880bc49a9f425f1e4f1887`; these files were **not independently
archived from CVMFS**, so this is not additional release-binary attestation:

| Helper | SHA256 |
|---|---|
| `CRVResponse/inc/MakeCrvPhotons.hh` | `bb5dc66d44bcf032eb09978c42cd6975c94d9b119a722e60c332b57317a2c19b` |
| `CRVResponse/src/MakeCrvPhotons.cc` | `b350e74158f13bae26615415750f85c6c16f5e09860c5237d36edc58fbf44157` |

The helper holds mutable yield, channel deviations and arrival vectors, owns its
LUT containers, and borrows the module's random generators by reference. Loading
has no random draws; arrival vectors are cleared at the beginning of every call.
The yield multiplies scintillation at implementation line 336, not the Cherenkov
mean. A source census of that checkout found one production allocation, one
production yield setter and one production `MakePhotons` call in the module;
this census does not cover external clients.

## Recommended narrow correction

Set the selected sector's yield immediately before **every** `MakePhotons` call:

```cpp
photonMaker->SetScintillationYield(_scintillationYields.at(CRVSectorNumber));
photonMaker->MakePhotons(/* existing arguments */);
```

This is illustrative patch guidance, not an applied or compiled patch. Retain the
initialization setter for a minimal diff. Per-step assignment keeps filename-only
LUT sharing, allocations, ownership and API unchanged and consumes no random
numbers itself. Setting only in the shared-object initialization branch would
replace first-sector-wins with last-sector-wins; setting once per event would
still fail on interleaved sectors.

This relies on the existing serial select/set/generate/read sequence. It is not
a thread-safety fix: deviations, arrival buffers and random-generator references
remain mutable and shared.

### Alternatives

- **Cache by `(filename, configured yield)`**, using exact equality, not a numeric
  tolerance. The existing initialization path would then produce correct helper
  state. Equal-yield groups still share; unequal-yield groups duplicate the LUT
  containers. Measure memory/startup overhead rather than assuming it is small.
  Keep the same module RNG references; no new engines, seeds or manual deletion.
  Reflector is already a per-call argument, and deviations are refreshed per step.
- **Require yield as a `MakePhotons` argument.** This is the cleaner explicit-state
  API but changes header, implementation and caller and requires an external-client
  compatibility check. Do not silently default the new argument. It still does
  not make the remaining shared mutable state thread-safe.

Prefer the per-call setter for a narrow correction unless maintainers request the
explicit API. LUT duplication is unnecessary for this particular state defect.

## Actual-module regression oracle

A standalone helper test that manually sets the yield cannot prove module/cache
wiring. Exercise the actual module selection/cache path. Use two **different
lookup filenames containing identical bytes** as the unshared oracle: even the
existing cache then constructs separate helpers with their correct initial yields.
Use identical engine state, step order, geometry, deviations and all other settings
for the shared-filename run and its oracle; verify the file hashes match.

| Fixture | Required assertion |
|---|---|
| Equal yields `(39794,39794)` and `(30482,30482)` | Exact per-call photon counts, arrival vectors and final RNG state match the unshared oracle. |
| Unequal yields `(39794,30482)` | Both sectors match their own oracle; a sufficiently responsive fixture must expose the unfixed behavior. |
| Reversed yield assignment `(30482,39794)` | Catches both first-wins and last-wins fixes. |
| `A,B,A,B` and `B,A,B,A`, repeated across events | Each sequence matches its identically ordered oracle, without stale yield/deviation/arrival state. |
| Unique LUTs and equal-yield sharing controls | Controlled same-build behavior remains exactly unchanged. |
| Distinct deviations, compatible signed reflectors, zero-response final call | Channel scaling, routing and output clearing remain correct. |

Keep sector/geometry ordering valid: module `beginRun` checks it at lines 245–251.
Swap the yield assignments on fixed sectors or remap the whole fixture consistently;
do not reverse only the `CRVSectors` list. Do not compare different step sequences
bitwise: sequential RNG consumption makes that the wrong oracle. Copy arrivals
before the next call clears them. Check construction/loading leave engine state
unchanged. Include a scintillation-only test and a mixed scintillation/Cherenkov
control.

## Saved-product validation after a user build

1. Reuse the existing finished 1000-event Run1A/Run2 stock outputs and exact
   source art. Verify highest-cycle event entries and unique full-width identities
   from the outputs, not log event estimates. Do not rerun Geant4 or an existing
   stock baseline merely to exercise this patch.
2. An isolated comparison to those saved products requires the same `v13_39_00`
   dependencies, assets, seed configuration, geometry, conditions and downstream
   settings, differing only by the patch. Run only the candidate into a new output
   directory after the user builds the chosen area.
3. Current-head PR validation is separate: comparing a head candidate to this
   release reference is cross-version, not an isolated patch test. It requires an
   appropriate same-build head baseline. Reuse one if available; otherwise report
   the missing baseline and agree on that work rather than silently generating it.
4. Audit every sector and all four SiPM indices, including zero-event/zero-response
   rows: photons if available, digis, saved ADC, pulse PE/time and clusters. Expect
   T2/T5 to use their declared lower scintillation yield, but do not impose
   `30482/39794` as an exact reconstructed-PE ratio: Cherenkov, thresholds and reco
   are nonlinear contributions.
5. Unaffected full-job sectors need statistical compatibility, not identical bytes:
   changed photon numbers change later RNG consumption. Report paired-event errors
   and retain the independent-seed-floor limitation until measured.
6. Preserve the fast shared-yield replay as a **stock-behavior diagnostic**, never
   relabel it patched-stock output. Keep the independent Run2 R4 physical-presence
   omission separate: stock already checks `hasCMB` before channel status.

No calibration compensation, spectrum-bin tuning, threshold relaxation or full
named V2 claim follows from this correction. Its C++ fixture and candidate runs
remain to be implemented and executed; no build command is issued until an
isolated source area and validation target have been agreed and prepared.
