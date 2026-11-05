# Same-saved-step deterministic stock response control

## Scope and build gate

This control runs stock `CaloShowerROMaker` on the original saved `CaloShowerStep` collection, not on a
reconstructed compression replay. The new `CaloResponseDumper` is an isolated analyzer; neither the stock
producer nor `CaloLibraryDumper` is modified. **User-built and exercised on an existing-art pilot**, with
exact observed deterministic-output comparisons to NumPy and eager/JIT JAX CPU primitives. This is scoped
same-input evidence, not validation of all conditions, JAX devices or the full simulation/reconstruction chain.
A fresh full user build and verification are mandatory after any C++ change before loading it. No transport, runtime packaging, library
population, change to production defaults or reconstruction is authorized by this preparation.

The original transport schedule need not contain `EventWindowMarker` or `ProtonBunchTimeMC`; unexecuted producer
configuration is not evidence of a persisted product. The control therefore schedules stock
`EventWindowMarkerProducer` with on-spill mode, initial phase zero and no reconstructed-time smearing. Its
products are explicitly process-qualified. This is a controlled response experiment, **not historical digitization
replay**. The original step energy, time, position, identities and event population remain unchanged.

## Explicit conditions and switches

`art/CaloFastSim/fcl/response_control.fcl` specifies:

- Single input `CaloShowerStepMaker::CaloLibraryCE`; no step maker, generator, G4, waveform or fitting module.
- Stock event-window producer with `SpillType=1`, `InitialPhaseShift=0`, `RecoFromMCTruth=true`,
  `RecoFromMCTruthErr=0`; export the resulting per-event double window length and float proton-bunch time.
- Birks and LRU enabled; PE Poisson fluctuations and photon travel time **disabled**.
- Explicit start/end/buffer, nominal file-backed `CalSimParams`, file-backed `EventTiming` and marker offset.
  Expanded configuration and original simulation search inputs must accompany the output.
- `PhysicsParams` nominal DR period and geometry crystal length narrowed to float exactly where stock does so.
  The actual fetched values, conditions CIDs and validity intervals are exported, not inferred from a nominal value.

The propagation ROOT file/histogram remain necessary: stock calls `CaloPhotonPropagation::buildTable()` in
`beginRun` even with travel time disabled. Record their resolved file, SHA256 and histogram name, but do not claim
that this control validates their inverse CDF or CLHEP random realizations. Matching seeds across CLHEP/JAX is
not a supplied-realization comparison. Exact stochastic-realization replay would need independently observed
draws/cache values. The separately executed travel, Poisson and combined statistical arms do not establish that
replay or production RNG certification; see their control documents and PAUSE_RESUME.md.

## Observations, predictions and joins

The analyzer exports only observations and fetched conditions:

- `events`: every input event, collection ProductIDs/counts including empty outputs, timing-product IDs, spill,
  window length, bunch time, marker offset, nominal period, crystal length, CalSimParams/EventTiming IOVs and CIDs.
- `steps`: original product/index, SimParticle product/key, crystal/readout base, persisted energies/time,
  source-float local z, LRU and the two fetched PE yields. Rejected-time and zero-PE steps remain present.
- `readouts`: observed output product/index, exact saved-step ProductID/key, SiPMID and every float PE time.
- `summaries`: observed stock grouped corrected energy/time and ordered exact saved-step member pointers.

The analyzer deliberately does **not** recompute response formulas and label them stock observations.
Per-step corrected energy, PE mean, double folded time and accept/reject flag are **not directly emitted by the
stock producer**. Grouped energy/time and per-step/readout counts/times constrain their predictions, but do not
independently certify every intermediate. In particular zero-PE step energy is observed only within a group;
stock compressed raw-constituent identity is a different, still-open question.

`check_response_control.py` uses exact event/product/index + SiPM keys, not time/energy/position matching.
SiPM-only stock sorting is not stable, so readout row order is not treated as a saved-step association. The checker:

1. Reconciles every event and saved step against the independently audited numeric export; requires the supplied
   expected finished-event count, status zero and exact original IDs/order/energies/time/float-local-z.
2. Validates collection registries, contiguous rows, positive IDs, conditions intervals and finite source inputs.
3. Predicts float energy/means/half-up PE counts, double fmod folding and lower-only time rejection. Very late
   finite times are neither clipped nor discarded. Accepted zero-PE steps still appear in stock summary membership.
4. Compares exact saved-step/SiPM sets, counts and all PE times. Uses stock ordered member pointers to compare
   sequential float group energy and minimum stored-float time; also requires precisely all accepted steps once,
   with one exact particle/crystal group and input-order members per summary, in stock crystal then
   SimParticle persistent-pointer map order.
5. Reports exact mismatches without association repair, tolerant count matching or silently changing conditions.
   Numeric failure retains its report and exits nonzero. `--backend jax` enables source-double x64 for that check;
   contraction-sensitive PE boundaries remain strict failures, not a tolerance on discrete counts.

No passing result certifies stochastic response, digitization, reconstruction or an unbiased library. No passing
synthetic fixture is stock execution evidence. The control preserves empty events but does not create an independent
incident manifest or certify that an empty transport response is a physical zero.

## Post-build execution and evidence

After the user completes a full build on the designated build host:

1. Verify the new plugin exists, postdates its source and both build inputs, has the pinned backing/envset,
   and passes direct-library/undefined-symbol checks in that environment. Do not use an old library dumper binary
   as evidence that the response analyzer was built.
2. Reuse the retained original pilot art through the existing authorized temporary-input mechanism; verify its
   checksum. Do not rerun transport or stage it in a shared data area. Check host availability and quota first.
3. Preserve original simulation geometry/search input precedence and relative FHiCL include semantics. Save the
   expanded control, resolved geometry/global-constants log, condition file, propagation file/histogram provenance,
   stock producer/analyzer/build hashes, runtime and process completion. Outputs/logs belong in the data area.
4. Run a tiny existing-art control, inspect all four tables and completion, then the full approved event population.
   The first run may still discover compile/configuration/product issues; preserve failures and do not relabel them.
5. Check the output with explicit original-export reference, expanded control, expected count and fresh report:

   ```bash
   PYTHONPATH=calo python3 calo/scripts/check_response_control.py CONTROL.root \
     --reference ORIGINAL_EXPORT.root --expanded-fcl EXECUTED.expanded.fcl \
     --expected-events COUNT --out FRESH_REPORT.json
   ```

   Repeat with `--backend jax` only in an environment containing JAX; add `--jax-jit` to compile the four
   response primitives, retaining exact discrete-count failures. Reports record versions/device/x64/mode and
   source hashes. Summary accumulation remains host NumPy sequential float in stock member order in both modes;
   this is not a device-resident grouping/reconstruction test. Counts of retained empty events, time rejections,
   accepted zero-PE steps and all compared PEs are reported without becoming incidence/physical-zero certificates.
   No throughput extrapolation from a tiny check. No removal of persistent pilot/evidence files without approval.

`scripts/response_control.sh` runs only against the verified satellite and an explicit input/plugin checksum,
checks plugin freshness against source/build inputs, restricts event counts to the approved pilot bound, requires
a fresh output directory, and records success/failure receipts. It preserves expanded FHiCL, simulation search-root
precedence, resolved condition/propagation input hashes, source/build hashes, geometry/global-constant logs and
process timing. It contains no build or transport command. Original transport timing and this control's timing
must remain separate; whole-process art timing with warm caches is not an isolated response throughput benchmark.
