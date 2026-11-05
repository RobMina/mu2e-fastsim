# Target-born DIO and RPC sources (opt-in)

`beamfastsim.target_sources` creates explicit source tables. It does not change
conversion-electron defaults or claim DIO/RPC validation against Geant4.

## DIO

Select the spectrum table from the Offline release of the reference production.
The two columns are **total electron energy** in MeV and density. The reader
follows Offline `BinnedSpectrum`: abscissae are left edges unless `BinCenter` is
true; only whole bins inside the requested interval survive; the density is
constant within each bin, as `CLHEP::RandGeneral` plus `BinnedSpectrum::sample`.
The table must be increasing, uniformly binned and nonnegative. Offline
`loadTable` does not append its final parsed row if EOF is set by the last value
(no trailing whitespace); this reader preserves that behavior. In particular,
keep the release table byte-for-byte: adding a newline can change the last bin.
Production job overrides, not the base FCL, determine the energy interval.

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.target_sources \
  --process dio --stops stops.npz --spectrum-table dio.tbl \
  --elow 95 --ehi "$DIO_EHI" --n 40000 --seed 1 --out dio_starts.npz
PYTHONPATH=core:beam python3 -m beamfastsim.cechain \
  --source dio_starts.npz --offline-root "$OFFLINE" \
  --geom Offline/Mu2eG4/geom/geom_run1_a.txt \
  --maps DSMap.header TSdMap.header --backend device --device-float float32 \
  --lanes 4096 --out dio_plane.npz
```

This Run1A example requires geometry, its complete include closure and field maps
from the same release as the reference. Setting `--offline-root` alone does not
select the geometry. These commands are interface examples, not a benchmark receipt.

Set `DIO_EHI` from the production job's expanded FCL or log, not from the
unoverridden generator template. The spectrum file's highest complete bin still
limits the effective interval when the configured upper cut is above the table.

The stop-table schema is `pdg, at_rest, foil, xstop, ystop, zstop, tstop`.
`pdg=13`, `at_rest=true` and `foil>=0` select stopped negative muons in the
target. Rows are resampled with replacement; directions are isotropic; the
emission time is the stop time plus an exponential muon lifetime in aluminium
(864 ns, shared with the CE source). This is one electron per resampled stop,
**conditional on the requested energy interval**, not a decay branching fraction
or an absolute yield per incident proton. Table and stop hashes, effective
interval and source provenance are stored in the source file's `metadata`.
`beam_point --stops-out` tables (`x, t, foil, q, muons`) are also accepted for
DIO: their stopped-muon charge selects PDG 13, never a fictitious pion.
`stop_idx` preserves the original stop identity. `cechain` output `idx` identifies
the source row, not the stop row; chunking slices the source without redrawing it.

### Explicit generated-event DIO adapter

The existing `target_sources --n` interface counts emitted electrons and is
unchanged. `target_source_events` is a separate opt-in interface for a campaign
specified in generated events. It requires an audited stopped-muon parent table
with complete event identities, multiplicities and the exact eligible PDG13 /
process32 selection. Each scheduled parent event emits one electron for **every**
eligible muon, preserving multi-electron correlations and repeated occurrences.
It stores generated-event and emitted-electron denominators separately.

The caller supplies a predeclared empirical population NPZ whose `event_index`
vector identifies parent-event occurrences, including repeated window exposure.
The schedule RNG draws events from that population, not flattened stop rows;
a separate source RNG draws electron kinematics and decay delays. Parent,
population and spectrum hashes are retained; output creation is exclusive.
This adapter does not choose campaign weights or establish parent independence,
source equivalence or absolute normalization. Bind it to a reviewed admission
receipt and frozen schedule before confirmation. Transport still joins source
rows through `idx`; per-event arrays remain in the hash-bound source file.

## RPC

Offline `RPCGun` draws photon energies from the midpoint-binned Bistirlich
spectrum (default 50–139.5 MeV, 0.1 MeV bins, zero above 134.530 MeV). Its
internal conversion uses the Kroll–Wada–Joseph density in pair invariant mass
and energy asymmetry, then isotropic electron direction and the correlated
positron opening angle. `internal_pair` retains both momentum vectors and pair
variables; `rpc-internal` supplies the electron marginal to the CE chain.
The positron is **not** transported by that electron-only adapter.

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.target_sources \
  --process rpc-internal --stops physical_pion_stops.npz --physical-pion-times \
  --elow 50 --ehi 139.5 --n 40000 --seed 1 --out rpc_starts.npz
```

RPC requires **negative pion stops** (`pdg=-211`), not beam muon stops. Retain
`run, subrun, evt`: `RPCGun` resamples an event and then chooses one of its
eligible pions uniformly. Flattening all pions and resampling rows instead
would overweight multi-stop events; the source adapter uses the two-stage draw.
The input must already apply the production stopping-code and stop-time cuts.
Use a physical-time pion ensemble and explicitly certify it with
`--physical-pion-times`. The emission is at the pion stop position and time,
with no muon lifetime delay. A decay-off production needs ancestry-based pion
survival weights and a weighted normalization adapter; this CLI is not that
adapter. Source generation rejects weighted stop ensembles, and the CE source
adapter rejects non-unit source weights rather than silently losing them.

External RPC (`--process rpc-external`) emits **photons** at pion stops, exactly
as `RPCGun` does. Photon propagation and conversion in the target/absorbers are
not supplied here. `cechain --source` rejects photons and positrons. An electron
placed at a pion stop cannot stand in for a material-converted photon; the
external-RPC source must not be presented as an end-to-end RPC simulation.

## References and validation boundary

`cereference --code` can extract DIO or internal-RPC electron starts from an
existing `ReadVirtualDetector` tree (creation codes 166 and 179; default CE 167).
Verify those codes and the production configuration in the chosen release.
Internal pairs can pass a production filter through either daughter, so kept
rows alone do not establish generated yields. External-conversion electrons
need their actual conversion start and ancestry, not the RPC primary code.

The source tests cover bin semantics, support, isotropy, reproducibility, stop
identity and timing, pair energy/invariant-mass conservation, and rejection of
unsupported source types. They are synthetic contracts, not evidence of
agreement with Geant4. Before physics use, audit the release/geometry/field,
energy restriction, production filter, pion survival convention and generated
normalization, then compare electron yield, energy, angle, radius and time at
VD 13 with paired errors. `SingleProcessGenerator` can emit for every stopped
muon in one event: a generated **event** count is not automatically an electron
count. `target_yield_compare` requires a reference-hash-bound exact parent-window
exposure receipt, even when the retained events each contain one DIO electron. The CE peak-window summary is not a sufficient
spectrum metric for DIO or RPC. `beamfastsim.target_compare` verifies all start
coordinates/momenta/times, uses the same VD selection, and reports paired
momentum, angle, radius, TOF and absolute-time differences. Histogram errors
cluster the paired differences by reference file, retaining flow bins; a single
file does not establish that variance. It explicitly labels kept-reference
transport as an electron-marginal check, not generated-yield closure.
`target_reference_audit` counts the highest persisted tree cycle and distinct
event/particle identities before reducing a reference.

`target_bench --source ...` (or `--ref ...`) times the same device CE loop with
setup/cold/warm timing kept separate and verifies repeat identity. Compare its
primary-only scope with the separately measured `production_timing` G4-module
and whole-art log rates; neither comparison is a full background/reco speedup.
No absolute-rate or physics-validity result is claimed by this opt-in generator.

### Parent and source audits without new transport

`target_parent_audit` reads a stock `ReadVirtualDetector` parent tree and selects
exactly Offline `stoppedMuMinusList`: negative muons with stopping process 32.
It checks distinct event/track identities, contiguous event order, target-foil
coverage and an explicit expected art-entry count. Its ordered multiplicities
support circular sequential-window exposure diagnostics matching the production
mixer's sequential read with a random initial skip. A full-pool mean is not the
exact electron denominator for one production window; that requires its skip or
original parent IDs. Neither an upstream generator count nor the number of kept
DIO rows supplies that denominator.

The source-only `fcl/target_rpc_source.fcl` runs the released internal-RPC generator
and its input stop selector, with no Geant4 or detector acceptance filter. Its
stock generator histograms can test energy, momentum components, pair mass and
stop-time/position marginals on an unbiased source population. That is source
validation only, not transport or detector-response closure.

`target_compare` also records base-only/candidate-only selection migrations and
the fast fate of each lost reference selection. Both-arm interior momentum/TOF
checks are explicitly shape diagnostics; they cannot certify an unbiased yield
or attribute a loss to production conditioning rather than transport physics.

With `--reference`, the parent audit can infer an exact production window by
joining every persisted DIO birth position uniquely and exactly (float32, the
stock tree precision) to the parent stop. Every row in each reference file must
imply the same `(parent_event_index - generated_event_offset) modulo pool size`.
Unmatched, ambiguous or inconsistent rows fail; there is no spatial-nearest
fallback. The inferred window's complete multiplicity sum is then the electron
denominator under the audited sequential-mixer contract. This resolves exposure
only; it does not prove production-filter implication or transport validity.
The exposure-corrected `target_yield_compare --exposure parent_audit.json` also
checks the reference hash and event/row counts. It remains explicitly diagnostic
until the VD selection's production-filter implication is established; an exact
denominator does not remove a retained-output selection bias.

`fcl/target_dio_source.fcl` similarly runs only the released DIO generator, using
a pass-all energy switch to activate `GenFilter` plotting (that module otherwise
returns before filling its tree). Its kinematic marginals can be compared with
`target_source_compare --process dio`; the plotting switch is not a production
filter change or a Geant4 sample. Pair-source comparisons use the default
`--process rpc-internal` and released `generate/RPCGun` histograms.

`target_conditioning` compares two independent stochastic draws of the same fast
model on identical starts. It contrasts unconditional selection rates with the
candidate acceptance after selecting base successes. This is a null diagnostic
for outcome conditioning, not a correction for the actual Geant4 production
filter. A conditional loss can coexist with identical unconditional acceptance;
that observation alone cannot assign the measured G4 discrepancy to conditioning.

`target_closure_inventory` reads saved inclusive reference/fast outputs for a
development-only precision inventory. It checks the hash-bound independent census,
complete starts and original-event correlations, then reports own-selection loss
core/tails, empirical CDF distances and explicitly development-coarse joint cells.
Its cap-scaled errors cover event statistics only: they neither quantify parent
uncertainty nor establish simultaneous coverage or certification power. Sparse or
zero tails are marked unsupported, not assigned zero uncertainty.

`target_parent_coverage` exactly joins saved source `stop_idx` identities to an
ordered full-parent table and inventories which original run/subrun groups were
used in development. An unused label is not proof of independent ancestry; every
upstream support and independence claim still needs a separate provenance audit.
The CLI accepts repeated `--draw SOURCE STOPS` pairs and an optional original
catalogue JSON, and refuses to overwrite its strict-JSON receipt.

Art `RootInput` event order need not equal the secondary mixer's physical Events
entry order. `target_parent_audit --parent-art parent.art` can recover that order
from `FileIndex` metadata: every event identity must map uniquely and the physical
entry numbers must cover the Events tree exactly. The exposure join is repeated
in that order and still requires unanimous offsets. This metadata-only join is
not an art physics reader; particle quantities come from the stock extracted tree.
Some historical custom-streamer metadata are byte-reversed by uproot. The explicit
`--fileindex-byteswap` workaround is recorded in the receipt; no automatic repair
or guessed ordering is permitted. Both the complete event-key/entry bijection and
the unanimous birth-derived production offsets must pass after any such adapter.

For a source-population-matched independent check:

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.target_window_stops \
  --parent parent.npz --exposure parent.json --reference reference.npz --out windows.npz
```

This constructs the exact empirical stop population over the audited production
windows. Circular wraps and overlapping windows retain each stop's exposure,
original parent index and production-file index. The receipt and embedded parent
metadata must agree and the reference hash must match. Sampling the complete
parent pool instead is a different source-population pilot, even when the total
electron denominator is exact. Neither choice certifies the detector filter or
reproduces the generator/mixer random history.

`target_rpc_parent_audit` is a limited exact-parent adapter for internal RPC:
each specified physical-pion parent must contain exactly one eligible negative
pion per event, with stopping process 78, time at least 350 ns, zero stop momentum
and target-foil consistency. Empty/multiple-pion cases fail rather than borrowing
a DIO normalization. It checks complete event identities and FileIndex order,
then infers one mixer phase per reference file from uniquely matching births.
Repeated physical stops can occur in distinct parent events: they are resolved
only by that anchored sequential phase, and every birth position **and time**
must then match exactly. There is no nearest-neighbour matching or timing-window
fallback. The output retains each production-window event occurrence, including
wraps; `rpc_event_occurrence` keeps these exposures distinct during the existing
event-then-pion source draw. Original run/subrun/event and parent row stay intact.

An exact internal-RPC receipt enables only an exposure-corrected **electron
marginal diagnostic** in `target_yield_compare`. Either daughter can pass the
production filter; the positron, pair detector response and filter completeness
remain uncertified. The source-only generator pilot does not identify which
parent files were used by an arbitrary retained reference: inspect its persisted
configuration before constructing the matching source population.

### Filter-completeness control

`fcl/target_filter_probe.fcl` reads an existing inclusive art sample, runs only
the production detector-step thresholds, and writes an all-primary
`ReadVirtualDetector` tree plus filter-selected ART. Read that saved ART with
`fcl/target_filter_passed_read.fcl` to produce the passed-primary tree; supply it
to `target_filter_probe` with `--passed-root`. Neither stage generates or transports
particles. `target_filter_probe` compares exact persisted primary identities and
counts VD p/rmax proxy selections missing from the passed arm, with a one-sided
95% exact binomial upper limit. An inclusive CE control can falsify a universal
proxy implication; even zero observed misses cannot certify DIO/RPC at different
energies, stop populations or releases. A future unfiltered matching-background
control must preserve rejected events and raw primary/VD/detector products and
independently audit all statuses before a completeness claim.

### Inclusive matching-background controls

`fcl/target_inclusive_observation.fcl` is an explicit observation overlay for an
already-audited released primary configuration, not a standalone generator.
Its default event cap is zero: a separately approved finite control must supply
the parent, seed and cap explicitly. All-event output keeps raw truth, statuses
and original trigger decisions independently of the stock production filter.
A rejected event must not suppress the independent census. Physics settings
are inherited from the reference's persisted configuration, not a newer default.

`fcl/target_inclusive_read.fcl` reads that saved raw art without generation or
transport. `target_inclusive_ledger` parses its stock PrintModule integer event,
StatusG4 and PrimaryPath records, refusing missing, duplicate or malformed
records. Nonzero status values remain explicit, never ordinary zero response.
`target_inclusive_compare` streams the raw primary/VD tree in bounded chunks,
joins complete integer event identities, counts actual DIO electron multiplicity
and both internal-RPC daughters, and reports proxy selections rejected by the
stock path. The event-level binomial limit is descriptive for that pilot only;
sequential shared parent support and source-library uncertainty are not covered.
No zero-miss pilot establishes a logical filter implication or certifies an
inclusive yield, distribution, full pair response or external RPC conversion.

The inclusive ledger CLI requires the independently captured generation log for
an inclusive comparison (`--generation-log`). It reconciles exact event keys,
statuses and counters despite changed read order; the generation-source hash
must survive into the census. `target_inclusive_transport` then requires that
hash-bound census, complete unique fast row coverage and explicit status-zero
truth. It reports event-clustered paired yield errors, with electron-iid errors
retained only under a named diagnostic field. Event clusters do not establish
between-file or parent-library uncertainty. Original production-path decisions
remain attached to primaries, permitting a clearly labelled retained-population
replay without hiding rejected events from the inclusive denominator.

### Fresh stopped-muon libraries

`target_library_pot.fcl` and `target_library_stops.fcl` define opt-in stock
Run1A proton-to-beam and beam-to-target-stop stages. Their default caps are zero.
Use an explicitly approved finite cap, distinct seed ranges and fresh output
identities for each parent arm. The POT stage inherits the stock upstream field;
this is not necessarily the upstream field used to produce a historical stop
library. Audit saved production settings and state source-population changes
explicitly rather than calling a fresh sample an identical regeneration. The
stop stage uses the DS transport field and retains every selected negative
muon capture without a stop prescale. Independent keep-all output observes every
generated event, including path rejections, alongside the selected art library.

`target_library_read.fcl` audits saved raw output without generation. Override
its process-qualified tags for the POT stage. A selected-parent read uses the
compressed target-stop collection and its matching physical-volume collection.
`target_library_audit` reconciles generation/read integer keys and every status
counter. With `--parent-art`, it also verifies selected ART counts. With
`--parent-root`, `--parent-art` and `--offline-root`, it additionally checks exact
accepted-stop keys, preserves multi-muon multiplicity and recovers physical mixer
order; ledger-only/POT admission does not certify those selected-stop properties. Nonzero statuses or
an empty library block admission; they do not authorize regeneration or refill.
Repeated beam draws retain source event identities but are not independent POT.
Source-library uncertainty clusters by independent fresh proton parent arms,
never by the number of resampled draws. Generation and transport validity remain
separate from absolute POT normalization and from historical-library equivalence.

`target_beam_parent_audit` additionally verifies the saved selected-beam tree
against POT admission's exact accepted event keys. Every event must contain a
PDG +/-13 muon with stopping process `mu2eKillerVolume` (54), matching the stock
beam selector. It inventories eligible multiplicity and momentum/time support;
it does not establish the compressed-volume mixing contract or audit upstream
source-event IDs. This closes a selected-content gap not covered by art entry
counts alone, without rerunning successful generation.

```bash
PYTHONPATH=core:beam python3 -m beamfastsim.target_library_campaign \
  --manifest campaign.json --out inventory.json
```

This reads only a declared completed set of parent arms. The manifest has an `arms` list;
each entry supplies a unique `arm_id`, an `arm_complete` marker and `pot`/`stops`
stage dictionaries. Each stage declares `cap`, `run`, `subrun`, `base_seed`,
`max_unique_engines` and paths `generation_log`, `read_log`, `admission`,
`selected_art`, `generation_complete`, `read_complete`, `selected_read_complete`,
`selected_tree`, `stage_checksums`. The last path is the saved stage `SHA256`
manifest binding selected ART, tree, generation/read logs and admission together.
Stops also supplies `parent_npz`. Paths are relative to the manifest or absolute;
admissions are named `admission.json`. Each cap is an integer from 1 through
20,000, and ledger identities must equal the declared run/subrun with events
1 through cap. The five checksum-bound artifacts (selected ART, selected tree,
generation log, read log and admission) must share the checksum manifest's
directory, with entries keyed by artifact basename. Preserve that manifest
when the stage closes, not after replacing an artifact.

The campaign inventory reconciles all event/status ledgers, exact event ranges,
artifact hashes and embedded parent receipts, selected tree rows, multiplicities
and physical FileIndex ordering. It rejects duplicate keys/artifacts and
overlapping declared seed reservation intervals. It reports foil/time support
without merging parent correlations. Partial or empty arms fail rather than
being refilled or silently omitted. Declared seeds are not a verification of
actual expanded configuration, and neither file labels nor hash differences
prove upstream independence. Source-event ancestry, absolute POT normalization,
historical-library equivalence and physics certification remain explicitly
false/unestablished. The completed inventory is not a live polling tool.

The [confirmation design](target_confirmation.md) records development engineering
margins and the still-unfrozen parent, weighting and simultaneous-statistics gates.
Library construction and passing audit contracts do not establish physics equivalence.
