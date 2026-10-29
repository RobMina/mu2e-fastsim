# Saved exposure metadata: available evidence and remaining contracts

`veto.exposure_metadata_inventory` reads two externally pinned saved JSON reports,
twelve exact-file SAM metadata texts and the footer of their bound exposure ledger.
It does not query SAM, read ntuples/selected tracks, decode ledger rows, recalculate
parent overlaps, rerun an earlier audit or generate a mixture.

## Contract

Derive only the twelve explicitly bound `sam_<sequencer>.txt` names. Check each
full source basename, full `dh.sequencer`, original metadata hash, event count,
file identity and sorted immediate-parent hash/count. Preserve every non-Parents
field, including checksum continuation lines and `dh.gencount`; do not reduce
metadata to a keyword-based claim that no other information exists.

The JSON inputs are each below 8 MiB, SAM texts each below 1 MiB, and their total
below 16 MiB before hashing. The ledger is at most 32 MiB; footer limits are
100,000 rows, 32 columns, 4,096 row groups and 64 MiB declared uncompressed data.
Only metadata/schema are decoded. The original selection path/hash and previous
diagnostics remain explicitly inherited, not freshly validated observations.
Selected-track and ntuple paths are never opened, hashed or inspected.

Source snapshots and consumed evidence are rechecked before exclusive JSON output,
limited to 1 MiB outside all evidence/source/selected-track directories. There is
no overwrite, atomic-completion or crash-durability claim. These bounds do not
promise a whole-process RSS ceiling.

```bash
python3 -m crvfastsim.veto.exposure_metadata_inventory /data/SAVED_METADATA \
  --expected-provenance-sha256 PROVENANCE_REPORT_SHA \
  --expected-selection-sha256 SELECTION_REPORT_SHA \
  --out /data/SEPARATE_OUTPUT/NEW_INVENTORY.json
```

## Completed observations

All twelve retained Neutrals metadata records declare:

- `dh.configuration = MDC2025ad`;
- `dh.dataset = dts.mu2e.NeutralsFlashCat.MDC2025ad.art`;
- `dh.gencount = 6700000000`;
- 5,000 immediate parents, matching each previously recorded parent-set hash.

The generation count was available in the full metadata but omitted from the
previous compact provenance summary. Its presence is useful, **not a certificate
of unique primaries, independent showers or additive POT**. No sum over aliases,
exposure renormalization or new source-weight model is formed. The inventory does
not establish intended reuse, deeper ancestry or cross-species independence.

The existing exposure ledger footer declares 91,226 rows, one row group and seven
fields: `run`, `subrun`, `event`, `nprotons`, `pbtime`, `measured_nprotons`,
`measured_pbtime`. It has no per-row source-file identifier or track-time field.
The original selected-track file remains a separate, unopened input. The prior
report's observation that measured `nprotons` is identically zero is inherited;
MC exposure stratification therefore remains a truth diagnostic, not a validated
measured-variable correction. Header metadata alone does not revalidate identity
uniqueness, field values, selected-track joins or time conventions.

All 70 targeted tests pass locally (0.94 s) and remotely (2.98 s), with synthetic
SAM records and real Parquet footers. Real inventory: 149,354 output bytes,
1.79 s whole-process wall time and 204,332 KiB peak RSS. An older remote PyArrow
rejected a new optional constructor keyword during tests; the portable footer-only
call now retains the same metadata bounds and no payload decoding. No real
inventory failed or was repeated for this portability fix.

## What must be bound next

1. **Generation-count semantics and reuse:** obtain the historical generated
   resampler and concatenation configurations, Production/Offline releases,
   exact input/subrun allocation, wrapping/skipping/event limits, and the metadata
   extraction rule that produced `dh.gencount`. Generic current source is not a
   substitute for the deployed campaign configuration.
2. **Deeper ancestry and uncertainty units:** trace the retained parent groups far
   enough to identify shared upstream generation, including across species. Define
   the physical reuse weights and resampling population before assigning library
   covariance. Disjoint immediate parents alone do not prove independence.
3. **Measured CE exposure:** bind the retained ntuple writer/configuration and
   determine whether a calibrated measured intensity observable was persisted.
   A default/unfilled zero field cannot become a correction variable. Establish
   bunch-time and selected-track time conventions before any joint-distribution
   diagnostic; MC-conditioned variants must remain explicitly truth-labelled.

A future joint truth diagnostic can reuse the saved exposure ledger and frozen
track selection without rereading ROOT, but it would require deliberate track
access and a separately specified matching/binning contract. It would not recover
a missing measured observable or justify multiplying independent marginals.

Historical retained-sample F remains conditional on its library and tracks.
`F=null`, `weight_model=null` and `finite_library_uncertainty=null` in this inventory;
C4 stays stock. No new simulation or broad extraction is authorized by this audit.
