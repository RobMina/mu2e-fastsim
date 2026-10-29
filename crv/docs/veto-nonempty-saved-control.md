# Bounded nonempty saved-control audit

This separate diagnostic consumes existing legacy pulse, recovered-membership and
same-pulse stock-finder products. Stock members are persisted native indices;
fast members were previously recovered and summary-validated, not persisted by
the original fast draw and not truth matches. This audit executes no finder,
response, fit, optical simulation or stock job. It does not change the completed
timing pilot.

## Admission and scope

The historical pulse schema is `ev, chan, time, pe, seq`. It does **not** retain
waveform identity, start TDC, original relative finder time or fit-quality flags.
The diagnostic stock adapter deliberately has empty digis and waveform links.
Those missing values must not be invented to satisfy `pilot_coincidence`.
The table-import adapter inserts every retained pulse without the production
pulse-finder's ignore-channel filtering. Native links with channel-status bits
are retained and counted, not silently removed. This historical policy is not the
new pilot policy, even with identical calibration tables. Consequently the new
adapter's nonempty structural-identity and channel-policy gates remain open.
Stock `poscalc` is int32 with values 0/1; recovery uses bool. Validate the domain
before normalizing that storage difference, without tolerances.

The new audit instead pins the historical recovery report and stock check, verifies
the saved pulse collection against the stock adapter in native per-event order
(with its explicit float32 PE serialization), and compares already-saved native
member sets. Original float64 PE is preserved separately. Time is the retained
reco-frame time; the legacy helper's reconstructed finder time is not used. It checks exact
full calibration equality between the retained control and bound fixed stock.
Numerical cluster properties are reported as exact mismatch counts and residuals,
never tolerance-promoted equality.

The complete frozen source-qualified registry and selected-track grid are reused
from the completed saved-pilot transaction and its externally pinned verifier.
No raw-source admission or track histogram regeneration is repeated. The input
payload hash is inherited provenance, not a new raw ROOT verification.

## Bounded selection and resources

Read at most one historical block: at most 3,000 census rows, 100,000 pulses,
1,000 coincidences and 5,120 calibration rows. Each ROOT file is at most 128 MiB;
each admitted tree's declared uncompressed bytes are at most 128 MiB before array
reads. Parquet admission limits file/uncompressed bytes and rows before decoding.
Counters text is capped at 2 MiB before parsing. These caps do not guarantee
process RSS. The ROOT file hashes are streamed before decode; no raw mixture is
opened.

Select at most four whole frames, using saved observations: the earliest
nonempty stock frame, earliest documented opposite-side attachment, earliest
passing frame with positive eligible historical-grid coverage, and earliest
zero-coincidence frame. Duplicate selections are one frame with multiple roles.
No row is dropped from a selected frame. Require at most 512 pulses per selected
frame and 2,048 total. This is deliberately nonrepresentative validation selection,
not an exposure sample. Missing required roles fail rather than silently reducing
scope. Coverage uses the unchanged historical grid and independent occupancy-mask
parity check. Both saved stock and saved recovered clusters are evaluated.

All read artifacts and executing source files are hash-checked before and after;
the pilot transaction inventory is rechecked. A new exclusive atomic output has
a 4 MiB pathname-byte cap and includes selected pulses, clusters, links, coverage,
source snapshots and summary. No F, error bar, finite-library uncertainty,
full-campaign admission or promotion certificate is produced. C4 stays stock.
