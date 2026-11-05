# Cosmic production control: distinct era and bounded readiness

The inventoried cosmic production is not the CE production with different event names. Its embedded configuration
and matching successful digitization log identify SimJob MDC2025au / Offline 13_33, environment p103, ROOT 6.32
and KinKal 3.7. The CE control uses Offline 13_41. Read the actual production-era modules before asserting equal
semantics, even though all 52 physics path names/bits/order/prescales agree. Seventeen shared module settings and
geometry differ in the recorded inventories.

## Complete source prerequisites

The matching cosmic job processed 118,934 stored preselection events from 100 exact resampled dts parents.
Catalogue parent size totals 4,870,760,515 bytes. The catalogue generated count is 500 million, not the retained
trigger denominator. Parent filenames do not imply a single subrun per parent: default numeric source traversal
must be checked against independently decoded ROOT identities within each parent, in actual opened-file order.
The original SeedService starts at one, not the CE control's seed. Preserve the entire original stochastic
module set, conditions, geometry and job-start history.

Production source names are worker-local; resolve exact catalogue DIDs to actual storage class paths, preserving
configured names separately for history validation. A path returned for the wrong storage class can be absent.
A transfer stalled after a cached prefix and has no complete source SHA receipt. Such a partial source is not
admitted to the complete-history control. A catalogue size/Adler-32 check and measured destination SHA can establish
cached bytes for a bounded startup test, but cannot be relabelled a verified source SHA receipt or full coverage.
No replacement Geant4 or counter-offset fitting is justified by storage unavailability.

## Calorimeter units differ

This older release stores integer CaloDigi `t0_` in ns; its digitizer writes a sampled time and fast hit maker uses
`t0 + peakpos * digiSampling - bunchOffset`. The newer CE release stores clock ticks and uses
`(t0 + peakpos) * digitizationPeriod - bunchOffset - channelOffset`. Do not normalize one into the other in a raw
serialized comparison or inject digis using the wrong era's unit convention.

## What a startup probe establishes

A 100-event job-start prefix with the original full digitization configuration can test the actual released
runtime, FCL and retained branch schema using a checksum-checked cached first parent. It is deliberately not
production-overlap reproducibility, full-history trigger agreement, cosmic efficiency or fast-digi validation.
The scoped startup probe completes successfully with 100 highest-cycle Events, all six required serialized
products present and the observed 78-byte EventAuxiliary layout. CaloDigi schema version one contains `t0_`,
consistent with the older ns contract. This checks startup/schema readiness only, not original payload equality.
Successful art is not enough: independently check finished highest-cycle Events and serialized metadata.
The original selected output and complete parent receipt remain required for the real control.

The matching cosmic production log attributes 10.2411 ms of trigger-module wall work per processed input event,
with 53.324 ms full-event mean. The robust multi-helix finder contributes 28.8%, merged-pattern helix fit 14.5%.
These are shared-label invocation-weighted attributions, not isolated path latency or a fast/stock multiplier.
