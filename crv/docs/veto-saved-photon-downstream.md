# Saved-photon downstream timing control

This bounded diagnostic consumes the four-arm photon ledgers from
`veto-optical-timing-probe.md`. It never regenerates optics, stock detector response,
mixtures, coincidences or F. It retains the selected event census but is not a
representative veto-loss sample. C4 and production response defaults are unchanged.

```bash
python3 -m crvfastsim.veto.saved_photon_downstream /data/optical_probe \
  --photon-map /cvmfs/mu2e.opensciencegrid.org/DataFiles/CRVConditions/v6_0/photonMap.root \
  --template /release/Offline/CRVResponse/data/singlePEWaveform_v3.txt \
  --out /data/new_downstream_control
python3 -m crvfastsim.veto.saved_photon_downstream /data/existing_downstream_control \
  --verify --out /data/new_saved_comparison
```

## Coupling and scope

The isolated `replay_sipm_control` and `replay_waveform_control` kernels are ported
without changes from the previously validated bounded controls. Their tests include
pixel/cascade coupling, channel partitioning, thermal insertion, full waveform tails,
zero-suppression starts, ADC clamping and full-length pulse fitting. They do not
replace production kernels.

- Each input has a separate downstream seed. Optical candidate root IDs are the
  saved immutable photon IDs; removal of other photons never shifts a sequential
  RNG stream. Primary/cascade variates use event/channel/root/generation/relative
  pixel keys. Pixel recovery is shared physical state and can change outcomes.
- Pixel policy is **fixed integer map bins only**. Thermal OFF and ON at `1e-4/ns`
  are separate models, not a combined timing/model change. Thermal draws are reused
  across all four arms and evolve the same pixel state as optical candidates.
- Thermal simulation covers the live **union of emitted optical channels**, including
  channels quiet after an arm's cuts. It does not cover all detector channels.
- FEB phase uses event/ROC/FEB keys; electronics noise uses event/channel/absolute
  TDC sample keys, independent of padding or waveform cluster ordinal.
- Historical/rejection-only use saved historical-relative times;
  arrival-only/stock-timing use saved raw-generated wrapped-relative times. All
  saved arm masks remain authoritative. No lower photon cut or waveform-tail clip
  is introduced.
- The SiPM control uses grouped breadth-first fast crosstalk and supports zero
  traps only. No stock queue, RNG or complete detector-response equivalence is claimed.

Every block must have one run/subrun, unique event numbers below `2**20` and at most
ten frames. Full event tuples are checked before the legacy helper's event-only
key is used. Separate per-input seeds prevent cross-block identity collisions.
Geometry/optical inputs were validated by the optical saved-report gate; the map
and single-PE template have exact pinned hashes. Conditions cover every channel,
and live flags must agree with the saved photon ledger.

## Outputs and comparisons

For each of four timing arms and two thermal models, save candidates, every avalanche
attempt, fired charges, complete waveform samples, digi ADCs, fitted pulses,
concatenated fit samples, channel conditions and phases. The optical source event
census and all zero-output channels remain recoverable from the bound inputs and
saved conditions. Summary counts include empty outputs.

Comparisons report additions/removals, exact differences, maximum common-row
residuals and affected event/channel pairs. Complete differing rows are saved.

| Stage | Join identity |
|---|---|
| Charges | event/channel/root/dark/cascade generation/pixel |
| Digi samples | event/channel/start TDC/sample ordinal |
| Pulses | event/channel/concatenated-waveform start TDC/fit sequence |

Pulse joins are **structural waveform/sequence comparisons, not truth matching**.
A shifted waveform start becomes unmatched rather than a nearest-time match.
No assumption of equality is imposed: numerical, ADC or fit changes are results,
not grounds to loosen tolerances. Full pulse fitting is used without the historical
round cap. No stock ADC refit or old boundary control is repeated.

`--verify` hash-checks the saved inputs/assets/code/outputs and reproduces every
stage comparison without simulating response. It does not independently regenerate
avalanche histories. Successful verification is published only after all contrasts
agree; existing output directories are refused.

## Interpretation

Unchanged ADCs can coexist with changed analog samples and charge times. Equal
pulse multiplicities can coexist with changed fitted PE/time. Neither proves
unchanged coincidence membership or F, which are not calculated. Downstream seed
variation, full-sample keyed optics, all-detector thermal coverage, CE joint exposure
and finite-library uncertainty remain separate promotion gates.
