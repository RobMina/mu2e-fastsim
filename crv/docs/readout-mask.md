# Physical readout mask

Offline `v13_39_00`, `CRVResponse/src/CrvSiPMChargeGenerator_module.cc`, checks `bar.hasCMB(SiPM % 2)` before the
channel status: a SiPM that is not physically there makes no charges, whatever its calibration status. Presence comes
from the counters file (`crs.sipmsAtSide{0,1}<sector>`, read by `CosmicRayShieldMaker`), not from calibration
status, the channel map or an optical reflector. Run 2 R4, for example, has reflector 0 but no side-1 sensors, so
its SiPMs 1 and 3 must not produce charges even when mapped and status-good.

## Default

`runchain` (including prepared contexts and `e2e-bench`) and `valhist.fasttables` apply the mask of the counters file
in use to every configuration. `readout.apply(conditions, geom, counters_cfg)` returns new conditions with
`readout_mask` set; channels beyond `4 * geom.nbars` stay present, conditions covering fewer channels are refused.
`readout.effective_status(conditions)` ORs status bit 0 into absent channels (a temporary execution status; the stored
status is not changed), and `chain.charges` applies it at its pre-SiPM boundary for NumPy, nonfused JAX and the fused
device response. Photon generation is unchanged, so the mask does not change any random draw for present channels.

`readout.sensor_mask(geom, counters_cfg)` returns the dense boolean mask of length `4 * geom.nbars` (entry
`4 * b + s` from `crs.sipmsAtSide{s % 2}{sector of b}`). Every sector must supply both flags as booleans; missing
flags, invalid geometry, duplicate names and unknown sector indices raise `ValueError`, and `crs.sectorNames`, when
present, must match the geometry's names and order.

## Effect by configuration

- **Extracted (`extracted_v04`) and Run 1A**: no change. The single-sided extracted sectors (T1 without side 0, M1-M8
  without side 1) have reflectors at their missing ends, so the photon stage routes no photons to an absent SiPM
  (`tests/test_readout_default.py`). Output is byte-identical with and without the mask (see
  [validation.md](validation.md#physical-readout-mask)).
- **Run 2 (`run2_v10`)**: 2,112 of 21,376 channels are absent. R4 (reflector 0, no side-1 sensors) loses the
  unphysical side-1 output; on a saved 1,000-event Run 2 population the R4 summed pulse PE moved to within
  -0.5 to -0.7 % of stock in every backend. The named configurations still have no tier-1b library, so this is not
  an end-to-end validation of Run 2.

## Tests

`tests/test_readout_mask.py` (validation and routing), `tests/test_readout_execution.py` (the NumPy, nonfused JAX and
fused JAX charge paths on CPU with synthetic tables: surviving and forbidden channels, status bits, unmodified
conditions, identical output and random stream for no mask versus all-true) and `tests/test_readout_default.py`
(loaders, prepared contexts, the extracted and Run 1A no-op).
