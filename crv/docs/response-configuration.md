# Versioned response configuration (thread C)

The production `reco-r3` path resolves `crvfastsim/configs/response_v1.json` before loading physics inputs. This
shipped JSON, not a hand-edited Python source file, is the nominal R3 preset. Existing dataclasses remain the
low-level API; a regression test keeps their compatibility defaults equal to the preset. This is the extracted
CRV response/reconstruction configuration, not a general Run1/Run2/transport/beam configuration migration.

## Use and precedence

```bash
python3 -m crvfastsim.response_config > response.json
python3 -m crvfastsim.response_config --describe
python3 -m crvfastsim reco-r3 INPUT.root --response-config response.json --calib calib.csv \
  --deviation photon.csv --objective --v05 --knob atten_T1=0.6 --emit result.json
```

A file may contain only changes, but must declare `"schema": 1`. For example:

```json
{
  "schema": 1,
  "sipm": {"overvoltage": 2.9, "p1": 0.607, "p2": 2.7, "capacitance": 8.84e-14},
  "response": {"photon_yield_variation_scale": 1.0},
  "timing": {"feb_phase_width": 10.0},
  "pulse_reco": {"min_adc_difference": 42.0},
  "finder": {"fiber_signal_speed": 140.0, "cuts": {"T1": {"pe_threshold": 9.0}}}
}
```

Precedence: shipped nominal preset -> JSON changes -> existing `--knob` overrides -> explicit CLI assets,
`--v05` (per-event clock phase, zero DB offsets) and `--no-noise`. Omitting `--no-noise` preserves the file setting.
Paths are resolved in the job's working directory, not relative to the config file; use absolute asset paths for
standalone jobs. Both the Python `r3.run(..., response_config=...)` and CLI accept a path or inline JSON; Python
also accepts a dictionary. Unknown keys, wrong types/units-by-schema, nonfinite numbers, duplicate JSON keys,
inconsistent bounds and unsupported settings fail before physics I/O. Numeric fields use the fixed units listed
by `--describe`; strings like `"3 V"` are invalid. This is structural/implementation validation, not a physics prior.

The existing kit task adds **one fixed `response_config` parameter**, as an object or inline JSON, validated
identically for EAF and the UVA delegate. Kit-host file paths are refused (they are not automatically staged).
The kit serializes the full config as one shell-quoted argument. Its period/conditions, background switch, v05
convention and existing scalar knobs keep their previous precedence. A custom kit config cannot override staged
asset paths or waveform files; use the explicit-input local/FNAL validation path for asset changes. No new scalar
BO dimensions are registered. A study must explicitly authorize a parameter/selection change and use a new
measurement definition when appropriate; do not mix configurations into an existing board merely because the kit
version is unchanged.

## Audit disposition

| Audit group | Schema-v1 handling |
|---|---|
| Yield/attenuation/optical speed, channel-deviation scale | Configurable in `response`; existing knobs override. Sector dictionaries include EX/T1/T2 and all eight taggers; module propagation overrides retain precedence. |
| SiPM overvoltage, firing-law p1/p2, capacitance, recovery and cross talk | Configurable in `sipm`, both response backends; reject nonphysical derived firing/CT probabilities and firing below 1e-12 (runtime numerical support). Strongly coupled to yield/gain/calibration; not automatically identifiable or commissioned. |
| Thermal rate/trap probabilities; trap lifetimes | Probabilities must stay zero. Lifetimes frozen because the process is inactive; no fictitious objective-sensitive tuning control. |
| Pixel layout/inactive map | Recorded but frozen, including exact integer element types; kernels retain fixed 41-index representation. |
| Waveform stretch, reference charge, threshold, electronic noise, ADC gain/pedestal | Configurable in `waveform`; template path is an asset, not a numeric knob. Template sampling/support and ADC/ZS structure frozen. |
| FEB phase | `timing.feb_phase_width` is a uniform width in ns, 0..12.5; scope is per FEB or per event. This is not a new arbitrary jitter distribution. Default preserves RNG draws exactly. |
| Pulse finder | Thresholds, beta/time/height bounds, fit-window samples and undershoot threshold configurable; passed to NumPy, JAX and full-length retry. |
| Pulse leading-edge factor | Recorded/frozen: R3 exports peak time, not `letime`; changing it would not change the objective. |
| Coincidence finder | Fibre signal speed (distinct from optical propagation), time interval/offset and all sector cuts configurable, used by both chain backends. Changing layer cuts can invalidate the fixed objective-term contract; failure is intentional, not permission to drop terms. |
| DB/ignored-channel/secondary policies, solver/capacity/precision | Recorded and frozen stock-compatibility or numerical controls, not physical BO parameters. |
| Calibration/deviation/time shifts, optical tables/maps, waveform template/background | Explicit named assets for standalone R3, resolved path and SHA256 recorded. Optical tables retain v6/extracted lengths, and photon maps require finite nonnegative 40x40 weights. Changing bytes is not by itself validation. Kit uses its staged assets. |
| Geometry/counters | Resolved extracted-v04 asset recorded; schema-v1 rejects geometry-path overrides because the objective hardcodes extracted bar decoding. General geometry requires a separate validated objective contract. |
| Geometry/world/material/Birks, production mixing/beam distributions | Not configurable by this cached-step response task. Need separate transport/assets/contracts; visible energy cannot be changed retroactively. |

The empirical background is added after waveform fitting; it is not regenerated by a change of SiPM/waveform knobs.
Treat background assumptions as model discrepancy, not detector calibration. The window is fixed to extracted
0..100000 ns for R3. Broader windows and arbitrary sensor/optical geometry remain unsupported.

## Results and reproducibility

`r3.json` gains `resolved_config`: schema, full effective values, canonical SHA256, implementation-support label,
and resolved SHA256s for calibration/deviation/background/time-shift/counters/template/photon-map/LUT files.
This configuration hash is **not** an automatically installed Mu2eBO board hash and is not a substitute for code,
input-event, reference or container identity. It records what ran. `objective/validation.py` continues to provide
separate immutable code/input provenance for explicit-list validation. No running replica checkout is modified.

Full-file nominal CPU equivalence was executed on GPVM07 with candidate CRV `0d58650`, core `d629620`, against
the saved pre-refactor closure (CRV `8d42b1b`, core `a554571`). Same input subrun 990, seed 20261025, one pass,
22,573 step + 2,603 empty = 25,176 exposure events, 30,145 clusters: **the entire `response_state` is exactly equal**.
Candidate wall 222.66 s on GPVM07 versus reference 247.84 s on GPVM04 is not a controlled speed comparison.
Receipt: `$CRVFS_DATA/thread_c_validation_20261025/config_nominal_replay.json`; candidate output
`config_nominal_cpu/r3.json` records the resolved config and eight asset hashes. Changed-settings physics and GPU
nominal equivalence remain separate gates; the ongoing held-out replicas keep their pre-refactor checkout frozen.

Local targeted CRV tests: 183 passed, 1 skipped. Isolated GPVM config/wiring/scoring/kit/comparison tests:
172 passed, 10 JAX-dependent skips; core config/admission tests: 87 passed. Existing multiprocessing tests warn
about fork/thread interaction but pass. No builds were run.

Tests cover nominal dataclass equality, old knob-map equivalence, default seeded phase identity, waveform/pulse
backend forwarding including full-length fallback, R3 resolution/serialization, all-sector cuts, strict JSON/types,
unsupported settings, inactive fields and derived-probability limits. Core admission tests check plain/delegate
refusals before submit, shell-token serialization and no accidental new BO dimensions. Full physics validation of
changed settings and parameter identifiability remains necessary; CPU synthetic tests do not certify GPU support
at every proposed physical value.
