# Opt-in staged on-spill CRV response

`crvfastsim.onspill` consumes already-made CrvSteps from all sources of an event,
not a transport or secondary-library model. It is separate from the default
extracted chain and from fasttables' fixed absolute crop. No defaults are promoted.

Supply `EventTiming(keys, mc_pbtime, reco_pbtime, event_length, period=1695)` with
one row per event, including zero-step events. Full integer run/subrun/event IDs
are mapped to dense row labels, never packed into a fixed-width event field.
`run(steps, pdg, geom, cond, cfg, rng, timing=..., counters_cfg=..., key=...)` uses
existing photon, SiPM, waveform, pulse and coincidence implementations. `pdg`
must be joined by particle identity within each input stream, not by time.
The returned `event_keys` maps every dense output label back to its original ID.

## Timing

For true window start `S=-mc_pbtime`, keep the stock period separate from actual
1675/1700 ns event lengths. Temporary folded endpoints decide whether a step can
contribute to the `[200,1500]` ns ZS gate with 100 ns prestart margin. Optics receives
the original interval, then each arrival is folded into event-local coordinates
with `fmod(t+mc_pbtime,period)` and a negative-remainder correction. NaN endpoint
steps are skipped and counted as in stock; infinite endpoints are rejected.

The SiPM upper limit is 1500 ns; photons before 100 ns are not removed because
they affect pixel recovery. Waveforms use the local FEB phase and inclusive
`[200,1500]` start gate; digi tails are not truncated at the gate. The opt-in JAX
waveform scan applies this gate before consuming a 12-sample digi, not as a
post-scan crop: a rejected pre-window start must not hide an in-window hit.
The stock `--i` after sample collection keeps the accepted-start advance at
12 samples, not 13. Default offspill kernel calls retain their existing path. Reco adds
`-reco_pbtime` separately from DB offset enable/scale, before coincidences.
Do not use smeared reconstructed bunch time for response or true bunch time for
reco. Returned charge/waveform times are local; detector TDCs need no translation.

## Thermal noise and scope

Stock on-spill thermal rate is 1e-4/ns/SiPM. The adapter supplies every physically
present, status-eligible event/channel label, including zero-step events. Thermal
attempts are Poisson with rate divided by the fully-charged firing probability;
times are uniform over `[100,1500]` and pixels uniform over the sensor, not its
fibre map. They share the photon recovery/cross-talk state, including in JAX.
Dark charges and descendants carry photon index -1. `sipm_jax.simulate` adds
keyword-only `t_start` and `thermal_chans`; default zero-thermal calls retain their
RNG/output path. Thermal device overflow raises rather than dropping candidates
or silently redrawing them in a host fallback.

Fused response, trap afterpulses, NZS and empirical post-fit background are not
supported by this adapter. The global default response contract still rejects
thermal noise; callers opt into this explicit complete-population path. The
supported kernel capacity/retry policy and precision settings are unchanged.
JAX thermal attempt generation, packing and bookkeeping still use the host;
this is GPU-backed, not fully device-resident.

Tests cover full-ID collisions, event windows, negative times and period crossings,
post-optics folding, true/reco separation, empty events, thermal statistics and
shared optical/dark recovery, pixel distributions, overflow and zero-thermal
regression. These synthetic tests do not certify stock physics closure or GPU
throughput on a mixed sample; the umbrella development driver provides that check.
