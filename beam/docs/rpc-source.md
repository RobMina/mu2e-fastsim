# Internal RPC source

`beamfastsim.rpcspec` is an opt-in source generator, independent of the CE
transport implementation. It follows `RPCGun`, `PionCaptureSpectrum` and
`BinnedSpectrum` in Mu2e Offline v13_00_04. Existing chains and defaults are
unchanged. NumPy is the only runtime dependency of the sampler.

## Physics and inputs

The `RPCInternalPhysical.MDC2025af` configuration uses:

- Bistirlich photon spectrum, 50–139.5 MeV, 0.1 MeV midpoint bins;
- Kroll–Wada–Joseph internal conversion, retaining the correlated electron and
  positron (one pair per generated capture);
- physical negative pion stops (`pionDecayOff: false`), stopping process 78,
  stop time at least 350 ns;
- a selected parent event, then a uniformly selected stopped pion in that event.

The fit ends at 134.530 MeV. As in the production binned spectrum, bins with
zero midpoint density are never sampled. Constants are the release's electron
mass 0.510998910 MeV, pion mass 139.57018 MeV, and Al atomic mass
26.981539 × 931.494028 MeV. This is the production Mg-fit approximation for Al,
not a newly fitted spectrum. The random stream is NumPy's, not CLHEP's; draws
match the distribution, not individual production seeds.

Input NPZ columns are equal-length arrays:

| Columns | Meaning |
|---|---|
| `run`, `subrun`, `evt` | Parent event identity; integral values |
| `pdg`, `at_rest`, `foil`, `codestop` | Species, boolean rest flag, target-foil index, stopping process |
| `xstop`, `ystop`, `zstop`, `tstop` | Position in the Mu2e frame [mm], physical time [ns] |

The caller supplies an already extracted physical-pion table. Muon stops are
not interchangeable with pion stops. Decay-disabled weighted pion samples are
not supported. Times are copied, with no muon decay delay or additional pion
lifetime factor. Sampling is conditional on an event surviving the stop
selection; it does not predict stop yields or an absolute RPC rate.

```bash
python3 -m beamfastsim.rpcspec --stops physical_pion_stops.npz \
  --n 1000000 --seed 20261105 --out rpc_pairs.npz
```

The output has electron `x,y,z,px,py,pz,time,pdg` columns for an explicit-source
transport adapter, plus `p,u`, `positron_p,positron_u,positron_px,positron_py,
positron_pz`, `photon_energy,pair_mass,asymmetry`, and the selected stop index
and event identity. Both particles share the stop position and time. The
`asymmetry` convention is `(E_positron-E_electron)/|p_pair|`, matching the density
argument in Offline; Offline's diagnostic `hy` histogram uses the opposite
sign. No transport is done by this command.

## Tests and saved-source comparison

Synthetic tests cover the photon-spectrum cut and chi2, KWJ support and
rejection envelope, numerical integration of its mass marginal, energy and
invariant-mass conservation, isotropy, event-then-pion sampling, physical-time
copying, stop cuts and repeatability.

For a saved, unfiltered stock `RPCGun` histogram file:

```bash
python3 -m beamfastsim.rpccompare --stock stock_source.root \
  --source rpc_pairs.npz --outdir rpc_source_comparison
```

This writes four source-kinematics overlays, bin pulls, histogram arrays and
`numbers.json`. The two-sample Pearson test uses independent kinematic draws,
equal-area normalization and pooled-null variances. Adjacent bins are pooled
for chi2 where sparse; underflow and overflow count. Pulls remain unpooled.
With 40,000 saved stock captures and 1,000,000 fast captures, chi2/ndf is
73.94/83 (photon energy), 148.97/125 (electron momentum), 138.17/125 (positron
momentum), and 116.38/111 (pair mass). These are source-only development
comparisons, not held-out VD13 validation or a full pair-response result.

## Boundaries

The electron-only CE chain cannot validate the pair's combined detector or
calorimeter response. The stock dts keeps detector-filtered events; a comparison
on those saved starts is conditional, not an inclusive generated-yield result.
Track and pair identities must remain intact when extending the comparison.

External RPC is not implemented here. It starts with a photon, and core
`eslab` returns photons without transporting them: photon conversion to a pair
in the target is missing. An electron placed directly at the pion stop is not
an external-RPC model. Full-chain throughput and a stock grid-slot multiplier
remain unmeasured.
