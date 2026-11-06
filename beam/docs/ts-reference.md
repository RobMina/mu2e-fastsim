# TS stock reference and paired comparison (opt-in)

`beamfastsim.tsreference` reads the two-stage Run1B TS1Beam -> TS1ToTS5s100
reference. Use entrance files and the downstream file that actually consumed them.
Retain every entrance, including those without downstream descendants. The
`tsread.fcl` stock analyzer reads actual Beam StepPointMC pre-step states, not
SimParticle terminal states. No Geant4 producer is run by that FCL. Its fixed
positive SeedService seed permits initialization of the imported simulation services;
it does not rerun or reseed the saved simulation.

```bash
python3 -m beamfastsim.tsreference --source source.root --downstream downstream.root \
  --source-art source_inputs/*.art --out reference.npz
python3 -m beamfastsim.tscompare --reference reference.npz --fast fast.npz \
  --shards SUBRUN_IDS --label development --geometry-label Run1B --out comparison
```

The fast NPZ must carry aligned input IDs, shard IDs, terminal state and fate.
Identity/ancestry joins use exact event/SimParticle keys, never spatial or time
proximity. The parent art Events census retains empty-source events. Missing
endpoints are zero outcomes, not deleted rows. Restart-stage identity rules must
not be applied to a same-stage raw Run1A reference.

`tscompare` reports charge-specific yields, momentum/time histograms and pulls
with paired independent-shard estimating-equation errors. Rates are normalized
to the corresponding source-charge cohort, not POT. Spectral diagonal chi2
includes yield and ignores bin covariance; it is descriptive, not a calibrated
p-value or an equivalence test. Keep under/overflows in total yields. Sparse
positive-charge endpoints do not establish precision agreement.

Choose settings on development shards; report disjoint held-out shards once.
No import changes the default beam/CE chains. Run tests from the umbrella with
the pinned core and `beam/` on `PYTHONPATH`:

```bash
python3 -m pytest -q beam/tests/test_tsreference.py beam/tests/test_tscompare.py
```
