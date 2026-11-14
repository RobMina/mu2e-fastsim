# Fixed device shapes (capacity files)

`runchain` on the JAX path pads each device call to a power-of-two bucket of its own size: photon rows, substeps and
arrivals per kernel chunk, the photons of a file, the SiPM chunk, waveform clusters and samples per call, and the
refit rows and samples. A new input file brings new bucket combinations, so a warm pass over new events recompiles.
On an A100 that was 31-47 compiles and 11-18 s per pass of 200,000 generated cosmics, a third of the pass.

With a capacity file every such call takes its shape from the file instead (`crvfastsim.capacity`):

```bash
python3 -m crvfastsim.analyses.runchain_bench gen.*.root ... --resident-response --capacities run1a_v01
python3 -m crvfastsim runchain gen.root ... --jax --capacities run1a_v01
```

`run1a_v01` names the packaged file `crvfastsim/configs/capacities_run1a_v01_runchain.json`; a path to any other file
works too. The resident response is required (it is turned on with the option).

## Draws that do not depend on the shapes

Padding a buffer does not move a random number when the draw is indexed by the hit itself. Two draws of the default path
are indexed by the padded layout, so with fixed shapes they are drawn differently:

- SiPM (`sipm_jax.device_sipm(draws="section")`): the firing uniforms of the photons and of each cross-talk generation
  come from their own keys, and the lrint pixel offsets are drawn as `[photons, 2]`. In the default path a cross-talk
  candidate's uniform sits after the padded photon buffer, so it moves when that buffer does.
- Waveform noise: one key per charge cluster (`fold_in(seed key, cluster index)`), drawn over the cluster's samples, so
  it does not depend on how clusters are grouped into calls or padded.

With those, the charges and digis are the same for any capacity file, and for the bucketed shapes
(`--record-capacities`). This is another realization of the same model than the default path, so the default is
unchanged and the mode is opt-in. Its held-out check against the default draws and stock is in the PR that added it.

## Making a capacity file

```bash
python3 -m crvfastsim.analyses.runchain_bench gen.*.root ... --record-capacities --out census.json
python3 -m crvfastsim.capacity census*.json --margin 1.25 --out caps.json
```

`--record-capacities` runs the capacity draws with bucketed shapes and records the largest demand of every buffer
(`runchain.capacity.usage` in the report). Each capacity is `ceil(margin x largest demand)`; sample lengths are rounded
up to 8. The substep chunk is clamped to the kernel's chunk limit (2^22), and the SiPM chunk size (`sipm_photons`, also
the chunk length of the resident SiPM loop) is set by `--sipm-photons` (default 2^22), not derived.

The coincidence hit buffers (`coinc_hits`, one per tier group size H: the hits of one device-packed kernel batch,
first-choice or promotion) are derived the same way for the tiers with large batches. The tiers whose largest
possible batch is small (`run1a_v01`: H 128 and up, at most 32,768 hits) get that batch x H, so they never overflow:
their demand depends on how many groups the chunks of a run promote together, which the census underestimates, and the
device builds a batch x H array for them anyway.

A call whose demand exceeds its capacity is not truncated. It runs with the bucketed shape: the output is the same,
but that call compiles. It is counted in `runchain.capacity.overflow`, so a file that is too small shows in the report.

## Bounded pulse scans

In capacity mode both pulse fits (the 12-sample fit and the device refit of flagged waveforms) run with
`pulsefit_jax.reconstruct_dense(bounded=True)`. The peak scans stop at the longest waveform, and the rounds stop once
no waveform found a pulse (a waveform that finds none never finds one later). The pulses are the same:
`runchain --save-reco` outputs are byte-identical with and without on one GPU, float32 and float64. The refit's 16
rounds no longer run over every padded row.
