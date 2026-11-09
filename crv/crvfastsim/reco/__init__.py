"""crvfastsim.reco -- fast CRV reconstruction for the response/alignment BO loop (docs/fastreco.md).

The path from SiPM charges to reco pulses:

    waveform  charges -> waveforms (MakeCrvWaveforms + MakeCrvDigis port) -> pulses (MakeCrvRecoPulses port)

Modules:
    reftables  readers for the CrvRecoTableDumper trees (events, charges, digimc, digis, pulses, clusters, calib)
    pulsefit   batched pulse finding + fixed-iteration Levenberg-Marquardt Gumbel fit (numpy; JAX twin in pulsefit_jax)
    waveform   batched waveform synthesis and digitisation from SiPM charges
    pulsefit_jax  the jitted fixed-shape JAX twin of pulsefit (GPU/CPU)
    synth      KPP-like synthetic waveforms for benchmarks without the reference tables
    compare    V-R1/V-R2 per-channel matching and the threshold summaries (joins/KS from mu2efastsim.match)
    cli        the reco-* subcommands of the package CLI

Run as:  python3 -m crvfastsim reco-<cmd> ...   (alias: python3 -m crvfastsim.reco <cmd>)
"""
