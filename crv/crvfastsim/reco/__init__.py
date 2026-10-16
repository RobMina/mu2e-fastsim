"""crvfastsim.reco -- fast CRV reconstruction for the response/alignment BO loop (docs/fastreco.md).

Two prototype paths from SiPM charges to reco pulses, compared on fidelity and cost:

    waveform  charges -> waveforms (MakeCrvWaveforms + MakeCrvDigis port) -> pulses (MakeCrvRecoPulses port)
    direct    charges -> pulses through a response model built from stock (charges, pulses) pairs

Modules:
    reftables  readers for the CrvRecoTableDumper trees (events, charges, digimc, digis, pulses, clusters, calib)
    pulsefit   batched pulse finding + fixed-iteration Levenberg-Marquardt Gumbel fit (numpy; JAX twin in pulsefit_jax)
    waveform   batched waveform synthesis and digitisation from SiPM charges
    direct     the direct charge -> pulse model
    pulsefit_jax  the jitted fixed-shape JAX twin of pulsefit (GPU/CPU)
    synth      KPP-like synthetic waveforms for benchmarks without the reference tables
    compare    V-R1/V-R2 per-channel matching and the threshold summaries (joins/KS from mu2efastsim.match)
    cli        the reco-* subcommands of the package CLI

Run as:  python3 -m crvfastsim reco-<cmd> ...   (alias: python3 -m crvfastsim.reco <cmd>)
"""
