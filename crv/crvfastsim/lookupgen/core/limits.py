"""Limits of the photon-chain random-number counter, without JAX (writers and manifests read them).

The device generator's counter is built from (bin, photon index, step, draw); a chain has fewer than 2**STEP_BITS
steps and a step fewer than 2**DRAW_BITS draws. Longer chains are stopped and counted (transport.Config.max_steps).
"""
STEP_BITS = 19                   # steps per photon chain < 2**19 (longer chains are stopped and counted)
DRAW_BITS = 10                   # random draws per step < 2**10
LEN_BINS = STEP_BITS             # chain-length histograms: bin k holds chains of 2**k ... 2**(k+1) - 1 steps
