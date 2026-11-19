"""Experiment-neutral core of the strip optical-photon table generator: counter, binning, optics, materials,
transport, run. Imports only the standard library, NumPy, JAX and modules of this package (tests enforce it), so it
can move to its own package. An experiment adapter supplies the counters, the bin edges and numbering, any further
optical models (optics.register) and the table format."""
