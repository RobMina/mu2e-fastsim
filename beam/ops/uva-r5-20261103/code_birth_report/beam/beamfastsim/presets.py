"""Named layer-physics presets of transport.Params (no JAX import, for CLIs and task definitions)."""
PHYSICS = {"legacy": dict(mean="local", law="fast", msc_path=False),
           "g4": dict(mean="range", law="g4", msc_path=True)}
DEFAULT_PHYSICS = "legacy"
