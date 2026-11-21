"""mu2efastsim.bench: the warm-rate timer of hep-simbench (hepsimbench.bench) with Mu2e's per-workflow GPU launch
settings (mu2efastsim.devloops) as an extension. Every name, option and report field is as before the extraction.

    python3 -m mu2efastsim.bench MODULE:FUNC --events N [--prepare MODULE:FUNC] [--kwargs JSON] [--dtype float32]
                                 [--workflow W2] [--boundary "stops -> reco CE track"] [--output-included]
                                 [--fresh-inputs [--seed S]] [--passes 3] [--out bench.json]
                                 [--device-loops auto|on|off] [--graph-concurrency auto|on|off]
                                 [--approx-preset default|fast|fidelity] [--approx NAME=LEVEL ...]
                                 [--approximations FILE]

See hepsimbench.bench for what is measured and reported; this module adds --device-loops and --graph-concurrency
(mu2efastsim.devloops, chosen before the entry point creates a backend) and the report's device_loops and
graph_concurrency fields.
"""
from hepsimbench.bench import *  # noqa: F401,F403
from hepsimbench.bench import Extension
from hepsimbench.bench import main as _main

PROG = "python3 -m mu2efastsim.bench"


class DevLoops(Extension):
    """XLA device-side loops and concurrent graph regions per workflow (mu2efastsim.devloops)."""

    def add_arguments(self, p):
        p.add_argument("--device-loops", choices=("auto", "on", "off"), default="auto",
                       help="XLA device-side loops (mu2efastsim.devloops): auto (default) picks the faster setting "
                            "for --workflow; on or off forces it")
        p.add_argument("--graph-concurrency", choices=("auto", "on", "off"), default="auto",
                       help="XLA concurrent graph regions (mu2efastsim.devloops): auto (default) follows --workflow; "
                            "on or off forces it (never on together with device loops for the CE chain, W2-W4)")

    def before(self, a):
        from . import devloops
        devloops.select(a.workflow, a.device_loops, a.graph_concurrency)

    def report(self, a):
        from . import devloops
        return {"device_loops": devloops.state(), "graph_concurrency": devloops.concurrency_state()}


def main(argv=None):
    return _main(argv, prog=PROG, extensions=(DevLoops(),))


if __name__ == "__main__":
    raise SystemExit(main())
