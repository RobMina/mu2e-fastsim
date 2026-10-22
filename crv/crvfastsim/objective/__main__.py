import argparse

from . import crvresponse


def main(argv=None):
    p = argparse.ArgumentParser(prog="crvfastsim.objective", description=crvresponse.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="EventNtuple files, or accumulator JSONs with --merge (globs allowed)")
    p.add_argument("--outdir", default="plots/crvresponse")
    p.add_argument("--emit", default=None, help="write the accumulator state to this JSON")
    p.add_argument("--merge", action="store_true", help="inputs are accumulator JSONs: sum them")
    p.add_argument("--compare", nargs="*", default=None, help="reference accumulator JSONs (objective of inputs vs these)")
    p.add_argument("--n0", type=float, default=crvresponse.N0, help="nominal entries of the size-free chi2/ndf scale")
    p.add_argument("--entry-stop", type=int, default=None, dest="entry_stop")
    p.add_argument("--window-ns", type=float, default=None, dest="window_ns")
    p.add_argument("--time-corr", default=None, help="timecal correction JSON ({'correction': [16 ns]}) added to pulse and "
                   "cluster times per module before accumulating (class-D module time calibration)")
    a = p.parse_args(argv)
    corr = None
    if a.time_corr:
        import json
        corr = json.load(open(a.time_corr))["correction"]
    return crvresponse.run(a.inputs, a.outdir, emit=a.emit, merge=a.merge, compare=a.compare, entry_stop=a.entry_stop,
                           window_ns=a.window_ns, n0=a.n0, time_corr=corr)


if __name__ == "__main__":
    main()
