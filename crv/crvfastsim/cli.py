"""Single CLI entrypoint:  .venv/bin/python3 -m crvfastsim <subcommand> <inputs> [...]

Validation subcommands (docs/plan.md section 5) and, later, the fast-sim drivers. Run from the repository root so the
package imports (Mu2e machines: $CRVFS_CODE; Polaris: the Eagle clone).
Common flags: --outdir DIR (default plots/<cmd>), --compare REF [...], --emit X.json / --merge for batchable ones.
Fast CRV reco: reco-vr2, reco-vr2stock, reco-vr2jax, reco-bench, reco-vr1, reco-direct, reco-cost, reco-inspect.
"""
import argparse
import sys

from .analyses import census, crossingpos, runchain, g4cost, g4muon, pulsematch, secondaries, tier1b, transport, v0check, v1compare, v2compare, v2origin, v4photons, v4sipm
from .reco import cli as reco_cli

ANALYSES = {
    "g4cost":    (g4cost.run,    "baseline cost of the stock G4 chain from TimeTracker SQLite (sim + digi), per accepted event"),
    "census":    (census.run,    "what reaches the CRV: visible energy and steps by origin (primary muon / descendants / non-muon)"),
    "transport": (transport.run, "fast muon transport on CRY primaries (gen tables) -> CrvStep tables; --stage A = air only"),
    "v0check":   (v0check.run,   "V0: CRV bar geometry (CosmicRayShieldMaker re-implementation) vs the Geant4 GDML, per bar"),
    "v1compare": (v1compare.run, "V1: primary-muon bar crossings (count, path length, visible energy, time) fast sim vs G4"),
    "v2compare": (v2compare.run, "V2: stock CRV response + reco on fast vs G4 CrvSteps (PE, per-bar rates, clusters, layer eff)"),
    "secondaries": (secondaries.run, "tier-1b design input: G4 steps beyond the primary muon (descendants by process/birth, non-muon primaries)"),
    "tier1b-build": (tier1b.build, "tier 1b: build the G4 library (muon descendants per crossing, non-muon showers) from training shards"),
    "tier1b-apply": (tier1b.apply, "tier 1b: add library descendants and non-muon showers to fast tier-1a tables (fast1b.*.root)"),
    "tier1b-check": (tier1b.check, "tier 1b: step-level closure per sector (descendant energy per crossing, non-muon) vs G4"),
    "v2origin":  (v2origin.run,  "V2 diagnostic: reco pulses by deposit origin (muon / descendant / non-muon / none) per PE bin, two arms"),
    "crossingpos": (crossingpos.run, "diagnostic: primary-muon crossing position along/across the bar, fast vs G4 (matched by event and bar)"),
    "pulsematch": (pulsematch.run, "diagnostic: per-channel PE of four V2 arms (fast, fast-mu, ref, ref-mu) on the same events"),
    "v4photons": (v4photons.run, "V4 photon stage: fast photons (response/photons.py) vs stock MakeCrvPhotons (PyROOT) on the same steps"),
    "v4sipm":    (v4sipm.run,    "V4 SiPM-charge stage: fast charges (response/sipm.py) vs stock MakeCrvSiPMCharges (PyROOT) on the same photons"),
    "runchain":  (runchain.run,  "the whole fast chain on CRY gen tables (no art): steps, charges, digis, pulses, clusters; stage costs"),
    "g4muon":    (g4muon.run,    "G4 reference CrvStep tables cut to the primary muon's own steps (tier-1a-like V2 arm)"),
}

# Subcommands that scale via the --emit (worker) / --merge (combiner) batch JSON pattern.
BATCHABLE = {"v1compare", "v2compare", "census"}


def build_parser():
    p = argparse.ArgumentParser(prog="crvfastsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, (fn, help_) in ANALYSES.items():
        sp = sub.add_parser(name, help=help_, description=sys.modules[fn.__module__].__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        sp.add_argument("inputs", nargs="+", help="input files (globs allowed, quote them)")
        sp.add_argument("--outdir", default=None, help="output dir (default plots/<cmd>)")
        sp.add_argument("--counters", default=None,
                        help="Offline CRV counters file (default Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt)")
        if name in BATCHABLE:
            sp.add_argument("--emit", default=None, metavar="JSON", help="write this batch's counts, no plots")
            sp.add_argument("--merge", action="store_true", help="inputs are --emit JSONs: sum and report")
        if name == "transport":
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, i)")
            sp.add_argument("--stage", default="A", choices=["A", "B", "C"],
                            help="world model (A: air, no MSC; B: air + MSC; C: + GDML world material)")
            sp.add_argument("--gdml", default=None, help="GDML world export for --stage C (geometry/README.md)")
        if name == "g4cost":
            sp.add_argument("--digi", required=True, help="the digi job's TimeTracker SQLite (timing_digi.db)")
        if name == "v0check":
            sp.add_argument("--g4log", required=True, help="mu2e log with the crs.verbosityLevel 5 bar printout")
            sp.add_argument("--table", default=None, metavar="CSV", help="also write the candidate bar table")
        if name == "v1compare":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT",
                            help="G4 reference CrvStep tables (crvsteps.*.root)")
        if name == "tier1b-build":
            sp.add_argument("--gen", nargs="+", default=None, metavar="ROOT", help="gen.*.root of the same shards")
            sp.add_argument("--out", default=None, metavar="ROOT", help="library file to write")
            sp.add_argument("--generated", type=int, default=None,
                            help="generated events behind the inputs when --gen is not their own gen stream (production)")
        if name == "v4photons":
            sp.add_argument("--deviation", default=None, metavar="CSV", help="CRVPhoton table, channel,deviation (default 0)")
            sp.add_argument("--max-steps", type=int, default=None, help="first N steps only (the stock arm is a Python loop)")
            sp.add_argument("--jax", action="store_true", help="fast arm = JAX twin (response/photons_jax.py)")
            sp.add_argument("--no-stock", action="store_true", help="reference = numpy port (no Offline needed)")
            sp.add_argument("--seed", type=int, default=1)
        if name == "runchain":
            sp.add_argument("--library", default=None, metavar="ROOT", help="tier-1b library")
            sp.add_argument("--calib", default=None, metavar="CSV", help="CRVCalib + channel map CSV")
            sp.add_argument("--deviation", default=None, metavar="CSV", help="CRVPhoton deviations")
            sp.add_argument("--gdml", default=None, help="GDML world for stage C transport")
            sp.add_argument("--jax", action="store_true", help="JAX twins for photons, SiPM charges, waveforms")
            sp.add_argument("--max-events", type=int, default=None, help="first N generated events per file")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="stock CrvRecoTableDumper files")
        if name == "v4sipm":
            sp.add_argument("--deviation", default=None, metavar="CSV", help="CRVPhoton table, channel,deviation (default 0)")
            sp.add_argument("--max-steps", type=int, default=None, help="first N steps only")
            sp.add_argument("--jax", action="store_true", help="fast arm = JAX twin (response/sipm_jax.py)")
            sp.add_argument("--no-stock", action="store_true", help="reference = numpy port (no Offline needed)")
            sp.add_argument("--seed", type=int, default=1)
        if name == "pulsematch":
            for a in ("--fast", "--fast-mu", "--ref", "--ref-mu"):
                sp.add_argument(a, nargs="+", default=None, metavar="ROOT", help="CrvRecoTableDumper files of that arm")
        if name == "crossingpos":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="G4 CrvStep tables, same shards")
        if name == "tier1b-check":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="G4 CrvStep tables, same shards")
        if name == "tier1b-apply":
            sp.add_argument("--library", default=None, metavar="ROOT", help="library from tier1b-build")
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, 1000+i)")
        if name == "secondaries":
            sp.add_argument("--gen", nargs="+", default=None, metavar="ROOT", help="gen.*.root tables (CRY rates per pdg)")
        if name == "v2origin":
            sp.add_argument("--steps", nargs="+", default=None, metavar="ROOT", help="CrvStep table behind the inputs")
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="reference CrvRecoTableDumper files")
            sp.add_argument("--compare-steps", nargs="+", default=None, metavar="ROOT", help="CrvStep tables behind the reference")
            sp.add_argument("--generated", type=int, default=None, help="generated events behind the inputs")
            sp.add_argument("--compare-generated", type=int, default=None, help="generated events behind the reference")
        if name == "v2compare":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT",
                            help="reference arm CrvRecoTableDumper files")
            sp.add_argument("--steps", nargs="+", default=None, metavar="ROOT",
                            help="CrvStep tables behind the candidate arm (enables the layer-efficiency diagnostic)")
            sp.add_argument("--compare-steps", nargs="+", default=None, metavar="ROOT",
                            help="CrvStep tables behind the reference arm")
            sp.add_argument("--generated", type=int, default=None,
                            help="generated events behind the inputs: rates per generated event")
            sp.add_argument("--compare-generated", type=int, default=None,
                            help="generated events behind the reference (default: --generated)")
    for name in reco_cli.RECO_CMDS:
        sp = sub.add_parser(f"reco-{name}", help=f"fast CRV reco: {name} (reco/cli.py)")
        reco_cli.add_reco_args(sp, name)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd.startswith("reco-"):
        return reco_cli.run_reco(args.cmd[len("reco-"):], args)
    kw = vars(args)
    cmd = kw.pop("cmd")
    inputs = kw.pop("inputs")
    outdir = kw.pop("outdir") or f"plots/{cmd}"
    ANALYSES[cmd][0](inputs, outdir, **kw)


if __name__ == "__main__":
    main()
