"""Single CLI entrypoint:  .venv/bin/python3 -m crvfastsim <subcommand> <inputs> [...]

Validation subcommands (docs/plan.md section 5) and, later, the fast-sim drivers. Run from the repository root so the
package imports (Mu2e machines: $CRVFS_CODE; Polaris: the Eagle clone).
Common flags: --outdir DIR (default plots/<cmd>), --compare REF [...], --emit X.json / --merge for batchable ones.
Fast CRV reco: reco-vr2, reco-vr2stock, reco-vr2jax, reco-bench, reco-vr1, reco-direct, reco-cost, reco-inspect.
"""
import argparse
import sys

from .analyses import census, crossingpos, runchain, g4cost, g4muon, pulsematch, secondaries, tier1b, transport, v0check, v1compare, v2compare, v2origin, v4photons, v4sipm
from . import tier1b_assets as T1B
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
    "runchain-merge": (runchain.merge, "combine runchain --emit files of a node job: totals and accepted events per node-hour"),
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
            sp.add_argument("--world-mods", default=None, help="--stage C: named world modification (trkfastsim.worldmods)")
            sp.add_argument("--run-config", default=None, help="named CRV run configuration (crvfastsim.runconfig): "
                            "counters, GDML world and the tagger-cut bars; default extracted_v04")
            sp.add_argument("--msc-tail", action="store_true",
                            help="stage C: single-scattering tail above the Gaussian MSC core, per height band")
            sp.add_argument("--tail-k", type=float, default=None,
                            help="with --msc-tail: tail threshold in units of theta0 (default: density crossing)")
            sp.add_argument("--sliced-msc", action="store_true", help="stage C: per-band |p| in the MSC")
            sp.add_argument("--first-muon-only", action="store_true",
                            help="transport only the first generated muon of each event (before 2026-10-23; default: all)")
            sp.add_argument("--no-absorbers", action="store_true",
                            help="stage C: no energy loss in the CRV aluminium (absorbers, module skins) between bars (before 2026-10-24)")
            sp.add_argument("--no-below-world", action="store_true",
                            help="run configurations with below_world: no material between the MSC plane and the bars")
            sp.add_argument("--below-msc", action="store_true",
                            help="run configurations with below_world: multiple scattering below the MSC plane, bar to "
                                 "bar (transport.scattered_crossings; default straight lines)")
            sp.add_argument("--below-margin", type=float, default=None, metavar="MM",
                            help="run configurations with below_world: read the below-plane material inside the CRV box "
                                 "grown by MM (default: the configuration's below_margin)")
            sp.add_argument("--unbanded-world", action="store_true",
                            help="stage C: world columns piece by piece (before 2026-10-24; the banded grid is the default)")
        if name in ("transport", "runchain"):
            sp.add_argument("--transport-backend", choices=["numpy", "jax"], default="numpy",
                            help="opt-in device transport (straight-line stages A/C; tier 1b remains on host)")
            sp.add_argument("--below-scatter-in", action="store_true",
                            help="experimental bounded below-plane marcher; requires --below-msc and stage C")
            sp.add_argument("--scatter-in-step-mm", type=float, default=50.0)
            sp.add_argument("--scatter-in-envelope-mm", type=float, default=1500.0)
            sp.add_argument("--scatter-in-max-steps", type=int, default=20000)
        if name == "g4cost":
            sp.add_argument("--digi", required=True, help="the digi job's TimeTracker SQLite (timing_digi.db)")
        if name == "v0check":
            sp.add_argument("--g4log", required=True, help="mu2e log with the crs.verbosityLevel 5 bar printout")
            sp.add_argument("--table", default=None, metavar="CSV", help="also write the candidate bar table")
        if name == "v1compare":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT",
                            help="G4 reference CrvStep tables (crvsteps.*.root)")
            sp.add_argument("--run-config", default=None, help="named run configuration: its counters file and one "
                            "group per sector type (default extracted_v04: EX, T1, T2, M)")
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
        if name in ("runchain", "tier1b-apply"):
            sp.add_argument("--tier1b", dest="tier1b_set", default=None, choices=sorted(T1B.SETS),
                            help=f"tier-1b asset set (library, placement sidecars and options; crvfastsim.tier1b_assets). "
                                 f"Default: {T1B.DEFAULT} for extracted_v04; the named run configurations have only "
                                 f"{T1B.LEGACY} (tier1b_v7 library, no placement: the default before 2026-10-28)")
            sp.add_argument("--tier1b-dir", default=None, metavar="DIR",
                            help="directory with the set's files (default $CRVFS_DATA/<set directory>)")
        if name == "runchain":
            sp.add_argument("--gen", choices=("table", "native", "device"), default="table",
                            help="generator backend; native CPU/device JAX take placeholder input 'native'")
            sp.add_argument("--gen-events", type=int, default=None, help="accepted native CRY events (required with --gen native)")
            sp.add_argument("--gen-seed", type=int, default=None, help="native CRY seed (default --seed; independent chain RNG)")
            sp.add_argument("--gen-subrun", type=int, default=1, help="native event subrun (default 1)")
            sp.add_argument("--library", default=None, metavar="ROOT",
                            help="custom tier-1b library, sampled without placement options (instead of --tier1b)")
            sp.add_argument("--conditions-manifest", default=None, help="validated named-condition bundle manifest")
            sp.add_argument("--geometry-readout", action="store_true",
                            help="opt-in physical SiPM mask; requires a Run1A/Run2 conditions manifest")
            sp.add_argument("--calib", default=None, metavar="CSV", help="CRVCalib + channel map CSV")
            sp.add_argument("--deviation", default=None, metavar="CSV", help="CRVPhoton deviations")
            sp.add_argument("--gdml", default=None, help="GDML world for stage C transport")
            sp.add_argument("--msc-tail", action="store_true",
                            help="stage C: single-scattering tail above the Gaussian MSC core, per height band")
            sp.add_argument("--tail-k", type=float, default=None,
                            help="with --msc-tail: tail threshold in units of theta0 (default: density crossing)")
            sp.add_argument("--sliced-msc", action="store_true", help="stage C: per-band |p| in the MSC")
            sp.add_argument("--first-muon-only", action="store_true",
                            help="transport only the first generated muon of each event (before 2026-10-23; default: all)")
            sp.add_argument("--no-absorbers", action="store_true",
                            help="stage C: no energy loss in the CRV aluminium (absorbers, module skins) between bars (before 2026-10-24)")
            sp.add_argument("--no-below-world", action="store_true",
                            help="run configurations with below_world: no material between the MSC plane and the bars")
            sp.add_argument("--below-msc", action="store_true",
                            help="run configurations with below_world: multiple scattering below the MSC plane, bar to "
                                 "bar (transport.scattered_crossings; default straight lines)")
            sp.add_argument("--below-margin", type=float, default=None, metavar="MM",
                            help="run configurations with below_world: read the below-plane material inside the CRV box "
                                 "grown by MM (default: the configuration's below_margin)")
            sp.add_argument("--unbanded-world", action="store_true",
                            help="stage C: world columns piece by piece (before 2026-10-24; the banded grid is the default)")
            sp.add_argument("--run-config", default=None, help="named CRV run configuration (crvfastsim.runconfig: "
                            "extracted_v04 default, run1a_v01, run1b_v01, run2_v10): counters, GDML, response, cuts, window")
            sp.add_argument("--window", default=None, help="digitisation window override: offspill, onspill")
            sp.add_argument("--jax", action="store_true", help="JAX twins for photons, SiPM charges, waveforms")
            sp.add_argument("--max-events", type=int, default=None, help="first N generated events per file")
            sp.add_argument("--seed", type=int, default=1)
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="stock CrvRecoTableDumper files")
            sp.add_argument("--reco-cpu", action="store_true", help="with --jax: pulses and coincidences on the JAX CPU backend")
            sp.add_argument("--prefetch", action="store_true",
                            help="read, transport and tier 1b of the next file in a thread while the device runs this one")
            sp.add_argument("--fused-response", action="store_true",
                            help="with --jax: photons -> SiPM charges in one device pass (response.device, as reco-r3)")
            sp.add_argument("--emit", default=None, metavar="JSON", help="per-shard accumulators for runchain-merge")
        if name == "runchain-merge":
            sp.add_argument("--wall", type=float, default=None, help="node job wall time [s] (default: slowest shard)")
        if name == "v4sipm":
            sp.add_argument("--deviation", default=None, metavar="CSV", help="CRVPhoton table, channel,deviation (default 0)")
            sp.add_argument("--max-steps", type=int, default=None, help="first N steps only")
            sp.add_argument("--jax", action="store_true", help="fast arm = JAX twin (response/sipm_jax.py)")
            sp.add_argument("--no-stock", action="store_true", help="reference = numpy port (no Offline needed)")
            sp.add_argument("--device", action="store_true", help="fast arm = fused device response (response/device.py)")
            sp.add_argument("--calib", default=None, metavar="CSV", help="CRVCalib CSV (status mask) for --device")
            sp.add_argument("--repeat", type=int, default=0, help="--device: N warm timed passes after the first (P2)")
            sp.add_argument("--seed", type=int, default=1)
        if name == "pulsematch":
            for a in ("--fast", "--fast-mu", "--ref", "--ref-mu"):
                sp.add_argument(a, nargs="+", default=None, metavar="ROOT", help="CrvRecoTableDumper files of that arm")
        if name == "crossingpos":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="G4 CrvStep tables, same shards")
        if name == "tier1b-check":
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="G4 CrvStep tables, same shards")
        if name == "tier1b-apply":
            sp.add_argument("--library", default=None, metavar="ROOT",
                            help="custom library from tier1b-build (instead of --tier1b); placement only as flagged:")
            sp.add_argument("--seed", type=int, default=1, help="base seed; file i uses (seed, 1000+i)")
            sp.add_argument("--crossing-context", action="store_true",
                            help="draw descendants within the crossing's layer/upstream context (library with ctx)")
            sp.add_argument("--contain-steps", action="store_true",
                            help="move same-bar descendant endpoints that leave the bar inside it")
            sp.add_argument("--point-placement", action="store_true",
                            help="other-bar descendants with a missing or other-axis target bar: bar at their position")
            sp.add_argument("--mirror-offsets", action="store_true",
                            help="reflect other-bar descendant offsets where library and fast muons lean opposite ways")
            sp.add_argument("--translate-offsets", action="store_true",
                            help="other-bar descendants to the bar at fast entry + library offset from the entry")
            sp.add_argument("--topology-sidecar", default=None, metavar="NPZ",
                            help="other-bar descendants keep their library layer topology around the receiving muon's "
                                 "visits (sidecar from valhist.descendant_topology sidecar for this library)")
            sp.add_argument("--template-sidecar", default=None, metavar="NPZ",
                            help="position-conditioned template components for this library "
                                 "(valhist.descendant_templates sidecar); turn them on with the next two flags")
            sp.add_argument("--unvisited-layers", action="store_true",
                            help="descendants in layer planes the library muon never visits: drawn by the crossing's "
                                 "position cell instead of with its offsets")
            sp.add_argument("--near-miss", action="store_true",
                            help="near-miss templates for generated muons of events with no fast crossing")
            sp.add_argument("--chord-placement", action="store_true",
                            help="same-bar descendants along the fast chord at their fraction of the library chord "
                                 "(template sidecar with library chords)")
            sp.add_argument("--follow-muon", action="store_true",
                            help="a muon's next crossing takes its library muon's next crossing when that lies in its "
                                 "draw cell (template sidecar with library chords)")
        if name == "secondaries":
            sp.add_argument("--gen", nargs="+", default=None, metavar="ROOT", help="gen.*.root tables (CRY rates per pdg)")
        if name == "v2origin":
            sp.add_argument("--steps", nargs="+", default=None, metavar="ROOT", help="CrvStep table behind the inputs")
            sp.add_argument("--compare", nargs="+", default=None, metavar="ROOT", help="reference CrvRecoTableDumper files")
            sp.add_argument("--compare-steps", nargs="+", default=None, metavar="ROOT", help="CrvStep tables behind the reference")
            sp.add_argument("--generated", type=int, default=None, help="generated events behind the inputs")
            sp.add_argument("--compare-generated", type=int, default=None, help="generated events behind the reference")
        if name == "v2compare":
            sp.add_argument("--run-config", default=None, help="named geometry and all-sector V2 groups")
            sp.add_argument("--candidate-receipt", help="prospective fasttable receipt; conditional integrity audit only")
            sp.add_argument("--conditions-manifest", help="strict bundle for candidate receipt revalidation")
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
    sub.add_parser("e2e-bench", help="provenance-checked cached-CRY to pulses/coincidences benchmark",
                   add_help=False)
    sub.add_parser("transport-validation", help="persisted-table audit and event-paired sector diagnostics",
                   add_help=False)
    sub.add_parser("transport-paths", help="saved-table identity-matched path geometry diagnostics",
                   add_help=False)
    for name in reco_cli.RECO_CMDS:
        sp = sub.add_parser(f"reco-{name}", help=f"fast CRV reco: {name} (reco/cli.py)")
        reco_cli.add_reco_args(sp, name)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "e2e-bench":
        from .analyses import e2e_bench
        return e2e_bench.main(argv[1:])
    if argv and argv[0] == "transport-validation":
        from .analyses import transport_validation
        return transport_validation.main(argv[1:])
    if argv and argv[0] == "transport-paths":
        from .analyses import transport_paths
        return transport_paths.main(argv[1:])
    args = build_parser().parse_args(argv)
    if args.cmd.startswith("reco-"):
        return reco_cli.run_reco(args.cmd[len("reco-"):], args)
    kw = vars(args)
    cmd = kw.pop("cmd")
    inputs = kw.pop("inputs")
    outdir = kw.pop("outdir") or f"plots/{cmd}"
    result = ANALYSES[cmd][0](inputs, outdir, **kw)
    if cmd == "v1compare" and (kw.get("merge") or not kw.get("emit")) and not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
