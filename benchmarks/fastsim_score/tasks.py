"""The benchmark's module tasks: what each task scores, at which boundary, and what its reference must hold.

A task is one directory of the reference tree (reference/<task>/). This table defines the tasks; the reference trees
themselves hold Mu2e simulation results and are never in this repository. `reference.json` of a task carries its
"task" id, so a reference built for one task cannot be scored as another.

    python -m benchmarks.fastsim_score.tasks            # the task table (Markdown)
"""
from __future__ import annotations

import sys

# kind: the histogram kinds the reference uses (score.py: counts, values, efficiency; yields: count or measured).
# floor: what plays the reseeded-stock role, or None when no floor exists yet (threshold 2 then).
# status: "scored" (a frozen reference exists and the coverage table scores it) or "defined" (card and scoring rules
# fixed, reference pending). W1-W8 are the Run 1A fast workflows; W9, CRVLY and PTGT are module tasks; B1-B4 are the
# Geant4 fast-simulation and GPU cards (FS-CALO-G4, FS-CRV-MOD, FS-TRKFIT, FS-FLASH-GPU), scored by the same scorer.
TASKS: dict[str, dict] = {
    "W1": {"title": "Muon beam", "boundary": "TS-entrance mu/pi -> stopping-target stops",
           "stock": "Geant4 TS + DS stages, expectation over stock re-seeds", "metrics": "stop foil, time, radius, "
           "DS-entrance momentum (paired per shard); stops per shard", "kind": "counts, paired per shard",
           "floor": "one stock re-seed", "rate_unit": "TS-entrance mu/pi per s"},
    "W2": {"title": "Conversion electrons", "boundary": "CE stops -> CE track observables",
           "stock": "full Geant4 events -> stock digi + LoopHelix reco", "metrics": "ValKalSeed histograms, paired "
           "shape per sub-shard (yields per generated electron are a separate study)", "kind": "counts, paired shape",
           "floor": None, "rate_unit": "events/s"},
    "W3": {"title": "DIO electrons", "boundary": "DIO births -> CE track observables",
           "stock": "full Geant4 events (every tracker step of the DIO dts) -> stock OffSpill digi + LoopHelix reco",
           "note": "the fast arm runs the default (full-event) reco response, so the stock arm digitises every tracker "
                   "step of the event, not only the DIO electron's own steps (that primary-only arm has 0.93 of the "
                   "full-event CE-selected yield)",
           "metrics": "CE-selected momentum "
           "spectrum 95-106 MeV/c; tail counts above 95 and 100 MeV/c and all CE-selected", "kind": "counts",
           "floor": None, "rate_unit": "generated DIO muons per s"},
    "W4": {"title": "RPC electrons", "boundary": "RPC births -> tracker entry",
           "stock": "Geant4 RPC sample at the tracker entrance", "metrics": "tracker-entry fiducial yield (no stock "
           "reco reference after the response yet)", "kind": "measured yield", "floor": None,
           "rate_unit": "generated RPC captures per s"},
    "W5": {"title": "CRV cosmics", "boundary": "CRY cosmics -> CRV coincidence clusters",
           "stock": "Geant4 steps -> stock CRV response + reco", "metrics": "ValCrv histograms, paired shape per shard; "
           "reco pulse yield", "kind": "counts, paired shape", "floor": "stock with another seed",
           "rate_unit": "events/s"},
    "W6": {"title": "Pileup mixing", "boundary": "primary steps + donor pools -> mixed tracker digis + CRV pulses",
           "stock": "stock mixing of the same primaries", "metrics": "digis per plane, pulses per sector (paired by "
           "event); tracker and CRV totals", "kind": "values + measured yields", "floor": None,
           "rate_unit": "events/s"},
    "W7": {"title": "Calorimeter", "boundary": "CE calorimeter entrants (disk front, inner edge) -> calo hits, clusters",
           "stock": "Geant4 calorimeter steps -> stock calo digi + reco", "metrics": "cluster E, t, R, N (paired shape "
           "per block); cluster yield", "kind": "counts, paired shape", "floor": "stock with another seed",
           "rate_unit": "events/s"},
    "W8": {"title": "Tracker trigger", "boundary": "StrawDigis -> tracker trigger path decisions",
           "stock": "the stock trigger on the same digis", "metrics": "accept fraction per path (measured, paired "
           "error, tolerance 2 % of the acceptance)", "kind": "measured yields", "floor": None,
           "rate_unit": "events/s",
           "note": "scored on main 213edff8 with the exact-hits trigger code (#283)"},
    "W9": {
        "title": "CRV design model",
        "note": "the model returns expected misses (mean miss probability), so both quantities are values with the "
                "paired error on the reference side",
        "boundary": "CRV module design parameters -> veto (in)efficiency",
        "stock": "full-simulation CRV veto comparison of the same design (stock arm of the design study)",
        "metrics": "inefficiency per momentum bin; efficiency per CRV sector",
        "kind": "values (per momentum bin and per sector, paired error) + measured inefficiency; the efficiency "
                "kind (passed/total) serves a model that returns counts",
        "floor": None,
        "rate_unit": "selected CE-proxy events per s",
        "stock_rate": "Geant4 + CRV reconstruction core-seconds per selected event of the stock comparison sample (one "
                      "CPU core); the model is timed from the CRV-entry states",
    },
    "CRVLY": {
        "title": "CRV light-yield (lookup) table generator",
        "boundary": "CRV counter geometry and optics -> photon arrival lookup tables",
        "stock": "the v6_0 lookup tables (2019 standalone CRV optical simulation; the generator's default v6 model)",
        "metrics": "bin-by-bin arrival probability per table type (scintillation, Cerenkov in scintillator, "
                   "Cerenkov in fibre), error binomial on the photons generated per bin; grid jobs and cost per table",
        "kind": "values (per-bin probability with its binomial error)",
        "floor": None,
        "rate_unit": "tables/s",
        "jobs": "stock_jobs_per_unit = grid jobs per stock table",
        "note": "reference pending: the generator's full tables and a measured stock cost per table",
        "status": "defined",
    },
    "PTGT": {
        "title": "Production-target chain",
        "boundary": "protons on target -> particles at the transport-solenoid entrance",
        "stock": "G4beamline on the same target configurations (the design board's rods)",
        "metrics": "stopped muons per POT and mu- + pi- per POT at the TS entrance per configuration (measured "
                   "yields); p, cos theta, r and t at the TS entrance (shape); throughput per POT",
        "kind": "yields: measured (value, error) per configuration; hists: counts in shape mode",
        "floor": "replicate G4beamline runs of the same configuration",
        "rate_unit": "POT/s",
        "note": "stock = G4beamline on the 33 design-board rods; stops via the same muon transport on both arms; TS-entrance "
                "shapes at the nominal rod with G4beamline's replicate as the floor; rate per POT for the target + PS stage; "
                "the fast arm crosses the TS-window steel rings as thin slabs (two seeds of 1M POT per rod, pooled)",
    },
    "B1": {
        "title": "Calorimeter shower model inside and outside Geant4 (FS-CALO-G4)",
        "boundary": "e+-/gamma state at a calorimeter disk front -> crystal deposits (stock calo digi + reco downstream)",
        "stock": "Mu2eG4 with the model off on the same CeEndpoint primaries, one Offline build",
        "metrics": "cluster E, time, x, y, radius, size, clusters per event, hit E; yields: events with a cluster "
                   "> 10 MeV, CE peak position (tolerance 0.2 %, se <= 0.067 %); classifier score on per-cluster "
                   "features (reported beside validity)",
        "kind": "counts + measured yields with yield_max_se; classifier (features.npz)",
        "floor": "model off with another seed",
        "rate_unit": "events/s",
        "stock_rate": "Mu2eG4 host: Mu2eG4 module time per event model-off / model-on on the same node and primaries; "
                      "device host: the stock rate per grid slot at the same boundary",
        "rules": "host declared (Mu2eG4 model or device chain); both dtypes; per-thread resident memory increase "
                 "<= 250 MB; the edge rule (entries near a disk edge stay in Geant4) is part of the submission; "
                 "splits by file (train, dev, reserved final + fresh primaries on reserved seeds); the reference "
                 "must beat two weak baselines (Gflash profiles; all energy in the entry crystal) on validity and "
                 "classifier score",
        "status": "scored",
    },
    "B2": {
        "title": "Geometry-portable CRV module interaction (FS-CRV-MOD)",
        "boundary": "entry state at a CRV module envelope (module frame) -> the module's CrvSteps + exiting particles",
        "stock": "full Geant4 per layout: dev = held-out nominal-layout shards; place = placement moves of the "
                 "current module type; design = the CRV design study's short-listed layouts (stock from its grid "
                 "production, reused)",
        "metrics": "per layout: DQMVAL CRV histograms (steps, pulses, coincidences), exit-particle yield and spectrum; "
                   "portability ratio = mean place validity / dev validity; adaptation score = mean design validity "
                   "at the declared capture budget; classifier score on per-coincidence features",
        "kind": "counts + yields per layout; one task per layout with group B2 and part dev / place / design; "
                "capture_budget on design tasks; classifier",
        "floor": "stock with another seed, per layout",
        "rate_unit": "events/s",
        "rules": "library in the loop: no retraining between dev and place; design moves may adapt within a fixed "
                 "capture budget per new module type (proposed 200,000 captured module entries), reported; layouts "
                 "defined in code (run config + pose table); weak baselines: the absolute-placement hybrid (should "
                 "pass dev and fail place) and the nominal library unadapted on design moves; ranking: final validity",
        "status": "defined",
    },
    "B3": {
        "title": "Batched track fits against KinKal (FS-TRKFIT)",
        "boundary": "frozen stock reco inputs per track (combo hits, time cluster, seed) -> fit parameters and "
                    "covariance at the tracker entrance",
        "stock": "stock KinKal on CPU on the same inputs (agreement 1 by construction; sets the throughput basis); "
                 "KinematicLine (extracted), CentralHelix cosmics, LoopHelix CE",
        "metrics": "agreement: fraction of tracks within 0.1 sigma of the stock fit in every parameter (ranking key); "
                   "good-fit ratio fast/stock (paired); pulls against MC truth; throughput scaling curve at batch "
                   "10, 10^3, 10^5 tracks with the crossover batch",
        "kind": "track_agreement (tracks.npz) + scaling_batches; one task per configuration",
        "floor": None,
        "rate_unit": "tracks/s",
        "stock_rate": "stock KinKal tracks per s per core",
        "rules": "float64 fit state (float32 a reported variant); device and batch sizes declared; splits by file "
                 "(train for tuning, dev, final); weak baseline: a least-squares helix without material or drift "
                 "(should fail agreement)",
        "status": "defined",
    },
    "B4": {
        "title": "GPU e+-/gamma transport for the flash (FS-FLASH-GPU)",
        "boundary": "flash primaries at one stage boundary fixed before any run (recommended: DS entrance) -> tracker "
                    "StrawGasSteps, the DS/tracker regions offloaded through G4VTrackingManager",
        "stock": "Mu2eG4 without offload on the same primaries and build; CPU controls for the stock behaviour the GPU "
                 "cannot copy (no low-energy kill; tracks killed on leaving the DS)",
        "metrics": "StrawGasStep edep, multiplicity, time and per-plane histograms; the flash_edep objective as a yield "
                   "(tolerance proposed 3 %)",
        "kind": "counts + measured yield",
        "floor": "stock with another seed",
        "rate_unit": "events/s",
        "stock_rate": "job wall time, CPU-only stock vs offload on the same node (1 A100 + 16 cores vs the 16 cores)",
        "rules": "engine selection: AdePT and Celeritas, the better score kept; splits by pool file and subrun; weak "
                 "baseline: the earlier direct-electron sketch (should fail)",
        "status": "defined",
    },
}


def table() -> str:
    rows = ["| task | what | boundary | stock reference | scored | floor | rate | status |", "|---" * 8 + "|"]
    for k, t in TASKS.items():
        rows.append(f"| {k} | {t['title']} | {t['boundary']} | {t['stock']} | {t['metrics']} | "
                    f"{t['floor'] or 'none yet (threshold 2)'} | {t['rate_unit']} | {t.get('status', 'scored')} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
    sys.exit(0)
