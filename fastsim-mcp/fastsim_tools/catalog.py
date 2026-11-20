"""The fast-chain workflows an agent can ask about: what each one starts from, what it produces, where it lives.

Plain data and lookups, no MCP and no repository imports. Numbers (rates, validity) are not copied here: each entry
points at the documents that hold them, so the catalogue cannot drift from the measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Workflow:
    key: str                         # the name agents pass around (list_workflows, describe_approximations)
    title: str
    boundary_in: str                 # what the workflow starts from
    boundary_out: str                # what it hands on
    packages: Tuple[str, ...]        # Python packages of the repository that implement it
    entry_points: Tuple[str, ...]    # command lines (python3 -m ...) that run it
    approximations: Tuple[str, ...]  # workflow names in docs/approximations.md whose approximations apply
    docs: Tuple[str, ...]            # repository-relative documents with its validation and rates
    kit_tasks: Tuple[str, ...] = ()  # fastsimkit tasks that run it on a compute endpoint
    run_design_point: Optional[str] = None   # how run_design_point exposes it (None: not through this server)
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> Dict:
        d = asdict(self)
        d["boundary"] = {"input": d.pop("boundary_in"), "output": d.pop("boundary_out")}
        return d


WORKFLOWS: Tuple[Workflow, ...] = (
    Workflow(
        "W1", "muon beam: TS entrance to stopped muons",
        "muons at the transport-solenoid entrance (or the DS entrance / Coll5_Out plane of a reference table)",
        "stopped muons in the stopping-target foils (stop positions, times, stops per generated event and per POT)",
        ("beamfastsim",), ("python3 -m beamfastsim.tsdevice", "python3 -m beamfastsim point"),
        ("W1",), ("docs/performance.md", "beam/docs/ts-transport.md", "beam/docs/target-geometry.md"),
        kit_tasks=("beam_point", "beam_sob", "beam_edep", "beam_edep_cpu"),
        run_design_point="first stage of every design point; stage 'stops' (gpu) returns the stopping metrics alone"),
    Workflow(
        "W2-W4", "tracker: stops, DIO or RPC electrons to tracker steps, digis and observables",
        "electron births (CE from the stops, DIO, RPC internal-conversion pairs)",
        "tracker StrawGasSteps, straw digis, and track observables from the reco response tables",
        ("beamfastsim", "trkfastsim"), ("python3 -m beamfastsim.cetracker",),
        ("W2-W4",), ("docs/performance.md", "docs/validation.md", "tracker/docs"),
        kit_tasks=("beam_edep", "beam_edep_cpu", "beam_centuple"),
        run_design_point="inside the CE chain (tracker transport and albedo of the conversion electrons)"),
    Workflow(
        "W5", "CRV: cosmic generator events to CRV pulses and coincidences",
        "cosmic generator events (CRY), or Geant4 CrvSteps from art (--steps-input)",
        "CRV steps, photons, SiPM charges, waveforms, reco pulses, coincidence clusters",
        ("crvfastsim",), ("python3 -m crvfastsim runchain",),
        ("W5",), ("docs/performance.md", "crv/docs/validation.md", "crv/docs/running.md"),
        kit_tasks=("chain", "chain_point", "crv_response"),
        run_design_point="run_crv_chain (GPU only, chain_point or chain; float32, the chain's default)"),
    Workflow(
        "W6", "pileup mixing",
        "donor frames of the pileup streams and a signal event",
        "mixed tracker and CRV digis",
        ("mixing",), ("python3 -m mixing.stream_benchmark",),
        ("W6",), ("mixing/README.md", "mixing/DEVICE_BACKEND.md")),
    Workflow(
        "W7", "calorimeter: disk entrants (front face, inner edge) to crystal hits and clusters",
        "particles entering a calorimeter disk through the front face or inner edge (or stock CaloShowerSteps)",
        "crystal hits and clusters (parametric response by default; digis with the waveform response)",
        ("calofastsim",), (), ("W7",), ("docs/performance.md", "docs/approximations.md"),
        kit_tasks=("beam_edep", "beam_edep_cpu", "beam_centuple"),
        run_design_point="inside the CE chain (shower library at the disk fronts, Geant4 inner-edge table)"),
    Workflow(
        "W8", "trigger: device digis to trigger decisions",
        "straw and calorimeter digis (fast or stock)",
        "per-path trigger decisions of the online menu (20 single-helix tracker paths on the device)",
        ("trigger",), ("python3 -m trigger.device.run decide",), ("W8",), ("docs/performance.md",),
        kit_tasks=("beam_centuple",)),
    Workflow(
        "ce_chain", "conversion-electron chain: stops to the tracker and calorimeter",
        "stopped muons of a target geometry (W1)",
        "CE electrons at the tracker entry, through the device tracker chain, to the calorimeter",
        ("beamfastsim", "calofastsim", "trkfastsim"),
        ("python3 -m beamfastsim.cechain", "python3 -m beamfastsim.cesob point"),
        ("W1", "W2-W4", "W7"), ("docs/performance.md", "beam/docs/edepana.md"),
        kit_tasks=("beam_sob", "beam_edep", "beam_edep_cpu"),
        run_design_point="stage 'sob' runs it with the calorimeter wall proxy and returns the built-in sensitivity"),
    Workflow(
        "edepana_writer", "EdepAna writer: the fast CE chain written as the EDepAna/tree of Offline's EdepAna",
        "a target geometry (stops, CE chain, calorimeter)",
        "nts.mu2efastsim.edep.root (EDepAna/tree), edep_summary.txt and the .edep.json run record: the inputs of "
        "the analysis server's approx_ce_sensitivity (S/sqrt(B)) and of the edep summary readers",
        ("beamfastsim",), ("python3 -m beamfastsim.edepana point",),
        ("W1", "W2-W4", "W7"), ("beam/docs/edepana.md",),
        kit_tasks=("beam_edep", "beam_edep_cpu"),
        run_design_point="stage 'edep' (default): one CPU core (or one GPU) per dtype, the file comes back with "
                         "fetch_outputs",
        notes=("event_trk_edep is low: no delta rays in the tracker (beam/docs/edepana.md, known gaps)",)),
    Workflow(
        "eventntuple_writer", "EventNtuple writer: the fast CE chain written as EventNtuple/ntuple",
        "a target geometry, donor events from stock files, the Mu2e environment (/cvmfs) for the PyROOT writer",
        "nts.mu2efastsim.centuple.root (EventNtuple/ntuple, stock split layout): the input of the analysis "
        "server's EventNtuple analyses (trigger_efficiency_ntuple, occupancy_ntuple, ...)",
        ("beamfastsim",), ("python3 -m beamfastsim.centuple point",),
        ("W1", "W2-W4", "W7", "W8"), ("beam/docs/eventntuple.md",),
        kit_tasks=("beam_centuple",),
        run_design_point="stage 'centuple' (gpu), on an endpoint with /cvmfs and FNAL storage",
        notes=("the kit refuses it on Slurm endpoints without FNAL storage",)),
)

BY_KEY: Dict[str, Workflow] = {w.key: w for w in WORKFLOWS}
KEYS: Tuple[str, ...] = tuple(BY_KEY)


def find(key: str) -> Workflow:
    """A workflow by key; also accepts W2, W3, W4 for W2-W4 and is case-insensitive."""
    k = key.strip()
    if k.upper() in ("W2", "W3", "W4"):
        k = "W2-W4"
    for w in WORKFLOWS:
        if w.key.lower() == k.lower():
            return w
    raise ValueError(f"unknown workflow {key!r}; choose from {', '.join(KEYS)}")


def describe_all() -> List[Dict]:
    return [w.describe() for w in WORKFLOWS]
