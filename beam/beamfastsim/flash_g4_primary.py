"""Saved-record primary-brems birth-foil diagnostic matched to fast S1, NOT S2.

A source electron/positron is a same-stage mu2ePrimary continuation with a
resolved, same-species earlier-stage parent. Earlier stages need not be adjacent;
upstream shower ancestry does not disqualify an input DS state. No ancestry walk
through a new-stage secondary is allowed. All original photon/event rows remain.

Uncollided means only geometry end processes on recorded completed steps BEFORE
the crossing step, not energy equality (saved float32 energy can hide scattering).
The crossing endpoint process occurs after an accepted interior crossing. For a
two-segment exit the continuation is the crossing step, so its transport anchor
is included in the prefix. Spatial continuity/ambiguity gates come from extract.
Missing records and no recorded exit are never interpreted as absorption. This
is neither full G4 photon-flux completeness nor coupled S1/S2 validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .flash_flux import ENERGY_EDGES
from .flash_g4_photons import extract, read_tables, _xyz
from .flash_geometry import from_offline
from .flash_inclusive_check import EVENT, PRODUCT, keys, join, matched

# Verified in Offline MCDataProducts/inc/ProcessCode.hh and
# Mu2eG4/src/Mu2eG4TrackingAction.cc::saveSimParticleStart.
EBREM, MU2E_PRIMARY = 16, 56
GEOMETRY_PROCESSES = (48, 49, 182)  # StepLimiter, Transportation, CoupledTransportation


def _unknown_process(code):
    # GammaGeneralProc does not resolve which physics interaction occurred.
    return np.isin(code, (0, 76, 184, 185, 186)) | (code < 0) | (code >= 195)


def select(tables, photons):
    """Return all extract columns plus conservative ancestry/history selections.

    Full uint64 event/product/particle and collection-index identities are used.
    Duplicate targets fail rather than choosing an ancestry arbitrarily. Collection
    indices order same-photon records; they are NOT G4 track step numbers. The
    unknown-history gate is event-conservative for unresolved photon pointers or
    declared/exported photon-step mismatches. This is still not export certification.
    """
    a, s, p, e = (tables[k] for k in ("particles", "steps", "products", "events"))
    ak, pk, ek = keys(a, (*PRODUCT, "key")), keys(photons, (*PRODUCT, "key")), keys(e, EVENT)
    for identity in (ak, pk, ek):
        join(identity, identity)
    particle, event = join(pk, ak), join(keys(photons, EVENT), ek)
    if np.any(particle < 0) or np.any(event < 0):
        raise ValueError("photon missing from particles or event ledger")
    if np.any(a["pdg"][particle] != 22):
        raise ValueError("foil photon identity is not gamma")
    parent = join(keys(a, (*EVENT, "parent_product", "parent_key")), ak)
    n = len(pk)
    primary, unknown = np.zeros(n, bool), _unknown_process(a["creation"][particle])
    brems = a["creation"][particle] == EBREM
    par = parent[particle]
    missing = (a["parent_state"][particle] != 1) | (par < 0)
    unknown |= brems & missing
    rows = np.flatnonzero(brems & ~missing)
    q = par[rows]
    continuation = np.isin(a["pdg"][q], (11, -11)) & (a["creation"][q] == MU2E_PRIMARY)
    unknown[rows] |= _unknown_process(a["creation"][q])
    rows, q = rows[continuation], q[continuation]
    prev = parent[q]
    resolved = (a["parent_state"][q] == 1) & (prev >= 0)
    valid = np.zeros(len(rows), bool)
    valid[resolved] = ((a["stage"][q[resolved]] == a["stage"][particle[rows[resolved]]])
                       & (a["stage"][prev[resolved]] < a["stage"][q[resolved]])
                       & (a["pdg"][prev[resolved]] == a["pdg"][q[resolved]]))
    primary[rows] = valid
    unknown[rows] |= ~valid

    ep = np.flatnonzero(p["scope"] == 0)
    pi = join(keys(s, PRODUCT), keys(p, PRODUCT)[ep])
    role = (p["kind"][ep] == 3) & (p["label"][ep] == "g4run") & (p["instance"][ep] == "photonSteps")
    join(keys(p, EVENT)[ep][role], keys(p, EVENT)[ep][role])
    recorded = matched(pi, role)
    owner_all = join(keys(s, (*EVENT, "particle_product", "particle_key")), pk)
    sr = np.flatnonzero(recorded & (s["particle_state"] == 1) & (owner_all >= 0))
    join(keys(s, (*PRODUCT, "index"))[sr], keys(s, (*PRODUCT, "index"))[sr])
    owner = owner_all[sr]
    two = photons["two_segment_reconstruction"]
    crossing = np.where(two, photons["continuation_stepindex"], photons["first_stepindex"])
    same_product = s["product"][sr] == photons["first_step_product"][owner]
    prefix = photons["exit_candidate"][owner] & (s["index"][sr] < crossing[owner])
    process = s["end_process"][sr]
    count = lambda mask: np.bincount(owner[mask], minlength=n)
    nongeometry = count(prefix & ~np.isin(process, GEOMETRY_PROCESSES))
    unknown_history = count(prefix & (_unknown_process(process) | ~same_product)) > 0
    at_crossing = same_product & (s["index"][sr] == crossing[owner])
    positions, endpoints = _xyz(s)[sr], _xyz(s, "post_")[sr]
    invalid = (~np.all(np.isfinite(positions) & np.isfinite(endpoints), axis=1)
               | (np.linalg.norm(endpoints - positions, axis=1) <= 0))
    for prefix_name in ("p", "post_p"):
        energy = np.linalg.norm(_xyz(s, prefix_name)[sr], axis=1)
        invalid |= ~np.isfinite(energy) | ((energy <= 0) if prefix_name == "p" else False)
    unknown_history |= count((prefix | at_crossing) & invalid) > 0
    at_anchor = same_product & (s["index"][sr] == photons["first_stepindex"][owner])
    intervening = two[owner] & prefix & (s["index"][sr] > photons["first_stepindex"][owner])
    reconstruction_bad = two & ((count(at_anchor) != 1) | (count(intervening) != 0)
                                | ~np.isin(photons["first_end_process"], (49, 182)))
    unknown_history |= photons["exit"] & ((count(at_crossing) != 1) | reconstruction_bad)
    unknown_history |= np.bincount(owner, minlength=n) != photons["recorded_steps"]
    # A broken pointer could hide a completed pre-crossing interaction.
    se = join(keys(s, EVENT), ek)
    sp = join(keys(s, (*EVENT, "particle_product", "particle_key")), ak)
    broken = recorded & ((s["particle_state"] != 1) | (sp < 0)) & (se >= 0)
    bad_event = np.bincount(se[broken], minlength=len(ek)) > 0
    bad_event |= (~photons["event_photon_steps_present"] | photons["event_photon_steps_size_mismatch"])
    unknown_history |= bad_event[event]
    spatial_bad = photons["ambiguous"] | ~photons["birth_inside_geometry"]
    unresolved = unknown | unknown_history | spatial_bad
    uncollided = primary & photons["exit"] & ~unresolved & (nongeometry == 0)
    out = dict(photons)
    out.update(primary_brems=primary, unknown_ancestry=unknown, unknown_history=unknown_history,
               prior_non_geometry_steps=nongeometry, uncollided_exit=uncollided,
               unresolved=unresolved, no_recorded_exit=~photons["exit"],
               evt_primary_births=np.bincount(event[primary], minlength=len(ek)),
               evt_primary_exits=np.bincount(event[uncollided], minlength=len(ek)),
               evt_primary_energy=np.bincount(event[uncollided],
                   weights=photons["pre_energy_mev"][uncollided], minlength=len(ek)))
    report = dict(scope="Recorded primary-brems birth-foil diagnostic only; not full G4 flux or coupled S1/S2",
                  input_events=len(ek), photon_rows=n, primary_birth_count=int(primary.sum()),
                  primary_uncollided_exit_count=int(uncollided.sum()),
                  primary_uncollided_exit_energy_mev=float(out["evt_primary_energy"].sum()),
                  ambiguous_count=int(photons["ambiguous"].sum()), unresolved_count=int(unresolved.sum()),
                  unknown_ancestry_count=int(unknown.sum()), unknown_history_count=int(unknown_history.sum()),
                  no_recorded_exit_count=int(out["no_recorded_exit"].sum()),
                  primary_no_recorded_exit_count=int((primary & ~photons["exit"]).sum()),
                  primary_prior_non_geometry_count=int((primary & (nongeometry > 0)).sum()),
                  two_segment_uncollided_count=int((uncollided & two).sum()),
                  source_states_declared=int(photons["event_input_states_declared"].sum()),
                  source_missing_events=int((~photons["event_input_states_present"]).sum()),
                  source_size_mismatch_events=int(photons["event_input_states_size_mismatch"].sum()),
                  geometry_process_ids=list(GEOMETRY_PROCESSES), energy_edges_mev=ENERGY_EDGES.tolist())
    for label, mask, energy in (("birth", primary, photons["birth_energy_mev"]),
                                ("exit", uncollided, photons["pre_energy_mev"])):
        report["primary_" + label + "_bins"] = np.histogram(energy[mask], ENERGY_EDGES)[0].tolist()
        report["primary_" + label + "_energy_bins_mev"] = np.histogram(
            energy[mask], ENERGY_EDGES, weights=energy[mask].astype(float))[0].tolist()
        report["primary_" + label + "_out_of_bins"] = int(np.count_nonzero(
            mask & (~np.isfinite(energy) | (energy < ENERGY_EDGES[0]) | (energy > ENERGY_EDGES[-1]))))
    out["spatial_report_json"] = photons["report_json"]
    out["report_json"] = np.asarray(json.dumps(report, sort_keys=True))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--geom", required=True)
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--events", required=True, type=int, help="Expected all-input count, not a read limit")
    parser.add_argument("--boundary-tolerance-mm", type=float, default=0.)
    args = parser.parse_args(argv)
    if args.events < 0:
        parser.error("--events must be nonnegative")
    tables = read_tables(args.input)
    photons = extract(tables, from_offline(args.geom, args.offline_root),
                      boundary_tolerance_mm=args.boundary_tolerance_mm)
    result = select(tables, photons)
    report = json.loads(str(result["report_json"]))
    report.update(input=str(args.input), geometry_file=args.geom, offline_root=args.offline_root,
                  expected_events=args.events, boundary_tolerance_mm=args.boundary_tolerance_mm)
    result["report_json"] = np.asarray(json.dumps(report, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        np.savez_compressed(stream, **result)
    args.out.with_suffix(".json").write_text(json.dumps(report, sort_keys=True) + "\n")
    print(result["report_json"])
    return int(result["input_event_count"] != args.events)


if __name__ == "__main__":
    raise SystemExit(main())
