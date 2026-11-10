"""Per-step and per-event energy accounting of linked component captures; diagnostic only, never bank admission.

Energies are total energies, so rest masses are explicit. For a step inside the component,

    R = E_pre - E_post(kept) - deposit - sum(E_birth of the step's children) + m_e * L_medium

where E_post is dropped when the step destroys the particle (it is killed by a decay or an absorbing EM process),
and L_medium is the net electron number the step takes from the material (Compton, photo-, ionisation electrons;
the atomic electron of an annihilation; minus the positrons of pair production). A nucleus that an EM step sets in
motion (the recoil of single Coulomb scattering) also comes from the material, so its rest mass is added to the
medium term too, and the step balances in kinetic energy. Steps of electromagnetic,
transport and decay processes must close (|R| within tolerance). Any other process is reported as an explicit
nuclear term Q = E_pre - E_post(kept) - deposit - sum(E_birth), never forced to zero.

Decay steps carry a declared term Q_decay = E_parent - sum(E_children), gated step by step against its expected
value. Geant4's G4MuonDecayChannel (a muon decaying to one e+- and two neutrinos) gives daughters that total
m_mu - m_e, so its expected Q_decay is m_e. This is Geant4's own behaviour, declared here, not a tuned value.
A nucleus (PDG >= 1e9) with no decay channel, such as a triton with radioactive decay off, is killed at rest by
G4Decay with no products; that step is declared too: the nucleus is not destroyed, its rest mass ends in the
material, and Q_decay = 0. Every other decay is expected to conserve energy (Q_decay = 0). A decay step off its
expected value fails.

Two declared terms come from dynamic masses that differ from the nominal ones at rounding level: a photon made in a
boosted decay carries a tiny dynamic mass (m <= 1e-6 E), and an EM process that absorbs it (conv, phot) shares
only its kinetic energy, so its pre-step mass is the declared term of that step. G4Decay makes the daughters at the
parent's dynamic mass m but boosts them with gamma = E/M, so a decay may give Q_decay = E (1 - m/M); this is the
alternative expected value of a decay step that is off zero, for |m - M| <= 1e-6 M, with M the most common mass of
that PDG in the file.

Per event, E_entrance + sum(m_e * L_medium) = deposits + first exits + exits at birth + energy left by particles that
end inside without being destroyed + sum(Q_nuclear) + sum(Q_decay) + sum(photon dynamic masses) + sum(R).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import capture
from .saved_g4 import frame, match, require, unique

TRANSPORT = {"Transportation", "CoupledTransportation", "msc", "muMsc", "StepLimiter", "UserLimit"}
EM = {"eIoni", "eBrem", "muIoni", "muBrems", "muPairProd", "hIoni", "ionIoni", "hBrems", "hPairProd", "ePairProd",
      "CoulombScat", "eCoulombScattering", "Rayl", "compt", "phot", "conv", "annihil", "GammaGeneralProc"}
DECAY = {"Decay", "DecayWithSpin"}
DESTROYING = {"annihil", "phot", "conv", "GammaGeneralProc", *DECAY}
KILLED = 2
ELECTRON, POSITRON = 11, -11
MUONS, NEUTRINOS = (13, -13), (12, -12, 14, -14, 16, -16)
PHOTON, MASS_ROUNDING = 22, 1e-6
DECLARED = {"G4MuonDecayChannel": "muon -> e+- and two neutrinos: daughters total m_mu - m_e, so Q_decay = m_e; "
                                  "Geant4 behaviour, not tuned",
            "G4DecayWithoutProducts": "a nucleus with no decay channel (e.g. 3H, radioactive decay off) is killed at rest "
                                      "by Decay with no products; its rest mass ends in the material; Geant4 behaviour",
            "G4PhotonDynamicMass": "a photon from a boosted decay has a rounding-level dynamic mass; an EM process "
                                   "that absorbs it shares its kinetic energy, so its pre-step mass is left; "
                                   "Geant4 behaviour",
            "G4DecayBoostMass": "G4Decay makes daughters at the parent's dynamic mass m and boosts with gamma = E/M, "
                                "so Q_decay = E (1 - m/M) for a rounding-level |m - M|; Geant4 behaviour"}


def report(tables, links, expected_events, *, rtol=1e-9, atol=1e-9):
    """Audit with links, verify the manifest, then account every inside step and every event."""
    require(np.isfinite(rtol) and np.isfinite(atol) and rtol >= 0 and atol > 0, "invalid accounting tolerance")
    audit = capture.audit(tables, expected_events, links=links)
    birth, surface, creator_rows = capture.check_links(tables, links)
    e, t, s = (tables[name] for name in ("events", "tracks", "steps"))
    events = unique(e, capture.EVENT, "event").assign(_event=np.arange(len(e["event"])))
    tracks = unique(t, capture.TRACK, "track").assign(_track=np.arange(len(t["track"])))
    sr = match(frame(s, capture.TRACK), tracks, capture.TRACK, "step")["_track"].to_numpy()
    se = match(frame(s, capture.EVENT), events, capture.EVENT, "step")["_event"].to_numpy()
    te = match(frame(t, capture.EVENT), events, capture.EVENT, "track")["_event"].to_numpy()
    n, nev = len(s["event"]), len(events)
    child = (t["parent"] != 0)
    rows = creator_rows[child]
    births_e = np.bincount(rows, weights=t["begin_energy"][child], minlength=n)
    pdg = t["pdg"][child]
    electrons = np.bincount(rows, weights=pdg == ELECTRON, minlength=n)
    positrons = np.bincount(rows, weights=pdg == POSITRON, minlength=n)
    neutrinos = np.bincount(rows, weights=np.isin(pdg, NEUTRINOS), minlength=n)
    nucleus_child = np.abs(pdg) >= 1000000000
    recoil_mass = np.bincount(rows, weights=np.where(nucleus_child, t["begin_mass"][child], 0.), minlength=n)
    recoils = np.bincount(rows, weights=nucleus_child, minlength=n)
    children = np.bincount(rows, minlength=n)
    process = s["post_process"].astype(str)
    killed = s["trackStatus"] == KILLED
    nucleus_kill = (np.isin(process, list(DECAY)) & killed & (children == 0) &
                    (np.abs(t["pdg"][sr]) >= 1000000000))
    destroyed = killed & np.isin(process, list(DESTROYING)) & ~nucleus_kill
    inside = s["preInside"].astype(bool)
    em_like = np.isin(process, list(TRANSPORT | EM))
    decay = np.isin(process, list(DECAY))
    parent_pdg = t["pdg"][sr]
    parent_lepton = np.where(parent_pdg == ELECTRON, 1, np.where(parent_pdg == POSITRON, -1, 0))
    taken = electrons - positrons - np.where(destroyed, parent_lepton, 0)
    lepton = np.isin(t["pdg"], [ELECTRON, POSITRON])
    masses = np.unique(t["begin_mass"][lepton])
    # Geant4 rebuilds the dynamic mass of boosted decay products from E and p: e+- masses scatter at rounding level.
    require(len(masses) == 0 or masses[-1] - masses[0] <= rtol * masses[-1] + atol, "inconsistent electron masses")
    muon_channel = decay & np.isin(parent_pdg, MUONS) & (children == 3) & (electrons + positrons == 1) & (neutrinos == 2)
    require(len(masses) > 0 or not (np.any(taken[em_like]) or np.any(muon_channel)),
            "electron-mass term without a recorded electron mass")
    electron_mass = float(np.median(t["begin_mass"][lepton])) if len(masses) else 0.
    medium = np.where(em_like, taken * electron_mass + recoil_mass, 0.)
    kept_post = np.where(destroyed, 0., s["post_energy"])
    q = s["pre_energy"] - kept_post - s["deposit"] - births_e
    tolerance = rtol * np.maximum(np.abs(s["pre_energy"]), 1.) + atol
    pre_mass = s["pre_mass"]
    photon_term = np.where(em_like & destroyed & (parent_pdg == PHOTON) & (pre_mass > 0) &
                           (pre_mass <= MASS_ROUNDING * np.abs(s["pre_energy"])), pre_mass, 0.)
    nominal = np.zeros(n)
    for p in np.unique(parent_pdg[decay]):
        values, counts = np.unique(t["begin_mass"][t["pdg"] == p], return_counts=True)
        nominal[decay & (parent_pdg == p)] = values[np.argmax(counts)] if len(values) else 0.
    near = decay & (nominal > 0) & (np.abs(pre_mass - nominal) <= MASS_ROUNDING * nominal)
    boost = np.where(near, s["pre_energy"] * (1. - pre_mass / np.where(near, nominal, 1.)), 0.)
    base_decay = np.where(muon_channel, electron_mass, 0.)
    boosted = (near & (boost != 0) & (np.abs(q - base_decay) > tolerance) &
               (np.abs(q - base_decay - boost) <= tolerance))
    expected_decay = base_decay + np.where(boosted, boost, 0.)
    residual = np.where(decay, q - expected_decay, q + medium - photon_term)
    em_steps = inside & em_like
    decay_steps = inside & decay
    nuclear = inside & ~em_like & ~decay
    bad = em_steps & (np.abs(residual) > tolerance)
    bad_decay = decay_steps & (np.abs(residual) > tolerance)
    worst = int(np.argmax(np.where(em_steps, np.abs(residual), -1.))) if em_steps.any() else -1

    def step_row(i):
        kids = np.flatnonzero(child)[rows == i]
        return {"run": int(s["run"][i]), "subrun": int(s["subrun"][i]), "event": int(s["event"][i]),
                "index": int(s["index"][i]), "process": process[i], "parent_pdg": int(parent_pdg[i]),
                "children_pdg": sorted(int(x) for x in t["pdg"][kids]), "q_MeV": float(q[i]),
                "expected_MeV": float(expected_decay[i]), "residual_MeV": float(residual[i])}

    # Event terms over inside steps and tracks.
    def per_event(values, mask, index):
        return np.bincount(index[mask], weights=values[mask], minlength=nev)
    primary = t["parent"] == 0
    entrance = per_event(t["entrance_energy"], primary, te)
    deposits = per_event(s["deposit"], inside, se)
    exits = per_event(s["post_energy"], inside & s["firstExit"].astype(bool), se)
    birth_exits = per_event(t["begin_energy"], birth, te)
    last = np.zeros(n, dtype=bool)
    order = frame(s, [*capture.TRACK, "number"]).assign(_row=np.arange(n)).sort_values([*capture.TRACK, "number"])
    last[order.groupby(capture.TRACK, sort=False).tail(1)["_row"].to_numpy()] = True
    ended_inside = last & inside & ~s["firstExit"].astype(bool) & ~destroyed
    stopped = per_event(s["post_energy"], ended_inside, se)
    zero_step = (t["steps"] == 0) & ~birth & child
    stopped += per_event(t["begin_energy"], zero_step, te)
    q_nuclear = per_event(q, nuclear, se)
    q_decay = per_event(expected_decay, decay_steps, se)
    q_photon = per_event(photon_term, em_steps, se)
    r_sum = per_event(residual, em_steps | decay_steps, se)
    medium_sum = per_event(medium, inside, se)
    left = entrance + medium_sum
    right = deposits + exits + birth_exits + stopped + q_nuclear + q_decay + q_photon + r_sum
    closure = left - right
    closure_tol = rtol * np.maximum(np.abs(left), 1.) + atol * np.maximum(1, np.bincount(se, minlength=nev))
    unprocessed = sorted(set(process[nuclear]))
    result = {
        "diagnostic_only": True, "bank_admissible": False, "audit": audit,
        "manifest_hash": str(links["manifest"]["manifestHash"][0]) if nev else "",
        "g4_version": str(links["manifest"]["g4Version"][0]) if nev else "",
        "tolerance": {"rtol": rtol, "atol_MeV": atol},
        "inside_steps": int(inside.sum()), "conserving_steps": int(em_steps.sum()),
        "nuclear_steps": int(nuclear.sum()), "nuclear_processes": unprocessed,
        "conserving_steps_out_of_tolerance": int(bad.sum()),
        "em_nuclear_recoils": int(recoils[inside & em_like].sum()),
        "decay_steps": int(decay_steps.sum()), "declared_muon_decay_steps": int((decay_steps & muon_channel).sum()),
        "declared_nucleus_kill_steps": int((decay_steps & nucleus_kill).sum()),
        "nucleus_kill_pdgs": sorted({int(x) for x in t["pdg"][sr][decay_steps & nucleus_kill]}),
        "declared_photon_mass_steps": int((em_steps & (photon_term > 0)).sum()),
        "max_declared_photon_mass_MeV": float(photon_term[em_steps].max()) if em_steps.any() else 0.,
        "declared_decay_boost_steps": int((decay_steps & boosted).sum()),
        "max_abs_declared_decay_boost_MeV": float(np.abs(np.where(decay_steps & boosted, boost, 0.)).max(initial=0.)),
        "decay_steps_out_of_tolerance": int(bad_decay.sum()),
        "max_abs_decay_residual_MeV": float(np.max(np.abs(residual[decay_steps]))) if decay_steps.any() else 0.,
        "decay_failures": [step_row(i) for i in np.flatnonzero(bad_decay)[:20]],
        "declared_terms": DECLARED,
        "max_abs_conserving_residual_MeV": float(np.max(np.abs(residual[em_steps]))) if em_steps.any() else 0.,
        "worst_step": ({"run": int(s["run"][worst]), "subrun": int(s["subrun"][worst]), "event": int(s["event"][worst]),
                        "index": int(s["index"][worst]), "process": process[worst],
                        "residual_MeV": float(residual[worst])} if worst >= 0 else None),
        "exits_at_birth": int(birth.sum()), "surface_born_secondaries": int((surface & child).sum()),
        "event_closure_out_of_tolerance": int(np.count_nonzero(np.abs(closure) > closure_tol)),
        "max_abs_event_closure_MeV": float(np.max(np.abs(closure))) if nev else 0.,
        "totals_MeV": {"entrance": float(entrance.sum()), "medium_electron_mass": float(medium_sum.sum()),
                       "deposit": float(deposits.sum()), "first_exit": float(exits.sum()),
                       "exit_at_birth": float(birth_exits.sum()), "ended_inside": float(stopped.sum()),
                       "nuclear_q": float(q_nuclear.sum()), "declared_decay_q": float(q_decay.sum()),
                       "declared_photon_mass": float(q_photon.sum()),
                       "conserving_residual": float(r_sum.sum())},
    }
    result["energy_accounting_passed"] = (result["conserving_steps_out_of_tolerance"] == 0 and
                                          result["decay_steps_out_of_tolerance"] == 0 and
                                          result["event_closure_out_of_tolerance"] == 0)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", default="dump")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-closure", action="store_true", help="exit nonzero, writing no report, on any open balance")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output already exists; refusing to clobber")
    tables = capture.read(args.input, args.directory)
    links = capture.read_links(args.input, args.directory)
    result = report(tables, links, args.expected_events)
    result["input"] = str(args.input)
    if args.require_closure:
        require(result["energy_accounting_passed"], "energy accounting does not close")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
