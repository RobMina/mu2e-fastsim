"""Mu2eG4 component captures (schema 1): the Mu2e reader and the Mu2e checks over the generic capture audit.

The audit, the links check and the table helpers are hep-simlib's (hepsimlib.capture, docs/recorder.md there), ported
from this module. What stays here is what only the Mu2eG4 recorder writes or needs: its ROOT layout (directory
"dump", art event and record identities), its event fields (capture and status products, StatusG4, SimParticle
overflow, step-limit and field-propagator kill counts, validation errors), the natural-boundary-stop rule of
Mu2eG4TrackingAction and its resource-kill process. ``audit`` runs the generic audit on a view of the tables (the
generic resource-kill count is supplied as zero: Mu2eG4 counts its kills in its own fields) with mu2e_checks as its
framework checks (passed in AUDIT_OPTIONS). Passing never certifies a physics bank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from hepsimlib.capture import audit as _generic
from hepsimlib.capture.audit import check_links, same, state_equal  # noqa: F401  (re-exported for the capture_* modules)
from hepsimlib.capture.tables import fnv1a, vectors  # noqa: F401

from .saved_g4 import frame, match, require, unique

EVENT = ["run", "subrun", "event"]
TRACK = [*EVENT, "track"]
TABLES = ("events", "tracks", "steps", "classifications", "volumes")
STATES = {"tracks": ("begin_", "end_", "entrance_"), "steps": ("pre_", "post_"),
          "classifications": ("state_",)}
FLOATS = "x y z px py pz lx ly lz lpx lpy lpz energy kinetic mass charge time properTime weight".split()
STATE = [*FLOATS, "volume", "placement", "material", "process", "stepStatus"]
FIELDS = {
    "events": "schema geometryTag physicsTag envelope translation rotation halfSize complete bankAdmissible "
              "recordComplete recordBankAdmissible recordRun recordSubrun recordEvent capturePresent statusPresent "
              "eventIDsMatch errors validationErrors nTracks nSteps nClassifications nVolumes nErrors g4Status "
              "statusClean nG4Tracks overflowSimParticles nKilledStepLimit nKilledByFieldPropagator cpuTime realTime",
    "tracks": "index track parent steps secondaries pdg status reason entered ended creator stopProcess",
    "steps": "index track number secondaries deposit nonIonizingDeposit length trackStatus preInside postInside firstExit",
    "classifications": "index track parent pdg status killed",
    "volumes": "index id copy name logical motherLogical material solid density translation rotation",
}
# Written by newer dumps; captures without them are still read and audited.
OPTIONAL_FIELDS = {"events": ("nNaturalBoundaryStops",)}
INT64 = set(EVENT + "index track parent steps secondaries number id copy recordRun recordSubrun recordEvent "
            "nTracks nSteps nClassifications nVolumes nErrors nNaturalBoundaryStops".split())
# Mu2eG4TrackingAction exempts only these from its field-propagator count when their range is below rangeToIgnore.
FIELD_KILL_EXEMPT_PDG = (11, 2212)
LINKS = ("links", "manifest")
LINK_FIELDS = "index track parent creatorStep bornOnSurface exitAtBirth".split()
MANIFEST_FIELDS = "linksPresent linksIDsMatch schema g4Version manifest manifestHash nLinks nBirthExits".split()
# Geant4 builds G4CoupledTransportation instead of G4Transportation whenever a G4ScoringManager exists.
PROCESS_CLASSES = {"CoupledTransportation": "Transportation"}
# Mu2eG4's own resource kill (a step-limit or special-cuts kill) is never a physical outcome.
RESOURCE_PROCESSES = ("mu2eSpecialCutsProcess",)
# Event fields only Mu2eG4 writes: checked by mu2e_checks, not by the generic audit.
MU2E_EVENT_FIELDS = ("recordComplete recordBankAdmissible recordRun recordSubrun recordEvent capturePresent "
                     "statusPresent eventIDsMatch validationErrors g4Status statusClean overflowSimParticles "
                     "nKilledStepLimit nKilledByFieldPropagator").split()


def is_process_field(field):
    """True for capture fields that hold a Geant4 process name."""
    return field in ("creator", "stopProcess") or field.endswith("process")


def process_class(values):
    """Map Geant4 process names to their process class; transport variants form one class."""
    return np.array([PROCESS_CLASSES.get(str(v), str(v)) for v in np.ravel(values)], dtype=object).reshape(np.shape(values))


def read(path, directory="dump"):
    """Read highest ROOT cycles; retain ragged metadata for mixed missing/present diagnostic products."""
    def column(key, value):
        if key in ("translation", "rotation", "halfSize"):
            counts = ak.to_numpy(ak.num(value, axis=1))
            if len(counts) and np.any(counts != counts[0]):
                return value
        return ak.to_numpy(value)
    with uproot.open(path) as root:
        return {name: {key: column(key, value) for key, value in
                       root[f"{directory}/{name}"].arrays(library="ak", how=dict).items()} for name in TABLES}


def read_links(path, directory="dump"):
    """Read the per-track links and per-event manifest trees, or None for captures written without them."""
    with uproot.open(path) as root:
        try:
            trees = {name: root[f"{directory}/{name}"] for name in LINKS}
        except KeyError:
            return None
        return {name: {key: ak.to_numpy(value) for key, value in tree.arrays(library="ak", how=dict).items()}
                for name, tree in trees.items()}


def natural_boundary_stops(tables):
    """Per-event count of tracks Mu2eG4 counts as field-propagator kills although they stopped naturally.

    Such a track ends by Transportation at zero kinetic energy on a last step that stays in one placement, and is
    neither an electron nor a proton (whose zero range is below rangeToIgnore).
    """
    e, t, s = tables["events"], tables["tracks"], tables["steps"]
    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    order = frame(s, [*TRACK, "number"]).assign(_row=np.arange(len(s["event"]))).sort_values([*TRACK, "number"])
    last = order.groupby(TRACK, sort=False).tail(1)
    rows = last["_row"].to_numpy()
    same = s["pre_placement"].astype(str)[rows] == s["post_placement"].astype(str)[rows]
    last = last.assign(_same=same)[[*TRACK, "_same"]]
    tracks = frame(t, TRACK).assign(_row=np.arange(len(t["track"]))).merge(last, on=TRACK, how="left")
    tracks = tracks.sort_values("_row")
    stopped = (t["ended"].astype(bool) & (t["stopProcess"].astype(str) == "Transportation") & (t["end_kinetic"] == 0)
               & ~np.isin(t["pdg"], FIELD_KILL_EXEMPT_PDG) & tracks["_same"].fillna(False).to_numpy(dtype=bool))
    rows = match(frame(t, EVENT), events, EVENT, "tracks")["_event"].to_numpy()
    return np.bincount(rows, weights=stopped, minlength=len(events)).astype(np.int64)


def diagnose(tables):
    """Summarize invalid smoke records without accepting them or changing any stored value."""
    e, t, s = (tables[name] for name in ("events", "tracks", "steps"))
    outside = ~s["preInside"]
    columns = [*EVENT, "track", "number", "deposit", "secondaries", "pre_material", "post_material",
               "preInside", "postInside", "pre_lz", "post_lz", "pre_kinetic", "post_kinetic", "post_process"]
    print(frame(e, [*EVENT, "complete", "errors", "nTracks", "nSteps"]).to_string(index=False))
    print(frame(s, columns).loc[outside].head(20).to_string(index=False))
    return {"diagnostic_only": True, "bank_admissible": False, "events": len(e["event"]),
            "tracks": len(t["track"]), "steps": len(s["track"]), "outside_steps": int(outside.sum()),
            "outside_deposit": float(s["deposit"][outside].sum()),
            "outside_secondaries": int(s["secondaries"][outside].sum())}


def mu2e_checks(tables, links=None):
    """The checks of a Mu2eG4 capture beyond the generic audit (which has run already): its own event fields, the
    record identities, StatusG4 and the field-propagator kills, each of which must be a natural boundary stop."""
    e = tables["events"]
    n = len(e["event"])
    for key in MU2E_EVENT_FIELDS + [k for k in OPTIONAL_FIELDS["events"] if k in e]:
        require(key in e, f"missing events fields: {{{key!r}}}")
        try:
            value = np.asarray(e[key])
        except ValueError as exc:
            raise ValueError(f"events: malformed {key}") from exc
        require(value.ndim > 0 and len(value) == n, f"events: malformed {key}")
        if key in INT64:
            require(value.dtype.kind == "i" and value.dtype.itemsize == 8, f"events: {key} must be int64")
        if value.dtype.kind == "f":
            require(np.isfinite(value).all(), f"events: nonfinite {key}")
    for field in "recordComplete capturePresent statusPresent eventIDsMatch statusClean".split():
        require(np.all(e[field] == True), f"event {field} is not true")  # noqa: E712
    for field in "recordBankAdmissible overflowSimParticles nKilledStepLimit".split():
        require(np.all(e[field] == 0), f"invalid event {field}")
    # Mu2eG4 sets status 1 for any field-propagator kill; each must be a natural stop recomputed from the tables.
    natural = natural_boundary_stops(tables)
    require(np.array_equal(e["nKilledByFieldPropagator"], natural), "invalid event nKilledByFieldPropagator")
    require(np.array_equal(e["g4Status"], np.where(natural > 0, 1, 0)), "invalid event g4Status")
    if "nNaturalBoundaryStops" in e:
        require(np.array_equal(e["nNaturalBoundaryStops"], natural), "natural boundary stop census mismatch")
    require(np.all(e["validationErrors"] == ""), "event validationErrors present")
    for field in EVENT:
        require(np.array_equal(e[field], e["record" + field.capitalize()]), "record event identity mismatch")


AUDIT_OPTIONS = dict(resource_processes=RESOURCE_PROCESSES, checks=(mu2e_checks,))


def generic_view(tables):
    """The tables as the generic audit reads them: the same arrays, plus a zero generic resource-kill count (Mu2eG4
    counts its kills in its own fields, checked by mu2e_checks), and nG4Tracks as int64 when it is an integer column
    (the Mu2eG4 schema never required int64 for it; the values are unchanged)."""
    view = dict(tables)
    e = tables.get("events")
    if isinstance(e, dict) and "event" in e:
        e = dict(e)
        if "nResourceKills" not in e:
            e["nResourceKills"] = np.zeros(len(e["event"]), dtype=np.int64)
        if "nG4Tracks" in e:
            g = np.asarray(e["nG4Tracks"])
            if g.dtype.kind in "iu" and g.dtype != np.int64:
                e["nG4Tracks"] = g.astype(np.int64)
        view["events"] = e
    return view


def audit(tables, expected_events, *, atol=1e-7, links=None):
    """Fail closed on missing/partial events; atol is an absolute geometry/state tolerance (mm, MeV, ns).

    The generic audit (hepsimlib.capture.audit) with the Mu2eG4 checks. With links, flagged exit-at-birth children
    (reason 4) and surface-born children are accepted.
    """
    return _generic.audit(generic_view(tables), expected_events, atol=atol, links=links, **AUDIT_OPTIONS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--directory", default="dump")
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true", help="report invalid data without certifying it")
    args = parser.parse_args(argv)
    tables = read(args.input, args.directory)
    result = diagnose(tables) if args.diagnose else audit(tables, args.expected_events)
    result["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
