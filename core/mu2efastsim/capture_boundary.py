"""Constructed boundary-born ownership fixture (five cases); constructed objects only, not physics transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import capture
from .saved_g4 import require

CASES = ("owned-inward-on-surface", "owned-outward-zero-length-exit", "exit-at-birth",
         "entry-step-child-starting-inside", "entry-step-child-starting-outside")
ERRORS = ("", "", "",
          "nontrivial_outside_transport\nsecondary_inside_from_outside_interaction",
          "nontrivial_outside_transport\nsecondary_born_outside_envelope\ntrack_never_entered_component")


def report(tables, links, run, subrun=0):
    """Require the pinned outcome of every case: owner, flags, creator step, reason and recorder errors."""
    require(links is not None, "boundary-born fixture requires the links and manifest tables")
    e, t, s, c = (tables[name] for name in ("events", "tracks", "steps", "classifications"))
    lk, mf = links["links"], links["manifest"]
    require(len(e["event"]) == 5 and np.array_equal(np.sort(e["event"]), np.arange(1, 6)), "expected events 1..5")
    require(np.all(e["run"] == run) and np.all(e["subrun"] == subrun), "run/subrun census mismatch")
    require(np.all(e["capturePresent"] == True) and np.all(e["eventIDsMatch"] == True), "missing capture")
    require(np.all(mf["linksPresent"] == True) and np.all(mf["linksIDsMatch"] == True), "missing links")
    hashes = [capture.fnv1a(str(text)) for text in mf["manifest"]]
    require(hashes == [str(h) for h in mf["manifestHash"]], "manifest hash mismatch")
    require(np.all(e["bankAdmissible"] == False) and np.all(e["recordBankAdmissible"] == False), "bank admissible")
    for field in ("track", "parent", "index"):
        require(np.array_equal(lk[field], t[field]) and np.array_equal(lk["event"], t["event"]), f"links {field} misaligned")
    cases = []
    for number in range(1, 6):
        ev = e["event"] == number
        errors = str(e["errors"][ev][0])
        require(errors == ERRORS[number - 1], f"case {number}: recorder errors {errors!r}")
        require(bool(e["recordComplete"][ev][0]) == (ERRORS[number - 1] == ""), f"case {number}: completeness")
        tv, sv = t["event"] == number, s["event"] == number
        require(np.count_nonzero(tv) == 2, f"case {number}: track census")
        child = tv & (t["parent"] == 1)
        require(np.count_nonzero(child) == 1, f"case {number}: child census")
        k = np.flatnonzero(child)[0]
        entry = number >= 4
        steps = np.flatnonzero(sv & (s["track"] == 1))
        require(len(steps) == 3, f"case {number}: parent step census")
        creator_index = 0 if entry else 2
        require(lk["creatorStep"][k] == creator_index, f"case {number}: creator step")
        require(s["secondaries"][steps][creator_index] == 1, f"case {number}: creator step secondaries")
        require(bool(s["preInside"][steps][creator_index]) == (not entry), f"case {number}: creator step location")
        expected = {1: (True, False, 1, True, 0), 2: (True, False, 2, True, 1), 3: (True, True, 4, False, 0),
                    4: (True, False, 1, True, 0), 5: (True, False, 1, False, 0)}[number]
        surface, birth, reason, entered, child_steps = expected
        require(bool(lk["bornOnSurface"][k]) == surface, f"case {number}: bornOnSurface")
        require(bool(lk["exitAtBirth"][k]) == birth, f"case {number}: exitAtBirth")
        require(t["reason"][k] == reason, f"case {number}: reason {t['reason'][k]}")
        require(bool(t["entered"][k]) == entered, f"case {number}: entered")
        require(t["steps"][k] == child_steps, f"case {number}: child steps")
        cv = (c["event"] == number) & (c["track"] == 2)
        require(np.count_nonzero(cv) == 1 and bool(c["killed"][cv][0]) == birth, f"case {number}: classification kill")
        if number == 2:
            exit_step = np.flatnonzero(sv & (s["track"] == 2))
            require(len(exit_step) == 1 and s["firstExit"][exit_step[0]] and s["length"][exit_step[0]] == 0.,
                    "case 2: zero-length first exit")
        cases.append({"case": CASES[number - 1], "event": number, "errors": errors,
                      "bornOnSurface": bool(lk["bornOnSurface"][k]), "exitAtBirth": bool(lk["exitAtBirth"][k]),
                      "reason": int(t["reason"][k]), "creatorStep": int(lk["creatorStep"][k])})
    return {"diagnostic_only": True, "bank_admissible": False, "physical_transport": False,
            "constructed_objects": True, "boundary_born_fixture_passed": True, "run": run, "subrun": subrun,
            "events": 5, "manifest_hash": hashes[0], "cases": cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--subrun", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output already exists; refusing to clobber")
    result = report(capture.read(args.input), capture.read_links(args.input), args.run, args.subrun)
    result["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
