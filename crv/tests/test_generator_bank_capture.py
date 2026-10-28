"""Synthetic strict-capture tests; no art, remote IO or builds."""
import json

import numpy as np
import pytest

from test_genassoc_fullgen import rootfile
from valhist import generator_bank_capture as B, genassoc_fullgen as F


def inventory():
    return {"candidate_only": True, "downstream_authorized": False, "events": 94_000_000,
            "files": [{"run": 1400, "subrun": s, "events_inferred": 1_000_000,
                       "path": f"/data/gen.001400_{s:08d}.root", "sha256": "a" * 64}
                      for s in range(2001, 2101) if s not in B.MISSING]}


def test_inventory_exact_partition():
    row = B.select_record(inventory(), 2001)
    assert row["art"].endswith(".001400_00002001.art")
    assert row["events_inferred"] == 1_000_000
    with pytest.raises(ValueError, match="absent"):
        B.select_record(inventory(), 2011)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "run", "count", "promoted", "path"])
def test_inventory_rejects(kind):
    source = inventory()
    if kind == "duplicate":
        source["files"][-1] = source["files"][0]
    elif kind == "missing":
        source["files"].pop()
    elif kind == "run":
        source["files"][0]["run"] += 1
    elif kind == "count":
        source["files"][0]["events_inferred"] -= 1
    elif kind == "promoted":
        source["downstream_authorized"] = True
    else:
        source["files"][0]["path"] = "/pnfs/data/gen.001400_00002001.root"
    with pytest.raises(ValueError):
        B.select_record(source, 2001)


def files(tmp_path, case=None):
    old, new = tmp_path / "old.root", tmp_path / "new.root"

    def mutate(p):
        p["gen"]["event"][:] = 1
        if "events" not in p:
            return
        e = p["events"]
        e["event"][:] = [1, 3 if case == "gap" else 2]
        e["ngen"] = e["n_gen"].copy()
        e["nsteps"] = np.full(2, 0 if case == "sentinel" else -1, dtype="int32")
        if case == "alias":
            e["ngen"][1] = 1
        if case == "payload":
            p["gen"]["px"][0] = np.nextafter(3., 4.)
    rootfile(old, 2001, mutation=mutate)
    rootfile(new, 2001, strict=True, mutation=mutate)
    if case == "payload":
        rootfile(old, 2001, mutation=lambda p: p["gen"]["event"].fill(1))
    return old, new


def test_explicit_count_and_empty_census(tmp_path):
    old, new = files(tmp_path)
    report = B.compare_file(old, new, 1400, 2001, 2)
    assert report["event_count"] == 2 and report["zero_generator_events"] == 1
    assert report["complete_event_range"] and report["merged_census_aliases"]
    assert F.GENERATED_PER_FILE == 50_000
    with pytest.raises(ValueError, match="50000"):
        F.compare_payload(old, new)


@pytest.mark.parametrize("case", ["gap", "sentinel", "alias", "payload", "partition"])
def test_complete_file_guards(tmp_path, case):
    old, new = files(tmp_path, case)
    with pytest.raises((ValueError, AssertionError)):
        B.compare_file(old, new, 1400, 2002 if case == "partition" else 2001, 2)


def test_failed_attempt_is_retained_and_not_reused(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    out = tmp_path / "attempt"
    with pytest.raises(ValueError, match="candidate"):
        B.capture(candidate, 2001, tmp_path, out)
    failure = json.loads((out / "failed.json").read_text())
    assert failure["downstream_authorized"] is False
    assert not (out / "certificate.json").exists()
    with pytest.raises(FileExistsError):
        B.capture(candidate, 2001, tmp_path, out)


def test_approved_configuration_closure():
    hashes = {"/area/" + name: digest for name, digest in B.CONFIG_SHA256.items()}
    assert B.approved_configs(hashes) == B.CONFIG_SHA256
    for bad in ({**hashes, "/area/extra_output.fcl": "bad"},
                {**hashes, "/area/CrvFastSim/fcl/dump_gen.fcl": "changed"}, {}):
        with pytest.raises(ValueError, match="configuration"):
            B.approved_configs(bad)
