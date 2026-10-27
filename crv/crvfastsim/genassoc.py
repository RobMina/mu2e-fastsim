"""Read-only schema-v2 generator association gate; no Offline or kinematic matching.

ProductIDs identify collections *within an event*, not across independent donor and
full-generator files. Joins retain the complete integer (run, subrun, event) tuple.
Only file/column loops and vectorized, logarithmic-depth ancestry traversal are used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

KEY = ["run", "subrun", "event"]
SCHEMA_VERSION = 2
EVENT_COLUMNS = KEY + ["schema_version", "associations_checked", "full_generator_stream",
                       "gen_product_id", "sim_product_id", "n_gen", "n_particles", "n_steps"]
GEN_COLUMNS = KEY + ["idx", "gen_product_id", "pdg", "e", "t"]
PART_COLUMNS = KEY + ["key", "sim_product_id", "parent", "parent_product_id", "genidx",
                      "gen_product_id", "isprim", "pdg", "se", "st"]
STEP_COLUMNS = KEY + ["sp", "sp_product_id"]


def _require(ok, message):
    if not bool(np.all(ok)):
        raise ValueError(message)


def _read(root, name, columns, report):
    treepath = "dump/" + name
    cycles = sorted(int(k.rsplit(";", 1)[1]) for k in root.keys(recursive=True, cycle=True)
                    if k.rsplit(";", 1)[0] == treepath)
    _require(bool(cycles), f"missing {treepath}; strict association validation requires schema v2 (re-dump inputs)")
    tree = root[f"{treepath};{cycles[-1]}"]
    missing = set(columns) - set(tree.keys())
    _require(not missing, f"{treepath}: missing schema-v2 branches {sorted(missing)}; re-dump inputs")
    frame = pd.DataFrame(tree.arrays(columns, library="np"))
    report[name] = {"rows": len(frame), "cycles": cycles, "selected_cycle": cycles[-1]}
    return frame


def _integer(frame, columns, label, dtype=None):
    for column in columns:
        a = frame[column].to_numpy()
        _require(a.dtype.kind in "iu", f"{label}.{column}: must be integer, not {a.dtype}")
        if dtype is not None:
            kind, size = dtype
            _require(a.dtype.kind == kind and a.dtype.itemsize == size,
                     f"{label}.{column}: requires {'uint' if kind == 'u' else 'int'}{8 * size}")
        # Avoid unsigned/signed merge promotion, and never pass identities through float.
        _require(a <= np.iinfo(np.int64).max, f"{label}.{column}: exceeds signed int64")
        frame[column] = a.astype(np.int64)


def _unique(frame, columns, label):
    _require(not frame.duplicated(columns).any(), f"{label}: duplicate keys {columns}")


def _lookup(left, right, left_columns, right_columns, label):
    """Exact integer many-to-one join, returning right row indices in left order."""
    target = right[right_columns].rename(columns=dict(zip(right_columns, left_columns))).copy()
    # Nullable integer avoids even temporary float conversion when a join is missing.
    target["_row"] = pd.array(np.arange(len(right), dtype=np.int64), dtype="Int64")
    joined = left[left_columns].merge(target, on=left_columns, how="left", sort=False,
                                      validate="many_to_one", indicator=True)
    _require((joined["_merge"] == "both").all(), f"{label}: missing referenced identity")
    return joined["_row"].to_numpy(dtype=np.int64)


def _events(root, report, donor):
    events = _read(root, "events", EVENT_COLUMNS, report)
    _integer(events, KEY + ["schema_version", "gen_product_id", "sim_product_id"], "events", ("u", 4))
    _integer(events, ["associations_checked", "full_generator_stream"], "events")
    _integer(events, ["n_gen", "n_particles", "n_steps"], "events", ("i", 8))
    _unique(events, KEY, "events")
    _require(len(events) > 0, "events: empty census cannot establish schema or stream role")
    _require(events.schema_version == SCHEMA_VERSION, "unsupported association schema; require schema_version=2")
    _require(events.associations_checked == (1 if donor else 0),
             "events: associations_checked must be 1 for donors, 0 for full generator streams")
    _require(events.full_generator_stream == (0 if donor else 1),
             "events: full_generator_stream must be 0 for donors, 1 for external denominators; accepted donor gen is not full exposure")
    _require(events[["n_gen", "n_particles", "n_steps"]].to_numpy() >= 0, "events: negative payload counts")
    _require(events.gen_product_id > 0, "events: null generator ProductID")
    if donor:
        _require(events.sim_product_id > 0, "events: null SimParticle ProductID")
    else:
        _require(events.sim_product_id == 0, "full generator stream: nonnull SimParticle ProductID")
        _require(events[["n_particles", "n_steps"]].to_numpy() == 0,
                 "full generator stream: nonzero particle/step census")
    return events


def _census(frame, events, name):
    _integer(frame, KEY, name, ("u", 4))
    rows = _lookup(frame, events, KEY, KEY, f"{name} event census")
    counts = np.bincount(rows, minlength=len(events))
    _require(counts == events["n_" + name].to_numpy(), f"{name}: event census payload count mismatch (including zero rows)")
    return rows


def _gen(root, events, report):
    gen = _read(root, "gen", GEN_COLUMNS, report)
    rows = _census(gen, events, "gen")
    _integer(gen, ["gen_product_id"], "gen", ("u", 4))
    _integer(gen, ["idx", "pdg"], "gen")
    _unique(gen, KEY + ["idx"], "gen")
    _require(gen.gen_product_id.to_numpy() == events.gen_product_id.to_numpy()[rows],
             "gen: generator ProductID differs from event census")
    _require((gen.idx.to_numpy() >= 0) & (gen.idx.to_numpy() < events.n_gen.to_numpy()[rows]),
             "gen: index out of range")
    _require(np.isfinite(gen[["e", "t"]].to_numpy()), "gen: nonfinite energy/time")
    return gen


def _report(path, trees, role):
    return {"path": str(Path(path).resolve()), "schema_version": SCHEMA_VERSION,
            "stream_role": role, "trees": trees,
            "counts": {name: info["rows"] for name, info in trees.items()}}


def validate_donor(path):
    """Reject incomplete/legacy/misassociated donors; return JSON-safe validation metadata.

    Every nonnull pointer is checked, including inherited generator pointers on
    secondaries (whose PDG need not equal the generator PDG). Every used step must
    reach an actual isprim root; a parentless nonprimary is not a primary.
    """
    trees = {}
    try:
        with uproot.open(path) as root:
            events = _events(root, trees, donor=True)
            gen = _gen(root, events, trees)
            parts = _read(root, "particles", PART_COLUMNS, trees)
            steps = _read(root, "steps", STEP_COLUMNS, trees)
        prows = _census(parts, events, "particles")
        srows = _census(steps, events, "steps")
        _integer(parts, ["sim_product_id", "parent_product_id", "gen_product_id"], "particles", ("u", 4))
        _integer(parts, ["key", "parent", "genidx", "isprim", "pdg"], "particles")
        _integer(steps, ["sp_product_id"], "steps", ("u", 4))
        _integer(steps, ["sp"], "steps")
        _unique(parts, KEY + ["key"], "particles")
        _require(parts.key >= 0, "particles: negative key")
        _require(parts.isprim.isin([0, 1]), "particles: isprim must be 0 or 1")
        _require(parts.sim_product_id.to_numpy() == events.sim_product_id.to_numpy()[prows],
                 "particles: SimParticle ProductID differs from census")
        _require(steps.sp_product_id.to_numpy() == events.sim_product_id.to_numpy()[srows],
                 "steps: SimParticle ProductID differs from census")
        used = _lookup(steps, parts, KEY + ["sp_product_id", "sp"],
                       KEY + ["sim_product_id", "key"], "steps SimParticle")
        for key, product in (("parent", "parent_product_id"), ("genidx", "gen_product_id")):
            _require(parts[key] >= -1, f"particles: invalid {key}")
            _require((parts[key] == -1) == (parts[product] == 0),
                     f"particles: null {product} requires key=-1 and vice versa")
        hasgen = parts.genidx >= 0
        referenced = _lookup(parts.loc[hasgen], gen, KEY + ["gen_product_id", "genidx"],
                             KEY + ["gen_product_id", "idx"], "particles generator ProductID/index")
        primary = parts.isprim.to_numpy() == 1
        _require(~primary | hasgen.to_numpy(), "primary: missing generator pointer")
        genprim = primary[hasgen.to_numpy()]
        _require(parts.loc[hasgen, "pdg"].to_numpy()[genprim] == gen.pdg.to_numpy()[referenced[genprim]],
                 "primary: signed PDG disagrees with referenced generator")
        _require(np.isfinite(parts.loc[primary, ["se", "st"]].to_numpy()), "primary: nonfinite energy/time")
        hasparent = parts.parent.to_numpy() >= 0
        _require(~primary | ~hasparent, "primary: must be an actual parentless isprim root")
        parent = np.arange(len(parts), dtype=np.int64)
        parent[hasparent] = _lookup(parts.loc[hasparent], parts, KEY + ["parent_product_id", "parent"],
                                   KEY + ["sim_product_id", "key"], "particles parent ProductID/key")
        ancestors = parent.copy()
        passes = 0
        # Pointer doubling finds roots; the original parent flag detects cycles that converge to self.
        for _ in range(len(parts).bit_length()):
            advanced = ancestors[ancestors]
            passes += 1
            if np.array_equal(advanced, ancestors):
                break
            ancestors = advanced
        _require(~hasparent[ancestors], "particles: ancestry cycle (no parentless root)")
        _require(primary[ancestors[used]], "steps: ancestry reaches a nonprimary/truncated root, not true isprim")
        report = _report(path, trees, "checked donor")
        report.update(associations_checked=True, ancestry_cycles=0, ancestry_passes=passes,
                      primary_particles=int(primary.sum()), used_step_particles=int(len(np.unique(used))))
        return report
    except ValueError as exc:
        raise ValueError(f"{path}: strict generator association validation failed: {exc}") from exc


def generator_event_keys(path):
    """Complete schema-v2 census, or explicitly legacy primary-row event inference."""
    with uproot.open(path) as root:
        if "dump/events" in root:
            return _events(root, {}, donor=False)[KEY]
        keys = pd.DataFrame(root["dump/gen"].arrays(KEY, library="np"))
    _integer(keys, KEY, "legacy gen")
    return keys.drop_duplicates()


def validate_generator_stream(path):
    """Check external denominator role, never compare its ProductIDs with donors.

    Historical full-gen files without an events tree remain usable but are explicitly
    unverified. A partially upgraded file (new gen ProductID without census) is not legacy.
    """
    trees = {}
    try:
        with uproot.open(path) as root:
            if "dump/events" not in root:
                gen = _read(root, "gen", KEY + ["idx", "pdg", "e", "t"], trees)
                _require("gen_product_id" not in root["dump/gen"].keys(),
                         "schema-v2 generator payload missing events census; not a legacy stream")
                _integer(gen, KEY + ["idx", "pdg"], "legacy gen")
                _unique(gen, KEY + ["idx"], "legacy gen")
                _require(gen.idx >= 0, "legacy gen: negative index")
                _require(np.isfinite(gen[["e", "t"]].to_numpy()), "legacy gen: nonfinite energy/time")
                report = _report(path, trees, "unverified legacy stream role")
                report["schema_version"] = None
                return report
            events = _events(root, trees, donor=False)
            _gen(root, events, trees)
            # Gen-only dumpers may create empty particle/step trees; never accept payload.
            for name in ("particles", "steps"):
                if "dump/" + name in root:
                    frame = _read(root, name, KEY, trees)
                    _census(frame, events, name)
        return _report(path, trees, "full generator stream")
    except ValueError as exc:
        raise ValueError(f"{path}: external generator denominator validation failed: {exc}") from exc
