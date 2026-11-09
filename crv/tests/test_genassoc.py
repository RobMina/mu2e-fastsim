"""Strict schema-v2 association tests using tiny real ROOT TTrees, no Offline."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim import genassoc as G, tier1b


def payload():
    # Large uint32 event components intentionally alias under packed 20-bit keys.
    events = pd.DataFrame(dict(run=[2**32 - 1, 2**31 + 3, 2**32 - 1],
                               subrun=[2**31 + 1, 1, 2**31 + 1],
                               event=[2**32 - 1, 2**32 - 1, 0],
                               schema_version=2, associations_checked=1, full_generator_stream=0,
                               gen_product_id=[2**32 - 2, 101, 101],
                               sim_product_id=[2**32 - 3, 202, 202],
                               n_gen=[1, 1, 0], n_particles=[2, 1, 0], n_steps=[1, 1, 0]))
    for c in G.EVENT_COLUMNS:
        events[c] = events[c].astype(np.int64 if c.startswith("n_") else np.uint32)
    gen = events.iloc[:2][G.KEY + ["gen_product_id"]].copy().reset_index(drop=True)
    gen = gen.assign(idx=np.int64(0), pdg=np.int32(-13), e=200., t=0.)
    parts = events.iloc[[0, 0, 1]][G.KEY + ["sim_product_id", "gen_product_id"]].copy().reset_index(drop=True)
    parts = parts.assign(key=np.array([2**60 + 7, 2**60 + 8, 2**60 + 7], dtype=np.int64),
                         parent=np.array([-1, 2**60 + 7, -1], dtype=np.int64),
                         parent_product_id=np.array([0, 2**32 - 3, 0], dtype=np.uint32),
                         genidx=np.int64(0), isprim=np.array([1, 0, 1], dtype=np.int32),
                         pdg=np.array([-13, 11, -13], dtype=np.int32), se=200., st=0.)
    steps = parts.iloc[[1, 2]][G.KEY + ["key", "sim_product_id"]].rename(
        columns={"key": "sp", "sim_product_id": "sp_product_id"}).reset_index(drop=True)
    return dict(events=events, gen=gen, particles=parts, steps=steps)


def write(path, frames, update=False):
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        for name, frame in frames.items():
            cols = {c: frame[c].to_numpy() for c in frame}
            root.mktree("dump/" + name, {c: a.dtype for c, a in cols.items()})
            if len(frame):
                root["dump/" + name].extend(cols)
    return path


def full_gen(frames):
    frames = deepcopy(frames)
    frames["events"] = frames["events"].assign(associations_checked=np.uint32(0),
        full_generator_stream=np.uint32(1), sim_product_id=np.uint32(0),
        n_particles=np.int64(0), n_steps=np.int64(0))
    frames["particles"] = frames["particles"].iloc[:0]
    frames["steps"] = frames["steps"].iloc[:0]
    return frames


def test_valid_large_integer_keys_multiple_runs_inherited_gen_and_empty_event(tmp_path):
    report = G.validate_donor(write(tmp_path / "donor.root", payload()))
    assert report["schema_version"] == 2
    assert report["counts"] == dict(events=3, gen=2, particles=3, steps=2)
    assert report["primary_particles"] == 2 and report["ancestry_cycles"] == 0
    assert report["associations_checked"] is True


@pytest.mark.parametrize("tree,column,row,value,match", [
    ("events", "schema_version", 0, 1, "unsupported association schema"),
    ("events", "associations_checked", 0, 0, "associations_checked"),
    ("events", "full_generator_stream", 0, 1, "full_generator_stream"),
    ("events", "n_gen", 2, 1, "census payload count mismatch"),
    ("events", "n_particles", 0, -1, "negative payload counts"),
    ("gen", "gen_product_id", 0, 101, "ProductID differs"),
    ("gen", "idx", 0, 1, "index out of range"),
    ("particles", "gen_product_id", 0, 101, "generator ProductID/index"),
    ("particles", "gen_product_id", 1, 101, "generator ProductID/index"),
    ("particles", "genidx", 0, 1, "generator ProductID/index"),
    ("particles", "genidx", 0, -2, "invalid genidx"),
    ("particles", "pdg", 0, 13, "signed PDG"),
    ("particles", "parent_product_id", 1, 202, "parent ProductID/key"),
    ("particles", "parent", 1, 100, "parent ProductID/key"),
    ("particles", "sim_product_id", 1, 202, "SimParticle ProductID differs"),
    ("particles", "parent_product_id", 0, 202, "null parent_product_id"),
    ("particles", "gen_product_id", 0, 0, "null gen_product_id"),
    ("steps", "sp_product_id", 0, 202, "SimParticle ProductID differs"),
    ("steps", "sp", 0, 100, "steps SimParticle"),
    ("steps", "event", 0, 17, "event census"),
    ("gen", "e", 0, np.nan, "nonfinite energy/time"),
    ("gen", "t", 0, np.inf, "nonfinite energy/time"),
    ("particles", "se", 0, np.nan, "primary: nonfinite"),
    ("particles", "st", 0, np.inf, "primary: nonfinite"),
])
def test_corruption_rejected(tmp_path, tree, column, row, value, match):
    frames = payload()
    frames[tree].loc[row, column] = value
    with pytest.raises(ValueError, match=match):
        G.validate_donor(write(tmp_path / "bad.root", frames))


@pytest.mark.parametrize("tree", ["events", "gen", "particles"])
def test_duplicate_keys(tmp_path, tree):
    frames = payload()
    frames[tree] = pd.concat([frames[tree], frames[tree].iloc[[0]]], ignore_index=True)
    # Preserve the census to ensure duplicate-key rejection is tested independently.
    if tree != "events":
        frames["events"].loc[0, "n_" + tree] += 1
    with pytest.raises(ValueError, match="duplicate keys"):
        G.validate_donor(write(tmp_path / "bad.root", frames))


def test_duplicate_steps_are_not_duplicate_particle_keys(tmp_path):
    frames = payload()
    frames["steps"] = pd.concat([frames["steps"], frames["steps"].iloc[[0]]], ignore_index=True)
    frames["events"].loc[0, "n_steps"] += 1
    assert G.validate_donor(write(tmp_path / "good.root", frames))["counts"]["steps"] == 3


@pytest.mark.parametrize("tree", ["gen", "particles", "steps"])
def test_partial_payload_and_zero_rows(tmp_path, tree):
    frames = payload()
    frames[tree] = frames[tree].iloc[:0]
    with pytest.raises(ValueError, match="census payload count mismatch"):
        G.validate_donor(write(tmp_path / "bad.root", frames))


def test_all_payload_trees_empty_but_census_present(tmp_path):
    frames = payload()
    frames["events"] = frames["events"].iloc[[2]]
    for tree in ("gen", "particles", "steps"):
        frames[tree] = frames[tree].iloc[:0]
    report = G.validate_donor(write(tmp_path / "empty.root", frames))
    assert report["counts"] == dict(events=1, gen=0, particles=0, steps=0)


def test_missing_tree_or_legacy_schema_refused(tmp_path):
    frames = payload()
    frames.pop("events")
    with pytest.raises(ValueError, match="requires schema v2"):
        G.validate_donor(write(tmp_path / "old.root", frames))
    frames = payload()
    frames["particles"] = frames["particles"].drop(columns="gen_product_id")
    with pytest.raises(ValueError, match="missing schema-v2 branches"):
        G.validate_donor(write(tmp_path / "old.root", frames))


@pytest.mark.parametrize("tree,column", [("events", "event"), ("particles", "key"), ("steps", "sp")])
def test_floating_identity_rejected(tmp_path, tree, column):
    frames = payload()
    frames[tree][column] = frames[tree][column].astype(float)
    with pytest.raises(ValueError, match="must be integer"):
        G.validate_donor(write(tmp_path / "float.root", frames))


def test_truncated_nonprimary_root_not_accepted_even_with_gen_pointer(tmp_path):
    frames = payload()
    frames["particles"].loc[0, "isprim"] = 0
    with pytest.raises(ValueError, match="not true isprim"):
        G.validate_donor(write(tmp_path / "bad.root", frames))


def test_primary_needs_generator_pointer(tmp_path):
    frames = payload()
    frames["particles"].loc[0, ["genidx", "gen_product_id"]] = [-1, 0]
    with pytest.raises(ValueError, match="primary: missing generator pointer"):
        G.validate_donor(write(tmp_path / "bad.root", frames))


def test_null_secondary_gen_pointer_allowed(tmp_path):
    frames = payload()
    frames["particles"].loc[1, ["genidx", "gen_product_id"]] = [-1, 0]
    G.validate_donor(write(tmp_path / "good.root", frames))


@pytest.mark.parametrize("self_cycle", [False, True])
def test_cycles_rejected_including_unused_particles(tmp_path, self_cycle):
    frames = payload()
    parts = frames["particles"]
    parts.loc[0, "isprim"] = 0
    parts.loc[0, "parent"] = parts.key.iloc[0 if self_cycle else 1]
    parts.loc[0, "parent_product_id"] = parts.sim_product_id.iloc[0]
    # No steps in the cyclic event: still reject its malformed ancestry.
    frames["steps"] = frames["steps"].iloc[[1]]
    frames["events"].loc[0, "n_steps"] = 0
    with pytest.raises(ValueError, match="ancestry cycle"):
        G.validate_donor(write(tmp_path / "cycle.root", frames))


def test_highest_cycle_is_selected_and_recorded(tmp_path):
    path = write(tmp_path / "cycles.root", payload())
    gen = payload()["gen"]
    write(path, {"gen": gen}, update=True)
    assert G.validate_donor(path)["trees"]["gen"] == dict(rows=2, cycles=[1, 2], selected_cycle=2)
    gen.loc[0, "pdg"] = 13
    write(path, {"gen": gen}, update=True)
    with pytest.raises(ValueError, match="signed PDG"):
        G.validate_donor(path)


def test_external_role_and_legacy_label(tmp_path):
    donor = write(tmp_path / "donor.root", payload())
    with pytest.raises(ValueError, match="external generator denominator"):
        G.validate_generator_stream(donor)
    frames = full_gen(payload())
    # Independent full-generator ProductIDs need not match donor IDs.
    frames["events"]["gen_product_id"] = np.uint32(99)
    frames["gen"]["gen_product_id"] = np.uint32(99)
    full = write(tmp_path / "full.root", frames)
    assert G.validate_generator_stream(full)["stream_role"] == "full generator stream"
    with pytest.raises(ValueError, match="associations_checked"):
        G.validate_donor(full)
    legacy = write(tmp_path / "legacy.root", {"gen": frames["gen"].drop(columns="gen_product_id")})
    report = G.validate_generator_stream(legacy)
    assert report["stream_role"] == "unverified legacy stream role" and report["schema_version"] is None
    partial = write(tmp_path / "partial.root", {"gen": frames["gen"]})
    with pytest.raises(ValueError, match="missing events census"):
        G.validate_generator_stream(partial)


def test_reach_normalization_includes_zero_generator_events(tmp_path):
    frames = full_gen(payload())
    frames["gen"]["pdg"] = np.int32(22)
    path = write(tmp_path / "full.root", frames)
    assert len(G.generator_event_keys(path)) == 3
    shower = pd.DataFrame({"apdg": [22], "kebin": [tier1b._ke_bin(np.array([200.]))[0]]})
    reach, count = tier1b._reach(shower, [str(path)], generated=300)
    assert count == 2 and reach.ngen.iloc[0] == 200


def test_external_census_count_required(tmp_path):
    frames = full_gen(payload())
    frames["events"].loc[2, "n_gen"] = 1
    with pytest.raises(ValueError, match="census payload count mismatch"):
        G.validate_generator_stream(write(tmp_path / "partial.root", frames))


def test_build_and_direct_build_one_fail_before_processing(tmp_path, monkeypatch):
    frames = payload()
    frames["particles"].loc[0, "pdg"] = 13
    path = write(tmp_path / "bad.root", frames)
    def forbidden(*args, **kwargs):
        pytest.fail("invalid donor reached bin edges or library reads")
    monkeypatch.setattr(tier1b, "_bin_edges", forbidden)
    monkeypatch.setattr(tier1b.tables, "read_many", forbidden)
    for operation in (lambda: tier1b.build([path], [], None),
                      lambda: tier1b._build_one([path], None, None)):
        with pytest.raises(ValueError, match="signed PDG"):
            operation()


def test_build_checks_each_unique_donor_once_and_records_roles(tmp_path, monkeypatch):
    path = write(tmp_path / "good.root", payload())
    legacy = write(tmp_path / "legacy.root", {"gen": payload()["gen"].drop(columns="gen_product_id")})
    original, checked = G.validate_donor, []
    def validate(p):
        checked.append(p)
        return original(p)
    monkeypatch.setattr(G, "validate_donor", validate)
    monkeypatch.setattr(tier1b, "_bin_edges", lambda *a: pd.DataFrame())
    def one(paths, geom, bins, *, _already_validated):
        assert str(path.resolve()) in _already_validated
        return {"libxing": pd.DataFrame({"xid": [0]}), "libdesc": pd.DataFrame({"xid": [0]}),
                "libshower": pd.DataFrame({"shid": [0]}), "libshowersteps": pd.DataFrame({"shid": [0]})}, dict(
                    crossings=1, desc_steps=1, desc_steps_unattached=0, showers=1, shower_steps=1, max_shower_sp=1)
    monkeypatch.setattr(tier1b, "_build_one", one)
    monkeypatch.setattr(tier1b, "_reach", lambda *a: (pd.DataFrame(), 0))
    lib, info = tier1b.build([path, path], [legacy], None)
    assert checked == [str(path.resolve())]
    assert info["files"] == 2 and info["showers"] == 2
    assert lib["libshower"].shid.tolist() == [0, 1]
    assert len(info["generator_associations"]) == 1
    assert info["generator_streams"][0]["stream_role"] == "unverified legacy stream role"
    with pytest.raises(ValueError, match="external generator denominator"):
        tier1b.build([path], [path], None)


def test_full_generator_without_particle_or_step_trees(tmp_path):
    frames = full_gen(payload())
    frames.pop("particles")
    frames.pop("steps")
    report = G.validate_generator_stream(write(tmp_path / "full.root", frames))
    assert report["counts"] == dict(events=3, gen=2)


def test_empty_census_cannot_establish_stream_role(tmp_path):
    frames = {name: frame.iloc[:0] for name, frame in payload().items()}
    with pytest.raises(ValueError, match="empty census"):
        G.validate_donor(write(tmp_path / "empty.root", frames))


@pytest.mark.parametrize("column,dtype", [("n_gen", np.int32), ("schema_version", np.int32),
                                          ("gen_product_id", np.uint64)])
def test_exact_census_schema_types(tmp_path, column, dtype):
    frames = payload()
    frames["events"][column] = frames["events"][column].astype(dtype)
    with pytest.raises(ValueError, match="requires (u?int)(32|64)"):
        G.validate_donor(write(tmp_path / "wrong_type.root", frames))


def test_vectorized_long_ancestry_and_three_cycle(tmp_path):
    frames = payload()
    n = 257
    # One full event and a zero-payload event, with a chain longer than eight levels.
    frames["events"] = frames["events"].iloc[[0, 2]].reset_index(drop=True)
    frames["events"].loc[0, "n_particles"] = n
    frames["gen"] = frames["gen"].iloc[[0]]
    parts = frames["particles"].iloc[np.zeros(n, dtype=int)].reset_index(drop=True)
    parts["key"] = 2**60 + np.arange(n, dtype=np.int64)
    parts["parent"] = np.r_[np.int64(-1), parts.key.to_numpy()[:-1]]
    parts["parent_product_id"] = np.r_[np.uint32(0), parts.sim_product_id.to_numpy()[1:]]
    parts["isprim"] = np.r_[np.int32(1), np.zeros(n - 1, dtype=np.int32)]
    frames["particles"] = parts
    frames["steps"] = frames["steps"].iloc[[0]].copy()
    frames["steps"]["sp"] = np.int64(parts.key.iloc[-1])
    report = G.validate_donor(write(tmp_path / "deep.root", frames))
    assert report["ancestry_passes"] <= n.bit_length()
    # A non-power-of-two cycle must not evade the pointer-doubling check.
    parts.loc[0, "isprim"] = 0
    parts.loc[0, "parent"] = parts.key.iloc[2]
    parts.loc[0, "parent_product_id"] = parts.sim_product_id.iloc[0]
    with pytest.raises(ValueError, match="ancestry cycle"):
        G.validate_donor(write(tmp_path / "cycle.root", frames))


def test_full_event_tuple_not_packed_or_event_only(tmp_path):
    frames = payload()
    # Same particle key/index/event, distinguish only subrun, with both ProductIDs equal.
    for name, frame in frames.items():
        second = frame.run == np.uint32(2**31 + 3)
        frame.loc[second, "run"] = np.uint32(2**32 - 1)
        frame.loc[second, "subrun"] = np.uint32(1)
        if "gen_product_id" in frame:
            frame.loc[second, "gen_product_id"] = np.uint32(2**32 - 2)
        if "sim_product_id" in frame:
            frame.loc[second, "sim_product_id"] = np.uint32(2**32 - 3)
        if "sp_product_id" in frame:
            frame.loc[second, "sp_product_id"] = np.uint32(2**32 - 3)
        if "pdg" in frame:
            frame.loc[second, "pdg"] = 22
    G.validate_donor(write(tmp_path / "tuples.root", frames))
    frames["particles"].loc[2, "pdg"] = -13
    with pytest.raises(ValueError, match="signed PDG"):
        G.validate_donor(write(tmp_path / "bad.root", frames))
