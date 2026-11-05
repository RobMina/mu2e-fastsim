"""Exact ancestry follows persistent identities, never time/distance or key alone."""
import numpy as np
import pytest

from calofastsim.ancestry import assign, forest


def particles():
    return {
        "run": np.ones(6, dtype=np.int64), "subrun": np.zeros(6, dtype=np.int64),
        "event": np.array([1, 1, 1, 1, 2, 2]),
        "sim_product_id": np.array([7, 8, 8, 7, 7, 8]), "sim_key": np.array([1, 1, 2, 2, 1, 1]),
        "parent_valid": np.array([False, True, True, True, False, True]),
        "parent_product_id": np.array([0, 7, 8, 7, 0, 7]), "parent_key": np.array([0, 1, 1, 1, 0, 1]),
    }


def incidents():
    return {"run": np.array([1, 1]), "subrun": np.array([0, 0]), "event": np.array([2, 1]),
            "product_id": np.array([7, 8]), "particle_key": np.array([1, 1])}


def refs(p, indices):
    return {name: p[name][indices] for name in ("run", "subrun", "event", "sim_product_id", "sim_key")}


def test_cross_product_and_event_identity_with_designated_internal_root():
    p = particles()
    assert np.array_equal(forest(p), [0, 0, 1, 0, 4, 4])
    assert np.array_equal(assign(p, incidents(), refs(p, [2, 1, 5, 4, 2])), [1, 1, 0, 0, 1])
    with pytest.raises(ValueError, match="not a descendant"):
        assign(p, incidents(), refs(p, [3]))


def test_permuted_particle_order_and_unsigned_mixed_identity_storage():
    p = {name: value[[5, 2, 3, 0, 4, 1]] for name, value in particles().items()}
    r = refs(p, [0, 1, 5])
    p["sim_product_id"] = p["sim_product_id"].astype(np.uint32)
    p["sim_key"] = p["sim_key"].astype(np.uint64)
    assert np.array_equal(assign(p, incidents(), r), [0, 1, 1])


@pytest.mark.parametrize("cycle", [[0], [0, 1], [0, 1, 2]])
def test_all_cycles_refused_including_unreferenced_components(cycle):
    p = particles()
    nxt = np.roll(cycle, -1)
    p["parent_valid"][cycle] = True
    p["parent_product_id"][cycle] = p["sim_product_id"][nxt]
    p["parent_key"][cycle] = p["sim_key"][nxt]
    with pytest.raises(ValueError, match="cyclic"):
        assign(p, incidents(), refs(p, [5]))


@pytest.mark.parametrize("change", ["missing_parent", "cross_event_parent", "duplicate", "zero_product", "float_key", "huge_key", "bad_flag", "short_flag", "short_parent"])
def test_bad_forest_refused(change):
    p = particles()
    if change == "missing_parent":
        p["parent_key"][1] = 99
    elif change == "cross_event_parent":
        p["parent_key"][5] = 2
    elif change == "duplicate":
        p["sim_key"][2] = 1
    elif change == "zero_product":
        p["sim_product_id"][0] = 0
    elif change == "float_key":
        p["sim_key"] = p["sim_key"].astype(float)
    elif change == "huge_key":
        p["sim_key"] = p["sim_key"].astype(np.uint64)
        p["sim_key"][0] = 2**63
    elif change == "bad_flag":
        p["parent_valid"] = p["parent_valid"].astype(int)
    elif change == "short_flag":
        p["parent_valid"] = p["parent_valid"][:-1]
    else:
        p["parent_key"] = p["parent_key"][:-1]
    with pytest.raises(ValueError):
        forest(p)


def test_empty_and_missing_incidents_references():
    p = particles()
    empty_incidents = {name: value[:0] for name, value in incidents().items()}
    assert assign(p, incidents(), refs(p, [])).size == 0
    assert assign(p, empty_incidents, refs(p, [])).size == 0
    assert forest({name: value[:0] for name, value in p.items()}).size == 0
    with pytest.raises(ValueError, match="not a descendant"):
        assign(p, empty_incidents, refs(p, [0]))
    bad = refs(p, [2])
    bad["sim_key"][0] = 99
    with pytest.raises(ValueError, match="missing from particles"):
        assign(p, incidents(), bad)
    missing = incidents()
    missing["particle_key"][0] = 99
    with pytest.raises(ValueError, match="missing from particles"):
        assign(p, missing, refs(p, []))
    duplicate = incidents()
    duplicate["event"][0] = 1
    with pytest.raises(ValueError, match="one designated"):
        assign(p, duplicate, refs(p, []))


def test_long_chain_pointer_doubling_and_zero_response_incident():
    n = 1025
    p = {"run": np.ones(n, dtype=int), "subrun": np.zeros(n, dtype=int),
         "event": np.ones(n, dtype=int), "sim_product_id": np.full(n, 7),
         "sim_key": np.arange(n), "parent_valid": np.arange(n) != 0,
         "parent_product_id": np.full(n, 7), "parent_key": np.maximum(np.arange(n) - 1, 0)}
    roots = {"run": np.array([1]), "subrun": np.array([0]), "event": np.array([1]),
             "product_id": np.array([7]), "particle_key": np.array([700])}
    assert np.array_equal(assign(p, roots, refs(p, [700, 1024])), [0, 0])
    assert assign(p, roots, refs(p, [])).size == 0
    with pytest.raises(ValueError, match="not a descendant"):
        assign(p, roots, refs(p, [699]))
