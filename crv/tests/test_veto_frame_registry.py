"""Frozen full source-qualified registry labels and keyed helper seed contract."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import frame_registry as F


def events():
    return pd.DataFrame({"input_sha256": ["a"*64, "a"*64, "b"*64, "a"*64],
                         "run": [1, 2, 1, 1], "subrun": [0, 0, 0, 1], "event": [7]*4,
                         "pbtime": [-224.46]*4, "intensity": [1e7]*4})


def test_full_identity_domains_canonical_and_bound_seed():
    ev = events()
    registry = F.freeze(ev)
    pd.testing.assert_frame_equal(registry, F.freeze(ev.iloc[::-1]))
    assert registry.frame_id.tolist() == [0, 1, 2, 3]
    assert F.digest(registry) == F.digest(F.freeze(ev.iloc[::-1]))
    assert F.helper_seed(registry, 7) != F.helper_seed(registry, 8)
    assert F.helper_seed(registry, 7) != F.helper_seed(F.freeze(ev.iloc[:2]), 7)
    assert F.manifest(registry, [7, 8])["frames"] == 4


def test_subset_partition_round_trip_preserves_original_ids():
    ev = events().assign(payload=[3., 5., 7., 9.])
    r = F.freeze(ev)
    encoded = F.encode(ev, r)
    assert encoded.event.nunique() == 4
    encoded_parts = pd.concat([F.encode(ev.iloc[:2], r), F.encode(ev.iloc[2:], r)])
    pd.testing.assert_frame_equal(encoded, encoded_parts)
    restored = F.restore(encoded, r)
    pd.testing.assert_frame_equal(restored[list(ev)].reset_index(drop=True), ev.reset_index(drop=True))
    np.testing.assert_array_equal(F.encode(ev.iloc[::-1], r).event, encoded.event.iloc[::-1])
    empty = F.restore(F.encode(ev.iloc[:0], r), r)
    assert all(empty[k].dtype == np.int64 for k in (*F.IDS, "frame_id"))


@pytest.mark.parametrize("case", ["duplicate", "float", "negative", "huge", "nan", "digest"])
def test_malformed_registry_inputs(case):
    ev = events()
    if case == "duplicate":
        ev = pd.concat([ev, ev.iloc[:1]])
    elif case == "float":
        ev["event"] = ev.event.astype(float)
    elif case == "negative":
        ev["run"] = -1
    elif case == "huge":
        ev["run"] = 2**32
    elif case == "nan":
        ev["pbtime"] = np.nan
    else:
        ev["input_sha256"] = "x"*64
    with pytest.raises(ValueError):
        F.freeze(ev)


def test_changed_valid_registry_cannot_share_digest():
    r = F.freeze(events())
    changed = r.copy()
    changed["intensity"] *= 2
    assert F.digest(changed) != F.digest(r)
    for key in ("frame_id",):
        bad = r.copy()
        bad[key] = bad[key].iloc[::-1].to_numpy()
        with pytest.raises(AssertionError):
            F.validate(bad)


def test_encoding_outside_registry_or_changed_exposure_fails():
    ev = events()
    r = F.freeze(ev)
    for bad in (ev.assign(run=99), ev.assign(pbtime=-10.), ev.assign(intensity=1.)):
        with pytest.raises(ValueError):
            F.encode(bad, r)
    for bad in (pd.DataFrame({"event": [-1]}), pd.DataFrame({"event": [4]}), pd.DataFrame({"event": [1.5]})):
        with pytest.raises(ValueError):
            F.restore(bad, r)


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [True], [2**64]])
def test_invalid_seed_schedule(seeds):
    with pytest.raises(ValueError):
        F.manifest(F.freeze(events()), seeds)
