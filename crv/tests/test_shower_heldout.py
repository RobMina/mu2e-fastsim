"""Held-out scalar features, native expectation and legacy census guards."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import uproot

from valhist import shower_heldout as H
from valhist.generator_bank_candidate import count_file
from valhist.heldout_certificate import contiguous_census


def test_means_equal_enumerated_template_law_including_no_shower():
    x = sparse.csr_matrix([[1., 2.], [3., 5.], [7., 11.]])
    m = H.means(x, np.array([0, 0, 1]), np.array([4, 2]), 2)
    np.testing.assert_array_equal(m, [[1., 1.75], [3.5, 5.5]])


def test_scalar_features_keep_nonmuon_origin_and_empty_events():
    census = pd.DataFrame({"run": [1400] * 3, "subrun": [21] * 3, "event": [1, 2, 3]})
    parts = pd.DataFrame({"run": [1400] * 4, "subrun": [21] * 4, "event": [1, 1, 2, 2],
                          "key": [1, 2, 1, 2], "parent": [-1, 1, -1, 1], "pdg": [13, 11, 22, 11]})
    steps = parts.rename(columns={"key": "sp"})[["run", "subrun", "event", "sp"]].assign(bar=[0, 1, 0, 1], edep=[9., 8., 2., 3.])
    geom = SimpleNamespace(sector=np.array([0, 1]))
    f = H.scalar_features(steps, parts, census, geom).toarray()
    np.testing.assert_array_equal(f[1], [1, 2, 1, 3, 0, 0, 0, 0])
    assert not f[[0, 2]].any()


def test_contiguous_population_is_not_just_event_count():
    g = pd.DataFrame({"run": [1400] * 3, "subrun": [21] * 3, "event": [1, 2, 3], "idx": [0] * 3})
    assert len(contiguous_census(g, 21, 3)) == 3
    with pytest.raises(ValueError, match="incomplete"):
        contiguous_census(g.assign(event=[1, 2, 4]), 21, 3)
    with pytest.raises(ValueError, match="duplicate"):
        contiguous_census(pd.concat([g, g.iloc[:1]]), 21, 3)


def test_candidate_bank_uses_highest_cycle_and_does_not_infer_exposure_from_subrun_primaries(tmp_path):
    path = tmp_path / "gen.root"
    g = {"run": np.array([1400] * 3, dtype=np.uint32), "subrun": np.array([2001] * 3, dtype=np.uint32),
         "event": np.array([1, 2, 3], dtype=np.uint32), "idx": np.zeros(3, dtype=np.int32),
         "pdg": np.array([13, 22, 22]), "e": np.array([2000., 100., 100.]), "t": np.zeros(3)}
    with uproot.recreate(path) as root:
        root.mktree("dump/gen", {k: v[:1] for k, v in g.items()})
        root.mktree("dump/subruns", {"primaries": np.array([1])})
    with uproot.update(path) as root:
        root.mktree("dump/gen", g)
    counts, report = count_file(path, pd.DataFrame({"apdg": [22], "kebin": [H.U.tier1b._ke_bin(np.array([100.]))[0]]}), 3)
    assert report["selected_cycle"] == 2 and report["events_inferred"] == 3
    assert report["primaries_subruns"] == 1 and counts[0] == 2
