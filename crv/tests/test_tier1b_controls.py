"""Factorial-control origin isolation and migration-budget contracts."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from valhist import descendant_migration_report as M
from valhist.tier1b_controls import showers


def test_shower_guard_does_not_compare_event_total_step_count():
    parts = pd.DataFrame({"run": [1400] * 4, "subrun": [1] * 4, "event": [1] * 4,
                          "key": [1, 2, 3, 4], "parent": [-1, 1, -1, 3], "pdg": [13, 11, 22, 11]})
    steps = pd.DataFrame({"run": [1400] * 4, "subrun": [1] * 4, "event": [1] * 4, "nstep": [4] * 4,
                          "sp": [1, 2, 3, 4], "bar": [0] * 4, "t0": [1., 2., 3., 4.], "edep": [1., 2., 3., 4.]})
    geom = SimpleNamespace(sector=np.zeros(1, dtype=int))
    got = showers(steps, parts, geom)
    assert list(got.sp) == [3, 4] and "nstep" not in got.columns
    pd.testing.assert_frame_equal(got, showers(steps.iloc[::-1].assign(nstep=20), parts, geom))


def test_migration_budget_separates_mc_error_from_physics():
    pre = np.tile([2., 4., 6., 8.], (32, 1)).T
    post = pre / 2
    replicas = {"reference": np.ones(32), "pre_samples": pre, "post_samples": post,
                "retained_post": post[0], "replicas": 4}
    expected = {"arms": {"context": {"sector_reference": [4.] * 8, "sector_mean": [20.] * 8,
                                      "sector_sampling_sd": [1.] * 8}}}
    placement = {"views": {"migration": [{"source_kind": "otherbar", "target_group": g,
                                          "target_kind": kind, "e_cand": value}
                                         for g in ("EX", "T1", "T2", "M")
                                         for kind, value in (("anchor", 1.), ("otherbar", 3.))]}}
    result = M.calculate(expected, replicas, placement)
    assert result["budget"][0]["post_mean_ratio"] == 2.5
    assert result["budget"][0]["migration_energy"] == -10
    assert result["budget"][0]["pre_replica_mean_minus_exact_over_mc_se"] == 0
    assert result["retained_placement"][0]["reclassified_fraction_of_energy_landing_in_sector"] == .25
    replicas["reference"] = np.full(32, 2.)
    with pytest.raises(ValueError, match="population differs"):
        M.calculate(expected, replicas, placement)
