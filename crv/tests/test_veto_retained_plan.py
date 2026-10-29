"""Whole-frame workload measurements and batch admission without optical generation."""
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import frame_registry as F, retained_plan as P
from crvfastsim.response import photons
from test_veto_keyed_optical_smoke import sample


class Tree:
    def __init__(self, steps, size):
        self.steps, self.size, self.num_entries = steps, size, len(steps)

    def iterate(self, names, **kwargs):
        for first in range(0, len(self.steps), self.size):
            yield {name: self.steps[name].iloc[first:first+self.size].to_numpy() for name in names}


def inputs():
    st, ev, geom, cfg, tables, dev, status = sample()
    er = pd.MultiIndex.from_frame(ev[list(P.IDS)]).get_indexer(pd.MultiIndex.from_frame(st[list(P.IDS)]))
    ledger = st[[*P.IDS, "bar", "t0", "t1", "edep"]].assign(step_row=np.arange(len(st)), event_row=er)
    return st, ev, ledger, geom, cfg, tables, dev, status


@pytest.mark.parametrize("chunk", [1, 5, 1000])
def test_exact_full_frame_work_and_basket_invariance(chunk):
    st, ev, ledger, geom, cfg, tables, dev, status = inputs()
    actual = P.workload(Tree(st, chunk), ev, ledger, geom, cfg, status)
    base = P.workload(Tree(st, 1000), ev, ledger, geom, cfg, status)
    pd.testing.assert_frame_equal(actual, base, check_exact=True)
    rows = list(photons.iter_rows(st, st.pdg.to_numpy(), geom, cfg, tables, dev, np.random.default_rng(7)))
    expected = sum(int(r["nsub"].sum()) for _, r in rows)
    assert actual.simulated_substeps.sum() == expected
    assert actual.steps.tolist() == [4, 4, 4, 0]
    assert actual.live_pairs.tolist() == [4]*4
    assert actual.optical_support_pairs.tolist() == [4, 4, 4, 0]
    assert actual.source_first.tolist() == [0, 4, 8, -1]
    assert actual.source_stop.tolist() == [4, 8, 12, -1]
    assert actual.contiguous.all()


@pytest.mark.parametrize("reflector,channels", [(-1, [1, 3]), (1, [0, 2])])
def test_reflector_output_support_and_nan_skip(reflector, channels):
    st, ev, ledger, geom, cfg, tables, dev, status = inputs()
    cfg.sectors["EX"].reflector = reflector
    st.loc[0, "t0"] = np.nan
    ledger.loc[0, "t0"] = np.nan
    a = P.workload(Tree(st, 3), ev, ledger, geom, cfg, status)
    assert a.finite_steps.tolist() == [3, 4, 4, 0]
    assert a.optical_support_pairs.tolist() == [2, 2, 2, 0]
    status[channels] = 1
    b = P.workload(Tree(st, 3), ev, ledger, geom, cfg, status)
    assert not b.optical_support_pairs.any()
    assert b.live_pairs.tolist() == [2]*4


def frames():
    st, ev, ledger, geom, cfg, tables, dev, status = inputs()
    frame = P.workload(Tree(st, 3), ev, ledger, geom, cfg, status)
    frame["input_sha256"] = "a"*64
    other = frame.copy()
    other["input_sha256"] = "b"*64
    combined = pd.concat([frame, other], ignore_index=True)
    registry = F.freeze(combined)
    return combined.merge(registry[[*F.FRAME_KEYS, "frame_id"]], on=list(F.FRAME_KEYS)), registry


def test_batch_plan_exact_once_all_frames_and_partition_independent_registry():
    f, r = frames()
    limits = dict(frames=2, steps=8, simulated_substeps=64, live_pairs=8)
    assigned, batches = P.batch_plan(f.iloc[::-1], limits)
    assert len(assigned) == 8 and assigned.frame_id.tolist() == list(range(8))
    assert all((batches[k] <= v).all() for k, v in limits.items())
    assert not assigned.frame_id.duplicated().any()
    assert (assigned.groupby("batch").input_sha256.nunique() == 1).all()
    assert batches.frames.sum() == 8 and batches.steps.sum() == 24
    assert batches.range_has_no_extra_rows.all()
    _, one = P.batch_plan(f, dict(limits, frames=1))
    assert len(one) == 8
    assert F.digest(F.freeze(assigned)) == F.digest(r)


def test_empty_only_batch_has_no_source_range_but_live_thermal_domain():
    f, _ = frames()
    f = f[f.steps == 0]
    _, batches = P.batch_plan(f, dict(frames=1, steps=1, simulated_substeps=1, live_pairs=8))
    assert (batches.source_first == -1).all() and (batches.source_stop == -1).all()
    assert (batches.ranged_rows == 0).all() and (batches.live_pairs == 4).all()


def test_noncontiguous_sources_reported_not_assumed():
    st, ev, ledger, geom, cfg, tables, dev, status = inputs()
    st = st.iloc[np.r_[0, 4, 1, 5, 2, 6, 3, 7, 8, 9, 10, 11]].reset_index(drop=True)
    ledger = st[[*P.IDS, "bar", "t0", "t1", "edep"]].assign(step_row=np.arange(len(st)), event_row=st.event-1)
    a = P.workload(Tree(st, 3), ev, ledger, geom, cfg, status)
    assert a.contiguous.tolist() == [False, False, True, True]
    a["input_sha256"] = "a"*64
    a["frame_id"] = np.arange(4)
    _, b = P.batch_plan(a, dict(frames=1, steps=8, simulated_substeps=64, live_pairs=8))
    assert b.range_has_no_extra_rows.tolist() == [False, False, True, True]


@pytest.mark.parametrize("limits", [dict(frames=0, steps=8, simulated_substeps=64, live_pairs=8),
                                    dict(frames=2, steps=2, simulated_substeps=64, live_pairs=8),
                                    dict(frames=2, steps=8, simulated_substeps=1, live_pairs=8),
                                    dict(frames=2, steps=8, simulated_substeps=64, live_pairs=2),
                                    dict(frames=2, steps=8, simulated_substeps=2**52, live_pairs=8)])
def test_invalid_or_individually_oversized_frames_fail(limits):
    f, _ = frames()
    with pytest.raises(ValueError):
        P.batch_plan(f, limits)


def test_total_work_limit_checked_across_baskets(monkeypatch):
    st, ev, ledger, geom, cfg, tables, dev, status = inputs()
    monkeypatch.setattr(P, "MAX_TOTAL_SUBSTEPS", 20)
    with pytest.raises(ValueError, match="total substep"):
        P.workload(Tree(st, 1), ev, ledger, geom, cfg, status)
