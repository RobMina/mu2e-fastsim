"""Bounded two-draw streaming optics and global root admission, with frozen laws."""
from dataclasses import FrozenInstanceError, replace
import hashlib

import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto import indexed_replay as I, keyed_optical_smoke as M, keyed_optics as K
from crvfastsim.veto import streaming_optics as S
from crvfastsim.veto.replay import IDS
from crvfastsim.veto.replay_timing import DIGI_END, PERIOD, stock_step_mask, wrap_arrivals
from test_veto_keyed_optics import SHA, fixture


def domain(n=24):
    steps, _, geom, cfg, tables, deviation = fixture(n=n, decay=1.5)
    events = pd.DataFrame({"run": [20, 20, 21, 21], "subrun": [2, 3, 2, 3],
                           "event": [7, 7, 7, 7], "source_row": [1, 4, 7, 10],
                           "frame_id": [100, 201, 308, 412],
                           "pbtime": [-224.46, -219.46, -214.46, -209.46],
                           "intensity": [10., 20., 30., 40.], "input_sha256": SHA})
    er = np.arange(n) % 3
    for name in (*IDS, "frame_id", "input_sha256"):
        steps[name] = events[name].to_numpy()[er]
    steps["sp"] = np.ones(n, dtype=np.int64)
    steps["pdg"] = np.array([13, -13, 211], np.int64)[er]
    w = -events.pbtime.to_numpy().astype(np.float32).astype(float)[er]
    steps["t0"] = w + np.resize([10., 400., 1499., 1550., 1701., -100., 3500., 2.6081538852432572e16], n)
    steps["t1"] = steps.t0 + .125
    particles = events.iloc[:3][[*IDS, "frame_id", "input_sha256"]].copy()
    particles["source_row"] = np.array([9, 14, 31], np.int64)
    particles["key"], particles["parent"], particles["isprim"] = 1, -1, 1
    particles["pdg"] = np.array([13, -13, 211], np.int32)
    selection = I.Selection(events, steps, particles, "a" * 64, SHA, 1234)
    return selection, geom, cfg, tables, deviation, np.array([0, 1, 2, 4], np.int64)


def run(args, **kw):
    return S.make_batch(*args, input_sha256=kw.pop("input_sha256", SHA), seed=kw.pop("seed", 717), **kw)


def reference(args, seed=717):
    selection, geom, cfg, tables, deviation, status = args
    steps, events = selection.steps, selection.events
    er = pd.MultiIndex.from_frame(events[list(IDS)]).get_indexer(pd.MultiIndex.from_frame(steps[list(IDS)]))
    w = -events.pbtime.to_numpy().astype(np.float32).astype(float)[er]
    shift = np.floor((steps.t0.to_numpy() - w) / PERIOD) * PERIOD
    folded = steps.assign(t0=steps.t0 - w - shift, t1=steps.t1 - w - shift)
    raw, historical = [K.make_photons(st, st.pdg.to_numpy(), geom, cfg, tables, deviation,
                                     input_sha256=SHA, seed=seed) for st in (steps, folded)]
    return M.timing_table(raw, historical, steps, events, status, SHA, seed)


def subset(selection, frames):
    return replace(selection, **{name: getattr(selection, name).loc[
        getattr(selection, name).frame_id.isin(frames)].copy() for name in ("events", "steps", "particles")})


def equal(a, b, *, local=False):
    if local:
        a, b = (x.drop(columns="event_row") for x in (a, b))
    pd.testing.assert_frame_equal(M.canonical(a), M.canonical(b), check_exact=True)


def test_exact_frozen_timing_laws_and_preserved_census():
    args = domain()
    selection, *_, status = args
    originals = {name: getattr(selection, name).copy(deep=True) for name in ("events", "steps", "particles")}
    q = run(args)
    equal(q, reference(args))
    assert len(q) > 100 and set(q.photon_type) == {0, 1, 2}
    assert len(selection.events) == 4 and 412 not in selection.steps.frame_id.to_numpy()
    assert q.source_row.isin(selection.steps.source_row).all()
    for name, original in originals.items():
        pd.testing.assert_frame_equal(getattr(selection, name), original, check_exact=True)
    live = (status[q.chan.to_numpy()] & 5) == 0
    np.testing.assert_array_equal(q.live_channel, live)
    np.testing.assert_array_equal(q.time_raw_wrapped_absolute, wrap_arrivals(q.time_raw_absolute, q.origin))
    si = pd.Index(selection.steps.source_row).get_indexer(q.source_row)
    er = pd.MultiIndex.from_frame(selection.events[list(IDS)]).get_indexer(
        pd.MultiIndex.from_frame(selection.steps[list(IDS)]))
    w = -selection.events.pbtime.to_numpy().astype(np.float32).astype(float)[er]
    kept = stock_step_mask(selection.steps.t0, selection.steps.t1, w)[si]
    np.testing.assert_array_equal(q.stock_step_kept, kept)
    np.testing.assert_array_equal(q.historical_kept, live & (q.time_historical_relative <= DIGI_END))
    np.testing.assert_array_equal(q.arrival_only_kept, live & (q.time_raw_wrapped_absolute <= q.origin + DIGI_END))
    np.testing.assert_array_equal(q.rejection_only_kept, q.historical_kept & kept)
    np.testing.assert_array_equal(q.stock_timing_kept, q.arrival_only_kept & kept)
    # Histories survive dead status, step rejection, early times and the upper cut.
    assert (~q.live_channel).any() and (~q.stock_step_kept).any()
    assert (q.time_historical_relative > DIGI_END).any()
    assert (q.time_historical_relative < 200).any()
    assert not q[q.time_historical_relative < 200].historical_kept.eq(False).all()


def test_only_two_calls_no_smoke_and_explicit_limits(monkeypatch):
    args = domain()
    calls = []
    original = K.make_photons
    def record(steps, *rest, **kw):
        calls.append((steps.copy(), kw.copy()))
        return original(steps, *rest, **kw)
    monkeypatch.setattr(K, "make_photons", record)
    monkeypatch.setattr(M, "generate", lambda *a, **k: pytest.fail("must not rerun smoke"))
    limits = S.Limits(max_substeps=7)
    run(args, limits=limits)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(calls[0][0], args[0].steps)
    for steps, kw in calls:
        assert kw == {"input_sha256": SHA, "seed": 717, "max_substeps": 7,
                      "max_total_substeps": 2_000_000, "max_photons": 300_000}
        np.testing.assert_array_equal(steps.source_row, args[0].steps.source_row)
    e, st = args[0].events, args[0].steps
    er = pd.Index(e.frame_id).get_indexer(st.frame_id)
    w = -e.pbtime.to_numpy().astype(np.float32).astype(float)[er]
    shift = np.floor((st.t0.to_numpy() - w) / PERIOD) * PERIOD
    np.testing.assert_array_equal(calls[1][0].t0, st.t0 - w - shift)
    np.testing.assert_array_equal(calls[1][0].t1, st.t1 - w - shift)


def test_subset_partition_internal_chunk_and_seed_invariance():
    args = domain()
    selection, *rest = args
    q = run(args)
    equal(q, run(args, limits=S.Limits(max_substeps=3)))
    pieces = [run((subset(selection, frames), *rest)) for frames in ([308, 412], [201], [100])]
    equal(q, pd.concat(pieces, ignore_index=True), local=True)
    take = subset(selection, [100, 412])
    equal(q[q.source_row.isin(take.steps.source_row)], run((take, *rest)), local=True)
    other = run(args, seed=718)
    assert set(other.draw_seed) == {718}
    assert not q.drop(columns="draw_seed").equals(other.drop(columns="draw_seed"))
    assert len(selection.events) == 4


@pytest.mark.parametrize("column", ["run", "subrun", "event"])
def test_full_event_tuple_changes_draws(column):
    args = domain()
    selection, *rest = args
    changed = replace(selection, **{name: getattr(selection, name).assign(**{
        column: getattr(selection, name)[column] + 1}) for name in ("events", "steps", "particles")})
    a, b = run(args), run((changed, *rest))
    assert not a.drop(columns=[column, "photon_id"]).equals(b.drop(columns=[column, "photon_id"]))


@pytest.mark.parametrize("mode", ["quiet", "nan", "zero"])
def test_typed_zero_photons_complete_events(mode):
    selection, *rest = domain()
    if mode == "quiet":
        selection = subset(selection, [412])
    else:
        st = selection.steps.assign(t0=np.nan) if mode == "nan" else selection.steps.assign(edep=0., len=0.)
        selection = replace(selection, steps=st)
    q = run((selection, *rest))
    assert q.empty and len(selection.events) == (1 if mode == "quiet" else 4)
    assert q.draw_seed.dtype == np.uint64 and q.source_row.dtype == np.int64
    assert q.photon_id.dtype == np.int64 and q.historical_kept.dtype == bool
    equal(q, reference((selection, *rest)))
    union = S.RootUnion(max_roots=1)
    union.add(q)
    assert union.summary()["roots"] == 0


def test_partial_nan_skips_without_reranking():
    args = domain()
    selection, *rest = args
    st = selection.steps.copy()
    st.loc[::3, "t1"] = np.nan
    a = run(args)
    b = run((replace(selection, steps=st), *rest))
    equal(a[~a.source_row.isin(st.loc[::3, "source_row"])], b)


@pytest.mark.parametrize("name,ceiling", [("max_frames", 4), ("max_steps", 20_000),
    ("max_total_substeps", 2_000_000), ("max_substeps", 200_000), ("max_photons", 300_000)])
def test_immutable_strict_limits(name, ceiling):
    for bad in (0, -1, True, 1., np.int64(1), ceiling + 1):
        with pytest.raises(ValueError, match=name):
            S.Limits(**{name: bad})
    with pytest.raises(FrozenInstanceError):
        setattr(S.Limits(), name, 1)


@pytest.mark.parametrize("kw,match", [({"max_frames": 3}, "cap"), ({"max_steps": 1}, "cap"),
    ({"max_total_substeps": 1}, "total substep"), ({"max_photons": 1}, "photon")])
def test_limits_fail_not_truncate(kw, match):
    with pytest.raises(ValueError, match=match):
        run(domain(), limits=S.Limits(**kw))


def test_zero_yield_work_preflight(monkeypatch):
    selection, *rest = domain()
    st = selection.steps.assign(x1=1e6, edep=0., len=0.)
    monkeypatch.setattr(K, "_rows", lambda *a, **k: pytest.fail("must preflight work"))
    with pytest.raises(ValueError, match="total substep"):
        run((replace(selection, steps=st), *rest))


def test_pre_timing_cap_and_exact_raw_folded_keys(monkeypatch):
    args = domain()
    raw = reference(args)[list(K.KEYS)].assign(time=1.)
    monkeypatch.setattr(M, "timing_table", lambda *a, **k: pytest.fail("must fail before timing allocation"))
    monkeypatch.setattr(K, "make_photons", lambda *a, **k: raw)
    with pytest.raises(ValueError, match="before timing"):
        run(args, limits=S.Limits(max_photons=1))
    values = iter([raw, raw.assign(ordinal=raw.ordinal + 1)])
    monkeypatch.setattr(K, "make_photons", lambda *a, **k: next(values))
    with pytest.raises(ValueError, match="full photon keys"):
        run(args)


@pytest.mark.parametrize("table,column,value", [
    ("events", "run", 1.), ("events", "frame_id", -1), ("events", "pbtime", np.inf),
    ("events", "intensity", -1.), ("events", "source_row", 1.),
    ("steps", "source_row", 1.), ("particles", "source_row", 1.),
    ("steps", "input_sha256", "b" * 64), ("events", "input_sha256", "b" * 64),
    ("particles", "input_sha256", "b" * 64), ("steps", "frame_id", 999),
    ("particles", "frame_id", 999), ("steps", "subrun", 999),
    ("particles", "run", 999), ("steps", "pdg", 11), ("particles", "pdg", 11),
    ("steps", "sp", 2), ("particles", "parent", 0), ("particles", "isprim", 0),
    ("steps", "t0", np.inf), ("steps", "t1", -np.inf),
    ("steps", "source_row", np.uint64(2**63)), ("particles", "key", 0),
])
def test_identity_and_numeric_fail_closed(table, column, value, monkeypatch):
    selection, *rest = domain()
    selection = replace(selection, **{table: getattr(selection, table).assign(**{column: value})})
    monkeypatch.setattr(K, "make_photons", lambda *a, **k: pytest.fail("identity admission must precede optics"))
    with pytest.raises(ValueError):
        run((selection, *rest))


@pytest.mark.parametrize("table", ["events", "steps", "particles"])
@pytest.mark.parametrize("mode", ["duplicate_row", "reverse", "missing", "duplicate_column"])
def test_each_table_source_order_and_schema(table, mode):
    selection, *rest = domain()
    frame = getattr(selection, table).copy()
    if mode == "duplicate_row":
        frame.iloc[1, frame.columns.get_loc("source_row")] = frame.source_row.iloc[0]
    elif mode == "reverse":
        frame = frame.iloc[::-1]
    elif mode == "missing":
        frame = frame.drop(columns="source_row")
    else:
        frame = pd.concat([frame, frame[["source_row"]]], axis=1)
    with pytest.raises(ValueError):
        run((replace(selection, **{table: frame}), *rest))


@pytest.mark.parametrize("mode", ["events", "frame_id", "particles", "empty", "binding", "registry"])
def test_duplicate_census_particle_and_binding(mode):
    selection, *rest = domain()
    if mode == "events":
        events = selection.events.copy()
        events.loc[1, list(IDS)] = events.loc[0, list(IDS)].to_numpy()
        selection = replace(selection, events=events)
    elif mode == "frame_id":
        selection = replace(selection, events=selection.events.assign(frame_id=1))
    elif mode == "particles":
        part = selection.particles.copy()
        part.loc[1, [*IDS, "frame_id"]] = part.loc[0, [*IDS, "frame_id"]].to_numpy()
        selection = replace(selection, particles=part)
    elif mode == "empty":
        selection = subset(selection, [])
    elif mode == "binding":
        selection = replace(selection, input_sha256="c" * 64)
    else:
        selection = replace(selection, registry_sha256="bad")
    with pytest.raises(ValueError):
        run((selection, *rest))


@pytest.mark.parametrize("status", [np.zeros(3, int), np.zeros(4, float), np.full(4, -1)])
def test_status_coverage(status):
    args = list(domain())
    args[-1] = status
    with pytest.raises(ValueError, match="status"):
        run(args)


def test_roots_partition_seed_union_exact_digest_and_preallocation():
    args = domain()
    a, b = run(args), run(args, seed=718)
    joined = pd.concat([a, b], ignore_index=True)
    union = S.RootUnion(max_roots=len(joined))
    storage = union._roots
    assert storage.dtype == np.int64 and storage.shape == (len(joined),)
    for q in (a.iloc[:0], b.iloc[::2], a, b.iloc[1::2], b.iloc[:0]):
        union.add(q)
        assert union._roots is storage
    expected = hashlib.sha256(np.sort(M.roots(joined)).astype("<i8").tobytes()).hexdigest()
    assert union.summary() == {"roots": len(joined), "sha256": expected,
                               "collision_free": True, "contract": M.ROOT_CONTRACT}
    other = S.RootUnion(max_roots=len(joined))
    other.add(joined.iloc[::-1])
    assert other.summary() == union.summary()
    with pytest.raises(ValueError, match="cap"):
        union.add(a.iloc[:1])
    assert union.summary() == other.summary()


def test_roots_duplicates_collision_across_seeds_and_failed_add_atomic(monkeypatch):
    q = run(domain()).iloc[:1].copy()
    union = S.RootUnion(max_roots=10)
    union.add(q)
    before = union.summary()
    with pytest.raises(ValueError, match="across"):
        union.add(q)
    wrong = q.assign(photon_id=q.photon_id + 1)
    with pytest.raises(ValueError, match="disagreement"):
        union.add(wrong)
    with pytest.raises(ValueError, match="duplicate full photon key"):
        union.add(pd.concat([q, q], ignore_index=True))
    # Force a collision for distinct full keys and seeds, not just duplicate rows.
    changed = q.assign(draw_seed=np.uint64(718))
    monkeypatch.setattr(M, "roots", lambda x: np.full(len(x), q.photon_id.iloc[0], np.int64))
    with pytest.raises(ValueError, match="across batches/seeds"):
        union.add(changed)
    within = pd.concat([changed, changed.assign(ordinal=changed.ordinal + 1)], ignore_index=True)
    with pytest.raises(ValueError, match="within batch"):
        union.add(within)
    assert union.summary() == before


@pytest.mark.parametrize("value", [0, -1, True, 1., np.int64(1), 1_800_001])
def test_root_limit_strict(value):
    with pytest.raises(ValueError, match="max_roots"):
        S.RootUnion(max_roots=value)


def test_root_cap_before_compute(monkeypatch):
    q = run(domain()).iloc[:2]
    monkeypatch.setattr(M, "roots", lambda *a, **k: pytest.fail("cap before root computation"))
    union = S.RootUnion(max_roots=1)
    with pytest.raises(ValueError, match="cap"):
        union.add(q)
    assert union.summary()["sha256"] == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("column,value", [("draw_seed", 1.), ("draw_seed", -1),
    ("source_row", 1.), ("photon_id", -1), ("run", np.uint64(2**63)),
    ("input_sha256", None), ("input_sha256", "bad"), ("input_sha256", 123)])
def test_root_key_validation(column, value):
    q = run(domain()).iloc[:1].assign(**{column: value})
    with pytest.raises(ValueError):
        S.RootUnion(max_roots=1).add(q)


@pytest.mark.parametrize("kw", [{"seed": -1}, {"seed": 2**64}, {"seed": True},
    {"seed": 1.}, {"input_sha256": "A" * 64}, {"input_sha256": None}])
def test_bad_draw_domain_before_optics(kw, monkeypatch):
    monkeypatch.setattr(K, "make_photons", lambda *a, **k: pytest.fail("draw domain admission first"))
    with pytest.raises(ValueError):
        run(domain(), **kw)


def test_uint64_seed_and_large_source_ids_survive_union():
    selection, *rest = domain(n=6)
    changed = {}
    for name in ("events", "steps", "particles"):
        table = getattr(selection, name).copy()
        table["source_row"] = np.arange(len(table), dtype=np.uint64) + np.uint64(2**63 - len(table))
        changed[name] = table
    selection = replace(selection, **changed)
    q = run((selection, *rest), seed=2**64 - 1)
    assert len(q) > 0 and q.source_row.dtype == np.int64
    assert q.draw_seed.eq(np.uint64(2**64 - 1)).all()
    union = S.RootUnion(max_roots=len(q))
    union.add(q)
    assert union.summary()["roots"] == len(q)


def test_quiet_frame_float32_origin_still_validated():
    selection, *rest = domain()
    events = selection.events.copy()
    events.loc[events.frame_id == 412, "pbtime"] = 1e100
    with pytest.raises(ValueError, match="float32 bunch origin"):
        run((replace(selection, events=events), *rest))


def test_nullable_missing_input_digest_rejected():
    selection, *rest = domain()
    events = selection.events.copy()
    events["input_sha256"] = events.input_sha256.astype("string")
    events.loc[0, "input_sha256"] = pd.NA
    with pytest.raises(ValueError, match="source input"):
        run((replace(selection, events=events), *rest))
