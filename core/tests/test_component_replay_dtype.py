"""Component replay in float32 and float64 (docs/component-response.md): float64 (the default) is bit-identical to a
frozen copy of the replay before the dtype argument; float32 gives the same trials, particle tree, track IDs and
refusals, and real outputs that differ from float64 only by float32 rounding of the transform and the time shift."""
import numpy as np
import pytest

from mu2efastsim import component_response as CR
from test_component_response import FixedRandom, bank, component, data, incoming

EPS32 = float(np.finfo(np.float32).eps)
TINY32 = float(np.finfo(np.float32).tiny)     # underflow: a float64 value below it may become 0 in float32
POINTS, VECTORS, TIMES, VALUES = ("x0", "x1", "position"), ("p0", "p1", "momentum"), ("t0", "t1", "time"), \
    ("length", "edep", "energy")


def _legacy_replay(library, placement, incoming, interaction, rng, *, next_track):
    """Frozen copy of replay() before the dtype argument (float64 only)."""
    incoming = CR._incoming(incoming)
    n = len(incoming["track"])
    interaction = CR._integer(interaction, "interaction", (n,))
    cursor = int(CR._integer(next_track, "next_track", ()))
    if placement.component != library.component:
        raise ValueError("incompatible component geometry/material/physics")
    if cursor <= max(incoming["track"].max(initial=0), incoming["parent"].max(initial=0)):
        raise ValueError("next_track must exceed input track/parent IDs and reserve unused IDs")
    key = np.column_stack((incoming["event"], interaction))
    if np.any(interaction < 0) or len(np.unique(key, axis=0)) != n:
        raise ValueError("interaction IDs must be nonnegative and unique within each event")
    local = {**incoming, "position": placement.local_points(incoming["position"]),
             "momentum": placement.local_vectors(incoming["momentum"])}
    valid = library.component.on_boundary(local["position"], local["momentum"], entering=True)
    if not valid.all():
        raise CR.UnsupportedEntrance(np.flatnonzero(~valid))
    trial = CR._choose(library, local, rng)
    owner, pidx, counts = CR._gather(library.particle_offsets, trial)
    starts = np.r_[0, np.cumsum(counts)[:-1]] if n else np.empty(0, dtype=np.int64)
    local_id = np.arange(len(pidx)) - starts[owner]
    isroot = local_id == 0
    children = int((~isroot).sum())
    track = np.empty(len(pidx), dtype=np.int64)
    track[isroot] = incoming["track"][owner[isroot]]
    track[~isroot] = cursor + np.arange(children)
    lp = library.particles["parent"][pidx]
    parent = track[starts[owner] + np.maximum(lp, 0)]
    parent[isroot] = incoming["parent"][owner[isroot]]

    def common(ii):
        return dict(event=incoming["event"][ii].copy(), interaction=interaction[ii].copy(),
                    instance=np.full(len(ii), placement.instance, dtype=np.int64))

    particles = dict(**common(owner), track=track, parent=parent,
                     pdg=library.particles["pdg"][pidx].copy(), exited=library.particles["exited"][pidx].copy())
    outputs = []
    for table, offsets, isstep in ((library.steps, library.step_offsets, True), (library.exits, library.exit_offsets, False)):
        ii, idx, _ = CR._gather(offsets, trial)
        pi = starts[ii] + table["particle"][idx]
        out = dict(**common(ii), track=track[pi], parent=parent[pi], pdg=particles["pdg"][pi])
        if isstep:
            out["volume"] = placement.volume_ids[table["volume"][idx]]
            for name in ("x0", "x1"):
                out[name] = placement.global_points(table[name][idx])
            for name in ("p0", "p1"):
                out[name] = placement.global_vectors(table[name][idx])
            for name in ("t0", "t1"):
                out[name] = table[name][idx] + incoming["time"][ii]
            for name in ("length", "edep"):
                out[name] = table[name][idx].copy()
        else:
            out["position"] = placement.global_points(table["position"][idx])
            out["momentum"] = placement.global_vectors(table["momentum"][idx])
            out["time"] = table["time"][idx] + incoming["time"][ii]
            out["energy"] = table["energy"][idx].copy()
        outputs.append(out)
    return CR.Replay(particles, *outputs, trial, cursor + children)


def _rotation(a, b, c):
    def rx(t):
        return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])

    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return rz(a) @ rx(b) @ rz(c)


def _placement(angles=(0.3, 1.1, -0.7), translation=(-3904.123, 1187.5, 10631.25)):
    return CR.Placement(component(), 7, _rotation(*angles), np.asarray(translation), np.asarray((100, 101)))


PLACEMENTS = [((0., 0., 0.), (0., 0., 0.)), ((0.3, 1.1, -0.7), (-3904.123, 1187.5, 10631.25)),
              ((2.9, 0.4, 1.3), (5.5, -2.25, 1.0))]


def _bitwise(a, b):
    return set(a) == set(b) and all(np.asarray(a[k]).dtype == np.asarray(b[k]).dtype and
                                    np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)


def _run(p, n=400, seed=3, **kw):
    inc = incoming(p, n=n)
    inc["time"] = inc["time"] * 37.25 + 1500.
    return CR.replay(bank(), p, inc, np.arange(n), np.random.default_rng(seed), next_track=10_000, **kw), inc


@pytest.mark.parametrize("angles,translation", PLACEMENTS)
def test_float64_default_is_bit_identical_to_the_previous_replay(angles, translation):
    p = _placement(angles, translation)
    inc = incoming(p, n=400)
    old = _legacy_replay(bank(), p, inc, np.arange(400), np.random.default_rng(3), next_track=10_000)
    for kw in ({}, {"dtype": np.float64}, {"dtype": "float64"}):
        new = CR.replay(bank(), p, inc, np.arange(400), np.random.default_rng(3), next_track=10_000, **kw)
        for t in ("particles", "steps", "exits"):
            assert _bitwise(getattr(new, t), getattr(old, t)), (kw, t)
        assert new.trials.tobytes() == old.trials.tobytes() and new.next_track == old.next_track


def agreement(r64, r32, placement, incoming_time, library):
    """float32 against float64 per real column: the largest |f32 - f64| over its float32 rounding bound, and the
    number of elements over the bound. Bounds: transform 8 eps32 (|R| |x_local| + |t|); time shift 2 eps32 (|t_local|
    + |t_in|); each plus 4 tiny32 for underflow (a coordinate of 1e-40 mm becomes 0); recorded values: the float32
    cast of the float64 value, exactly."""
    out = {}
    absr, t = np.abs(placement.rotation), np.abs(placement.translation)
    for tname, table, offsets in (("steps", library.steps, library.step_offsets),
                                  ("exits", library.exits, library.exit_offsets)):
        ii, idx, _ = CR._gather(offsets, r64.trials)
        a64, a32 = getattr(r64, tname), getattr(r32, tname)
        for k, v in a64.items():
            if np.asarray(v).dtype.kind != "f":
                continue
            x64, x32 = np.asarray(v), np.asarray(a32[k])
            assert x32.dtype == np.float32, (tname, k)
            d = np.abs(x32.astype(np.float64) - x64)
            if k in POINTS:
                bound = 8 * EPS32 * (np.abs(table[k][idx]) @ absr.T + t) + 4 * TINY32
            elif k in VECTORS:
                bound = 8 * EPS32 * (np.abs(table[k][idx]) @ absr.T) + 4 * TINY32
            elif k in TIMES:
                bound = 2 * EPS32 * (np.abs(table[k][idx]) + np.abs(incoming_time[ii])) + 4 * TINY32
            else:
                bound = None
            under = int(np.count_nonzero((x32 == 0) & (x64 != 0)))
            if bound is None:
                over = int(np.count_nonzero(x32 != x64.astype(np.float32)))
                ratio = 0.0 if over == 0 else float("inf")
            else:
                safe = np.where(bound > 0, bound, np.inf)
                over = int(np.count_nonzero(d > bound))
                ratio = float((d / safe).max()) if d.size else 0.0
            rel = float((d / np.maximum(np.abs(x64), 1e-300)).max()) if d.size else 0.0
            out[f"{tname}.{k}"] = dict(n=int(d.size), max_abs=float(d.max()) if d.size else 0.0, max_rel=rel,
                                       max_over_bound=ratio, over_bound=over, underflow_to_zero=under)
    return out


@pytest.mark.parametrize("angles,translation", PLACEMENTS)
def test_float32_same_trials_and_identities_and_rounding_only(angles, translation):
    p = _placement(angles, translation)
    r64, inc = _run(p)
    r32, _ = _run(p, dtype=np.float32)
    assert r32.trials.tobytes() == r64.trials.tobytes() and r32.next_track == r64.next_track
    assert _bitwise(r32.particles, r64.particles)
    for t in ("steps", "exits"):
        a64, a32 = getattr(r64, t), getattr(r32, t)
        assert set(a64) == set(a32)
        for k, v in a64.items():
            if np.asarray(v).dtype.kind != "f":
                assert np.asarray(a32[k]).tobytes() == np.asarray(v).tobytes(), (t, k)
    rep = agreement(r64, r32, p, inc["time"], bank())
    assert rep and all(r["over_bound"] == 0 for r in rep.values()), rep
    e64 = np.bincount(r64.steps["interaction"], weights=r64.steps["edep"])
    e32 = np.bincount(r32.steps["interaction"], weights=r32.steps["edep"].astype(np.float64))
    assert np.allclose(e32, e64, rtol=EPS32, atol=0)


def test_float32_refuses_the_same_rows_before_drawing():
    p = _placement()
    inc = incoming(p, n=5)
    inc["energy"][2] = 11.
    inc["momentum"][2] *= 11. / np.linalg.norm(inc["momentum"][2])
    rows = []
    for dt in (np.float64, np.float32):
        rng = np.random.default_rng(1)
        with pytest.raises(CR.UnsupportedEntrance) as e:
            CR.replay(bank(), p, inc, np.arange(5), rng, next_track=100, dtype=dt)
        rows.append(e.value.rows.tolist())
        assert rng.random() == np.random.default_rng(1).random()          # no draw consumed
    assert rows[0] == rows[1] == [2]


def test_empty_batch_and_bad_dtype():
    p = _placement()
    r = CR.replay(bank(), p, incoming(p, n=0), np.empty(0, np.int64), np.random.default_rng(1), next_track=10,
                  dtype=np.float32)
    assert r.steps["edep"].dtype == np.float32 and len(r.steps["edep"]) == 0
    with pytest.raises(ValueError, match="dtype"):
        CR.replay(bank(), p, incoming(p), np.arange(1), np.random.default_rng(1), next_track=10, dtype=np.float16)


def test_bank_manifest_declares_both_dtypes():
    m = CR.bank_manifest_fields(bank())
    assert m["dtype"]["compute"] == ["float32", "float64"]


def test_float32_underflow_of_a_tiny_coordinate_is_within_the_bound():
    """A recorded coordinate below the float32 range (here 1e-50 mm) becomes 0 in float32 (identity placement):
    within the bound's underflow term, and counted."""
    args = data()
    args["steps"]["x1"][0, 0] = 1e-50                  # below the float32 subnormals
    lib = CR.Library(**args)
    p = CR.Placement(component(), 7, np.eye(3), np.zeros(3), np.asarray((100, 101)))
    inc = incoming(p)
    r64 = CR.replay(lib, p, inc, [0], FixedRandom(.1), next_track=10)
    r32 = CR.replay(lib, p, inc, [0], FixedRandom(.1), next_track=10, dtype=np.float32)
    rep = agreement(r64, r32, p, inc["time"], lib)
    assert rep["steps.x1"]["underflow_to_zero"] >= 1
    assert all(r["over_bound"] == 0 for r in rep.values()), rep
