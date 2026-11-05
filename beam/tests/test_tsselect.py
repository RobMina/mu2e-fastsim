"""Synthetic explicit-stream selector tests; no transport or external map assets."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from beamfastsim import tsfield, tsselect


MATRIX = np.array([[2.0, -3.0, 0.5], [-1.0, 4.0, 2.0], [0.25, 2.0, -5.0]])
OFFSET = np.array([0.5, -1.0, 2.5])


def affine(points):
    return np.asarray(points) @ MATRIX.T + OFFSET


def loaded(tmp_path, name, *, lo=(0, 0, 0), n=(2, 3, 3), d=(1, 0.5, 0.5),
           shift=(0, 0, 0), mirror=False, delta=(0, 0, 0)):
    """Standalone modern-map fixture with the same affine layout as tsfaces."""
    path = tmp_path / (name + ".header")
    lo, d, shift = np.asarray(lo), np.asarray(d), np.asarray(shift)
    origin = lo + shift
    text = "# Origin shift for Mu2e: " + " ".join(str(v) for v in shift) + "\n"
    text += (f"grid X0={origin[0]:.17g} Y0={origin[1]:.17g} Z0={origin[2]:.17g} "
             f"nX={n[0]} nY={n[1]} nZ={n[2]} "
             f"dX={d[0]:.17g} dY={d[1]:.17g} dZ={d[2]:.17g}\n")
    if mirror:
        text += "extendY flip=By\n"
    path.write_text(text + "data\n")
    nodes = np.stack(np.meshgrid(*(np.arange(count) for count in n), indexing="ij"), axis=-1)
    values = affine(lo + nodes * d) + np.asarray(delta)
    path.with_suffix(".bin").write_bytes(struct.pack("<I", tsfield.MARKER)
                                         + np.asarray(values, dtype="<f8").tobytes())
    return tsfield.load(path)


@pytest.fixture
def touching(tmp_path):
    left = loaded(tmp_path, "left", shift=(-3904, 7, 7929))
    right = loaded(tmp_path, "right", lo=(1, 0, 0), delta=(3, 4, 0))
    return left, right


def state(values):
    return np.asarray(values, dtype=np.int64)


def test_fresh_face_and_opposite_histories_need_not_agree(touching):
    left, right = touching
    selector = tsselect.MapSelector(inner=touching)
    points = np.array([[1, 0.5, 0.5], [1, 0.5, 0.5]])
    fresh, selected = selector.field(points, state([-1, -1]))
    np.testing.assert_array_equal(selected, [0, 0])
    np.testing.assert_allclose(fresh, affine(points))
    values, selected = selector.field(points, state([0, 1]))
    np.testing.assert_array_equal(selected, [0, 1])
    np.testing.assert_allclose(values, affine(points) + [[0, 0, 0], [3, 4, 0]])
    reverse = tsselect.MapSelector(inner=(right, left))
    values, selected = reverse.field(points, state([-1, -1]))
    np.testing.assert_array_equal(selected, [0, 0])
    np.testing.assert_allclose(values, affine(points) + [3, 4, 0])


def test_side_then_face_then_next_carries_state(touching):
    selector = tsselect.MapSelector(inner=touching)
    points = np.array([[0.75, 0.5, 0.5], [1.25, 0.5, 0.5]])
    _, previous = selector.field(points, state([-1, -1]))
    np.testing.assert_array_equal(previous, [0, 1])
    points[:, 0] = 1
    face, previous = selector.field(points, previous)
    np.testing.assert_array_equal(previous, [0, 1])
    np.testing.assert_allclose(face[1] - face[0], [3, 4, 0])
    points[:, 0] = [1.25, 0.75]
    values, previous = selector.field(points, previous)
    np.testing.assert_array_equal(previous, [1, 0])
    np.testing.assert_allclose(values, affine(points) + [[3, 4, 0], [0, 0, 0]])


def test_leaving_inner_resets_history_and_explicit_reset(touching, tmp_path):
    outer = loaded(tmp_path, "outer", lo=(-2, 0, 0), n=(6, 3, 3), delta=(20, 0, 0))
    selector = tsselect.MapSelector(inner=touching, outer=(outer,))
    _, previous = selector.field([[1.5, 0.5, 0.5]], state([-1]))
    np.testing.assert_array_equal(previous, [1])
    _, reset = selector.field([[1, 0.5, 0.5]], state([-1]))
    np.testing.assert_array_equal(reset, [0])
    _, previous = selector.field([[2.5, 0.5, 0.5]], previous)
    np.testing.assert_array_equal(previous, [2])
    _, previous = selector.field([[1, 0.5, 0.5]], previous)
    np.testing.assert_array_equal(previous, [0])
    _, previous, covered = selector.field([[10, 0.5, 0.5]], state([1]), return_coverage=True)
    np.testing.assert_array_equal(previous, [-1])
    np.testing.assert_array_equal(covered, [False])
    _, previous = selector.field([[1, 0.5, 0.5]], previous)
    np.testing.assert_array_equal(previous, [0])


def test_noncovering_third_prior_falls_back_to_first_inner(touching, tmp_path):
    third = loaded(tmp_path, "third", lo=(3, 0, 0))
    selector = tsselect.MapSelector(inner=touching + (third,))
    values, selected = selector.field([[1, 0.5, 0.5]], state([2]))
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_allclose(values, affine([[1, 0.5, 0.5]]))


def test_outer_priority_ignores_previous_outer_and_inner_wins(touching, tmp_path):
    first = loaded(tmp_path, "outer_first", lo=(-2, 0, 0), n=(6, 3, 3), delta=(20, 0, 0))
    second = loaded(tmp_path, "outer_second", lo=(-2, 0, 0), n=(8, 3, 3), delta=(40, 0, 0))
    selector = tsselect.MapSelector(inner=touching, outer=(first, second))
    points = np.array([[2.5, 0.5, 0.5], [1, 0.5, 0.5], [4.5, 0.5, 0.5]])
    values, selected = selector.field(points, state([3, 3, 2]))
    np.testing.assert_array_equal(selected, [2, 0, 3])
    np.testing.assert_allclose(values, affine(points) + [[20, 0, 0], [0, 0, 0], [40, 0, 0]])
    # No inner maps is valid, and outer state still has no preferential effect.
    values, selected = tsselect.MapSelector(outer=(first, second)).field(points, state([1, 1, 0]))
    np.testing.assert_array_equal(selected, [0, 0, 1])
    np.testing.assert_allclose(values, affine(points) + [[20, 0, 0], [20, 0, 0], [40, 0, 0]])


def test_mirrored_boxes_shared_negative_face_and_by_flip(tmp_path):
    mirror = loaded(tmp_path, "mirror", lo=(0, 0.5, 0), n=(2, 4, 3), mirror=True,
                    shift=(10, 13, -8))
    negative = loaded(tmp_path, "negative", lo=(1, -2, 0), n=(2, 4, 3), delta=(7, 8, 9))
    selector = tsselect.MapSelector(inner=(mirror, negative))
    points = np.array([[1, -1, 0.5], [1, -1, 0.5], [0, -2, 1], [0, 2, 1]])
    values, selected = selector.field(points, state([0, 1, -1, -1]))
    np.testing.assert_array_equal(selected, [0, 1, 0, 0])
    expected = affine(points * [1, -1, 1])
    expected[:, 1] *= -1
    expected[1] = affine(points[1]) + [7, 8, 9]
    expected[3] = affine(points[3])
    np.testing.assert_allclose(values, expected)


def test_mirror_y_gap_is_not_coverage_or_inner_overlap(tmp_path):
    mirror = loaded(tmp_path, "mirror", lo=(0, 2, 0), mirror=True)
    gap = loaded(tmp_path, "gap", lo=(0, -0.5, 0))
    selector = tsselect.MapSelector(inner=(mirror, gap))
    points = np.array([[0.5, -2.5, 0.5], [0.5, 0, 0.5], [0.5, 2.5, 0.5], [0.5, 1, 0.5]])
    values, selected, covered = selector.field(points, state([-1, 0, -1, 1]), return_coverage=True)
    np.testing.assert_array_equal(selected, [0, 1, 0, -1])
    np.testing.assert_array_equal(covered, [True, True, True, False])
    np.testing.assert_allclose(values[1], affine(points[1]))
    assert np.isnan(values[3]).all()
    # The unchanged strict bank also understands the mirror gap correctly.
    tsfield.FieldMapBank(inner=(mirror, gap))


@pytest.mark.parametrize("lo", [(1, 0, 0), (1, 1, 0), (1, 1, 1)])
def test_touching_face_edge_corner_allowed_but_strict_bank_unchanged(tmp_path, lo):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=lo, delta=(1, 2, 3))
    selector = tsselect.MapSelector(inner=(left, right))
    point = np.asarray(lo, dtype=float)
    values, selected = selector.field(point, state(1))
    assert selected.shape == ()
    assert selected.item() == 1
    np.testing.assert_allclose(values, affine(point) + [1, 2, 3])
    with pytest.raises(ValueError, match="shared boundaries"):
        tsfield.FieldMapBank(inner=(left, right))


@pytest.mark.parametrize("lo", [(0.5, 0, 0), (0, 0, 0)])
def test_positive_volume_overlap_rejected(tmp_path, lo):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=lo)
    with pytest.raises(ValueError, match="positive-volume"):
        tsselect.MapSelector(inner=(left, right))
    with pytest.raises(ValueError, match="positive-volume"):
        tsselect.MapSelector(inner=(left, left))
    # Outer overlaps, including overlap with an inner map, remain allowed.
    tsselect.MapSelector(inner=(left,), outer=(left, right))


def test_mirrored_positive_volume_overlap_rejected(tmp_path):
    mirror = loaded(tmp_path, "mirror", lo=(0, 2, 0), mirror=True)
    negative = loaded(tmp_path, "negative", lo=(0, -2.75, 0))
    with pytest.raises(ValueError, match="positive-volume"):
        tsselect.MapSelector(inner=(mirror, negative))


@pytest.mark.parametrize("previous", [
    [-1], np.array([-1.0]), np.array([0.5]), np.array([np.nan]), np.array([np.inf]),
    np.array([False]), np.array([True]), np.array([0], dtype=np.int32),
    np.array([0], dtype=np.uint64), np.array([2**64 - 1], dtype=np.uint64),
    np.array([0], dtype=object), state([-2]), state([2]), state([2**63 - 1]),
    state(-1), state([[-1]]), state([-1, -1]),
])
def test_invalid_previous_rejected(touching, previous):
    with pytest.raises(ValueError, match="previous"):
        tsselect.MapSelector(inner=touching).field([[1, 0.5, 0.5]], previous)


@pytest.mark.parametrize("points", [[], [1, 2], [[1, 2]], np.zeros((2, 3, 1)),
                                   [[np.nan, 0, 0]], [[0, np.inf, 0]], [[0, 0, -np.inf]]])
def test_invalid_positions_rejected(touching, points):
    with pytest.raises(ValueError, match="positions"):
        tsselect.MapSelector(inner=touching).field(points, state([-1]))


@pytest.mark.parametrize("role", ["inner", "outer"])
def test_invalid_maps_rejected(role):
    with pytest.raises(TypeError, match="FieldMap"):
        tsselect.MapSelector(**{role: [object()]})


def test_uncovered_error_and_opt_in_nan_mask(touching):
    selector = tsselect.MapSelector(inner=touching)
    points = np.array([[[0, 0, 0], [9, 0, 0]], [[1, 1, 1], [1, 1.1, 1]]], dtype=float)
    previous = state([[-1, 1], [1, 0]])
    before = previous.copy()
    with pytest.raises(tsfield.CoverageError) as caught:
        selector.field(points, previous)
    np.testing.assert_array_equal(caught.value.indices, [1, 3])
    np.testing.assert_array_equal(caught.value.covered, [[True, False], [True, False]])
    values, selected, covered = selector.field(points, previous, return_coverage=True)
    np.testing.assert_array_equal(selected, [[0, -1], [1, -1]])
    np.testing.assert_array_equal(covered, caught.value.covered)
    np.testing.assert_allclose(values[covered], affine(points[covered]) + [[0, 0, 0], [3, 4, 0]])
    assert np.isnan(values[~covered]).all()
    np.testing.assert_array_equal(previous, before)


@pytest.mark.parametrize("batch_shape", [(), (0,), (2, 0, 4), (2, 2)])
def test_empty_and_batched_shapes_with_fresh_output(touching, batch_shape):
    selector = tsselect.MapSelector(inner=touching)
    points = np.full(batch_shape + (3,), 0.5)
    previous = np.full(batch_shape, -1, dtype=np.int64)
    previous.setflags(write=False)
    points.setflags(write=False)
    values, selected, covered = selector.field(points, previous, return_coverage=True)
    assert values.shape == points.shape
    assert selected.shape == covered.shape == batch_shape
    assert selected.dtype == np.int64
    assert covered.dtype == bool
    assert not np.shares_memory(selected, previous)
    assert not np.shares_memory(values, points)
    np.testing.assert_array_equal(selected, np.zeros(batch_shape, dtype=np.int64))
    assert covered.all()
    np.testing.assert_allclose(values, affine(points))
    np.testing.assert_array_equal(previous, np.full(batch_shape, -1, dtype=np.int64))
    plain, plain_state = selector.field(points, previous)
    np.testing.assert_array_equal(plain_state, selected)
    np.testing.assert_allclose(plain, values)


def test_no_maps_empty_is_valid_and_nonempty_uncovered():
    selector = tsselect.MapSelector()
    values, selected = selector.field(np.empty((0, 3)), state([]))
    assert values.shape == (0, 3)
    assert selected.shape == (0,)
    with pytest.raises(tsfield.CoverageError):
        selector.field([0, 0, 0], state(-1))
    values, selected, covered = selector.field([0, 0, 0], state(-1), return_coverage=True)
    assert np.isnan(values).all()
    assert selected.item() == -1 and not covered.item()
    with pytest.raises(ValueError, match="previous"):
        selector.field([0, 0, 0], state(0))


def test_lane_permutation_and_compaction_explicitly_carry_state(touching):
    selector = tsselect.MapSelector(inner=touching)
    sides = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5],
                      [1.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    _, previous = selector.field(sides, np.full(4, -1, dtype=np.int64))
    before = previous.copy()
    face = np.tile([1, 0.5, 0.5], (4, 1))
    full_values, full_state = selector.field(face, previous)
    perm = np.array([2, 0, 3, 1])
    values, selected = selector.field(face[perm], previous[perm])
    np.testing.assert_allclose(values, full_values[perm])
    np.testing.assert_array_equal(selected, full_state[perm])
    active = np.array([3, 1])  # No inactive-lane protocol: only these lanes are queried.
    values, selected = selector.field(face[active], previous[active])
    np.testing.assert_allclose(values, full_values[active])
    np.testing.assert_array_equal(selected, full_state[active])
    next_state = previous.copy()
    next_state[active] = selected
    np.testing.assert_array_equal(previous, before)
    np.testing.assert_array_equal(next_state, full_state)


def test_no_hidden_history_or_randomness(touching, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("selector must not use RNG")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "random", forbidden)
    selector = tsselect.MapSelector(inner=touching)
    points = np.array([[1, 0.5, 0.5]])
    values, selected = selector.field(points, state([-1]))
    selector.field([[1.5, 0.5, 0.5]], state([-1]))
    again, again_state = selector.field(points, state([-1]))
    np.testing.assert_array_equal(again, values)
    np.testing.assert_array_equal(again_state, selected)
    assert set(vars(selector)) == {"inner", "outer"}


def test_audit_maps_literal_contract_and_explicit_stream_scope(touching, tmp_path):
    outer = loaded(tmp_path, "outer", lo=(-2, 0, 0), n=(6, 3, 3), mirror=True)
    selector = tsselect.MapSelector(inner=touching, outer=(outer,))
    report = selector.audit()
    assert report["schema"] == "beamfastsim.tsselect.audit.v1"
    assert report["selection_contract"] == (
        "if prior inner covers point retain it; else configured first inner; "
        "else configured first outer; uncovered fail with CoverageError "
        "(return_coverage=True gives NaNs, selected_map=-1, covered=False)"
    )
    assert "independent" in report["scope"]
    assert "not Geant4 global cache" in report["scope"]
    assert "same selector/bank" in report["state_contract"]
    assert "cannot detect another bank" in report["state_contract"]
    for flag in ("transport_validated", "inclusive_population_validated",
                 "stock_endpoint_equivalence", "bank_policy_changed"):
        assert report[flag] is False
    for role, maps in [("inner", touching), ("outer", (outer,))]:
        assert len(report[role]) == len(maps)
        for entry, item in zip(report[role], maps):
            assert entry["shape"] == list(item.B.shape)
            for key, path in [("header", item.header.path),
                              ("binary", item.header.path.with_suffix(".bin"))]:
                assert entry[key]["path"] == str(path.resolve())
                assert entry[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(report["outer"][0]["coverage_boxes_mm"]) == 2
    assert "need not be identical" in " ".join(report["limitations"])
    assert "clamped endpoints" in " ".join(report["limitations"])
    json.dumps(report, allow_nan=False)


def test_import_requires_neither_jax_nor_core():
    package_root = str(Path(__file__).resolve().parents[1])
    program = (
        "import builtins, sys\n"
        f"sys.path.insert(0, {package_root!r})\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    level = args[3] if len(args) > 3 else kwargs.get('level', 0)\n"
        "    if name == 'jax' or name.startswith('jax.') or name == 'beamfastsim.core' or (name == 'core' and level == 0):\n"
        "        raise AssertionError('unexpected JAX/core dependency: ' + name)\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "from beamfastsim import tsselect\n"
        "assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_receipt_distinguishes_evaluated_array_from_current_disk(touching):
    selector = tsselect.MapSelector(inner=touching)
    before = selector.audit()["inner"][0]
    path = touching[0].header.path.with_suffix(".bin")
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    after = selector.audit()["inner"][0]
    assert before["binary"]["sha256"] != after["binary"]["sha256"]
    assert before["evaluated_array"] == after["evaluated_array"]
    assert after["evaluated_grid"]["lo"] == touching[0].lo.tolist()
    assert "not a proof" in after["disk_hash_scope"]
