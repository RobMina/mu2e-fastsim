"""Straight-segment ownership covers; not a curved or native crossing test."""

import numpy as np
import pytest

from beamfastsim import tschord as c, tslocate as l, tsnav as g


def scene(*obstacles):
    mother = l.Volume(g.Box([10, 10, 10]), 'vac', 'mother', 'root', True)
    daughters = tuple(l.Volume(s, f'metal{i}', f'leaf{i}', 'root', False)
                      for i, s in enumerate(obstacles))
    return l.Partition((l.Cell(mother, daughters),))


def run(geometry, start=None, end=None, **kwargs):
    start = [[-2., 0, 0]] if start is None else start
    end = [[2., 0, 0]] if end is None else end
    return c.partition(start, end, geometry, assume_straight=True, **kwargs)


def test_vacuum_and_zero_length_chords_are_certified_without_nudges():
    out = run(scene(), [[-2., 0, 0], [0, 0, 0]], [[2., 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(out.source_index, [0, 1])
    np.testing.assert_array_equal(out.status, c.RESOLVED)
    np.testing.assert_array_equal(out.volume_index, 0)
    np.testing.assert_allclose(out.width, [4, 0])
    out = run(scene(), [[10., 0, 0]], [[10., 0, 0]])
    assert out.status[0] == c.BOUNDARY_PENDING
    assert out.volume_index[0] == -1


def test_thin_intervening_material_not_skipped_with_same_vacuum_endpoints():
    slab = g.Placed(g.Box([.001, 9, 9]), [.5, 0, 0], np.eye(3))
    out = run(scene(slab), atol=1e-7)
    assert np.all(out.source_index == 0)
    assert out.fraction_lo[0] == 0 and out.fraction_hi[-1] == 1
    np.testing.assert_array_equal(out.fraction_hi[:-1], out.fraction_lo[1:])
    unresolved = out.status != c.RESOLVED
    assert unresolved.any() and np.all(out.width[unresolved] <= 1e-7)
    assert np.all(out.volume_index[unresolved] == -1)
    metal = (out.status == c.RESOLVED) & (out.volume_index == 1)
    assert metal.any()
    assert abs(np.sum(out.width[metal]) - .002) <= np.sum(out.width[unresolved]) + 1e-14
    for face in (.499, .501):
        fraction = (face + 2) / 4
        assert np.any(unresolved & (out.fraction_lo <= fraction) & (out.fraction_hi >= fraction))
    assert np.all(out.depth < 64)


def test_depth_cap_cannot_invent_material_crossing_or_owner():
    out = run(scene(g.Box([.01, 9, 9])), max_depth=0)
    assert out.status[0] == c.DEPTH_LIMIT
    assert out.volume_index[0] == -1
    assert out.midpoint_status[0] == l.MATERIAL
    with pytest.raises(ValueError, match='max_intervals'):
        run(scene(g.Box([.01, 9, 9])), max_intervals=2)


def test_batch_order_reverse_and_no_input_mutation():
    a = np.array([[-2., 0, 0], [2., 0, 0]])
    b = -a
    saved = a.copy()
    out = run(scene(g.Box([.01, 9, 9])), a, b, atol=1e-5)
    assert np.all(out.source_index[:-1] <= out.source_index[1:])
    np.testing.assert_array_equal(a, saved)
    for index in (0, 1):
        use = out.source_index == index
        np.testing.assert_array_equal(out.fraction_hi[use][:-1], out.fraction_lo[use][1:])
        assert out.fraction_lo[use][0] == 0 and out.fraction_hi[use][-1] == 1
        assert np.sum(out.width[use]) == 4


def test_same_material_root_joint_remains_unresolved_not_implicitly_welded():
    a = l.Volume(g.Placed(g.Box([2, 2, 2]), [-2, 0, 0], np.eye(3)), 'vac', 'a', 'a', True)
    b = l.Volume(g.Placed(g.Box([2, 2, 2]), [2, 0, 0], np.eye(3)), 'vac', 'b', 'b', True)
    out = run(l.Partition((l.Cell(a), l.Cell(b))), atol=1e-6)
    assert np.any(out.status == c.BOUNDARY_PENDING)
    assert set(out.volume_index[out.status == c.RESOLVED]) == {0, 1}


def test_explicit_contracts_invalid_inputs_and_empty_batches():
    with pytest.raises(ValueError, match='assume_straight'):
        c.partition([[0, 0, 0]], [[1, 0, 0]], scene())
    for kwargs in ({'atol': 0}, {'max_depth': 65}, {'max_depth': True},
                   {'max_intervals': 0}, {'max_intervals': 1.5}):
        with pytest.raises(ValueError):
            run(scene(), **kwargs)
    for start, end in (([0, 0, 0], [1, 0, 0]), ([[np.nan, 0, 0]], [[1, 0, 0]]),
                       ([[1e308, 0, 0]], [[-1e308, 0, 0]])):
        with pytest.raises(ValueError):
            run(scene(), start, end)
    assert run(scene(), np.empty((0, 3)), np.empty((0, 3))).status.size == 0


def test_lossy_endpoint_reconstruction_fails_before_certifying_wrong_chord():
    mother = l.Volume(g.HalfSpace([1, 0, 0], .5), 'vac', 'mother', 'root', True)
    geometry = l.Partition((l.Cell(mother),))
    a, b = [[-float(2**54), 0, 0]], [[1., 0, 0]]
    for start, end in ((a, b), (b, a)):
        with pytest.raises(ValueError, match='preserve both supplied endpoints'):
            run(geometry, start, end)


def test_cli_chord_receipt_keeps_unresolved_intervals_and_alias_guard(tmp_path):
    import json
    from test_tsgeom import fixture
    path = fixture(tmp_path)
    out = tmp_path / 'chord.json'
    argv = [str(path), '--root', 'Mother', '--vacuum-material', 'vac', '--assume-straight',
            '--chord', '110', '20', '30', '114', '20', '30', '--out', str(out)]
    c.main(argv)
    receipt = json.loads(out.read_text())
    assert not receipt['native_query_certified']
    assert not receipt['curved_path_certified']
    assert not receipt['material_transport']
    assert c.BOUNDARY_PENDING in receipt['status']
    assert 1 in receipt['volume_index']
    summary = receipt['per_chord']
    assert summary['interval_count'] == [len(receipt['status'])]
    assert abs(summary['resolved_material_mm'][0] - 1) <= summary['unresolved_mm'][0]
    assert abs(summary['resolved_vacuum_mm'][0] + summary['resolved_material_mm'][0]
               + summary['unresolved_mm'][0] - 4) < 1e-14
    saved = path.read_bytes()
    alias = tmp_path / 'alias.json'
    alias.symlink_to(path)
    with pytest.raises(SystemExit):
        c.main(argv[:-1] + [str(alias)])
    assert path.read_bytes() == saved
