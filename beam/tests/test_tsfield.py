import numpy as np
import pytest

from beamfastsim.tsfield import FieldBank, from_config
from mu2efastsim import ehelix


def header(tmp_path, name, offset=(0, 0, 0), value=(1, 2, 3), reflect=False,
           spacing=(1, 1, 1), grid=None):
    if grid is None:
        grid = np.broadcast_to(np.array(value, float), (2, 2, 2, 3))
    n = grid.shape[:3]
    path = tmp_path / f"{name}.header"
    path.write_text(f"grid X0={offset[0]} Y0={offset[1]} Z0={offset[2]} "
                    f"nX={n[0]} nY={n[1]} nZ={n[2]} "
                    f"dX={spacing[0]} dY={spacing[1]} dZ={spacing[2]}\n"
                    + ("extendY flip=By\n" if reflect else "") + "data\n")
    with path.with_suffix(".bin").open("wb") as f:
        f.write(np.array([0xDEADBEEF], np.uint32).tobytes())
        f.write(np.asarray(grid, dtype=float).tobytes())
    return path


def test_reflect_and_closed_upper_boundary(tmp_path):
    bank = FieldBank([header(tmp_path, "map", reflect=True)])
    field, valid = bank.evaluate([[1, -1, 1], [0.5, 0.5, 0.5], [2, 0, 0]])
    np.testing.assert_allclose(field, [[1, -2, 3], [1, 2, 3], [0, 0, 0]])
    np.testing.assert_array_equal(valid, [True, True, False])


def test_first_containing_no_summing(tmp_path):
    bank = FieldBank([header(tmp_path, "a", value=(1, 0, 0)),
                      header(tmp_path, "b", offset=(0.5, 0, 0), value=(0, 1, 0))], scale=2)
    np.testing.assert_allclose(bank([[0.75, 0, 0], [1.25, 0, 0]]), [[2, 0, 0], [0, 2, 0]])


def test_from_config(tmp_path):
    header(tmp_path, "a")
    conf = tmp_path / "field.txt"
    conf.write_text('vector<string> bfield.innerMaps={"a.header"};\n'
                    'vector<string> bfield.outerMaps={};\ndouble bfield.scaleFactor=1;\n')
    bank = from_config(conf, tmp_path, tmp_path)
    np.testing.assert_allclose(bank([[0, 0, 0]]), [[1, 2, 3]])


def test_errors_and_empty(tmp_path):
    with pytest.raises(ValueError):
        FieldBank([])
    bank = FieldBank([header(tmp_path, "a")])
    assert bank(np.empty((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        bank([[np.nan, 0, 0]])
    with pytest.raises(ValueError):
        FieldBank([header(tmp_path, "bad", offset=(0, 1, 0), reflect=True)])


def test_shifted_upper_map_face(tmp_path):
    bank = FieldBank([header(tmp_path, "shifted", offset=(-1, 0, -1))])
    b, valid = bank.evaluate([[0, 1, 0]])
    assert valid[0]
    np.testing.assert_allclose(b, [[1, 2, 3]], atol=1e-14)


def test_fetch_follows_release_includes_before_selecting_maps(tmp_path, monkeypatch):
    from pathlib import Path
    from beamfastsim import tsfield
    fetched = []
    configs = {
        "Offline/top.txt": '#include "Offline/base.txt"\nvector<string> bfield.innerMaps={"BFieldMaps/new.header"};',
        "Offline/base.txt": 'vector<string> bfield.innerMaps={"BFieldMaps/old.header"};\n'
                            'vector<string> bfield.outerMaps={};\ndouble bfield.scaleFactor=1;',
    }
    class Repo:
        def fetch(self, source, dest):
            fetched.append(source)
            p = Path(dest)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(configs.get(source.removeprefix("/Musings/Offline/test/"), "map"))
    monkeypatch.setattr(tsfield.cvmfsget, "Repo", lambda *args: Repo())
    report = tsfield.fetch(tmp_path, "test", "Offline/top.txt")
    assert report["map_names"] == ["BFieldMaps/new.header"]
    assert "/Musings/Offline/test/Offline/base.txt" in fetched
    assert "/DataFiles/BFieldMaps/new.bin" in fetched
    assert not any("old." in path for path in fetched)


def baseline_evaluate(headers, x, scale=1.0):
    """Frozen pre-optimization evaluation, with independent core interpolators."""
    x = np.asarray(x, dtype=float)
    out, covered = np.zeros_like(x), np.zeros(len(x), bool)
    for path in headers:
        lo, spacing, n, grid = ehelix.read_g4bl(str(path))
        hi = lo + spacing * (n - 1)
        reflect = "extendY flip=By" in path.read_text()
        interp = ehelix.grid_field(np.zeros(3), np.ones(3), n, grid)
        pos = x.copy() if reflect else x
        if reflect:
            pos[:, 1] = np.abs(pos[:, 1])
        mask = ~covered & np.all((pos >= lo) & (pos <= hi), axis=1)
        if mask.any():
            grid_position = (pos[mask] - lo) / spacing
            out[mask] = interp(np.clip(grid_position, 0, np.nextafter((n-1).astype(float), 0)))
            if reflect:
                out[mask, 1] *= np.where(x[mask, 1] < 0, -1.0, 1.0)
        covered |= mask
    return scale * out, covered


@pytest.fixture
def mixed_maps(tmp_path):
    rng = np.random.default_rng(20261105)
    specs = [((-2.3, 0, -1.1), (0.7, 0.3, 0.9), (6, 5, 4), True),
             ((-0.9, -1.7, -0.2), (0.6, 0.8, 0.4), (5, 5, 6), False),
             ((0.3, 0, -2.5), (0.3, 0.7, 1.1), (7, 4, 5), True),
             ((5.1, -3.3, 0.7), (0.2, 0.9, 0.5), (4, 3, 5), False)]
    headers, points = [], [rng.uniform([-4, -5, -4], [8, 5, 5], (1000, 3))]
    for k, (lo, spacing, shape, reflect) in enumerate(specs):
        grid = rng.normal(size=(*shape, 3)) * [0.003, 11, 700]
        headers.append(header(tmp_path, f"random{k}", lo, reflect=reflect,
                              spacing=spacing, grid=grid))
        lo, spacing, n, _ = ehelix.read_g4bl(str(headers[-1]))
        hi = lo + spacing * (n - 1)
        # All grid vertices, including closed faces/edges/corners, plus adjacent ULPs.
        vertices = np.stack(np.meshgrid(*(lo[i] + spacing[i] * np.arange(n[i])
                                         for i in range(3)), indexing="ij"), axis=-1).reshape(-1, 3)
        local = [rng.uniform(lo, hi, (500, 3)), vertices]
        for axis in range(3):
            for direction in (-np.inf, np.inf):
                adjacent = vertices.copy()
                adjacent[:, axis] = np.nextafter(adjacent[:, axis], direction)
                local.append(adjacent)
        local = np.concatenate(local)
        points.append(local)
        if reflect:
            reflected = local.copy()
            reflected[:, 1] *= -1  # Includes negative zero on the reflection plane.
            points.append(reflected)
    return headers, np.concatenate(points)


@pytest.mark.parametrize("selection", ["all_covered", "partial", "missing", "empty"])
@pytest.mark.parametrize("scale", [1.0, -1.7, 0.0, -0.0])
def test_exact_baseline_mixed_nonconstant_maps(mixed_maps, selection, scale):
    headers, x = mixed_maps
    _, covered = baseline_evaluate(headers, x)
    assert covered.any() and (~covered).any()
    if selection == "all_covered":
        x = x[covered]
    elif selection == "missing":
        x = x[~covered]
    elif selection == "empty":
        x = x[:0]
    # Exercise a read-only, noncontiguous input and ensure reflection never mutates it.
    storage = np.zeros((len(x), 6))
    storage[:, ::2] = x
    x = storage[:, ::2]
    x.flags.writeable = False
    original = storage.copy()
    expected, expected_covered = baseline_evaluate(headers, x, scale)
    actual, actual_covered = FieldBank(headers, scale).evaluate(x)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(actual_covered, expected_covered)
    np.testing.assert_array_equal(storage, original)


def test_multiple_overlapping_maps_priority(tmp_path):
    headers = [header(tmp_path, "a", value=(1, 2, 3), reflect=True),
               header(tmp_path, "b", offset=(0.5, 0, 0), value=(4, 5, 6), reflect=True),
               header(tmp_path, "c", offset=(0.75, 0, 0), value=(7, 8, 9), reflect=True),
               header(tmp_path, "d", offset=(1, 0, 0), value=(10, 11, 12), reflect=True)]
    field, covered = FieldBank(headers).evaluate([[1, -1, 1], [1.5, -1, 1],
                                                 [1.75, -1, 1], [2, -1, 1], [2.1, -1, 1]])
    np.testing.assert_allclose(field, [[1, -2, 3], [4, -5, 6], [7, -8, 9],
                                       [10, -11, 12], [0, 0, 0]], atol=1e-14)
    np.testing.assert_array_equal(covered, [True, True, True, True, False])


@pytest.mark.parametrize("count", [0, 1, 3])
def test_prunes_domain_checks_and_reuses_clip_limits(tmp_path, monkeypatch, count):
    bank = FieldBank([header(tmp_path, str(i), offset=(2*i, 0, 0), reflect=True)
                      for i in range(4)])
    sizes = []
    original_all = np.all

    def checked_all(a, *args, **kwargs):
        if kwargs.get("axis") == 1:
            sizes.append(len(a))
        return original_all(a, *args, **kwargs)

    def unexpected_nextafter(*args, **kwargs):
        raise AssertionError("clip limits must be precomputed")

    with monkeypatch.context() as m:
        m.setattr(np, "all", checked_all)
        m.setattr(np, "nextafter", unexpected_nextafter)
        field, covered = bank.evaluate(np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])[:count])
    # Each visited map checks remaining rows, then core checks the one selected row.
    assert sizes == [size for remaining in range(count, 0, -1) for size in (remaining, 1)]
    assert covered.all()
    assert field.shape == (count, 3)


@pytest.mark.parametrize("signed_zero", [False, True])
def test_ordered_corner_interpolation_matches_core_exactly(signed_zero):
    from beamfastsim.tsfield import _grid_field
    rng = np.random.default_rng(20261106)
    n = np.array([7, 5, 9])
    grid = rng.normal(size=(*n, 3)) * np.exp2(rng.integers(-90, 90, size=(*n, 3)))
    if signed_zero:
        grid = np.copysign(np.zeros_like(grid), grid)
    vertices = np.indices(n).reshape(3, -1).T.astype(float)
    points = np.concatenate((rng.uniform(-1, n, (2000, 3)), vertices,
                             np.nextafter(vertices, -np.inf),
                             np.nextafter(vertices, np.inf), [[-0.0, -0.0, -0.0]]))
    expected = ehelix.grid_field(np.zeros(3), np.ones(3), n, grid)(points)
    actual = _grid_field(n, grid)(points)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert _grid_field(n, grid)(np.empty((0, 3))).shape == (0, 3)


def test_safety_reflected_rectangle_faces_and_outside(tmp_path):
    bank = FieldBank([header(tmp_path, "reflected", offset=(-2, 0, -3),
                             spacing=(4, 2, 6), reflect=True)])
    x = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0],
                  [-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0],
                  [0, 0, -3], [0, 0, 3], [2, 2, 3],
                  [np.nextafter(2., np.inf), 0, 0],
                  [0, np.nextafter(-2., -np.inf), 0], [0, 0, 4]])
    x.flags.writeable = False
    expected = np.nextafter(np.array([2, 1, 1] + [0]*10, float), 0)
    np.testing.assert_array_equal(bank.safety(x), expected)
    assert bank.safety(np.empty((0, 3))).shape == (0,)


def test_safety_overlap_uses_best_containing_map_not_selected_map(tmp_path):
    bank = FieldBank([header(tmp_path, "first", offset=(-2, -2, -2), spacing=(2, 4, 4)),
                     header(tmp_path, "second", offset=(-1, -3, -3), spacing=(4, 6, 6),
                            value=(4, 5, 6))])
    x = np.array([[0, 0, 0], [-0.5, 0, 0], [-1.5, 0, 0],
                  [1, 0, 0], [-2, 0, 0], [3, 0, 0], [4, 0, 0]])
    expected = np.nextafter(np.array([1, 0.5, 0.5, 2, 0, 0, 0]), 0)
    np.testing.assert_array_equal(bank.safety(x), expected)
    np.testing.assert_allclose(bank(x[:1]), [[1, 2, 3]])
    reversed_bank = FieldBank([tmp_path / "second.header", tmp_path / "first.header"])
    np.testing.assert_array_equal(bank.safety(x), reversed_bank.safety(x))


def test_safety_never_joins_disconnected_domains(tmp_path):
    bank = FieldBank([header(tmp_path, "left", offset=(-3, 0, 0)),
                     header(tmp_path, "right", offset=(2, 0, 0))])
    np.testing.assert_array_equal(bank.safety([[0, 0.5, 0.5], [-2, 0.5, 0.5],
                                             [2, 0.5, 0.5]]), [0, 0, 0])


def test_safety_certifies_covered_spheres(mixed_maps):
    headers, x = mixed_maps
    bank = FieldBank(headers)
    safety = bank.safety(x)
    _, covered = baseline_evaluate(headers, x)
    assert np.all(safety >= 0)
    assert np.all(safety[~covered] == 0)
    rng = np.random.default_rng(17)
    directions = rng.normal(size=(32, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    interior = safety > 1e-10
    probes = (x[interior, None, :] + (safety[interior, None, None]
                                    * (1 - 1e-12) * directions)).reshape(-1, 3)
    assert baseline_evaluate(headers, probes)[1].all()


@pytest.mark.parametrize("bad", [[0, 0, 0], [[0, 0]], [[np.nan, 0, 0]], [[0, np.inf, 0]]])
def test_safety_rejects_invalid_positions(tmp_path, bad):
    bank = FieldBank([header(tmp_path, "a")])
    with pytest.raises(ValueError, match="finite N by 3"):
        bank.safety(bad)


@pytest.mark.parametrize('scale', [0., 1., -2.])
def test_global_bound_covers_interpolated_reflected_fields(mixed_maps, scale):
    headers, x = mixed_maps
    bank = FieldBank(headers, scale=scale)
    norms = np.linalg.norm(bank(x), axis=1)
    assert np.all(norms <= bank.max_field_t)
    if scale:
        assert bank.max_field_t > 0
    else:
        assert bank.max_field_t == 0
