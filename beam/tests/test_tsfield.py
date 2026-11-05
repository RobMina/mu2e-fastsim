"""Synthetic CPU-reference tests; no field assets, JAX, or transport required."""

import hashlib
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from beamfastsim import tsfield


MATRIX = np.array([[2.0, -3.0, 0.5], [-1.0, 4.0, 2.0], [0.25, 2.0, -5.0]])
OFFSET = np.array([0.3, -1.2, 2.5])


def affine(points):
    return np.asarray(points) @ MATRIX.T + OFFSET


def write_map(tmp_path, name="map", *, lo=(0, 0, 0), n=(3, 4, 5),
              d=(1.25, 0.5, 2.0), shift=(0, 0, 0), explicit=True,
              mirror=False, order="<", constant=None):
    path = tmp_path / (name + ".header")
    lo = np.asarray(lo, dtype=float)
    d = np.asarray(d, dtype=float)
    origin = lo + np.asarray(shift)
    grid = (f"grid X0={origin[0]:.17g} Y0={origin[1]:.17g} Z0={origin[2]:.17g} "
            f"nX={n[0]} nY={n[1]} nZ={n[2]} "
            f"dX={d[0]:.17g} dY={d[1]:.17g} dZ={d[2]:.17g}\n")
    text = "# synthetic G4BL binary map\n"
    if explicit:
        text += "# Origin shift for Mu2e: " + " ".join(str(v) for v in shift) + "\n"
    text += "param maxline=128 current=1 gradient=1 normB=1 normE=1\n" + grid
    if mirror:
        text += "extendY flip=By\n"
    path.write_text(text + "data\n")
    nodes = np.stack(np.meshgrid(*(np.arange(count) for count in n), indexing="ij"), axis=-1)
    points = lo + nodes * d
    values = affine(points) if constant is None else np.broadcast_to(constant, points.shape)
    path.with_suffix(".bin").write_bytes(struct.pack(order + "I", tsfield.MARKER)
                                          + np.asarray(values, dtype=order + "f8").tobytes())
    return path


def loaded(tmp_path, name, **kwargs):
    return tsfield.load(write_map(tmp_path, name, **kwargs))


@pytest.mark.parametrize("order", ["<", ">"])
def test_affine_batch_axis_order_and_endianness(tmp_path, order):
    item = loaded(tmp_path, "affine", lo=(-8, 1, -4), order=order)
    fractions = np.array([[[0.0, 0.0, 0.0], [0.17, 0.63, 0.91]],
                          [[0.83, 0.21, 0.45], [1.0, 1.0, 1.0]]])
    points = item.lo + fractions * (item.hi - item.lo)
    values, covered = item.field(points, return_coverage=True)
    np.testing.assert_allclose(values, affine(points), atol=2e-14)
    assert covered.shape == (2, 2)
    assert np.all(covered)
    assert item.B.shape == (3, 4, 5, 3)
    assert item.byte_order == ("little" if order == "<" else "big")
    np.testing.assert_allclose(item.B[-1, 1, 2], affine(item.lo + [2.5, 0.5, 4]))
    assert not item.B.flags.writeable
    assert not item.header.d.flags.writeable


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_and_explicit_origin_shift(tmp_path, explicit):
    shift = (5, 2, -7) if explicit else tsfield.LEGACY_ORIGIN_SHIFT
    item = loaded(tmp_path, "shift", lo=(-10, 3, 11), shift=shift, explicit=explicit)
    np.testing.assert_array_equal(item.lo, [-10, 3, 11])
    np.testing.assert_array_equal(item.header.origin_shift, shift)
    assert item.header.explicit_origin is explicit
    point = item.lo + [0.3, 0.2, 0.5]
    np.testing.assert_allclose(item.field(point), affine(point))


def test_y_mirror_uses_mu2e_plane_and_flips_only_by(tmp_path):
    item = loaded(tmp_path, "mirror", lo=(-4, 0, 2), shift=(3, 5, -8), mirror=True)
    positive = np.array([[-3.7, 0.7, 3.4], [-2.0, 1.5, 10.0]])
    negative = positive * [1, -1, 1]
    points = np.concatenate([positive, negative, [[-4, 0, 2]]])
    expected = np.concatenate([affine(positive), affine(positive) * [1, -1, 1], affine([[-4, 0, 2]])])
    np.testing.assert_allclose(item.field(points), expected, atol=2e-14)
    assert np.all(item.coverage(points))
    assert len(item.header.boxes) == 2


def test_mirrored_y_gap_is_not_covered(tmp_path):
    item = loaded(tmp_path, "gap", lo=(0, 2, 0), mirror=True)
    points = np.array([[0, 0, 0], [0, 1.99, 0], [0, -1.99, 0], [0, -2, 0], [0, 2, 0]])
    values, covered = item.field(points, return_coverage=True)
    np.testing.assert_array_equal(covered, [False, False, False, True, True])
    assert np.all(np.isnan(values[:3]))
    with pytest.raises(tsfield.CoverageError):
        item.field(points)


@pytest.mark.parametrize("mirror", [False, True])
def test_closed_bounds_all_corners_faces_and_adjacent_outside(tmp_path, mirror):
    item = loaded(tmp_path, "bounds", lo=(-0.3, 0, -0.7), d=(0.1, 0.2, 0.3), mirror=mirror)
    corners = np.where(np.array(list(itertools.product((False, True), repeat=3))), item.hi, item.lo)
    midpoint = (item.lo + item.hi) / 2
    faces = np.broadcast_to(midpoint, (6, 3)).copy()
    axes = np.repeat(np.arange(3), 2)
    faces[np.arange(6), axes] = np.stack([item.lo, item.hi], axis=1).reshape(-1)
    points = np.concatenate([corners, faces, midpoint[None, :]])
    np.testing.assert_allclose(item.field(points), affine(points), atol=2e-14)
    np.testing.assert_array_equal(item.field(item.hi), item.B[-1, -1, -1])
    outside = np.broadcast_to(midpoint, (6, 3)).copy()
    lower = item.lo.copy()
    if mirror:
        lower[1] = -item.hi[1]
    limits = np.stack([lower, item.hi], axis=1).reshape(-1)
    direction = np.tile([-np.inf, np.inf], 3)
    outside[np.arange(6), axes] = np.nextafter(limits, direction)
    assert not np.any(item.coverage(outside))
    with pytest.raises(tsfield.CoverageError) as error:
        item.field(outside)
    np.testing.assert_array_equal(error.value.indices, np.arange(6))


def test_empty_single_and_bad_queries(tmp_path):
    item = loaded(tmp_path, "query")
    bank = tsfield.FieldMapBank(inner=[item])
    for source in (item, bank):
        values, covered = source.field(np.empty((2, 0, 3)), return_coverage=True)
        assert values.shape == (2, 0, 3)
        assert covered.shape == (2, 0)
        assert source.field(item.lo).shape == (3,)
        for bad in (0, [1, 2], [[1, 2]], [np.nan, 0, 0], [0, np.inf, 0]):
            with pytest.raises(ValueError):
                source.field(bad)


def test_bank_inner_priority_outer_order_and_fail_closed(tmp_path):
    inner = loaded(tmp_path, "inner", n=(2, 2, 2), d=(1, 1, 1), constant=[1, 2, 3])
    outer1 = loaded(tmp_path, "outer1", lo=(-1, -1, -1), n=(2, 2, 2), d=(3, 3, 3), constant=[4, 5, 6])
    outer2 = loaded(tmp_path, "outer2", lo=(-2, -2, -2), n=(2, 2, 2), d=(5, 5, 5), constant=[7, 8, 9])
    bank = tsfield.FieldMapBank(inner=[inner], outer=[outer1, outer2])
    points = np.array([[0.5, 0.5, 0.5], [2, 2, 2], [3, 3, 3], [4, 0, 0]])
    values, covered = bank.field(points, return_coverage=True)
    np.testing.assert_allclose(values[:3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_array_equal(covered, [True, True, True, False])
    np.testing.assert_array_equal(bank.map_indices(points), [0, 1, 2, -1])
    assert np.all(np.isnan(values[-1]))
    with pytest.raises(tsfield.CoverageError) as error:
        bank.field(points)
    np.testing.assert_array_equal(error.value.indices, [3])
    reverse = tsfield.FieldMapBank(outer=[outer2, outer1])
    np.testing.assert_allclose(reverse.field([0, 0, 0]), [7, 8, 9])
    empty = tsfield.FieldMapBank()
    with pytest.raises(tsfield.CoverageError):
        empty.field([0, 0, 0])


def test_inner_overlap_and_shared_face_rejected(tmp_path):
    first = loaded(tmp_path, "first", n=(2, 2, 2), d=(1, 1, 1), constant=[1, 2, 3])
    overlap = loaded(tmp_path, "overlap", lo=(0.5, 0, 0), n=(2, 2, 2), d=(1, 1, 1))
    with pytest.raises(ValueError, match="overlapping inner coverage"):
        tsfield.FieldMapBank(inner=[first, overlap])
    with pytest.raises(ValueError, match="overlapping inner coverage"):
        tsfield.FieldMapBank(inner=[first, first])
    touching = loaded(tmp_path, "touching", lo=(1, 0, 0), n=(2, 2, 2), d=(1, 1, 1), constant=[4, 5, 6])
    with pytest.raises(ValueError, match="shared boundaries"):
        tsfield.FieldMapBank(inner=[first, touching])


def test_inner_overlap_accounts_for_reflection_and_does_not_fill_gap(tmp_path):
    mirror = loaded(tmp_path, "symmetric", lo=(0, 2, 0), mirror=True)
    reflected = loaded(tmp_path, "negative", lo=(0, -3, 0))
    with pytest.raises(ValueError, match="overlapping inner coverage"):
        tsfield.FieldMapBank(inner=[mirror, reflected])
    gap = loaded(tmp_path, "gap_inner", lo=(0, -1, 0), n=(2, 2, 2), d=(1, 2, 1))
    bank = tsfield.FieldMapBank(inner=[mirror, gap])
    np.testing.assert_array_equal(bank.map_indices([[0, -2.5, 0], [0, 0, 0], [0, 2.5, 0]]), [0, 1, 0])


@pytest.mark.parametrize("extension", ["extendX flip=Bx", "extendY flip=Bx", "extendZ", "extendY flip=By extra=1",
                                        "extendY flip=By\nextendY flip=By", "extendY flip=By\nextendX flip=Bx"])
def test_unsupported_extensions_rejected(tmp_path, extension):
    path = write_map(tmp_path)
    path.write_text(path.read_text().replace("data\n", extension + "\ndata\n"))
    with pytest.raises(ValueError, match="extendY flip=By"):
        tsfield.load(path)


@pytest.mark.parametrize("old,new", [("nX=3", "nX=2.5"), ("nX=3", "nX=1"),
                                      ("nX=3", "nX=-3"), ("nX=3", "nX=NaN"),
                                      ("nX=3", "nX=1e99"), ("nX=3", "nX=3.00000000000000001"),
                                      ("nX=3", "nX=2000000000 nY=2000000000"),
                                      ("dX=1.25", "dX=0"), ("dX=1.25", "dX=-1"),
                                      ("dX=1.25", "dX=inf"), ("X0=0", "X0=nan"),
                                      ("X0=0", "X0=1e999"), ("X0=0", "X0=1e300"),
                                      ("nX=3", "points=3"), ("dX=1.25", "dX=1.25 dX=2"),
                                      ("dX=1.25", "dX=1.25 unknown=1")])
def test_malformed_grid_rejected(tmp_path, old, new):
    path = write_map(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        tsfield.load(path)


@pytest.mark.parametrize("record", ["# Origin shift for Mu2e: 0 0", "# Origin shift for Mu2e: 0 NaN 0",
                                     "# Origin shift for Mu2e: 0 0 0 extra", "# Origin shift for Mu2e 0 0 0"])
def test_malformed_origin_shift_rejected(tmp_path, record):
    path = write_map(tmp_path)
    path.write_text(path.read_text().replace("# Origin shift for Mu2e: 0 0 0", record))
    with pytest.raises(ValueError):
        tsfield.load(path)


def test_ambiguous_header_records_and_negative_mirror_grid(tmp_path):
    path = write_map(tmp_path)
    original = path.read_text()
    grid = next(line for line in original.splitlines() if line.startswith("grid"))
    for changed in (original.replace(grid, grid + "\n" + grid),
                    original + "# Origin shift for Mu2e: 0 0 0\n",
                    original.replace(grid, ""), original + "data\n", original + "param foo=1\n"):
        path.write_text(changed)
        with pytest.raises(ValueError):
            tsfield.load(path)
    negative = write_map(tmp_path, "negative_mirror", lo=(0, -1, 0), mirror=True)
    with pytest.raises(ValueError, match="nonnegative"):
        tsfield.load(negative)


def test_header_numeric_notation_comments_and_optional_data(tmp_path):
    path = write_map(tmp_path)
    text = path.read_text().replace("nX=3", "nX=3e0").replace("nY=4", "nY=4.0")
    text = text.replace("dX=1.25", "dX=1.25e+0").replace("data\n", "# no text payload\n")
    path.write_text(text)
    np.testing.assert_array_equal(tsfield.load(path).n, [3, 4, 5])


@pytest.mark.parametrize("suffix", [".txt", ".bin", ".header.gz", ".csv", ".header.extra"])
def test_non_header_asset_rejected(tmp_path, suffix):
    with pytest.raises(ValueError, match="only .header"):
        tsfield.load(tmp_path / ("bad" + suffix))


@pytest.mark.parametrize("mutation", ["truncated_marker", "short", "trailing", "old_coordinates", "marker", "nan", "inf"])
def test_corrupt_binary_rejected(tmp_path, mutation):
    path = write_map(tmp_path)
    binary = path.with_suffix(".bin")
    payload = binary.read_bytes()
    if mutation == "truncated_marker":
        payload = payload[:3]
    elif mutation == "short":
        payload = payload[:-8]
    elif mutation == "trailing":
        payload += b"extra"
    elif mutation == "old_coordinates":
        payload += payload[4:]
    elif mutation == "marker":
        payload = b"BAD!" + payload[4:]
    elif mutation in ("nan", "inf"):
        payload = payload[:4] + struct.pack("<d", float(mutation)) + payload[12:]
    binary.write_bytes(payload)
    with pytest.raises(ValueError):
        tsfield.load(path)


def test_missing_binary_and_direct_array_validation(tmp_path):
    path = write_map(tmp_path)
    item = tsfield.load(path)
    for values in (np.zeros((2, 2, 2, 3)), np.full(item.B.shape, np.nan)):
        with pytest.raises(ValueError, match="field array"):
            tsfield.FieldMap(item.header, values)
    path.with_suffix(".bin").unlink()
    with pytest.raises(OSError):
        tsfield.load(path)
    with pytest.raises(TypeError):
        tsfield.FieldMapBank(inner=[path])


def test_npts_overflow_is_rejected_before_opening_binary(tmp_path):
    path = write_map(tmp_path)
    path.write_text(path.read_text().replace("nX=3", "nX=2000000000")
                    .replace("nY=4", "nY=2000000000"))
    path.with_suffix(".bin").unlink()
    with pytest.raises(ValueError, match="npts"):
        tsfield.load(path)


def test_cpu_module_import_does_not_require_jax():
    package_root = str(Path(__file__).resolve().parents[1])
    program = (
        "import builtins, sys\n"
        f"sys.path.insert(0, {package_root!r})\n"
        "original = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name == 'jax' or name.startswith('jax.'):\n"
        "        raise AssertionError('unexpected JAX dependency')\n"
        "    return original(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "from beamfastsim import tsfield\n"
        "assert not any(name == 'jax' or name.startswith('jax.') for name in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_audit_hashes_symmetry_and_coverage(tmp_path, capsys):
    path = write_map(tmp_path, mirror=True, order=">")
    assert tsfield.main(["--inner", str(path), "--point", "0", "-0.5", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    entry = report["inner"][0]
    assert report["backend"] == "numpy-cpu-reference"
    assert report["transport_validated"] is False
    assert entry["header"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["binary"]["sha256"] == hashlib.sha256(path.with_suffix(".bin").read_bytes()).hexdigest()
    assert entry["shape"] == [3, 4, 5, 3]
    assert entry["bounds_mm"] == {"lo": [0, 0, 0], "hi": [2.5, 1.5, 8]}
    assert entry["npts"] == 60
    assert entry["byte_order"] == "big"
    assert "flip=By" in entry["symmetry"]
    assert len(entry["coverage_boxes_mm"]) == 2
    assert report["points"]["covered"] == [True]
    np.testing.assert_allclose(report["points"]["field_tesla"], affine([[0, 0.5, 0]]) * [1, -1, 1])
    assert tsfield.main(["--outer", str(path), "--point", "1e6", "0", "0"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["points"]["covered"] == [False]
    assert report["points"]["field_tesla"] == [[None, None, None]]
    assert report["points"]["map_indices"] == [-1]
    assert tsfield.main(["--inner", str(path), "--inner", str(path)]) == 1
    assert "overlapping inner coverage" in capsys.readouterr().err
