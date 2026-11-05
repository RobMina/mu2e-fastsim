"""Synthetic opt-in shared-boundary diagnostics; no transport or field assets."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from beamfastsim import tsfaces, tsfield


MATRIX = np.array([[2.0, -3.0, 0.5], [-1.0, 4.0, 2.0], [0.25, 2.0, -5.0]])
OFFSET = np.array([0.5, -1.0, 2.5])


def affine(points):
    return np.asarray(points) @ MATRIX.T + OFFSET


def loaded(tmp_path, name, *, lo=(0, 0, 0), n=(2, 3, 3), d=(1, 0.5, 0.5),
           shift=(0, 0, 0), mirror=False, field=affine, delta=(0, 0, 0)):
    """Write modern synthetic maps using the test_tsfield fixture layout."""
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
    values = field(lo + nodes * d) + np.asarray(delta)
    path.with_suffix(".bin").write_bytes(struct.pack("<I", tsfield.MARKER)
                                         + np.asarray(values, dtype="<f8").tobytes())
    return tsfield.load(path)


def pair(left, right, **kwargs):
    return tsfaces.audit([left, right], **kwargs)["pairs"][0]


def test_exact_face_closed_endpoints_sources_and_bank_unchanged(tmp_path):
    left = loaded(tmp_path, "left", shift=(-3904, 7, 7929))
    right = loaded(tmp_path, "right", lo=(1, 0, 0))
    receipt = tsfaces.audit([left, right])
    row = receipt["pairs"][0]
    assert row["kind"] == "face"
    assert row["samples"] == 9
    assert row["gate"] is True
    assert row["history_independent_within_atol"] is True
    assert row["intersection_closed"] is True
    assert row["intersections_mm"] == [{"lo": [1, 0, 0], "hi": [1, 1, 1],
                                         "closed": True, "kind": "face"}]
    for key in tsfaces._METRICS:
        assert row[key] == 0.0
    for source, item in zip(row["sources"], [left, right]):
        for key, path in [("header", item.header.path), ("binary", item.header.path.with_suffix(".bin"))]:
            assert source[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert source[key]["path"] == str(path.resolve())
        assert source["bounds_mm"] == {"lo": item.lo.tolist(), "hi": item.hi.tolist()}
    assert receipt["transport_validated"] is False
    assert receipt["stock_endpoint_equivalence"] is False
    assert receipt["inclusive_population_validated"] is False
    assert receipt["bank_policy_changed"] is False
    assert "approximate" in receipt["diagnostic"]
    assert "BFGridMap" in " ".join(receipt["limitations"])
    json.dumps(receipt, allow_nan=False)
    with pytest.raises(ValueError, match="shared boundaries"):
        tsfield.FieldMapBank(inner=[left, right])


def test_face_mismatch_statistics_and_componentwise_tolerance(tmp_path):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0, 0), delta=(3, 4, 0))
    row = pair(left, right)
    assert row["samples"] == 9
    assert row["max_abs_component_tesla"] == 4
    assert row["max_vector_difference_tesla"] == 5
    assert row["rms_component_difference_tesla"] == pytest.approx(np.sqrt(25 / 3))
    assert row["history_independent_within_atol"] is False
    assert row["gate"] is False
    assert pair(left, right, atol=4)["history_independent_within_atol"] is True
    assert pair(left, right, atol=np.nextafter(4.0, 0))["gate"] is False


@pytest.mark.parametrize("lo,kind,samples", [((1, 1, 0), "edge", 3), ((1, 1, 1), "corner", 1)])
def test_edge_and_corner_singleton_axes(tmp_path, lo, kind, samples):
    row = pair(loaded(tmp_path, "left"), loaded(tmp_path, "right", lo=lo), max_points=samples)
    assert row["kind"] == kind
    assert row["samples"] == samples
    assert row["gate"] is True
    assert row["max_abs_component_tesla"] == 0


def test_disjoint_uses_exact_boxes_including_reflection_gap(tmp_path, monkeypatch):
    mirror = loaded(tmp_path, "mirror", lo=(0, 2, 0), mirror=True)
    gap = loaded(tmp_path, "gap", lo=(0, -0.5, 0))
    far = loaded(tmp_path, "far", lo=(5, 0, 0))
    def forbidden(*args, **kwargs):
        raise AssertionError("disjoint maps must not be sampled")
    monkeypatch.setattr(tsfield.FieldMap, "field", forbidden)
    receipt = tsfaces.audit([mirror, gap, far], max_points=1)
    assert len(receipt["pairs"]) == 3
    assert receipt["gate"] is True
    for row in receipt["pairs"]:
        assert row["kind"] == "disjoint"
        assert row["samples"] == 0
        assert row["intersections_mm"] == []
        assert row["history_independent_within_atol"] is None


def test_volume_overlap_unsupported_without_field_evaluation(tmp_path, monkeypatch):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(0.5, 0, 0))
    def forbidden(*args, **kwargs):
        raise AssertionError("volume overlaps must not be sampled")
    monkeypatch.setattr(tsfield.FieldMap, "field", forbidden)
    row = pair(left, right, max_points=1)
    assert row["kind"] == "volume_overlap"
    assert row["supported"] is False
    assert row["gate"] is False
    assert row["samples"] == 0
    assert row["history_independent_within_atol"] is False
    assert row["max_abs_component_tesla"] is None


def symmetric(points):
    x, y, z = np.moveaxis(points, -1, 0)
    return np.stack((x + 2 * np.abs(y) + z, y, 2 * x - np.abs(y) + z), axis=-1)


def test_reflected_y_knots_flip_and_mu2e_shift(tmp_path):
    left = loaded(tmp_path, "mirror", lo=(0, 0.5, 0), n=(2, 4, 3), mirror=True,
                  shift=(10, 13, -8), field=symmetric)
    right = loaded(tmp_path, "negative", lo=(1, -2, 0), n=(2, 5, 3), d=(1, 0.375, 0.5),
                   field=symmetric)
    row = pair(left, right, atol=1e-13)
    assert row["kind"] == "face"
    assert row["intersections_mm"][0]["lo"] == [1, -2, 0]
    assert row["intersections_mm"][0]["hi"] == [1, -0.5, 1]
    assert row["samples"] == 21  # seven Y union knots, three Z knots
    assert row["gate"] is True
    assert row["sources"][0]["symmetry"].startswith("extendY flip=By")
    wrong_by = loaded(tmp_path, "wrong_by", lo=(1, -2, 0), n=(2, 5, 3), d=(1, 0.375, 0.5),
                      field=symmetric, delta=(0, 1, 0))
    assert pair(left, wrong_by)["max_abs_component_tesla"] == pytest.approx(1)


def test_both_reflected_maps_and_zero_seam_are_not_double_counted(tmp_path):
    left = loaded(tmp_path, "left", mirror=True, field=symmetric)
    right = loaded(tmp_path, "right", lo=(1, 0, 0), mirror=True, field=symmetric)
    row = pair(left, right, max_points=15)
    assert row["samples"] == 15  # five physical Y knots (including zero) times three Z
    assert len(row["intersections_mm"]) == 1
    assert row["gate"] is True
    assert row["componentwise_interpolant_bound_applicable"] is True


def test_nonzero_reflected_seam_explicitly_limits_bound_claim(tmp_path):
    left = loaded(tmp_path, "left", mirror=True)
    right = loaded(tmp_path, "right", lo=(1, 0, 0), mirror=True)
    row = pair(left, right)
    assert row["history_independent_within_atol"] is None
    assert row["sampled_agreement_within_atol"] is True
    assert row["gate"] is False
    assert row["reflection_seam_discontinuous"] is True
    assert row["componentwise_interpolant_bound_applicable"] is False
    assert "one-sided" in row["warning"]


def test_mixed_tangential_common_refinement_and_internal_mismatch(tmp_path):
    left = loaded(tmp_path, "left", n=(2, 4, 3), d=(1, 1, 0.5))
    right = loaded(tmp_path, "right", lo=(1, 0, 0), n=(2, 3, 5), d=(1, 1.5, 0.25))
    row = pair(left, right, atol=1e-13)
    assert row["samples"] == 25  # Y: 0,1,1.5,2,3; Z: 0,.25,.5,.75,1
    assert row["gate"] is True
    # End corners still agree, but a tangential knot inside the shared face differs.
    data = right.B.copy()
    data[0, 1, 2, 0] += 2
    perturbed = tsfield.FieldMap(right.header, data)
    row = pair(left, perturbed)
    assert row["samples"] == 25
    assert row["max_abs_component_tesla"] == 2
    assert row["gate"] is False


def test_partial_face_adds_intersection_endpoints_not_on_other_grid(tmp_path):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0.25, 0.125), n=(2, 2, 2), d=(1, 0.5, 0.5))
    row = pair(left, right, atol=1e-13)
    assert row["intersections_mm"][0] == {"lo": [1, 0.25, 0.125], "hi": [1, 0.75, 0.625],
                                            "kind": "face", "closed": True}
    assert row["samples"] == 9
    assert row["gate"] is True


def test_two_reflected_pieces_preserve_gap_and_cap_total_points(tmp_path):
    left = loaded(tmp_path, "mirror", lo=(0, 0.5, 0), mirror=True, field=symmetric)
    right = loaded(tmp_path, "right", lo=(1, -1.5, 0), n=(2, 7, 3), field=symmetric)
    row = pair(left, right)
    assert len(row["intersections_mm"]) == 2
    assert row["samples"] == 18
    assert row["gate"] is True
    with pytest.raises(ValueError, match="requires 18 points"):
        pair(left, right, max_points=17)


def test_atol_does_not_expand_geometric_intersection(tmp_path):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(np.nextafter(1.0, np.inf), 0, 0))
    row = pair(left, right, atol=100)
    assert row["kind"] == "disjoint"
    assert row["samples"] == 0


def test_cap_raises_before_meshgrid_no_truncation(tmp_path, monkeypatch):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0, 0))
    assert pair(left, right, max_points=9)["samples"] == 9
    def forbidden(*args, **kwargs):
        raise AssertionError("cap must be checked before meshgrid allocation")
    monkeypatch.setattr(np, "meshgrid", forbidden)
    with pytest.raises(ValueError, match="requires 9 points, exceeds max_points=8"):
        pair(left, right, max_points=8)


@pytest.mark.parametrize("atol", [-1, np.nan, np.inf, -np.inf, True, None, "0", [0]])
def test_invalid_atol(tmp_path, atol):
    with pytest.raises(ValueError, match="atol"):
        tsfaces.audit([loaded(tmp_path, "map")], atol=atol)


@pytest.mark.parametrize("cap", [0, -1, 1.5, np.nan, np.inf, True, None, "2"])
def test_invalid_max_points(tmp_path, cap):
    with pytest.raises(ValueError, match="max_points"):
        tsfaces.audit([loaded(tmp_path, "map")], max_points=cap)


def test_empty_wrong_duplicate_and_missing_sources(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        tsfaces.audit([])
    with pytest.raises(TypeError, match="FieldMap"):
        tsfaces.audit([Path("not-loaded.header")])
    item = loaded(tmp_path, "map")
    with pytest.raises(ValueError, match="distinct"):
        tsfaces.audit([item, item])
    alias_header = tsfield.GridHeader(item.header.path.parent / "." / item.header.path.name,
                                      item.header.grid_origin, item.header.origin_shift,
                                      item.n, item.d)
    with pytest.raises(ValueError, match="distinct"):
        tsfaces.audit([item, tsfield.FieldMap(alias_header, item.B)])
    receipt = tsfaces.audit([item], atol=np.float64(0), max_points=np.int64(1))
    assert receipt["gate"] is True
    assert receipt["pairs"] == []
    item.header.path.with_suffix(".bin").unlink()
    with pytest.raises(OSError):
        tsfaces.audit([item])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_sampled_fields_rejected(tmp_path, monkeypatch, bad):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0, 0))
    monkeypatch.setattr(tsfield.FieldMap, "field", lambda self, points: np.full(points.shape, bad))
    with pytest.raises(ValueError, match="sampled fields must be finite"):
        pair(left, right)


def test_finite_fields_with_overflowing_difference_fail_closed(tmp_path, monkeypatch):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0, 0))
    monkeypatch.setattr(tsfield.FieldMap, "field",
                        lambda self, points: np.full(points.shape, 1e308 if self is left else -1e308))
    with pytest.raises(ValueError, match="differences must be finite"):
        pair(left, right)


def test_cli_receipt_pair_summaries_and_exit_codes(tmp_path, capsys):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0, 0), delta=(0.25, 0, 0))
    out = tmp_path / "receipt" / "faces.json"
    argv = ["--inner", str(left.header.path), "--inner", str(right.header.path), "--out", str(out)]
    assert tsfaces.main(argv) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "face"
    assert summary["gate"] is False
    assert json.loads(out.read_text())["pairs"][0]["max_abs_component_tesla"] == 0.25
    assert tsfaces.main(argv + ["--atol", "0.25", "--max-points", "9"]) == 0
    assert json.loads(capsys.readouterr().out)["gate"] is True
    assert tsfaces.main(argv + ["--max-points", "8"]) == 1
    assert "no truncation" in capsys.readouterr().err
    assert tsfaces.main(argv + ["--atol", "nan"]) == 1
    assert "atol" in capsys.readouterr().err
    assert tsfaces.main(["--inner", str(left.header.path), "--inner", str(left.header.path),
                         "--out", str(out)]) == 1
    assert "distinct" in capsys.readouterr().err
    volume = loaded(tmp_path, "volume", lo=(0.5, 0, 0))
    assert tsfaces.main(["--inner", str(left.header.path), "--inner", str(volume.header.path),
                         "--out", str(out)]) == 2
    assert json.loads(capsys.readouterr().out)["kind"] == "volume_overlap"
    far = loaded(tmp_path, "far", lo=(4, 0, 0))
    assert tsfaces.main(["--inner", str(left.header.path), "--inner", str(far.header.path),
                         "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "disjoint"


def test_alias_output_cannot_destroy_map_sources(tmp_path):
    import os

    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0, 0))
    argv = ["--inner", str(left.header.path), "--inner", str(right.header.path)]
    for source in (left.header.path, left.header.path.with_suffix(".bin")):
        before = source.read_bytes()
        assert tsfaces.main(argv + ["--out", str(source)]) == 1
        assert source.read_bytes() == before
        for kind in ("symlink", "hardlink"):
            alias = tmp_path / f"{source.suffix[1:]}-{kind}.json"
            if kind == "symlink":
                alias.symlink_to(source)
            else:
                os.link(source, alias)
            assert tsfaces.main(argv + ["--out", str(alias)]) == 1
            assert source.read_bytes() == before


def test_evaluated_array_hash_is_separate_from_unchanged_disk_hash(tmp_path):
    left = loaded(tmp_path, "left")
    right = loaded(tmp_path, "right", lo=(1, 0, 0))
    original = pair(left, right)
    modified = tsfield.FieldMap(right.header, right.B + [1.0, 0, 0])
    changed = pair(left, modified)
    assert original["sources"][1]["binary"] == changed["sources"][1]["binary"]
    assert original["sources"][1]["evaluated_array"]["sha256"] != changed["sources"][1]["evaluated_array"]["sha256"]
    assert changed["sources"][1]["evaluated_grid"]["lo"] == right.lo.tolist()


def test_mirrored_nonmirrored_seam_sample_agreement_is_not_gate(tmp_path):
    def constant_by(points):
        result = np.zeros_like(points, dtype=float)
        result[..., 1] = 1
        return result

    def neighbor_by(points):
        result = np.zeros_like(points, dtype=float)
        result[..., 1] = np.where(points[..., 1] == 0, 1, -1)
        return result

    left = loaded(tmp_path, "mirrored", mirror=True, n=(2, 3, 3), field=constant_by)
    right = loaded(tmp_path, "neighbor", lo=(1, -1, 0), field=neighbor_by)
    row = pair(left, right)
    assert row["max_abs_component_tesla"] == 0
    assert row["sampled_agreement_within_atol"] is True
    assert row["history_independent_within_atol"] is None
    assert row["gate"] is False
    query = np.array([[1, -1e-6, 0.5]])
    assert np.max(np.abs(left.field(query) - right.field(query))) > 1.99
