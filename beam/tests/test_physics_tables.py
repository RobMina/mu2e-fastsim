"""Synthetic ASCII-format/unit and deterministic reference tests; no collaboration inputs."""
from pathlib import Path

import numpy as np
import pytest

from beamfastsim import physics_tables as pt, rangestudy


def _table(path, vectors):
    pieces = [str(len(vectors))]
    for energy, values in vectors:
        pieces += ["2", f"{energy[0]} {energy[-1]} {len(energy)}", str(len(energy))]
        pieces += [f"{x:.17g} {y:.17g}" for x, y in zip(energy, values)]
    path.write_text("\n".join(pieces) + "\n")


def _export(tmp_path):
    (tmp_path / "couple.dat").write_text(
        "COUPLE-V3.0\n2\n0\nG4_AIR\nDefaultRegionForTheWorld\n.01\n.01\n.01\n0\n"
        "1\nStoppingTarget_Al\nDefaultRegionForTheWorld\n.01\n.01\n.01\n0\n")
    (tmp_path / "material.dat").write_text("MATERIAL-V3.0\n2\nG4_AIR .0012\nStoppingTarget_Al 2.699\n")
    (tmp_path / "cut.dat").write_text(
        "CUT-V3.0\n2\n.01 .99\n.01 .99\n.01 1\n.01 34.10575\n.01 1\n.01 33.72249\n0 0\n0 0\n")
    energy = np.geomspace(.001, 100., 101)
    for name in pt.TABLES:
        values = energy / 2 if name in ("Range", "CSDARange") else np.full(len(energy), 2.)
        _table(tmp_path / f"{name}.muIoni.mu-.asc", [(energy, values * .01), (energy, values)])
    return energy


def test_named_couple_and_cut_units(tmp_path):
    energy = _export(tmp_path)
    tables, info, paths = pt.load_target(tmp_path)
    assert info["index"] == 1 and info["density_g_cm3"] == 2.699
    assert info["energy_cuts_MeV"]["e-"] == pytest.approx(.03410575)
    assert info["range_cuts_mm"]["e-"] == .01
    np.testing.assert_array_equal(tables["DEDX"]["energy"], energy)
    np.testing.assert_array_equal(tables["DEDX"]["values"], np.full(len(energy), 2.))
    assert len(paths) == 8
    grid, ranges = pt.integrated_range(tables)
    np.testing.assert_allclose(ranges, (grid - .1) / 2, atol=1e-12)
    summary = pt.summarize(tables, info)
    assert summary["range_consistency"]["Range"]["max_abs_mm"] < 1e-10


def test_not_a_knot_cubic_and_domain_guard():
    energy = np.geomspace(.1, 10., 12)
    vector = dict(energy=energy, values=1 + energy + energy**3)
    query = np.geomspace(.1, 10., 83)
    np.testing.assert_allclose(pt.interpolate(vector, query), 1 + query + query**3, rtol=2e-14)
    for value in (.099, 10.01, np.nan):
        with pytest.raises(ValueError, match="domain"):
            pt.interpolate(vector, value)


@pytest.mark.parametrize("change", ["count", "index", "cuts", "material", "empty", "unordered", "trailing"])
def test_malformed_mapping_or_vector_fails(tmp_path, change):
    energy = _export(tmp_path)
    if change == "count":
        path = tmp_path / "cut.dat"
        path.write_text(path.read_text().replace("CUT-V3.0\n2", "CUT-V3.0\n3"))
    elif change == "index":
        path = tmp_path / "couple.dat"
        path.write_text(path.read_text().replace("\n1\nStoppingTarget", "\n9\nStoppingTarget"))
    elif change == "cuts":
        path = tmp_path / "cut.dat"
        path.write_text(path.read_text().replace(".01 34.10575", ".02 34.10575"))
    elif change == "material":
        path = tmp_path / "couple.dat"
        path.write_text(path.read_text().replace("StoppingTarget_Al", "Other_Al"))
    elif change == "empty":
        _table(tmp_path / "DEDX.muIoni.mu-.asc", [(energy, energy), (energy, np.zeros(len(energy)))])
    elif change == "unordered":
        energy[10] = energy[9]
        _table(tmp_path / "DEDX.muIoni.mu-.asc", [(energy, energy), (energy, energy)])
    else:
        path = tmp_path / "DEDX.muIoni.mu-.asc"
        path.write_text(path.read_text() + "garbage\n")
    with pytest.raises(ValueError):
        pt.load_target(tmp_path)


def test_range_report_uses_same_frozen_rows(tmp_path):
    _export(tmp_path)
    tables, info, _ = pt.load_target(tmp_path)
    frozen = dict(row=np.array([7, 2, 19]), entry_MeV=np.array([.09, 2., 3.]), length_mm=np.array([.2, 2., .1]))
    report, stored = rangestudy.compare_integrators(frozen, tables, info, fractions=(.1, .01))
    assert report["models"]["g4_unrestricted"]["stop_rows"] == [7, 2]
    assert report["g4_native_range_check"]["g4_unrestricted"]["disagree_integral"] == 0
    assert report["adaptive"]["g4_unrestricted"]["0.01"]["disagree_integral"] == 0
    np.testing.assert_array_equal(stored["row"], frozen["row"])
    for fraction in (.1, .01):
        assert report["adaptive"]["g4_unrestricted"][f"{fraction:g}"]["maximum_fraction"] <= fraction * (1 + 1e-14)


def test_reject_trailing_cut_garbage_and_bad_density(tmp_path):
    _export(tmp_path)
    path = tmp_path / "cut.dat"
    path.write_text(path.read_text() + "garbage\n")
    with pytest.raises(ValueError):
        pt.load_target(tmp_path)
    _export(tmp_path)
    path = tmp_path / "material.dat"
    path.write_text(path.read_text().replace("2.699", "nan"))
    with pytest.raises(ValueError, match="density"):
        pt.load_target(tmp_path)


def test_mixed_charge_entries_are_not_compared_to_one_table():
    frozen = dict(row=np.array([2, 0, 1]), entry_MeV=np.array([1., 2., 3.]), length_mm=np.ones(3))
    g = dict(pdg=np.array([13, -13, 13]), foil=np.array([0, 0, -1]))
    selected, info = rangestudy.negative_entries(frozen, g)
    assert selected["row"].tolist() == [2, 0]
    assert selected["entry_MeV"].tolist() == [1., 2.]
    assert info["positive_excluded"] == 1 and info["positive_g4_first"] == 1
    g["pdg"][0] = 11
    with pytest.raises(ValueError, match="non-muon"):
        rangestudy.negative_entries(frozen, g)
