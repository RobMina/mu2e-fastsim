"""Synthetic geometric truth diagnostics: no JAX, transport or production files needed."""
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import uproot

from beamfastsim import flash_ancestry as fa
from beamfastsim import flash_regions as fr


@pytest.fixture
def geom():
    beam = SimpleNamespace(axis_x=-100.0, foil_z=np.array([0.0, 5.0]),
                           foil_rin=np.array([2.0, 3.0]), foil_rout=np.array([8.0, 7.0]),
                           foil_ht=np.array([0.2, 0.4]), ipa_r=10.0, ipa_t=1.0,
                           ipa_z0=10.0, ipa_z1=20.0, opa_z0=30.0, opa_z1=40.0)
    return SimpleNamespace(beam=beam, opa_rin0=20.0, opa_rin1=30.0, opa_t=2.0,
                           ring_t=2.0, ring_hl=0.5)


def points(geom, rz):
    rz = np.asarray(rz, dtype=float).reshape(-1, 2)
    return np.column_stack((rz[:, 0] + geom.beam.axis_x, np.zeros(len(rz)), rz[:, 1]))


def test_axis_offset_annular_foils_and_upstream_other(geom):
    xyz = points(geom, [(4, 0), (1, 0), (8.1, 0), (4, 0.21), (4, 5.3), (2.5, 5), (4, -50)])
    assert fr.classify(geom, xyz).tolist() == ["foil", "other", "other", "other", "foil", "other", "other"]
    assert fr.classify(geom, [[-100, 4, 0], [4, 0, 0]]).tolist() == ["foil", "other"]
    assert fr.classify(geom, points(geom, [(2, 0.2), (8, -0.2)]), 0).tolist() == ["foil", "foil"]


def test_exact_foil_union_not_nearest_centre(geom):
    geom.beam.foil_z = np.array([0.0, 1.0])
    geom.beam.foil_ht = np.array([0.8, 0.05])
    assert fr.classify(geom, points(geom, [(4, 0.7)]), 0).tolist() == ["foil"]
    geom.beam.foil_ht[1] = 0.4
    assert fr.classify(geom, points(geom, [(4, 0.7)]), 0).tolist() == ["ambiguous"]


def test_ipa_shell_and_both_end_rings(geom):
    xyz = points(geom, [(10.5, 15), (9, 15), (11.1, 15), (10.5, 9),
                        (9, 10.5), (9, 20.5), (9, 19.5), (7.9, 10.5), (9, 21.1), (10, 10.5)])
    assert fr.classify(geom, xyz, 0).tolist() == ["ipa", "other", "other", "other", "ring", "ring", "other", "other", "other", "ambiguous"]
    geom.ring_t = 0
    assert fr.classify(geom, points(geom, [(10, 10.5), (9, 10.5)]), 0).tolist() == ["ipa", "other"]
    geom.beam.ipa_t = 0
    assert fr.classify(geom, points(geom, [(10, 15)]), 0).tolist() == ["other"]


def test_opa_linear_cone_and_z_limits(geom):
    xyz = points(geom, [(21, 30), (26, 35), (31, 40), (24.9, 35), (27.1, 35), (21, 29), (31, 41)])
    assert fr.classify(geom, xyz, 0).tolist() == ["opa", "opa", "opa", "other", "other", "other", "other"]
    geom.opa_t = 0
    assert fr.classify(geom, points(geom, [(26, 35)])).tolist() == ["other"]


def test_tolerance_overlap_is_not_precedence(geom):
    assert fr.classify(geom, points(geom, [(4, 0.2005)]), 0).tolist() == ["other"]
    assert fr.classify(geom, points(geom, [(4, 0.2005)])).tolist() == ["foil"]
    assert fr.classify(geom, points(geom, [(10.0005, 10.5)]), 0).tolist() == ["ipa"]
    assert fr.classify(geom, points(geom, [(10.0005, 10.5)])).tolist() == ["ambiguous"]
    geom.beam.opa_z0, geom.beam.opa_z1 = 10, 20
    geom.opa_rin0 = geom.opa_rin1 = 10
    assert fr.classify(geom, points(geom, [(10.5, 15)])).tolist() == ["ambiguous"]


def test_missing_birth_empty_and_invalid(geom):
    assert fr.classify(geom, [[np.nan, 0, 0], [0, np.inf, 0]]).tolist() == ["missing_birth", "missing_birth"]
    assert fr.classify(geom, np.empty((0, 3))).shape == (0,)
    for tolerance in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="tolerance"):
            fr.classify(geom, np.empty((0, 3)), tolerance)
    with pytest.raises(ValueError, match="shape"):
        fr.classify(geom, [])
    geom.beam.foil_ht = np.array([0.1])
    with pytest.raises(ValueError, match="foil arrays"):
        fr.classify(geom, np.empty((0, 3)))


def fixture_tables(geom):
    p = {name: np.zeros(7) for name in fa.PARTICLE}
    p.update(run=np.ones(7, dtype=int), subrun=np.ones(7, dtype=int), event=np.ones(7, dtype=int),
             key=np.arange(1, 8), parent=np.array([-1, 1, 2, 2, 4, -1, 6]),
             pdg=np.array([13, 22, 11, 22, 11, 22, 11]), genidx=np.array([0, -1, -1, -1, -1, 1, -1]),
             sx=geom.beam.axis_x + np.array([0, 4, 0, 10.5, 0, np.nan, 0]),
             sz=np.array([0, 0, 0, 15, 0, 0, 0]))
    s = {name: np.zeros(6) for name in fa.STEP}
    s.update(run=np.ones(6, dtype=int), subrun=np.array([1, 1, 1, 1, 1, 2]), event=np.ones(6, dtype=int),
             sp=np.array([3, 5, 1, 999, 7, 3]), eion=np.array([1., 2., 3., 4., 5., 6.]))
    g = {name: np.ones(2, dtype=int) for name in fa.GEN}
    g.update(idx=np.array([0, 1]), pdg=np.array([13, 2212]))
    return s, p, g


def write_root(path, tables):
    with uproot.recreate(path) as root:
        for name, table in zip(("sgsteps", "particles", "gen"), tables):
            root.mktree(f"dump/{name}", {key: value.dtype for key, value in table.items()}).extend(table)


def test_nearest_photon_missing_and_conserved_group_totals(geom):
    s, p, g = fixture_tables(geom)
    a = fa.trace_steps(s, p, g)
    assert a["photon_key"][:2].tolist() == [2, 4]
    report = fr.summarize(s, a, geom)
    by_region = report["by_region"]
    counts = dict(zip(by_region["region"], by_region["step_count"]))
    assert counts == dict(foil=1, ipa=1, ring=0, opa=0, other=0, ambiguous=0, missing_birth=1, no_photon=3)
    assert report["step_count"] == sum(by_region["step_count"]) == 6
    assert report["eion_sum_mev"] == sum(by_region["eion_sum_mev"]) == 21
    joint = report["by_subrun_region_root_genpdg"]
    assert sum(joint["step_count"]) == 6
    assert sum(joint["eion_sum_mev"]) == 21
    assert set(joint["root_gen_pdg"]) == {-1, 13, 2212}
    assert set(joint["subrun"]) == {1, 2}
    assert "normalized_totals" not in report
    assert "eion_mev_per_generated_event" not in by_region
    assert sum(report["photon_status"]["eion_sum_mev"]) == 21
    normalized = fr.summarize(s, a, geom, certified_generated_events=10)
    assert normalized["normalized_totals"] == dict(step_count_per_generated_event=0.6, eion_mev_per_generated_event=2.1)
    assert sum(normalized["by_region"]["eion_mev_per_generated_event"]) == pytest.approx(2.1)


def test_empty_conservation_and_bad_denominator_energy(geom):
    s, p, g = fixture_tables(geom)
    empty = [{key: value[:0] for key, value in table.items()} for table in (s, p, g)]
    a = fa.trace_steps(*empty)
    report = fr.summarize(empty[0], a, geom)
    assert report["step_count"] == report["eion_sum_mev"] == 0
    assert report["by_subrun_region_root_genpdg"]["region"] == []
    for denominator in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="certified"):
            fr.summarize(empty[0], a, geom, certified_generated_events=denominator)
    s["eion"][0] = np.nan
    with pytest.raises(ValueError, match="eion"):
        fr.summarize(s, fa.trace_steps(s, p, g), geom)


def test_root_highest_cycle_hash_and_cli(tmp_path, geom, monkeypatch):
    path = tmp_path / "sample.root"
    tables = fixture_tables(geom)
    tables[2]["event"][1] = 2
    write_root(path, tables)
    s = {key: value[:1] for key, value in tables[0].items()}
    with uproot.update(path) as root:
        root.mktree("dump/sgsteps", {key: value.dtype for key, value in s.items()}).extend(s)
    report = fr.analyze(path, geom)
    assert report["step_count"] == 1
    assert report["input"]["trees"]["sgsteps"] == dict(selected_key="dump/sgsteps;2", entries=1, cycles={"1": 6, "2": 1})
    assert len(report["input"]["sha256"]) == 64
    assert report["represented_any_tree_events"] == 2
    with pytest.raises(ValueError, match="smaller than represented"):
        fr.analyze(path, geom, certified_generated_events=1)
    monkeypatch.setattr(fr, "_load_geometry", lambda *args: (geom, {"source_files": ["test"]}))
    out = tmp_path / "report.json"
    assert fr.main(["--input", str(path), "--geom", "overlay", "--offline-root", "offline", "--out", str(out), "--generated-events", "10"]) == 0
    saved = json.loads(out.read_text())
    assert saved["normalized_totals"]["eion_mev_per_generated_event"] == 0.1
    assert saved["geometry"]["source_files"] == ["test"]


def test_geometry_reader_include_hashes(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from beamfastsim import cechain, simpleconfig

    @dataclass
    class TestGeometry:
        beam: dict

    base = tmp_path / "base.txt"
    base.write_text("double x = 1;\n")
    overlay = tmp_path / "overlay.txt"
    overlay.write_text('#include "base.txt"\ndouble x = 2;\n')
    sentinel = TestGeometry({"foil_z": np.array([1., 2.])})
    calls = []

    def from_offline(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(cechain, "from_offline", from_offline)
    loaded, provenance = fr._load_geometry(str(overlay), str(tmp_path))
    assert loaded is sentinel
    assert calls == [(str(overlay), str(tmp_path))]
    assert simpleconfig.read(overlay, tmp_path)["x"] == 2
    assert len(provenance["source_files"]) == 2
    assert all(len(item["sha256"]) == 64 for item in provenance["source_files"])
    assert provenance["parameters"]["beam"]["foil_z"] == [1., 2.]


def test_module_import_does_not_load_jax():
    result = subprocess.run([sys.executable, "-c", "import sys; from beamfastsim.flash_regions import classify; assert 'jax' not in sys.modules; assert 'beamfastsim.cechain' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_all_regions_joint_species_and_energy_conservation(geom):
    xyz = points(geom, [(4, 0), (10.5, 15), (9, 10.5), (26, 35),
                        (0, -100), (10, 10.5), (np.nan, 0), (4, 0)])
    n = len(xyz)
    s = dict(run=np.array([1, 1, 1, 1, 2, 2, 2, 2]), subrun=np.ones(n, dtype=int),
             eion=np.arange(1, n + 1, dtype=float))
    a = dict(photon_index=np.array([0, 0, 0, 0, 0, 0, 0, -1]),
             photon_sx=xyz[:, 0], photon_sy=xyz[:, 1], photon_sz=xyz[:, 2],
             root_gen_pdg=np.array([13, 13, 2212, 2212, -1, -1, -1, -1]),
             photon_status=np.array(["found"] * 7 + ["missing_particle"]))
    report = fr.summarize(s, a, geom, certified_generated_events=100)
    assert report["by_region"]["step_count"] == [1] * len(fr.REGIONS)
    assert report["by_region"]["eion_sum_mev"] == list(range(1, n + 1))
    assert report["eion_sum_mev"] == sum(report["by_subrun_region_root_genpdg"]["eion_sum_mev"]) == 36
    assert sum(report["by_subrun_region_root_genpdg"]["step_count"]) == n
    assert report["normalized_totals"]["eion_mev_per_generated_event"] == 0.36
    json.dumps(report, allow_nan=False)
