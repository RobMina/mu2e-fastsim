"""Synthetic identity/energy provenance; no transport, remote data or model alteration."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mu2efastsim import tables as core_tables
from crvfastsim import tier1b
from valhist import denominator_provenance as P
from valhist import library_variants as V


def gen_frame(run=1, pdgs=(2112, 2212, 22), kes=(0.5, 70., 10.)):
    pdgs, kes = np.array(pdgs), np.array(kes)
    mass = pd.Series(abs(pdgs)).map(tier1b.MASS).fillna(0).to_numpy()
    n = len(pdgs)
    return pd.DataFrame(dict(run=np.full(n, run, dtype=np.int64), subrun=np.zeros(n, dtype=np.int64),
                             event=np.arange(n, dtype=np.int64) + 2 ** 40, idx=np.zeros(n, dtype=np.int64),
                             pdg=pdgs, genid=np.full(n, 7), e=mass + kes,
                             px=np.sqrt(kes * (kes + 2 * mass)), py=np.zeros(n), pz=np.zeros(n),
                             x=np.zeros(n), y=np.full(n, 15400.), z=np.zeros(n), t=np.zeros(n)))


def donor(path, run=1):
    gen = gen_frame(run)
    parts = gen[P.KEY + ["pdg"]].assign(key=1, parent=-1, isprim=1, genidx=0, ccode=1, scode=2,
                                       se=gen.e.astype(np.float32), spx=gen.px.astype(np.float32),
                                       spy=np.float32(0), spz=np.float32(0), st=np.float32(0),
                                       sx=gen.x, sy=gen.y, sz=gen.z)
    child = parts.iloc[[0]].assign(key=2, parent=1, isprim=0, genidx=-1, pdg=11)
    parts = pd.concat([parts, child], ignore_index=True)
    steps = gen[P.KEY].assign(sp=1)
    steps = pd.concat([steps, steps.iloc[[0]].assign(sp=2)], ignore_index=True)
    core_tables.write(str(path), dict(gen=gen, particles=parts, steps=steps), as_ttree=True)
    return gen, parts, steps


def append(path, name, frame):
    with uproot.update(path) as root:
        root.mktree("dump/" + name, {c: a.to_numpy().dtype for c, a in frame.items()})
        root["dump/" + name].extend({c: a.to_numpy() for c, a in frame.items()})


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "_code_identity", lambda: {"synthetic_builder": "frozen"})
    donors, gens = [], []
    for i in range(2):
        p = tmp_path / ("donor%d.root" % i)
        donor(p, i + 1)
        donors.append(str(p))
    for i in range(2):
        p = tmp_path / ("gen%d.root" % i)
        g = gen_frame(i + 10, (2112, 2212, 22, 13), (60., 70., 10., 100.))
        core_tables.write(str(p), {"gen": g}, as_ttree=True)
        gens.append(str(p))
    nominal = tmp_path / "nominal.json"
    nominal.write_bytes(V._json_bytes(dict(inputs=donors, gen=gens, generated_scaled_to=200)))
    nominal.with_suffix(".root").write_bytes(b"synthetic nominal")
    counters = tmp_path / "counters.txt"
    counters.write_text("synthetic geometry")
    V.plan(nominal, tmp_path / "plan", 100, bootstrap=1)

    def fake_build(paths, gen_paths, counters, generated, output):
        blocks, offset = [], 0
        for path in paths:
            frame, _ = P.donor_showers(path)
            blocks.append(frame[["shid", "apdg", "kebin", "ke"]].assign(shid=frame.shid + offset))
            offset += len(frame)
        showers = pd.concat(blocks, ignore_index=True)
        reach, nonmu = tier1b._reach(showers, gen_paths, generated)
        core_tables.write(str(output), dict(libshower=showers, libgen=reach), as_ttree=True)
        return dict(generated_scaled_to=generated, generated_nonmu=nonmu, files=len(paths))

    monkeypatch.setattr(V, "_build_library", fake_build)
    V.build(tmp_path / "plan/plan.json", "bootstrap_0001", counters, tmp_path / "base")
    return tmp_path / "base/library.json"


def test_energy_total_not_kinetic_and_precision():
    g = gen_frame()
    e = P.energy(g)
    np.testing.assert_allclose(e.ke, [0.5, 70., 10.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(e.ke, e.diagnostic_ke_from_p, atol=1e-12)
    np.testing.assert_allclose(e.inferred_mass[:2], [939.565, 938.272], atol=1e-10)
    assert not e.diagnostic_bin_disagreement.any()
    assert e.underflow.tolist() == [True, False, False]


def test_negative_ke_not_hidden_and_energy_convention_diagnostic():
    g = gen_frame()
    g.loc[0, "e"] = 50.
    e = P.energy(g)
    assert e.ke.iloc[0] < 0 and e.kebin.iloc[0] == 0 and e.unclipped_kebin.iloc[0] == -1
    assert e.negative_ke.iloc[0]
    assert e.energy_minus_on_shell.iloc[0] < -800
    assert e.inferred_mass.iloc[0] < 50
    assert abs(e.inferred_mass.iloc[0] - e.native_mass.iloc[0]) > 800


def test_bin_edges_exact_right_side_and_flow():
    values = np.r_[-1., 0., np.nextafter(tier1b.KE_EDGES[1], -np.inf), tier1b.KE_EDGES[1], 1e7]
    g = gen_frame(pdgs=(22,) * len(values), kes=(1.,) * len(values))
    g["e"] = values
    a = P.energy(g)
    assert a.kebin.tolist() == [0, 0, 0, 1, 17]
    assert a.unclipped_kebin.tolist() == [-1, -1, 0, 1, 18]


def test_unknown_mass_and_sign():
    g = gen_frame(pdgs=(-2212, 999999), kes=(70., 4.))
    a = P.energy(g)
    assert a.apdg.tolist() == [2212, 999999]
    assert a.mass_known.tolist() == [True, False]
    assert a.native_mass.iloc[1] == 0


def test_donor_identity_and_descendant_deduplication(tmp_path):
    path = tmp_path / "donor.root"
    donor(path)
    sh, info = P.donor_showers(path)
    assert len(sh) == 3 and sh.shid.tolist() == [0, 1, 2]
    assert sh.event.iloc[0] == 2 ** 40
    assert not sh.gen_join_missing.any() and not sh.pdg_disagreement.any()
    assert info["steps"]["rows"] == 4
    assert abs(sh.start_minus_gen_e.iloc[0]) < 1e-4
    assert sh.ke.iloc[0] == pytest.approx(0.5)


def test_pdg_join_mismatch_is_visible_not_repaired(tmp_path):
    path = tmp_path / "donor.root"
    g, _, _ = donor(path)
    g.loc[0, "pdg"] = 2212
    append(path, "gen", g)
    sh, info = P.donor_showers(path)
    assert info["gen"]["selected_cycle"] == 2
    assert sh.pdg_disagreement.iloc[0]
    assert sh.ke.iloc[0] != sh.gen_pdg_native_ke.iloc[0]


def test_missing_join_nan_retained_and_native_bin_reported(tmp_path):
    path = tmp_path / "donor.root"
    g, _, _ = donor(path)
    append(path, "gen", g.iloc[1:])
    sh, _ = P.donor_showers(path)
    assert sh.gen_join_missing.iloc[0] and sh.nonfinite.iloc[0]
    assert sh.kebin.iloc[0] == 17
    assert P.records(sh)[0]["ke"] is None
    V._json_bytes(P.records(sh))


@pytest.mark.parametrize("tree", ["gen", "particles"])
def test_duplicate_identity_refused(tmp_path, tree):
    path = tmp_path / "donor.root"
    g, p, _ = donor(path)
    frame = g if tree == "gen" else p
    append(path, tree, pd.concat([frame, frame.iloc[[0]]]))
    with pytest.raises(ValueError, match="ambiguous"):
        P.donor_showers(path)


def test_support_missing_vs_rounding_and_native_saturation():
    showers = pd.DataFrame(dict(apdg=[11, 22, 2112], kebin=[0, 0, 0]))
    blocks = [pd.DataFrame(dict(apdg=[22, 2112], kebin=[0, 0]))]
    existing = pd.DataFrame(dict(apdg=[22, 2112], kebin=[0, 0], ngen=[0, 0], nshower=[1, 1]))
    result = P.support(showers, blocks, 100, 1, existing)
    assert result.missing_raw_support.tolist() == [True, False, False]
    assert result.rounded_to_zero.tolist() == [False, True, True]
    existing.loc[0, "ngen"] = 1
    with pytest.raises(ValueError, match="libgen"):
        P.support(showers, blocks, 100, 1, existing)


def test_end_to_end_sealed_base_read_only(base, tmp_path):
    before = {str(p): V.sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "report.json"
    result = P.run(base, out)
    assert result["exact_shower_projection_closure"] and result["exact_libgen_closure"]
    missing = [r for r in result["support"] if r["missing_raw_support"]]
    assert [(r["apdg"], r["kebin"]) for r in missing] == [(2112, 0)]
    assert result["provenance_summary"]["all"]["occurrence_weighted_showers"] == result["shower_tree"]["rows"]
    assert result["provenance_summary"]["missing_support"]["occurrence_weighted_showers"] > 0
    assert result["external_persisted_events"] == 8
    assert result["selection"]["generated"] == 200
    assert all(V.sha256(p) == h for p, h in before.items())
    assert result == __import__("json").loads(out.read_text())
    with pytest.raises(FileExistsError):
        P.run(base, out)


def test_dangling_output_refused_before_input(tmp_path):
    out = tmp_path / "report.json"
    out.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        P.run(tmp_path / "absent.json", out)


def test_unsealed_base_refused(tmp_path):
    p = tmp_path / "fake.json"
    p.write_text("{}")
    with pytest.raises(FileNotFoundError):
        P.run(p, tmp_path / "report.json")


def test_source_drift_refuses_publication(base, tmp_path, monkeypatch):
    original = P.D._base
    calls = []

    def drift(path):
        result = original(path)
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("source drift")
        return result

    monkeypatch.setattr(P.D, "_base", drift)
    with pytest.raises(ValueError, match="drift"):
        P.run(base, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_real_bound_source_guard(base, tmp_path, monkeypatch):
    monkeypatch.setattr(V, "_code_identity", lambda: {"synthetic_builder": "unbound"})
    with pytest.raises(ValueError, match="build code changed"):
        P.run(base, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_shower_projection_mismatch_refused(base, tmp_path, monkeypatch):
    original = P.donor_showers

    def mismatch(path):
        sh, info = original(path)
        sh.loc[0, "ke"] += 1
        return sh, info

    monkeypatch.setattr(P, "donor_showers", mismatch)
    with pytest.raises(AssertionError):
        P.run(base, tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


def test_projection_matches_real_build_one_without_transport(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / "donor.root"
    g, parts, steps = donor(path)
    mu = gen_frame(1, (13,), (100.,)).assign(event=2 ** 40 + 3)
    g = pd.concat([g, mu], ignore_index=True)
    mp = parts.iloc[[0]].copy().assign(event=2 ** 40 + 3, pdg=13, se=mu.e.iloc[0], spx=mu.px.iloc[0])
    parts = pd.concat([parts, mp], ignore_index=True)
    steps = pd.concat([steps, mu[P.KEY].assign(sp=1)], ignore_index=True)
    steps = steps.assign(bar=0, t0=1., t1=2., x0=0., y0=0., z0=0., x1=1., y1=0., z1=0.,
                         px0=100., py0=0., pz0=0., p1=90., len=1., edep=1.)
    core_tables.write(str(path), dict(gen=g, particles=parts, steps=steps), as_ttree=True)
    geom = SimpleNamespace(sector=np.array([0]), center=np.zeros((1, 3)), local_to_world=np.array([[0, 1, 2]]))
    libbins = pd.DataFrame(dict(kind=[0, 1], group=[0, 0], edge=[50., 0.5]))
    actual, _ = tier1b._build_one([str(path)], geom, libbins)
    reconstructed, _ = P.donor_showers(path)
    pd.testing.assert_frame_equal(reconstructed[["shid", "apdg", "kebin", "ke"]], actual["libshower"],
                                  check_dtype=False, check_exact=True)


def test_identity_safe_json_and_nonfinite_diagnostic():
    g = gen_frame()
    g.loc[0, "event"] = 2 ** 60 + 7
    g.loc[0, "e"] = np.nan
    frame = P.energy(g)
    data = P.records(frame)
    assert data[0]["event"] == 2 ** 60 + 7
    assert data[0]["e"] is None and data[0]["nonfinite"]
    V._json_bytes(data)


def test_weighted_provenance_summary_and_unavailable_extrema():
    rows = pd.DataFrame(dict(apdg=[2112, 11, 22], kebin=[0, 5, 5], ke=[-20., 55., 60.],
        start_ke_native_mass=[80., 90., 60.], pdg_disagreement=[True, True, False],
        gen_join_missing=[False] * 3, negative_ke=[True, False, False]))
    donors = [dict(showers=P.records(rows), multiplicity=3),
              dict(showers=P.records(rows.iloc[[2]]), multiplicity=1)]
    cells = pd.DataFrame(dict(apdg=[2112], kebin=[0], missing_raw_support=[True]))
    got = P.provenance_summary(donors, cells)
    assert got['all']['unique_showers'] == 4
    assert got['all']['occurrence_weighted_showers'] == 10
    assert got['pdg_mismatch']['unique_showers'] == 2
    assert got['pdg_mismatch']['occurrence_weighted_showers'] == 6
    assert got['missing_support']['pdg_mismatch_weighted'] == 3
    rows = rows.iloc[[0]].assign(ke=np.nan, start_ke_native_mass=np.nan, gen_join_missing=True)
    got = P.provenance_summary([dict(showers=P.records(rows), multiplicity=2)], cells)
    assert got['all']['native_ke_min'] is None
    assert got['all']['diagnostic_start_ke_max'] is None
    V._json_bytes(got)
