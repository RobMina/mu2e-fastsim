"""Production reuse cannot upgrade a selected S1 sample to full generated-event evidence."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crvfastsim.analyses import production_reuse as P


def fixture_events():
    u = pd.DataFrame({"run": [1] * 4, "subrun": [2] * 4, "event": [1, 2, 3, 4]})
    x = pd.DataFrame({"run": [1] * 3, "subrun": [2] * 3, "event": [1, 1, 2],
                      "len": [10., 20., 30.], "edep": [1., 2., 4.]})
    return u, x


def test_event_moments_include_zeros_and_within_event_correlation():
    u, x = fixture_events()
    m = P.event_moments(x, u)
    v = np.array([[2., 1., 0., 0.], [30., 30., 0., 0.], [3., 4., 0., 0.]])
    np.testing.assert_array_equal(m["totals"], v.sum(1))
    np.testing.assert_array_equal(m["cross"], v @ v.T)
    p = P.precision(m)
    np.testing.assert_allclose(p["rate"]["single_arm_relative_se"], v[0].std(ddof=1) / 2 / v[0].mean())
    assert m["contributing_events"] == 2


def test_length_mean_cluster_se_matches_centered_influence():
    u, x = fixture_events()
    p = P.precision(P.event_moments(x, u))
    influence = np.array([30., 30., 0., 0.]) / 60 - np.array([2., 1., 0., 0.]) / 3
    expected = np.sqrt(4 / 3 * (influence @ influence))
    assert p["length_mean"]["single_arm_relative_se"] == pytest.approx(expected)
    assert p["length_mean"]["equal_distribution_independent_ratio_se"]["20"] == pytest.approx(expected * np.sqrt(1.05))
    assert p["length_mean"]["infinite_reference_ratio_se"] == pytest.approx(expected)


def test_disjoint_merge_equals_whole_event_moments():
    u, x = fixture_events()
    a = P.event_moments(x[x.event <= 2], u[u.event <= 2])
    b = P.event_moments(x[x.event > 2], u[u.event > 2])
    assert P.merge_moments(a, b) == P.event_moments(x, u)


def test_out_of_universe_crossing_refused():
    u, x = fixture_events()
    with pytest.raises(ValueError, match="outside"):
        P.event_moments(x, u[u.event != 1])
    with pytest.raises(ValueError, match="duplicate"):
        P.event_moments(x, pd.concat([u, u]))


def test_empty_sector_unsupported_not_zero_error():
    u, x = fixture_events()
    assert not P.precision(P.event_moments(x.iloc[:0], u))["supported"]


def sample_log():
    return ("output sim.test.art\nTrigReport Events total = 2500000 passed = 106796 failed = 2393204\n"
            "Total live time simulated: 29.5618\n")


def test_original_log_binds_generated_and_retained_normalization():
    m = {"file_name": "sim.test.art", "event_count": 106796, "dh.gencount": 106796}
    got = P.normalization(sample_log(), m)
    assert got["generated_log"] == 2500000
    assert got["retained_log"] == 106796
    assert not got["catalog_gencount_is_generated"]
    assert got["generated_livetime_log_s"] == 29.5618


@pytest.mark.parametrize("change", ["count", "name", "ambiguous", "missing_live"])
def test_normalization_fails_closed(change):
    m = {"file_name": "sim.test.art", "event_count": 106796, "dh.gencount": 106796}
    log = sample_log()
    if change == "count":
        m["event_count"] += 1
    elif change == "name":
        m["file_name"] = "other.art"
    elif change == "ambiguous":
        log += sample_log()
    else:
        log = log.replace("Total live time simulated: 29.5618", "")
    with pytest.raises(ValueError):
        P.normalization(log, m)


def test_s1_conversion_config_has_no_transport_or_false_gen_stream():
    path = Path(__file__).parents[1] / "art/CrvFastSim/fcl/dump_run2_s1.fcl"
    text = path.read_text()
    assert "CrvStepsFromStepPointMCs" in text
    assert 'stepPointsModuleLabels : [ "cosmicFilter" ]' in text
    assert 'simParticles : "cosmicFilter"' in text
    assert "fullGeneratorStream : false" in text
    assert "g4run" not in text and "Mu2eG4" not in text


def test_production_dump_is_census_bound_and_conditional(tmp_path, monkeypatch):
    import uproot
    from crvfastsim import tables
    u, x = fixture_events()
    steps = x.assign(bar=0, sp=1, t0=0.)
    parts = u.assign(key=1, pdg=13, isprim=1)
    gen = u.assign(idx=0)
    census = u.assign(n_gen=1, n_particles=1, n_steps=[2, 1, 0, 0],
                      associations_checked=1, full_generator_stream=0)
    subruns = pd.DataFrame({"run": [1], "subrun": [2], "livetime": [10.]})
    path = tmp_path / "s1.root"
    tables.write(path, steps, parts, gen, subruns)
    with uproot.update(path) as f:
        f["dump/events"] = {c: census[c].to_numpy() for c in census}
    geom = type("Geometry", (), {"sector": np.array([0])})()
    rc = type("Config", (), {"geometry": lambda self: geom})()
    monkeypatch.setattr(P.runconfig, "get", lambda *a: rc)
    monkeypatch.setattr(P.VS, "groups_of", lambda *a: {"sector": [0]})
    metadata = {"file_name": "test.art", "event_count": 4, "dh.gencount": 4}
    log = "test.art\nTrigReport Events total = 100 passed = 4 failed = 96\nTotal live time simulated: 10\n"
    got = P.production_evidence(path, metadata, log)
    assert got["events"] == 4 and got["step_rows"] == 3
    assert "not an all-generated" in got["scope"]
    assert got["groups"]["sector"]["crossings"] == 2
    census["full_generator_stream"] = 1
    with uproot.update(path) as f:
        f["dump/events"] = {c: census[c].to_numpy() for c in census}
    with pytest.raises(ValueError, match="selected-event"):
        P.production_evidence(path, metadata, log)
    census["full_generator_stream"] = 0
    with uproot.update(path) as f:
        values = {c: census[c].to_numpy() for c in census}
        values["gen_tag"] = ["generate::Primary"] * len(census)
        schema = {c: v.dtype for c, v in values.items() if c != "gen_tag"}
        f.mktree("dump/events", dict(schema, gen_tag="string"))
        f["dump/events"].extend(values)
    got = P.production_evidence(path, metadata, log)
    assert got["events"] == 4 and got["persisted_event_cycles"]["dump/events;3"] == 4


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_invalid_crossing_value_refused(bad):
    u, x = fixture_events()
    x.loc[0, "len"] = bad
    with pytest.raises(ValueError, match="crossing value"):
        P.event_moments(x, u)


def test_log_world_mismatch_is_not_hidden_by_equal_target_box():
    a = "Ref. point: (-3904,20000,10171)\nTarget box: -10000, 3000, -5000, 5000, -5000, 21000\nWorld: -36786.6, 36786.6, -3530.2, 15365.4, -52264.6, 52264.6\n"
    b = a.replace("-36786.6, 36786.6", "-531787, 531787")
    got = P.geometry_log_comparison(a, b)
    assert got["Target box"]["equal"] and got["Ref. point"]["equal"]
    assert not got["World"]["equal"]
    with pytest.raises(ValueError, match="ambiguous"):
        P.geometry_log_comparison(a + a, b)


def test_projection_uses_generator_identity_not_crossing_position(monkeypatch):
    gen = pd.DataFrame({"run": [1] * 3, "subrun": [2] * 3, "event": [1, 2, 3], "idx": [0] * 3,
                        "pdg": [13] * 3, "x": [0., 5., 5.], "y": [5., 4., 5.], "z": [0.] * 3,
                        "px": [0., -1., 0.], "py": [-1., -1., -1.], "pz": [0.] * 3})
    gen = gen.assign(genid=1, e=100., t=0.)
    x = gen[["run", "subrun", "event"]].assign(genidx=0, bar=0, sector=0)
    monkeypatch.setattr(P.tables, "read", lambda *a: gen)
    monkeypatch.setattr(P.TV, "primary_crossings", lambda *a: x)
    monkeypatch.setattr(P.VS, "groups_of", lambda *a: {"test": [0]})
    got = P.projection_diagnostic("reference", gen, object(), [-1., 1., -1., 1., -1., 1.])
    assert got["generated_muons"] == 3
    assert got["generated_ray_misses_smaller_world"] == 1
    assert got["generated_ray_side_entry"] == 1
    assert got["groups"]["test"]["truth_muon_bar_crossings"] == 3


def test_compressed_gen_index_is_remapped_by_exact_source_values():
    fields = {"run": [1, 1], "subrun": [2, 2], "event": [3, 3], "idx": [0, 1],
              "pdg": [11, 13], "genid": [1, 1], "x": [0., 1.], "y": [2., 3.], "z": [4., 5.],
              "px": [0., 1.], "py": [0., -2.], "pz": [0., 3.], "e": [1., 110.], "t": [0., 0.]}
    full = pd.DataFrame(fields)
    retained = full.iloc[[1]].copy().assign(idx=0)
    got = P.generator_identity_map(retained, full)
    assert got.genidx.iloc[0] == 0 and got.source_genidx.iloc[0] == 1
    retained.loc[:, "e"] += .001
    with pytest.raises(ValueError, match="exact subset"):
        P.generator_identity_map(retained, full)


def test_unresolved_muon_source_identity_is_unsupported_not_a_repaired_join(monkeypatch):
    full = pd.DataFrame({"run": [1], "subrun": [2], "event": [3], "idx": [0], "pdg": [11],
                         "genid": [1], "x": [0.], "y": [5.], "z": [0.], "px": [0.], "py": [-1.],
                         "pz": [0.], "e": [1.], "t": [0.]})
    monkeypatch.setattr(P.tables, "read", lambda *a: full)
    monkeypatch.setattr(P.TV, "primary_crossings", lambda *a: full[["run", "subrun", "event"]].assign(genidx=0))
    got = P.projection_diagnostic("reference", full, object(), [-1., 1., -1., 1., -1., 1.])
    assert not got["supported"] and got["unresolved_crossings"] == 1
    assert "groups" not in got
