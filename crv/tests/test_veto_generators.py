"""Retained generator ingestion preserves all species, source keys and total energies."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest

from crvfastsim.veto.generators import prepare
from crvfastsim.veto.primary_chain import rekey, run


def fixture():
    selection = ak.Array(dict(run=[2, 1, 1], subrun=[4, 4, 4], event=[5, 5, 6], shard=[0, 0, 0]))
    census = pd.DataFrame(dict(run=[1, 1, 2], subrun=[4, 4, 4], event=[5, 6, 5], n_gen=[2, 0, 1]))
    gen = pd.DataFrame(dict(run=[1, 1, 2], subrun=[4, 4, 4], event=[5, 5, 5],
        gen_product_id=[7, 7, 9], idx=[3, 8, 3], pdg=[13, -11, 2112], genid=[38, 38, 38],
        x=[1.123456789012, 2., 3.], y=[0., 0., 0.], z=[4., 5., 6.],
        px=[1., 2., 3.], py=[0., 0., 0.], pz=[0., 0., 0.], e=[106., 2.064, 940.], t=[7.123456789012, 8., 9.]))
    return selection, gen, census


def test_all_species_empty_events_and_precision():
    selection, raw, census = fixture()
    gen, report = prepare(selection, raw, census)
    assert report["primary_particles"] == 3 and report["primary_muons"] == 1
    assert report["nonmuon_primaries"] == 2 and report["empty_generator_events"] == 1
    pd.testing.assert_frame_equal(gen.drop(columns="shard"), raw)
    again, _ = prepare(selection, gen, census)
    pd.testing.assert_frame_equal(gen, again)
    mapped = rekey(selection, gen, "retained")
    assert list(mapped.event) == [1, 2, 2] and list(mapped.source_idx) == [3, 3, 8]
    assert list(mapped.source_run) == [2, 1, 1] and list(mapped.gen_product_id) == [9, 7, 7]
    assert list(mapped.e) == [940., 106., 2.064]
    assert mapped.x.iloc[1] == raw.x.iloc[0] and mapped.t.iloc[1] == raw.t.iloc[0]
    assert list(mapped.idx) == [0, 0, 1]
    assert mapped.index.is_unique


@pytest.mark.parametrize("failure", ["missing_event", "missing_generator", "duplicate", "nan", "bad_energy"])
def test_bad_or_incomplete_collection_rejected(failure):
    selection, gen, census = fixture()
    if failure == "missing_event":
        census = census.iloc[:2]
    elif failure == "missing_generator":
        gen = gen.iloc[:2]
    elif failure == "duplicate":
        gen = pd.concat([gen, gen.iloc[:1]])
    elif failure == "nan":
        gen.loc[0, "t"] = np.nan
    else:
        gen.loc[0, "e"] = 0.
    with pytest.raises(ValueError):
        prepare(selection, gen, census)


@pytest.mark.parametrize("table,field,value", [
    ("selection", "event", 5.5), ("selection", "run", np.nan),
    ("census", "subrun", -1), ("census", "n_gen", 2.5),
    ("gen", "event", 5.5), ("gen", "idx", -1), ("gen", "idx", 0.5),
    ("gen", "gen_product_id", 0), ("gen", "pdg", 13.5),
    ("gen", "e", np.inf), ("gen", "px", np.nan),
])
def test_malformed_identity_counts_and_states(table, field, value):
    selection, gen, census = fixture()
    if table == "selection":
        values = np.asarray(selection[field], dtype=float)
        values[0] = value
        selection = ak.with_field(selection, values, field)
    else:
        target = gen if table == "gen" else census
        target[field] = target[field].astype(float)
        target.loc[0, field] = value
    with pytest.raises(ValueError):
        prepare(selection, gen, census)


@pytest.mark.parametrize("which", ["selection", "census"])
def test_duplicate_event_identity(which):
    selection, gen, census = fixture()
    if which == "selection":
        selection = ak.concatenate([selection, selection[:1]])
    else:
        census = pd.concat([census, census.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        prepare(selection, gen, census)


def test_census_product_is_checked_when_present():
    selection, gen, census = fixture()
    census["gen_product_id"] = [7, 7, 9]
    prepare(selection, gen, census)
    gen.loc[0, "gen_product_id"] = 12
    with pytest.raises(ValueError, match="ProductID differs"):
        prepare(selection, gen, census)


def test_empty_census_still_requires_nonnull_advertised_product():
    selection, gen, census = fixture()
    census["n_gen"] = 0
    census["gen_product_id"] = [7, 0, 9]
    with pytest.raises(ValueError, match="null generator ProductID"):
        prepare(selection, gen.iloc[:0], census)


def test_out_of_grid_support_is_flagged_even_if_endpoint_cell_exists():
    from crvfastsim import tier1b
    from crvfastsim.veto.generators import shower_support
    gen = pd.DataFrame(dict(pdg=[22], e=[1e9]))
    cells = pd.DataFrame(dict(apdg=[22], kebin=[len(tier1b.KE_EDGES) - 2], ngen=[100], nshower=[10]))
    result = shower_support(gen, {"libgen": cells})
    assert result["status_counts"] == {"supported": 1}
    assert result["outside_energy_grid"] == 1 and result["incomplete_response_support"]


def test_identical_states_and_indices_across_products_are_distinct():
    selection, gen, census = fixture()
    gen.loc[1] = gen.loc[0]
    gen.loc[1, "gen_product_id"] = 8
    prepared, _ = prepare(selection, gen, census)
    assert len(prepared) == 3
    mapped = rekey(selection, prepared, "retained")
    assert list(mapped.gen_product_id) == [9, 7, 8]
    assert list(mapped.source_idx) == [3, 3, 3]
    assert list(mapped.idx) == [0, 0, 1]


@pytest.mark.parametrize("offset", [2**32, 2**53])
def test_exact_large_event_keys_survive_prepare_and_rekey(offset):
    selection, gen, census = fixture()
    for field in ("run", "subrun", "event"):
        selection = ak.with_field(selection, np.asarray(selection[field], dtype=np.int64) + offset, field)
        gen[field] += offset
        census[field] += offset
    gen["idx"] += offset
    gen["gen_product_id"] += offset
    prepared, _ = prepare(selection, gen, census)
    mapped = rekey(selection, prepared, "retained")
    np.testing.assert_array_equal(mapped.source_event, gen.event.iloc[[2, 0, 1]])
    np.testing.assert_array_equal(mapped.source_idx, gen.idx.iloc[[2, 0, 1]])
    assert mapped.source_event.dtype.kind in "iu"


def test_unsigned_identity_overflow_is_rejected():
    selection, gen, census = fixture()
    gen["idx"] = np.array([2**63, 8, 3], dtype=np.uint64)
    with pytest.raises(ValueError, match="64-bit"):
        prepare(selection, gen, census)


def test_all_empty_retained_collections_are_valid_inputs():
    selection, gen, census = fixture()
    census["n_gen"] = 0
    prepared, report = prepare(selection, gen.iloc[:0], census)
    assert report["empty_generator_events"] == 3 and len(prepared) == 0
    assert len(rekey(selection, prepared, "retained")) == 0


def test_missing_library_support_is_not_labelled_a_physical_zero():
    from crvfastsim import tier1b
    from crvfastsim.veto.generators import shower_support
    gen = pd.DataFrame(dict(pdg=[11, -11, 22, 2112, 999999], e=[11., 11., 10., 950., 10.]))
    cells = pd.DataFrame(dict(apdg=[11, 22, 2112],
        kebin=tier1b._ke_bin(np.array([11. - tier1b.MASS[11], 10., 950. - tier1b.MASS[2112]])),
        ngen=[100, 100, 0], nshower=[10, 0, 0]))
    result = shower_support(gen, {"libgen": cells})
    assert result["status_counts"] == dict(supported=2, zero_shower_cell=1, zero_denominator=1, unknown_mass=1)
    result = shower_support(gen, {"libgen": cells.iloc[1:]})
    assert result["status_counts"]["missing_cell"] == 2
    assert result["incomplete_response_support"]
    assert {r["pdg"] for r in result["cells"]} == {11, -11, 22, 2112, 999999}


def test_represented_route_does_not_accept_nonmuons():
    selection, gen, _ = fixture()
    with pytest.raises(ValueError, match="muons only"):
        rekey(selection, gen)
    mu = gen[gen.pdg == 13]
    from mu2efastsim.eloss import MMU
    represented = rekey(selection, mu)
    np.testing.assert_array_equal(represented.e, np.sqrt(mu.px**2 + mu.py**2 + mu.pz**2 + MMU**2))
    assert "source_idx" not in represented


def test_nonmuons_cannot_be_silently_ignored_without_library():
    selection, gen, census = fixture()
    with pytest.raises(ValueError, match="require the shower library"):
        run(selection, gen, "unused", "unused", "unused", input_kind="retained", census=census)
    with pytest.raises(ValueError, match="independent event census"):
        run(selection, gen, "unused", "unused", "unused", input_kind="retained")


def test_development_only_preparation_does_not_import_other_events():
    selection, gen, census = fixture()
    selected, report = prepare(selection[:1], gen, census)
    assert len(selected) == 1 and selected.pdg.iloc[0] == 2112
    assert report["selected_events"] == 1


def test_retained_run_sends_all_inputs_to_transport_and_sampler(monkeypatch):
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from crvfastsim import chain, readout, runconfig, transport, tier1b, tier1b_assets
    from crvfastsim.response import photons, sipm
    from crvfastsim.veto import primary_chain
    selection, gen, census = fixture()
    selection = ak.with_field(selection, np.zeros(len(selection)), "pbtime_mc")
    config = SimpleNamespace(jax=True, response=None, below_margin=0., below_msc=False,
        geometry=lambda: None, counters_cfg=lambda: None, chain_config=lambda **kw: SimpleNamespace(response=None),
        tagger_bar=lambda geom: None, bar_cut=lambda geom: None)
    monkeypatch.setattr(runconfig, "get", lambda name: config)
    monkeypatch.setattr(chain.Conditions, "from_csv", lambda *args: None)
    monkeypatch.setattr(readout, "apply", lambda *args: None)
    monkeypatch.setattr(transport, "load_world", lambda *a, **kw: (None, None))
    monkeypatch.setattr(transport, "load_below", lambda *a, **kw: None)
    monkeypatch.setattr(transport, "msc_plane", lambda geom: None)
    monkeypatch.setattr(photons, "load_tables", lambda response: {})
    monkeypatch.setattr(sipm, "photon_map", lambda: None)
    cells = pd.DataFrame(dict(apdg=[11, 2112], kebin=[0, 0], ngen=[10, 10], nshower=[0, 0]))
    monkeypatch.setattr(tier1b_assets, "load", lambda *args: {"libgen": cells})
    seen = {}
    def capture_transport(inputs, *args, **kwargs):
        seen["transport"] = inputs.copy()
        return pd.DataFrame(dict(event=np.zeros(0, dtype=np.int64), t0=[], t1=[])), pd.DataFrame()
    def capture_sampler(steps, parts, inputs, *args, **kwargs):
        seen["sampler"] = inputs.copy()
        assert kwargs["showers"] and kwargs["per_primary"]
        return steps, parts, {"shower_steps": 0}
    monkeypatch.setattr(transport, "transport", capture_transport)
    monkeypatch.setattr(tier1b, "sample", capture_sampler)
    monkeypatch.setattr(primary_chain, "onspill_charges", lambda *a: {"time": np.zeros(0)})
    fast, report = run(selection, gen, "unused", "unused", "unused", library_dir="library",
                       input_kind="retained", census=census)
    expected = rekey(selection, prepare(selection, gen, census)[0], "retained")
    for actual in seen.values():
        pd.testing.assert_frame_equal(actual, expected)
    assert set(seen) == {"transport", "sampler"}
    assert len(fast) == 3 and ak.sum(ak.num(fast.coincs)) == 0
    assert report["input_kind"] == "retained" and report["nonmuon_primaries"] == 2
    assert report["library_sampling"] == {"shower_steps": 0}
    assert report["inputs"]["shower_support"]["incomplete_response_support"]


def test_chain_cli_saves_source_mapping_and_census_requirement(tmp_path, monkeypatch):
    from crvfastsim.veto import primary_chain
    selection, gen, census = fixture()
    ak.to_parquet(selection, tmp_path / "selection.parquet")
    gen.to_parquet(tmp_path / "generators.parquet", index=False)
    census.to_parquet(tmp_path / "events.parquet", index=False)
    args = ["--selection", str(tmp_path / "selection.parquet"), "--primaries", str(tmp_path / "generators.parquet"),
            "--gdml", "unused", "--calib", "unused", "--deviation", "unused", "--outdir", str(tmp_path / "output"),
            "--input-kind", "retained"]
    with pytest.raises(SystemExit):
        primary_chain.main(args)
    def capture_run(selected, primaries, *args):
        assert len(selected) == 3 and len(primaries) == 3
        assert args[-2] == "retained"
        return selected, {}
    monkeypatch.setattr(primary_chain, "run", capture_run)
    primary_chain.main([*args, "--generator-events", str(tmp_path / "events.parquet")])
    mapped = pd.read_parquet(tmp_path / "output/generator_map.parquet")
    expected = rekey(selection, prepare(selection, gen, census)[0], "retained").reset_index(drop=True)
    pd.testing.assert_frame_equal(mapped, expected)


def test_preparation_cli_keeps_development_population_and_census(tmp_path):
    import json
    from crvfastsim.veto.generators import main
    selection, gen, census = fixture()
    selection = ak.with_field(selection, [0, 22, 0], "shard")
    ak.to_parquet(selection, tmp_path / "selection.parquet")
    gen.to_parquet(tmp_path / "generators.parquet", index=False)
    census.to_parquet(tmp_path / "events.parquet", index=False)
    main(["--selection", str(tmp_path / "selection.parquet"), "--generators", str(tmp_path / "generators.parquet"),
          "--events", str(tmp_path / "events.parquet"), "--outdir", str(tmp_path / "prepared")])
    result = json.loads((tmp_path / "prepared/inputs.json").read_text())
    assert result["selected_events"] == 2 and result["primary_particles"] == 1
    assert result["empty_generator_events"] == 1 and result["nonmuon_primaries"] == 1
    assert len(pd.read_parquet(tmp_path / "prepared/generator_events.parquet")) == 2
