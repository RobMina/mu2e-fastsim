"""Counting-error and truth identity contracts for transport physics diagnostics."""
import json
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd
import uproot

from crvfastsim.analyses import transport_validation as V


def test_paired_error_includes_empty_events():
    universe = pd.DataFrame(dict(run=1, subrun=1, event=np.arange(4)))
    a = universe.iloc[[0, 0, 1]].copy()
    b = universe.iloc[[0, 1]].copy()
    out = V.paired_count(a, b, universe)
    assert out["relative"] == .5
    np.testing.assert_allclose(out["se"], np.sqrt(4 * .5 / 3) / 2)
    assert V.paired_count(a, a, universe)["se"] == 0.
    assert V.paired_count(a, b.iloc[:0], universe)["relative"] is None


def test_count_moments_merge_equals_full_sample():
    universe = pd.DataFrame(dict(run=1, subrun=1, event=np.arange(6)))
    a = universe.iloc[[0, 0, 1, 4]].copy()
    b = universe.iloc[[0, 1, 4, 5]].copy()
    full = V.paired_count(a, b, universe)
    first = V.paired_count(a[a.event < 3], b[b.event < 3], universe.iloc[:3])
    last = V.paired_count(a[a.event >= 3], b[b.event >= 3], universe.iloc[3:])
    assert V.merge_counts([first, last]) == full


def test_merge_rejects_different_diagnostics_or_bins(tmp_path):
    import json
    import pytest
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    row = {"run_config": "synthetic", "diagnostic_sha256": "one", "p_bin_edges_MeV": [0, 300, None]}
    a.write_text(json.dumps(row))
    b.write_text(json.dumps(dict(row, diagnostic_sha256="two")))
    with pytest.raises(ValueError, match="incompatible diagnostic"):
        V.merge([str(a), str(b)], tmp_path / "out.json")
    b.write_text(json.dumps(dict(row, p_bin_edges_MeV=[0, 600, None])))
    with pytest.raises(ValueError, match="momentum bins"):
        V.merge([str(a), str(b)], tmp_path / "out.json")


def test_cli_routes_transport_validation(monkeypatch):
    from crvfastsim import cli
    seen = []
    monkeypatch.setattr(V, "main", lambda argv: seen.extend(argv))
    cli.main(["transport-validation", "--summary", "example.json"])
    assert seen == ["--summary", "example.json"]


def test_highest_cycle_inventory(tmp_path):
    p = tmp_path / "tables.root"
    with uproot.recreate(p) as f:
        f.mktree("dump/gen", {k: "int32" for k in V.tables.KEY})
        f["dump/gen"].extend(dict(run=[1], subrun=[2], event=[3]))
    with uproot.update(p) as f:
        f.mktree("dump/gen", {k: "int32" for k in V.tables.KEY})
        f["dump/gen"].extend(dict(run=[1, 1], subrun=[2, 2], event=[3, 4]))
    r = V.audit(p)["trees"]["gen"]
    assert r == dict(cycles=[1, 2], selected_cycle=2, rows=2, unique_ids=2)


def test_multiple_primary_muons_remain_distinct(monkeypatch):
    gen = pd.DataFrame(dict(run=1, subrun=1, event=1, idx=[0, 2], pdg=[13, -13]))
    parts = pd.DataFrame(dict(run=1, subrun=1, event=1, key=[1, 2], isprim=1, pdg=[13, -13]))
    steps = pd.DataFrame(dict(run=1, subrun=1, event=1, sp=[1, 2], bar=0, t0=[0., 1.],
                              px0=0., py0=[-100., -200.], pz0=0., edep=2., len=10.))
    monkeypatch.setattr(V.tables, "read", lambda path, name: steps if name == "steps" else parts)
    from types import SimpleNamespace
    x = V.primary_crossings("synthetic", gen, SimpleNamespace(sector=np.array([0])), fast_format=True)
    assert len(x) == 2 and x.genidx.tolist() == [0, 2]
    np.testing.assert_array_equal(x.p, [100., 200.])
    import pytest
    with pytest.raises(ValueError, match="require genidx"):
        V.primary_crossings("synthetic", gen, SimpleNamespace(sector=np.array([0])))
    parts["key"] = [7, 99]
    parts["genidx"] = [2, 0]
    steps["sp"] = [99, 7]
    x = V.primary_crossings("synthetic", gen, SimpleNamespace(sector=np.array([0])))
    assert x.genidx.tolist() == [0, 2]
    np.testing.assert_array_equal(x.p, [100., 200.])


@pytest.fixture
def synthetic_audit(tmp_path, monkeypatch):
    """Tiny persisted tables: three generated events, two particles in one accepted event."""
    geom = SimpleNamespace(sector=np.array([0]), sector_names=["S"])
    rc = SimpleNamespace(geometry=lambda: geom, counters="synthetic",
                         counters_cfg=lambda: {"crs.sectorTypeS": 0})
    counters = tmp_path / "counters.txt"
    counters.write_text("synthetic geometry")
    monkeypatch.setattr(V.runconfig, "get", lambda name: rc)
    monkeypatch.setattr(V.runconfig, "sector_type_names", lambda geom, cfg: {0: "all"})
    monkeypatch.setattr(V.geometry, "find_counters", lambda name: counters)
    monkeypatch.setattr(V, "repo_state", lambda module: {"head": "test", "source_sha256": {"test": "one"}})

    def make(subrun=1):
        gen = pd.DataFrame(dict(run=1, subrun=subrun, event=[1, 1, 2, 3], idx=[0, 1, 0, 0],
                                pdg=[13, -13, 13, 13], px=0., py=-100., pz=0., x=0., y=0., z=0., t=0.))
        parts = pd.DataFrame(dict(run=1, subrun=subrun, event=1, key=[1, 2], genidx=[0, 1],
                                  isprim=1, pdg=[13, -13]))
        steps = pd.DataFrame(dict(run=1, subrun=subrun, event=1, sp=[1, 2], bar=0, t0=[0., 1.],
                                  px0=0., py0=[-100., -200.], pz0=0., edep=2., len=10.))
        paths = {arm: tmp_path / f"{arm}_{subrun}.root" for arm in ("gen", "reference", "baseline", "candidate")}
        V.tables.write(str(paths["gen"]), None, None, gen=gen, as_ttree=True)
        for arm in ("reference", "baseline", "candidate"):
            particles = parts if arm == "reference" else parts.drop(columns="genidx")
            V.tables.write(str(paths[arm]), steps, particles, gen=gen, as_ttree=True)
        return paths

    return make


def run_synthetic(paths, output, baseline=False, **kwargs):
    return V.run(paths["gen"], paths["reference"], paths["baseline"] if baseline else None,
                 paths["candidate"], "synthetic", 3, output, **kwargs)


def test_no_baseline_run_and_summary(synthetic_audit, tmp_path, capsys):
    paths = synthetic_audit()
    paths["baseline"].unlink()  # No baseline file may be read or inventoried.
    output = tmp_path / "audit.json"
    out = run_synthetic(paths, output)
    assert set(out["inputs"]) == {"gen", "reference", "candidate"}
    assert out["persisted_reference_accepted_events"] == 1
    assert out["expected_accepted"] is None and out["accepted_count_gate"] is None
    assert "requires external job evidence" in out["reference_completeness"]
    for values in out["sectors"].values():
        assert set(values) == {"candidate_vs_reference"}
        assert values["candidate_vs_reference"]["a"] == 2
        assert values["candidate_vs_reference"]["moments"]["events"] == 3
    assert json.loads(output.read_text()) == out
    capsys.readouterr()
    V.show_summary(output)
    printed = capsys.readouterr().out
    assert "candidate_vs_reference" in printed and "baseline" not in printed
    assert "(truth diagnostics)" in printed and "matched" in printed and "edep/path" in printed
    assert "length 0.0 +/- None" in printed


def test_baseline_positional_api_preserved(synthetic_audit, tmp_path):
    out = run_synthetic(synthetic_audit(), tmp_path / "audit.json", baseline=True, expected_accepted=1)
    assert "baseline" in out["inputs"]
    assert set(out["sectors"]["S"]) == {"baseline_vs_reference", "candidate_vs_reference", "candidate_vs_baseline"}
    assert out["expected_accepted"] == 1 and out["accepted_count_gate"] is True
    assert "count agreement does not verify" in out["reference_completeness"]


@pytest.mark.parametrize("expected", [0, 2, 3])
def test_accepted_mismatch_fails_before_output(synthetic_audit, tmp_path, expected):
    output = tmp_path / "audit.json"
    with pytest.raises(ValueError, match=f"reference accepted event count mismatch: persisted 1, expected {expected}"):
        run_synthetic(synthetic_audit(), output, expected_accepted=expected)
    assert not output.exists()


@pytest.mark.parametrize("expected", [-1, 1.5, True])
def test_accepted_gate_rejects_invalid_count(expected):
    with pytest.raises(ValueError, match="nonnegative integer"):
        V.accepted_count_evidence(1, expected)


def test_zero_accepted_gate():
    assert V.accepted_count_evidence(0, 0)["accepted_count_gate"] is True


@pytest.mark.parametrize("gate", [True, False])
def test_no_baseline_merge_and_summary(synthetic_audit, tmp_path, capsys, gate):
    audits = []
    for subrun in (1, 2):
        output = tmp_path / f"audit_{subrun}.json"
        run_synthetic(synthetic_audit(subrun), output, expected_accepted=1 if gate or subrun == 1 else None)
        audits.append(str(output))
    merged_path = tmp_path / "merged.json"
    out = V.merge(audits, merged_path)
    assert out["persisted_generated_events"] == 6
    assert out["persisted_reference_accepted_events"] == 2
    assert out["expected_accepted"] == (2 if gate else None)
    assert out["accepted_count_gate"] == (True if gate else None)
    if not gate:
        assert "requires external job evidence" in out["reference_completeness"]
    assert set(out["inventory_totals"]) == {"gen", "reference", "candidate"}
    assert out["inventory_totals"]["reference"]["particles"]["unique_ids"] == 2
    for values in out["sectors"].values():
        assert set(values) == {"candidate_vs_reference"}
        assert values["candidate_vs_reference"]["a"] == 4
        assert values["candidate_vs_reference"]["se"] == 0.
        c = values["candidate_vs_reference"]
        assert c["length_mean_se"] == c["edep_mean_se"] == c["energy_per_path_se"] == 0.
        assert c["matched_only"]["paired_measure_moments"]["length_mean"]["events"] == 6
        assert c["identity_matched"] == 4
        assert c["entry_p_difference_MeV_quantiles"] is None
        assert c["entry_p_difference_MeV_shard_quantiles"] == [[0., 0., 0.], [0., 0., 0.]]
    capsys.readouterr()
    V.show_summary(merged_path)
    printed = capsys.readouterr().out
    assert "candidate_vs_reference" in printed and "baseline" not in printed
    assert "(truth diagnostics)" in printed and "matched" in printed and "edep/path" in printed
    assert "length 0.0 +/- 0.0" in printed


@pytest.mark.parametrize("reverse", [False, True])
def test_merge_rejects_different_comparison_sets(synthetic_audit, tmp_path, reverse):
    audits = []
    for subrun in (1, 2):
        output = tmp_path / f"audit_{subrun}.json"
        run_synthetic(synthetic_audit(subrun), output, baseline=subrun == 1)
        audits.append(str(output))
    output = tmp_path / "merged.json"
    with pytest.raises(ValueError, match="incompatible comparison sets"):
        V.merge(audits[::-1] if reverse else audits, output)
    assert not output.exists()


@pytest.mark.parametrize("baseline", [None, "old.root"])
def test_cli_optional_baseline_and_accepted_count(monkeypatch, baseline):
    calls = []
    monkeypatch.setattr(V, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    args = ["--gen", "gen.root", "--reference", "g4.root", "--candidate", "new.root",
            "--config", "synthetic", "--expected-events", "3", "--output", "audit.json",
            "--expected-accepted", "1"]
    if baseline is not None:
        args.extend(["--baseline", baseline])
    V.main(args)
    assert calls[0][0][:7] == ("gen.root", "g4.root", baseline, "new.root", "synthetic", 3, "audit.json")
    assert calls[0][1] == {"expected_accepted": 1}


@pytest.fixture
def paired_sample():
    """Two disjoint subruns, multi-primary clusters, selection differences and zero events."""
    universe = pd.DataFrame(dict(run=1, subrun=np.repeat([1, 2], 4), event=np.tile(np.arange(4), 2)))
    a = universe.iloc[[0, 0, 1, 4, 4, 6]].reset_index(drop=True)
    b = universe.iloc[[0, 0, 1, 2, 4, 5]].reset_index(drop=True)
    for frame in (a, b):
        frame['genidx'] = [0, 1, 0, 0, 1, 0]
        frame['bar'] = 0
        frame['p'] = 200.
    # Both primaries in event 0 match, but event 1 has different bars.
    b.loc[2, 'bar'] = 1
    b.loc[4, 'genidx'] = 0
    a['length'], b['length'] = [2., 5., 4., 9., 6., 3.], [3., 3., 7., 2., 5., 4.]
    a['edep'], b['edep'] = [1., 4., 2., 7., 2., 3.], [2., 1., 3., 2., 2., 1.]
    a['p'] += np.arange(len(a)) * 10
    return a, b, universe


def direct_event_ratio(a, b, universe, numerator, weight):
    """Independent centered influence-covariance calculation, not raw-moment subtraction."""
    arms = []
    for frame in (a, b):
        grouped = frame.groupby(V.tables.KEY)
        sums = grouped[numerator].sum()
        weights = grouped.size() if weight == 'count' else grouped[weight].sum()
        columns = pd.DataFrame({'sum': sums, 'weight': weights})
        arms.append(universe.merge(columns, on=V.tables.KEY, how='left')
                    [['sum', 'weight']].fillna(0.).to_numpy())
    x = np.column_stack(arms)
    sa, wa, sb, wb = x.sum(axis=0)
    ratio = (sa / wa) / (sb / wb)
    g = np.array([wb / (wa * sb), -ratio / wa, -ratio / sb, ratio / wb])
    influence = (x - x.mean(axis=0)) @ g
    se = np.sqrt(len(x) * np.var(influence, ddof=1))
    return ratio - 1, se, x


@pytest.mark.parametrize('name,numerator,weight', [
    ('length_mean', 'length', 'count'), ('edep_mean', 'edep', 'count'),
    ('energy_per_path', 'edep', 'length')])
@pytest.mark.parametrize('unequal_counts', [False, True])
def test_mean_errors_direct_event_influence(paired_sample, name, numerator, weight, unequal_counts):
    a, b, universe = paired_sample
    if unequal_counts:
        a = a.iloc[:-1]
    out = V.compare(a, b, universe)
    shift, error, x = direct_event_ratio(a, b, universe, numerator, weight)
    np.testing.assert_allclose([out[name + '_relative'], out[name + '_se']], [shift, error], rtol=1e-12)
    moments = out['paired_measure_moments'][name]
    assert moments['events'] == 8  # Includes two events empty in both arms.
    np.testing.assert_allclose(moments['totals'], x.sum(axis=0))
    np.testing.assert_allclose(moments['cross_products'], x.T @ x)
    if name == 'length_mean' and not unequal_counts:
        np.testing.assert_array_equal(x, [[7, 2, 6, 2], [4, 1, 7, 1], [0, 0, 2, 1], [0, 0, 0, 0],
                                         [15, 2, 5, 1], [0, 0, 4, 1], [3, 1, 0, 0], [0, 0, 0, 0]])
        # Independent-arm propagation drops the same-event A/B covariance and is wrong here.
        sa, wa, sb, wb = x.sum(axis=0)
        ratio = sa / wa / (sb / wb)
        g = np.array([wb / wa / sb, -ratio / wa, -ratio / sb, ratio / wb])
        covariance = np.cov(x, rowvar=False, ddof=1)
        unpaired = np.sqrt(len(x) * (g[:2] @ covariance[:2, :2] @ g[:2]
                                      + g[2:] @ covariance[2:, 2:] @ g[2:]))
        assert not np.isclose(error, unpaired)
    json.dumps(out, allow_nan=False)


def test_multiple_crossings_are_one_statistical_cluster():
    universe = pd.DataFrame(dict(run=1, subrun=1, event=np.arange(4)))
    a = universe.iloc[[0, 0, 1, 1]].copy()
    a['genidx'] = [0, 1, 0, 1]
    a['bar'], a['p'], a['length'], a['edep'] = 0, 200., [2., 2., 6., 6.], [1., 1., 3., 3.]
    b = a.assign(length=2., edep=1.)
    out = V.compare(a, b, universe)
    # Event influences [-.5, +.5, 0, 0]; crossing independence would underestimate the error.
    for name in ('length_mean', 'edep_mean'):
        assert out[name + '_relative'] == 1.
        assert out[name + '_se'] == pytest.approx(np.sqrt(2 / 3))
    assert out['energy_per_path_se'] == 0.


def test_identical_arms_zero_mean_errors(paired_sample):
    a, _, universe = paired_sample
    out = V.compare(a, a.sample(frac=1, random_state=1), universe)
    for component in (out, out['matched_only']):
        for name in ('length_mean', 'edep_mean', 'energy_per_path'):
            assert component[name + '_relative'] == 0.
            assert component[name + '_se'] == 0.
    assert out['a_only']['count'] == out['b_only']['count'] == 0
    assert out['identity_matched'] == len(a)


@pytest.mark.parametrize('empty', ['a', 'b', 'both', 'universe'])
def test_empty_mean_diagnostics_are_strict_json(paired_sample, empty):
    a, b, universe = paired_sample
    if empty in ('a', 'both', 'universe'):
        a = a.iloc[:0]
    if empty in ('b', 'both', 'universe'):
        b = b.iloc[:0]
    if empty == 'universe':
        universe = universe.iloc[:0]
    out = V.compare(a, b, universe)
    for component in (out, out['matched_only']):
        for name in ('length_mean', 'edep_mean', 'energy_per_path'):
            assert component[name + '_relative'] is None
            assert component[name + '_se'] is None
    assert out['identity_matched'] == 0
    assert out['entry_p_difference_MeV_quantiles'] is None
    json.dumps(out, allow_nan=False)


def test_only_one_contributing_event_does_not_claim_zero_uncertainty(paired_sample):
    a, b, universe = paired_sample
    out = V.compare(a.iloc[:2], b.iloc[:2], universe)
    assert out["length_mean_se"] is None
    assert out["paired_measure_moments"]["length_mean"]["contributing_events"] == 1
    assert out["paired_measure_moments"]["length_mean"]["events"] == len(universe)


def test_single_event_and_zero_denominator(paired_sample):
    a, b, universe = paired_sample
    a, b, universe = a.iloc[:2], b.iloc[:2], universe.iloc[:1]
    out = V.compare(a, b, universe)
    assert out['length_mean_relative'] == pytest.approx(7 / 6 - 1)
    assert out['length_mean_se'] is None
    zero = V.compare(a, b.assign(length=0., edep=0.), universe)
    assert zero['length_mean_relative'] is None
    assert zero['edep_mean_relative'] is None
    assert zero['energy_per_path_relative'] is None
    json.dumps(zero, allow_nan=False)


def test_matched_identity_selection_and_path_ratio(paired_sample):
    a, b, universe = paired_sample
    out = V.compare(a, b, universe)
    # Matching uses genidx AND bar, and does not confuse repeated event IDs across subruns.
    assert out['identity_matched'] == 3
    ma, mb = a.iloc[[0, 1, 3]], b.iloc[[0, 1, 4]]
    matched = out['matched_only']
    assert matched['a'] == matched['b'] == 3
    assert matched['relative'] == matched['se'] == 0.
    assert matched['crossings_per_generated_event'] == {'a': 3 / 8, 'b': 3 / 8}
    for name, numerator, weight in [('length_mean', 'length', 'count'), ('edep_mean', 'edep', 'count'),
                                     ('energy_per_path', 'edep', 'length')]:
        shift, error, _ = direct_event_ratio(ma, mb, universe, numerator, weight)
        np.testing.assert_allclose([matched[name + '_relative'], matched[name + '_se']], [shift, error])
    assert matched['energy_per_path_a'] == pytest.approx(ma.edep.sum() / ma.length.sum())
    assert not np.isclose(matched['energy_per_path_a'], (ma.edep / ma.length).mean())
    assert out['a_only'] == V.selection_summary(3, {'length': 13., 'edep': 7.})
    assert out['b_only'] == V.selection_summary(3, {'length': 13., 'edep': 6.})
    np.testing.assert_allclose(out['entry_p_difference_MeV_quantiles'], [2., 10., 26.])
    for arm in ('a', 'b'):
        assert out[arm] == matched[arm] + out[arm + '_only']['count']
        for col in ('length', 'edep'):
            assert out['sums'][col][arm] == matched['sums'][col][arm] + out[arm + '_only']['sums'][col]
    assert 'Truth diagnostics' in out['scope']
    with pytest.raises(pd.errors.MergeError):
        V.compare(pd.concat([a, a.iloc[:1]]), b, universe)


def test_disjoint_mean_diagnostic_merge_equals_pooled(paired_sample):
    a, b, universe = paired_sample
    pooled = V.compare(a, b, universe)
    shards = [V.compare(a[a.subrun == s], b[b.subrun == s], universe[universe.subrun == s]) for s in (1, 2)]
    merged = V.merge_counts(shards)
    assert merged.pop('entry_p_difference_MeV_shard_quantiles') == [
        r['entry_p_difference_MeV_quantiles'] for r in shards]
    assert merged.pop('entry_p_difference_MeV_quantiles') is None  # Cannot pool quantiles from quantiles.
    pooled.pop('entry_p_difference_MeV_quantiles')
    assert merged == pooled  # Includes raw moments, matched, unmatched, means, SEs and bins.
    assert merged['length_mean_se'] != np.mean([r['length_mean_se'] for r in shards])
    json.dumps(merged, allow_nan=False)



def test_nonempty_arms_without_common_identities(paired_sample):
    a, b, universe = paired_sample
    out = V.compare(a, b.assign(bar=b.bar + 10), universe)
    assert out['identity_matched'] == 0
    assert out['a_only']['count'] == len(a)
    assert out['b_only']['count'] == len(b)
    assert out['length_mean_se'] is not None
    assert out['matched_only']['length_mean_relative'] is None
    assert out['matched_only']['length_mean_se'] is None
    json.dumps(out, allow_nan=False)


def test_materially_negative_variance_fails_closed():
    x = np.array([[2., 1., 2., 1.], [4., 1., 4., 1.]])
    moments = V.ratio_moments(x, ["s_a", "n_a", "s_b", "n_b"])
    moments["cross_products"][0][0] = -1e6
    with pytest.raises(ValueError, match="materially negative"):
        V.ratio_summary(moments)


def test_negative_roundoff_is_clamped():
    x = np.array([[2., 1., 2., 1.], [4., 1., 4., 1.]])
    moments = V.ratio_moments(x, ["s_a", "n_a", "s_b", "n_b"])
    moments["cross_products"][0][0] = np.nextafter(moments["cross_products"][0][0], -np.inf)
    assert V.ratio_summary(moments)["se"] == 0.


def test_near_identical_arms_bound_raw_moment_cancellation():
    rng = np.random.default_rng(71)
    a = rng.uniform(1e4, 2e4, 1000)
    b = a + rng.normal(0., .1, len(a))
    x = np.column_stack([a, np.ones(len(a)), b, np.ones(len(a))])
    moments = V.ratio_moments(x, ["s_a", "n_a", "s_b", "n_b"])
    sa, wa, sb, wb = x.sum(axis=0)
    r = sa / wa / (sb / wb)
    g = np.array([wb / wa / sb, -r / wa, -r / sb, r / wb])
    direct = np.sqrt(len(x) / (len(x) - 1) * np.sum((x @ g) ** 2))
    assert V.ratio_summary(moments)["se"] == pytest.approx(direct, rel=1e-3)


def test_zero_numerator_is_supported(paired_sample):
    a, b, universe = paired_sample
    out = V.compare(a.assign(edep=0.), b, universe)
    assert out['edep_mean_relative'] == out['energy_per_path_relative'] == -1.
    assert out['edep_mean_se'] == out['energy_per_path_se'] == 0.


@pytest.mark.parametrize('contract,reason', [
    ('source', 'duplicate generator'), ('events', 'disjoint run/subrun'),
    ('physics', 'incompatible code'), ('geometry', 'incompatible code'),
    ('sectors', 'sector coverage')])
def test_merge_retains_provenance_contracts(synthetic_audit, tmp_path, contract, reason):
    paths = []
    rows = []
    for subrun in (1, 2):
        output = tmp_path / f'audit_{subrun}.json'
        rows.append(run_synthetic(synthetic_audit(subrun), output))
        paths.append(str(output))
    if contract == 'source':
        rows[1]['inputs']['gen']['sha256'] = rows[0]['inputs']['gen']['sha256']
    elif contract == 'events':
        rows[1]['event_ranges'] = rows[0]['event_ranges']
    elif contract == 'physics':
        rows[1]['code']['crv']['source_sha256']['test'] = 'different'
    elif contract == 'geometry':
        rows[1]['geometry'][0]['sha256'] = 'different'
    else:
        rows[1]['sectors'].pop('S')
    from pathlib import Path
    Path(paths[1]).write_text(json.dumps(rows[1], allow_nan=False))
    output = tmp_path / 'merged.json'
    with pytest.raises(ValueError, match=reason):
        V.merge(paths, output)
    assert not output.exists()
