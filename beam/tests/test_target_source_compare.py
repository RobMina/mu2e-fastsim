"""Source histogram contracts, including flow bins and sparse shape tests."""
import numpy as np
import pytest

from beamfastsim.target_source_compare import histogram_compare


def test_identical_shape_and_scaled_exposure_are_null():
    a = np.array([1, 100, 120, 1])
    r = histogram_compare(a, a * 3, np.array([0., 1., 2.]))
    assert r['chi2'] == 0 and r['pvalue'] == 1
    assert r['max_cdf_difference'] == 0
    assert r['flow_stock'] == [1, 1] and r['flow_fast'] == [3, 3]


def test_shape_difference_is_detected_without_discarding_flows():
    r = histogram_compare(np.array([200, 200, 200, 200]), np.array([0, 400, 400, 0]), np.array([0., 1., 2.]))
    assert r['chi2'] > 100 and r['pvalue'] < 1e-10
    assert r['max_cdf_difference'] == 0.25
    with pytest.raises(ValueError):
        histogram_compare(np.zeros(4), np.ones(4), np.arange(3))


def test_sparse_histogram_has_no_spurious_degrees_of_freedom():
    r = histogram_compare(np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([0., 1.]))
    assert r['ndf'] == 0 and r['pvalue'] is None


def test_rpc_histogram_directory_and_pair_marginals(tmp_path):
    uproot = pytest.importorskip('uproot')
    from beamfastsim.target_sources import internal_pair
    from beamfastsim.target_source_compare import compare
    a = internal_pair(np.random.default_rng(17), np.full(3000, 100.))
    a.update(x=np.zeros(3000), y=np.zeros(3000), z=np.full(3000, 6000.), time=np.full(3000, 400.))
    source, stock = tmp_path / 'source.npz', tmp_path / 'stock.root'
    np.savez(source, **a)
    ee, ep = np.hypot(a['p'], 0.510998910), np.hypot(a['positron_p'], 0.510998910)
    q = dict(hmomentum=ee + ep, hElecMom=a['p'], hPosiMom=a['positron_p'], hMee=a['pair_mass'],
             hMeeOverE=a['pair_mass'] / (ee + ep), hy=-a['asymmetry'], hTime=a['time'], hStopZ=a['z'])
    for axis, k in enumerate(('x', 'y', 'z')):
        q['hElecP' + k] = a['p'] * a['u'][:, axis]
        q['hPosiP' + k] = a['positron_p'] * a['positron_u'][:, axis]
    with uproot.recreate(stock) as root:
        for k, v in q.items():
            lo, hi = v.min(), v.max()
            edges = np.linspace(lo - 0.01, hi + 0.01, 21)
            root['generate/RPCGun/' + k] = np.histogram(v, edges)
    r = compare(stock, source)
    assert r['source_marginal_test_passed'] and not r['transport_validated']
    assert len(r['kinematics']) == 12
    assert r['repeated_parent_stop_diagnostics']['hTime']['pvalue'] is None


def test_dio_source_only_tree_marginals(tmp_path):
    uproot = pytest.importorskip('uproot')
    from beamfastsim.target_source_compare import compare_dio
    from beamfastsim.cechain import isotropic
    rng = np.random.default_rng(19)
    p, u = rng.uniform(95., 105., 4000), isotropic(rng, 4000)
    source, stock = tmp_path / 'source.npz', tmp_path / 'stock.root'
    np.savez(source, p=p, u=u)
    a = dict(energy=np.hypot(p, 0.510998910), mom=p, momT=p * np.hypot(u[:, 0], u[:, 1]), cosTheta=u[:, 2])
    with uproot.recreate(stock) as root:
        root.mktree('GenFilter/GenAna', {k: 'float64' for k in a}).extend(a)
    r = compare_dio(stock, source)
    assert r['stock_electrons'] == 4000 and r['source_marginal_test_passed']
    assert len(r['kinematics']) == 4 and not r['transport_validated']
