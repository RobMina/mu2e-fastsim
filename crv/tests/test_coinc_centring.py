"""Reduced-precision hit centring in the coincidence kernel: per-group first-hit t/x/y subtracted on the host and
cluster times restored; checked here in float64, where it must reproduce the uncentred clusters."""
import numpy as np
import pytest

pytest.importorskip('jax')
from mu2efastsim import precision as dprec
from crvfastsim.reco import coinc as C
from crvfastsim.reco import coinc_jax as J

from test_coinc_attachment_jax import basic, geometry, pulses


def _clusters(rows, t_shift=0.0):
    p = pulses([(b, s, t + t_shift, pe, q) for b, s, t, pe, q in rows])
    return J.finish(J.find_async(np.zeros(len(p['idx']), int), *[p[k] for k in ('idx', 'bar', 'sipm', 'time', 'pe', 'seq')],
                                 geometry(), np.zeros(32, int), C.FinderConfig(), tiers=((16, 8, 2),), split=False,
                                 with_pos=True))


@pytest.mark.pr_params("87654.3")
@pytest.mark.parametrize('t_shift', [0.0, 87_654.3])
def test_centred_groups_give_the_same_clusters(monkeypatch, t_shift):
    rows = basic() + [(b, 0, 300. + 3 * b, 7., 0) for b in (1, 3, 5, 7)]
    ref = _clusters(rows, t_shift)
    monkeypatch.setattr(dprec, 'SINGLE', True)        # centring on, kernel still float64 (tests pin it)
    got = _clusters(rows, t_shift)
    assert J.cluster_keys(got) == J.cluster_keys(ref)
    for k in ('cl_pe', 'cl_start', 'cl_end', 'cl_slope', 'cl_avgtime', 'cl_pos'):
        np.testing.assert_allclose(got[k], ref[k], rtol=0, atol=1e-9 * max(1.0, t_shift), err_msg=k)
    np.testing.assert_array_equal(got['cl_poscalc'], ref['cl_poscalc'])


def test_pack_keeps_references_only_when_centring(monkeypatch):
    p = pulses(basic())
    G, cfg = geometry(), C.FinderConfig()
    tables = J.hit_tables(G, np.zeros(32, int), cfg)
    cols = (np.zeros(len(p['idx']), int), *[p[k] for k in ('idx', 'bar', 'sipm', 'time', 'pe', 'seq')])
    monkeypatch.setattr(dprec, 'SINGLE', False)
    plain, _, _ = J.pack(*cols, G, tables, 3, H=16)
    assert 'tref' not in plain
    monkeypatch.setattr(dprec, 'SINGLE', True)
    cen, _, _ = J.pack(*cols, G, tables, 3, H=16)
    v = cen['valid']
    for k in ('t', 'x', 'y'):
        np.testing.assert_allclose(cen[k][v] + np.broadcast_to(cen[k + 'ref'][:, None], v.shape)[v], plain[k][v])
        assert np.all(cen[k][:, 0][v[:, 0]] == 0)
