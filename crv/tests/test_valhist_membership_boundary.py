"""Synthetic exact-boundary tests for the retained membership diagnostic."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crvfastsim import geometry
from crvfastsim.reco import coinc as C
from valhist import membership_boundary as M


def test_strict_slope_difference():
    c = C.EXTRACTED_V04['T2']
    x = np.array([[0., 0., 3., 3.], [0., 0., np.nextafter(3., np.inf), np.nextafter(3., np.inf)]])
    y = np.broadcast_to(np.arange(4.), x.shape)
    t = np.zeros_like(x)
    ok, tm, sm, dm = M.combination_mask(x, y, t, c)
    assert ok.tolist() == [True, False]
    assert dm[0] == 0 and dm[1] < 0


def test_time_boundary_is_inclusive():
    c = C.EXTRACTED_V04['T2']
    x = np.zeros((2, 4))
    y = np.broadcast_to(np.arange(4.), x.shape)
    t = np.array([[0, 0, 0, 10.], [0, 0, 0, np.nextafter(10., np.inf)]])
    assert M.combination_mask(x, y, t, c)[0].tolist() == [True, False]


def test_nonfinite_slopes_refused():
    with pytest.raises(ValueError, match='nonfinite'):
        M.combination_mask(np.zeros((1, 4)), np.zeros((1, 4)), np.zeros((1, 4)), C.EXTRACTED_V04['T2'])


def test_stock_order_addition():
    g = geometry.CRVGeometry(['T2'], np.zeros(4, int), np.zeros(4, int), np.arange(4),
                            np.full(4, 3), np.zeros((4, 3)), np.ones((4, 3)),
                            np.tile([1, 0, 2], (4, 1)))
    c = {'crs.layerDirectionT2': [0, 1, 0], 'crs.gapDirectionT2': [1, 0, 0],
         'crs.offsetDirectionT2': [1, 0, 0], 'crs.nLayers': 4,
         'crs.scintillatorBarThickness': 20., 'crs.gapBetweenLayers': [9., 9., 9.],
         'crs.layerOffset': 8., 'crs.scintillatorBarWidth': 50., 'crs.gapSmall': .1,
         'crs.gapLarge': .2, 'crs.gapBetweenModules': 1., 'crs.nCountersPerModuleT2': 32,
         'crs.firstCounterT2': [10000., 4000., 20000.]}
    out = M.stock_order_geometry(g, c)
    expected = np.asarray(c['crs.firstCounterT2']) + np.arange(4)[:, None] * np.array([8., 29., 0.])
    expected = expected + np.array([50.2 + 2 * 50.1, 0, 0])
    np.testing.assert_array_equal(out.center, expected)
    np.testing.assert_array_equal(g.center, np.zeros((4, 3)))


def test_exclusive_output(tmp_path):
    out = tmp_path / 'exists.json'
    out.write_text('old')
    with pytest.raises(FileExistsError):
        M.create('missing', out, 'missing')
    assert out.read_text() == 'old'
