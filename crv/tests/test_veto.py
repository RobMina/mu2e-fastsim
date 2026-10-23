"""crvfastsim.veto: card parsing, vectorised application, Jeffreys fractions."""
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crvfastsim.veto import VetoCard, fraction, jeffreys, vetoed  # noqa: E402
from crvfastsim.veto.apply import gate  # noqa: E402


def _coincs():
    # event 0: one coinc 100 ns before the track; event 1: none; event 2: one at +10 ns (after the track) and a
    # low-PE one 50 ns before; event 3: a flash coinc starting at 300 ns plus one 20 ns before the track
    return ak.zip({"time": ak.Array([[900.0], [], [1010.0, 950.0], [310.0, 980.0]]),
                   "timeStart": ak.Array([[890.0], [], [1000.0, 940.0], [300.0, 970.0]]),
                   "timeEnd": ak.Array([[910.0], [], [1020.0, 960.0], [320.0, 990.0]]),
                   "PEs": ak.Array([[40.0], [], [60.0, 8.0], [500.0, 30.0]]),
                   "nLayers": ak.Array([[3], [], [4, 3], [4, 3]]),
                   "sectorType": ak.Array([[1], [], [2, 1], [1, 3]])})


def test_window_pe_sector_and_gate():
    t = ak.Array([[1000.0], [1000.0], [1000.0], [1000.0]])
    co = _coincs()
    base = VetoCard(dt_lo=0.0, dt_hi=150.0)
    assert ak.to_list(ak.firsts(vetoed(t, co, base))) == [True, False, True, True]
    assert ak.to_list(ak.firsts(vetoed(t, co, VetoCard(dt_lo=0.0, dt_hi=150.0, min_pe=10)))) == [True, False, False, True]
    assert ak.to_list(ak.firsts(vetoed(t, co, VetoCard(sectors=(3,))))) == [False, False, False, True]
    sym = VetoCard(dt_lo=-150.0, dt_hi=150.0, min_pe=10)
    assert ak.to_list(ak.firsts(vetoed(t, co, sym))) == [True, False, True, True]     # +10 ns coinc counts now
    by = VetoCard(min_pe=10, min_pe_by_sector={1: 35.0})
    assert ak.to_list(ak.firsts(vetoed(t, co, by))) == [True, False, False, True]    # sector 1 at 35 PE; event 3 vetoed by its 30-PE sector-3 coinc
    v, sec = vetoed(t, co, base, return_sector=True)
    assert ak.to_list(ak.firsts(sec)) == [1, -1, 1, 3]
    g = gate(co, VetoCard(gate_start_lo=420.0, gate_end_hi=1700.0))
    assert g.tolist() == [True, True, True, False]                                   # event 3 has a flash coinc


def test_multiple_tracks_per_event():
    t = ak.Array([[1000.0, 2000.0], [], [1000.0]])
    co = ak.zip({k: ak.Array([[900.0], [5.0], [1200.0]]) for k in ("time", "timeStart", "timeEnd", "PEs")})
    v = vetoed(t, co, VetoCard())
    assert ak.to_list(v) == [[True, False], [], [False]]


def test_card_sha_and_cosmic_ana(tmp_path):
    a = VetoCard(dt_lo=0, dt_hi=150, min_pe_by_sector={1: 10.0, 3: 12.0}, source="x")
    b = VetoCard(dt_lo=0, dt_hi=150, min_pe_by_sector={3: 12.0, 1: 10.0}, source="y")
    assert a.sha() == b.sha() and a.sha() != VetoCard(dt_lo=-150, dt_hi=150).sha()
    pytest.importorskip("yaml")
    y = tmp_path / "cuts.yaml"
    y.write_text("""defaults:
  thresholds: {lo_crv_start_ns: 420.0, hi_crv_end_ns: 1700.0, lo_veto_dt_ns: -150.0, hi_veto_dt_ns: 150.0}
  active: {unvetoed: true, within_coinc_start_time: false, within_coinc_end_time: false}
cutsets:
  dev:
    active: {within_coinc_start_time: true, within_coinc_end_time: true}
    thresholds: {lo_veto_dt_ns: 0.0}
  noveto:
    active: {unvetoed: false}
""")
    d = VetoCard.from_mu2e_cosmic_ana(str(y), "dev")
    assert (d.dt_lo, d.dt_hi, d.gate_start_lo, d.gate_end_hi) == (0.0, 150.0, 420.0, 1700.0)
    assert VetoCard.from_mu2e_cosmic_ana(str(y), "defaults").gate_start_lo is None
    with pytest.raises(ValueError):
        VetoCard.from_mu2e_cosmic_ana(str(y), "noveto")


def test_jeffreys_and_fraction():
    lo, hi = jeffreys(0, 10)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = jeffreys(10, 21423)                                                     # 58017's 10 unvetoed
    assert lo < 10 / 21423 < hi
    f = fraction(np.array([True, False, False, True]))
    assert (f["k"], f["n"], f["frac"]) == (2, 4, 0.5)


def test_deadtime_full_and_factorised():
    from crvfastsim.veto.deadtime import factorised, full
    co = ak.zip({"time": ak.Array([[1000.0], [], [300.0]]), "timeStart": ak.Array([[995.0], [], [295.0]]),
                 "timeEnd": ak.Array([[1005.0], [], [305.0]]), "PEs": ak.Array([[50.0], [], [500.0]])})
    t = ak.Array([[1050.0], [1050.0], [1050.0]])
    card = VetoCard(dt_lo=0.0, dt_hi=150.0, gate_start_lo=420.0)
    S = full(t, co, card)
    assert (S["k"], S["vetoed_passing_gate"], S["gate_fail"]) == (2, 1, 1)      # event 0 vetoed, event 2 flash
    F = factorised(np.array([1050.0, 1050.0, 1050.0]), co, card)
    assert F["frames"] == 3 and abs(F["frac"] - 2 / 3) < 1e-12
    coinc = VetoCard(dt_lo=0.0, dt_hi=150.0, gate_start_lo=420.0, gate_mode="coinc")
    assert full(t, co, coinc)["k"] == 1                                          # the flash coinc can no longer veto
    assert abs(factorised(np.array([1050.0]), co, coinc)["frac"] - 1 / 3) < 1e-12
    # t0 spread: coverage (1000, 1150) of tracks uniform on 900-1300 is 150/400 for the one busy frame
    ts = np.linspace(900.0, 1300.0, 40001)[:-1] + 0.005
    F = factorised(ts, co[:2], VetoCard(dt_lo=0.0, dt_hi=150.0), step=0.25)
    assert abs(F["frac"] - 0.5 * 150 / 400) < 2e-3


def test_deadtime_gate_mode_sha():
    assert VetoCard().sha() == VetoCard(gate_mode="event").sha() != VetoCard(gate_mode="coinc").sha()
    with pytest.raises(ValueError):
        VetoCard.from_dict({"gate_mode": "frame"})
