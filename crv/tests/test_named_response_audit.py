"""Saved-product paired diagnostics retain zero events and full-width identities."""
import json
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from crvfastsim.analyses import named_response_audit as A


def test_ratio_paired_covariance():
    x = np.array([0., 1., 2., 3.])
    r = A.paired_ratio(2 * x, x)
    assert r["relative"] == 1 and r["relative_se"] == 0
    assert r["events"] == 4 and r["reference_mean"] == 1.5
    assert A.paired_ratio(x, np.zeros(4))["status"] == "unsupported"
    assert A.paired_ratio([1], [1])["status"] == "unsupported"
    with pytest.raises(ValueError):
        A.paired_ratio([np.nan], [1])


def test_fullwidth_identity_and_foreign_rows():
    ids = np.array([[1400, 1, 1], [1400, 1, 2**20 + 1], [1400, 2, 1]])
    rows = ak.Array(dict(zip(A.KEY, ids[[2, 1, 1]].T)))
    assert A.row_events(rows, ids).tolist() == [2, 1, 1]
    for bad in ([1400, 0, 1], [1400, 1, 2], [1400, 3, 1]):
        with pytest.raises(ValueError, match="outside"):
            A.row_events(ak.Array(dict(zip(A.KEY, np.array([bad]).T))), ids)


def root_file(path, scale=1, duplicate=False, foreign=False):
    with uproot.recreate(path) as f:
        f["crvdump/events"] = {"run": np.array([1400, 1400, 1400]), "subrun": np.ones(3, int),
                               "event": np.array([1, 2, 2 if duplicate else 3])}
        keys = {"run": np.array([1400, 1400]), "subrun": np.ones(2, int),
                "event": np.array([1, 4 if foreign else 2])}
        f["crvdump/digis"] = {**keys, "bar": np.array([0, 1]), "sipm": np.array([0, 3]),
                              "adc": ak.Array([[10 + scale, 10 + 2*scale], [10 + scale]])}
        f["crvdump/pulses"] = {**keys, "bar": np.array([0, 1]), "sipm": np.array([0, 3]),
                               "pe": np.array([10., 20.]) * scale, "seq": np.zeros(2, int),
                               "time": np.array([20., 100001.])}


def test_audit_all_sectors_and_zero_event(monkeypatch, tmp_path):
    ref, cand, man = (tmp_path / x for x in ("ref.root", "cand.root", "manifest.json"))
    root_file(ref)
    root_file(cand, scale=2)
    man.write_text("{}")
    geom = SimpleNamespace(nbars=2, sector_names=["T", "L"], sector=np.array([0, 1]))
    cond = SimpleNamespace(pedestal=np.full(8, 10.), pulsearea=np.ones(8))
    monkeypatch.setattr(A.NC, "load_conditions", lambda *args: (cond, {"config_sha256": "synthetic"}))
    rc = SimpleNamespace(name="synthetic", geometry=lambda: geom)
    outdir = tmp_path / "report"
    result = A.audit(ref, cand, man, rc, outdir)
    assert result["events"] == 3 and set(result["sectors"]) == {"T", "L"}
    for sector in result["sectors"].values():
        assert set(sector) == {"all", "0", "1", "2", "3"}
        m = sector["all"]["paired_event_metrics"]
        assert m["digi_count"]["relative"] == 0
        assert m["adc_pedestal_subtracted"]["relative"] == 1
        assert m["pulse_pe"]["relative_se"] == 0
    assert result["sectors"]["T"]["3"]["paired_event_metrics"]["digi_count"]["status"] == "unsupported"
    assert json.loads((outdir / "audit.json").read_text()) == result
    assert "| L |" in (outdir / "table.md").read_text()
    with pytest.raises(FileExistsError):
        A.audit(ref, cand, man, rc, outdir)
    root_file(cand, duplicate=True)
    with pytest.raises(ValueError, match="unique"):
        A.audit(ref, cand, man, rc, tmp_path / "bad")
    root_file(cand, foreign=True)
    with pytest.raises(ValueError, match="outside"):
        A.audit(ref, cand, man, rc, tmp_path / "bad")
