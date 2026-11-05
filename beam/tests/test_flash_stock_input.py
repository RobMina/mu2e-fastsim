"""The stock boundary reader supports only exactly represented float identities."""
import numpy as np
import pytest
import uproot
from beamfastsim.flash_input import stock_summary


def test_stock_ntvd_highest_cycle_and_identity_limit(tmp_path):
    p = tmp_path / "stock.root"
    fields = ("run", "subrun", "evt", "trk", "sid", "pdg", "x", "y", "z", "px", "py", "pz", "time")
    a = {k: np.zeros(2, np.float32) for k in fields}
    a.update(run=np.ones(2, np.float32), subrun=np.ones(2, np.float32), evt=np.array([1, 2], np.float32),
             trk=np.ones(2, np.float32), sid=np.full(2, 8, np.float32), pdg=np.array([11, -11], np.float32))
    with uproot.recreate(p) as root:
        root.mktree("input/ntvd", {k: v.dtype for k, v in a.items()}).extend(a)
    report = stock_summary(p)
    assert report["summary"]["electron_or_positron_rows"] == 2
    assert report["summary"]["represented_events"] == 2
    a["trk"][0] = 2**24
    with uproot.update(p) as root:
        root.mktree("input/ntvd", {k: v.dtype for k, v in a.items()}).extend(a)
    with pytest.raises(ValueError, match="Float"):
        stock_summary(p)
