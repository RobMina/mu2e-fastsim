"""State summaries cannot invent a boundary or borrow another event's identity."""
import numpy as np
import pytest
from beamfastsim.flash_input import summarize, VD
from beamfastsim.flash_ancestry import PARTICLE


def test_state_identity_is_event_scoped():
    s = {k: np.zeros(2) for k in VD}
    s.update(run=np.ones(2, int), subrun=np.ones(2, int), event=np.array([1, 2]), sp=np.array([1, 1]),
             vdid=np.array([8, 8]), px=np.array([3., 0.]), py=np.array([4., 0.]))
    p = {k: np.zeros(1) for k in PARTICLE}
    p.update(run=np.array([1]), subrun=np.array([1]), event=np.array([1]), key=np.array([1]), pdg=np.array([11]))
    result = summarize(s, p)
    assert result["state_rows"] == 2
    assert result["represented_events"] == 2
    assert result["unresolved_particle_rows"] == 1
    assert result["electron_or_positron_rows"] == 1
    assert result["momentum_quantiles_mev"][-1] == 5


def test_empty_states_and_nonfinite_rejection():
    s = {k: np.array([]) for k in VD}
    p = {k: np.array([]) for k in PARTICLE}
    assert summarize(s, p)["momentum_quantiles_mev"] is None
    s = {k: np.zeros(1) for k in VD}
    s["x"] = np.array([np.nan])
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize(s, p)
