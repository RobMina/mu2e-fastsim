import numpy as np
import pytest

from beamfastsim.tsbench import compare


def test_saved_comparison_ignores_cost_but_checks_all_states(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    data = dict(id=np.array([[1, 2, 3, 4]]), shard=np.array([2]), pdg=np.array([13]),
                x=np.array([[1., 2., 3.]]), fate=np.array([1], np.int8), pion_decay=np.array([False]))
    np.savez(a, **data, seconds=4.)
    np.savez(b, **data, seconds=2.)
    result = compare(a, b)
    assert result["all_exact"] and result["measured_ratio"] == 2
    data["x"][0, 1] += 1e-8
    np.savez(b, **data, seconds=2.)
    result = compare(a, b)
    assert not result["all_exact"] and result["fields"]["x"]["different_elements"] == 1
    assert result['position_or_fate_changed_rows']['id'] == [[1, 2, 3, 4]]
    assert result['position_or_fate_changed_rows']['after_fate'] == [1]
    data["id"][0, 0] += 1
    np.savez(b, **data, seconds=2.)
    with pytest.raises(ValueError, match="alignment"):
        compare(a, b)
