"""Independent generated-census matching and resolved-window guards."""
import numpy as np
import pytest

from crvfastsim.objective import joint_identity as J
from crvfastsim.objective import validation as V


def keys(run=1400):
    return J.key_rows(dict(run=np.array([run, run]), subrun=np.array([2, 3]), event=np.array([4, 4])))


def test_exact_source_keys_including_empty_and_reordered_rows():
    expected = [keys(), keys(1401)]
    rows = np.repeat(np.concatenate(expected), 4)
    source = np.repeat([0, 1], 8)
    slots = np.tile(np.arange(4), 4)
    order = np.arange(16)[::-1]
    out = J.audit_capture(rows[order], source[order], slots[order], expected)
    assert [x['events'] for x in out] == [2, 2]


@pytest.mark.parametrize('kind', ['wrong_run', 'missing', 'duplicate', 'source', 'slot'])
def test_mislabeled_keys_do_not_pass_on_counts_alone(kind):
    rows = np.repeat(keys(), 4)
    source, slots = np.zeros(8, int), np.tile(np.arange(4), 2)
    if kind == 'wrong_run':
        rows['run'] += 1
    elif kind == 'missing':
        rows, source, slots = rows[:-1], source[:-1], slots[:-1]
    elif kind == 'duplicate':
        rows[-1] = rows[0]
    elif kind == 'source':
        source[-1] = 2
    else:
        slots[-1] = 2
    with pytest.raises(ValueError):
        J.audit_capture(rows, source, slots, [keys()])


def test_real_resolved_window_is_checked(monkeypatch):
    monkeypatch.setattr(V, '_result_counts', lambda result: {'events': 2})
    result = dict(response_state={'exposure_ns': 200000.},
                  resolved_config={'values': {'timing': {'window': [0., 400000.]}}})
    with pytest.raises(ValueError, match='time exposure differs'):
        V._validate_measurement(result, {}, {})
