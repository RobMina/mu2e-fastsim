"""Endpoint tables are diagnostic counts, not causal physics labels."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import migration_endpoints


def test_loss_gain_and_both_fail_pairs():
    stock = np.array([[13], [10], [10], [13]], dtype=np.uint16)
    fast = np.array([[6], [9], [6], [9]], dtype=np.uint16)
    row = migration_endpoints.summarize(stock, fast, ['p'], {'p': ['Digi', 'Seed', 'Fit']},
        {'p': ['Seed', 'Fit']})['p']
    assert row['lost'] == {'events': 1, 'endpoint_pairs': [{'stock': '<end-of-path>', 'fast': 'Seed', 'events': 1}]}
    assert row['gained']['events'] == 1
    assert row['both_fail_different_endpoint']['events'] == 0


def test_empty_migration_groups_and_invalid_state():
    words = np.array([[5]], dtype=np.uint16)
    row = migration_endpoints.summarize(words, words, ['p'], {'p': ['Cut']}, {'p': ['Cut']})['p']
    assert row['lost'] == {'events': 0, 'endpoint_pairs': []}
    with pytest.raises(ValueError, match='Ready/Exception'):
        migration_endpoints.summarize(words, np.array([[0]], dtype=np.uint16), ['p'], {'p': ['Cut']}, {'p': ['Cut']})
