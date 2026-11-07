"""Independent four-event aggregated accessor fixture check; not physics closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import flash_board_export as export
from .flash_inclusive_check import EVENT, keys, join


def _equal(actual, expected, what):
    if not np.array_equal(actual, expected):
        raise ValueError(f"Fixture mismatch: {what}")


def check(tables):
    """Check exact analytic payload and pointer grouping of FlashBoardFixture."""
    t, selected = export.validate(tables, expected_events=4, expected_subruns=2)
    e, p, s, c, r = (t[name] for name in ('events', 'products', 'straw_steps', 'calo_steps', 'crv_steps'))
    _equal(e['run'], [1, 1, 1, 1], 'run')
    _equal(e['subrun'], [1, 1, 2, 2], 'ordered SubRuns')
    _equal(e['nproducts'], [3, 3, 3, 3], 'collection counts')
    ek = keys(e, EVENT)
    se, ce = join(keys(s, EVENT), ek), join(keys(c, EVENT), ek)
    _equal(se, [0, 0, 1, 3], 'tracker event order')
    _equal(ce, [1], 'calorimeter event order')
    _equal(s['index'], [0, 1, 0, 0], 'tracker collection indices')
    _equal(c['index'], [0], 'calorimeter collection index')
    _equal(s['ionizing_mev'], np.array([2e-6, 5e-7, 5e-7, 5e-7], np.float32), 'tracker ionization')
    _equal(c['birks_mev'], np.array([.02], np.float32), 'calorimeter Birks energy')
    _equal(s['momentum_mev_c'], np.ones(4, np.float32), 'tracker momentum accessor')
    _equal(c['momentum_mev_c'], np.ones(1, np.float32), 'calorimeter momentum accessor')
    _equal(s['particle_key'], [17, 17, 17, 17], 'tracker particle keys')
    _equal(c['particle_key'], [17], 'calorimeter particle key')
    _equal(join(keys(r, EVENT), ek), [0], 'CRV event order')
    _equal(r['index'], [0], 'CRV collection index')
    _equal(r['visible_mev'], np.array([2e-6], np.float32), 'CRV visible energy')
    _equal(r['momentum_mev_c'], np.array([5.], np.float32), 'CRV momentum accessor')
    _equal(r['particle_key'], [17], 'CRV particle key')
    a, b = s['particle_product'][:2]
    if a == b:
        raise ValueError('Fixture mismatch: identical key must identify two distinct particle products')
    _equal(s['particle_product'], np.array([a, b, a, a], np.uint64), 'tracker particle products')
    _equal(c['particle_product'], np.array([b], np.uint64), 'calorimeter particle product')
    _equal(r['particle_product'], np.array([b], np.uint64), 'CRV particle product')
    if len(t['requirements']['tag']) != 3:
        raise ValueError('Fixture mismatch: exactly three configured collections required')
    for role, tag in ((1, 'fixture:tracker'), (2, 'fixture:calo'), (3, 'fixture:crv')):
        rows = p['detector'] == role
        _equal(p['tag'][rows], np.full(4, tag), 'configured detector tag')
        if len(np.unique(p['product'][rows])) != 1:
            raise ValueError('Fixture mismatch: one persistent ProductID per detector branch')
    _equal(t['subruns']['run'], [1, 1], 'counter run')
    _equal(t['subruns']['subrun'], [1, 2], 'counter SubRuns')
    _equal(t['subruns']['generated_events'], [10, 20], 'independent generated counters')
    _equal(t['subruns']['tag'], ['fixture', 'fixture'], 'counter tag')
    _equal(selected['tracker'], [True, False, False, False], 'tracker acceptance')
    _equal(selected['calo'], [False, True, False, False], 'calorimeter acceptance')
    _equal(selected['crv'], [True, False, False, False], 'CRV acceptance')
    _equal(selected['retained'], [True, True, False, False], 'detector OR')
    return export.summarize(t, expected_events=4, expected_subruns=2,
                            input_scope='filter-input', complete_subruns=True)


def compare(left, right):
    """Require two individually valid fixtures to have identical typed payloads."""
    check(left)
    check(right)
    for name, spec in export.SCHEMA.items():
        for field, dtype in spec.items():
            a, b = np.asarray(left[name][field]), np.asarray(right[name][field])
            if dtype != 'str' and a.dtype != b.dtype:
                raise ValueError(f'Fixture mismatch: reread dtype {name}.{field}')
            _equal(a, b, f'reread {name}.{field}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reread', type=Path)
    args = parser.parse_args(argv)
    original = export.read_tables(args.input)
    result = check(original)
    if args.reread:
        compare(original, export.read_tables(args.reread))
    print(json.dumps(dict(synthetic_fixture_passed=True, reread_compared=bool(args.reread),
                          retained_events=result['retained_events'], generated_events=result['generated_events'],
                          tracker_ionizing_mev=result['tracker_ionizing_mev'], physics_validated=False), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
