"""Retained Cat primary diagnostics preserve the distinction between keys and ProductIDs."""
import json

import numpy as np
import pandas as pd
import pytest
import uproot

from crvfastsim.veto.cat_primary_audit import run, summarize


def frames():
    parts = pd.DataFrame({'run': [1, 1], 'subrun': [0, 1], 'event': [7, 7], 'key': [2, 2],
                         'parent': [-1, -1], 'isprim': [1, 1], 'genidx': [0, 0],
                         'pdg': [11, 22], 'ccode': [56, 56], 'se': [2., 3.], 'st': [4., 5.]})
    gen = pd.DataFrame({'run': [1, 1], 'subrun': [0, 1], 'event': [7, 7], 'idx': [0, 0],
                       'pdg': [22, 22], 'e': [9., 9.], 't': [0., 0.]})
    return parts, gen


def test_full_tuple_join_and_no_false_product_claim():
    parts, gen = frames()
    report, linked = summarize(parts, gen)
    assert report['linked_primary_pdg_mismatches'] == 1
    assert report['linked_primary_with_parent'] == 0
    assert report['generator_ProductIDs_checked'] is False
    assert report['mismatch_types'] == [{'pdg': 11, 'gen_pdg': 22, 'ccode': 56, 'particles': 1}]
    assert len(linked) == 2
    parts['gen_product_id'] = [101, 102]
    gen['gen_product_id'] = [101, 102]
    assert summarize(parts, gen)[0]['generator_ProductIDs_checked'] is True
    gen.loc[0, 'gen_product_id'] = 103
    with pytest.raises(ValueError, match='ProductID mismatch'):
        summarize(parts, gen)


@pytest.mark.parametrize('case', ['duplicate', 'missing', 'partial_product', 'floating_id', 'flag'])
def test_bad_keys_fail_closed(case):
    parts, gen = frames()
    if case == 'duplicate':
        gen = pd.concat([gen, gen])
    elif case == 'missing':
        gen = gen.iloc[:1]
    elif case == 'partial_product':
        parts['gen_product_id'] = 101
    elif case == 'floating_id':
        parts['event'] = parts.event.astype(float)
    else:
        parts.loc[0, 'isprim'] = 2
    with pytest.raises(ValueError):
        summarize(parts, gen)


def test_root_audit_hash_cycle_and_event(tmp_path):
    parts, gen = frames()
    source, out = tmp_path / 'dump.root', tmp_path / 'audit.json'
    with uproot.recreate(source) as root:
        for name, frame in [('particles', parts), ('gen', gen)]:
            root.mktree('dump/' + name, {k: v.dtype for k, v in frame.items()})
            root['dump/' + name].extend({k: v.to_numpy() for k, v in frame.items()})
    result = run(source, out, [1, 0, 7])
    assert len(result['requested_event_linked_particles']) == 1
    assert result['requested_event_linked_particles'][0]['pdg'] == 11
    assert result['entries_highest_cycle'] == {'particles': 2, 'gen': 2}
    assert json.loads(out.read_text()) == result
    with pytest.raises(ValueError, match='new report'):
        run(source, out)


def test_staged_cat_cannot_become_retained_primary_cosmic_donor(tmp_path):
    from test_genassoc import payload, write
    from crvfastsim.genassoc import validate_donor
    data = payload()
    data['events']['schema_version'] = np.uint32(3)
    with pytest.raises(ValueError, match='unsupported association schema'):
        validate_donor(write(tmp_path / 'staged.root', data))
