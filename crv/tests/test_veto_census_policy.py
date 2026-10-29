"""Policy separation, event completeness and exact payload checks on synthetic flat TTrees."""
import copy
import json

import numpy as np
import pytest
import uproot

from crvfastsim.veto import census_policy as P
from test_genassoc import payload


def write(path, data):
    with uproot.recreate(path) as root:
        for name, frame in data.items():
            values = {k: frame[k].to_numpy() for k in frame}
            root.mktree('dump/' + name, {k: 'string' if v.dtype.kind == 'O' else v.dtype for k, v in values.items()})
            if len(frame):
                root['dump/' + name].extend(values)
    return path


def staged():
    data = payload()
    data['events']['schema_version'] = np.uint32(3)
    data['events']['association_policy'] = 'staged_cat'
    data['events']['ngen'] = data['events'].n_gen.astype(np.uint64)
    data['events']['nsteps'] = data['events'].n_steps
    data['particles'].loc[0, 'pdg'] = 2112
    return data


def check(data, policy='staged_cat'):
    return P.validate_policy(*(data[k] for k in ('events', 'gen', 'particles', 'steps')), policy)


def test_staged_keeps_pointer_checks_without_ancestor_pdg_assumption(tmp_path):
    data = staged()
    assert check(copy.deepcopy(data))['linked_primary_pdg_mismatches'] == 1
    path = write(tmp_path / 'cat.root', data)
    result = P.read_policy(path, 'staged_cat')
    assert result['events'] == 3
    assert 'unsupported association schema' in result['cosmic_donor_rejected']
    assert result['pointer_ProductIDs_keys_checked']
    data['events']['schema_version'] = np.uint32(2)
    data['events']['association_policy'] = 'retained_primary'
    with pytest.raises(ValueError, match='signed PDG'):
        check(data, 'retained_primary')


@pytest.mark.parametrize('tree,column,row,value', [
    ('events', 'schema_version', 0, 2), ('events', 'association_policy', 0, 'retained_primary'),
    ('events', 'associations_checked', 0, 0), ('events', 'full_generator_stream', 0, 1),
    ('events', 'ngen', 0, 9), ('events', 'nsteps', 0, 9), ('events', 'sim_product_id', 0, 1),
    ('particles', 'gen_product_id', 0, 101), ('particles', 'genidx', 0, 6),
    ('particles', 'parent_product_id', 1, 202), ('particles', 'parent', 1, -1),
    ('steps', 'sp_product_id', 0, 202), ('steps', 'sp', 0, 99),
    ('gen', 'e', 0, np.nan), ('gen', 'idx', 0, -1),
])
def test_policy_and_pointer_corruption_rejected(tree, column, row, value):
    data = staged()
    data[tree].loc[row, column] = value
    with pytest.raises(ValueError):
        check(data)


def test_unfinished_job_and_wrong_event_count(tmp_path):
    probe = tmp_path / 'probe'
    probe.mkdir()
    write(probe / 'dump.root', staged())
    log = probe / 'job.log'
    log.write_text('TrigReport Events total = 3\nArt has completed and will exit with status 1.\n')
    out = tmp_path / 'report.json'
    with pytest.raises(ValueError, match='successful art'):
        P.run(probe, out, 'staged_cat')
    log.write_text('TrigReport Events total = 3\nArt has completed and will exit with status 0.\n')
    with pytest.raises(ValueError, match='event count'):
        P.run(probe, out, 'staged_cat', events=4)
    result = P.run(probe, out, 'staged_cat', events=3)
    assert json.loads(out.read_text()) == result
    with pytest.raises(ValueError, match='new report'):
        P.run(probe, out, 'staged_cat')


def test_payload_equality_all_branches_and_nans(tmp_path):
    data = staged()
    data['subruns'] = data['events'][['run', 'subrun']].drop_duplicates().assign(livetime=np.nan)
    ref = write(tmp_path / 'ref.root', data)
    candidate = copy.deepcopy(data)
    candidate['events']['added'] = 42
    new = write(tmp_path / 'new.root', candidate)
    report = P.compare_payload(new, ref)
    assert report['events']['added_branches'] == ['added']
    candidate['particles'].loc[0, 'pdg'] = 11
    bad = write(tmp_path / 'bad.root', candidate)
    with pytest.raises(ValueError, match='changed reference payload particles'):
        P.compare_payload(bad, ref)


@pytest.mark.parametrize('guard', ['product', 'primary'])
def test_negative_controls_require_expected_guard_exit_and_no_payload(tmp_path, guard):
    data = staged()
    for k in data:
        data[k] = data[k].iloc[:0]
    write(tmp_path / 'dump.root', data)
    text = ('references GenParticle (101, 0); selected wrong ProductID 202 size 1' if guard == 'product'
            else 'primary/gen signed PDG mismatch')
    (tmp_path / 'job.log').write_text(text + '\nArt has completed and will exit with status 1.\n')
    status = tmp_path / 'exit_status.txt'
    status.write_text('1\n')
    assert P.check_negative(tmp_path, guard, 0)['completed_event_prefix'] == 0
    with pytest.raises(ValueError, match='unexpected payload'):
        P.check_negative(tmp_path, guard, 1)
    status.write_text('0\n')
    with pytest.raises(ValueError, match='exit one'):
        P.check_negative(tmp_path, guard, 0)


def test_negative_control_wrong_guard_is_not_success(tmp_path):
    (tmp_path / 'job.log').write_text('unrelated crash\nArt has completed and will exit with status 1.\n')
    (tmp_path / 'exit_status.txt').write_text('1')
    with pytest.raises(ValueError, match='ProductID rejection'):
        P.check_negative(tmp_path, 'product', 0)


def test_compact_alias_comparison_keeps_real_contract_mismatch(tmp_path):
    from test_veto_catsteps_v2 import converted, rewrite
    first, _ = converted(tmp_path, 'a')
    second, meta = converted(tmp_path, 'b')
    report = P.check_compact(first, second)
    assert 'duplicate content_sha256' in report['joint_pool_rejection']
    meta['extraction']['dumper_sha256'] = '9' * 64
    rewrite(second, metadata=meta)
    report = P.check_compact(first, second)
    assert report['joint_pool_rejection'] == 'pool extraction metadata disagreement'
    assert report['identical_compact_content']['frames'] == 4


def test_exact_comparison_rejects_dtype_change(tmp_path):
    data = staged()
    data['subruns'] = data['events'][['run', 'subrun']].drop_duplicates()
    ref = write(tmp_path / 'ref.root', data)
    data['gen']['e'] = data['gen'].e.astype(np.float32)
    new = write(tmp_path / 'new.root', data)
    with pytest.raises(ValueError, match='changed reference payload gen'):
        P.compare_payload(new, ref)
