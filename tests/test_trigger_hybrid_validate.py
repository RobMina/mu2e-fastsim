"""Full hybrid postflight measures real migrations but rejects altered retained reference statuses."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import hybrid_replay
import integrity
from test_trigger_full_validate import case


def adapt(case):
    path = case.prepared / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['reference_integrity'] = integrity.files(case.paths)
    manifest['population'] = 'bounded fixture population'
    manifest['full_production_history_certified'] = False
    path.write_text(json.dumps(manifest))


def test_migration_is_measured_not_failure(case):
    adapt(case)
    tree = case.trees[str(case.path)]
    tree.fast[0, 0] = 6
    case.log = case.log.replace('ProductPrint art::TriggerResults_TriggerResults__TriggerReplay\n  pass 10 p',
        'ProductPrint art::TriggerResults_TriggerResults__TriggerReplay\n  fail 10 p', 1)
    report = hybrid_replay.validate(case.path, case.prepared, case.config, case.log)
    assert report['all_scheduled_paths_measured']
    assert report['accepted_mismatches'] == 1
    assert report['decision_migrations']['p']['stock_only'] == 1
    assert report['expected_fast_stock_equality'] is False
    assert report['full_production_history_certified'] is False


def test_retained_reference_status_change_fails(case):
    adapt(case)
    case.trees[str(case.path)].stock[0, 1] = 10
    with pytest.raises(ValueError, match='retained original'):
        hybrid_replay.validate(case.path, case.prepared, case.config, case.log)


def campaign_fixture(case):
    adapt(case)
    path = case.prepared / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['injection_campaign'] = {'runtime': {'OFFLINE_IMAGE': '/cvmfs/pinned/image',
        'TRIGGER_AREA': '/cvmfs/released/area'}}
    path.write_text(json.dumps(manifest))
    (case.path.parent / 'replay.expanded.fcl').write_text((case.prepared / 'replay.fcl').read_text())
    case.log = 'image=/cvmfs/pinned/image\nMUSE_WORK_DIR=/cvmfs/released/area stub\n' + case.log


def test_campaign_validates_executed_fcl_and_runtime(case):
    campaign_fixture(case)
    report = hybrid_replay.validate(case.path, case.prepared, case.config, case.log)
    assert report['injection_campaign']['runtime']['OFFLINE_IMAGE'] == '/cvmfs/pinned/image'


@pytest.mark.parametrize('change', ['image', 'release', 'source', 'filter', 'service'])
def test_campaign_rejects_changed_executed_runtime_configuration(case, change):
    campaign_fixture(case)
    if change in ('image', 'release'):
        case.log = case.log.replace('/cvmfs/pinned/image' if change == 'image' else '/cvmfs/released/area', '/cvmfs/wrong')
    else:
        expanded = case.path.parent / 'replay.expanded.fcl'
        text = expanded.read_text()
        if change == 'source':
            text = text.replace('unfiltered_1.art', 'other.art')
        elif change == 'filter':
            text = text.replace('TestFilter', 'OtherFilter')
        else:
            text = text.replace('DbService: {', 'DbService: { purpose: "other"')
        expanded.write_text(text)
    with pytest.raises(ValueError, match='runtime|executed hybrid'):
        hybrid_replay.validate(case.path, case.prepared, case.config, case.log)
