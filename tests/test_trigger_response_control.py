"""Full-stock controls reuse F response machinery without primary-only admission or stronger claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import uproot

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import integrity
import replay
import response_control as control
from test_trigger_step_handoff import CONFIG


def test_separate_scope_and_same_response_contract():
    a, _ = control.step_handoff.prepare(CONFIG, ['fast-source.art'], 'fast-table.root', 20261104)
    b, manifest, text = control.intended(CONFIG, 'stock-source.art', 'stock-table.root', 20261104)
    control.compare_fast_configuration(a, b)
    assert manifest['admission_kind'] == control.KIND
    assert manifest['primary_only_admission_used'] is False
    assert manifest['particle_state_lossless'] is False
    assert manifest['electronics_only'] is False
    assert 'not pure secondaries' in manifest['scope']
    assert manifest['fcl_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert 'primary-only' not in manifest['scope']
    assert control.step_handoff.prepare(CONFIG, ['x'], 'y')[1]['scope'].startswith('primary-only')


@pytest.mark.parametrize('mutation', ['seed', 'label', 'input', 'schedule', 'geometry', 'output', 'timing'])
def test_fast_response_contract_changes_fail(mutation):
    a, _ = control.step_handoff.prepare(CONFIG, ['a'], 'fast.root', 20261104)
    b = deepcopy(a)
    if mutation == 'seed':
        b['services']['SeedService']['baseSeed'] = replay.Atom('20261105')
    elif mutation == 'label':
        b['process_name'] = 'DifferentProcess'
    elif mutation == 'input':
        b['physics']['producers']['R7FastSD']['StrawGasStepModules'] = ['compressDetStepMCs']
    elif mutation == 'schedule':
        b['physics']['Inject'].reverse()
    elif mutation == 'geometry':
        b['services']['GeometryService']['inputFile'] = 'other'
    elif mutation == 'output':
        b['outputs']['Combined']['outputCommands'] = ['drop *']
    else:
        b['physics']['producers']['R7FastSD']['EventWindowMarker'] = 'new-marker'
    with pytest.raises(ValueError, match='construction'):
        control.compare_fast_configuration(a, b)


def test_dump_explicit_new_particle_and_step_tags_without_redigitization(tmp_path):
    directory = tmp_path / 'dump'
    control.dump_prepare(CONFIG, tmp_path / 'saved.art', directory)
    parsed = replay.parse_expanded((directory / 'job.fcl').read_text())
    dump = parsed['physics']['analyzers']['dump']
    assert dump['strawGasSteps'] == 'R7FastSteps::R7TrackerInject'
    assert dump['simParticles'] == 'R7FastSteps::R7TrackerInject'
    assert dump['genParticles'] == 'compressDigiMCs::Digitize'
    assert 'producers' not in parsed['physics']
    assert 'outputs' not in parsed
    with pytest.raises(ValueError, match='reuse'):
        control.dump_prepare(CONFIG, tmp_path / 'saved.art', directory)


def test_save_refuses_overwrite_and_symlink(tmp_path):
    file = tmp_path / 'receipt.json'
    control.save(file, {'a': 1})
    with pytest.raises(FileExistsError):
        control.save(file, {'a': 2})
    link = tmp_path / 'link.json'
    link.symlink_to(file)
    with pytest.raises(FileExistsError):
        control.save(link, {'a': 3})
    assert json.loads(file.read_text()) == {'a': 1}


def test_completed_bundle_requires_actual_importer_and_dumper(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    plugins = runtime / 'plugins'
    plugins.mkdir(parents=True)
    files = []
    for name in ('StrawGasStepsFromTable', 'StrawGasStepTableDumper'):
        path = plugins / f'libtrkfastsim_{name}_module.so'
        path.write_bytes(name.encode())
        files.append(path)
    monkeypatch.setattr(control, 'IMPORTER', integrity.digest(files[0])['sha256'])
    monkeypatch.setattr(control, 'DUMPER', integrity.digest(files[1])['sha256'])
    runner = runtime / 'offline_plugins.sh'
    runner.write_text('saved runner')
    control.save(runtime / 'runtime.json', {'plugins': integrity.files(files),
        'adapted_runner_sha256': integrity.digest(runner)['sha256']})
    plan = {'runtime': {'TRIGGER_PLUGINS': 'arbitrary-input'}, 'parents': [{'injection': str(tmp_path)}]}
    completed = {'injection_evidence': integrity.files([*files, runner, runtime / 'runtime.json'])}
    result = control.inspect_plugins(plan, completed)
    assert result['runner'] == integrity.digest(runner)
    with pytest.raises(ValueError, match='completed F'):
        control.inspect_plugins(plan, {'injection_evidence': []})
    files[0].write_bytes(b'changed')
    with pytest.raises(ValueError, match='bundle changed'):
        control.inspect_plugins(plan, completed)


@pytest.mark.parametrize('flag', ['admission_kind', 'primary_only_admission_used', 'electronics_only', 'particle_state_lossless',
                                  'source_simparticle_product_ids_authenticated'])
def test_preflight_rejects_false_admission_kind_or_promoted_claims(tmp_path, flag):
    value = {'admission_kind': control.KIND, 'primary_only_admission_used': False,
             'electronics_only': False, 'particle_state_lossless': False,
             'source_simparticle_product_ids_authenticated': False}
    value[flag] = 'primary-only' if flag == 'admission_kind' else True
    path = tmp_path / 'plan.json'
    control.save(path, value)
    with pytest.raises(ValueError, match='separate full-stock'):
        control.preflight(path, 1)


class _Metadata(dict):
    """Only the ROOT metadata interface; physics payload decoding is never available."""
    def keys(self, recursive=False):
        return super().keys()


class _ArtRoot(_Metadata):
    file = SimpleNamespace(streamers={})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __getitem__(self, key):
        return super().__getitem__('Events;1' if key == 'Events' else key)


def _art_metadata(identities, tag=None):
    tree = _Metadata()
    tree.classname, tree.num_entries = 'TTree', len(identities)
    tree.identities = identities
    if tag is not None:
        stem = 'mu2e::StrawGasSteps_' + tag.replace(':', '_')
        present = SimpleNamespace(typename='bool', num_entries=len(identities),
                                  array=lambda *, library: np.ones(len(identities), dtype=np.bool_))
        obj = SimpleNamespace(typename='std::vector<mu2e::StrawGasStep>', num_entries=len(identities))
        parent = _Metadata({stem + '.present': present, stem + '.obj': obj})
        parent.typename = 'art::Wrapper<std::vector<mu2e::StrawGasStep> >'
        parent.num_entries = len(identities)
        tree[stem + '.'] = parent
    return _ArtRoot({'Events;1': tree})


@pytest.fixture(scope='module')
def completed_control(tmp_path_factory):
    """Real receipts/files/split/presence API, but no art jobs or large ROOT decoding."""
    directory = tmp_path_factory.mktemp('completed-response-control')
    counts = [2805] * 4 + [2804] * 6
    seed = 20261104
    roots, sources, exports, parents, ordered_keys = {}, [], [], [], []
    config = directory / 'production.fcl'
    config.write_text(CONFIG)
    table = directory / 'original-export.root'
    # Descending parent runs expose accidental global sorting of the ledger.
    for ordinal, count in enumerate(counts, 1):
        numeric = np.column_stack((np.full(count, 1440 - ordinal, dtype=np.uint64),
                                   np.full(count, 55, dtype=np.uint64),
                                   np.arange(1, count + 1, dtype=np.uint64)))
        ordered_keys.append(numeric)
        source = directory / f'stock-{ordinal}.art'
        export = directory / f'original-{ordinal}.art'
        source.write_bytes(f'saved Digitize parent {ordinal}'.encode())
        export.write_bytes(f'original Primary parent {ordinal}'.encode())
        roots[source] = _art_metadata(numeric[::-1], 'compressDigiMCs::Digitize')
        roots[export] = _art_metadata(numeric[::-1], 'compressDetStepMCs::Primary')
        prepared = directory / 'fast-parents' / f'parent_{ordinal}'
        prepared.mkdir(parents=True)
        fast, _ = control.step_handoff.prepare(CONFIG, [str(source)], f'fast-{ordinal}.root', seed)
        (prepared / 'job.fcl').write_text(control.render(fast))
        parents.append(dict(ordinal=ordinal, events=count, source=integrity.digest(source),
                            prepared=str(prepared), injection=str(directory / f'injection-{ordinal}')))
        sources.append(source)
        exports.append(export)
    denominator = np.concatenate(ordered_keys)
    # Sparse but real export: two SGS rows per parent, preserving a payload column.
    table_keys = np.concatenate([values[[0, -1]] for values in ordered_keys])
    with uproot.recreate(table) as root:
        root.mktree('dump/steps', {'run': 'uint64', 'subrun': 'uint64', 'event': 'uint64', 'payload': 'int32'})
        root['dump/steps'].extend(dict(zip(('run', 'subrun', 'event'), table_keys.T),
                                       payload=np.arange(20, dtype=np.int32)))
    original_receipt = directory / 'original-table-receipt.json'
    control.save(original_receipt, {'inputs': integrity.files([table, *sources]),
        'events': 28044, 'source_birth_identities_verified': True})
    runtime = Path(parents[0]['injection']) / 'runtime'
    plugins = runtime / 'plugins'
    plugins.mkdir(parents=True)
    plugin_files = []
    for name in ('StrawGasStepsFromTable', 'StrawGasStepTableDumper'):
        plugin = plugins / f'libtrkfastsim_{name}_module.so'
        plugin.write_bytes(f'completed F {name}'.encode())
        plugin_files.append(plugin)
    runner = runtime / 'offline_plugins.sh'
    runner.write_text('completed F frozen runner\n')
    control.save(runtime / 'runtime.json', {'plugins': integrity.files(plugin_files),
        'adapted_runner_sha256': integrity.digest(runner)['sha256']})
    completed = {'all_planned_parents_verified': True,
                 'injection_evidence': integrity.files([*plugin_files, runner, runtime / 'runtime.json'])}
    fast_plan = dict(events=28044, parents=parents, config=integrity.digest(config),
        digitizer_seed=seed,
        digitizer_seed_policy='reset_each_parent; correlated streams, not original production draws or independent replicas',
        runtime={'OFFLINE_IMAGE': '/frozen/offline.sif', 'TRIGGER_AREA': '/frozen/Offline',
                 'TRIGGER_PLUGINS': str(plugins)})
    fast_plan_path = directory / 'fast-plan.json'
    control.save(fast_plan_path, fast_plan)
    fast_prepared = directory / 'saveddecision'
    fast_prepared.mkdir()
    control.save(fast_prepared / 'manifest.json', {'production_history_receipt': {
        'parent_census': {'original_parent_files': integrity.files(exports)}}})
    (fast_prepared / 'replay.fcl').write_text('saved full-history replay configuration\n')
    fast_art = directory / 'saveddecision.art'
    fast_art.write_bytes(b'persisted full-history decision ledger')
    roots[fast_art] = _art_metadata(denominator)
    comparison = directory / 'saveddecision-comparison.json'
    control.save(comparison, {'events': 28044, 'all_decisions_equal': True})
    flow = {'events': 28044, 'paths': [f'path_{index}' for index in range(52)]}
    audit_calls, completed_calls, flow_calls = [], [], []
    real_open = uproot.open

    def open_root(path, *args, **kwargs):
        path = Path(path)
        return roots[path] if path in roots else real_open(path, *args, **kwargs)

    def audit(tables, keys):
        # The expensive numerical audit is mocked, not file binding or splitting.
        audit_calls.append(([str(path) for path in tables], len(keys)))
        return {'events': len(keys), 'inputs': integrity.files(tables),
                'event_keys_sha256': hashlib.sha256(keys.tobytes()).hexdigest()}

    def completed_reference(plan):
        completed_calls.append(deepcopy(plan))
        assert plan == fast_plan
        return completed

    def stageflow(art, cfg, prepared, saved_comparison):
        flow_calls.append((art, cfg, prepared, saved_comparison))
        assert (art, cfg, prepared, saved_comparison) == (fast_art, config, fast_prepared, comparison)
        return flow

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(control.uproot, 'open', open_root)
        patch.setattr(control.event_keys, 'read', lambda tree: tree.identities)
        patch.setattr(control.response_presence.serialized, 'entry_bytes',
                      lambda branch: ak.Array(np.ones((branch.num_entries, 1), dtype=np.uint8)))
        patch.setattr(control.injection_campaign, 'completed', completed_reference)
        patch.setattr(control.stage_flow, 'analyze', stageflow)
        patch.setattr(control.response_table, 'audit', audit)
        patch.setattr(control, 'IMPORTER', integrity.digest(plugin_files[0])['sha256'])
        patch.setattr(control, 'DUMPER', integrity.digest(plugin_files[1])['sha256'])
        output = directory / 'control'
        plan = control.create(fast_plan_path, [table], original_receipt, output,
                              fast_prepared, fast_art, comparison)
        yield SimpleNamespace(plan=plan, plan_path=output / 'plan.json', output=output,
            roots=roots, fast_art=fast_art, flow=flow, sources=sources, exports=exports,
            denominator=denominator, table=table, table_receipt=original_receipt,
            fast_plan_path=fast_plan_path, fast_prepared=fast_prepared, comparison=comparison,
            counts=counts, audit_calls=audit_calls, completed_calls=completed_calls, flow_calls=flow_calls)


def test_create_and_all_ten_preflights_bind_exact_presence_ledgers_and_split(completed_control):
    fixture = completed_control
    plan = fixture.plan
    assert control.load(fixture.plan_path) == plan
    assert plan['events'] == 28044 and len(plan['parents']) == 10
    assert len(fixture.completed_calls) == len(fixture.flow_calls) == 1
    assert plan['admission_kind'] == control.KIND and plan['scope'] == control.SCOPE
    for flag in ('primary_only_admission_used', 'electronics_only', 'particle_state_lossless',
                 'source_simparticle_product_ids_authenticated'):
        assert plan[flag] is False
    for filename, tag, sources in (
            ('export-source-presence.json', 'compressDetStepMCs::Primary', fixture.exports),
            ('source-presence.json', 'compressDigiMCs::Digitize', fixture.sources)):
        ledger = control.load(fixture.output / filename)
        assert ledger['schema'] == 'response-presence-parents-v1'
        assert ledger['tag'] == tag and ledger['all_present'] is True
        assert ledger['parents'] == 10 and ledger['events'] == ledger['total_events'] == 28044
        assert ledger['event_keys_numeric'] == fixture.denominator.tolist()
        assert [row['events'] for row in ledger['files']] == fixture.counts
        assert [row['source_before'] for row in ledger['files']] == integrity.files(sources)
        for row in ledger['files']:
            assert row['tag'] == row['product']['tag'] == tag
            assert row['product']['branch'] == 'mu2e::StrawGasSteps_' + tag.replace(':', '_') + '.'
            assert row['source_before'] == row['source_after']
            assert row['simparticles_presence_checked'] is False
            assert row['event_keys_physical'][0] == row['event_keys_numeric'][-1]
        assert integrity.digest(fixture.output / filename) in plan['evidence']
    assert plan['source_presence'] == control.load(fixture.output / 'source-presence.json')
    split = control.load(fixture.output / 'tables/split.json')
    assert split['events'] == 28044 and len(split['parents']) == 10
    assert split['export_tree_rows'] == {'steps': 20}
    assert split['source_files'] == integrity.files(fixture.sources)
    assert split['export_files'] == integrity.files([fixture.table])
    assert split['output_files'] == [row['table'] for row in plan['parents']]
    for ordinal, row in enumerate(plan['parents'], 1):
        verified, parent = control.preflight(fixture.plan_path, ordinal)
        assert verified == plan and parent == row
        assert row['ordinal'] == ordinal and row['events'] == fixture.counts[ordinal - 1]
        with uproot.open(row['table']['path']) as root:
            values = root['dump/steps'].arrays(library='np')
        assert values['run'].tolist() == [1440 - ordinal] * 2
        assert values['event'].tolist() == [1, row['events']]
        assert values['payload'].tolist() == [2 * (ordinal - 1), 2 * ordinal - 1]
        assert split['parents'][ordinal - 1]['tree_rows'] == {'steps': 2}
        manifest = control.load(Path(row['prepared']) / 'manifest.json')
        assert manifest['admission_kind'] == control.KIND
        assert manifest['primary_only_admission_used'] is False
    assert fixture.audit_calls[:11] == [([str(fixture.table)], 28044)] + [
        ([row['table']['path']], row['events']) for row in plan['parents']]


@pytest.mark.parametrize('mutation, message', [
    ('runtime', 'runtime or reset policy'),
    ('digitizer_seed', 'runtime or reset policy'),
    ('digitizer_seed_policy', 'runtime or reset policy'),
    ('config', 'runtime or reset policy'),
    ('parent-source', 'parent mapping'),
    ('parent-fast-prepared', 'parent mapping'),
    ('parent-events', 'parent mapping'),
    ('parent-order', 'parent order'),
    ('runtime-runner', 'runner not from completed F'),
    ('admission_kind', 'separate full-stock'),
    ('primary_only_admission_used', 'separate full-stock'),
    ('electronics_only', 'separate full-stock'),
    ('particle_state_lossless', 'separate full-stock'),
    ('source_simparticle_product_ids_authenticated', 'separate full-stock'),
])
def test_success_plan_mutations_rejected_preflight(completed_control, tmp_path, mutation, message):
    plan = deepcopy(completed_control.plan)
    if mutation == 'runtime':
        plan['runtime']['OFFLINE_IMAGE'] = '/different/offline.sif'
    elif mutation == 'digitizer_seed':
        plan[mutation] += 1
    elif mutation == 'digitizer_seed_policy':
        plan[mutation] = 'continuous stream; no parent reset'
    elif mutation == 'config':
        # A valid alternate hash receipt must still match the completed F config.
        config = tmp_path / 'alternate.fcl'
        config.write_text(CONFIG.replace('production-geometry', 'different-geometry'))
        plan['config'] = integrity.digest(config)
    elif mutation == 'parent-source':
        plan['parents'][0]['source'] = deepcopy(plan['parents'][1]['source'])
    elif mutation == 'parent-fast-prepared':
        plan['parents'][0]['fast_prepared'] = plan['parents'][1]['fast_prepared']
    elif mutation == 'parent-events':
        plan['parents'][0]['events'] -= 1
        plan['parents'][1]['events'] += 1
    elif mutation == 'parent-order':
        plan['parents'][0], plan['parents'][1] = plan['parents'][1], plan['parents'][0]
    elif mutation == 'runtime-runner':
        runner = tmp_path / 'offline_plugins.sh'
        runner.write_bytes(Path(plan['runtime_binding']['runner']['path']).read_bytes())
        plan['runtime_binding']['runner'] = integrity.digest(runner)
    else:
        plan[mutation] = 'primary-only' if mutation == 'admission_kind' else True
    path = tmp_path / 'mutated-plan.json'
    control.save(path, plan)
    audits_before = len(completed_control.audit_calls)
    with pytest.raises(ValueError, match=message):
        control.preflight(path, 1)
    assert len(completed_control.audit_calls) == audits_before


@pytest.mark.parametrize('evidence', ['source-presence.json', 'export-source-presence.json',
                                     'tables/split.json'])
def test_preflight_rejects_changed_presence_or_split_receipt(completed_control, evidence):
    path = completed_control.output / evidence
    original = path.read_bytes()
    try:
        path.write_bytes(original + b'\n')
        with pytest.raises(ValueError, match='control evidence.*changed'):
            control.preflight(completed_control.plan_path, 1)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize('mutation, message', [('decision-order', 'decision-art identities/order'),
                                              ('decision-paths', 'decision admission')])
def test_create_requires_saveddecision_order_and_full_path_ledger(completed_control, tmp_path, mutation, message):
    fixture = completed_control
    tree = fixture.roots[fixture.fast_art]['Events']
    original_keys, original_paths = tree.identities, fixture.flow['paths']
    try:
        if mutation == 'decision-order':
            tree.identities = tree.identities[::-1]
        else:
            fixture.flow['paths'] = fixture.flow['paths'][:-1]
        output = tmp_path / 'rejected-control'
        with pytest.raises(ValueError, match=message):
            control.create(fixture.fast_plan_path, [fixture.table], fixture.table_receipt, output,
                           fixture.fast_prepared, fixture.fast_art, fixture.comparison)
        assert not output.exists()
    finally:
        tree.identities, fixture.flow['paths'] = original_keys, original_paths
