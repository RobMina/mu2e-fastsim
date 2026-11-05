"""Separate response postflight gates using opaque art mocks and real numerical dump TTrees."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import integrity
import replay
import response_control as control
import response_roundtrip as rt
import response_validate as validator


TAG = 'compressDetStepMCs::Compress'
KEYS = np.array([[1430, 55, 2], [1430, 55, 9]], dtype='u8')


def save(path, value):
    path.write_text(json.dumps(value))


class Branch:
    def __init__(self, typename, values):
        self.typename, self.values, self.num_entries = typename, values, 2

    def array(self, **kwargs):
        assert kwargs == {'library': 'np'}
        assert self.typename == 'bool', 'never deserialize SGS or particle payloads'
        return self.values


class Parent(dict):
    num_entries = 2

    def keys(self, recursive=False):
        return super().keys()


class Tree(dict):
    classname, num_entries = 'TTree', 2

    def keys(self, recursive=False):
        return super().keys()


def add_product(tree, friendly, tag, typename, values):
    stem = friendly + '_' + tag.replace(':', '_')
    parent = Parent({stem + '.present': Branch('bool', np.ones(2, dtype='?')),
                     stem + '.obj': Branch(typename, ak.Array(values))})
    parent.typename = f'art::Wrapper<{typename}>'
    tree[stem + '.'] = parent
    return parent


def tree(physical, injected=False):
    result = Tree()
    result.identities = physical.copy()
    result.words = np.stack([np.arange(56, dtype='u2') + key[2] for key in physical]).astype('u2')
    for friendly, label in validator.hybrid_input.STOCK:
        add_product(result, friendly, f'{label}::Digitize', 'Value', [[int(k[2]), 42] for k in physical])
    # Includes pointer-like bytes; no assumption about meaning or resolvability.
    add_product(result, 'mu2e::StrawGasSteps', TAG, 'std::vector<mu2e::StrawGasStep>',
                [[0, int(k[2]), 255] for k in physical])
    if injected:
        add_product(result, 'mu2e::StrawGasSteps', validator.SGS_TAG,
                    'std::vector<mu2e::StrawGasStep>', [[0, 11], [0, 0]])
        for friendly in ('mu2e::StrawDigis', 'mu2e::StrawDigiADCWaveforms'):
            add_product(result, friendly, validator.DIGI_TAG, 'Value', [[1, 2], [0, 0]])
    return result


class Root:
    def __init__(self, latest):
        stale = Tree()
        stale.num_entries = 0
        self.trees = {'Events;1': stale, 'Events;7': latest}
        self.file = SimpleNamespace(streamers={})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def keys(self, recursive=False):
        return self.trees.keys()

    def __getitem__(self, name):
        if name == 'Events':
            name = 'Events;' + str(max(int(k.split(';')[1]) for k in self.trees))
        return self.trees[name]


def table_payload(zero=False):
    def columns(schema, which):
        result = {name: np.zeros(len(which), dtype=dtype) for name, dtype in schema.items()}
        for i, name in enumerate(rt.KEYS):
            result[name] = KEYS[which, i].astype(schema[name])
        return result
    gen = columns(rt.GEN_SCHEMA, [0, 1])
    gen['pdg'][:] = 11
    gen['e'][:] = 105
    sgs = columns(rt.SGS_SCHEMA, [] if zero else [0])
    sgs['nsgs'][:] = 1
    sgs['eion'][:] = 1.25
    particles = columns(rt.PARTICLE_SCHEMA, [0, 1])
    particles['pdg'][:] = 11
    particles['parent'][:] = -1
    particles['ismcprim'][:] = 1
    return {'gen': gen, 'sgsteps': sgs, 'particles': particles}


def write_table(path, payload):
    with uproot.recreate(path) as root:
        for name, columns in payload.items():
            root.mktree('dump/' + name, {k: v.dtype for k, v in columns.items()})
            root['dump/' + name].extend(columns)


def config_text():
    physics = {'producers': {'makeSD': {'module_type': 'StrawDigisFromStrawGasSteps',
                    'StrawGasStepModules': ['compressDetStepMCs'], 'AllHitsStraw': replay.Atom('91')}},
               'trigger_paths': [f'p{i}' for i in range(56)]}
    physics.update({f'p{i}': [] for i in range(56)})
    return control.render({'Digitize': {'services': {
        'GeometryService': {'inputFile': 'production'}, 'SeedService': {'baseSeed': replay.Atom('9')},
        'DbService': {'purpose': 'production'}, 'RandomNumberGenerator': {},
        'OtherElectronics': {'threshold': replay.Atom('0.03')}}, 'physics': physics}})


@pytest.fixture
def sample(tmp_path, monkeypatch):
    directory, prepared = tmp_path / 'execution', tmp_path / 'prepared'
    directory.mkdir()
    prepared.mkdir()
    source, candidate, config = tmp_path / 'source.art', directory / 'unfiltered_1.art', tmp_path / 'production.fcl'
    source.write_bytes(b'opaque original art bytes')
    candidate.write_bytes(b'opaque injected art bytes')
    config.write_text(config_text())
    table = tmp_path / 'parent.root'
    write_table(table, table_payload())
    admission = prepared / 'table-admission.json'
    save(admission, {'events': 2})
    _, manifest, text = control.intended(config.read_text(), source, table, 12345)
    (prepared / 'job.fcl').write_text(text)
    save(prepared / 'manifest.json', manifest)
    (directory / 'inject.expanded.fcl').write_text(text.replace('services:', '&services:', 1))
    runtime = tmp_path / 'completed-F/runtime'
    plugins = runtime / 'plugins'
    plugins.mkdir(parents=True)
    for name in ('StrawGasStepsFromTable', 'StrawGasStepTableDumper', 'StrawDigiTableDumper'):
        (plugins / f'libtrkfastsim_{name}_module.so').write_bytes(name.encode())
    runner, receipt = runtime / 'offline_plugins.sh', runtime / 'runtime.json'
    runner.write_bytes(b'frozen runner bytes')
    receipt.write_text('{}')
    bound = {'plugins': integrity.files(sorted(plugins.glob('*.so'))),
             'runner': integrity.digest(runner), 'receipt': integrity.digest(receipt)}
    row = {'ordinal': 1, 'events': 2, 'source': integrity.digest(source), 'table': integrity.digest(table),
           'table_admission': integrity.digest(admission), 'prepared': str(prepared)}
    roots = {source: Root(tree(KEYS[::-1])), candidate: Root(tree(KEYS, True))}
    original_open = uproot.open
    monkeypatch.setattr(validator.uproot, 'open', lambda path, **kwargs:
                        roots[Path(path)] if Path(path) in roots else original_open(path, **kwargs))
    monkeypatch.setattr(validator.event_keys, 'read', lambda value: value.identities)
    monkeypatch.setattr(validator.serialized, 'entry_bytes', lambda branch: branch.values)
    monkeypatch.setattr(validator.path_state, 'check_schema', lambda root: None)

    def status(value, process, expected_paths):
        assert process == 'Digitize'
        if value.words.shape[1] != expected_paths:
            raise ValueError('wrong packed word count')
        return value.words, 'original-schedule-id'
    monkeypatch.setattr(validator.path_state, 'read_status', status)
    monkeypatch.setattr(validator.hybrid_input.schedule_registry, 'verify',
                        lambda *args: {'registry_schedule_verified': True})
    presence = control.response_presence.admit([source], [2], TAG)
    plan = {'admission_kind': control.KIND, 'parents': [row], 'events': 2,
            'config': integrity.digest(config), 'runtime_binding': bound,
            'runtime': {'OFFLINE_IMAGE': '/pinned/production.sif', 'TRIGGER_AREA': '/pinned/area'},
            'source_presence': presence}
    plan_path = tmp_path / 'plan.json'
    save(plan_path, plan)
    calls = []

    def preflight(path, ordinal):
        assert Path(path) == plan_path
        calls.append(ordinal)
        if type(ordinal) is not int or ordinal != 1:
            raise ValueError('invalid original parent ordinal')
        return plan, row
    monkeypatch.setattr(control, 'preflight', preflight)
    save(directory / 'execution-binding.json', {'plan': integrity.digest(plan_path), 'parent': row,
        'runtime': plan['runtime'], 'scope': control.SCOPE, 'admission_kind': control.KIND})
    dump = directory / 'roundtrip'
    dump.mkdir()
    dump_prepared = directory / 'dump-prepared'
    control.dump_prepare(config.read_text(), candidate, dump_prepared)
    (dump / 'dump.expanded.fcl').write_text((dump_prepared / 'job.fcl').read_text())
    write_table(dump / 'source-tables.root', table_payload())
    for location, prefix in ((directory, 'inject'), (dump, 'dump')):
        save(location / 'runtime-selection.json', {'runner': str(runner), 'plugin_directory': str(plugins)})
        (location / 'art.exit').write_text('0\n')
        (location / f'{prefix}.log').write_text('start image=/pinned/production.sif\n'
            'MUSE_WORK_DIR=/pinned/area MUSE_ENVSET=production\n' + validator.SUCCESS + '\n')
    return SimpleNamespace(directory=directory, prepared=prepared, dump=dump, dump_prepared=dump_prepared,
        source=source, candidate=candidate, roots=roots, a=roots[source]['Events'], b=roots[candidate]['Events'],
        table=table, plan=plan, row=row, plan_path=plan_path, calls=calls, runner=runner, plugins=plugins)


def run(sample):
    return validator.validate(sample.plan_path, 1, sample.directory)


def product(value, tag=TAG, suffix='.obj'):
    stem = 'mu2e::StrawGasSteps_' + tag.replace(':', '_')
    return value[stem + '.'][stem + suffix]


def rewrite_fcl(path, change):
    config = validator._configuration(path)
    change(config)
    path.write_text(control.render(config))


def test_success_all_evidence_and_no_strong_promotion(sample):
    report = run(sample)
    assert sample.calls == [1, 1], 'both injection and dump must run separate control preflight'
    assert report['validated'] and report['numericalSGS']
    assert not report['actual_consumed_ProductID']
    assert not report['original_SimParticle_ProductID_resolved']
    assert not report['particle_state_lossless'] and not report['electronics_only']
    assert report['highest_finished_Events_cycles'] == {'source': 7, 'candidate': 7}
    assert report['event_keys_numeric_original_parent_order'] == KEYS.tolist()
    assert report['original_SGS_serialized_bytes_preserved']
    assert report['consumes_configuration_exact']
    assert report['selected_producer_configuration']['StrawGasStepModules'] == [validator.SGS_TAG]
    assert len(report['handoff']['files'][0]['stock_products']) == 6
    assert report['roundtrip']['comparison']['zero_step_events'] == {'original': 1, 'imported': 1}
    assert report['roundtrip']['comparison']['sgs']['equal']
    assert report['inputs_before'] == report['inputs_after']
    assert len(report['outputs']) == 2 and all(row['bytes'] > 0 for row in report['outputs'])
    assert not list(sample.directory.glob('*validation*.json')), 'API is read-only'
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('location,prefix', [('directory', 'inject'), ('dump', 'dump')])
@pytest.mark.parametrize('change', ['exit', 'success', 'image', 'area', 'prefix-image', 'prefix-area'])
def test_both_jobs_require_successful_exact_runtime(sample, location, prefix, change):
    directory = getattr(sample, location)
    if change == 'exit':
        (directory / 'art.exit').write_text('1')
    else:
        path = directory / f'{prefix}.log'
        text = path.read_text()
        old, new = {'success': (validator.SUCCESS, 'not finished'),
                    'image': ('production.sif', 'wrong.sif'), 'area': ('/pinned/area', '/wrong/area'),
                    'prefix-image': ('production.sif', 'production.sif.other'),
                    'prefix-area': ('/pinned/area ', '/pinned/area.other ')}[change]
        path.write_text(text.replace(old, new))
    with pytest.raises(ValueError):
        run(sample)


@pytest.mark.parametrize('change', ['service', 'extra-service', 'source', 'producer', 'schedule',
                                   'output', 'extra-key', 'ambiguous-services'])
def test_all_executed_injection_semantics_are_exact(sample, change):
    path = sample.directory / 'inject.expanded.fcl'
    def modify(config):
        if change == 'service':
            config['services']['OtherElectronics']['threshold'] = replay.Atom('0.03000000001')
        elif change == 'extra-service':
            config['services']['Unadmitted'] = {}
        elif change == 'source':
            config['source']['fileNames'] = ['another-parent.art']
        elif change == 'producer':
            config['physics']['producers']['R7FastSD']['StrawGasStepModules'] = ['compressDetStepMCs']
        elif change == 'schedule':
            config['physics']['Inject'] = ['R7FastSteps']
        elif change == 'output':
            config['outputs']['Combined']['outputCommands'] = ['drop *']
        elif change == 'extra-key':
            config['unadmitted'] = 'anything'
        else:
            config['&services'] = config['services']
    rewrite_fcl(path, modify)
    with pytest.raises(ValueError, match='executed|ambiguous'):
        run(sample)


def test_semantic_numbers_not_float_tolerance(sample):
    path = sample.directory / 'inject.expanded.fcl'
    path.write_text(path.read_text().replace('0.03', '3e-2').replace('12345', '1.2345e4'))
    assert run(sample)['validated']


@pytest.mark.parametrize('change', ['plan', 'parent', 'runtime', 'scope', 'extra'])
def test_execution_receipt_is_exact_preflight_cli_output(sample, change):
    path = sample.directory / 'execution-binding.json'
    binding = control.load(path)
    if change == 'plan':
        binding['plan']['sha256'] = '0' * 64
    elif change == 'parent':
        binding['parent']['ordinal'] = 2
    elif change == 'runtime':
        binding['runtime']['OFFLINE_IMAGE'] = '/other.sif'
    elif change == 'scope':
        binding['scope'] = 'electronics only'
    else:
        binding['extra'] = True
    save(path, binding)
    with pytest.raises(ValueError, match='execution-binding'):
        run(sample)


@pytest.mark.parametrize('location', ['directory', 'dump'])
@pytest.mark.parametrize('change', ['runner', 'bundle', 'extra-plugin', 'changed-plugin',
                                   'runner-byte', 'runtime-receipt-byte', 'generated-runtime'])
def test_direct_completed_F_runner_and_exact_plugin_bytes(sample, tmp_path, location, change):
    directory = getattr(sample, location)
    selection = directory / 'runtime-selection.json'
    if change == 'runner':
        other = tmp_path / 'other-runner.sh'
        other.write_bytes(sample.runner.read_bytes())
        value = control.load(selection)
        value['runner'] = str(other)
        save(selection, value)
    elif change == 'bundle':
        value = control.load(selection)
        value['plugin_directory'] = str(tmp_path / 'new-plugins')
        save(selection, value)
    elif change == 'extra-plugin':
        (sample.plugins / 'libextra.so').write_bytes(b'new')
    elif change == 'changed-plugin':
        next(sample.plugins.glob('*.so')).write_bytes(b'changed')
    elif change == 'runner-byte':
        sample.runner.write_bytes(b'changed runner')
    elif change == 'runtime-receipt-byte':
        sample.runner.with_name('runtime.json').write_text('{"changed": true}')
    else:
        (directory / 'runtime').mkdir()
    with pytest.raises(ValueError):
        run(sample)


@pytest.mark.parametrize('change', ['order', 'duplicate', 'foreign', 'negative', 'count', 'latest-empty',
                                   'latest-not-tree', 'split-output', 'words-count', 'word-change'])
def test_complete_highest_finished_population_original_order_and_56_words(sample, change):
    if change == 'order':
        sample.b.identities = sample.b.identities[::-1]
    elif change == 'duplicate':
        sample.b.identities[1] = sample.b.identities[0]
    elif change == 'foreign':
        sample.b.identities[1, 2] = 10
    elif change == 'negative':
        sample.b.identities = sample.b.identities.astype('i8')
        sample.b.identities[0, 0] = -1
    elif change == 'count':
        sample.b.num_entries = 1
    elif change == 'latest-empty':
        extra = Tree()
        extra.num_entries = 0
        sample.roots[sample.candidate].trees['Events;8'] = extra
    elif change == 'latest-not-tree':
        sample.b.classname = 'ROOT::RNTuple'
    elif change == 'split-output':
        (sample.directory / 'unfiltered_2.art').write_bytes(b'split')
    elif change == 'words-count':
        sample.b.words = sample.b.words[:, :52]
    else:
        sample.b.words[1, 55] += 1
    with pytest.raises(ValueError):
        run(sample)


@pytest.mark.parametrize('change', ['source-SGS-byte', 'injected-missing', 'injected-absent',
                                   'injected-presence-dtype', 'injected-empty-serialization',
                                   'injected-type', 'injected-count', 'stock-byte', 'stock-absent'])
def test_presence_original_raw_SGS_and_six_retained_products(sample, change):
    if change == 'source-SGS-byte':
        product(sample.b).values = ak.Array([[0, 2, 254], [0, 9, 255]])
    elif change == 'injected-missing':
        del sample.b['mu2e::StrawGasSteps_R7FastSteps__R7TrackerInject.']
    elif change == 'injected-absent':
        product(sample.b, validator.SGS_TAG, '.present').values[1] = False
    elif change == 'injected-presence-dtype':
        product(sample.b, validator.SGS_TAG, '.present').values = np.ones(2, dtype='u1')
    elif change == 'injected-empty-serialization':
        product(sample.b, validator.SGS_TAG).values = ak.Array([[1], []])
    elif change == 'injected-type':
        product(sample.b, validator.SGS_TAG).typename = 'wrong'
    elif change == 'injected-count':
        product(sample.b, validator.SGS_TAG).num_entries = 1
    else:
        stem = 'mu2e::CaloDigis_CaloDigiMaker__Digitize'
        parent = sample.b[stem + '.']
        if change == 'stock-byte':
            parent[stem + '.obj'].values = ak.Array([[2, 43], [9, 42]])
        else:
            parent[stem + '.present'].values[0] = False
    with pytest.raises(ValueError):
        run(sample)


def test_original_exact_tag_cannot_fall_back_to_different_process(sample):
    sample.plan['source_presence']['tag'] = 'compressDetStepMCs::Other'
    with pytest.raises(ValueError, match='exact original SGS presence'):
        run(sample)


@pytest.mark.parametrize('change', ['explicit-tag', 'source', 'producer', 'trigger', 'services', 'manifest'])
def test_dump_is_bound_explicit_tag_read_only_preparation(sample, change):
    if change == 'manifest':
        path = sample.dump_prepared / 'manifest.json'
        value = control.load(path)
        value['scope'] = 'lossless'
        save(path, value)
    else:
        path = sample.dump / 'dump.expanded.fcl'
        def modify(config):
            if change == 'explicit-tag':
                config['physics']['analyzers']['dump']['strawGasSteps'] = 'compressDetStepMCs'
            elif change == 'source':
                config['source']['fileNames'] = [str(sample.source)]
            elif change == 'producer':
                config['physics']['producers'] = {'rerun': {'module_type': 'StrawDigisFromStrawGasSteps'}}
            elif change == 'trigger':
                config['physics']['trigger_paths'] = ['newPath']
            else:
                config['services']['DbService']['purpose'] = 'other'
        rewrite_fcl(path, modify)
    with pytest.raises(ValueError):
        run(sample)


@pytest.mark.parametrize('change', ['sgs-value', 'missing-zero-event', 'generated-value'])
def test_numerical_report_false_is_fatal(sample, change):
    payload = table_payload()
    if change == 'sgs-value':
        payload['sgsteps']['t'][0] = 1.
    elif change == 'missing-zero-event':
        for name in ('gen', 'particles'):
            payload[name] = {k: v[:1] for k, v in payload[name].items()}
    else:
        payload['gen']['e'][1] += 1.
    write_table(sample.dump / 'source-tables.root', payload)
    with pytest.raises(ValueError, match='roundtrip is not equal'):
        run(sample)


def test_reexported_originalPrimaryParticle_flags_are_diagnostic_not_gating(sample):
    payload = table_payload()
    payload['particles']['ismcprim'][:] = 0
    payload['particles']['ccode'][:] = 99
    write_table(sample.dump / 'source-tables.root', payload)
    report = run(sample)
    particles = report['roundtrip']['comparison']['particles']
    assert not particles['equal'] and not particles['gating']
    assert report['numericalSGS']
    assert not report['roundtrip']['reexported_originalPrimaryParticle_flags_gating']


def test_zero_SGS_events_remain_present_and_compare_even_if_all_zero(sample):
    payload = table_payload(zero=True)
    write_table(sample.table, payload)
    write_table(sample.dump / 'source-tables.root', payload)
    report = run(sample)
    assert report['events'] == 2 and report['injected_SGS']['present_all_events']
    assert report['roundtrip']['comparison']['zero_step_events'] == {'original': 2, 'imported': 2}
    assert report['roundtrip']['comparison']['sgs']['matched_rows'] == 0
    product(sample.b, validator.SGS_TAG, '.present').values[0] = False
    with pytest.raises(ValueError, match='presence'):
        run(sample)


def test_preflight_is_required_before_any_art_reader(sample, monkeypatch):
    def fail(*args):
        raise ValueError('control preflight rejection')
    monkeypatch.setattr(control, 'preflight', fail)
    monkeypatch.setattr(validator.uproot, 'open', lambda *args, **kwargs: pytest.fail('read before admission'))
    with pytest.raises(ValueError, match='control preflight rejection'):
        run(sample)


def test_evidence_changes_during_comparison_fail(sample, monkeypatch):
    original = rt.compare
    def compare(*args):
        report = original(*args)
        (sample.dump / 'dump.log').write_text('modified after read')
        return report
    monkeypatch.setattr(rt, 'compare', compare)
    with pytest.raises(ValueError, match='evidence bytes changed'):
        run(sample)


def cli(sample, monkeypatch, output, *extras):
    monkeypatch.setattr(sys, 'argv', ['response_validate.py', '--plan', str(sample.plan_path),
        '--ordinal', '1', '--directory', str(sample.directory), '--output', str(output), *extras])
    validator.main()


def test_cli_success_nooverwrite_and_dump_only(sample, monkeypatch, tmp_path):
    output = tmp_path / 'report.json'
    cli(sample, monkeypatch, output)
    assert control.load(output)['numericalSGS']
    prior = output.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        cli(sample, monkeypatch, output)
    assert output.read_bytes() == prior
    dump_report = tmp_path / 'dump-report.json'
    cli(sample, monkeypatch, dump_report, '--dump-only')
    assert control.load(dump_report)['equal']


def test_cli_refuses_dangling_symlink_without_reading(sample, monkeypatch, tmp_path):
    output = tmp_path / 'report.json'
    output.symlink_to(tmp_path / 'missing')
    monkeypatch.setattr(control, 'preflight', lambda *args: pytest.fail('must check overwrite first'))
    with pytest.raises(ValueError, match='overwrite'):
        cli(sample, monkeypatch, output)
    assert output.is_symlink()


def test_cli_failed_gate_writes_no_report(sample, monkeypatch, tmp_path):
    (sample.dump / 'art.exit').write_text('42')
    output = tmp_path / 'report.json'
    with pytest.raises(ValueError):
        cli(sample, monkeypatch, output)
    assert not output.exists()
