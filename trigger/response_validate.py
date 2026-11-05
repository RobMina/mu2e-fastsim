"""Postflight for one separately admitted full-stock SGS response-control parent.

validate(plan_path, ordinal, directory) returns evidence, never writes it. Required
in directory: execution-binding.json (response_control preflight CLI), art.exit,
inject.log, inject.expanded.fcl, unfiltered_1.art and runtime-selection.json.
The latter is exactly {"runner": "/completed-F/runtime/offline_plugins.sh",
"plugin_directory": "/completed-F/runtime/plugins"}; use those files directly,
not a freshly generated plugin runtime. dump-prepared/{job.fcl,manifest.json}
is made by response_control prepare-dump with this candidate. roundtrip/ contains
art.exit, dump.log, dump.expanded.fcl, source-tables.root, runtime-selection.json.
Both jobs use the same completed-F runner/bundle and pinned runtime. No trigger
rerun, ProductID authentication, lossless-particle or electronics-only claim.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

import event_keys
import hybrid_input
import injection_campaign
import integrity
import path_state
import replay
import response_control
import response_presence
import response_roundtrip
import serialized
import step_handoff

SGS_TAG = 'R7FastSteps::R7TrackerInject'
DIGI_TAG = 'R7FastSD::R7TrackerInject'
SUCCESS = 'Art has completed and will exit with status 0.'
DUMP_SCOPE = 'read-only imported SGS numerical roundtrip; no digitization or trigger rerun'


def _load(path):
    return response_control.load(path)


def _new(path):
    if path.exists() or path.is_symlink():
        raise ValueError('refuse response validation report overwrite')


def _configuration(path):
    result = replay.parse_expanded(path.read_text())
    # Some art config dumps name the service table &services. It is the same
    # table, not permission to ignore it or accept two conflicting service sets.
    if '&services' in result:
        if 'services' in result:
            raise ValueError('ambiguous executed services and &services')
        result['services'] = result.pop('&services')
    return result


def _executed(prepared, expanded):
    intended, executed = _configuration(prepared), _configuration(expanded)
    if injection_campaign.semantic(executed) != injection_campaign.semantic(intended):
        raise ValueError('executed configuration differs from prepared job (including all services and keys)')
    return executed


def _runtime(plan, directory, prefix):
    bound = plan['runtime_binding']
    selection = _load(directory / 'runtime-selection.json')
    plugins = bound['plugins']
    parents = {Path(item['path']).resolve().parent for item in plugins}
    if len(parents) != 1:
        raise ValueError('need one completed F plugin bundle')
    bundle = parents.pop()
    expected = {'runner': str(Path(bound['runner']['path']).resolve()),
                'plugin_directory': str(bundle)}
    if (not isinstance(selection, dict) or set(selection) != set(expected)
            or any(not isinstance(selection[key], str) or
                   str(Path(selection[key]).resolve()) != value for key, value in expected.items())):
        raise ValueError('selected runtime is not the direct completed F runner and plugin bundle')
    if (directory / 'runtime').exists() or (directory / 'runtime').is_symlink():
        raise ValueError('new plugin runtime generation is not allowed for response control')
    actual_plugins = integrity.files(sorted(bundle.glob('*.so')))
    response_control.same_files(sorted(plugins, key=lambda item: item['path']), actual_plugins,
                                'selected completed F plugins')
    for key in ('runner', 'receipt'):
        response_control.same_files([bound[key]], integrity.files(bound[key]['path']), 'selected F ' + key)
    if (directory / 'art.exit').read_text().strip() != '0':
        raise ValueError('art did not exit zero')
    log = (directory / f'{prefix}.log').read_text()
    for token in (f"image={plan['runtime']['OFFLINE_IMAGE']}",
                  f"MUSE_WORK_DIR={plan['runtime']['TRIGGER_AREA']}"):
        if re.search(r'(?<!\S)' + re.escape(token) + r'(?=\s|$)', log) is None:
            raise ValueError('log lacks exact pinned runtime image or work area')
    if SUCCESS not in log:
        raise ValueError('log lacks successful art completion marker')
    return {'selection': selection, 'binding': bound, 'art_exit': 0, 'success_marker': SUCCESS}


def _keys(tree, expected):
    keys = event_keys.read(tree)
    if (type(expected) is not int or expected <= 0 or tree.num_entries != expected
            or not isinstance(keys, np.ndarray) or keys.shape != (expected, 3)
            or keys.dtype.kind not in 'iu' or np.any(keys < 0)
            or np.any(keys[:, [0, 2]] == 0) or np.any(keys >= 2**32 - 1)
            or len(np.unique(keys, axis=0)) != expected):
        raise ValueError('invalid finished Events count or numeric event identities')
    return keys


def wrapper(tree, friendly_type, tag):
    """General exact-tag persisted wrapper check; never deserialize pointer payloads.

    Unlike response_presence discovery, this also accepts the injected label.
    A present-empty collection is legal: its serialized framing is nonempty.
    """
    if not isinstance(tag, str) or not hybrid_input.TAG.fullmatch(tag):
        raise ValueError('wrapper requires explicit label:instance:process tag')
    stem = friendly_type + '_' + tag.replace(':', '_')
    if list(tree.keys(recursive=False)).count(stem + '.') != 1:
        raise ValueError('missing or ambiguous exact wrapper branch: ' + tag)
    parent = tree[stem + '.']
    children = list(parent.keys(recursive=False))
    if any(children.count(stem + suffix) != 1 for suffix in ('.present', '.obj')):
        raise ValueError('missing or ambiguous wrapper object/present child')
    present, obj = parent[stem + '.present'], parent[stem + '.obj']
    if any(branch.num_entries != tree.num_entries for branch in (parent, present, obj)):
        raise ValueError('wrapper entry count differs from finished Events')
    canonical = lambda text: ''.join(text.split()).replace('std::', '')
    typename = canonical(obj.typename)
    if canonical(parent.typename) != f'art::Wrapper<{typename}>':
        raise ValueError('unsupported wrapper typename')
    if friendly_type == response_presence.FRIENDLY_TYPE and typename not in response_presence.OBJECT_TYPES:
        raise ValueError('unsupported SGS object typename')
    flags = present.array(library='np')
    if (not isinstance(flags, np.ndarray) or np.ma.isMaskedArray(flags)
            or flags.dtype != np.dtype('bool') or flags.shape != (tree.num_entries,)
            or not np.all(flags)):
        raise ValueError('absent or malformed wrapper presence')
    raw = serialized.entry_bytes(obj)
    if len(raw) != tree.num_entries or np.any(ak.to_numpy(ak.num(raw, axis=1)) <= 0):
        raise ValueError('empty serialized entry or incorrect SGS count')
    return raw, obj.typename


def _base_files(plan_path, plan, row):
    files = [Path(plan_path), Path(plan['config']['path']), Path(row['source']['path']),
             Path(row['table']['path']), Path(row['table_admission']['path']),
             Path(row['prepared']) / 'job.fcl', Path(row['prepared']) / 'manifest.json']
    files += [Path(item['path']) for item in plan.get('evidence', []) + plan.get('original_tables', [])]
    if 'fast_plan' in plan:
        files.append(Path(plan['fast_plan']['path']))
    if 'fast_prepared' in row:
        files.append(Path(row['fast_prepared']) / 'job.fcl')
    bound = plan['runtime_binding']
    files += [Path(item['path']) for item in bound['plugins'] + [bound['runner'], bound['receipt']]]
    return files


def _snapshot(paths):
    return integrity.files(sorted({Path(path).resolve() for path in paths}))


def _unchanged(before):
    after = integrity.files([item['path'] for item in before])
    if after != before:
        raise ValueError('evidence bytes changed during response postflight')
    return after


def _original_keys(plan, row):
    with uproot.open(row['source']['path']) as root:
        tree, cycle = response_presence._latest(root)
        keys = _keys(tree, row['events'])
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return keys[order], cycle


def validate_dump(plan_path, ordinal, directory):
    """Validate directory/dump-prepared and directory/roundtrip; require exact numerical equality.

    Includes zero-SGS events via original event keys. Particle diagnostics (including
    originalPrimaryParticle/ismcprim reconstruction) are not equality gates.
    """
    plan, row = response_control.preflight(plan_path, ordinal)
    directory = Path(directory)
    prepared, dump = directory / 'dump-prepared', directory / 'roundtrip'
    candidate, table = directory / 'unfiltered_1.art', dump / 'source-tables.root'
    paths = _base_files(plan_path, plan, row) + [candidate, table, prepared / 'job.fcl',
        prepared / 'manifest.json', dump / 'runtime-selection.json', dump / 'art.exit',
        dump / 'dump.log', dump / 'dump.expanded.fcl']
    before = _snapshot(paths)
    keys, _ = _original_keys(plan, row)
    with uproot.open(candidate) as root:
        tree, _ = response_presence._latest(root)
        if not np.array_equal(_keys(tree, row['events']), keys):
            raise ValueError('dump candidate does not retain original numeric parent order')
    result, receipt = step_handoff.prepare(Path(plan['config']['path']).read_text(), [str(candidate)])
    result['physics']['analyzers']['dump'].update(strawGasSteps=SGS_TAG, simParticles=SGS_TAG,
        genParticles='compressDigiMCs::Digitize', surfaceSteps=SGS_TAG, vdSteps='R7FastSteps:virtualdetector:R7TrackerInject')
    text = response_control.render(result)
    expected = dict(receipt, fcl_sha256=hashlib.sha256(text.encode()).hexdigest(), scope=DUMP_SCOPE)
    if (prepared / 'job.fcl').read_text() != text or _load(prepared / 'manifest.json') != expected:
        raise ValueError('dump preparation differs from explicit-tag read-only configuration')
    executed = _executed(prepared / 'job.fcl', dump / 'dump.expanded.fcl')
    runtime = _runtime(plan, dump, 'dump')
    report = response_roundtrip.compare([Path(row['table']['path'])], table, keys)
    if report.get('equal') is not True or report.get('events') != row['events']:
        raise ValueError('numerical SGS/generated-particle roundtrip is not equal')
    return {'equal': True, 'events': row['events'], 'comparison': report, 'runtime': runtime,
            'explicit_dump_configuration': replay.plain(executed),
            'inputs_before': before, 'inputs_after': _unchanged(before),
            'output': integrity.digest(table), 'numericalSGS': True, 'actual_consumed_ProductID': False,
            'original_SimParticle_ProductID_resolved': False, 'particle_state_lossless': False,
            'electronics_only': False, 'scope': DUMP_SCOPE,
            'reexported_originalPrimaryParticle_flags_gating': False}


def validate(plan_path, ordinal, directory):
    """Validate exactly one parent, returning a strict-JSON-compatible evidence report."""
    plan, row = response_control.preflight(plan_path, ordinal)
    directory = Path(directory)
    source, candidate = Path(row['source']['path']), directory / 'unfiltered_1.art'
    if sorted(path.name for path in directory.glob('unfiltered_*.art')) != ['unfiltered_1.art']:
        raise ValueError('response parent output missing or split across unexpected files')
    paths = _base_files(plan_path, plan, row) + [candidate, directory / 'execution-binding.json',
        directory / 'runtime-selection.json', directory / 'art.exit', directory / 'inject.log',
        directory / 'inject.expanded.fcl']
    before = _snapshot(paths)
    binding = {'plan': integrity.digest(plan_path), 'parent': row, 'runtime': plan['runtime'],
               'scope': response_control.SCOPE, 'admission_kind': response_control.KIND}
    if _load(directory / 'execution-binding.json') != binding:
        raise ValueError('execution-binding differs from response_control preflight CLI receipt')
    runtime = _runtime(plan, directory, 'inject')
    config = _executed(Path(row['prepared']) / 'job.fcl', directory / 'inject.expanded.fcl')
    producers = replay.plain(config['physics']['producers'])
    if (producers['R7FastSD']['module_type'] != 'StrawDigisFromStrawGasSteps'
            or producers['R7FastSD']['StrawGasStepModules'] != [SGS_TAG]
            or producers['R7FastSteps']['module_type'] != 'StrawGasStepsFromTable'):
        raise ValueError('actual selected producer does not consume the exact injected SGS tag')
    original, original_cycle = _original_keys(plan, row)
    presence = plan['source_presence']
    receipt = presence['files'][ordinal - 1]
    tag = presence['tag']
    if receipt.get('tag') != tag or presence.get('all_present') is not True:
        raise ValueError('plan lacks consistent exact original SGS presence')
    source_presence = response_presence.verify(source, receipt, row['events'], tag=tag)
    if not np.array_equal(np.asarray(source_presence['event_keys_numeric']), original):
        raise ValueError('source presence denominator differs from original numeric parent order')
    classes = dict(uproot.classes, **{'art::HLTGlobalStatus': path_state.Model_art_3a3a_HLTGlobalStatus})
    with uproot.open(source, custom_classes=classes) as a, uproot.open(candidate, custom_classes=classes) as b:
        ta, _ = response_presence._latest(a)
        tb, cycle = response_presence._latest(b)
        ka, kb = _keys(ta, row['events']), _keys(tb, row['events'])
        if not np.array_equal(kb, original):
            raise ValueError('candidate Events differ from original numeric per-parent order')
        order = np.lexsort((ka[:, 2], ka[:, 1], ka[:, 0]))
        raw_a, type_a = wrapper(ta, response_presence.FRIENDLY_TYPE, tag)
        raw_b, type_b = wrapper(tb, response_presence.FRIENDLY_TYPE, tag)
        if type_a != type_b or not np.all(serialized.serialized_equal(raw_a[order], raw_b)):
            raise ValueError('original SGS serialized raw bytes changed')
        _, injected_type = wrapper(tb, response_presence.FRIENDLY_TYPE, SGS_TAG)
        # hybrid_input.audit checks all six retained products, all packed words,
        # schedule registry and replacement digi presence. Pin the original 56-word
        # menu here as well; the 52 decision paths alone are not the full menu.
        production = replay.production_config(Path(plan['config']['path']).read_text(), 'Digitize')
        if len(replay.paths(production)) != 56:
            raise ValueError('expected original 56-word Digitize menu')
        for root, tree in ((a, ta), (b, tb)):
            path_state.check_schema(root)
            words, _ = path_state.read_status(tree, 'Digitize', 56)
            if words.shape != (row['events'], 56):
                raise ValueError('original TriggerResults must retain all 56 words')
    handoff = hybrid_input.audit([source], [candidate], Path(plan['config']['path']).read_text(),
                                 row['events'], straw_tag=DIGI_TAG)
    if handoff.get('same_event_handoff_metadata_passed') is not True or handoff.get('events') != row['events']:
        raise ValueError('original six-product/56-word handoff failed')
    roundtrip = validate_dump(plan_path, ordinal, directory)
    return {'schema': 'full-stock-response-postflight-v1', 'validated': True, 'ordinal': ordinal,
        'events': row['events'], 'admission_kind': response_control.KIND, 'scope': response_control.SCOPE,
        'plan': integrity.digest(plan_path), 'parent': row, 'execution_binding': binding, 'runtime': runtime,
        'highest_finished_Events_cycles': {'source': original_cycle, 'candidate': cycle},
        'event_keys_numeric_original_parent_order': original.tolist(), 'complete_parent_order_verified': True,
        'source_presence': source_presence, 'original_SGS_serialized_bytes_preserved': True,
        'injected_SGS': {'tag': SGS_TAG, 'present_all_events': True, 'typename': injected_type},
        'selected_producer_configuration': producers['R7FastSD'], 'consumes_configuration_exact': True,
        'handoff': handoff, 'roundtrip': roundtrip, 'numericalSGS': True,
        'actual_consumed_ProductID': False, 'original_SimParticle_ProductID_resolved': False,
        'source_simparticle_product_ids_authenticated': False, 'particle_state_lossless': False,
        'electronics_only': False, 'primary_only_admission_used': False,
        'outputs': integrity.files([candidate, directory / 'roundtrip/source-tables.root']),
        'inputs_before': before, 'inputs_after': _unchanged(before),
        'limitations': ['Exact configured consumes is not runtime art consumed-ProductID authentication.',
            'Preserved original SGS pointer bytes do not resolve the original SimParticle ProductID.',
            'Numerical SGS/gen equality does not establish lossless particles or electronics-only response.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ordinal', type=int, required=True)
    parser.add_argument('--dump-only', action='store_true', help='validate only the separate read-only re-export')
    args = parser.parse_args()
    _new(args.output)
    report = (validate_dump if args.dump_only else validate)(args.plan, args.ordinal, args.directory)
    response_control.save(args.output, report)
    print(f"validated response {'dump' if args.dump_only else 'parent'} {args.ordinal}: {report['events']} events")


if __name__ == '__main__':
    main()
