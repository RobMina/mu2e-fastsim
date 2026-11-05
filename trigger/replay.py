"""Inventory and prepare replay from an expanded, embedded production configuration."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Atom:
    text: str


TOKEN = re.compile(r'\s+|#[^\n]*|"(?:[^"\\]|\\.)*"|[{}\[\]:,]|[^\s{}\[\]:,"#]+')


def parse_expanded(text: str) -> dict:
    """Read the literal subset emitted by config_dumper, not general FHiCL."""
    tokens = []
    end = 0
    for match in TOKEN.finditer(text):
        if match.start() != end:
            raise ValueError(f"unsupported syntax at character {end}")
        end = match.end()
        token = match.group()
        if not token.isspace() and not token.startswith('#'):
            tokens.append(token)
    if end != len(text):
        raise ValueError(f"unsupported syntax at character {end}")
    pos = 0

    def take():
        nonlocal pos
        if pos == len(tokens):
            raise ValueError("truncated expanded configuration")
        token = tokens[pos]
        pos += 1
        return token

    def value():
        token = take()
        if token == '{':
            return table('}')
        if token == '[':
            result = []
            while pos < len(tokens) and tokens[pos] != ']':
                result.append(value())
                if pos < len(tokens) and tokens[pos] == ',':
                    take()
            if take() != ']':
                raise ValueError("unterminated sequence")
            return result
        if token.startswith('"'):
            return json.loads(token)
        if token in ('}', ']', ':', ',') or token.startswith('@'):
            raise ValueError(f"not literal expanded FHiCL: {token}")
        return Atom(token)

    def table(close=None):
        result = {}
        while pos < len(tokens) and (close is None or tokens[pos] != close):
            key = take()
            if key.startswith('"'):
                key = json.loads(key)
            if key in ('{', '[', '}', ']', ':', ',') or take() != ':':
                raise ValueError(f"invalid table key {key}")
            if key in result:
                raise ValueError(f"duplicate key {key}")
            result[key] = value()
        if close and take() != close:
            raise ValueError("unterminated table")
        return result

    return table()


def production_config(text: str, process: str) -> dict:
    match = re.search(rf'(?m)^{re.escape(process)}\s*:\s*\{{', text)
    if not match:
        raise ValueError(f"embedded process {process!r} not found")
    return parse_expanded(text[match.start():])[process]


def render(value, depth=0):
    if isinstance(value, Atom):
        return value.text
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return '[ ' + ', '.join(render(x, depth) for x in value) + ' ]'
    if isinstance(value, dict):
        lines = [f"{'  ' * (depth + 1)}{k}: {render(v, depth + 1)}" for k, v in value.items()]
        return '{\n' + '\n'.join(lines) + '\n' + '  ' * depth + '}'
    raise TypeError(type(value).__name__)


def plain(value):
    if isinstance(value, Atom):
        return value.text
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def trigger_bit(text):
    if not isinstance(text, str) or not re.fullmatch(r'0|[1-9][0-9]*', text):
        raise ValueError(f"trigger bit must be canonical nonnegative decimal: {text!r}")
    return int(text)


def paths(config):
    physics = config['physics']
    result = []
    seen_names, seen_bits = set(), set()
    for entry in plain(physics['trigger_paths']):
        bit, separator, name = entry.partition(':')
        if not separator:
            name, bit = bit, None
        elif trigger_bit(bit) > 2147483647:
            raise ValueError(f"trigger bit exceeds std::stoi navigator range: {bit}")
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name):
            raise ValueError(f"unsupported trigger path name: {name}")
        if name in seen_names or (bit is not None and bit in seen_bits):
            raise ValueError("duplicate trigger path name or bit")
        seen_names.add(name)
        if bit is not None:
            seen_bits.add(bit)
        sequence = plain(physics[name])
        if not isinstance(sequence, list) or not all(isinstance(x, str) for x in sequence):
            raise ValueError(f"non-literal path {name}")
        result.append((bit, name, sequence))
    return result


def inventory(config):
    physics = config['physics']
    modules = {**physics.get('producers', {}), **physics.get('filters', {})}
    result = []
    for bit, name, sequence in paths(config):
        if name == 'DigitizePath' or name.startswith('Triggerable'):
            continue
        prescales = []
        for label in sequence:
            module = modules.get(label)
            if module is None:
                raise ValueError(f"undefined scheduled module {label}")
            if plain(module['module_type']) == 'PrescaleEvent':
                prescales.append({'label': label, 'eventModeConfig': plain(module['eventModeConfig'])})
        result.append({'bit': bit, 'path': name, 'prescales': prescales,
                       'sequence': sequence,
                       'modules': {label: plain(modules[label]) for label in sequence}})
    return {'paths': result, 'digitize_modules': plain(physics['DigitizePath']),
            'output_selection': plain(config['outputs']['Output']['SelectEvents']),
            'geometry': plain(config['services']['GeometryService']),
            'conditions': plain(config['services']['DbService'])}


def validate_inputs(producers, filters, removed_mc, original_process):
    """Reject known unsafe bindings; this is not a complete art consumes graph."""
    bindings = []
    fields = {
        'StrawHitReco': {'StrawDigiCollectionTag': 'StrawDigis',
                        'EventWindowMarker': 'EventWindowMarker',
                        'ProtonBunchTimeTag': 'ProtonBunchTime'},
        'CaloHitMakerFast': {'caloDigiCollection': 'CaloDigis',
                            'ProtonBunchTimeTag': 'ProtonBunchTime'},
        'PrescaleEvent': {'EventWindowMarker': 'EventWindowMarker'},
    }
    modules = {**producers, **filters}
    for label, module in modules.items():
        kind = plain(module['module_type'])
        selected_fields = dict(fields.get(kind, {}))
        if kind == 'DigiFilter':
            if plain(module.get('useStrawDigi')) == 'true':
                selected_fields['strawDigiCollection'] = 'StrawDigis'
            if plain(module.get('useCaloDigi')) == 'true':
                selected_fields['caloDigiCollection'] = 'CaloDigis'
        for field, product in selected_fields.items():
            tag = plain(module.get(field, 'EWMProducer' if kind == 'PrescaleEvent' else ''))
            if not isinstance(tag, str) or not tag:
                raise ValueError(f"{label}.{field} needs an explicit digi/timing tag")
            bindings.append({'consumer': label, 'field': field, 'product': product, 'tag': tag})

        def inspect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key != 'module_label':
                        inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)
            elif isinstance(value, str):
                source, *tail = value.split(':')
                if source in removed_mc:
                    raise ValueError(f"{label} references removed MC producer {source}")
                if source in modules and len(tail) == 2 and tail[-1] == original_process:
                    raise ValueError(f"{label} binds old reconstruction process: {value}")

        inspect(plain(module))
    return bindings


def stateful_modules(modules):
    reasons = {}
    for label, module in modules.items():
        kind = plain(module['module_type'])
        if kind == 'LumiStreamFilter':
            reasons[label] = 'processed-event buffering counter and first-in-subrun state'
        if kind == 'HelixFilter' and any(plain(module.get(side, {}).get('prescaleUsingD0Phi')) == 'true'
                for side in ('posHelicitySelection', 'negHelicitySelection')):
            reasons[label] = 'helix-phi prescale uses filter invocation count, not art event number'
    return reasons


def prepare(config, original_process='Digitize', replay_process='TriggerReplay', quiet=False,
            straw_tag=None, calo_tag=None):
    if original_process == replay_process:
        raise ValueError("replay needs a distinct process name")
    physics = config['physics']
    digitizers = set(plain(physics['DigitizePath']))
    inv = inventory(config)
    modules = {**physics.get('producers', {}), **physics.get('filters', {})}
    removed_mc = {label for label, m in modules.items() if plain(m['module_type']) == 'SelectRecoMC'}
    selected = []
    sequences = {}
    for bit, name, sequence in paths(config):
        if name == 'DigitizePath' or name.startswith('Triggerable'):
            continue
        if bit is None:
            raise ValueError(f"physics path has no explicit trigger bit: {name}")
        selected.append(f'{bit}:{name}')
        sequences[name] = [label for label in sequence if label not in digitizers | removed_mc]
    active = {label for sequence in sequences.values() for label in sequence}
    producers = deepcopy({k: v for k, v in physics.get('producers', {}).items() if k in active})
    filters = deepcopy({k: v for k, v in physics.get('filters', {}).items() if k in active})
    for label, module in producers.items():
        if plain(module['module_type']) == 'StrawDigisFromArtdaqFragments':
            producers[label] = {'module_type': 'NullProducer'}
    bindings = validate_inputs(producers, filters, removed_mc, original_process)
    substitutions = []
    for product, replacement in [('StrawDigis', straw_tag), ('CaloDigis', calo_tag)]:
        if replacement is None:
            continue
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*:[A-Za-z_0-9]*:[A-Za-z_][A-Za-z_0-9]*', replacement):
            raise ValueError('substitution needs an explicit label:instance:process digi tag')
        if replacement.split(':')[0] in active or replacement.split(':')[2] == replay_process:
            raise ValueError('injected digi producer must not also be a replay module or process')
        for binding in bindings:
            if binding['product'] == product:
                module = producers.get(binding['consumer'], filters.get(binding['consumer']))
                module[binding['field']] = replacement
                substitutions.append({**binding, 'replacement': replacement})
    bindings = validate_inputs(producers, filters, removed_mc, original_process)
    printer = {'module_type': 'PrintModule', 'PrintSubRun': Atom('false'),
               'triggerResultsPrinter': {'verbose': Atom('2'),
                 'inputTags': [f'TriggerResults::{original_process}', f'TriggerResults::{replay_process}']}}
    replay = {
        'process_name': replay_process,
        'source': {'module_type': 'RootInput', 'inputCommands': [
            'keep *', 'drop mu2e::HelixSeed*_*_*_*', 'drop mu2e::KalSeed*_*_*_*',
            'drop mu2e::TriggerInfo*_*_*_*', 'drop mu2e::CaloClusters_*_*_*',
            'drop mu2e::TimeClusters_*_*_*', 'drop mu2e::CosmicTrackSeeds_*_*_*',
            'drop mu2e::IntensityInfo*_*_*_*', 'drop mu2e::RecoCount_*_*_*',
            'drop mu2e::PrescaleFilterFraction_*_*_*']},
        'services': config['services'],
        'physics': {'producers': producers, 'filters': filters,
                    'analyzers': {'decisions': printer}, **sequences,
                    'trigger_paths': selected, 'DecisionOutput': ['decisions', 'DecisionProducts'],
                    'end_paths': ['DecisionOutput']},
        'outputs': {'DecisionProducts': {'module_type': 'RootOutput', 'fileName': 'decisions.art',
            'outputCommands': ['drop *', 'keep art::TriggerResults_*_*_*']}}}
    stateful = stateful_modules(modules)
    manifest = {'original_process': original_process, 'replay_process': replay_process,
                'paths': [p['path'] for p in inv['paths']], 'bits': [p['bit'] for p in inv['paths']],
                'removed_digitizers': sorted(digitizers), 'removed_mc_matching': sorted(removed_mc),
                'external_bindings': bindings, 'digi_substitutions': substitutions,
                'dependency_audit': 'known digi/timing fields and removed/old-process references only',
                'comparison': 'accepted_boolean_only', 'denominator': 'all stored input events',
                'stock_output_selection': inv['output_selection'],
                'stateful_modules': stateful,
                'stateful_paths': [name for _, name, sequence in paths(config)
                    if any(label in stateful for label in sequence)],
                'stateful_warning': 'Selected input may omit events that advanced buffering counters.'}
    manifest['mode'] = 'quiet_timing' if quiet else 'decision_comparison'
    if quiet:
        replay['physics']['analyzers'] = {}
        replay['physics']['DecisionOutput'] = ['DecisionProducts']
        replay['services'] = deepcopy(replay['services'])
        service = replay['services'].setdefault('message', {})
        category = service.setdefault('destinations', {}).setdefault('log', {}).setdefault('categories', {})
        category.setdefault('ArtReport', {})['limit'] = Atom('0')
    return replay, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='config_dumper -P output')
    parser.add_argument('--process', default='Digitize')
    parser.add_argument('--output', type=Path, required=True, help='private output directory')
    parser.add_argument('--quiet', action='store_true', help='no per-event printer; timing mode, not decision comparison')
    parser.add_argument('--straw-digi-tag', help='injected digi label:instance:process in the same input events')
    parser.add_argument('--calo-digi-tag', help='injected digi label:instance:process in the same input events')
    args = parser.parse_args()
    text = args.config.read_text()
    config = production_config(text, args.process)
    replay, manifest = prepare(config, args.process, quiet=args.quiet,
                               straw_tag=args.straw_digi_tag, calo_tag=args.calo_digi_tag)
    manifest['config_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    source_inventory = inventory(config)
    (args.output / 'inventory.json').write_text(json.dumps(source_inventory, indent=2) + '\n')
    replay_inventory = deepcopy(source_inventory)
    replay_inventory['digitize_modules'] = []
    active_modules = {**replay['physics']['producers'], **replay['physics']['filters']}
    for row in replay_inventory['paths']:
        row['sequence'] = replay['physics'][row['path']]
        row['modules'] = {label: plain(active_modules[label]) for label in row['sequence']}
    (args.output / 'replay_inventory.json').write_text(json.dumps(replay_inventory, indent=2) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'replay.fcl').write_text('\n'.join(f'{k}: {render(v)}' for k, v in replay.items()) + '\n')
    print(f"Prepared {len(manifest['paths'])} physics paths; accepted decisions only, not full art states")


if __name__ == '__main__':
    main()
