"""Versioned extracted-response configuration; parsing/admission requires only the standard library."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name('configs') / 'response_v1.json'


@dataclass(frozen=True)
class Field:
    unit: str
    role: str
    minimum: float | None = None
    maximum: float | None = None
    positive: bool = False
    kind: str = 'number'
    fixed: bool = False


def fields(unit, role, names, **kwargs):
    return {k: Field(unit, role, **kwargs) for k in names.split()}


FIELDS = {
    'response': fields('1', 'calibration', 'photon_yield_scale photon_yield_variation_scale', minimum=0),
    'sipm': {
        **fields('V', 'calibration', 'overvoltage p2', minimum=0, positive=True),
        **fields('ns', 'calibration', 'time_constant', minimum=0, positive=True),
        **fields('ns', 'inactive_process', 'trap0_lifetime trap1_lifetime', fixed=True),
        **fields('F', 'calibration', 'capacitance', minimum=0, positive=True),
        **fields('1', 'calibration', 'p1', minimum=0, maximum=1, positive=True),
        **fields('1', 'calibration', 'cross_talk', minimum=0, maximum=1),
        **fields('1/ns', 'unsupported', 'thermal_rate', fixed=True),
        **fields('1', 'unsupported', 'trap0_prob trap1_prob', fixed=True),
        **fields('count', 'asset_dependent_design', 'n_pixels_x n_pixels_y', kind='integer', fixed=True),
        'inactive': Field('pixel index', 'asset_dependent_design', kind='array', fixed=True)},
    'waveform': {
        **fields('ns', 'numerical', 'single_pe_precision single_pe_max_time', fixed=True),
        **fields('1', 'calibration', 'single_pe_stretch', minimum=0, positive=True),
        **fields('C', 'calibration', 'single_pe_reference_charge', minimum=0, positive=True),
        **fields('V', 'calibration', 'min_voltage noise', minimum=0),
        **fields('ADC/V', 'calibration', 'adc_conversion', minimum=0, positive=True),
        **fields('ADC', 'conditions', 'pedestal', minimum=0, maximum=4095),
        **fields('ADC', 'asset_dependent_design', 'min_adc max_adc', kind='integer', fixed=True),
        **fields('count', 'asset_dependent_design', 'samples_zs', kind='integer', fixed=True),
        'single_pe_file': Field('path', 'asset', kind='path')},
    'pulse_reco': {
        **fields('ADC', 'reconstruction', 'min_adc_difference', minimum=0, positive=True),
        **fields('ADC', 'reconstruction', 'pedestal_undershoot_threshold', minimum=0),
        **fields('ns', 'reconstruction', 'default_beta min_beta max_beta max_time_difference', minimum=0, positive=True),
        **fields('1', 'reconstruction', 'min_pulse_height_ratio max_pulse_height_ratio', minimum=0, positive=True),
        **fields('1', 'unused_output', 'le_time_factor', fixed=True),
        **fields('count', 'reconstruction', 'samples_before samples_after', kind='integer', minimum=0),
        **fields('1', 'stock_compatibility', 'use_time_offset_db ignore_channels', kind='boolean', fixed=True),
        **fields('1', 'stock_compatibility', 'time_offset_scale', fixed=True),
        **fields('count', 'numerical', 'max_pulses lm_iterations', kind='integer', fixed=True),
        **fields('1', 'numerical', 'limit_tolerance', fixed=True),
        **fields('ADC', 'numerical', 'knife_edge_adc', fixed=True)},
    'finder': {
        **fields('mm/ns', 'reconstruction', 'fiber_signal_speed', minimum=0, positive=True),
        **fields('ns', 'reconstruction', 'max_time_interval', minimum=0, positive=True),
        **fields('ns', 'reconstruction', 'time_offset'),
        'compensate_status_bits': Field('bit index', 'stock_compatibility', kind='array', fixed=True),
        'ignore_secondary_pulses': Field('1', 'stock_compatibility', kind='boolean', fixed=True),
        'big_cluster_threshold': Field('count', 'numerical', kind='integer', fixed=True)},
    'timing': {
        'window': Field('ns', 'scenario', kind='array', fixed=True),
        'feb_phase_width': Field('ns', 'calibration', minimum=0, maximum=12.5),
        'phase_scope': Field('1', 'scenario', kind='enum'),
        'timeoffset_mode': Field('1', 'conditions', kind='enum')},
}
SECTOR_FIELDS = {'table': Field('LUT id', 'asset', kind='string', fixed=True),
                 'reflector': Field('1', 'asset_dependent_design', kind='integer', fixed=True),
                 'yield_per_mev': Field('photons/MeV', 'calibration', minimum=0)}
CUT_FIELDS = {
    **fields('PE', 'reconstruction', 'pe_threshold min_cluster_pes', minimum=0),
    **fields('ns', 'reconstruction', 'max_dt_adjacent max_dt', minimum=0, positive=True),
    **fields('1', 'reconstruction', 'min_slope max_slope'),
    **fields('1', 'reconstruction', 'max_slope_diff', minimum=0),
    **fields('count', 'reconstruction', 'coincidence_layers', kind='integer', minimum=2, maximum=4),
    **fields('mm', 'reconstruction', 'initial_max_distance final_max_distance', minimum=0),
}
PROP_KEYS = (('EX', 'T1', 'T2', 'TAG') + tuple(f'EX{i}' for i in range(4)) + ('T10', 'T11', 'T20', 'T21')
             + tuple(f'M{i}' for i in range(1, 9)))
ASSETS = ('calib', 'deviation', 'background', 'time_shift', 'counters', 'photon_map', 'lookup_dir')


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f'duplicate configuration key: {k}')
        out[k] = v
    return out


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'nonfinite JSON value: {x}')))


def defaults():
    return loads(DEFAULT_PATH.read_text())


def _field(name, value, spec, nominal):
    if spec.kind in ('number', 'integer'):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f'{name}: expected finite {spec.kind} in {spec.unit}')
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or (spec.kind == 'integer' and not isinstance(value, int)):
            raise ValueError(f'{name}: expected finite {spec.kind}')
        if spec.minimum is not None and (value < spec.minimum or (spec.positive and value == spec.minimum)):
            raise ValueError(f'{name}: value below allowed domain ({spec.unit})')
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f'{name}: value above allowed domain ({spec.unit})')
    elif spec.kind == 'boolean' and not isinstance(value, bool):
        raise ValueError(f'{name}: expected boolean')
    elif spec.kind in ('path', 'string', 'enum') and (not isinstance(value, str) or not value or '\x00' in value):
        raise ValueError(f'{name}: expected nonempty string')
    elif spec.kind == 'array' and not isinstance(value, list):
        raise ValueError(f'{name}: expected array')
    if spec.kind == 'array':
        def same_types(v, n):
            if isinstance(n, list):
                return isinstance(v, list) and len(v) == len(n) and all(same_types(x, y) for x, y in zip(v, n))
            if isinstance(n, int):
                return type(v) is int
            return type(v) in (int, float) and math.isfinite(v)
        if not same_types(value, nominal):
            raise ValueError(f'{name}: array element types/shape must match the supported configuration')
    if spec.fixed and value != nominal:
        raise ValueError(f'{name}: unsupported change to {spec.role}; required {nominal!r}')


def _block(name, values, specs, nominal):
    if not isinstance(values, dict) or set(values) != set(specs):
        raise ValueError(f'{name}: fields must be {sorted(specs)}')
    for key, spec in specs.items():
        _field(f'{name}.{key}', values[key], spec, nominal[key])


def validate(config, *, kit=False):
    """Structural/backend support, not a commissioned physics-validity envelope or BO prior."""
    d = defaults()
    if not isinstance(config, dict) or set(config) != set(d):
        raise ValueError('configuration has unknown or missing top-level fields')
    if type(config['schema']) is not int or config['schema'] != 1:
        raise ValueError('response configuration requires schema=1')
    if config['scenario'] != 'extracted_v04':
        raise ValueError('response configuration supports only extracted_v04')
    for block, specs in FIELDS.items():
        nested = {'response': ('sectors', 'speed_ratio', 'atten_ratio'), 'finder': ('cuts',)}.get(block, ())
        v = config[block]
        if not isinstance(v, dict) or set(v) != set(specs) | set(nested):
            raise ValueError(f'{block}: unknown or missing fields')
        _block(block, {k: v[k] for k in specs}, specs, d[block])
    for block, key, specs in [('response', 'sectors', SECTOR_FIELDS), ('finder', 'cuts', CUT_FIELDS)]:
        v, nom = config[block][key], d[block][key]
        if not isinstance(v, dict) or set(v) != set(nom):
            raise ValueError(f'{block}.{key}: require all extracted sectors, including M1-M8')
        for sector in v:
            _block(f'{block}.{key}.{sector}', v[sector], specs, nom[sector])
    for kind in ('speed_ratio', 'atten_ratio'):
        v = config['response'][kind]
        if not isinstance(v, dict) or set(v) - set(PROP_KEYS):
            raise ValueError(f'response.{kind}: unknown propagation keys')
        for k, val in v.items():
            _field(f'{kind}.{k}', val, Field('1', 'calibration', minimum=0, positive=True), 1)
    if config['timing']['phase_scope'] not in ('per_feb', 'per_event'):
        raise ValueError('timing.phase_scope must be per_feb or per_event')
    if config['timing']['timeoffset_mode'] not in ('database', 'zero'):
        raise ValueError('timing.timeoffset_mode must be database or zero')
    r = config['pulse_reco']
    if not r['min_beta'] < r['default_beta'] < r['max_beta'] or \
            not r['min_pulse_height_ratio'] < r['max_pulse_height_ratio']:
        raise ValueError('pulse_reco: inconsistent fit bounds')
    for name, c in config['finder']['cuts'].items():
        if c['min_slope'] >= c['max_slope'] or c['max_dt_adjacent'] > c['max_dt']:
            raise ValueError(f'finder.cuts.{name}: inconsistent slope/time bounds')
    s = config['sipm']
    fire = s['p1'] * (1 - math.exp(-s['overvoltage'] / s['p2']))
    qct = (1 - (1 - s['cross_talk']) ** 0.25) / fire if fire else math.inf
    if not math.isfinite(fire) or fire < 1e-12 or fire > 1 or not 0 <= qct <= 1:
        raise ValueError('sipm: firing/cross-talk probabilities incompatible with overvoltage/PDE law')
    _field('background_scale', config['background_scale'], Field('1', 'discrepancy', minimum=0), 1)
    _field('noise', config['noise'], Field('1', 'scenario', kind='boolean'), True)
    if not isinstance(config['assets'], dict) or set(config['assets']) != set(ASSETS):
        raise ValueError('assets: unknown or missing asset names')
    if config['assets']['counters'] is not None:
        raise ValueError('assets.counters: schema-v1 objective supports only stock extracted_v04 geometry')
    for name, path in config['assets'].items():
        if path is not None:
            _field(f'assets.{name}', path, Field('path', 'asset', kind='path'), None)
        if kit and path != d['assets'][name]:
            raise ValueError(f'assets.{name}: kit configuration cannot change staged assets')
    if kit and config['waveform']['single_pe_file'] != d['waveform']['single_pe_file']:
        raise ValueError('kit configuration cannot change the staged waveform asset')
    return config


def _merge(base, patch, trail=''):
    if not isinstance(patch, dict):
        raise ValueError(f'{trail or "configuration"}: expected object')
    for key, value in patch.items():
        if key not in base:
            if trail in ('response.speed_ratio', 'response.atten_ratio') and key in PROP_KEYS:
                base[key] = value
                continue
            raise ValueError(f'unknown configuration field: {trail}.{key}')
        if isinstance(base[key], dict):
            _merge(base[key], value, f'{trail}.{key}'.strip('.'))
        else:
            base[key] = value
    return base


def parse(source=None, *, kit=False):
    """Load a JSON path/object or inline JSON; omitted fields inherit the shipped schema-v1 preset."""
    if source is None or source == '':
        patch = {}
    elif isinstance(source, dict):
        patch = deepcopy(source)
    else:
        text = str(source)
        patch = loads(text if text.lstrip().startswith('{') else Path(text).read_text())
    if not isinstance(patch, dict):
        raise ValueError('response configuration must be a JSON object')
    if patch and (type(patch.get('schema')) is not int or patch.get('schema') != 1):
        raise ValueError('nonempty configuration must declare schema=1')
    return validate(_merge(defaults(), patch), kit=kit)


def apply_knobs(config, knobs):
    """Existing response-knob namespace only; override file values and record resolved numbers."""
    c = deepcopy(config)
    for k, v in (knobs or {}).items():
        if v == '' or v is None:
            continue
        if isinstance(v, bool):
            raise ValueError(f'knob {k}: expected a number')
        try:
            v = float(v)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f'knob {k}: expected a finite number') from None
        if k.startswith(('speed_', 'atten_')):
            kind, key = k.split('_', 1)
            if key not in PROP_KEYS:
                raise ValueError(f'unknown knob {k}')
            c['response'][kind + '_ratio'][key] = v
        elif k.startswith('yield_') and k[6:] in ('EX', 'T1', 'T2', 'TAG'):
            for name in ([f'M{i}' for i in range(1, 9)] if k[6:] == 'TAG' else [k[6:]]):
                c['response']['sectors'][name]['yield_per_mev'] = v
        elif k in ('time_constant', 'cross_talk'):
            c['sipm'][k] = v
        elif k in ('min_voltage', 'wf_noise', 'pe_stretch'):
            c['waveform'][{'wf_noise': 'noise', 'pe_stretch': 'single_pe_stretch'}.get(k, k)] = v
        elif k == 'photon_yield_scale':
            c['response'][k] = v
        elif k == 'background_scale':
            c[k] = v
        else:
            raise ValueError(f'unknown knob {k}')
    return validate(c)


def canonical(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)


def record(config):
    """Resolved settings identity; asset-byte and execution identities are recorded separately by R3."""
    text = canonical(config)
    return {'schema': 1, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'values': config,
            'backend_support': ['numpy', 'jax'], 'validation': 'implementation support; not physics commissioning'}


def validate_loaded_assets(config, tables, photon_map):
    """Keep configurable optical assets inside the extracted sensor/table representation."""
    import numpy as np
    p = np.asarray(photon_map)
    if p.shape != (40, 40) or not np.isfinite(p).all() or np.any(p < 0) or not np.isfinite(p.sum()) or p.sum() <= 0:
        raise ValueError('photon_map: require finite nonnegative 40x40 weights with positive sum')
    for name, half_length in (('6000_0', 3000), ('2370_1', 1185), ('3200_0', 1600)):
        table = tables[name]
        if not np.isclose(table.half_length, half_length) or table.version[0] != 6:
            raise ValueError(f'lookup table {name}: unsupported geometry or version')


def asset_manifest(config, counters=None):
    """Hash the exact local condition/geometry/optical/template assets used by an R3 execution."""
    from . import geometry
    from .response.lookup import lookup_dir
    a = config['assets']
    root = Path(a['lookup_dir'] or lookup_dir())
    paths = {k: v for k, v in a.items() if v and k != 'lookup_dir'}
    paths['counters'] = counters or geometry.find_counters(a['counters'] or
                                                          'Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt')
    paths['waveform'] = geometry.find_counters(config['waveform']['single_pe_file'])
    paths['photon_map'] = a['photon_map'] or str(root / 'photonMap.root')
    paths.update({f'table_{s}': str(root / f'LookupTable_{s}') for s in
                  sorted({v['table'] for v in config['response']['sectors'].values()})})
    out = {}
    for key, path in paths.items():
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        out[key] = {'path': str(Path(path).resolve()), 'sha256': h.hexdigest()}
    return out


def build(config, *, jax=False, background=None):
    """Construct the existing dataclasses; no second physics implementation."""
    from .chain import ChainConfig
    from .response.photons import ResponseConfig, SectorConfig
    from .response.sipm import SiPMConfig
    from .reco.waveform import WaveformConfig
    from .reco.pulsefit import RecoConfig
    from .reco.coinc import FinderConfig, SectorCuts
    validate(config)
    r, s, f = deepcopy(config['response']), deepcopy(config['sipm']), deepcopy(config['finder'])
    r['sectors'] = {k: SectorConfig(**v) for k, v in r['sectors'].items()}
    s['inactive'] = tuple(tuple(x) for x in s['inactive'])
    f['cuts'] = {k: SectorCuts(**v) for k, v in f['cuts'].items()}
    f['compensate_status_bits'] = tuple(f['compensate_status_bits'])
    return ChainConfig(response=ResponseConfig(**r), sipm=SiPMConfig(**s), finder=FinderConfig(**f),
                       waveform=WaveformConfig(**config['waveform']), pulse_reco=RecoConfig(**config['pulse_reco']),
                       feb_phase_width=config['timing']['feb_phase_width'], window=tuple(config['timing']['window']),
                       noise=config['noise'], jax=jax, background=background, background_scale=config['background_scale'])


if __name__ == '__main__':
    import argparse
    from dataclasses import asdict
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', help='JSON path; omit to print the full nominal configuration')
    ap.add_argument('--describe', action='store_true', help='print units, roles and supported domains')
    args = ap.parse_args()
    print(json.dumps({'fields': {k: {n: asdict(v) for n, v in s.items()} for k, s in FIELDS.items()},
                      'sector_fields': {k: asdict(v) for k, v in SECTOR_FIELDS.items()},
                      'cut_fields': {k: asdict(v) for k, v in CUT_FIELDS.items()}}
                     if args.describe else parse(args.config), indent=2))
