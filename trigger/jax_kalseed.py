"""Synthetic-tested BO selection prototype for two TPR KalSeedFilters, not reconstruction.

Inputs must be DERIVED MEASURED quantities already reconstructed with the exact
released C++ definitions, including signed d0, 1/tan(theta), momentum magnitude,
t0-segment momentum error and chi2/nDOF. No truth inputs or proxy derivations are
supported. The separate measured feature table is validated by dedicated I/O and
comparison tools; decisions.art alone contains path statuses, not these quantities.
This kernel alone is not a validated production-exact emulator, a full-path decision,
a source-equivalence certification, or a speedup measurement.

Call validate_features outside jit, then evaluate(features, contract). For jit use
jax.jit(lambda features: evaluate(features, contract)); the JSON-safe compiled
contract is a static closure, not a traced argument. The caller must enable x64;
this module never changes JAX's global configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

if __package__:
    from . import replay
else:
    import replay

SOURCE_URL = 'https://raw.githubusercontent.com/Mu2e/Offline/b132ab6567ac021851353e05b7a1feb45bfb7c14/TrkFilters/src/KalSeedFilter_module.cc'
SOURCE_SHA256 = '51b6dbfe6f5dcd115afdfb8b9ada7aa187b8610f717cdbbc1527e7b573fabd39'
PATH_FILTERS = {
    'tpr_TrkDe_80m70p_D0200': 'TprTrkDe80m70pD0200KSFilter',
    'tpr_TrkDe_80m70p': 'TprTrkDe80m70pKSFilter',
}
BOOL_FEATURES = ('valid_track', 'status_has_SeedOK', 'has_t0_segment', 'supported_fit')
FLOAT_FEATURES = ('pz', 'momentum', 'tan_dip', 'd0', 'momentum_error', 'chi2_dof', 'fit_consistency')
INT_FEATURES = ('fit_particle', 'active_hits')
FEATURES = BOOL_FEATURES + FLOAT_FEATURES + INT_FEATURES
NUMERIC_CUTS = ('minFitCons', 'minNStrawHits', 'minMomentum', 'maxMomentum',
                'minTanDip', 'maxTanDip', 'maxChi2DOF', 'maxMomErr', 'minD0', 'maxD0')


def _require_x64():
    if not jax.config.x64_enabled:
        raise ValueError('jax_enable_x64 must already be true (caller responsibility)')


def _table(value, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be a literal table')
    return value


def _fields(table, allowed, where):
    unknown = set(table) - set(allowed)
    if unknown:
        raise ValueError(f'unsupported {where} fields: {sorted(unknown)}')


def _number(value, name):
    if not isinstance(value, replay.Atom):
        raise ValueError(f'{name} must be a literal number')
    try:
        result = float(value.text)
    except ValueError as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _integer(value, name, unsigned=False):
    result = _number(value, name)
    low, high = (0, 2**32 - 1) if unsigned else (-2**31, 2**31 - 1)
    if result != math.trunc(result) or not low <= result <= high:
        raise ValueError(f'{name} must be an in-range integer')
    return int(result)


def _boolean(value, name):
    if not isinstance(value, replay.Atom) or value.text not in ('true', 'false'):
        raise ValueError(f'{name} must be a literal boolean')
    return value.text == 'true'


def compile_selection(config_text: str, path_name: str) -> dict:
    """Compile only the final selection from embedded Digitize provenance.

    Thresholds come from the supplied config, never current-head defaults. Supported
    variations are numeric thresholds, minNTrks and explicit boolean noFilter.
    Other path modules (including upstream filters/prescales) are NOT emulated.
    Repeated supported cut rows are ORed; each track is counted only once.
    """
    _require_x64()
    if path_name not in PATH_FILTERS:
        raise ValueError(f'unsupported path: {path_name}')
    config = replay.production_config(config_text, 'Digitize')
    physics = _table(config.get('physics'), 'physics')
    scheduled = {name: sequence for _, name, sequence in replay.paths(config)}
    label = PATH_FILTERS[path_name]
    sequence = scheduled.get(path_name, [])
    if sequence.count(label) != 1 or sequence.count('TrigKSFDe') != 1:
        raise ValueError('supported path must schedule its exact filter and TrigKSFDe once')
    if sequence.index('TrigKSFDe') >= sequence.index(label):
        raise ValueError('TrigKSFDe must precede the selection filter')
    filters = _table(physics.get('filters'), 'filters')
    module = _table(filters.get(label), label)
    _fields(module, ('module_type', 'module_label', 'debugLevel', 'KalSeedCuts',
                     'kalSeedCollections', 'minNTrks', 'noFilter', 'noInfo', 'momSurface'), label)
    if replay.plain(module.get('module_type')) != 'KalSeedFilter':
        raise ValueError('expected module_type KalSeedFilter')
    if 'module_label' in module and replay.plain(module['module_label']) != label:
        raise ValueError('module_label does not match supported filter')
    if module.get('kalSeedCollections') != ['TrigKSFDe']:
        raise ValueError('only the exact single collection TrigKSFDe is supported')
    if 'momSurface' in module:
        raise ValueError('momSurface is unsupported; features must use the t0 segment')
    if _boolean(module.get('noInfo', replay.Atom('false')), 'noInfo'):
        raise ValueError('noInfo changes are unsupported')
    if 'debugLevel' in module:
        _integer(module['debugLevel'], 'debugLevel')
    min_tracks = _integer(module.get('minNTrks', replay.Atom('1')), 'minNTrks', unsigned=True)
    no_filter = _boolean(module.get('noFilter', replay.Atom('false')), 'noFilter')
    rows = module.get('KalSeedCuts')
    if not isinstance(rows, list) or not rows:
        raise ValueError('KalSeedCuts must be a nonempty literal sequence')
    cuts = []
    for row in rows:
        row = _table(row, 'KalSeedCuts row')
        _fields(row, NUMERIC_CUTS + ('fitparticle', 'fitdirection', 'requireCaloCluster',
                                    'minNStereo', 'minNPlanes', 'seedFitFlag'), 'cut')
        required = set(NUMERIC_CUTS) | {'fitparticle', 'fitdirection', 'requireCaloCluster'}
        if required - row.keys():
            raise ValueError(f'missing required cut fields: {sorted(required - row.keys())}')
        particle = _integer(row['fitparticle'], 'fitparticle')
        if particle not in (11, -11) or replay.plain(row['fitdirection']) != 'downstream':
            raise ValueError('only required fitparticle 11/-11 and downstream are supported')
        if _boolean(row['requireCaloCluster'], 'requireCaloCluster'):
            raise ValueError('calorimeter requirements are unsupported')
        for name in ('minNStereo', 'minNPlanes'):
            if _integer(row.get(name, replay.Atom('0')), name, unsigned=True) != 0:
                raise ValueError(f'nonzero {name} is unsupported')
        if row.get('seedFitFlag', ['SeedOK']) != ['SeedOK']:
            raise ValueError('only seedFitFlag ["SeedOK"] is supported')
        cut = {name: _number(row[name], name) for name in NUMERIC_CUTS}
        cut['minNStrawHits'] = _integer(row['minNStrawHits'], 'minNStrawHits', unsigned=True)
        cut.update(fitparticle=particle, fitdirection='downstream', requireCaloCluster=False,
                   minNStereo=0, minNPlanes=0, seedFitFlag=['SeedOK'])
        cuts.append(cut)
    if {cut['fitparticle'] for cut in cuts} != {11, -11}:
        raise ValueError('both electron and positron cut rows are required')
    return {
        'schema': 'jax-kalseed-selection-prototype-v1',
        'status': 'prototype; synthetic tests only; no measured features exported',
        'scope': 'final KalSeedFilter selection only, not full trigger path or reconstruction',
        'config_sha256': hashlib.sha256(config_text.encode('utf-8')).hexdigest(),
        'process': 'Digitize', 'path': path_name, 'filter': label,
        'collections': ['TrigKSFDe'],
        'source': {'url': SOURCE_URL, 'sha256': SOURCE_SHA256, 'release': 'v13_41_01',
                   'binding': 'read-only released source; not a source-equivalence certification'},
        'features': {
            'shape': ['events', 'padded_tracks'],
            'fields': {**{name: 'bool' for name in BOOL_FEATURES},
                       **{name: 'float64' for name in FLOAT_FEATURES},
                       'fit_particle': 'signed integer PDG', 'active_hits': 'nonnegative integer'},
            'origin': 'derived measured quantities with exact released C++ definitions; no truth',
            'supported_fit': 'loopHelixFit, centralHelixFit or kinematicLineFit',
            'padding': 'valid_track masks padding; empty track dimension supported',
            'nonfinite': 'C++ comparisons; no blanket finite veto; NOT(pz < 0) for downstream',
            'validation': 'call validate_features outside jit before evaluate',
        },
        'cuts': cuts, 'minNTrks': min_tracks, 'noFilter': no_filter,
        'noFilter_explicit': 'noFilter' in module,
        'momSurface': None, 'noInfo': False,
    }


def _layout(features):
    if not isinstance(features, Mapping) or set(features) != set(FEATURES):
        raise ValueError(f'features must contain exactly {FEATURES}; unknown and missing fields refused')
    shape = None
    for name in FEATURES:
        value = features[name]
        if not hasattr(value, 'shape') or not hasattr(value, 'dtype') or len(value.shape) != 2:
            raise ValueError(f'{name} must be an (events, padded_tracks) array')
        if shape is None:
            shape = value.shape
        if value.shape != shape:
            raise ValueError(f'{name} shape differs from the other feature arrays')
        dtype = np.dtype(value.dtype)
        if name in BOOL_FEATURES and dtype != np.dtype('bool'):
            raise ValueError(f'{name} must have bool dtype')
        if name in FLOAT_FEATURES and dtype != np.dtype('float64'):
            raise ValueError(f'{name} must have float64 dtype')
        if name == 'fit_particle' and dtype.kind != 'i':
            raise ValueError('fit_particle must have signed integer dtype')
        if name == 'active_hits' and dtype.kind not in 'iu':
            raise ValueError('active_hits must have integer dtype')
    return shape


def validate_features(features) -> None:
    """Host-side input gate; call outside jit. Nonfinite floats are intentionally legal."""
    _require_x64()
    _layout(features)
    if np.any(np.asarray(features['active_hits']) < 0):
        raise ValueError('active_hits must be nonnegative, including padding')


def _less(left, right):
    """IEEE float64 less-than via bits, retaining subnormals on FTZ backends."""
    a = jax.lax.bitcast_convert_type(jnp.asarray(left, dtype=jnp.float64), jnp.uint64)
    b = jax.lax.bitcast_convert_type(jnp.asarray(right, dtype=jnp.float64), jnp.uint64)
    sign = jnp.uint64(1 << 63)
    magnitude = sign - jnp.uint64(1)
    infinity = jnp.uint64(0x7ff0000000000000)
    a_abs, b_abs = a & magnitude, b & magnitude
    a_key = jnp.where((a & sign) != 0, ~a, a ^ sign)
    b_key = jnp.where((b & sign) != 0, ~b, b ^ sign)
    return ((a_key < b_key) & (a_abs <= infinity) & (b_abs <= infinity) &
            ~((a_abs == 0) & (b_abs == 0)))


def evaluate(features, contract):
    """Return (track_pass[E,T], event_pass[E]) JAX booleans; no per-event Python loops.

    The contract must come from compile_selection and remain unchanged. Layout
    checks are trace-safe, but data-dependent count validation belongs outside jit.
    """
    _require_x64()
    shape = _layout(features)
    f = {name: jnp.asarray(features[name]) for name in FEATURES}
    base = (f['valid_track'] & f['status_has_SeedOK'] & f['has_t0_segment'] &
            f['supported_fit'] & ~_less(f['pz'], 0.0))
    track_pass = jnp.zeros(shape, dtype=jnp.bool_)
    for cut in contract['cuts']:
        passed = (base & (f['fit_particle'] == cut['fitparticle']) &
                  (f['active_hits'] >= cut['minNStrawHits']) &
                  _less(cut['minMomentum'], f['momentum']) & _less(f['momentum'], cut['maxMomentum']) &
                  _less(f['momentum_error'], cut['maxMomErr']) & _less(f['chi2_dof'], cut['maxChi2DOF']) &
                  _less(cut['minFitCons'], f['fit_consistency']) &
                  _less(cut['minTanDip'], f['tan_dip']) & _less(f['tan_dip'], cut['maxTanDip']) &
                  _less(cut['minD0'], f['d0']) & _less(f['d0'], cut['maxD0']))
        track_pass = track_pass | passed
    event_pass = jnp.sum(track_pass, axis=1, dtype=jnp.int64) >= contract['minNTrks']
    if contract['noFilter']:
        event_pass = jnp.ones(shape[0], dtype=jnp.bool_)
    return track_pass, event_pass


def main():
    parser = argparse.ArgumentParser(description='Compile a BO selection prototype contract; no measured validation')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--path', choices=tuple(PATH_FILTERS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse selection contract overwrite')
    contract = compile_selection(args.config.read_text(), args.path)
    with args.output.open('x') as output:
        output.write(json.dumps(contract, indent=2, allow_nan=False) + '\n')
    print(f"compiled {contract['path']}: prototype only, no measured hybrid validation")


if __name__ == '__main__':
    main()
