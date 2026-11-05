"""Adaptive accepted-prefix diagnostics on existing maps/exports, no simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beamfastsim import tscarry, tsfield, tsgeom, tsvac


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(root):
    scenes = {}
    for arm in ('run1a', 'run1b'):
        path = root/f'geometry_{arm}_v1/ts_geometry_{arm}.gdml'
        scene = tsgeom.load(path, [f'TS{i}Vacuum' for i in range(1, 6)],
                            vacuum_materials=['PSVacuum', 'DSVacuum'])
        scenes[arm] = scene.partition, digest(path)
    choices = (('PSMap', [3904., 0, -3504.], [0., 0, 1.], 50.),
               ('PSMap', [3904., 0, -3000.], [0., 0, 1.], 150.),
               ('TSuMap_fix', [3904., 0, -2800.], [0., 0, 1.], 50.),
               ('TSuMap_fix', [20., 0, 0], [-1., 0, 0], 32.),
               ('TSdMap', [-3904., 0, 3000.], [0., 0, 1.], 50.),
               ('TSdMap', [-3904., 0, 2925.], [0., 0, 1.], 10.),
               ('TSdMap', [-2., 0, 0], [1., 0, 0], 4.),
               ('TSdMap', [0., 0, 0], [1., 0, 0], 1.))
    contexts, records = {}, []
    for name, point, direction, length in choices:
        path = root/f'assets_v1/maps/{name}.header'
        if name not in contexts:
            contexts[name] = tscarry.MapContext(tsfield.load(path))
        ctx = contexts[name]
        step = 2. if length <= 4. else 5.
        record = dict(map=name, header_sha256=digest(path), binary_sha256=digest(path.with_suffix('.bin')),
                      context_fingerprint=ctx.fingerprint, x_mm=point, direction=direction,
                      momentum_mev=100., pdg=[13, -13, 211, -211], requested_mm=length, max_step_mm=step, geometry={})
        for arm, (geometry, sha) in scenes.items():
            initial = tscarry.State(ctx, np.tile(point, (4, 1)), np.tile(direction, (4, 1)),
                100., [13, -13, 211, -211], position_error_mm=0., direction_error=0.)
            result = tsvac.walk(initial, geometry, length, field_error_tesla=0., position_error_mm=0.,
                direction_error=0., geometry_error_mm=0., max_position_error_mm=2., max_direction_error=.125,
                max_step=step, min_step=2**-12, max_trials=256, max_steps=128)
            if not np.array_equal(result.state.momentum, initial.momentum):
                raise ValueError('fixed momentum changed')
            if not np.array_equal(result.state.source_index, initial.source_index):
                raise ValueError('row tags changed')
            entry = dict(gdml_sha256=sha, **{key: getattr(result, key).tolist() for key in
                ('status', 'progress_mm', 'remaining_mm', 'trials', 'steps', 'last_rejection_reason', 'last_rejected_mm',
                 'initial_geometry_status', 'initial_owner_index', 'initial_clearance_margin_mm')},
                nominal_x_mm=result.state.x.tolist(), nominal_direction=result.state.direction.tolist(),
                position_error_mm=result.state.position_error_mm.tolist(), direction_error=result.state.direction_error.tolist(),
                fixed_momentum=True, row_tags_preserved=True,
                ledger={key: getattr(result.ledger, key).tolist() for key in result.ledger.__dataclass_fields__})
            record['geometry'][arm] = entry
        records.append(record)
    return dict(schema='ts-adaptive-vacuum-reference-audit-v1', diagnostics=records,
        terminal_status_labels=tsvac.STATUS, rejection_reason_labels=tsvac.REASON,
        max_step_mm=5., min_failed_step_mm=2**-12, max_trials=256, max_steps=128,
        position_budget_mm=2., direction_budget=.125, errors_are_caller_assumptions=True,
        initial_position_error_mm=0., initial_direction_error=0., field_error_tesla=0.,
        evaluation_position_error_mm=0., evaluation_direction_error=0., geometry_error_mm=0.,
        zero_errors_are_numerical_idealization=True, floating_point_certified=False,
        native_query_certified=False, physical_endpoint=False, material_transport=False,
        production_tracks=False, transmission_measured=False, failed_trial_is_not_root_bracket=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('audit output must not exist; preserve earlier receipts')
    receipt = audit(args.root)
    with args.out.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    for item in receipt['diagnostics']:
        for arm, result in item['geometry'].items():
            print(json.dumps(dict(map=item['map'], start=item['x_mm'], arm=arm,
                **{key: result[key] for key in ('status', 'progress_mm', 'trials', 'steps',
                    'last_rejection_reason', 'position_error_mm', 'direction_error')}), allow_nan=False))


if __name__ == '__main__':
    main()
