"""Bounded carried-reference diagnostics on existing maps/GDMLs, no simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beamfastsim import tscarry, tsfield, tsgeom


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
    records = []
    points = {'PSMap': [3904., 0, -3504.], 'TSuMap_fix': [3904., 0, -2800.],
              'TSdMap': [-3904., 0, 3000.]}
    for name, point in points.items():
        path = root/f'assets_v1/maps/{name}.header'
        ctx = tscarry.MapContext(tsfield.load(path))
        record = dict(map=name, header_sha256=digest(path), binary_sha256=digest(path.with_suffix('.bin')),
                      context_fingerprint=ctx.fingerprint, x_mm=point, direction=[0, 0, 1],
                      momentum_mev=100., pdg=[13, -13, 211, -211], geometry={})
        for arm, (geometry, sha) in scenes.items():
            initial = tscarry.State(ctx, np.tile(point, (4, 1)), np.tile([0., 0, 1.], (4, 1)),
                100., [13, -13, 211, -211], position_error_mm=0., direction_error=0.)
            state, history = initial, []
            for index in range(10):
                segment = tscarry.Segment(state, 5., field_error_tesla=0.,
                                          position_error_mm=0., direction_error=0.)
                result = tscarry.attempt(segment, geometry, geometry_error_mm=0.)
                state = result.state
                if not np.array_equal(state.momentum, initial.momentum):
                    raise ValueError('fixed scalar momentum changed')
                if not np.array_equal(state.source_index, initial.source_index):
                    raise ValueError('source row tags changed')
                history.append(dict(step=index+1, status=result.status.tolist(),
                    owner_index=result.owner_index.tolist(), clearance_margin_mm=result.clearance_margin_mm.tolist(),
                    path_mm=state.path_mm.tolist(), nominal_x_mm=state.x.tolist(),
                    nominal_direction=state.direction.tolist(), position_error_mm=state.position_error_mm.tolist(),
                    direction_error=state.direction_error.tolist(), momentum_unchanged=True))
                if not np.all(result.accepted):
                    break
            record['geometry'][arm] = dict(gdml_sha256=sha, attempts=history)
        records.append(record)
    return dict(schema='ts-carried-reference-audit-v1', maps=records, status_labels=tscarry.STATUS,
        step_mm=5., max_attempts=10, production_tracks=False, transmission_measured=False,
        physical_endpoint=False, segmented_nominal_reference=True, floating_point_certified=False,
        native_query_certified=False, rk4_error_certified=False, material_transport=False,
        errors_are_caller_assumptions=True, initial_position_error_mm=0., initial_direction_error=0.,
        field_error_tesla=0., evaluation_position_error_mm=0., evaluation_direction_error=0.,
        geometry_error_mm=0., zero_errors_are_numerical_idealization=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('audit output must not exist; preserve earlier receipts')
    text = json.dumps(audit(args.root), indent=2, allow_nan=False)
    with args.out.open('x') as stream:
        stream.write(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
