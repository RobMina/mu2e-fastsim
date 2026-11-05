"""Read-only staged-map bounds and conditional selected-geometry diagnostics."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from beamfastsim import tsbound, tsfield, tsgeom, tsvary


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(root):
    maps = ((root/'assets_v1/maps', 'PSMap'), (root/'assets_v1/maps', 'TSuMap_fix'),
            (root/'assets_v1/maps', 'TSdMap'), (root/'assets_v1/maps', 'PSAreaMap'),
            (root/'assets_v1/maps', 'WorldMap'), (root/'assets_dump_v1/maps', 'PStoDumpAreaMap'),
            (root/'assets_dump_v1/maps', 'ProtonDumpAreaMap'))
    scenes = {}
    for arm in ('run1a', 'run1b'):
        path = root/f'geometry_{arm}_v1/ts_geometry_{arm}.gdml'
        scene = tsgeom.load(path, [f'TS{i}Vacuum' for i in range(1, 6)],
                            vacuum_materials=['PSVacuum', 'DSVacuum'])
        scenes[arm] = (scene.partition, digest(path))
    records = []
    points = {'PSMap': [3904., 0, -3504.], 'TSuMap_fix': [3904., 0, -2800.],
              'TSdMap': [-3904., 0, 3000.]}
    for directory, name in maps:
        path = directory/f'{name}.header'
        item = tsfield.load(path)
        record = dict(map=name, header_sha256=digest(path), binary_sha256=digest(path.with_suffix('.bin')),
                      magnitude=asdict(tsbound.magnitude(item)), spatial=asdict(tsbound.spatial(item)))
        if name in points:
            model = tsvary.MapEnclosure(np.tile(points[name], (4, 1)), np.tile([0., 0, 100.], (4, 1)),
                                       [13, -13, 211, -211], item, 50.,
                                       field_error_tesla=0., position_error_mm=0.)
            source = np.repeat(np.arange(4), 3)
            lo, hi = np.zeros(12), np.tile([1., 10., 50.], 4)
            balls = model.balls(source, lo, hi)
            record['diagnostic'] = dict(x_mm=points[name], p3_mev=[0, 0, 100], pdg=[13, -13, 211, -211],
                source_index=balls.source_index.tolist(), path_mm=hi.tolist(),
                model_position_error_mm=balls.endpoint_model_error_mm.tolist(),
                midpoint_radius_mm=balls.midpoint_radius_mm.tolist(),
                chord_radius_mm=balls.chord_radius_mm.tolist(), geometry={})
            for arm, (geometry, sha) in scenes.items():
                owner = model.owners(source, lo, hi, geometry, geometry_error_mm=0.)
                record['diagnostic']['geometry'][arm] = dict(gdml_sha256=sha,
                    **{key: getattr(owner, key).tolist() for key in owner.__dataclass_fields__})
        records.append(record)
    return dict(schema='ts-single-map-model-enclosure-audit-v1', maps=records,
                errors_are_caller_assumptions=True, field_error_tesla=0., position_error_mm=0.,
                geometry_error_mm=0., zero_errors_are_numerical_idealization=True,
                floating_point_certified=False, native_query_certified=False,
                rk4_error_certified=False, propagated_endpoint=False, material_transport=False,
                production_tracks=False, transmission_measured=False,
                status_labels={0: 'unresolved', 1: 'conditional_vacuum', 2: 'conditional_material'})


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
