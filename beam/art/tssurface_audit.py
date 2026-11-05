"""Read-only planar-entry diagnostics in existing exports, not material transport."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beamfastsim import tscarry, tsgeom, tssurface, tsfield, tsbound


def audit(root):
    item = tsfield.load(root/'assets_v1/maps/TSdMap.header')
    ctx = tscarry.MapContext(item)
    records = []
    for arm in ('run1a', 'run1b'):
        gdml = root/f'geometry_{arm}_v1/ts_geometry_{arm}.gdml'
        scene = tsgeom.load(gdml, [f'TS{i}Vacuum' for i in range(1, 6)],
                            vacuum_materials=['PSVacuum', 'DSVacuum'])
        with gdml.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        for index, volume in enumerate(scene.partition.volumes):
            if volume.vacuum or 'Ti6Al4V' not in volume.material:
                continue
            try:
                faces = tssurface.faces(scene.partition, index)
            except ValueError as exc:
                records.append(dict(arm=arm, gdml_sha256=sha, instance=volume.instance,
                    physical=volume.physical, material=volume.material, supported=False, reason=str(exc)))
                continue
            for face in faces:
                point = face.origin-face.inward_normal*.03125
                if not tsbound.box_coverage(ctx.item, (point-.0625)[None, :],
                                            (point+.0625)[None, :], continuous=True)[0]:
                    raise ValueError('chosen AP diagnostic lacks continuous map coverage')
                state = tscarry.State(ctx, np.tile(point, (4, 1)), np.tile(face.inward_normal, (4, 1)),
                    100., [13, -13, 211, -211], position_error_mm=1e-6, direction_error=1e-9)
                seg = tscarry.Segment(state, .0625, field_error_tesla=1e-10,
                    position_error_mm=1e-6, direction_error=1e-9)
                result = tssurface.entry(seg, face, plane_error_mm=1e-6, projection_error=1e-9,
                    geometry_error_mm=1e-6, path_atol=1e-4)
                records.append(dict(arm=arm, gdml_sha256=sha, supported=True, volume_index=index,
                    instance=volume.instance, instance_path=list(volume.instance_path), physical=volume.physical,
                    logical=volume.logical, material=volume.material, face=face.key,
                    face_origin_mm=face.origin.tolist(), inward_normal=face.inward_normal.tolist(),
                    initial_x_mm=point.tolist(), hypothetical_vacuum_extension_mm=.0625,
                    **{key: getattr(result, key).tolist() for key in
                       ('status', 'unique_entry', 'lo_mm', 'hi_mm', 'derivative_lower', 'guard_margin_mm', 'iterations')}))
    if not any(r.get('supported') for r in records):
        raise ValueError('no supported chosen AP faces; diagnostic selection needs review')
    return dict(schema='ts-planar-material-first-entry-audit-v1', records=records, labels=tssurface.STATUS,
        source='Existing original exports and staged TSd map only; chosen starts, not production tracks',
        initial_position_error_mm=1e-6, initial_direction_error=1e-9, field_error_tesla=1e-10,
        evaluation_position_error_mm=1e-6, evaluation_direction_error=1e-9, plane_error_mm=1e-6,
        projection_error=1e-9, geometry_error_mm=1e-6, path_atol_mm=1e-4,
        error_declarations_are_unproved_assumptions=True, floating_point_certified=False,
        native_query_certified=False, material_transport=False, transmission_measured=False,
        incoming_vacuum_extension_not_valid_after_contact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must not exist; preserve original receipts')
    receipt = audit(args.root)
    with args.out.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    for record in receipt['records']:
        print(json.dumps(record, allow_nan=False))


if __name__ == '__main__':
    main()
