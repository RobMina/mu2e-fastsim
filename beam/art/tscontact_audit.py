"""Conditional contact sets on original exports, not restart or material physics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beamfastsim import tscontact, tscarry, tsgeom, tssurface, tsfield


def audit(root):
    ctx = tscarry.MapContext(tsfield.load(root/'assets_v1/maps/TSdMap.header'))
    records = []
    for arm in ('run1a', 'run1b'):
        path = root/f'geometry_{arm}_v1/ts_geometry_{arm}.gdml'
        scene = tsgeom.load(path, [f'TS{i}Vacuum' for i in range(1, 6)],
                            vacuum_materials=['PSVacuum', 'DSVacuum'])
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        for index, volume in enumerate(scene.partition.volumes):
            if volume.vacuum or 'Ti6Al4V' not in volume.material:
                continue
            for face in tssurface.faces(scene.partition, index):
                start = face.origin-face.inward_normal*.03125
                initial = tscarry.State(ctx, np.tile(start, (4, 1)), np.tile(face.inward_normal, (4, 1)),
                    100., [13, -13, 211, -211], position_error_mm=1e-6, direction_error=1e-9)
                segment = tscarry.Segment(initial, .0625, field_error_tesla=1e-10,
                    position_error_mm=1e-6, direction_error=1e-9)
                entry = tssurface.entry(segment, face, plane_error_mm=1e-6, projection_error=1e-9,
                    geometry_error_mm=1e-6, path_atol=1e-4)
                contact = tscontact.contact(entry)
                if len(contact.row_index) != 4:
                    raise ValueError('chosen near-normal diagnostic no longer has four conditional entries')
                records.append(dict(arm=arm, gdml_sha256=sha, instance=volume.instance,
                    instance_path=list(volume.instance_path), material=volume.material, physical=volume.physical,
                    logical=volume.logical, face=face.key, origin_mm=face.origin.tolist(),
                    inward_normal=face.inward_normal.tolist(),
                    **{key: getattr(contact, key).tolist() for key in ('row_index', 'source_index', 'entry_status',
                        'local_lo_mm', 'local_hi_mm', 'x_center_mm', 'position_radius_mm', 'direction_vector',
                        'direction_radius', 'momentum', 'pdg')},
                    cumulative_path_lo_mm=contact.cumulative_path.lo.tolist(),
                    cumulative_path_hi_mm=contact.cumulative_path.hi.tolist(),
                    center_plane_residual_lo_mm=contact.center_plane_residual.lo.tolist(),
                    center_plane_residual_hi_mm=contact.center_plane_residual.hi.tolist(),
                    intended_plane_slab_lo_mm=contact.intended_plane_slab.lo.tolist(),
                    intended_plane_slab_hi_mm=contact.intended_plane_slab.hi.tolist(),
                    stored_plane_root_arithmetic_outward=entry.plane_root_arithmetic_outward))
    return dict(schema='ts-conditional-first-contact-set-audit-v1', records=records,
        errors_are_unproved_assumptions=True, initial_position_error_mm=1e-6, initial_direction_error=1e-9,
        evaluation_position_error_mm=1e-6, evaluation_direction_error=1e-9,
        field_error_tesla=1e-10, plane_representation_error_mm=1e-6,
        projection_representation_error=1e-9, geometry_error_mm=1e-6,
        local_arithmetic='Outward under IEEE binary64 basic-op/sqrt/gradual-underflow assumptions',
        evaluated_at_precontact_lower_endpoint=True,
        direction_center='Mathematical normalization of stored direction_vector, not rounded unit coordinates',
        position_set='Ball intersected with original conditional finite face, no projection',
        incoming_direction_only=True, floating_point_certified=False, native_query_certified=False,
        restart_state=False, material_transport=False, production_tracks=False, transmission_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must not exist')
    receipt = audit(args.root)
    with args.out.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    for r in receipt['records']:
        print(json.dumps({key: r[key] for key in ('arm', 'face', 'local_lo_mm', 'local_hi_mm',
            'position_radius_mm', 'direction_radius', 'center_plane_residual_lo_mm', 'center_plane_residual_hi_mm')}))


if __name__ == '__main__':
    main()
