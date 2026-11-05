"""Whole incoming vacuum sets at selected existing exported faces, not transport."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beamfastsim import tsfield, tsgeom, tssurface, tstube


def audit(root):
    ctx = tstube.Context(tsfield.load(root/'assets_v1/maps/TSdMap.header'))
    records = []
    for arm in ('run1a', 'run1b'):
        path = root/f'geometry_{arm}_v1/ts_geometry_{arm}.gdml'
        scene = tsgeom.load(path, [f'TS{i}Vacuum' for i in range(1, 6)], vacuum_materials=['PSVacuum', 'DSVacuum'])
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        for index, volume in enumerate(scene.partition.volumes):
            if volume.vacuum or 'Ti6Al4V' not in volume.material:
                continue
            for face in tssurface.faces(scene.partition, index):
                x = np.tile(face.origin-face.inward_normal*.03125, (4, 1))
                raw = np.tile(face.inward_normal, (4, 1))
                tube = tstube.enclose(ctx, x, raw, 100., np.array([13, -13, 211, -211]), 0., .0625,
                    position_error_mm=1e-6, direction_error=1e-9)
                geometry = tstube.guard(tube, face)
                records.append(dict(arm=arm, gdml_sha256=sha, instance=volume.instance, instance_path=list(volume.instance_path),
                    material=volume.material, face=face.key, x_mm=x.tolist(), pdg=tube.pdg.tolist(), path_mm=[0., .0625],
                    field_error_tesla=tube.query.evaluation_error_tesla.tolist(),
                    model_position_mm=tube.model.position_mm.tolist(), model_direction=tube.model.direction.tolist(),
                    position_lo_mm=tube.position.lo.tolist(), position_hi_mm=tube.position.hi.tolist(),
                    direction_lo=tube.direction.lo.tolist(), direction_hi=tube.direction.hi.tolist(),
                    guard_lo_mm=geometry.guard.clearance.lo.tolist(), guard_hi_mm=geometry.guard.clearance.hi.tolist(),
                    primitive_plane_lo_mm=geometry.primitive_plane.lo.tolist(), primitive_plane_hi_mm=geometry.primitive_plane.hi.tolist(),
                    plane_discrepancy_upper_mm=geometry.plane_discrepancy_upper_mm.tolist()))
    return dict(schema='ts-whole-interval-stored-model-v1', records=records, initial_position_error_mm=1e-6,
        initial_direction_error=1e-9, initial_errors_are_assumptions=True, field_and_helix_errors_computed=True,
        whole_interval=True, arithmetic_environment_assumed=True, floating_point_certified=False,
        native_query_certified=False, intended_geometry_certified=False, unique_contact_claimed=False,
        hypothetical_incoming_vacuum_only=True, restart_state=False, material_transport=False, production_tracks=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('preserve existing receipts')
    result = audit(args.root)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
