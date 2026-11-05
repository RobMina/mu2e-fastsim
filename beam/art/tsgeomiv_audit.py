"""Bound stored transformed geometry on original exports, without native calls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from beamfastsim import tsgeom, tsgeomiv, tsinterval as iv, tssurface


def audit(root):
    records = []
    for arm in ('run1a', 'run1b'):
        path = root/f'geometry_{arm}_v1/ts_geometry_{arm}.gdml'
        scene = tsgeom.load(path, [f'TS{i}Vacuum' for i in range(1, 6)], vacuum_materials=['PSVacuum', 'DSVacuum'])
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        center_by_instance = {r['instance']: r['center_world_mm'] for r in scene.receipt['placed_volumes']}
        volumes = []
        for index, volume in enumerate(scene.partition.volumes):
            center = np.asarray([center_by_instance[volume.instance]], dtype=np.float64)
            points = iv.inflate(iv.Interval(center), iv.Interval(1e-6))
            result = tsgeomiv.evaluate(volume.solid, points)
            volumes.append(dict(index=index, instance=volume.instance, instance_path=list(volume.instance_path),
                material=volume.material, physical=volume.physical, logical=volume.logical, center_world_mm=center[0].tolist(),
                box_radius_mm=1e-6, clearance_lo=result.clearance.lo.tolist(), clearance_hi=result.clearance.hi.tolist(),
                visited_nodes=result.visited_nodes))
        faces = []
        for index, volume in enumerate(scene.partition.volumes):
            if volume.vacuum or 'Ti6Al4V' not in volume.material:
                continue
            for face in tssurface.faces(scene.partition, index):
                center = face.origin[None, :]
                points = iv.inflate(iv.Interval(center), iv.Interval(.0625))
                result = tsgeomiv.face_bounds(face, points)
                faces.append(dict(index=index, instance=volume.instance, instance_path=list(volume.instance_path),
                    material=volume.material, physical=volume.physical, face=face.key, origin_mm=face.origin.tolist(),
                    box_radius_mm=.0625, primitive_plane_lo_mm=result.primitive_plane.lo.tolist(),
                    primitive_plane_hi_mm=result.primitive_plane.hi.tolist(), guard_lo=result.guard.clearance.lo.tolist(),
                    guard_hi=result.guard.clearance.hi.tolist(), plane_discrepancy_upper_mm=result.plane_discrepancy_upper_mm.tolist(),
                    normal_difference_upper=result.normal_difference_upper, visited_nodes=result.guard.visited_nodes))
        records.append(dict(arm=arm, gdml_sha256=sha, volumes=volumes, faces=faces))
    return dict(schema='ts-stored-transformed-geometry-v1', records=records, stored_expression_arithmetic_outward=True,
        arithmetic_environment_assumed=True, unit_lipschitz_certified=False, floating_point_certified=False,
        decimal_and_trig_geometry_conversion_bounded=False, native_query_certified=False, unmodeled_siblings_excluded=False,
        ownership_or_contacts_advanced=False, defaults_changed=False, material_transport=False, production_tracks=False)


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
