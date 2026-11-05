"""Read stored fields and compare bounded uniform endpoints, not transport."""

import argparse
import json
from pathlib import Path

import numpy as np

from beamfastsim import tsfield, tshelix, tsquery
from mu2efastsim.ehelix import helix


def audit(root):
    records = []
    for directory, names in (('assets_v1', ('PSMap', 'TSuMap_fix', 'TSdMap', 'PSAreaMap', 'WorldMap')),
                             ('assets_dump_v1', ('PStoDumpAreaMap', 'ProtonDumpAreaMap'))):
        for name in names:
            item = tsfield.load(root/directory/'maps'/f'{name}.header')
            x = np.broadcast_to(item.lo+.375*(item.hi-item.lo), (4, 3)).copy()
            raw = np.broadcast_to([.25, .125, 1.], (4, 3)).copy()
            p, s, pdg = np.full(4, 100.), np.full(4, 5.), np.array([13, -13, 211, -211])
            query = tsquery.evaluate(item, x)
            receipt = tshelix.evaluate(x, raw, p, pdg, query.value_tesla, s)
            q = np.where((pdg == 13) | (pdg == -211), -1., 1.)
            xx, uu = helix(x, raw/np.hypot.reduce(raw, axis=1)[:, None], p, q, query.value_tesla, s)
            errors = tshelix.difference(receipt, xx, uu)
            records.append(dict(map=name, x_mm=x.tolist(), direction_raw=raw.tolist(), p_mev=100., path_mm=5.,
                pdg=pdg.tolist(), field_tesla=query.value_tesla.tolist(),
                field_evaluation_error_tesla=query.evaluation_error_tesla.tolist(),
                core_position_error_mm=errors.position_mm.tolist(), core_direction_error=errors.direction.tolist(),
                enclosure_position_radius_mm=receipt.position_radius_mm.tolist(),
                enclosure_direction_radius=receipt.direction_radius.tolist(), phase_upper=receipt.phase_upper.tolist(),
                direction_tail=receipt.direction_tail.tolist(), position_tail_mm=receipt.position_tail_mm.tolist()))
    return dict(schema='ts-outward-uniform-helix-v1', records=records, evaluation_arithmetic_outward=True,
        transcendental_library_assumed=False, arithmetic_environment_assumed=True, floating_point_certified=False,
        native_query_certified=False, varying_field_trajectory_bounded=False, geometry_certified=False,
        defaults_changed=False, material_transport=False, production_tracks=False,
        field_error_separate_from_frozen_field_helix_errors=True)


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
