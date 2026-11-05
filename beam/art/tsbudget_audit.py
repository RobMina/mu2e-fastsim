"""Read-only outward bound scans on existing maps; no transport or simulation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from beamfastsim import tsbound, tsbudget, tsfield


def audit(root):
    records = []
    for directory, names in (('assets_v1', ('PSMap', 'TSuMap_fix', 'TSdMap', 'PSAreaMap', 'WorldMap')),
                             ('assets_dump_v1', ('PStoDumpAreaMap', 'ProtonDumpAreaMap'))):
        for name in names:
            item = tsfield.load(root/directory/'maps'/f'{name}.header')
            bounds = tsbudget.scan(item)
            old_b, old_g = tsbound.magnitude(item), tsbound.spatial(item)
            lengths = np.array([.0625, 1., 5., 50.])
            errors = tsbudget.errors(bounds, np.full(4, 100.), lengths,
                position_error_mm=1e-6, direction_error=1e-9, field_error_tesla=1e-10)
            records.append(dict(map=name, bounds=asdict(bounds), old_magnitude_tesla=old_b.magnitude_tesla,
                old_gradient_tesla_per_mm=old_g.lipschitz_tesla_per_mm, momentum_mev=100.,
                path_mm=lengths.tolist(), **{key: getattr(errors, key).tolist() for key in
                    ('position_mm', 'direction', 'curvature_per_mm')}))
    return dict(schema='ts-outward-map-model-budget-v1', records=records,
        initial_position_error_mm=1e-6, initial_direction_error=1e-9, field_evaluation_error_tesla=1e-10,
        initial_evaluation_errors_remain_unproved_assumptions=True, bound_arithmetic_outward=True,
        arithmetic_environment_assumed=True, floating_point_certified=False, native_query_certified=False,
        domain_coverage_checked=False, field_interpolation_certified=False, helix_certified=False,
        propagated_states=False, material_transport=False, production_tracks=False)


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
