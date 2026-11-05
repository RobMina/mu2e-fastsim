"""Read existing stored maps; no tracks, native calls or material transport."""

import argparse
import json
from pathlib import Path

import numpy as np

from beamfastsim import tsfield, tsquery


def audit(root):
    records = []
    for directory, names in (('assets_v1', ('PSMap', 'TSuMap_fix', 'TSdMap', 'PSAreaMap', 'WorldMap')),
                             ('assets_dump_v1', ('PStoDumpAreaMap', 'ProtonDumpAreaMap'))):
        for name in names:
            item = tsfield.load(root/directory/'maps'/f'{name}.header')
            points = np.stack((item.lo+.375*(item.hi-item.lo), item.lo+.625*(item.hi-item.lo)))
            result = tsquery.evaluate(item, points, position_radius_mm=1e-6)
            records.append(dict(map=name, points_mm=points.tolist(), radius_mm=1e-6,
                nominal_values_tesla=result.value_tesla.tolist(),
                evaluation_error_tesla=result.evaluation_error_tesla.tolist(),
                region_error_tesla=result.region_error_tesla.tolist(),
                nominal_lo_tesla=result.nominal_field.lo.tolist(), nominal_hi_tesla=result.nominal_field.hi.tolist(),
                region_lo_tesla=result.region_field.lo.tolist(), region_hi_tesla=result.region_field.hi.tolist(),
                cell_count=result.cell_count.tolist()))
    return dict(schema='ts-stored-grid-query-v1', records=records, query_arithmetic_outward=True,
        stored_binary64_metadata_semantics=True, decimal_header_conversion_bounded=False,
        native_query_certified=False, floating_point_certified=False, arithmetic_environment_assumed=True,
        field_seam_continuity_claimed=False, helix_certified=False, geometry_certified=False,
        transport_defaults_changed=False, material_transport=False, production_tracks=False)


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
