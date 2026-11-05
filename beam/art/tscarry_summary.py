"""Compact summary of a persisted carried-reference audit, no re-propagation."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.input.read_text())
    if receipt.get('schema') != 'ts-carried-reference-audit-v1':
        parser.error('expected carried-reference audit')
    for item in receipt['maps']:
        for arm, geometry in item['geometry'].items():
            attempts = geometry['attempts']
            last = attempts[-1]
            print(json.dumps(dict(map=item['map'], arm=arm, attempts=len(attempts),
                status=last['status'], path_mm=last['path_mm'],
                position_error_mm=last['position_error_mm'], direction_error=last['direction_error'],
                fixed_momentum_all=all(row['momentum_unchanged'] for row in attempts),
                owners=last['owner_index'], conditional_diagnostic_only=True), allow_nan=False))


if __name__ == '__main__':
    main()
