"""Validate selected fresh-beam muon identities without inferring POT normalization."""
import argparse
import json
from pathlib import Path

import numpy as np

from .target_source_events import _integers
from .target_parent_audit import checksum


COLUMNS = ('run', 'subrun', 'evt', 'trk', 'pdg', 'codestop', 'pstop', 'xstop', 'ystop', 'zstop', 'tstop')
KILLER_VOLUME = 54


def selected_beam(columns, accepted_keys):
    """Require every accepted selected event to contain a stock-selector muon."""
    if any(k not in columns for k in COLUMNS):
        raise ValueError('missing selected beam columns')
    a = {k: np.asarray(columns[k]) for k in COLUMNS}
    n = a['pdg'].size
    if not n or any(v.shape != (n,) for v in a.values()):
        raise ValueError('nonempty scalar beam columns required')
    if any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError('finite selected beam columns required')
    identity = np.column_stack([_integers(a[k], k, (n,)) for k in ('run', 'subrun', 'evt', 'trk')])
    if np.unique(identity, axis=0).shape[0] != n:
        raise ValueError('duplicate selected beam particle identity')
    accepted = np.asarray(accepted_keys)
    if accepted.ndim != 2 or accepted.shape[1] != 3 or not len(accepted):
        raise ValueError('nonempty accepted event keys required')
    accepted = _integers(accepted, 'accepted_keys')
    if np.unique(accepted, axis=0).shape[0] != len(accepted):
        raise ValueError('duplicate accepted event identity')
    expected = np.unique(accepted, axis=0)
    if not np.array_equal(np.unique(identity[:, :3], axis=0), expected):
        raise ValueError('selected beam event keys differ from accepted path')
    if not np.isin(a['pdg'], [13, -13]).all() or np.any(a['pstop'] < 0):
        raise ValueError('selected reader must contain only finite muon states')
    code = _integers(a['codestop'], 'codestop', (n,))
    eligible = code == KILLER_VOLUME
    keys, inverse = np.unique(identity[:, :3], axis=0, return_inverse=True)
    counts = np.bincount(inverse[eligible], minlength=len(keys))
    if not eligible.any() or np.any(counts == 0):
        raise ValueError('accepted beam event without stock killed-volume muon')
    multiplicity, frequency = np.unique(counts, return_counts=True)
    return dict(mode='selected-fresh-beam-identity-content-audit', selected_events=len(keys),
        selected_muon_rows=n, eligible_muons=int(eligible.sum()),
        eligible_mu_minus=int(np.count_nonzero(eligible & (a['pdg'] == 13))),
        eligible_mu_plus=int(np.count_nonzero(eligible & (a['pdg'] == -13))),
        event_multiplicity={str(int(k)): int(v) for k, v in zip(multiplicity, frequency)},
        eligible_momentum_range=[float(a['pstop'][eligible].min()), float(a['pstop'][eligible].max())],
        eligible_time_range=[float(a['tstop'][eligible].min()), float(a['tstop'][eligible].max())],
        exact_accepted_event_keys_verified=True, selected_particle_identities_unique=True,
        selection='PDG +/-13 and mu2eKillerVolume stopping process 54',
        source_event_ancestry_audited=False, compressed_volume_contract_verified=False,
        POT_normalization_inferred=False, generated_yield_validated=False, certificate_established=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'admission', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError('refusing existing beam audit output')
    receipt = json.loads(a.admission.read_text())
    if (receipt.get('process') != 'R1LibraryPOT' or receipt.get('all_event_status_zero') is not True
            or receipt.get('generation_census_reconciled') is not True):
        raise ValueError('completed reconciled POT admission required')
    import uproot
    with uproot.open(a.root) as root:
        tree = root['cetree/ntpart']
        columns = tree.arrays(list(COLUMNS), library='np')
        cycles = {k: int(root[k].num_entries) for k in root.keys() if k.split(';')[0] == 'cetree/ntpart'}
    report = selected_beam(columns, receipt['accepted_event_keys'])
    if report['selected_events'] != receipt['accepted_events']:
        raise ValueError('selected beam event count differs from admission')
    report.update(selected_tree_sha256=checksum(a.root), admission_sha256=checksum(a.admission), tree_cycles=cycles)
    with a.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
