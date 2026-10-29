"""Diagnose primary/generator PDG assumptions from retained flat Cat dumps, without repairing them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from .replay import sha256

IDS = ['run', 'subrun', 'event']
PART = [*IDS, 'key', 'parent', 'isprim', 'genidx', 'pdg', 'ccode', 'se', 'st']
GEN = [*IDS, 'idx', 'pdg', 'e', 't']


def summarize(parts, gen):
    """Event-scoped key join only; legacy dumps cannot establish collection ProductID equality."""
    for rows, key in ((parts, 'key'), (gen, 'idx')):
        if rows.duplicated([*IDS, key]).any():
            raise ValueError('duplicate event-scoped particle/generator keys')
        for column in [*IDS, key]:
            if rows[column].dtype.kind not in 'iu':
                raise ValueError('integer identities required')
    if not parts.isprim.isin([0, 1]).all() or not parts.genidx.ge(-1).all():
        raise ValueError('invalid primary flag or generator index')
    linked = parts.loc[parts.genidx >= 0].merge(
        gen.rename(columns={'idx': 'genidx', 'pdg': 'gen_pdg', 'e': 'gen_e', 't': 'gen_t'}),
        on=[*IDS, 'genidx'], how='left', validate='many_to_one', indicator=True)
    if not linked['_merge'].eq('both').all():
        raise ValueError('missing generator key')
    linked = linked.drop(columns='_merge')
    mismatch = linked.pdg != linked.gen_pdg
    primary = linked.isprim == 1
    product_columns = ('gen_product_id' in parts, 'gen_product_id' in gen)
    checked = all(product_columns)
    if any(product_columns) and not checked:
        raise ValueError('incomplete ProductID evidence')
    if checked:
        if not linked.gen_product_id_x.eq(linked.gen_product_id_y).all():
            raise ValueError('generator ProductID mismatch')
    summary = {'particles': len(parts), 'linked_particles': len(linked),
        'primary_particles': int(parts.isprim.sum()), 'primary_missing_generator': int(((parts.isprim == 1) & (parts.genidx < 0)).sum()),
        'linked_pdg_mismatches': int(mismatch.sum()), 'linked_primary_pdg_mismatches': int((primary & mismatch).sum()),
        'linked_primary_with_parent': int((primary & (linked.parent >= 0)).sum()),
        'generator_ProductIDs_checked': checked,
        'mismatch_types': linked.loc[primary & mismatch].groupby(['pdg', 'gen_pdg', 'ccode'], sort=True).size().reset_index(name='particles').to_dict('records')}
    return summary, linked


def run(path, out, event=None):
    path, out = Path(path), Path(out)
    if out.exists():
        raise ValueError('choose new report')
    before = sha256(path)
    with uproot.open(path) as root:
        frames = {}
        for tree, columns in (('particles', PART), ('gen', GEN)):
            node = root['dump/' + tree]
            requested = columns + (['gen_product_id'] if 'gen_product_id' in node else [])
            frames[tree] = pd.DataFrame(node.arrays(requested, library='np'))
        counts = {k: root['dump/' + k].num_entries for k in ('events', 'steps', 'particles', 'gen') if 'dump/' + k in root}
    result, linked = summarize(frames['particles'], frames['gen'])
    if event is not None:
        selected = linked.loc[np.all(linked[IDS].to_numpy() == np.asarray(event), axis=1)]
        result['requested_event'] = list(event)
        result['requested_event_linked_particles'] = selected.replace({np.nan: None}).to_dict('records')
    if sha256(path) != before:
        raise ValueError('dump changed during audit')
    result.update(input=str(path), input_sha256=before, entries_highest_cycle=counts, code_sha256=sha256(__file__),
        scope='Diagnostic only; no ancestry repair or collection-identity inference for legacy dumps. '
              'A primary flag plus generator key does not by itself establish original generated-particle identity in staged samples.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--out', required=True)
    parser.add_argument('--event', type=int, nargs=3, metavar=('RUN', 'SUBRUN', 'EVENT'))
    args = parser.parse_args(argv)
    run(args.input, args.out, args.event)


if __name__ == '__main__':
    main()
