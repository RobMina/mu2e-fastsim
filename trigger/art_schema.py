"""Inspect retained trigger/event metadata branch schemas without reading physics data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def describe(path):
    import uproot

    with uproot.open(path) as root:
        if 'Events' not in root or root['Events'].classname != 'TTree':
            raise ValueError('missing art Events TTree')
        tree = root['Events']
        branches = []
        tree_counts = {}
        for tree_name in ('Events', 'SubRuns', 'Runs'):
            if tree_name not in root:
                continue
            current = root[tree_name]
            if current.classname != 'TTree':
                raise ValueError(f'{tree_name} is not an art TTree')
            tree_counts[tree_name] = current.num_entries
            for name, branch in current.items(recursive=True):
                if any(token in name for token in ('TriggerResults', 'EventAuxiliary', 'Lumi', 'IntensityInfo', 'EventHeader', 'FilterFraction')):
                    branches.append({'tree': tree_name, 'name': name, 'typename': branch.typename,
                                     'interpretation': type(branch.interpretation).__name__})
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False) if key.startswith('Events;')]
        status_models = {}
        for name in ('art::HLTGlobalStatus', 'art::HLTPathStatus', 'art::TriggerResults', 'art::EventAuxiliary', 'art::EventID', 'art::SubRunID', 'art::RunID'):
            if name in root.file.streamers:
                status_models[name] = {str(version): {'elements': [{'name': element.name, 'typename': element.typename} for element in streamer.elements]}
                    for version, streamer in root.file.streamers[name].items()}
        return {'events': tree.num_entries, 'highest_cycle': max(cycles), 'branches': branches,
                'tree_entries': tree_counts, 'status_streamers': status_models,
                'streamer_classes': sorted(name for name in root.file.streamers
                    if any(token in name for token in ('TriggerResults', 'HLT', 'EventAuxiliary', 'IntensityInfo', 'EventHeader'))),
                'scope': 'ROOT branch/streamer metadata only; no product payload or path-state comparison'}


def describe_sgs(path):
    """Inventory persisted SGS/particle branches and streamers, without physics decoding."""
    import uproot
    import integrity
    before = integrity.digest(path)
    with uproot.open(path) as root:
        cycles = [int(key.rsplit(';', 1)[1]) for key in root.keys(recursive=False) if key.startswith('Events;')]
        if not cycles:
            raise ValueError('missing Events tree')
        tree = root[f'Events;{max(cycles)}']
        if tree.classname != 'TTree':
            raise ValueError('highest Events cycle is not a TTree')
        tokens = ('StrawGasStep', 'SimParticle', 'GenParticle', 'ProductRegistry', 'BranchDescription', 'ProductProvenance')
        branches = [{'name': name, 'typename': branch.typename, 'entries': branch.num_entries}
                    for name, branch in tree.items(recursive=True) if any(token in name for token in tokens)]
        streamers = {name: {str(version): [{'name': item.name, 'typename': item.typename} for item in streamer.elements]
                           for version, streamer in versions.items()}
                     for name, versions in root.file.streamers.items() if any(token in name for token in tokens)}
        result = {'source': before, 'events': tree.num_entries, 'highest_cycle': max(cycles),
                  'branches': branches, 'streamers': streamers,
                  'scope': 'persisted SGS branch and streamer metadata only; no payload decoding or presence claim'}
    if integrity.digest(path) != before:
        raise ValueError('source changed during metadata inventory')
    return result


def probe_status(path):
    import awkward as ak
    import uproot

    probes = []
    with uproot.open(path) as root:
        for name, branch in root['Events'].items(recursive=True):
            if 'TriggerResults' in name and name.endswith('.obj'):
                try:
                    values = branch.array(entry_stop=1, library='ak')
                    probes.append({'branch': name, 'type': str(ak.type(values)),
                                   'fields': ak.fields(values)})
                except Exception as error:
                    probes.append({'branch': name, 'error': f'{type(error).__name__}: {error}'})
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('art', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-status', action='store_true', help='probe one retained TriggerResults entry, not physics products')
    parser.add_argument('--sgs', action='store_true', help='inventory SGS branch/streamer metadata only')
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('refuse metadata inventory overwrite')
    report = describe_sgs(args.art) if args.sgs else describe(args.art)
    if args.probe_status:
        report['status_probe'] = probe_status(args.art)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Events={report['events']} metadata branches={len(report['branches'])}")


if __name__ == '__main__':
    main()
