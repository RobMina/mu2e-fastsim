"""Read the trigger menu (a `fhicl-dump`-style replay configuration) into the per-path cut tables the device uses.

Only the tracker paths the device evaluates are extracted; every threshold comes from the configuration file, none
is restated here. The parser understands the subset of FHiCL that `fhicl-dump` writes (nested tables, sequences,
strings, numbers, booleans, `#` comments) plus dotted top-level overrides.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r'\s*(?:(#[^\n]*)|("(?:[^"\\]|\\.)*")|([{}\[\],])|([A-Za-z_@][\w.:@-]*)\s*:(?!:)|([^\s{}\[\],]+))')


def _tokens(text):
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == '':
                return
            raise ValueError(f'cannot parse FHiCL near: {text[pos:pos + 60]!r}')
        pos = m.end()
        comment, string, punct, key, atom = m.groups()
        if comment:
            continue
        if string is not None:
            yield ('val', string[1:-1])
        elif punct:
            yield (punct, punct)
        elif key:
            yield ('key', key)
        elif atom is not None:
            yield ('val', _atom(atom))


def _atom(s):
    if s in ('true', 'false'):
        return s == 'true'
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _value(toks, i):
    kind, v = toks[i]
    if kind == '{':
        return _table(toks, i + 1)
    if kind == '[':
        out, i = [], i + 1
        while toks[i][0] != ']':
            if toks[i][0] == ',':
                i += 1
                continue
            item, i = _value(toks, i)
            out.append(item)
        return out, i + 1
    return v, i + 1


def _table(toks, i):
    out = {}
    while i < len(toks) and toks[i][0] != '}':
        kind, key = toks[i]
        if kind != 'key':
            raise ValueError(f'expected a key, got {toks[i]!r}')
        val, i = _value(toks, i + 1)
        _assign(out, key, val)
    return out, i + 1


def _assign(table, dotted, val):
    parts = dotted.split('.')
    for p in parts[:-1]:
        table = table.setdefault(p, {})
    table[parts[-1]] = val


def parse_fcl(text):
    """FHiCL text -> nested dict (later assignments override earlier ones, as in FHiCL)."""
    toks = list(_tokens(text))
    out, i = {}, 0
    while i < len(toks):
        kind, key = toks[i]
        if kind != 'key':
            raise ValueError(f'expected a top-level key, got {toks[i]!r}')
        val, i = _value(toks, i + 1)
        _assign(out, key, val)
    return out


# module types the device evaluates, by role in a path
ROLES = {'TimeClusterFilter': 'tc', 'HelixFilter': 'hs', 'KalSeedFilter': 'ks', 'DigiFilter': 'sd'}
PATTERN_RECOGNITION = {'TimeClusterFinder': 'tpr', 'TZClusterFinder': 'apr', 'SimpleTimeCluster': 'cst'}


def device_paths(cfg, names=None):
    """Per tracker path: its filters' settings in path order and which time-cluster producer feeds them.

    Returns {path: {'family': tpr|apr|cst, 'tc_producer': label, 'pitch': float, 'filters': [(role, label, cfg)]}}
    for every path that has a TimeClusterFilter and no calorimeter or CRV filter.
    """
    phys = cfg['physics']
    modules = {**phys.get('producers', {}), **phys.get('filters', {})}
    out = {}
    for entry in phys['trigger_paths']:
        path = entry.split(':', 1)[-1]
        if names and path not in names:
            continue
        labels = phys.get(path, [])
        types = [modules.get(lbl, {}).get('module_type') for lbl in labels]
        if 'TimeClusterFilter' not in types:
            continue
        filters = [(ROLES[t], lbl, modules[lbl]) for lbl, t in zip(labels, types) if t in ROLES]
        tcf = next(c for r, _, c in filters if r == 'tc')
        producer = tcf['timeClusterCollection'].split(':')[0]
        family = PATTERN_RECOGNITION.get(modules[producer]['module_type'])
        if family is None:
            continue
        unsupported = [t for t in types if t and t.endswith('Filter') and t not in ROLES
                       and t not in ('PrescaleEvent', 'TriggerInfoMerger')]
        out[path] = {'family': family, 'tc_producer': producer, 'filters': filters,
                     'pitch': float(modules[producer].get('AveragePitch', 0.6)), 'unsupported': unsupported,
                     'min_helices': min([int(c.get('minNHelices', 1)) for r, _, c in filters if r == 'hs'] or [1])}
    return out


def prescales(cfg, path, event_mode):
    """PrescaleEvent factor of a path in an event mode (OnSpill/OffSpill); -1 means disabled."""
    phys = cfg['physics']
    for lbl in phys.get(path, []):
        m = phys.get('filters', {}).get(lbl, {})
        if m.get('module_type') == 'PrescaleEvent':
            for row in m.get('eventModeConfig', []):
                if row.get('eventMode') == event_mode:
                    return int(row['prescale'])
    return 1
