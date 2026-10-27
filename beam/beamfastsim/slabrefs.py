"""Read selected monoenergetic G4 references without generating or modifying inputs."""
import json
from pathlib import Path

import numpy as np

from . import foilreport
from .reference_audit import _identity


def _identity_fields(item, label):
    if (not isinstance(item, dict) or not isinstance(item.get('sha256'), str)
            or len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256'])
            or not isinstance(item.get('bytes'), int) or isinstance(item['bytes'], bool) or item['bytes'] < 0):
        raise ValueError(f'missing hash/size identity for {label}')


def _checked(item, path, label):
    _identity_fields(item, label)
    actual = _identity(path)
    if any(actual[k] != item[k] for k in ('sha256', 'bytes')):
        raise ValueError(f'changed hash/size identity for {label}: {path}')
    return actual


def _local(item, directory):
    if not isinstance(item, dict) or not item.get('path'):
        raise ValueError('missing artifact/output identity path')
    return Path(foilreport._local(item['path'], directory)).resolve()


def _case(receipt, pdg, thickness, energy, cosine):
    if receipt.get('complete') is not True:
        raise ValueError('incomplete G4 receipt')
    case = receipt.get('case', {})
    expected = dict(pdg=pdg, thickness_mm=thickness, energy_MeV=energy, cosine=cosine)
    for key, value in expected.items():
        if key not in case or not np.isclose(case[key], value, rtol=0, atol=1e-12):
            raise ValueError(f'G4 receipt {key} differs from selected cell/PDG')
    if (not np.isclose(case.get('energy_max_MeV', energy), energy, rtol=0, atol=1e-12)
            or not np.isclose(case.get('cosine_min', cosine), cosine, rtol=0, atol=1e-12)):
        raise ValueError('G4 receipt is not the selected monoenergetic cell')
    for key in ('events', 'seed'):
        value = case.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < (1 if key == 'events' else 0):
            raise ValueError(f'missing or invalid G4 receipt {key}')
    return case


def _arrays(g, case, census):
    required = ('entry_MeV', 'entry_cos', 'exit_MeV', 'exit', 'physical_stop', 'other', 'exit_z_mm')
    if any(key not in g for key in required):
        raise ValueError('missing G4 response arrays')
    n = case['events']
    if any(np.asarray(value).shape != (n,) for value in g.values()):
        raise ValueError('G4 array length differs from receipt event census')
    for key in ('exit', 'physical_stop', 'other'):
        if not np.all(np.isin(g[key], [False, True])):
            raise ValueError(f'invalid G4 outcome flags: {key}')
    ex, stop = g['exit'].astype(bool), g['physical_stop'].astype(bool)
    energy = g['exit_MeV']
    if (np.any(g['other']) or np.any(ex == stop) or np.any(~np.isfinite(energy))
            or np.any(energy < 0) or np.any(ex & (energy <= 0)) or np.any(stop & (energy != 0))
            or np.any(~np.isfinite(g['exit_z_mm']))):
        raise ValueError('unresolved or inconsistent G4 outcomes')
    if (not np.allclose(g['entry_MeV'], case['energy_MeV'], rtol=0, atol=1e-5)
            or not np.allclose(g['entry_cos'], case['cosine'], rtol=0, atol=1e-6)):
        raise ValueError('G4 entries differ from selected cell')
    if 'event' in g and not np.array_equal(g['event'], np.arange(1, n + 1)):
        raise ValueError('G4 persisted event coverage mismatch')
    measured = dict(primaries=n, exits=int(ex.sum()), physical_stops=int(stop.sum()),
                    threshold_stops=int(np.sum(energy < .1)), other=0,
                    back_exits=int(np.sum(ex & (g['exit_z_mm'] < 0))))
    if not isinstance(census, dict) or 'primaries' not in census:
        raise ValueError('missing audited G4 census')
    if any(census[key] != value for key, value in measured.items() if key in census):
        raise ValueError('G4 response differs from audited census')


def _provenance(receipt, directory, modern):
    records = []
    if not isinstance(receipt.get('artifacts'), list) or 'output' not in receipt:
        raise ValueError('missing G4 artifact/output identities')
    for item in receipt['artifacts']:
        records.append(dict(_checked(item, _local(item, directory), 'G4 artifact'), role='artifact'))
    output = receipt['output']
    path = _local(output, directory)
    if path.is_file():
        output = dict(_checked(output, path, 'G4 output'), role='output')
        records.append(output)
    else:
        if not modern:
            raise ValueError(f'missing G4 ROOT output: {path}')
        _identity_fields(output, 'G4 output')
        output = dict(output, path=str(path), available=False, verified=False)
    return records, output


def _selected(manifest, cell, pdg, thickness, energy, cosine):
    for tag, run in manifest.get('runs', {}).items():
        if 'scan' in run:
            if run['scan'] != cell or run.get('pdg') != pdg:
                continue
            if not np.isclose(run.get('thickness_mm', np.nan), thickness, rtol=0, atol=1e-12):
                raise ValueError('G4 manifest thickness differs from selected cell')
            mode = run.get('mode')
            if not isinstance(mode, str) or not mode:
                raise ValueError('missing G4 mode label')
            yield tag, run, mode, True
        else:
            matches = [c for c in manifest.get('cases', []) if c[0] == cell]
            modes = [mode for mode in manifest.get('modes', {}) if tag == f'{cell}_{mode}']
            if not matches or not modes:
                continue
            if len(matches) != 1 or len(modes) != 1:
                raise ValueError('ambiguous legacy G4 cell/mode')
            if not np.allclose(matches[0][1:], [energy, thickness, cosine], rtol=0, atol=1e-12):
                raise ValueError('G4 manifest geometry/energy differs from selected cell')
            yield tag, run, modes[0], False


def load_references(campaigns, cell, pdg, thickness, energy, cosine):
    """Return ``({mode: {column: ndarray}}, identity_records)`` for complete runs.

    Campaigns may be one directory or an iterable. Samples are concatenated in
    full, never paired to fast rows. Every flat record is an existing verified
    file, suitable for slabstudy.verify; a dropped modern ROOT identity remains
    in its receipt record's ``recorded_output`` with ``available=False``.
    """
    if isinstance(campaigns, (str, Path)):
        campaigns = [campaigns]
    pooled, records, seeds, identities = {}, [], set(), set()
    for root in campaigns:
        root = Path(root).resolve()
        path = root / 'manifest.json'
        manifest_record = dict(_identity(path), role='manifest')
        manifest = json.loads(path.read_text())
        if manifest.get('complete') is not True:
            raise ValueError(f'incomplete G4 campaign: {root}')
        used = False
        for tag, run, mode, modern in _selected(manifest, cell, pdg, thickness, energy, cosine):
            directory = root / tag
            receipt_path = directory / 'receipt.json'
            record = dict(_checked(run.get('receipt'), receipt_path, 'G4 receipt'), role='receipt',
                          campaign=str(root), tag=tag, mode=mode)
            receipt = json.loads(receipt_path.read_text())
            if not modern:
                receipt_pdg = receipt.get('case', {}).get('pdg')
                if receipt_pdg not in (13, -13):
                    raise ValueError('missing or invalid G4 receipt pdg')
                if receipt_pdg != pdg:
                    continue
            case = _case(receipt, pdg, thickness, energy, cosine)
            provenance, output = _provenance(receipt, directory, modern)
            record['recorded_output'] = output
            record['seed'] = case['seed']
            sample_ids = {('receipt', record['sha256']), ('output', output['sha256'])}
            if case['seed'] in seeds or sample_ids & identities:
                raise ValueError('duplicate independent G4 sample identity/seed')
            if modern:
                scan = dict(_checked(run.get('scan_npz'), directory / 'scan.npz', 'G4 scan'), role='scan')
                if ('scan', scan['sha256']) in identities:
                    raise ValueError('duplicate independent G4 scan identity')
                sample_ids.add(('scan', scan['sha256']))
                with np.load(scan['path'], allow_pickle=False) as saved:
                    g = {key: saved[key] for key in saved.files}
                provenance.append(scan)
            else:
                report, g, audited = foilreport.audit(directory)
                if audited['case'] != case:
                    raise ValueError('G4 receipt changed during audit')
                _arrays(g, case, report)
            _arrays(g, case, run.get('census'))
            _checked(run['receipt'], receipt_path, 'G4 receipt')
            seeds.add(case['seed'])
            identities.update(sample_ids)
            if not used:
                records.append(manifest_record)
                used = True
            records.extend([record, *provenance])
            if mode in pooled and set(g) != set(pooled[mode][0]):
                raise ValueError(f'inconsistent G4 array schema for mode {mode}')
            pooled.setdefault(mode, []).append(g)
        _checked(manifest_record, path, 'G4 manifest')
    if not pooled:
        raise ValueError('no matching G4 references for selected cell/PDG')
    refs = {mode: {key: np.concatenate([g[key] for g in runs]) for key in runs[0]}
            for mode, runs in pooled.items()}
    return refs, records
