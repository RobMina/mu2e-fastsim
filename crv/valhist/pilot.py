"""Provenance, FHiCL rendering and persisted-entry certification for a fresh A1/E2E pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import uproot


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tree_inventory(path, tree, ids=('run', 'subrun', 'event')):
    with uproot.open(path) as f:
        cycles = sorted(int(k.rsplit(';', 1)[1]) for k in f.keys(recursive=True, cycle=True)
                        if k.rsplit(';', 1)[0] == tree)
        if not cycles:
            raise ValueError(f'{path}: missing {tree}')
        t = f[f'{tree};{cycles[-1]}']
        a = t.arrays(list(ids), library='np')
        columns = [a[k] for k in ids]
        keys = np.stack(columns, axis=1)
        unique = np.unique(keys.astype('<i8'), axis=0)
        return dict(cycles=cycles, selected_cycle=cycles[-1], rows=t.num_entries,
                    unique_events=len(unique), event_keys_sha256=hashlib.sha256(unique.tobytes()).hexdigest())


def revision(path):
    git = lambda *a: subprocess.check_output(['git', '-C', str(path), *a], text=True).strip()
    return dict(path=str(path), head=git('rev-parse', 'HEAD'), status=git('status', '--short'))


def prepare(out, fastc, library, area, core, gen_art, seed):
    out, fastc, library, area, core = map(lambda p: Path(p).resolve(), (out, fastc, library, area, core))
    code = Path(__file__).resolve().parents[1]
    if out.exists():
        raise FileExistsError(f'refusing existing pilot output: {out}')
    for p in (fastc, library, area / 'backing/Offline/CRVDQM/data/CRVDQM_binning_v3_extracted.txt'):
        if not p.is_file():
            raise FileNotFoundError(p)
    inv = tree_inventory(fastc, 'dump/gen')
    if inv['unique_events'] <= 0:
        raise ValueError('empty generated-event population')
    files = [fastc, library, area / 'backing/Offline/CRVDQM/data/CRVDQM_binning_v3_extracted.txt']
    files += sorted((area / 'CrvFastSim').rglob('*.fcl'))
    files += sorted((area / 'CrvFastSim/src').glob('*'))
    files += sorted((area / 'build').glob('*/CrvFastSim/lib/*.so'))
    files += sorted((area / 'build').glob('*/DQM/lib/*Crv*so'))
    for package in ('crvfastsim', 'valhist'):
        files += sorted((code / package).rglob('*.py'))
    files += sorted((core / 'mu2efastsim').rglob('*.py'))
    files += sorted((code / 'valhist').glob('*.sh'))
    manifest = dict(crv=revision(code), core=revision(core), area=str(area),
                    backing=str((area / 'backing').resolve()), dqm=revision(area / 'DQM'),
                    fastc=str(fastc), library=str(library), seed=seed, gen_art=gen_art,
                    generated=inv, files={str(p): digest(p) for p in files if p.is_file()},
                    scope='A1/E2E using retained corrected transport; not an aluminium-only A/B')
    data = out.parents[1]
    assets = [data / 'geometry/crvcalib_sim_best_v1_5_run1400.csv',
              data / 'geometry/crvphoton_sim_best_v1_5_run1400.csv']
    from crvfastsim.reco.waveform import WaveformConfig
    from crvfastsim.geometry import find_counters
    from crvfastsim.response.lookup import lookup_dir
    assets += sorted(Path(lookup_dir()).iterdir())
    assets += [Path(find_counters(WaveformConfig().single_pe_file))]
    manifest['files'].update({str(p): digest(p) for p in assets if p.is_file()})
    manifest['asset_paths'] = [str(p) for p in assets]
    manifest['asset_paths_missing'] = [str(p) for p in assets if not p.is_file()]
    if manifest['asset_paths_missing']:
        raise FileNotFoundError(manifest['asset_paths_missing'])
    out.mkdir(parents=True)
    for d in ('tables', 'logs', 'a1', 'e2e'):
        (out / d).mkdir()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    table = out / 'tables' / fastc.name.replace('fastC.', 'fast1b.')
    reco = out / 'tables/chain.root'
    for arm, base in [('a1', 'valhist_faststeps.fcl'), ('e2e', 'valhist_fromtable.fcl')]:
        text = f'#include "CrvFastSim/fcl/{base}"\n'
        if arm == 'a1':
            text += f'physics.producers.fastCrvSteps.tableFile : "{table}"\n'
        else:
            for name in ('CrvDigi', 'CrvRecoPulses', 'CrvCoincidenceClusterFinder'):
                text += f'physics.producers.{name}.tableFile : "{reco}"\n'
        text += 'physics.analyzers.crvdump : { module_type : CrvRecoTableDumper mc : false }\n'
        text += 'physics.ValPath : [ Validation, crvDigiDQM, crvRecoDQM, dqmCrv, crvdump ]\n'
        (out / f'{arm}.fcl').write_text(text)
    print(f'prepared {out}: {inv["unique_events"]} persisted generated events', flush=True)


def verify(out):
    out = Path(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    expected = manifest['generated']['unique_events']
    evidence = {}
    passed = True
    for arm in ('a1', 'e2e'):
        p = out / arm / 'valhist.root'
        inventories = {t: tree_inventory(p, 'crvdump/' + t) for t in ('events', 'digis', 'pulses', 'clusters')}
        events = inventories['events']
        log = (out / 'logs' / f'{arm}.log').read_text()
        rc = (out / 'logs' / f'{arm}.exit').read_text().strip()
        ok = (events['rows'] == events['unique_events'] == expected and rc == '0' and
              events['event_keys_sha256'] == manifest['generated']['event_keys_sha256'] and
              'Art has completed and will exit with status 0.' in log)
        evidence[arm] = dict(path=str(p), sha256=digest(p), trees=inventories, completed=ok)
        passed &= ok
    drift = [p for p, sha in manifest['files'].items() if digest(p) != sha]
    passed &= not drift
    result = dict(expected_generated_events=expected, arms=evidence, changed_inputs=drift, completed=bool(passed))
    (out / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('out', 'fastc', 'library', 'area', 'core', 'gen-art'):
        prep.add_argument('--' + name, required=True)
    prep.add_argument('--seed', type=int, required=True)
    check = sub.add_parser('verify')
    check.add_argument('out')
    a = vars(p.parse_args())
    command = a.pop('command')
    (prepare if command == 'prepare' else verify)(**a)


if __name__ == '__main__':
    main()
