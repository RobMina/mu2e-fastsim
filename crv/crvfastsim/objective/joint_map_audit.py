"""Read-only candidate ordinal-map agreement and retained P4 config evidence, not cabling certification."""
import argparse
from pathlib import Path
import re

import numpy as np

from . import joint_compare as J, validation as V


def map_rows(values):
    a = np.asarray(values)
    J.require(a.ndim == 2 and a.shape[1] == 4 and np.all(np.isfinite(a)) and
              np.all(a == np.floor(a)) and np.all(a[:, 0] >= 0) and np.all(a[:, 1:] >= -1) and
              np.all(a < 2**31), "invalid ordinal map")
    a = a.astype(np.int64)
    J._unique(a[:, 0], "offline channels")
    missing = np.any(a[:, 1:] == -1, axis=1)
    J.require(np.all(a[missing, 1:] == -1), "partial unavailable online identity")
    online = a[~missing, 1:]
    J.require(len(np.unique(online, axis=0)) == len(online), "duplicate online channels")
    return a[np.argsort(a[:, 0])]


def compare_maps(calib, candidate):
    calib, candidate = map_rows(calib), map_rows(candidate)
    cm = calib[np.all(calib[:, 1:] >= 0, axis=1)]
    cand = candidate[np.all(candidate[:, 1:] >= 0, axis=1)]
    common, ci, mi = np.intersect1d(cm[:, 0], cand[:, 0], return_indices=True)
    mismatch = np.any(cm[ci, 1:] != cand[mi, 1:], axis=1)
    return dict(calibration_mapped=len(cm), calibration_unmapped=len(calib)-len(cm), candidate_mapped=len(cand),
                common_channels=len(common), mismatched_common=int(mismatch.sum()),
                mismatch_channels=common[mismatch].tolist(),
                calibration_only_channels=np.setdiff1d(cm[:, 0], cand[:, 0]).tolist(),
                candidate_only_channels=np.setdiff1d(cand[:, 0], cm[:, 0]).tolist(),
                exact_mapped_agreement=np.array_equal(cm, cand))


def config_evidence(text):
    process = text.split('CrvPass1:', 1)
    J.require(len(process) == 2, "missing retained CrvPass1 process")
    body = process[1].split('\nDL0:', 1)[0]
    def block(name):
        found = re.findall(r'\b'+re.escape(name)+r':\s*\{([^{}]*)\}', body)
        J.require(len(found) == 1, f"ambiguous or missing {name} block")
        return found[0]
    def value(content, name):
        found = re.findall(r'\b'+re.escape(name)+r':\s*("[^"\n]*"|[^\s{},]+)', content)
        J.require(len(found) == 1, f"ambiguous or missing {name}")
        return found[0].strip('"')
    ordinal, digi, db = block('crvOrdinal'), block('CrvDigi'), block('DbService')
    J.require(value(ordinal, 'useDb') == 'false', "retained map was not file-based")
    J.require(value(digi, 'useROC4asROC2') == 'true', "retained ROC remapping differs")
    J.require(value(digi, 'module_type') == 'CrvDigisFromArtdaqFragmentsFEBII', "unexpected unpacker")
    alias = 'Offline/Mu2eG4/geom/geom_common_extracted.txt'
    J.require(f'inputFile: "{alias}"' in body, "missing retained geometry alias")
    path = re.findall(r'^Processing file:\s*(\S+)', text, flags=re.M)
    J.require(len(path) == 1, "missing config source file")
    return dict(config_source_art=path[0], ordinal_use_db=False, ordinal_path=value(ordinal, 'filePath'),
                useROC4asROC2=True, geometry_alias=alias,
                database={k: value(db, k) for k in ('purpose', 'version', 'nearestMatch')})


def run(timing, calib, candidate, alternate, config, sources, out):
    J.require(not Path(out).exists(), "output already exists")
    receipts = {}
    for source in (Path(__file__), Path(J.__file__), Path(V.__file__)):
        J._receipt(source, receipts)
    report = J._json(timing, receipts)
    asset = report['map']['asset']
    J.require(J._receipt(calib, receipts)['sha256'] == asset['sha256'], "frozen calibration hash mismatch")
    c = np.loadtxt(calib, delimiter=',', comments='#', ndmin=2)[:, [0, 6, 7, 8]]
    maps = {}
    for label, path in (('candidate', candidate), ('alternate', alternate)):
        J._receipt(path, receipts)
        maps[label] = compare_maps(c, np.loadtxt(path, skiprows=1, ndmin=2))
    J._receipt(config, receipts)
    config_result = config_evidence(Path(config).read_text())
    for path in sources:
        J._receipt(path, receipts)
    result = dict(schema=1, comparison='joint-map-audit', completed=True,
                  historical_map_certified=False, physical_cabling_certified=False,
                  frozen_calibration=asset, current_local_map_comparisons=maps,
                  retained_configuration=config_result, frozen_data_inputs=report['data']['provenance']['inputs'],
                  evidence_boundary=[
                    'Candidate/alternate maps and source receipts are current local snapshots, not historical job-resolved bytes.',
                    'Matching rec and ntuple basename is a lineage clue, not an independently verified parentage certificate.',
                    'Retained configuration selects a moving geometry alias; effective release, include bytes and ordinal map are not recorded here.',
                    'Actual online FEB table payload and physical wiring validation are absent; projected map labels remain qualified.'],
                  required_for_historical_certification=[
                    'Ntuple-to-rec parentage with input checksum and production EventNtuple configuration/version.',
                    'Historical reconstruction release/build/patches and effective file search path.',
                    'Resolved geometry alias/include/ordinal map bytes or job-bound runtime ordinal dump.',
                    'Run125026 online boardreader/DTC/link/FEB table payload, not just a version label.'])
    for record in receipts.values():
        J.require(V.digest(record['path']) == record['sha256'], 'audit input/source changed')
    result['files'] = list(receipts.values())
    return V._report(out, result)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('timing', 'calib', 'candidate', 'alternate', 'config', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--source', action='append', default=[])
    a = p.parse_args(argv)
    run(a.timing, a.calib, a.candidate, a.alternate, a.config, a.source, a.out)
    print(f'joint-map-audit: completed; historical/physical certification withheld; {a.out}')


if __name__ == '__main__':
    main()
