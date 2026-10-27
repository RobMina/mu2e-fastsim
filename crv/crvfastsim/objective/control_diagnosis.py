"""Read-only comparison of frozen replicas and completed single-knob controls.

The reused diagnostic summary calls its candidate ``best``. Here that key means
exactly the point named by sample_role, NOT necessarily the five-knob full-best.
No ROOT inputs, response execution, calibration fit, or objective rescore occurs.
"""
import argparse
import json
from pathlib import Path
import subprocess

from . import diagnose as D, validation as V


CONTROLS = {'t1atten': {'atten_T1': 0.4367132104009329},
            'exspeed': {'speed_EX': 0.9840676938698677}}
POINT_FILES = ('r3', 'provenance', 'comparison')
FIXED_FLAGS = ('jax', 'noise', 'v05', 'batch_events', 'background', 'time_shift', 'device')
CAUTION = ('Single-knob contrasts are not a linear decomposition of the five-knob full-best. '
           'Interactions and response-dependent selection/random draws are not held fixed. '
           'No significance or paired-difference uncertainty is inferred.')


def _equal(a, b):
    """Order-independent JSON equality, without conflating booleans and numbers."""
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


class _Inputs:
    """Pin bytes before reading and check every pinned file again before output."""
    def __init__(self):
        self.hashes = {}

    def pin(self, path, expected=None):
        path = str(Path(path).resolve())
        sha = V.digest(path)
        if expected is not None and sha != expected:
            raise ValueError(f'input hash mismatch: {path}')
        if path in self.hashes and sha != self.hashes[path]:
            raise ValueError(f'input changed during diagnosis: {path}')
        self.hashes[path] = sha
        return dict(path=path, sha256=sha)

    def read(self, spec):
        receipt = self.pin(spec['path'], spec.get('sha256'))
        value = json.loads(Path(receipt['path']).read_text())
        self.pin(receipt['path'], receipt['sha256'])
        return value

    def verify(self):
        for path, sha in self.hashes.items():
            self.pin(path, sha)

    def receipts(self):
        return [dict(path=p, sha256=h) for p, h in sorted(self.hashes.items())]


def _point(files, inputs):
    if set(files) != set(POINT_FILES) or any(not row.get('sha256') for row in files.values()):
        raise ValueError('point requires hashed r3/provenance/comparison receipts')
    point = {key: inputs.read(files[key]) for key in POINT_FILES}
    counts = V._validate_point(point['r3'], point['comparison'], point['provenance'])
    D.validate_state(point['r3']['response_state'])
    return point, counts


def _fixed_result(nominal, candidate, label):
    for key in FIXED_FLAGS:
        if (key in nominal) != (key in candidate) or not _equal(nominal.get(key), candidate.get(key)):
            raise ValueError(f'{label}: fixed result flag {key} mismatch')
    if not _equal(nominal['response_state']['exposure_ns'], candidate['response_state']['exposure_ns']):
        raise ValueError(f'{label}: time exposure mismatch')


def validate_control(nominal, control, knobs, label):
    """Only the exact advertised knob may differ in the complete provenance."""
    pn, pc = nominal['provenance'], control['provenance']
    if nominal['r3']['knobs'] != {}:
        raise ValueError(f'{label}: original nominal must have empty knobs')
    if not _equal(pc['knobs'], knobs) or not _equal(control['r3']['knobs'], knobs):
        raise ValueError(f'{label}: wrong single-knob control')
    if not _equal({k: v for k, v in pn.items() if k != 'knobs'},
                  {k: v for k, v in pc.items() if k != 'knobs'}):
        raise ValueError(f'{label}: provenance differs from original nominal outside knobs')
    _fixed_result(nominal['r3'], control['r3'], label)
    if not _equal(nominal['comparison'].get('n0'), control['comparison'].get('n0')):
        raise ValueError(f'{label}: score normalization mismatch')


def _supplement(reference, saved, ref_provenance, inputs):
    if saved is None:
        return reference, None
    calibration = saved['calibration']
    fixed = ref_provenance['manifest']['assets']['time_shift']
    if not _equal(calibration, fixed):
        raise ValueError('timing supplement calibration differs from fixed calibration')
    config = inputs.read(calibration)
    files = saved['files']
    if not files or any(not row.get('sha256') for row in files):
        raise ValueError('timing supplement requires hashed file receipts')
    paths = [Path(row['path']).resolve() for row in files]
    if len(set(paths)) != len(paths) or len({p.parent for p in paths}) != 1:
        raise ValueError('timing supplement requires unique files in one directory')
    if [p.name for p in paths] != config['data_files']:
        raise ValueError('timing supplement file list differs from fixed calibration')
    for row in files:
        inputs.pin(row['path'], row['sha256'])
    updated, receipt = D.supplement_timing(reference, calibration['path'], paths[0].parent)
    for expected, actual in zip(files, receipt['files']):
        if expected['sha256'] != actual['sha256']:
            raise ValueError('timing supplement changed during diagnosis')
    inputs.verify()
    return updated, receipt


def code_identity():
    try:
        identity = D.code_identity()
    except (OSError, subprocess.CalledProcessError) as exc:
        # A copied checkout may lack its git metadata; never claim it is clean.
        identity = dict(path=str(Path(__file__).resolve().parents[2]), head=None,
                        dirty=True, git_identity_unavailable=str(exc))
    root = Path(__file__).resolve().parents[2]
    identity.setdefault('source_sha256', {}).update({
        str(Path(module.__file__).resolve().relative_to(root)): V.digest(module.__file__)
        for module in (D, V, D.R, D.bars)})
    identity['source_sha256'][str(Path(__file__).resolve().relative_to(root))] = V.digest(__file__)
    return identity


def _shift(samples, field='mean'):
    old = samples.get('nominal', {}).get(field)
    new = samples.get('best', {}).get(field)
    return None if old is None or new is None else V._finite(new - old, 'diagnostic shift')


def _summary(reference, nominal, candidate, name, role, threshold):
    ns, cs = nominal['comparison'], candidate['comparison']
    summary = D.summarize_block(dict(data=reference, nominal=nominal['r3']['response_state'],
                                    best=candidate['r3']['response_state']), threshold,
                                ns.get('n0', D.R.N0))
    summary['sample_role'] = dict(data='frozen_data_reference', nominal='original_nominal',
                                  best=f'{name}: {role}')
    summary['candidate_role'] = role
    summary['candidate_point'] = name

    def delta(n, c):
        return dict(nominal=n, candidate=c,
                    candidate_minus_nominal=V._finite(c - n, 'saved score difference'))

    summary['saved_score_deltas'] = dict(
        source='Saved comparison.json values; no objective rescore or combined uncertainty.',
        objective=delta(ns['objective'], cs['objective']),
        terms={k: delta(ns['terms'][k]['chi2ndf'], cs['terms'][k]['chi2ndf']) for k in sorted(ns['terms'])})
    summary['timing_mean_shifts_ns'] = {
        slot: dict(dt_cluster=_shift(summary['sectors'][slot]['scalar']['dt_cluster']['samples']),
                   module_timing={mode: _shift(summary['sectors'][slot]['module_timing'][mode])
                                  for mode in ('marginal', 'support_marginal', 'standardized')})
        for slot in ('T1', 'T2', 'TAG')}
    return summary


def diagnose(baseline_report, controls_a, controls_b, out, min_entries=100):
    """Read only saved inputs; create a new control_diagnosis.json after validation."""
    if type(min_entries) is not int or min_entries <= 0:
        raise ValueError('min_entries must be a positive integer')
    dest = Path(out)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    inputs = _Inputs()
    baseline_receipt = inputs.pin(baseline_report)
    baseline = inputs.read(baseline_receipt)
    if baseline.get('kind') != 'saved-response-diagnosis' or baseline.get('schema') != 1:
        raise ValueError('baseline must be a saved response diagnosis report')
    pairing = baseline['paired_receipt']
    names = {f'{b}_{a}' for b in ('a', 'b') for a in ('nominal', 'best')}
    if pairing.get('passed') is not True or set(pairing['inputs']) != names:
        raise ValueError('baseline requires a passing four-replica paired receipt')
    points, baseline_inputs = {}, {}
    for name in sorted(names):
        row = pairing['inputs'][name]
        points[name], counts = _point(row['files'], inputs)
        if not _equal(counts, row['counts']):
            raise ValueError(f'{name}: baseline receipt counts mismatch')
        baseline_inputs[name] = row
    refspec = baseline['reference']
    for name, point in points.items():
        if not _equal(point['provenance']['manifest']['assets']['reference'], refspec):
            raise ValueError(f'{name}: reference identity mismatch')
    if not refspec.get('sha256'):
        raise ValueError('missing fixed reference hash')
    reference = inputs.read(refspec)
    reference, supplement = _supplement(reference, baseline.get('timing_supplement'),
                                        points['a_nominal']['provenance'], inputs)
    D.validate_state(reference)

    controls, control_inputs, markers = {}, {}, {}
    for block, directory in (('a', controls_a), ('b', controls_b)):
        root = Path(directory)
        marker = inputs.pin(root / 'complete.txt')
        if Path(marker['path']).read_text().strip() != 'complete':
            raise ValueError(f'{block}: incomplete control marker (expected complete)')
        inputs.pin(marker['path'], marker['sha256'])
        markers[block] = marker
        nominal = points[f'{block}_nominal']
        for control, knobs in CONTROLS.items():
            name = f'{block}_{control}'
            files = {key: inputs.pin(root / name / f'{key}.json') for key in POINT_FILES}
            point, counts = _point(files, inputs)
            validate_control(nominal, point, knobs, name)
            controls[name] = point
            control_inputs[name] = dict(files=files, counts=counts)

    report = dict(schema=1, kind='saved-control-response-diagnosis', min_entries=min_entries,
                  limits=D.LIMITS + [CAUTION], diagnostic_code=code_identity(),
                  sample_role_note=('In every reused summary, best is the candidate named in sample_role; '
                                    'inside controls it means the single-knob control, NOT full-best.'),
                  baseline_report=baseline_receipt, paired_receipt=pairing,
                  reference=refspec, timing_supplement=supplement,
                  baseline_timing_supplement=baseline.get('timing_supplement'),
                  input_receipts=dict(baseline=baseline_inputs, controls=control_inputs, complete=markers),
                  blocks={})
    for block in ('a', 'b'):
        nominal, best = points[f'{block}_nominal'], points[f'{block}_best']
        _fixed_result(nominal['r3'], best['r3'], f'{block}_best')
        if not _equal(nominal['comparison'].get('n0'), best['comparison'].get('n0')):
            raise ValueError(f'{block}_best: score normalization mismatch')
        report['blocks'][block] = dict(
            full_best=_summary(reference, nominal, best, f'{block}_best', 'full-best (not a control)', min_entries),
            controls={control: _summary(reference, nominal, controls[f'{block}_{control}'],
                                        f'{block}_{control}', 'single-knob control (not full-best)', min_entries)
                      for control in CONTROLS})
    inputs.verify()
    report['input_receipts']['all_files'] = inputs.receipts()
    # Serialize before creating the destination, so malformed/nonfinite data cannot leave a report.
    payload = json.dumps(report, indent=2, allow_nan=False)
    inputs.verify()
    dest.mkdir(parents=True, exist_ok=False)
    with (dest / 'control_diagnosis.json').open('x') as stream:
        stream.write(payload + '\n')
    return report


def print_summary(report):
    def fmt(value):
        return 'n/a' if value is None else f'{value:.6g}'

    for block, row in report['blocks'].items():
        for item in [row['full_best'], *row['controls'].values()]:
            score = item['saved_score_deltas']
            total = score['objective']
            print(f"{item['candidate_point']} [{item['candidate_role']}]: saved total "
                  f"{fmt(total['nominal'])} -> {fmt(total['candidate'])}; "
                  f"delta={fmt(total['candidate_minus_nominal'])}")
            for term in ('pe_layer', 'tspread', 'dt_cluster'):
                delta = score['terms'][f'{term}_T1']['candidate_minus_nominal']
                samples = item['sectors']['T1']['scalar'][term]['samples']
                print(f"  T1 {term}: saved term delta={fmt(delta)}; data/nominal/candidate q50="
                      + '/'.join(fmt(samples[k]['q50']) for k in ('data', 'nominal', 'best')))
            for slot, timing in item['timing_mean_shifts_ns'].items():
                module = timing['module_timing']
                print(f"  {slot} mean shift ns: dt_cluster={fmt(timing['dt_cluster'])}; "
                      'module marginal/support/standardized='
                      + '/'.join(fmt(module[k]) for k in ('marginal', 'support_marginal', 'standardized')))
    print(CAUTION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-report', required=True)
    parser.add_argument('--controls-a', required=True, help='contains a_t1atten, a_exspeed and complete.txt')
    parser.add_argument('--controls-b', required=True, help='contains b_t1atten, b_exspeed and complete.txt')
    parser.add_argument('--out', required=True, help='new directory; never overwrite')
    parser.add_argument('--min-entries', '--min-support', type=int, default=100,
                        help='common-support entries required separately in every sample/cell (default 100)')
    args = parser.parse_args()
    try:
        report = diagnose(args.baseline_report, args.controls_a, args.controls_b, args.out, args.min_entries)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print_summary(report)
    print(f'wrote {args.out}/control_diagnosis.json')


if __name__ == '__main__':
    main()
