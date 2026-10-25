"""Validate general-chain pulse dispatch on frozen stock digis; never modify reference products."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
import uproot

from crvfastsim.reco import chain_jax, cli, compare, pulsefit
from dataclasses import replace


def run(path, outdir, max_events=None, require_gpu=False, batch=4096):
    if require_gpu and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU validation requested, but JAX is not using a GPU')
    cfg = pulsefit.RecoConfig()
    W = cli._prepare([path], max_events, cfg)
    W.update(adc=W['mat'], bar=W['chan'] // 4, sipm=W['chan'] % 4)
    ref = pulsefit.reconstruct_full(*(W[k] for k in ('adc', 'n', 'start', 'ped', 'area', 'height')), cfg)
    rows = np.arange(len(W['n']))
    pending = chain_jax._pulse_dispatch(W, rows, replace(cfg, max_pulses=3), batch, 12)
    wf, seq, pe, tm, nredo = chain_jax._pulse_collect(pending, W, rows, cfg)
    identity = np.array_equal(wf, ref['wf']) and np.array_equal(seq, ref['seq'])
    parity = dict(pulse_identity=identity, numpy_pulses=len(ref['wf']), chain_pulses=len(wf),
                  numpy_redo_waveforms=nredo, long_waveforms=int((W['n'] > 12).sum()),
                  max_samples=int(np.max(W['n'], initial=0)),
                  max_sequence=int(np.max(ref['seq'], initial=-1)))
    if identity:
        parity['max_abs_time_ns'] = float(np.max(np.abs(tm - ref['time']), initial=0))
        parity['max_abs_pe'] = float(np.max(np.abs(pe - ref['pe']), initial=0))
        parity['pass'] = bool(np.allclose(pe, ref['pe'], rtol=1e-6, atol=1e-6) and
                              np.allclose(tm, ref['time'], rtol=0, atol=1e-4))
    else:
        parity['pass'] = False
    fitted, _, cli_redo, _ = cli._run_jax(W, cfg, L=12, batch=batch, rounds=3)
    fast = dict(key=compare.pulse_key(W['ev'][fitted['wf']], W['digi0'][fitted['wf']], fitted['seq']),
                pe=fitted['pe'], time=cli._with_offsets(fitted, W, cfg), failed=fitted['failed'])
    near = np.stack([W['ev'], W['digi0']], 1)[fitted['wf_near_threshold']]
    stock = compare.vr2(fast, cli._pulses([path], W['events']), near)
    with uproot.open(path) as f:
        trees = {k: dict(entries=int(f[k].num_entries), classname=f[k].classname)
                 for k in ('crvdump/events', 'crvdump/digis', 'crvdump/pulses')}
    with open(path, 'rb') as src:
        digest = hashlib.file_digest(src, 'sha256').hexdigest()
    root = Path(__file__).resolve().parents[1]
    provenance = dict(input=str(Path(path).resolve()), input_sha256=digest, trees=trees,
                      crv_head=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                      crv_status=subprocess.check_output(['git', '-C', str(root), 'status', '--short'], text=True),
                      jax_version=jax.__version__, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()])
    result = dict(provenance=provenance, selected_events=len(W['events']), waveforms=len(W['n']),
                  general_vs_full_numpy=parity, cli_numpy_redo_waveforms=cli_redo, stock_vr2=stock)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'reco_contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not parity['pass'] or not stock['pass']:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input')
    p.add_argument('--outdir', required=True)
    p.add_argument('--max-events', type=int)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--require-gpu', action='store_true')
    a = p.parse_args()
    run(a.input, a.outdir, a.max_events, a.require_gpu, a.batch)


if __name__ == '__main__':
    main()
