"""Audit denominator support and propagate block resampling through native reach acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import tables
from . import denominator_variants as D
from . import library_variants as V


def propagate(counts, events, showers, generated, replicas, seed):
    """Vectorized file bootstrap of the existing RNG acceptance, never a library builder."""
    counts, events, showers = (np.asarray(x, dtype=float) for x in (counts, events, showers))
    if (counts.ndim != 2 or events.shape != (counts.shape[0],) or showers.shape != (counts.shape[1],)
            or counts.shape[0] < 2 or replicas < 2 or seed < 0 or generated <= 0
            or not all(np.isfinite(a).all() for a in (counts, events, showers))
            or np.any(counts < 0) or np.any(events <= 0) or np.any(showers < 0)):
        raise ValueError('invalid per-file count/exposure arrays')
    nominal = np.rint(counts.sum(axis=0) * generated / events.sum())
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(events), np.full(len(events), 1 / len(events)), size=replicas)
    denominators = np.rint((weights @ counts) * generated / (weights @ events)[:, None])
    raw = np.divide(showers, denominators, out=np.zeros_like(denominators), where=denominators > 0)
    nominal_raw = np.divide(showers, nominal, out=np.zeros_like(showers), where=nominal > 0)
    unsupported = (showers > 0) & (nominal == 0)
    saturated = nominal_raw > 1
    invalid = ((showers[None] > 0) & (denominators == 0)) | (raw > 1)
    effective = np.minimum(raw, 1)
    covariance = np.cov(effective, rowvar=False, ddof=1)
    return dict(replicas=replicas, seed=seed, nominal_scaled_denominator=nominal.tolist(),
                nominal_showers=showers.tolist(), nominal_raw_ratio=nominal_raw.tolist(),
                nominal_effective_acceptance=np.minimum(nominal_raw, 1).tolist(),
                nominal_missing_support=unsupported.tolist(), nominal_saturated=saturated.tolist(),
                any_invalid_replicas=int(invalid.any(axis=1).sum()),
                invalid_replicas_per_bin=invalid.sum(axis=0).tolist(),
                zero_denominator_replicas_per_bin=(denominators == 0).sum(axis=0).tolist(),
                saturation_replicas_per_bin=(raw > 1).sum(axis=0).tolist(),
                effective_acceptance_mean=effective.mean(axis=0).tolist(),
                effective_acceptance_sd=effective.std(axis=0, ddof=1).tolist(),
                effective_acceptance_covariance=np.atleast_2d(covariance).tolist(),
                effective_acceptance_quantile_levels=[0.025, 0.5, 0.975],
                effective_acceptance_quantiles=np.quantile(effective, [0.025, 0.5, 0.975], axis=0).tolist())


def run(library_json, out, replicas=1000, seed=20261025):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    metadata, plan, base = D._base(library_json)
    code = D._code_identity()
    code[str(Path(__file__).resolve())] = V.sha256(__file__)
    sources = [dict(r, bin_counts=D._counts(r)) for r in plan['gen']]
    showers = tables.read(base['root']['path'], 'libshower').groupby(D.BIN).size().rename('nshower')
    existing = tables.read(base['root']['path'], 'libgen').set_index(D.BIN)
    keys = existing.index.union(showers.index).sort_values()
    blocks = []
    for record in sources:
        frame = pd.DataFrame(record['bin_counts'], columns=D.BIN + ['count']).set_index(D.BIN)
        blocks.append(frame['count'].reindex(keys, fill_value=0).to_numpy())
    counts = np.stack(blocks)
    events = np.array([r['events']['unique_events'] for r in sources])
    ns = showers.reindex(keys, fill_value=0).to_numpy()
    result = propagate(counts, events, ns, metadata['generated_scaled_to'], replicas, seed)
    nominal = np.array(result['nominal_scaled_denominator'])
    old = existing.reindex(keys)
    present = old.ngen.notna().to_numpy()
    if not np.array_equal(nominal[present], old.ngen.to_numpy()[present]):
        raise ValueError('nominal generator counts do not reproduce existing libgen')
    if not np.array_equal(ns[present], old.nshower.to_numpy()[present]):
        raise ValueError('nominal shower counts do not reproduce existing libgen')
    result.update(schema='valhist.denominator_audit.v1', base=base, source_files=sources,
                  bins=[dict(apdg=int(a), kebin=int(b)) for a, b in keys],
                  nominal_libgen_missing_bins=(~present).tolist(),
                  code_sha256=code, generated_exposure=metadata['generated_scaled_to'], limitations=[
                      'Parameter-level denominator propagation only; NO downstream DQM/rate covariance or library written.',
                      'One generator-file multiplicity vector applies globally to every species/energy bin.',
                      'Effective acceptance reproduces native uniform<ratio semantics: ratio>1 always accepts, absent/zero denominator rejects.',
                      'Saturation is observed native behavior, not a new clipping repair; unsupported cells remain flagged.',
                      'Whole-file bootstrap assumes ten independent exchangeable generator blocks; conditional on fixed donor realization.',
                      'No conditional draws are discarded; all bootstrap support failures and native boundary behavior remain included.',
                      'This diagnostic does not authorize strict denominator-variant library builds that fail support validation.'])
    D._base(library_json)
    if D._code_identity() != {k: v for k, v in code.items() if k != str(Path(__file__).resolve())}:
        raise ValueError('source drift')
    if V.sha256(__file__) != code[str(Path(__file__).resolve())]:
        raise ValueError('audit driver drift')
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('nominal missing', sum(result['nominal_missing_support']), 'saturated', sum(result['nominal_saturated']),
          'invalid draws', result['any_invalid_replicas'], '/', replicas)
    for j, key in enumerate(result['bins']):
        print(key, 'ngen', nominal[j], 'nshower', ns[j], 'raw', result['nominal_raw_ratio'][j],
              'effectiveSD', result['effective_acceptance_sd'][j],
              'invalid', result['invalid_replicas_per_bin'][j])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('library_json')
    p.add_argument('--out', required=True)
    p.add_argument('--replicas', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261025)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
