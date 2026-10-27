"""Strict corrected-denominator gates preserve every draw and bind retained inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot

from valhist import corrected_denominator as C
from valhist.library_variants import sha256


def test_repair_removes_nominal_hole_but_sparse_bank_still_blocks():
    counts = np.array([[10, 1, 0], [10, 0, 0]])
    events = np.array([10, 10])
    before = C.support_report(counts, events, np.array([1, 1, 1]), 200, 1000, 7)
    after = C.support_report(counts, events, np.array([2, 1, 0]), 200, 1000, 7)
    assert before['any_invalid_replicas'] == 1000
    assert not before['nominal_support_pass']
    assert after['nominal_support_pass']
    assert 0 < after['any_invalid_replicas'] < 1000
    assert not after['ensemble_support_pass']
    assert not after['downstream_propagation_authorized']
    assert after['status'] == 'BLOCKED_SUPPORT'
    assert after['exact_zero_support_probability'] == [0, .25, 1]
    assert after['effective_acceptance_mean'][1] < after['nominal_raw_ratio'][1]


def test_shared_multiplicity_keeps_cross_bin_covariance():
    result = C.support_report(np.array([[1, 9], [9, 1]]), np.array([10, 10]),
                              np.array([1, 1]), 200, 1000, 9)
    cov = np.asarray(result['effective_acceptance_covariance'])
    assert cov[0, 1] < 0
    assert result['ensemble_support_pass']
    assert not result['downstream_propagation_authorized']
    assert result['status'] == 'SUPPORT_PASS_ONLY'


def test_rounding_and_unequal_exposure_are_native_ratio_of_totals():
    args = (np.array([[1], [8]]), np.array([2, 8]), np.array([2]), 100, 200, 1)
    result = C.support_report(*args)
    assert result['nominal_scaled_denominator'] == [90]
    assert result == C.support_report(*args)


def test_saturation_is_a_failure_not_a_clipping_repair():
    result = C.support_report(np.array([[1], [0]]), np.array([10, 10]), np.array([3]), 20, 100, 3)
    assert result['nominal_saturated'] == [True]
    assert result['any_invalid_replicas'] == 100
    assert result['status'] == 'BLOCKED_SUPPORT'


@pytest.mark.parametrize('replicas,seed,generated', [(1, 0, 20), (2, True, 20), (2, -1, 20), (2, 0, 0)])
def test_bad_scalar_options(replicas, seed, generated):
    with pytest.raises(ValueError):
        C.support_report(np.array([[1], [2]]), np.array([10, 10]), np.array([1]), generated, replicas, seed)


@pytest.mark.parametrize('counts', [np.array([[1.5], [2.]]), np.array([[-1], [2]]), np.array([[True], [False]])])
def test_counts_must_be_nonnegative_integers(counts):
    with pytest.raises(ValueError):
        C.support_report(counts, np.array([10, 10]), np.array([1]), 20, 10, 1)


def test_exclusive_output_refusal_before_input_read(tmp_path):
    out = tmp_path / 'report.json'
    out.write_text('preserved')
    with pytest.raises(FileExistsError):
        C.run('missing', 'missing', out)
    assert out.read_text() == 'preserved'
    link = tmp_path / 'dangling'
    link.symlink_to(tmp_path / 'absent')
    with pytest.raises(FileExistsError):
        C.run('missing', 'missing', link)


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    libdir = root / 'library'
    libdir.mkdir(parents=True)
    cache = tmp_path / 'cache'
    cache.mkdir()
    cells = dict(apdg=np.array([22, 2112]), kebin=np.array([5, 12]))
    reach = dict(**cells, ngen=np.array([200, 10]), nshower=np.array([2, 1]))
    sh = dict(shid=np.arange(3), apdg=np.array([22, 22, 2112]), kebin=np.array([5, 5, 12]))
    lib = libdir / 'library.root'
    with uproot.recreate(lib) as f:
        f.mktree('dump/libgen', reach)
        f.mktree('dump/libshower', sh)
    counts = np.array([[10, 1], [10, 0]])
    events = np.array([10, 10])
    np.savez(cache / 'index.npz', **cells, cell=np.array([0, 0, 1]), gen_counts=counts,
             gen_events=events, donor_events=np.array([[1400, 99, 1]]))
    donors = [dict(path='donor', validation=dict(schema_version=2, associations_checked=True, ancestry_cycles=0))]
    lm = dict(generated_scaled_to=200, gen=['g1', 'g2'], inputs=['donor'])
    (libdir / 'library.json').write_text(json.dumps(lm))
    meta = dict(campaign=str(root), library=str(lib), generated=200, donors=donors, gen=['g1', 'g2'],
                input_sha256={str(lib): sha256(lib)}, output_sha256={str(cache / 'index.npz'): sha256(cache / 'index.npz')})
    (cache / 'cache.json').write_text(json.dumps(meta))
    monkeypatch.setattr(C, 'verify_library', lambda _: None)
    monkeypatch.setattr(C.U, 'code_identity', lambda: {})
    ids = C.U.event_index(pd.DataFrame([[1400, 1, 1], [1400, 2, 1]], columns=C.U.KEY))
    monkeypatch.setattr(C, 'certify_generators', lambda *a: (counts, events, ids, []))
    return cache, lib, counts


def test_full_report_binds_root_and_refuses_drift(tmp_path, monkeypatch):
    cache, lib, _ = fixture(tmp_path, monkeypatch)
    report = C.run(cache, 'mock-certificate', tmp_path / 'out.json', replicas=100, seed=3, legacy_library=lib)
    assert report['generator_events'] == 20
    assert report['corrected']['nominal_support_pass']
    assert report['legacy']['any_invalid_replicas'] == report['corrected']['any_invalid_replicas']
    assert report['library_sha256'] == sha256(lib)
    lib.write_bytes(b'changed')
    with pytest.raises(Exception):
        C.run(cache, 'mock-certificate', tmp_path / 'drift.json', replicas=100, seed=3)
    assert not (tmp_path / 'drift.json').exists()


def test_cached_denominator_mismatch_refused(tmp_path, monkeypatch):
    cache, _, counts = fixture(tmp_path, monkeypatch)
    counts[0, 0] += 1
    with pytest.raises(ValueError, match='fresh denominator census'):
        C.run(cache, 'mock-certificate', tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()


def test_real_full_generator_certificate_counts_empty_events(tmp_path, monkeypatch):
    from test_genassoc_fullgen import rootfile
    from valhist import genassoc_fullgen as F
    monkeypatch.setattr(F, 'GENERATED_PER_FILE', 2)
    cert = dict(schema='valhist.genassoc_fullgen.v1', cross_file_event_overlap=0, files={},
                input_sha256_pre={}, input_sha256_post={}, code_sha256_pre={}, code_sha256_post={}, evidence_sha256={})
    paths = []
    for shard in (1, 2):
        old, new = tmp_path / f'old{shard}.root', tmp_path / f'new{shard}.root'
        rootfile(old, shard=shard)
        rootfile(new, shard=shard, strict=True)
        report, _ = F.compare_payload(old, new)
        cert['files'][str(old)] = dict(v2path=str(new), digest=sha256(new), **report)
        cert['input_sha256_pre'][str(old)] = cert['input_sha256_post'][str(old)] = sha256(old)
        cert['evidence_sha256'][str(new)] = sha256(new)
        paths.append(str(old))
    path = tmp_path / 'cert.json'
    path.write_text(json.dumps(cert))
    cells = pd.DataFrame(dict(apdg=[22], kebin=[5]))
    hashes = {}
    counts, events, ids, reports = C.certify_generators(paths, path, cells, hashes)
    np.testing.assert_array_equal(counts, [[0], [0]])
    np.testing.assert_array_equal(events, [2, 2])
    assert len(ids) == 4 and len(reports) == 2
    assert reports[0]['zero_generator_events'] == 1
    assert hashes[str(path)] == sha256(path)
    cert['files'][paths[0]]['event_count'] = 3
    path.write_text(json.dumps(cert))
    with pytest.raises(ValueError, match='census differs'):
        C.certify_generators(paths, path, cells, {})
    cert['files'][paths[0]]['event_count'] = 2
    cert['input_sha256_post'][paths[0]] = 'bad'
    path.write_text(json.dumps(cert))
    with pytest.raises(ValueError, match='certificate'):
        C.certify_generators(paths, path, cells, {})
