"""Generated-event counts cannot silently become generated-electron denominators."""
import numpy as np
import pytest

from beamfastsim import target_yield_compare as T


def test_multi_electron_reference_fails_closed(tmp_path, monkeypatch):
    ref = tmp_path / "ref.npz"
    np.savez(ref, creation_code=166, ifile=np.array([0, 0]), gencount=np.array([100]),
             subrun=np.array([1, 1]), evt=np.array([1, 1]))
    monkeypatch.setattr(T.CC, "g4_plane", lambda *a: {})
    monkeypatch.setattr(T.CC, "fast_plane", lambda *a: {})
    with pytest.raises(ValueError, match="not an electron denominator"):
        T.compare(str(ref), "unused", None, None)


def test_exposure_requires_exact_windows_even_without_kept_multiplicity():
    from beamfastsim.target_yield_compare import denominators
    ref = dict(ifile=np.array([0, 1]), subrun=np.array([0, 1]), evt=np.array([1, 1]), gencount=np.array([5000, 5000]))
    with pytest.raises(ValueError, match='exact parent-window'):
        denominators(ref)
    report = dict(production_exposure=dict(exact_production_electron_exposure_known=True,
        files=[dict(ifile=0, generated_electrons=5012, generated_events=5000, matched_electrons=1),
               dict(ifile=1, generated_electrons=5008, generated_events=5000, matched_electrons=1)]))
    assert denominators(ref, report).tolist() == [5012, 5008]
    report['production_exposure']['files'][1]['matched_electrons'] = 2
    with pytest.raises(ValueError, match='does not match'):
        denominators(ref, report)


def test_exposure_ratio_respects_clustered_counts_and_unequal_exposures():
    a = T.exposure_ratio(np.array([10, 20]), np.array([100, 200]), 30, 300)
    assert a['ratio'] == 1 and a['g4_clustered_err'] == 0
    b = T.exposure_ratio(np.array([10, 30]), np.array([100, 100]), 40, 200)
    assert b['g4_clustered_err'] == pytest.approx(0.1)
    assert b['ratio_err'] == pytest.approx(np.hypot(np.sqrt(0.2 * 0.8 / 200) / 0.2, 0.1 / 0.2))


def test_rpc_requires_matching_parent_audit_not_bare_generated_counts(tmp_path, monkeypatch):
    ref = tmp_path / 'rpc.npz'
    np.savez(ref, creation_code=179, ifile=np.array([0]), gencount=np.array([100]),
             subrun=np.array([0]), evt=np.array([1]))
    monkeypatch.setattr(T.CC, 'g4_plane', lambda *a: {})
    monkeypatch.setattr(T.CC, 'fast_plane', lambda *a: {})
    with pytest.raises(ValueError, match='exact one-pion'):
        T.compare(str(ref), 'unused', None, None)
    report = dict(exact_production_electron_exposure_known=True,
        files=[dict(ifile=0, generated_electrons=100, generated_events=100, matched_electrons=1)])
    with np.load(ref) as r:
        assert T.denominators(r, report).tolist() == [100]
