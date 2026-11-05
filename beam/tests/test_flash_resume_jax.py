"""Exact restart contracts only; not a physics validation of this birth diagnostic."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("jax.numpy")
from mu2efastsim import eslab_jax as SJ
from beamfastsim import cechain as C
from beamfastsim import flash_target_jax as F
from beamfastsim import geometry as G

pytestmark = pytest.mark.skipif(not hasattr(SJ, "lane_step_emissions"), reason="full emission interface required")


def _geom():
    b = G.BeamGeometry(0., np.array([5., 12.]), np.zeros(2), np.full(2, 75.), np.full(2, 2.),
                       -50., 1., 100., 200., 300., .5, 100., 200., 900., 100., 200.)
    return C.CEGeometry(b, 900., 900., 1., z_plane=20., z_up=-20., z_ds=-30.,
                        coll5_z0=-29., coll5_z1=-28.)


GEOM = _geom()


def _args(dtype=np.float64, **changes):
    args = dict(key=jax.random.PRNGKey(73), geom=GEOM, field=[0., 0., 1.],
                x=np.array([[1., 0., 5.], [2., 0., 0.], [3., 0., 5.], [4., 0., 10.],
                            [5., 0., 10.], [6., 0., 5.]]),
                u=np.tile([.1, 0., 1.], (6, 1)), p=[100., 10., 1., .0001, .0001, 100.],
                t=np.array([1e12 + .125, 2e12, 3e12, 4e12, 5e12, 6e12]),
                pdg=[11, -11, -11, -11, 11, 11], capacity=32, msc=True, dtype=dtype)
    args.update(changes)
    return args


def _flat(value, prefix=""):
    out = {}
    for name, child in value.items():
        path = prefix + name
        if isinstance(child, dict):
            out.update(_flat(child, path + "/"))
        else:
            out[path] = np.asarray(child)
    return out


def _exact(a, b):
    a, b = _flat(a), _flat(b)
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].dtype == b[name].dtype, name
        assert a[name].shape == b[name].shape, name
        assert a[name].tobytes() == b[name].tobytes(), name


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixed_fixed_batches_exact_chunks_checkpoint_and_legacy(dtype, tmp_path):
    args = _args(dtype)
    initial = F.initialize_source(**args)
    original = F.export_state(initial)
    first = F.advance_source(initial, 1)
    assert set(first["species"]) == {11, -11}
    assert np.asarray(first["species"][11]["state"]["inl"]).any()
    # Partial slab projection must never become entry state for continuation.
    before = F.export_state(first)
    partial = F.source_result(first)
    F.source_result(first)
    _exact(before, F.export_state(first))
    raw = first["species"][11]["state"]
    ids = first["species"][11]["ids"]
    assert np.any(np.asarray(partial["state"]["s"])[ids] > np.asarray(raw["s"]))
    whole = F.advance_source(initial, 201)
    chunked = F.advance_source(F.advance_source(first, 7), 193)
    _exact(F.source_result(whole), F.source_result(chunked))
    _exact(original, F.export_state(initial))  # no mutation of input session
    checkpoint = tmp_path / "checkpoint.npz"
    np.savez_compressed(checkpoint, **before)
    with np.load(checkpoint, allow_pickle=False) as arrays:
        restored = F.import_state(F.initialize_source(**args), arrays)
    _exact(before, F.export_state(restored))
    _exact(F.source_result(whole), F.source_result(F.advance_source(restored, 200)))
    _exact(F.source_result(whole), F.target_source(**args, max_steps=201))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_initial_rest_never_repeated_even_with_zero_work_restore(dtype):
    args = _args(dtype, x=[[1., 0., 5.], [2., 0., 10.]], u=[[0., 0., 1.]] * 2,
                 p=.0001, pdg=-11, t=[1e12, 2e12], capacity=2)
    initial = F.initialize_source(**args)
    result = F.source_result(initial)
    assert np.array_equal(result["count"], [2, 2])
    assert not np.asarray(initial["total_iterations"]).any()
    restored = F.import_state(F.initialize_source(**args), F.export_state(initial))
    for _ in range(3):
        restored = F.advance_source(restored, 100001)
    _exact(result, F.source_result(restored))
    assert np.array_equal(F.source_result(restored)["nann"], [2, 2])


def test_vacuum_continues_past_100k_and_path_loopers_remain_fail_closed():
    args = _args(np.float64, x=[[1., 0., 8.]], u=[[1., 0., 0.]],
                 p=.1, pdg=11, t=0., s_max=1800., capacity=2, msc=False)
    start = F.initialize_source(**args)
    first = F.advance_source(start, 100000)
    r = F.source_result(first)
    assert r["capped"][0] and int(r["state"]["nadapt"][0]) == 100000
    resumed = F.import_state(F.initialize_source(**args), F.export_state(first))
    final = F.advance_source(resumed, 10000)
    assert float(F.source_result(final)["state"]["s"][0]) > float(r["state"]["s"][0])
    _exact(F.source_result(final), F.source_result(F.advance_source(start, 110000)))
    assert bool(F.source_result(final)["loopers"][0])
    with pytest.raises(ValueError, match="loopers=1"):
        F.validate_complete(F.source_result(final))


@pytest.mark.parametrize("change", ["missing", "extra", "dtype", "shape", "nan", "ids", "key", "pdg",
                                   "negative", "overflow", "rng", "valid", "count", "lty", "fate"])
def test_malformed_checkpoints_rejected(change):
    initial = F.initialize_source(**_args())
    checkpoint = F.export_state(F.advance_source(initial, 3))
    state = "species/11/state/"
    if change == "missing":
        del checkpoint[state + "slab/pend"]
    elif change == "extra":
        checkpoint["unknown"] = np.array(1)
    elif change == "dtype":
        checkpoint[state + "x"] = checkpoint[state + "x"].astype(np.float32)
    elif change == "shape":
        checkpoint[state + "x"] = checkpoint[state + "x"][:1]
    elif change == "nan":
        checkpoint[state + "slab/x"][:] = np.nan
    elif change in ("ids", "key"):
        checkpoint["species/11/" + change][0] += 1
    elif change == "pdg":
        checkpoint["pdg"][:] = -11
    elif change == "negative":
        checkpoint[state + "nvac"][:] = -1
    elif change == "overflow":
        checkpoint["total_iterations"][:] = np.iinfo(np.int32).max
    elif change == "rng":
        checkpoint[state + "slab/it"][...] += 1
    elif change == "valid":
        checkpoint[state + "records/valid"][:] = True
    elif change == "count":
        checkpoint[state + "count"][:] = np.iinfo(np.int32).max
    elif change == "lty":
        checkpoint[state + "lty"][:] = 100000
    elif change == "fate":
        checkpoint[state + "fate"][:] = 10
    with pytest.raises(ValueError):
        F.import_state(initial, checkpoint)


def test_limits_absent_species_and_import_storage_are_safe():
    initial = F.initialize_source(**_args())
    for bad in (True, np.bool_(True), 0, 1.5, F._ITERATION_LIMIT + 1):
        with pytest.raises(ValueError):
            F.advance_source(initial, bad)
    later = F.advance_source(initial, 3)
    with pytest.raises(ValueError, match="freshly"):
        F.import_state(later, F.export_state(later))
    checkpoint = F.export_state(later)
    restored = F.import_state(initial, checkpoint)
    before = F.export_state(restored)
    checkpoint["species/11/state/x"][:] = 777.
    _exact(before, F.export_state(restored))
    fake = dict(initial, total_iterations=np.array([F._ITERATION_LIMIT, 0], np.int32))
    with pytest.raises(ValueError):
        F.advance_source(fake, 1)
    single = F.initialize_source(**_args(pdg=11))
    checkpoint = F.export_state(single)
    checkpoint["total_iterations"][1] = 1
    with pytest.raises(ValueError):
        F.import_state(single, checkpoint)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_map_runtime_checkpoint_has_no_object_payload_or_field_replay(dtype, monkeypatch):
    from beamfastsim import fieldmap as FM
    fm = FM.FieldMap(np.full(3, -100.), np.full(3, 200.), np.full(3, 2),
                     np.broadcast_to([0., 0., 1.], (2, 2, 2, 3)))
    args = _args(dtype, field=(fm,))
    initial = F.initialize_source(**args)
    first = F.advance_source(initial, 2)
    arrays = F.export_state(first)
    assert all(isinstance(a, np.ndarray) and a.dtype.kind in "biuf" for a in arrays.values())
    assert "species/11/state/slab/pend" in arrays
    assert "species/-11/state/e1" in arrays
    template = F.initialize_source(**args)
    expected = F.source_result(F.advance_source(initial, 201))
    def forbidden(*args, **kwargs):
        raise AssertionError("import must not initialize, rebuild or replay maps")
    monkeypatch.setattr(F, "initialize_source", forbidden)
    monkeypatch.setattr(F.CJ, "_build", forbidden)
    restored = F.import_state(template, arrays)
    _exact(expected, F.source_result(F.advance_source(restored, 199)))


def test_overflow_remains_visible_across_restore():
    args = _args(capacity=1)
    initial = F.initialize_source(**args)
    first = F.advance_source(initial, 5)
    restored = F.import_state(F.initialize_source(**args), F.export_state(first))
    resumed = F.advance_source(restored, 196)
    _exact(F.source_result(resumed), F.source_result(F.advance_source(initial, 201)))
    assert np.asarray(F.source_result(resumed)["overflow"]).sum() > 0
    with pytest.raises(ValueError, match="dropped births"):
        F.validate_complete(F.source_result(resumed))


@pytest.mark.parametrize("change", ["schema_version", "capacity", "species_codes", "parent_index", "t0",
                                   "negative_time", "unused_record", "nonfinite", "object", "last_gt_total"])
def test_additional_fail_closed_invariants(change):
    initial = F.initialize_source(**_args())
    arrays = F.export_state(F.advance_source(initial, 1))
    if change in ("schema_version", "capacity"):
        arrays[change][...] += 1
    elif change in ("species_codes", "parent_index", "t0"):
        arrays[change][0] += 1
    elif change == "negative_time":
        arrays["species/11/state/t"][:] = -1.
    elif change == "unused_record":
        arrays["species/11/state/records/energy"][:, -1] = 3.
    elif change == "nonfinite":
        arrays["species/11/state/slab/pend"][:] = np.inf
    elif change == "object":
        arrays["species/11/state/x"] = arrays["species/11/state/x"].astype(object)
    elif change == "last_gt_total":
        arrays["iterations"][:] = 2
    with pytest.raises(ValueError):
        F.import_state(initial, arrays)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_chunk_matches_independent_prerefactor_source(dtype):
    path = os.environ.get("FLASH_TARGET_BASELINE_SOURCE")
    if not path or not Path(path).is_file():
        pytest.skip("Provide an independent pre-refactor target source for byte regression")
    spec = importlib.util.spec_from_file_location("beamfastsim._flash_target_baseline", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args = _args(dtype)
    _exact(old.target_source(**args, max_steps=201), F.target_source(**args, max_steps=201))
