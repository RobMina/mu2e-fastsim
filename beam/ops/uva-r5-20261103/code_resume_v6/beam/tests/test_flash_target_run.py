"""The pilot runner verifies the exact certified input before transport."""
import argparse
import json

import numpy as np
import pytest

pytest.importorskip("jax")
from beamfastsim import flash_target_run as R


def test_uncertified_or_changed_source_rejected_before_geometry(tmp_path):
    source = tmp_path / "states.npz"
    np.savez(source, pdg=np.array([11]))
    cert = tmp_path / "certificate.json"
    args = argparse.Namespace(certificate=cert, states=source)
    cert.write_text(json.dumps(dict(certified=False)))
    with pytest.raises(ValueError, match="certificate"):
        R.run(args)
    cert.write_text(json.dumps(dict(schema="flash-step-input-v1", certified=True,
                                  artifacts=dict(states=dict(sha256="mismatch")))))
    with pytest.raises(ValueError, match="artifact"):
        R.run(args)


def test_source_schema_rejected_before_transport(tmp_path):
    source = tmp_path / "states.npz"
    np.savez(source, pdg=np.array([11]))
    cert = tmp_path / "certificate.json"
    cert.write_text(json.dumps(dict(schema="flash-step-input-v1", certified=True,
                                  artifacts=dict(states=dict(sha256=R._sha256(source))))))
    with pytest.raises(ValueError, match="schema"):
        R.run(argparse.Namespace(certificate=cert, states=source))


def _runner_fixture(tmp_path, monkeypatch):
    from beamfastsim import cechain as C, cechain_jax as CJ, simpleconfig
    from test_flash_target_jax import _geom
    source = tmp_path / "source.npz"
    np.savez(source, run=np.array([1, 1]), subrun=np.array([2, 2]), event=np.array([3, 4]),
             particle_key=np.array([5, 6]), pdg=np.array([11, -11]),
             x=np.array([[1., 0., 5.], [2., 0., 10.]]),
             p=np.array([[0., 0., 100.], [0., 0., 100.]]), time_ns=np.array([1e12 + .125, 4.]))
    cert = tmp_path / "certificate.json"
    cert.write_text(json.dumps(dict(schema="flash-step-input-v1", certified=True, rows=2,
                                  artifacts=dict(states=dict(sha256=R._sha256(source))))))
    overlay = tmp_path / "overlay.txt"
    overlay.write_text("synthetic geometry identity\n")
    header = tmp_path / "map.header"
    header.write_text("synthetic map identity\n")
    header.with_suffix(".bin").write_bytes(b"synthetic map bytes")
    monkeypatch.setattr(C, "from_offline", lambda *args: _geom(half=2.))
    monkeypatch.setattr(CJ, "device_maps", lambda *args, **kwargs: [0., 0., 1.])

    def read(*args, source_files=None, **kwargs):
        source_files.append(str(overlay))
    monkeypatch.setattr(simpleconfig, "read", read)
    return argparse.Namespace(states=source, certificate=cert, parents=None, geom=overlay,
                              offline_root=tmp_path, maps=[header], seed=1, dtype="float64", capacity=16,
                              max_steps=1, h=1., s_max=100., out=tmp_path / "first.json",
                              checkpoint=tmp_path / "first_checkpoint.json", resume=None, chunk_steps=None)


def test_runner_numeric_checkpoint_restart_matches_single_pass(tmp_path, monkeypatch):
    from beamfastsim import flash_checkpoint as CP
    args = _runner_fixture(tmp_path, monkeypatch)
    assert R.run(args) == 2
    first_report = json.loads(args.out.read_text())
    assert first_report["checkpoint"]["total_species_iterations"] == [1, 1]
    first_checkpoint = args.checkpoint
    resumed = argparse.Namespace(**vars(args))
    resumed.resume = first_checkpoint
    resumed.max_steps = 19
    resumed.out = tmp_path / "resumed.json"
    resumed.checkpoint = tmp_path / "resumed_checkpoint.json"
    R.run(resumed)
    resumed_arrays = np.load(resumed.out.with_suffix(".npz"), allow_pickle=False)
    uninterrupted = argparse.Namespace(**vars(args))
    uninterrupted.max_steps = 20
    uninterrupted.out = tmp_path / "uninterrupted.json"
    uninterrupted.checkpoint = None
    R.run(uninterrupted)
    with np.load(uninterrupted.out.with_suffix(".npz"), allow_pickle=False) as whole:
        assert whole.files == resumed_arrays.files
        for name in whole.files:
            assert whole[name].dtype == resumed_arrays[name].dtype
            assert whole[name].tobytes() == resumed_arrays[name].tobytes(), name
    resumed_arrays.close()
    resumed_manifest = json.loads(resumed.checkpoint.read_text())
    first_manifest = json.loads(first_checkpoint.read_text())
    assert resumed_manifest["parent_sha256"] == first_manifest["sha256"]
    changed = argparse.Namespace(**vars(resumed))
    changed.seed = 5
    changed.out = tmp_path / "foreign.json"
    changed.checkpoint = tmp_path / "foreign_checkpoint.json"
    with pytest.raises(ValueError, match="binding mismatch"):
        R.run(changed)
    assert not changed.out.exists()


def test_runner_chunk_receipts_preserve_every_parent(tmp_path, monkeypatch):
    args = _runner_fixture(tmp_path, monkeypatch)
    args.max_steps = 6
    args.chunk_steps = 2
    args.checkpoint = tmp_path / "chunks.json"
    assert R.run(args) == 2
    receipts = sorted(tmp_path.glob("chunks.chunk*.json"))
    assert len(receipts) == 3
    for previous, current in zip(receipts, receipts[1:]):
        assert json.loads(current.read_text())["parent_sha256"] == json.loads(previous.read_text())["sha256"]
    report = json.loads(args.out.read_text())
    assert report["parents"] == 2 and report["checkpoint"]["chunks"] == 3
    assert report["checkpoint"]["total_species_iterations"] == [6, 6]
