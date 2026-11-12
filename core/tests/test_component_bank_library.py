"""Component banks on the shared library core: save_library / load_library round trip with a checked manifest
(kind replay, bank_admissible false), replays bit-identical with the loaded bank, the trial choice unchanged by the
_matches refactor (frozen copy of the previous _choose), and refused rows counted without a draw."""
import numpy as np
import pytest

from mu2efastsim import component_response as CR
from mu2efastsim.library import manifest as LM, store as LS
from test_component_response import bank, incoming, placement


def _legacy_choose(library, local, rng):
    n, donors = len(local["pdg"]), library.entrances
    left, right = [], []
    chunk = max(1, 2_000_000 // len(donors["pdg"]))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        good = local["pdg"][start:stop, None] == donors["pdg"][None, :]
        for key in ("position", "momentum", "energy"):
            a, b = local[key][start:stop], donors[key]
            atol, rtol = (CR.POSITION_ATOL, 0.) if key == "position" else (CR.STATE_ATOL, CR.STATE_RTOL)
            close = np.isclose(a[:, None], b[None, :], atol=atol, rtol=rtol)
            good &= np.all(close, axis=-1) if a.ndim == 2 else close
        ii, jj = np.nonzero(good)
        left.append(ii + start)
        right.append(jj)
    left = np.concatenate(left) if left else np.empty(0, dtype=np.int64)
    right = np.concatenate(right) if right else np.empty(0, dtype=np.int64)
    counts = np.bincount(left, minlength=n)
    if np.any(counts == 0):
        raise CR.UnsupportedEntrance(np.flatnonzero(counts == 0))
    pick = np.cumsum(counts) - counts + (rng.random(n) * counts).astype(np.int64)
    return right[pick]


def _same(a, b):
    return set(a) == set(b) and all(np.asarray(a[k]).dtype == np.asarray(b[k]).dtype and
                                    np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)


def test_bank_round_trip_and_manifest(tmp_path):
    lib = bank()
    m = CR.save_library(tmp_path / "bank.npz", lib, seeds={"capture": 314159})
    assert m["name"] == "secondaries.component.finite-segmented-A" and m["kind"] == "replay"
    assert m["validation"]["bank_admissible"] is False and not LM.complete(m)
    assert {"generator", "source"} <= set(m["unrecorded"]) and "seeds" not in m["unrecorded"]
    back = CR.load_library(tmp_path / "bank.npz")
    assert back.component == lib.component
    assert _same(CR.bank_arrays(back), CR.bank_arrays(lib))


def test_replay_is_bit_identical_with_the_loaded_bank(tmp_path):
    lib = bank()
    CR.save_library(tmp_path / "bank.npz", lib)
    back = CR.load_library(tmp_path / "bank.npz")
    p = placement(angle=0.3, translation=(5., -2., 1.))
    inc = incoming(p, n=50)
    a = CR.replay(lib, p, inc, np.arange(50), np.random.default_rng(3), next_track=1000)
    b = CR.replay(back, p, inc, np.arange(50), np.random.default_rng(3), next_track=1000)
    for t in ("particles", "steps", "exits"):
        assert _same(getattr(a, t), getattr(b, t))
    assert np.array_equal(a.trials, b.trials) and a.next_track == b.next_track


def test_choice_is_unchanged_by_the_refactor():
    lib = bank()
    local = {k: np.asarray(v) for k, v in lib.entrances.items()}
    local = {k: np.repeat(v, 400, axis=0) for k, v in local.items()}
    new = CR._choose(lib, local, np.random.default_rng(9))
    old = _legacy_choose(lib, local, np.random.default_rng(9))
    assert new.dtype == old.dtype and new.tobytes() == old.tobytes()


def test_a_tampered_bank_or_another_library_is_refused(tmp_path):
    arrays = {k: np.array(v) for k, v in CR.bank_arrays(bank()).items()}
    m = CR.save_library(tmp_path / "bank.npz", bank())
    arrays["steps_edep"][0] += 1e-9
    np.savez(tmp_path / "bad.npz", **arrays, __manifest__=np.str_(LM.dumps(m)))
    with pytest.raises(LS.LibraryError, match="content hash"):
        CR.load_library(tmp_path / "bad.npz")


def test_domain_counts_refused_rows_without_drawing():
    lib, p = bank(), placement()
    inc = incoming(p, n=4)
    inc["energy"][1] = 11.                                            # on the boundary, no recorded state
    inc["momentum"][1] *= np.sqrt(11. ** 2) / np.linalg.norm(inc["momentum"][1])
    inc["position"][2] = p.global_points(np.array([0., 0., -0.5]))  # inside, not on the boundary
    rec = CR.domain_counts(lib, p, inc).record()
    assert rec["lookups"] == 4 and rec["refused"] == 2
    assert rec["axes"]["boundary"]["above"] == 1 and rec["axes"]["state"]["above"] == 1
    with pytest.raises(CR.UnsupportedEntrance):
        CR.replay(lib, p, inc, np.arange(4), np.random.default_rng(1), next_track=10)


def test_the_caller_cannot_replace_fields_the_replay_defines(tmp_path):
    lib = bank()
    base = CR.bank_name(lib.component)
    for bad in (dict(kind="binned"), dict(dtype={}), dict(rng={}), dict(domain={}), dict(outputs=[]),
                dict(validation={"bank_admissible": True}), dict(validation={"reason": "closed"}),
                dict(name="secondaries.component.other"), dict(name=base + "."), dict(name=base + "x")):
        with pytest.raises(ValueError):
            CR.save_library(tmp_path / "bad.npz", lib, **bad)
    m = CR.save_library(tmp_path / "ok.npz", lib, name=base + ".front_center_normal.k013", version=2,
                        validation={"status": "not validated", "record": "none yet"})
    assert m["name"] == base + ".front_center_normal.k013" and m["version"] == 2
    assert m["validation"]["bank_admissible"] is False and "reason" in m["validation"]
    assert CR.load_library(tmp_path / "ok.npz").component == lib.component
