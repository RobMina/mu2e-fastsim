"""Complete source-ledger propagation, independent of transport physics/parity."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from beamfastsim.flash_flux import exit_records, main, summarize
from test_flash_flux import fixture


def ledger():
    # Unsorted input and an entirely source-free SubRun.
    return np.array([[1, 9, 9], [1, 4, 2], [1, 2, 1], [1, 2, 3]], np.int64)


def test_ledger_zero_fills_only_verified_ids_and_preserves_denominator():
    payload, beam = fixture()
    photons = exit_records(payload, beam)
    report, events = summarize(payload, photons, source_events=4, event_ledger=ledger())
    assert report["source_events"] == 4
    assert report["represented_source_events"] == 2
    assert report["complete_event_ledger"]
    assert report["zero_source_events"] == 2
    np.testing.assert_array_equal(events["event"], [[1, 2, 1], [1, 2, 3], [1, 4, 2], [1, 9, 9]])
    np.testing.assert_array_equal(events["source_parents"], [2, 0, 1, 0])
    assert report["per_source_event"]["transparent_photons"] == .5
    for name in ("transparent_photons", "uncollided_photons", "transparent_energy_mev", "uncollided_energy_mev"):
        np.testing.assert_array_equal(events[name][[1, 3]], [0., 0.])
        assert events[name].sum() == pytest.approx(report["totals"][name])
    np.testing.assert_array_equal(report["source_shards"], [[1, 2], [1, 4], [1, 9]])
    np.testing.assert_array_equal(report["by_shard"]["transparent_photons"], [2., 0., 0.])
    # A scalar denominator alone must never manufacture event identities.
    legacy_report, legacy = summarize(payload, photons, source_events=4)
    assert len(legacy["event"]) == 2
    assert not legacy_report["complete_event_ledger"]
    assert legacy_report["zero_source_events"] is None


@pytest.mark.parametrize("bad,match", [
    (np.array([[1, 2, 1], [1, 2, 1]]), "Duplicate"),
    (np.array([[1., 2., 1.], [1., 4., 2.]]), "integer"),
    (np.array([[1, 2, -1], [1, 4, 2]]), "integer"),
    (np.ones((2, 3), bool), "integer"),
    (np.array([1, 2, 3]), "shape"),
    (np.array([[1, 2, 1]]), "size"),
    (np.array([[1, 2, 1], [1, 4, 99]]), "absent"),
])
def test_invalid_or_incomplete_ledger_rejected(bad, match):
    payload, beam = fixture()
    with pytest.raises(ValueError, match=match):
        summarize(payload, exit_records(payload, beam), source_events=2, event_ledger=bad)


def test_embedded_ledger_and_explicit_override_must_agree():
    payload, beam = fixture()
    payload["source_event_ledger"] = ledger()
    photons = exit_records(payload, beam)
    report, events = summarize(payload, photons, source_events=4, event_ledger=ledger()[::-1])
    assert report["complete_event_ledger"] and len(events["event"]) == 4
    bad = ledger()
    bad[0, 2] += 1
    with pytest.raises(ValueError, match="disagrees"):
        summarize(payload, photons, source_events=4, event_ledger=bad)
    with pytest.raises(ValueError, match="size"):
        summarize(payload, photons, source_events=5)


@pytest.mark.parametrize("base", [2**53, 2**63, 2**64 - 3])
def test_exact_mixed_integer_identities_join_to_ledger(base):
    payload, beam = fixture()
    payload["source_event"] = np.array([base, base, base + 1], np.uint64)
    events = np.array([[1, 2, base], [1, 4, base + 1], [1, 4, base + 2]], np.uint64)
    report, out = summarize(payload, exit_records(payload, beam), source_events=3, event_ledger=events)
    np.testing.assert_array_equal(out["event"], events)
    np.testing.assert_array_equal(out["source_parents"], [2, 1, 0])
    assert report["zero_source_events"] == 1
    events[0, 2] += np.uint64(1)
    with pytest.raises(ValueError, match="absent"):
        summarize(payload, exit_records(payload, beam), source_events=3, event_ledger=events)


def test_no_parent_payload_keeps_verified_zero_events():
    payload, beam = fixture()
    payload = {k: v[:0] for k, v in payload.items()}
    report, events = summarize(payload, exit_records(payload, beam), source_events=4, event_ledger=ledger())
    assert report["parents"] == report["represented_source_events"] == 0
    assert report["zero_source_events"] == 4
    np.testing.assert_array_equal(events["source_parents"], np.zeros(4))
    np.testing.assert_array_equal(events["transparent_photons"], np.zeros(4))


def _cli(tmp_path, source, output):
    return [*source, "--geom", str(tmp_path / "overlay.txt"), "--offline-root", str(tmp_path),
            "--source-events", "4", "--out", str(output)]


def test_births_cli_explicit_ledger_and_missing_file(tmp_path, monkeypatch):
    payload, beam = fixture()
    monkeypatch.setattr("beamfastsim.flash_geometry.from_offline", lambda *_: SimpleNamespace(beam=beam))
    births = tmp_path / "births.npz"
    sidecar = tmp_path / "states_events.npz"
    np.savez(births, **payload)
    np.savez(sidecar, event=ledger())
    out = tmp_path / "flux"
    assert main(_cli(tmp_path, ["--births", str(births), "--event-ledger", str(sidecar)], out)) == 0
    with np.load(out / "event_totals.npz") as z:
        np.testing.assert_array_equal(z["source_parents"], [2, 0, 1, 0])
    assert json.loads((out / "flux.json").read_text())["event_ledger"] == str(sidecar)
    with pytest.raises(FileNotFoundError):
        main(_cli(tmp_path, ["--births", str(births), "--event-ledger", str(tmp_path / "missing.npz")], tmp_path / "bad"))


def test_states_cli_autoloads_and_embeds_sidecar_for_births_reprocessing(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    payload, beam = fixture()
    source = {k.removeprefix("source_"): v for k, v in payload.items() if k.startswith("source_")}
    source.update(x=np.zeros((3, 3)), p=np.tile([0., 0., 3.], (3, 1)))
    states = tmp_path / "states.npz"
    sidecar = tmp_path / "states_events.npz"
    np.savez(states, **source)
    np.savez(sidecar, event=ledger())
    monkeypatch.setattr("beamfastsim.flash_geometry.from_offline", lambda *_: SimpleNamespace(beam=beam))
    monkeypatch.setattr("beamfastsim.cechain_jax.device_maps", lambda *_, **__: ())
    # Stub only charged transport: actual CLI, birth postprocessing, and summary run.
    result = dict(state={}, records={k.removeprefix("photon_"): v for k, v in payload.items()
                                     if k.startswith("photon_")},
                  diagnostics=dict(iterations=np.array([1, 1])),
                  **{k: payload[k] for k in ("overflow", "capped", "loopers")})
    calls = []
    def transport(*args, **kwargs):
        calls.append(1)
        return result
    monkeypatch.setattr("beamfastsim.flash_target_jax.target_source", transport)
    out = tmp_path / "flux"
    assert main(_cli(tmp_path, ["--states", str(states), "--maps", "fake.header"], out)) == 0
    assert len(calls) == 1
    with np.load(out / "births.npz") as z:
        assert "source_event_ledger" in z
    reprocessed = tmp_path / "reprocessed"
    assert main(_cli(tmp_path, ["--births", str(out / "births.npz")], reprocessed)) == 0
    with np.load(out / "event_totals.npz") as a, np.load(reprocessed / "event_totals.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert json.loads((reprocessed / "flux.json").read_text())["event_ledger"] == "embedded in births"
    # An incomplete sidecar must fail before starting expensive charged transport.
    np.savez(sidecar, event=ledger()[:3])
    with pytest.raises(ValueError, match="size"):
        main(_cli(tmp_path, ["--states", str(states), "--maps", "fake.header"], tmp_path / "bad"))
    assert len(calls) == 1
