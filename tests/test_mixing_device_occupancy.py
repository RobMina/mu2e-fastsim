"""On-device occupancy output: reduction, persistence, census and equality with the full-table path."""
import json

import numpy as np
import pytest

from mixing import device_occupancy as occ


def test_sector_map_contract(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"geometry": "g", "coverage": "c", "sector_names": ["T1", "T2"], "bars_per_sector": [3, 2]}))
    mapping, names, info = occ.load_sector_map(path)
    np.testing.assert_array_equal(mapping, [0, 0, 0, 1, 1])
    assert names == ["T1", "T2"] and info == {"geometry": "g", "coverage": "c"}
    path.write_text(json.dumps({"sector_names": ["A"], "bar_to_sector": [0, 1]}))
    with pytest.raises(ValueError, match="out-of-range"):
        occ.load_sector_map(path)


def test_compact_receipt_drops_only_per_donor_arrays():
    receipt = {"intensity": np.arange(3), "selections": {"S": {"counts": 1, "donors": 2, "mask": 3,
               "source_event_ids": 4, "initial_cursor": 5, "next_cursor": 6}}}
    compact = occ.compact_receipt(receipt)
    assert compact["selections"]["S"] == {"counts": 1, "initial_cursor": 5, "next_cursor": 6}
    assert compact["intensity"] is receipt["intensity"] and "donors" in receipt["selections"]["S"]


def test_reduce_matches_host_counts():
    jax = pytest.importorskip("jax")
    rng = np.random.default_rng(3)
    b, n, m = 5, 64, 40
    sid = (rng.integers(0, 36, n) << 10) | (rng.integers(0, 6, n) << 7) | rng.integers(0, 96, n)
    tracker = {"sid": sid.astype(np.uint32), "ev": rng.integers(0, b, n), "mask": rng.random(n) < .8}
    pulses = {"bar": rng.integers(0, 5, m), "ev": rng.integers(0, b, m), "mask": rng.random(m) < .7}
    digis = {"ev": rng.integers(0, b, m), "mask": rng.random(m) < .5}
    clusters = {"ev": rng.integers(0, b, 7), "mask": rng.random(7) < .5}
    mapping = np.array([0, 0, 1, 1, 1])
    output = {"tracker": tracker, "pulses": pulses, "digis": digis, "clusters": clusters}
    result, errors = jax.device_get(jax.jit(lambda o, s: occ.reduce(o, s, 2, b))(output, mapping))
    assert not any(errors.values())
    planes = np.zeros((b, 36), int)
    np.add.at(planes, (tracker["ev"][tracker["mask"]], (sid[tracker["mask"]] & 0xFC00) >> 10), 1)
    sectors = np.zeros((b, 2), int)
    np.add.at(sectors, (pulses["ev"][pulses["mask"]], mapping[pulses["bar"][pulses["mask"]]]), 1)
    np.testing.assert_array_equal(result["planes"], planes)
    np.testing.assert_array_equal(result["sectors"], sectors)
    np.testing.assert_array_equal(result["crv_digis"], np.bincount(digis["ev"][digis["mask"]], minlength=b))
    np.testing.assert_array_equal(result["crv_clusters"], np.bincount(clusters["ev"][clusters["mask"]], minlength=b))
    pulses["bar"][np.flatnonzero(pulses["mask"])[0]] = 9
    _, errors = jax.device_get(jax.jit(lambda o, s: occ.reduce(o, s, 2, b))(output, mapping))
    assert errors["occupancy/bar"] and not errors["occupancy/plane"]


def test_write_census_and_refuse_overwrite(tmp_path):
    keys = np.array([[1, 2, 3], [1, 2, 4]], np.uint64)
    data = {"planes": np.ones((4, 36), np.int32), "sectors": np.full((4, 2), 2, np.int32),
            "crv_digis": np.arange(4, dtype=np.int32), "crv_clusters": np.zeros(4, np.int32)}
    path = tmp_path / "occupancy.npz"
    occ.write(path, keys, 10, data, ["T1", "T2"])
    census = occ.census(path, 10, 2, keys)
    assert census == {"events": 2, "tracker_digis": 72, "crv": {"pulses": 8, "digis": 1, "clusters": 0},
                      "tracker_empty_events": 0}
    with pytest.raises(ValueError, match="realization"):
        occ.census(path, 11, 2, keys)
    with pytest.raises(ValueError, match="identities"):
        occ.census(path, 10, 2, keys[::-1])
    with pytest.raises(FileExistsError):
        occ.write(path, keys, 10, data, ["T1", "T2"])


def test_compare_cli_pairs_measured_batches(tmp_path):
    run = tmp_path / "run"
    keys = np.array([[1, 1, e] for e in range(6)], np.uint64)
    rng = np.random.default_rng(1)
    planes, sectors = rng.integers(0, 9, (6, 36)), rng.integers(0, 4, (6, 2))
    rows = []
    for index, (phase, lo, hi) in enumerate((("warmup", 0, 2), ("measure", 2, 5), ("measure", 5, 6))):
        directory = run / f"{phase}_{index:08d}"
        directory.mkdir(parents=True)
        occ.write(directory / "occupancy.npz", keys[lo:hi], lo, {"planes": planes[lo:hi], "sectors": sectors[lo:hi],
                  "crv_digis": np.zeros(hi - lo, int), "crv_clusters": np.zeros(hi - lo, int)}, ["T1", "T2"])
        rows.append({"phase": phase, "directory": directory.name})
    (run / "batches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    stock = tmp_path / "stock.npz"
    np.savez(stock, event_ids=keys[::-1], planes=planes[::-1], sectors=sectors[::-1] + 1)
    occ.main([str(run), "--stock", str(stock), "--selection", "test", "--out", str(tmp_path / "paired.npz")])
    report = json.loads((tmp_path / "paired.json").read_text())
    assert report["events"] == 4 and report["paired"]["matched_events"] == 4
    assert report["paired"]["stock_only_events"] == 2
    assert report["paired"]["planes_changed_events"] == 0
    np.testing.assert_allclose(report["paired"]["sectors"]["difference"], [-1, -1])
