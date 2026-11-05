"""Memberwise CaloShowerStep blob decoding and the on-spill EWM timing rule."""
import struct

import numpy as np

from calofastsim.artsteps import decode_blobs, onspill_timing


def _blob(steps):
    """Encode steps as art writes vector<CaloShowerStep> memberwise (big-endian)."""
    n = len(steps)
    body = b"".join(struct.pack(">IH", 0x40000020, 10) + struct.pack(">IH", 0x40000012, 10)
                    + struct.pack(">IHI", 0x40000006, 10, s["product"]) + struct.pack(">IH", 0x40000002, 10)
                    + struct.pack(">Q", s["key"]) for s in steps)
    for name, fmt in (("volume", ">i"), ("ncomp", ">I"), ("time", ">d"), ("eg4", ">f"), ("ebirks", ">f"),
                      ("mom", ">f")):
        body += b"".join(struct.pack(fmt, s[name]) for s in steps)
    body += b"".join(struct.pack(">IH3d", 0x4000001E, 11, *s["pos"]) for s in steps)
    payload = struct.pack(">HHII", 0x400A, 0, 0x67265319, n) + body
    return struct.pack(">I", 0x40000000 | len(payload)) + payload


def test_decode_matches_encoded_values():
    rng = np.random.default_rng(1)
    events = []
    for n in (3, 0, 5):
        events.append([{"product": int(rng.integers(1, 2**31)), "key": int(rng.integers(0, 2**40)),
                        "volume": int(rng.integers(0, 1348)), "ncomp": int(rng.integers(1, 9)),
                        "time": float(rng.uniform(0, 2000)), "eg4": float(np.float32(rng.uniform(0, 50))),
                        "ebirks": float(np.float32(rng.uniform(0, 50))), "mom": float(np.float32(rng.uniform(0, 100))),
                        "pos": tuple(rng.uniform(-100, 200, 3))} for _ in range(n)])
    blobs = [_blob(e) if e else b"" for e in events]
    data = np.frombuffer(b"".join(blobs), np.uint8)
    offsets = np.r_[0, np.cumsum([len(b) for b in blobs])]
    col = decode_blobs(data, offsets)
    flat = [s for e in events for s in e]
    np.testing.assert_array_equal(col["entry"], [0, 0, 0, 2, 2, 2, 2, 2])
    np.testing.assert_array_equal(col["volume"], [s["volume"] for s in flat])
    np.testing.assert_array_equal(col["sim_key"], [s["key"] for s in flat])
    np.testing.assert_array_equal(col["time"], [s["time"] for s in flat])
    np.testing.assert_array_equal(col["energy_birks"], np.float32([s["ebirks"] for s in flat]))
    np.testing.assert_array_equal(col["pos_z"], [s["pos"][2] for s in flat])


def test_onspill_timing_cycle():
    t = onspill_timing(np.arange(1, 11))
    np.testing.assert_array_equal(t["pbtime"], np.float32([-205, -210, -215, -220, -200] * 2))
    np.testing.assert_array_equal(t["event_length"], [1700.0] * 10)
