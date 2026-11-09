"""barId -> (sector slot, alignment body, layer) for the extracted v04 CRV (EX 0..255, T1 256..383, T2 384..511, muon
taggers 512..575); the same decode as crvextrap.crvgeom.bar_slot_body_layer, kept here so the objective has no
dependency on the analysis repository."""
import numpy as np

N_TOP_BARS = 512
SLOT_NAME = ("EX", "T1", "T2", "TAG")
BODY_NAMES = ([f"EX{m}" for m in range(4)] + [f"T1{m}" for m in range(2)] + [f"T2{m}" for m in range(2)]
              + [f"M{i + 1}" for i in range(8)])
N_BODY = 16


def bar_slot_body_layer(bid):
    """Vectorised barId -> (sector slot, alignment body, layer), muon taggers included."""
    bid = np.asarray(bid, np.int64)
    slot = np.select([bid < 256, bid < 384, bid < N_TOP_BARS], [0, 1, 2], 3)
    off = np.select([slot == 0, slot == 1, slot == 2], [0, 256, 384], N_TOP_BARS)
    rel = bid - off
    body = np.select([slot == 0, slot == 1, slot == 2], [rel // 64, 4 + rel // 64, 6 + rel // 64], 8 + rel // 8)
    layer = np.where(slot < 3, (rel % 64) // 16, (rel % 8) // 2)
    return slot, body, layer
