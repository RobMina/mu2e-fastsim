"""Write a CRVPropagation DbService text table (per-bar fibre signal speed and attenuation length ratios relative to the
lookup tables; Offline PR crv-fiber-atten) from per-module or per-sector values, for the extracted v04 CRV (576 bars).

Keys are alignment bodies (EX0..EX3, T10, T11, T20, T21, M1..M8) or sector slots (EX, T1, T2, TAG); a body value wins
over its sector. Unlisted bars get 1.

    python3 write_crv_propagation.py OUT.txt [--speed EX=1.08 T2=1.02] [--atten T1=0.58 T10=0.55] [--json KNOBS.json]

KNOBS.json: {"speed": {key: ratio}, "atten": {key: ratio}}. Use in the digi job (stock arm):
    services.ProditionsService.crvLightPropagation.useDb : true
    services.DbService.textFile : [ "OUT.txt" ]
"""
import argparse
import json

import numpy as np

from crvfastsim.objective import bars as G

N_BARS = 576


def per_bar(values):
    slot, body, _ = G.bar_slot_body_layer(np.arange(N_BARS))
    out = np.ones(N_BARS)
    for k, v in values.items():
        if k not in G.SLOT_NAME and k not in G.BODY_NAMES:
            raise KeyError(f"unknown key {k}; use {G.SLOT_NAME} or {G.BODY_NAMES}")
    for k, v in values.items():                                    # sectors first, then bodies override
        if k in G.SLOT_NAME:
            out[slot == G.SLOT_NAME.index(k)] = v
    for k, v in values.items():
        if k in G.BODY_NAMES:
            out[body == G.BODY_NAMES.index(k)] = v
    if not np.all(out > 0):
        raise ValueError("ratios must be positive")
    return out


def kv(items):
    return {k: float(v) for k, v in (s.split("=") for s in items or [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--speed", nargs="*", help="KEY=ratio, fibre signal speed relative to the table")
    ap.add_argument("--atten", nargs="*", help="KEY=ratio, attenuation length relative to the table")
    ap.add_argument("--json", help='{"speed": {...}, "atten": {...}}')
    a = ap.parse_args()
    sp, at = kv(a.speed), kv(a.atten)
    if a.json:
        j = json.load(open(a.json))
        sp.update(j.get("speed", {}))
        at.update(j.get("atten", {}))
    s, t = per_bar(sp), per_bar(at)
    with open(a.out, "w") as f:
        f.write("TABLE CRVPropagation\n")
        f.write(f"# speed {json.dumps(sp)} atten {json.dumps(at)}\n")
        f.write("#bar,signalSpeedRatio,attenuationLengthRatio\n")
        for b in range(N_BARS):
            f.write(f"{b},{s[b]:.4f},{t[b]:.4f}\n")
    print(f"wrote {a.out}: {N_BARS} bars, speed {sp or 'all 1'}, atten {at or 'all 1'}")


if __name__ == "__main__":
    main()
