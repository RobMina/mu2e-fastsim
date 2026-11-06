"""Describe unfinished source lanes in a saved flux run without rerunning transport."""
import argparse
import json
from pathlib import Path

import numpy as np


def describe(path):
    with np.load(path, allow_pickle=False) as saved:
        a = {k: saved[k] for k in saved.files}
    n = len(a["source_pdg"])
    out = {}
    for name in ("capped", "loopers"):
        mask = a[name].astype(bool)
        ids = np.flatnonzero(mask)
        rows = {}
        for key in ("source_run", "source_subrun", "source_event", "source_particle_key", "source_pdg",
                    "state_x", "state_u", "state_T", "state_s", "state_t", "state_nvac", "state_nslab", "state_cnt"):
            if key in a:
                rows[key] = a[key][ids].tolist()
        remaining = float(a["state_T"][ids].sum())
        out[name] = dict(count=len(ids), parents=n, rows=rows, remaining_kinetic_mev=remaining,
                         scope="Remaining primary kinetic energy only; not missing cascade/annihilation energy")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("births", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    text = json.dumps(describe(a.births), indent=2, allow_nan=False) + "\n"
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
