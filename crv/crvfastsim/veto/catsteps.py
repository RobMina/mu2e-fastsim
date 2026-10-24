"""Compact CrvStep tables of beam-frame "*Cat" datasets for deadtime method F from beam frames (crvfastsim.veto.mixframes).

Input: a CrvStepTableDumper file of one *Cat dts file (dump_crvsteps_cat.fcl: trees gen, steps, particles). Output: a
parquet with one row per CrvStep of the frames that have any (pdg joined, float32 kinematics) plus the frame count of
the input file in the metadata, so the mixer can draw frames uniformly, stepless ones included, without the full table.

    python -m crvfastsim.veto.catsteps crvsteps.<file>.root --dataset NeutralsFlashCat --out steps.<file>.parquet
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

F32 = ("edep", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "p1", "len")


def convert(path, dataset, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import uproot
    with uproot.open(path) as f:
        g = f["dump/gen"].arrays(["subrun", "event"], library="np")
        s = f["dump/steps"].arrays(library="pd")
        p = f["dump/particles"].arrays(["run", "subrun", "event", "key", "pdg"], library="pd")
    frames = int(np.unique((g["subrun"].astype(np.int64) << 20) | g["event"].astype(np.int64)).size)
    s = s.drop(columns=[c for c in ("nstep",) if c in s.columns])
    s = s.merge(p.rename(columns={"key": "sp"}), on=["run", "subrun", "event", "sp"], how="left")
    s["pdg"] = s["pdg"].fillna(0).astype(np.int32)
    s = s.astype({c: np.float32 for c in F32 if c in s.columns})
    stepped = int(np.unique((s.subrun.to_numpy().astype(np.int64) << 20) | s.event.to_numpy().astype(np.int64)).size)
    meta = {"dataset": dataset, "source": os.path.basename(path), "frames": frames, "stepped_frames": stepped,
            "steps": int(len(s))}
    t = pa.Table.from_pandas(s, preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"crvfastsim": json.dumps(meta).encode()})
    pq.write_table(t, out, compression="zstd")
    return meta


def read_meta(path):
    import pyarrow.parquet as pq
    return json.loads(pq.read_schema(path).metadata[b"crvfastsim"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = convert(a.input, a.dataset, a.out)
    print(json.dumps(m) + f" -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
