"""Convert 1D histograms from ROOT files (stock Validation/DQM files, or refills with the same binning) into the
benchmark's hists.json. Several files with the same booking are taken as shards (one per file), which turns on the
paired per-shard statistic when both arms share shard names.

    python -m benchmarks.fastsim_score.from_root OUT.json FILE.root [FILE.root ...] [--select REGEX]
        [--yield NAME=HIST ...] [--shard-names stem]

--yield NAME=HIST stores the entries of HIST (booked bins plus flow) as the yield NAME. Shard names default to the
file stems, so name the shard files of the two arms alike (e.g. shard_03.root on both sides).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def read(files: list[Path], select: str = "", yields: dict[str, str] | None = None) -> dict:
    import uproot  # only this converter needs uproot

    pat = re.compile(select) if select else None
    hists: dict[str, dict] = {}
    ycounts: dict[str, dict] = {n: {"shards": {}} for n in (yields or {})}
    for f in files:
        sid = f.stem
        with uproot.open(f) as rf:
            for key, cls in rf.classnames().items():
                name = key.rsplit(";", 1)[0]
                if not cls.startswith("TH1") or (pat and not pat.search(name)):
                    continue
                h = rf[key]
                vals, edges = h.to_numpy(flow=False)
                e = hists.setdefault(name, {"edges": edges.tolist(), "shards": {}})
                if len(e["edges"]) != len(edges):
                    raise ValueError(f"{f.name}:{name}: binning differs from the first file")
                e["shards"][sid] = [float(v) for v in vals]
            for yname, hname in (yields or {}).items():
                ycounts[yname]["shards"][sid] = float(rf[hname].values(flow=True).sum())
    return {"hists": hists, "yields": ycounts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--select", default="")
    ap.add_argument("--yield", dest="yields", action="append", default=[], metavar="NAME=HIST")
    a = ap.parse_args(argv)
    ys = dict(y.split("=", 1) for y in a.yields)
    a.out.write_text(json.dumps(read(a.files, a.select, ys)) + "\n")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
