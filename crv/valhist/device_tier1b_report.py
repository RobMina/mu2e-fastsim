"""Print or combine device-tier1b audits without recomputing inputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from valhist import stephist as H, device_tier1b_moments as M, steplayers as S, descendant_audit as A
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out")
    a = p.parse_args()
    runs = [json.loads(Path(path).read_text()) for path in a.inputs]
    hist, pair, records = {k: {} for k in ("numpy", "jax")}, {}, []
    seen = set()
    for run in runs:
        for rec in run["records"]:
            if rec["shard"] in seen:
                raise ValueError("overlapping shards")
            seen.add(rec["shard"])
            records.append(rec)
        for name in hist:
            H.add(hist[name], H.from_json(run["hists"][name]))
        H.add_paired(pair, {k: np.asarray(v) for k, v in run["paired"].items()})
    raw = {name: {} for name in hist}
    complete_moments = all("raw_moments" in r and r["raw_moments"] for r in runs)
    if complete_moments:
        for run in runs:
            for name in raw:
                M.add(raw[name], {k: np.asarray(v) if isinstance(v, list) else v
                                  for k, v in run["raw_moments"][name].items()})
    cmp = H.compare(hist["numpy"], hist["jax"], pair)
    components = {}
    for k in hist["numpy"]:
        if not k.startswith("_total"):
            continue
        components[k] = {}
        for c in H.COMPONENTS[1:]:
            pieces = [r.get("component_yields", {}).get(k, {}).get(c, {}) for r in runs]
            d2 = (sum((v["n_ref"] * v["rel_se_paired"]) ** 2 for v in pieces)
                  if all(v.get("rel_se_paired") is not None for v in pieces) else None)
            components[k][c] = H.yield_stat(hist["numpy"][k], hist["jax"][k], c, None if d2 is None else np.array([d2]))
    census = {}
    for run in runs:
        reconstructed = {}
        for view, rows in run.get("layer_census", {}).items():
            frame = pd.DataFrame(rows)
            keep = [c for c in frame if c not in S.SUMS and not c.startswith(("n_ratio", "n_delta", "e_ratio", "e_delta"))]
            reconstructed[view] = frame[keep + S.SUMS]
        census = S._merge_views(census, reconstructed)
    result = dict(shards=sorted(seen), events=sum(r["events"] for r in records),
                  max_replay_abs=max(r["replay"]["max_abs"] for r in records),
                  flagged=[k for k, v in cmp.items() if v.get("flag")], comparison=cmp,
                  device_array_bytes=runs[0]["device_array_bytes"],
                  library_load_s=runs[0]["library_load_s"], device_prepare_s=runs[0]["device_prepare_s"],
                  warm={name: [r.get("warm", {}).get(name) for r in records] for name in hist},
                  moments={n: M.summarize(v) for n, v in raw.items()} if complete_moments else None,
                  moment_pieces=[r.get("moments") for r in runs],
                  moments_combined=bool(complete_moments), component_yields=components, layer_census=A.summarize(census), records=records,
                  hists={n: H.to_json(v) for n, v in hist.items()}, paired={k: v.tolist() for k, v in pair.items()})
    text = json.dumps(result, indent=2, allow_nan=False)
    if a.out:
        with open(a.out, "x") as stream:
            stream.write(text)
    print(text)


if __name__ == "__main__":
    main()
