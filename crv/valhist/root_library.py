"""Library with step-tree roots (tier1b.build roots=True) from legacy G4 CrvStep tables, for the root-coherent draws.

The muon-descendant part comes from the donors below; bin edges, non-muon showers and their reach stay those of an
existing library, because the legacy tables predate the generator association repair (tier1b.build refuses them). The
sidecar tools accept the written manifest.

    python -m valhist.root_library --donors 'DIR/crvsteps.{s}.root' --shards 21-219:2 --base LIB.root --generated N --out NEW.root
    python -m valhist.root_library --base NEW.root --templates-from TEMPLATES.npz --out NEW_TEMPLATES.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mu2efastsim import tables as core_tables

from crvfastsim import geometry, tables, tier1b as T
from crvfastsim.analyses.census import classify
from . import descendant_audit as A
from . import descendant_templates as DT
from . import steplayers as S


def build(pattern, shards, base, out, generated, counters=None):
    if Path(out).exists():
        raise FileExistsError(out)
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    libbins = core_tables.read(base, "libbins")
    paths = [pattern.format(s=S.shard(k), k=k) for k in shards]
    parts, xoff, info = {"libxing": [], "libdesc": []}, 0, []
    for f in paths:
        lib, inf = T._build_one([f], geom, libbins, _already_validated={str(Path(f).resolve()): "legacy table"}, roots=True)
        lib["libxing"]["xid"] += xoff
        lib["libdesc"]["xid"] += xoff
        xoff += inf["crossings"]
        for k in parts:
            parts[k].append(lib[k])
        info.append({k: inf[k] for k in ("crossings", "desc_steps", "desc_steps_unattached")})
        print(f, info[-1], flush=True)
    new = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    new["libbins"] = libbins
    for k in ("libshower", "libshowersteps", "libgen"):
        new[k] = core_tables.read(base, k)
    T.write_library(out, new)
    man = {"inputs": paths, "base": str(base), "base_sha256": A.digest(base), "generated": generated, "files": len(paths),
           "crossings": int(xoff), "desc_steps": int(sum(i["desc_steps"] for i in info)),
           "desc_steps_unattached": int(sum(i["desc_steps_unattached"] for i in info)), "output_sha256": A.digest(out)}
    Path(out).with_suffix(".json").write_text(json.dumps(man, indent=2))
    print({k: v for k, v in man.items() if k != "inputs"}, flush=True)


def _crossing_table(path, geom, offset, muoff):
    """Sidecar v3 rows of one donor's crossings: bar, entry, |p|, last flag, muon, exit, path, next crossing; and the visits."""
    steps = tables.read(path, "steps")
    parts = tables.read(path, "particles", DT.PART_COLS)
    s = classify(steps, parts, geom.sector)
    mus = s.loc[s.cls.eq("mu_primary")]
    x = T._crossings(mus)
    ex = mus.sort_values(T.KEY + ["bar", "t0"], kind="stable").groupby(T.KEY + ["bar"], sort=False)[["x1", "y1", "z1"]].last()
    x["qx"], x["qy"], x["qz"] = ex.x1.to_numpy(), ex.y1.to_numpy(), ex.z1.to_numpy()
    mu, _, visits = T.muon_visits(x, geom)
    t = pd.DataFrame({"xid": np.arange(offset, offset + len(x)), "bar": x.bar.to_numpy().astype(np.int64),
                      "ex": x.x0.to_numpy(), "ey": x.y0.to_numpy(), "ez": x.z0.to_numpy(), "p": x.p.to_numpy(),
                      "last": T.last_crossing(x).astype(np.int64), "mu": mu.astype(np.int64) + muoff,
                      "qx": x.qx.to_numpy(), "qy": x.qy.to_numpy(), "qz": x.qz.to_numpy(), "len": x.len.to_numpy(),
                      "nxt": DT.next_crossing(mu, x.t0.to_numpy(), offset)})
    return t, visits.assign(mu=visits.mu.to_numpy().astype(np.int64) + muoff), (int(mu.max() + 1) if len(mu) else 0)


def templates(library, base_templates, out, counters=None):
    """Template sidecar v3: crossings and unvisited rows of library, near-miss templates and census of base_templates."""
    if Path(out).exists():
        raise FileExistsError(out)
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    man = json.loads(Path(library).with_suffix(".json").read_text())
    ld = core_tables.read(library, "libdesc")
    xs, vs, offset, muoff = [], [], 0, 0
    for f in man["inputs"]:
        x, v, nmu = _crossing_table(f, geom, offset, muoff)
        xs.append(x)
        vs.append(v)
        offset += len(x)
        muoff += nmu
    xing, vis = pd.concat(xs, ignore_index=True), pd.concat(vs, ignore_index=True)
    lid, _ = T.layer_slots(geom)
    other = np.flatnonzero(ld.samebar.to_numpy() == 0)
    xid = ld.xid.to_numpy()[other]
    db = T.bar_containing(geom.center[xing.bar.to_numpy()[xid]] + ld[["dcx", "dcy", "dcz"]].to_numpy()[other], geom)
    vkey = np.unique(vis.mu.to_numpy() * 1_000_000 + vis.lid.to_numpy().astype(np.int64))
    uv = ~np.isin(xing.mu.to_numpy()[xid] * 1_000_000 + lid[db], vkey)
    uvf = pd.DataFrame({"row": other[uv], "bar": db[uv]})
    z = np.load(base_templates)
    base = json.loads(Path(str(base_templates) + ".json").read_text())
    payload = {**{f"xing_{k}": xing[k].to_numpy() for k in xing.columns}, **{f"uv_{k}": uvf[k].to_numpy() for k in uvf.columns},
               **{k: z[k] for k in z.files if k.startswith(("nmt_", "nms_", "nmc_"))}}
    np.savez_compressed(out, **payload)
    meta = {"library": str(library), "library_sha256": A.digest(library), "base_templates": str(base_templates),
            "generated": base["generated"], "census_events": base["census_events"], "unvisited_rows": int(uv.sum()),
            "sidecar_sha256": A.digest(out)}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--donors", help="G4 CrvStep table pattern, {s} = shard id")
    p.add_argument("--shards", help="first-last:step")
    p.add_argument("--templates-from", help="only write the template sidecar --out of library --base from these templates")
    p.add_argument("--base", required=True, help="library giving bin edges, showers and reach")
    p.add_argument("--generated", type=int, help="generated events behind the donors")
    p.add_argument("--out", required=True)
    p.add_argument("--counters")
    a = p.parse_args()
    if a.templates_from:
        return templates(a.base, a.templates_from, a.out, a.counters)
    lo, _, st = a.shards.partition(":")
    x, _, y = lo.partition("-")
    build(a.donors, list(range(int(x), int(y or x) + 1, int(st or 1))), a.base, a.out, a.generated, a.counters)


if __name__ == "__main__":
    main()
