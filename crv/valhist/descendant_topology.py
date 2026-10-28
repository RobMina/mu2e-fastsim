"""Layer topology of muon descendants: library sidecar and a truth census of any step arm.

sidecar: reconstructs, from the library's own donor tables, each library crossing's bar, muon and sector-layer visit
(tier1b.muon_visits), checks them against the library crossings (bin, context, direction) and that every other-bar
offset is an exact bar-centre offset, and writes them next to nothing else (the library is not rewritten).
`tier1b.attach_topology` turns the sidecar into the per-descendant topology the opt-in topology placement uses.

census: per arm, every muon-descendant step relative to its own ancestor muon's crossings (full integer identities,
never nearest position): in a bar the ancestor crosses (own), in a sector layer the ancestor visits at a lean-signed
side offset k beyond the crossed bars (side+k / side-k, |k| > 3 pooled), in a sector layer it does not visit
(unvisited), or with no ancestor crossing at all (no_crossing). Step counts and visible energy with event-clustered
paired moments against the first arm. A truth diagnostic, not a response model.

    python -m valhist.descendant_topology sidecar --manifest LIBRARY.json --out NEW.npz
    python -m valhist.descendant_topology census --arm g4='DIR/g4all.{s}.root' --arm ccp='DIR/fast1b.{s}.root' \
        --shards 11-20 --out NEW.json
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, tier1b as T
from crvfastsim.analyses.census import classify
from . import descendant_audit as A
from . import steplayers as S

SIDE_MAX = 3
VIEWS = {"topo": ["grp", "topo"], "lay_topo": ["grp", "lay", "topo"]}


# ----------------------------------------------------------------------------------------------------------- sidecar
def donor_crossings(path, geom):
    """Library crossings of one donor table, in library order, with muon visits (muon ids local to the file)."""
    steps, parts = S._read(path)
    s = classify(steps, parts, geom.sector)
    x = T._crossings(s.loc[s.cls.eq("mu_primary")])
    mu, vis, visits = T.muon_visits(x, geom)
    return x, mu, vis, visits


def sidecar(manifest, out, counters=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tick = time.monotonic()
    m = json.loads(Path(manifest).read_text())
    library = str(Path(manifest).with_suffix(".root"))
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    from mu2efastsim import tables as core_tables
    lx = core_tables.read(library, "libxing")
    bins = core_tables.read(library, "libbins")
    if not np.array_equal(lx.xid.to_numpy(), np.arange(len(lx))):
        raise ValueError("library crossings are not ordered contiguous ids")
    cross, visits, offset, muoff, records = [], [], 0, 0, []
    for f in m["inputs"]:
        start = time.monotonic()
        x, mu, vis, v = donor_crossings(f, geom)
        z = lx.iloc[offset:offset + len(x)]
        np.testing.assert_array_equal(z.bin, T._bin(bins, T._group_index(geom, x.bar), x.p.to_numpy(), x.len.to_numpy()))
        if "ctx" in z.columns:
            np.testing.assert_array_equal(z.ctx, T.crossing_context(x, geom))
        np.testing.assert_allclose(z[["ux", "uy", "uz"]], T._unit(x[["px0", "py0", "pz0"]].to_numpy()), rtol=1e-7,
                                   atol=1e-12)
        cross.append(pd.DataFrame({"xid": np.arange(offset, offset + len(x)), "bar": x.bar.to_numpy().astype(np.int64),
                                   "mu": mu + muoff, "vis": vis}))
        visits.append(v.assign(mu=v.mu.to_numpy().astype(np.int64) + muoff))
        records.append({"path": f, "sha256": A.digest(f), "crossings": len(x), "muons": int(mu.max() + 1) if len(mu) else 0,
                        "visits": len(v), "seconds": time.monotonic() - start})
        offset += len(x)
        muoff += records[-1]["muons"]
        print(json.dumps(records[-1]), flush=True)
    if offset != len(lx):
        raise ValueError("library crossing count differs from the donor reconstruction")
    c, v = pd.concat(cross, ignore_index=True), pd.concat(visits, ignore_index=True)
    lib = {"libxing": lx, "libdesc": core_tables.read(library, "libdesc")}
    lib = T.attach_topology(lib, c, v, geom)
    ld = lib["libdesc"]
    other = ld.samebar.eq(0).to_numpy()
    has = other & (ld.tvis.to_numpy() != T.TOPO_NONE)
    grp = np.asarray(T.GROUPS)[(lx.bin.to_numpy() // (T.N_PBINS * T.N_LBINS))[ld.xid.to_numpy()]]
    census = (pd.DataFrame({"grp": grp[other], "has": has[other], "tvis": np.where(has, ld.tvis, 99)[other],
                            "tside": np.where(has, ld.tside, 0)[other].astype(int), "e": ld.edep.to_numpy()[other]})
              .groupby(["grp", "has", "tvis", "tside"]).e.agg(["size", "sum"]).reset_index())
    np.savez_compressed(out, **{f"crossings_{k}": c[k].to_numpy() for k in c.columns},
                        **{f"visits_{k}": v[k].to_numpy() for k in v.columns})
    result = {"library": library, "library_sha256": A.digest(library), "manifest": str(manifest),
              "geometry_sha256": A.digest(geometry.find_counters(counters or S.COUNTERS)),
              "code_sha256": {p: A.digest(p) for p in (__file__, T.__file__)}, "donors": records,
              "other_bar_rows": int(other.sum()), "with_topology_rows": int(has.sum()),
              "other_bar_energy": float(ld.edep.to_numpy()[other].sum()),
              "with_topology_energy": float(ld.edep.to_numpy()[has].sum()),
              "census": json.loads(census.to_json(orient="records")), "sidecar_sha256": A.digest(out),
              "seconds": time.monotonic() - tick}
    Path(str(out) + ".json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("other_bar_rows", "with_topology_rows", "other_bar_energy",
                                             "with_topology_energy", "seconds")}), flush=True)
    return result


def read_sidecar(path, library=None):
    """(crossings, visits) frames of a sidecar; with library, require the library hash it was made for."""
    from crvfastsim.analyses.tier1b import read_topology_sidecar
    return read_topology_sidecar(path, library)


# ------------------------------------------------------------------------------------------------------------ census
def topo_labels(steps, parts, geom):
    """Muon-descendant steps with their topology label relative to their own ancestor muon's crossings."""
    s = classify(steps, parts, geom.sector)
    x = (s.loc[s.cls.eq("mu_primary")].sort_values("t0", kind="stable")
         .groupby(tables.KEY + ["prim_key", "bar"], sort=False).first().reset_index())
    d = s.loc[s.cls.eq("mu_descendant")].copy()
    lid, widx = T.layer_slots(geom)
    d["lay"] = T.layer_rank(geom)[d.bar.to_numpy()]
    d["grp"] = d["group"]
    d["w"] = d.edep.to_numpy(dtype=np.float64)
    topo = np.full(len(d), "no_crossing", dtype=object)
    if len(x) and len(d):
        xk = pd.MultiIndex.from_frame(x[tables.KEY + ["prim_key"]].astype(np.int64))
        dk = pd.MultiIndex.from_frame(d[tables.KEY + ["prim_key"]].astype(np.int64))
        crossed = dk.isin(xk)
        own = pd.MultiIndex.from_frame(d[tables.KEY + ["prim_key", "bar"]].astype(np.int64)).isin(
            pd.MultiIndex.from_frame(x[tables.KEY + ["prim_key", "bar"]].astype(np.int64)))
        # per (ancestor, sector layer): width ranks crossed and the lean of the ancestor there
        xb = x.bar.to_numpy().astype(np.int64)
        ax_w = geom.local_to_world[xb, 1]
        u = T._unit(x[["px0", "py0", "pz0"]].to_numpy())
        lay = pd.DataFrame({**{c: x[c].to_numpy().astype(np.int64) for c in tables.KEY + ["prim_key"]},
                            "lid": lid[xb], "w": widx[xb], "s": T._lean(u, ax_w)})
        lay = lay.groupby(tables.KEY + ["prim_key", "lid"], sort=False).agg(
            wmin=("w", "min"), wmax=("w", "max"), s=("s", "first")).reset_index()
        q = d[tables.KEY + ["prim_key"]].astype(np.int64).assign(lid=lid[d.bar.to_numpy()], w=widx[d.bar.to_numpy()],
                                                                  row=np.arange(len(d)))
        q = q.merge(lay, on=tables.KEY + ["prim_key", "lid"], how="left").sort_values("row")
        visited = q.wmin.notna().to_numpy()
        w, wmin, wmax = q.w.to_numpy(), q.wmin.fillna(0).to_numpy(), q.wmax.fillna(0).to_numpy()
        k = np.where(w > wmax, w - wmax, np.where(w < wmin, w - wmin, 0)) * q.s.fillna(1).to_numpy()
        names = np.array(["side<-3"] + [f"side{v:+d}" for v in range(-SIDE_MAX, SIDE_MAX + 1)] + ["side>+3"], object)
        side = names[np.clip(k, -SIDE_MAX - 1, SIDE_MAX + 1).astype(int) + SIDE_MAX + 1]
        topo = np.select([own, crossed & visited & (k != 0), crossed & visited, crossed],
                         [np.full(len(d), "own", object), side, np.full(len(d), "inrange", object),
                          np.full(len(d), "unvisited", object)], default="no_crossing")
    d["topo"] = topo
    d["ev"] = S.evkey(d)
    return d


def census(arms, shard_list, out):
    """Topology census of every arm against the first, per shard summed (paired event moments)."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    geom = geometry.from_offline_config(geometry.find_counters(S.COUNTERS))
    acc = {name: {} for name, _ in arms[1:]}
    tick = time.monotonic()
    for k in shard_list:
        ref = topo_labels(*S._read(arms[0][1].format(s=S.shard(k), k=k)), geom)
        for name, pat in arms[1:]:
            cand = topo_labels(*S._read(pat.format(s=S.shard(k), k=k)), geom)
            acc[name] = S._merge_views(acc[name], {v: S.accumulate(ref, cand, cols) for v, cols in VIEWS.items()})
        print(f"{S.shard(k)} done", flush=True)
    result = {"arms": dict(arms), "shards": shard_list, "uncertainty": "paired event Poisson-bootstrap moments, fixed library",
              "views": {name: A.summarize(v) for name, v in acc.items()}, "seconds": time.monotonic() - tick}
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sidecar")
    s.add_argument("--manifest", required=True, help="library JSON (its inputs are the donor tables)")
    s.add_argument("--out", required=True, help="new .npz")
    s.add_argument("--counters")
    c = sub.add_parser("census")
    c.add_argument("--arm", action="append", required=True, metavar="NAME=PATTERN")
    c.add_argument("--shards", default="11-20")
    c.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "sidecar":
        sidecar(a.manifest, a.out, a.counters)
    else:
        census([tuple(x.split("=", 1)) for x in a.arm], S.shards(a.shards), a.out)


if __name__ == "__main__":
    main()
