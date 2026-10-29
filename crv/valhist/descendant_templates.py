"""Template sidecar for the opt-in position-conditioned tier-1b components (crvfastsim.tier1b unvisited, nearmiss).

Built from the library's own donor tables plus a generator census; the library is not rewritten.

- Library crossings (checked against the library: bin, context, direction, count): entry point, |p|, whether it is
  the last crossing of its muon, and the muon (for the groups it crosses); these give each crossing its position
  cell (tier1b.uv_cells).
- Unvisited-layer rows: the other-bar library rows whose bar lies in a layer plane their crossing's own muon never
  visits (tier1b.muon_visits of the donor crossings), with that bar.
- Near-miss templates: the muon-descendant steps of donor events with no primary-muon crossing (the steps the library
  build leaves unattached), one template per primary muon, with its generated kinetic energy and the straight-line
  closest approach of its generated ray to the library groups (tier1b.closest_approach).
- Library chords (sidecar v3, for the opt-in chord placement and follow draw): per library crossing its exit point,
  path length and the next crossing in time of its muon.
- Near-miss census: every generated muon of the census events with no G4 primary-muon crossing (G4 CrvStep tables and
  the full gen streams of the same shards), with the same cell variables; the reach denominator, scaled by the
  library's generated exposure over the census events (as the shower denominator).

    python -m valhist.descendant_templates sidecar --manifest LIBRARY.json \
        --census-steps 'DIR/crvsteps.{s}.root' --census-gen 'DIR/gen.{s}.root' --census-shards 1-10 \
        [--census-steps 'DIR2/crvsteps.{s}.root' --census-gen 'DIR2/gen.{s}.root' --census-shards 21,23] --out NEW.npz
    python -m valhist.descendant_templates cells SIDE.npz --library LIBRARY.root   (reach table and cell occupancy)
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import genassoc, geometry, tables, tier1b as T
from crvfastsim.analyses.census import classify
from . import descendant_audit as A
from . import steplayers as S

KEY = tables.KEY
PART_COLS = KEY + ["key", "parent", "pdg", "isprim", "genidx"]


def donor_pass(path, geom):
    """Crossings (library order), muon visits and near-miss templates of one donor table."""
    steps = tables.read(path, "steps")
    parts = tables.read(path, "particles", PART_COLS)
    gen = tables.read(path, "gen", KEY + ["idx", "pdg", "x", "y", "z", "px", "py", "pz", "e", "t"])
    s = classify(steps, parts, geom.sector)
    mus = s.loc[s.cls.eq("mu_primary")]
    x = T._crossings(mus)
    # exit point of each crossing (last step end, the order of _crossings) for the opt-in chord placement
    ex = mus.sort_values(KEY + ["bar", "t0"], kind="stable").groupby(KEY + ["bar"], sort=False)[["x1", "y1", "z1"]].last()
    ex = ex.reset_index()
    if not (ex[KEY + ["bar"]].to_numpy() == x[KEY + ["bar"]].to_numpy()).all():
        raise ValueError(f"{path}: crossing exits out of the crossing order")
    x["qx"], x["qy"], x["qz"] = ex.x1.to_numpy(), ex.y1.to_numpy(), ex.z1.to_numpy()
    mu, vis, visits = T.muon_visits(x, geom)
    crossed = pd.MultiIndex.from_frame(x[KEY].drop_duplicates().astype(np.int64))
    d = s.loc[s.cls.eq("mu_descendant")]
    d = d[~pd.MultiIndex.from_frame(d[KEY].astype(np.int64)).isin(crossed)].copy()
    m = d[KEY + ["prim_key"]].drop_duplicates().reset_index(drop=True)
    pr = parts.rename(columns={"key": "prim_key"})[KEY + ["prim_key", "genidx", "pdg"]]
    m = m.merge(pr, on=KEY + ["prim_key"], how="left", validate="one_to_one")
    m = m.merge(gen.rename(columns={"idx": "genidx", "pdg": "gpdg"}), on=KEY + ["genidx"], how="left",
                validate="many_to_one")
    if m.gpdg.isna().any() or (m.gpdg.abs() != tables.PRIMARY_MUON_PDG).any() or (m.pdg != m.gpdg).any():
        raise ValueError(f"{path}: near-miss template muon without its generator row")
    g, dist, _ = T.closest_approach(m[["x", "y", "z"]].to_numpy(), m[["px", "py", "pz"]].to_numpy(), geom)
    m["group"], m["dist"], m["ke"] = g, dist, m.e.to_numpy() - T.MASS[13]
    m["tloc"] = np.arange(len(m))
    d = d.merge(m[KEY + ["prim_key", "tloc", "t"]], on=KEY + ["prim_key"])
    spdg = d.merge(parts.rename(columns={"key": "sp", "pdg": "spdg"})[KEY + ["sp", "spdg"]], on=KEY + ["sp"],
                   how="left").spdg.fillna(11).to_numpy().astype(np.int32)
    nms = pd.DataFrame({"tloc": d.tloc.to_numpy(), "sp": d.sp.to_numpy().astype(np.int64), "pdg": spdg,
                        "bar": d.bar.to_numpy().astype(np.int64), "dt0": (d.t0 - d.t).to_numpy(),
                        "dt1": (d.t1 - d.t).to_numpy(),
                        **{c: d[c].to_numpy().astype(float) for c in T.UV_STEP_COLS + ["px0", "py0", "pz0", "p1",
                                                                                        "len", "edep"]}})
    agg = nms.groupby("tloc").agg(nsteps=("edep", "size"), edep=("edep", "sum"))
    nmt = m[KEY + ["prim_key", "tloc", "group", "dist", "ke"]].join(agg, on="tloc")
    return x, mu, vis, visits, nmt, nms


def next_crossing(mu, t0, offset=0):
    """Per crossing, the id (offset + row) of the next crossing in time of the same muon, -1 for its last."""
    n = len(mu)
    order = np.lexsort((np.arange(n), t0, mu))
    nxt = np.full(n, -1, np.int64)
    same = mu[order][1:] == mu[order][:-1]
    nxt[order[:-1][same]] = order[1:][same] + offset
    return nxt


def census_pass(steps_path, gen_path, geom):
    """Generated muons of the events without a G4 primary-muon crossing: (group, dist, ke) per muon, and events."""
    steps, parts = S._read(steps_path, KEY + ["bar", "sp", "edep"])
    s = classify(steps, parts, geom.sector)
    crossed = pd.MultiIndex.from_frame(s.loc[s.cls.eq("mu_primary"), KEY].drop_duplicates().astype(np.int64))
    gen = tables.read(gen_path, "gen", KEY + ["idx", "pdg", "x", "y", "z", "px", "py", "pz", "e", "t"])
    mu = T.muon_rays(gen)
    nc = mu[~pd.MultiIndex.from_frame(mu[KEY].astype(np.int64)).isin(crossed)]
    g, dist, _ = T.closest_approach(nc[["x", "y", "z"]].to_numpy(), nc[["px", "py", "pz"]].to_numpy(), geom)
    events = len(genassoc.generator_event_keys(gen_path))
    return pd.DataFrame({"group": g, "dist": dist, "ke": nc.ke.to_numpy()}), events, len(mu)


def sidecar(manifest, census_steps, census_gen, census_shards, out, counters=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tick = time.monotonic()
    man = json.loads(Path(manifest).read_text())
    library = str(Path(manifest).with_suffix(".root"))
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    from mu2efastsim import tables as core_tables
    lx = core_tables.read(library, "libxing")
    bins = core_tables.read(library, "libbins")
    if not np.array_equal(lx.xid.to_numpy(), np.arange(len(lx))):
        raise ValueError("library crossings are not ordered contiguous ids")
    xs, cross, visits, nmts, nmss, records = [], [], [], [], [], []
    offset = muoff = toff = 0
    for f in man["inputs"]:
        start = time.monotonic()
        x, mu, vis, v, nmt, nms = donor_pass(f, geom)
        z = lx.iloc[offset:offset + len(x)]
        np.testing.assert_array_equal(z.bin, T._bin(bins, T._group_index(geom, x.bar), x.p.to_numpy(),
                                                     x.len.to_numpy()))
        if "ctx" in z.columns:
            np.testing.assert_array_equal(z.ctx, T.crossing_context(x, geom))
        np.testing.assert_allclose(z[["ux", "uy", "uz"]], T._unit(x[["px0", "py0", "pz0"]].to_numpy()), rtol=1e-7,
                                   atol=1e-12)
        xs.append(pd.DataFrame({"xid": np.arange(offset, offset + len(x)), "bar": x.bar.to_numpy().astype(np.int64),
                                "ex": x.x0.to_numpy(), "ey": x.y0.to_numpy(), "ez": x.z0.to_numpy(),
                                "p": x.p.to_numpy(), "last": T.last_crossing(x).astype(np.int64),
                                "mu": mu.astype(np.int64) + muoff, "qx": x.qx.to_numpy(), "qy": x.qy.to_numpy(),
                                "qz": x.qz.to_numpy(), "len": x.len.to_numpy(),
                                "nxt": next_crossing(mu, x.t0.to_numpy(), offset)}))
        cross.append(mu + muoff)
        visits.append(v.assign(mu=v.mu.to_numpy().astype(np.int64) + muoff))
        nmts.append(nmt.assign(tid=nmt.tloc + toff))
        nmss.append(nms.assign(tid=nms.tloc + toff))
        nmu = int(mu.max() + 1) if len(mu) else 0
        records.append({"path": f, "sha256": A.digest(f), "crossings": len(x), "muons": nmu,
                        "nearmiss_templates": len(nmt), "nearmiss_steps": len(nms),
                        "nearmiss_edep": float(nms.edep.sum()), "seconds": time.monotonic() - start})
        offset += len(x)
        muoff += nmu
        toff += len(nmt)
        print(json.dumps(records[-1]), flush=True)
    if offset != len(lx):
        raise ValueError("library crossing count differs from the donor reconstruction")
    xing = pd.concat(xs, ignore_index=True)
    mu_all = np.concatenate(cross)
    vis_all = pd.concat(visits, ignore_index=True)
    nmt = pd.concat(nmts, ignore_index=True).drop(columns="tloc")
    nms = pd.concat(nmss, ignore_index=True).drop(columns="tloc")
    if len(nms) != int(man.get("desc_steps_unattached", len(nms))):
        raise ValueError(f"near-miss steps {len(nms)} differ from the library's unattached descendant steps "
                         f"{man.get('desc_steps_unattached')}")
    # unvisited-layer rows: other-bar library rows in a layer plane the crossing's own muon never visits
    ld = core_tables.read(library, "libdesc")
    lid, _ = T.layer_slots(geom)
    other = np.flatnonzero(ld.samebar.to_numpy() == 0)
    xid = ld.xid.to_numpy()[other]
    tgt = geom.center[xing.bar.to_numpy()[xid]] + ld[["dcx", "dcy", "dcz"]].to_numpy()[other]
    db = T.bar_containing(tgt, geom)
    if (db < 0).any() or not np.allclose(tgt, geom.center[np.maximum(db, 0)], rtol=0, atol=1e-6):
        raise ValueError("library other-bar offset is not an exact bar centre offset")
    vkey = np.unique(vis_all.mu.to_numpy().astype(np.int64) * 1_000_000 + vis_all.lid.to_numpy().astype(np.int64))
    uv = ~np.isin(mu_all[xid].astype(np.int64) * 1_000_000 + lid[db], vkey)
    uvf = pd.DataFrame({"row": other[uv], "bar": db[uv]})
    grp = np.asarray(T.GROUPS)[T._group_index(geom, db[uv])]
    uv_census = (pd.DataFrame({"grp": grp, "e": ld.edep.to_numpy()[other[uv]]}).groupby("grp").e
                 .agg(["size", "sum"]).reset_index())
    # near-miss census
    cens, cev, cmu, crec = [], 0, 0, []
    if isinstance(census_steps, str):
        census_steps, census_gen, census_shards = [census_steps], [census_gen], [census_shards]
    if not (len(census_steps) == len(census_gen) == len(census_shards)):
        raise ValueError("census steps, gen and shards must be given the same number of times")
    sources = [(st, ge, k) for st, ge, ks in zip(census_steps, census_gen, census_shards) for k in ks]
    paths = [st.format(s=S.shard(k), k=k) for st, _, k in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("repeated census table")
    for cs, cg, k in sources:
        sp, gp = (p.format(s=S.shard(k), k=k) for p in (cs, cg))
        c, ev, nm = census_pass(sp, gp, geom)
        cens.append(c)
        cev += ev
        cmu += nm
        crec.append({"steps": sp, "gen": gp, "steps_sha256": A.digest(sp), "gen_sha256": A.digest(gp), "events": ev,
                     "muons": nm, "no_crossing_muons": len(c)})
        print(json.dumps(crec[-1]), flush=True)
    nmc = pd.concat(cens, ignore_index=True)
    generated = man.get("generated_scaled_to") or man.get("generated")
    if not generated:
        raise ValueError("library manifest has no generated exposure (generated_scaled_to)")
    payload = {**{f"xing_{k}": xing[k].to_numpy() for k in xing.columns},
               **{f"uv_{k}": uvf[k].to_numpy() for k in uvf.columns},
               **{f"nmt_{k}": nmt[k].to_numpy() for k in nmt.columns},
               **{f"nms_{k}": nms[k].to_numpy() for k in nms.columns},
               **{f"nmc_{k}": nmc[k].to_numpy() for k in nmc.columns}}
    np.savez_compressed(out, **payload)
    result = {"library": library, "library_sha256": A.digest(library), "manifest": str(manifest),
              "geometry_sha256": A.digest(geometry.find_counters(counters or S.COUNTERS)),
              "code_sha256": {p: A.digest(p) for p in (__file__, T.__file__)}, "donors": records, "census": crec,
              "generated": int(generated), "census_events": int(cev), "census_muons": int(cmu),
              "census_no_crossing_muons": int(len(nmc)), "nearmiss_templates": int(len(nmt)),
              "nearmiss_steps": int(len(nms)), "nearmiss_edep": float(nms.edep.sum()),
              "other_bar_rows": int(len(other)), "unvisited_rows": int(uv.sum()),
              "unvisited_energy": float(ld.edep.to_numpy()[other[uv]].sum()),
              "unvisited_by_group": json.loads(uv_census.to_json(orient="records")),
              "sidecar_sha256": A.digest(out), "seconds": time.monotonic() - tick}
    Path(str(out) + ".json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("nearmiss_templates", "nearmiss_steps", "census_events",
                                             "census_no_crossing_muons", "unvisited_rows", "unvisited_energy",
                                             "seconds")}), flush=True)
    return result


def read_sidecar(path, library=None):
    from crvfastsim.analyses.tier1b import read_template_sidecar
    return read_template_sidecar(path, library)


def cells(path, library=None, counters=None):
    """Reach table of the near-miss cells and occupancy of the unvisited-layer position cells."""
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    side = read_sidecar(path, library)
    t, c = side["nmt"], side["nmc"]
    tc = T.nearmiss_cell(t.ke.to_numpy(), t.group.to_numpy(), t.dist.to_numpy())
    cc = T.nearmiss_cell(c.ke.to_numpy(), c.group.to_numpy(), c.dist.to_numpy())
    scale = side["meta"]["generated"] / side["meta"]["census_events"]
    tt = pd.DataFrame({"cell": tc, "e": t.edep.to_numpy(), "n": t.nsteps.to_numpy()}).groupby("cell").agg(
        ntemp=("e", "size"), steps=("n", "sum"), edep=("e", "sum"))
    r = pd.concat([pd.Series(cc).value_counts().rename("ncensus"), tt], axis=1).fillna(0)
    r["nden"] = r.ncensus * scale
    r["prob"] = np.where(r.nden > 0, r.ntemp / np.maximum(r.nden, 1e-300), np.nan)
    r.index = [f"{T.GROUPS[i // 100]} d{(i // 10) % 10} k{i % 10}" for i in r.index]
    print(r.round(5).to_string())
    lost = r[(r.nden == 0) & (r.ntemp > 0)]
    print(f"templates in cells without census muons: {int(lost.ntemp.sum())} ({lost.edep.sum():.1f} MeV of "
          f"{tt.edep.sum():.1f})")
    xs = side["xing"]
    mask = T.group_mask(xs.mu.to_numpy(), xs.bar.to_numpy(), geom)
    cl = T.uv_cells(xs.bar.to_numpy(), xs[["ex", "ey", "ez"]].to_numpy(), xs.p.to_numpy(), xs["last"].to_numpy(), mask,
                    geom)
    for j in range(T.N_UV_LEVELS):
        n = pd.Series(cl[:, j]).value_counts()
        print(f"uv level {j}: cells {len(n)}, crossings in cells >= {T.MIN_UV_CROSSINGS}: "
              f"{n[n >= T.MIN_UV_CROSSINGS].sum() / n.sum():.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sidecar")
    s.add_argument("--manifest", required=True, help="library JSON (its inputs are the donor tables)")
    s.add_argument("--census-steps", required=True, action="append",
                   help="G4 CrvStep tables of the census shards ({s} pattern); repeat with --census-gen and "
                        "--census-shards for several table sets")
    s.add_argument("--census-gen", required=True, action="append", help="full gen streams of the same shards")
    s.add_argument("--census-shards", required=True, action="append", help="e.g. 1-10 or 21,23,25 (S.shards)")
    s.add_argument("--out", required=True, help="new .npz")
    s.add_argument("--counters")
    c = sub.add_parser("cells")
    c.add_argument("sidecar")
    c.add_argument("--library")
    c.add_argument("--counters")
    a = p.parse_args()
    if a.cmd == "sidecar":
        sidecar(a.manifest, a.census_steps, a.census_gen, [S.shards(x) for x in a.census_shards], a.out, a.counters)
    else:
        cells(a.sidecar, a.library, a.counters)


if __name__ == "__main__":
    main()
