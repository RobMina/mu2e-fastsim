"""tier1b-build / tier1b-apply -- the G4 library of muon descendants and non-muon primaries (crvfastsim/tier1b.py).

    tier1b-build crvsteps.<train>.root [...] --gen gen.<train>.root [...] --out lib.root
    tier1b-apply fastC.*.root --library lib.root --outdir DIR [--seed N]   (writes fast1b.<shard>.root)
        [--crossing-context --contain-steps --point-placement --topology-sidecar SIDE.npz]   (opt-in placement)
        [--template-sidecar TEMPLATES.npz --unvisited-layers --near-miss]   (opt-in templates)
        [--chord-placement --follow-muon]   (opt-in same-bar placement and chain draw; template sidecar v3)

Build on training shards and apply to others (the library never sees the events it is compared with). The fast
tables carry their gen tree, which supplies the non-muon CRY primaries.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .. import geometry, tables, tier1b

DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def _geom(counters):
    return geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))


def build(inputs, outdir, gen=None, out=None, generated=None, counters=None, **kw):
    if not gen or not out:
        raise SystemExit("tier1b-build needs --gen and --out")
    lib, info = tier1b.build(tables.expand(inputs), tables.expand(gen), _geom(counters), generated=generated)
    tier1b.write_library(out, lib)
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump({"inputs": tables.expand(inputs), "gen": tables.expand(gen), **info}, fh, indent=2)
    print(info)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_topology_sidecar(path, library=None):
    """(crossings, visits) of a topology sidecar (valhist.descendant_topology sidecar): per library crossing its bar,
    muon and visit, and the library muons' visits. With library, require the library it was made for."""
    meta = json.loads(open(str(path) + ".json").read())
    if _sha256(path) != meta["sidecar_sha256"]:
        raise ValueError("topology sidecar changed since it was written")
    if library is not None and _sha256(library) != meta["library_sha256"]:
        raise ValueError("topology sidecar was made for another library")
    z = np.load(path)
    part = lambda pre: pd.DataFrame({k[len(pre):]: z[k] for k in z.files if k.startswith(pre)})
    return part("crossings_"), part("visits_")


TEMPLATE_PARTS = ("xing", "uv", "nmt", "nms", "nmc")


def read_template_sidecar(path, library=None):
    """Frames of a template sidecar (valhist.descendant_templates sidecar) and its metadata under 'meta': per library
    crossing its entry, |p| and last flag, the unvisited-layer rows, the near-miss templates and the census. With
    library, require the library it was made for."""
    meta = json.loads(open(str(path) + ".json").read())
    if _sha256(path) != meta["sidecar_sha256"]:
        raise ValueError("template sidecar changed since it was written")
    if library is not None and _sha256(library) != meta["library_sha256"]:
        raise ValueError("template sidecar was made for another library")
    z = np.load(path)
    part = lambda pre: pd.DataFrame({k[len(pre):]: z[k] for k in z.files if k.startswith(pre)})
    return {**{k: part(k + "_") for k in TEMPLATE_PARTS}, "meta": meta}


def apply(inputs, outdir, library=None, seed=1, counters=None, crossing_context=False, contain_steps=False,
          point_placement=False, mirror_offsets=False, translate_offsets=False, topology_sidecar=None,
          template_sidecar=None, unvisited_layers=False, near_miss=False, chord_placement=False, follow_muon=False,
          **kw):
    if not library:
        raise SystemExit("tier1b-apply needs --library")
    if (unvisited_layers or near_miss) and not template_sidecar:
        raise SystemExit("--unvisited-layers / --near-miss need --template-sidecar")
    opts = {"context": crossing_context, "contain": contain_steps, "point": point_placement, "mirror": mirror_offsets,
            "translate": translate_offsets, "topology": bool(topology_sidecar)}
    if unvisited_layers or near_miss:
        opts.update(unvisited=bool(unvisited_layers), nearmiss=bool(near_miss))
    if (chord_placement or follow_muon) and not template_sidecar:
        raise SystemExit("--chord-placement / --follow-muon need --template-sidecar (with library chords)")
    if chord_placement or follow_muon:
        opts.update(chord=bool(chord_placement), follow=bool(follow_muon))
    geom = _geom(counters)
    lib = tier1b.read_library(library)
    if topology_sidecar:
        lib = tier1b.attach_topology(lib, *read_topology_sidecar(topology_sidecar, library), geom)
    if template_sidecar:
        lib = tier1b.attach_templates(lib, read_template_sidecar(template_sidecar, library), geom)
    for i, f in enumerate(tables.expand(inputs)):
        rng = np.random.default_rng([int(seed), 1_000 + i])
        st, pa, gen = tables.read(f, "steps"), tables.read(f, "particles"), tables.read(f, "gen")
        sub = tables.read(f, "subruns")
        steps, parts, info = tier1b.sample(st, pa, gen, lib, geom, rng, **opts)
        name = os.path.basename(f).replace("fastC.", "fast1b.").replace("fastB.", "fast1b.").replace("fastA.", "fast1b.")
        tables.write(os.path.join(outdir, name), steps, parts, gen, sub, as_ttree=True)
        print(f"# {f}: {info}", flush=True)


GROUPS = ("EX", "T1", "T2", "M")
EXTRA_Q = (0.5, 0.9, 0.99)


def _per_crossing(paths, geom):
    """Per primary-muon crossing: muon visible energy and the descendants' visible energy in the same bar; plus the
    descendant energy outside the muon's bars and the non-muon energy, per sector group (totals)."""
    from .census import SECTOR_GROUP, classify
    steps = tables.read_many(paths, "steps", tables.KEY + ["bar", "sp", "edep"])
    parts = tables.read_many(paths, "particles", tables.KEY + ["key", "parent", "pdg"])
    parts["isprim"] = (parts.parent < 0).astype(int)
    s = classify(steps, parts, geom.sector)
    s["group"] = SECTOR_GROUP[geom.sector[s.bar.to_numpy()]]
    mu = s[s.cls == "mu_primary"].groupby(tables.KEY + ["bar"]).edep.sum().rename("mu").reset_index()
    d = s[s.cls == "mu_descendant"].groupby(tables.KEY + ["bar"]).edep.sum().rename("extra").reset_index()
    x = mu.merge(d, on=tables.KEY + ["bar"], how="left").fillna({"extra": 0.0})
    x["group"] = SECTOR_GROUP[geom.sector[x.bar.to_numpy()]]
    out = d.merge(mu[tables.KEY + ["bar"]].assign(m=1), on=tables.KEY + ["bar"], how="left")
    out = out[out.m.isna()]
    out["group"] = SECTOR_GROUP[geom.sector[out.bar.to_numpy()]]
    nm = s[s.cls.str.startswith("nonmu")]
    return x, out.groupby("group").extra.sum(), nm.groupby("group").edep.sum(), \
        nm.groupby(tables.KEY + ["bar"]).edep.sum().reset_index().assign(group=lambda f: SECTOR_GROUP[geom.sector[f.bar.to_numpy()]])


def _desc_geometry(paths, geom):
    """Muon-descendant steps: local midpoint |offset|/half-length in thickness, width and length, step length, |p|."""
    from .census import classify
    steps = tables.read_many(paths, "steps", tables.KEY + ["bar", "sp", "edep", "len", "x0", "y0", "z0", "x1", "y1",
                                                          "z1", "px0", "py0", "pz0"])
    parts = tables.read_many(paths, "particles", tables.KEY + ["key", "parent", "pdg"])
    s = classify(steps, parts, geom.sector)
    d = s[s.cls == "mu_descendant"]
    b = d.bar.to_numpy()
    mid = 0.5 * (d[["x0", "y0", "z0"]].to_numpy() + d[["x1", "y1", "z1"]].to_numpy())
    rel = np.abs(mid - geom.center[b]) / geom.halflen[b]
    ax = geom.local_to_world[b]
    r = np.arange(len(b))
    seg = np.linalg.norm(d[["x1", "y1", "z1"]].to_numpy() - d[["x0", "y0", "z0"]].to_numpy(), axis=1)
    return {"thick": rel[r, ax[:, 0]], "width": rel[r, ax[:, 1]], "length": rel[r, ax[:, 2]], "seg": seg,
            "len": d.len.to_numpy(), "p": np.linalg.norm(d[["px0", "py0", "pz0"]].to_numpy(), axis=1),
            "edep": d.edep.to_numpy()}


def check(inputs, outdir, compare=None, counters=None, **kw):
    """tier1b-check FAST1B.root [...] --compare crvsteps.root [...]: step-level tier-1b closure per sector group."""
    from mu2efastsim import plots
    if not compare:
        raise SystemExit("tier1b-check needs --compare <G4 CrvStep tables of the same (held-out) shards>")
    geom = _geom(counters)
    xf, of, nf, bf = _per_crossing(tables.expand(inputs), geom)
    xr, orr, nr, br = _per_crossing(tables.expand(compare), geom)
    res = {}
    for g in GROUPS:
        a, b = xf[xf.group == g], xr[xr.group == g]
        if not len(b):
            continue
        res[g] = {"crossings_fast": len(a), "crossings_ref": len(b),
                  "extra_over_muon_fast": float(a.extra.sum() / a.mu.sum()), "extra_over_muon_ref": float(b.extra.sum() / b.mu.sum()),
                  "frac_with_extra_fast": float((a.extra > 0).mean()), "frac_with_extra_ref": float((b.extra > 0).mean()),
                  "extra_q_fast": [float(v) for v in np.quantile(a.extra, EXTRA_Q)],
                  "extra_q_ref": [float(v) for v in np.quantile(b.extra, EXTRA_Q)],
                  "bar_total_q_fast": [float(v) for v in np.quantile(a.mu + a.extra, EXTRA_Q)],
                  "bar_total_q_ref": [float(v) for v in np.quantile(b.mu + b.extra, EXTRA_Q)],
                  "desc_outside_muon_bars_per_crossing_fast": float(of.get(g, 0.0) / len(a)),
                  "desc_outside_muon_bars_per_crossing_ref": float(orr.get(g, 0.0) / len(b)),
                  "nonmu_edep_fast_over_ref": float(nf.get(g, 0.0) / max(nr.get(g, 0.0), 1e-9)),
                  "nonmu_bar_hits_fast_over_ref": float((bf.group == g).sum() / max((br.group == g).sum(), 1))}
    gf, gr = _desc_geometry(tables.expand(inputs), geom), _desc_geometry(tables.expand(compare), geom)
    geo = {}
    for k in gf:
        w = {"fast": gf, "ref": gr}
        geo[k] = {a: {"q10_50_90": [float(v) for v in np.quantile(w[a][k], (0.1, 0.5, 0.9))],
                      "edep_weighted_mean": float(np.average(w[a][k], weights=np.maximum(w[a]["edep"], 0)))}
                  for a in w}
    for a, w in (("fast", gf), ("ref", gr)):
        geo.setdefault("on_face_frac", {})[a] = float(((w["thick"] >= 1 - 1e-6) | (w["width"] >= 1 - 1e-6)).mean())
        geo.setdefault("seg_over_len_q10", {})[a] = float(np.quantile(w["seg"] / np.maximum(w["len"], 1e-9), 0.1))
    res["desc_geometry"] = geo
    plots.write_json(outdir, "tier1b_check.json", res)
    for g, r in res.items():
        if g not in GROUPS:
            continue
        print(f"{g:3s} extra/mu {r['extra_over_muon_ref']:.3f}->{r['extra_over_muon_fast']:.3f}  with extra "
              f"{r['frac_with_extra_ref']:.3f}->{r['frac_with_extra_fast']:.3f}  extra q50/90/99 "
              f"{np.round(r['extra_q_ref'], 2)}->{np.round(r['extra_q_fast'], 2)}  bar total q50/90/99 "
              f"{np.round(r['bar_total_q_ref'], 2)}->{np.round(r['bar_total_q_fast'], 2)}  desc outside/xing "
              f"{r['desc_outside_muon_bars_per_crossing_ref']:.3f}->{r['desc_outside_muon_bars_per_crossing_fast']:.3f}  "
              f"nonmu edep f/r {r['nonmu_edep_fast_over_ref']:.3f}, bar hits f/r {r['nonmu_bar_hits_fast_over_ref']:.3f}")
    print("descendant steps (ref -> fast): " + "; ".join(
        f"{k} q50 {geo[k]['ref']['q10_50_90'][1]:.3g}->{geo[k]['fast']['q10_50_90'][1]:.3g} "
        f"wmean {geo[k]['ref']['edep_weighted_mean']:.3g}->{geo[k]['fast']['edep_weighted_mean']:.3g}"
        for k in ("thick", "width", "length", "seg", "len", "p")))
    print(f"  midpoint on a bar face: {geo['on_face_frac']['ref']:.4f} -> {geo['on_face_frac']['fast']:.4f}; "
          f"chord/path q10 {geo['seg_over_len_q10']['ref']:.3f} -> {geo['seg_over_len_q10']['fast']:.3f}")
    return res
