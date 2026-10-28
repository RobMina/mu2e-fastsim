"""Tier-1b placement arms against Geant4 steps on the same generated events: ValCrvStep closure and descendant topology.

Per shard, each arm runs crvfastsim.tier1b.sample on the retained stage-C transport table with the tier1b-apply seed
rule (seed = shard, stream [shard, 1000]), so `default` equals `tier1b-apply` output; a reseeded copy of an arm
(stream [shard, 1000, 7]) gives the sampling floor of the comparison. Nothing is written unless --write is given.
The reference is the G4 CrvStep table of the same generated events (its refill equals the art A0g histograms).

    python -m valhist.placement_closure run --library LIB.root [--sidecar SIDE.npz] \
        --transport 'DIR/fastC.{s}.root' --reference 'DIR/crvsteps.{s}.root' --shards 21-220 \
        --arm default= --arm ccp=context,contain,point --arm topo=context,contain,point,topology \
        [--templates TEMPLATES.npz --arm tmpl=context,contain,point,topology,unvisited,nearmiss] \
        --reseed topo --out NEW.json
    python -m valhist.placement_closure report RUN.json [RUN2.json ...]

`report` sums runs over disjoint shards and prints, per arm against G4 and against its reseed, the deck statistic
(chi2/ndf, Poisson variance; per-sector y/z with the exact clustered variance), the exact clustered and paired
statistics, and the yields with their clustered errors.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import geometry, tables, tier1b as T
from . import descendant_audit as A
from . import descendant_topology as D
from . import stephist as H
from . import steplayers as S

OPTIONS = ("context", "contain", "point", "mirror", "translate", "topology", "unvisited", "nearmiss")
RESEED = 7
REPORT_KEYS = list(H.TARGET) + ["SMom2", "posx"] + [f"{g}/{v}" for g in S.POS_GROUPS for v in ("y", "z")]


def parse_arm(spec):
    """NAME=OPT,OPT[@N]: tier1b options of an arm and its seed tag (0: the tier1b-apply stream [shard, 1000];
    N > 0: an independent stream [shard, 1000, N] of the same arm, for sampling floors)."""
    name, _, rest = spec.partition("=")
    opts, _, tag = rest.partition("@")
    flags = [o for o in opts.split(",") if o]
    bad = set(flags) - set(OPTIONS)
    if bad:
        raise ValueError(f"unknown tier1b options {sorted(bad)}")
    return name, {**{o: (o in flags) for o in OPTIONS}, "seed_tag": int(tag or 0)}


def run(library, transport, reference, shard_list, arms, out, sidecar=None, reseed=(), write=None, retained=None,
        counters=None, templates=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if len(set(shard_list)) != len(shard_list):
        raise ValueError("repeated shard")
    tick = time.monotonic()
    geom = geometry.from_offline_config(geometry.find_counters(counters or S.COUNTERS))
    lib = T.read_library(library)
    hashes = {str(library): A.digest(library), T.__file__: A.digest(T.__file__), H.__file__: A.digest(H.__file__),
              __file__: A.digest(__file__)}
    if sidecar:
        lib = T.attach_topology(lib, *D.read_sidecar(sidecar, library), geom)
        hashes[str(sidecar)] = A.digest(sidecar)
    if any(o["topology"] for _, o in arms) and not sidecar:
        raise ValueError("topology arms need --sidecar")
    if templates:
        from . import descendant_templates as DT
        lib = T.attach_templates(lib, DT.read_sidecar(templates, library), geom)
        hashes[str(templates)] = A.digest(templates)
    if any(o.get("unvisited") or o.get("nearmiss") for _, o in arms) and not templates:
        raise ValueError("unvisited / nearmiss arms need --templates")
    edges = S.booking_edges()
    runs = [(name, opts, opts.get("seed_tag", 0)) for name, opts in arms]
    runs += [(f"{name}~reseed", opts, RESEED) for name, opts in arms if name in reseed]
    hist = {name: {} for name, _, _ in runs}
    hist["g4"] = {}
    pair = {name: {} for name, _, _ in runs}
    floor = {name: {} for name in reseed}
    topo = {name: {} for name, _, _ in runs}
    why, records = {name: {} for name, _, _ in runs}, []
    for k in shard_list:
        t0 = time.monotonic()
        f, r = (p.format(s=S.shard(k), k=k) for p in (transport, reference))
        fx, fp, gen = tables.read(f, "steps"), tables.read(f, "particles"), tables.read(f, "gen")
        events = gen[tables.KEY].drop_duplicates()
        rs, rp = S._read(r, H.STEP_COLS)
        hr, ir = H.fill(rs, rp, events, edges)
        H.add(hist["g4"], hr)
        tr = D.topo_labels(rs, rp, geom)
        rec = {"shard": k, "transport": f, "reference": r, "events": len(events), "g4_steps": len(rs), "arms": {}}
        hashes.update({f: A.digest(f), r: A.digest(r)})
        idx = {}
        for name, opts, tag in runs:
            base = name.split("~")[0]
            rng = np.random.default_rng([k, 1000, tag] if tag else [k, 1000])
            steps, parts, info = T.sample(fx, fp, gen, lib, geom, rng,
                                          **{o: v for o, v in opts.items() if o in OPTIONS})
            if retained and name in retained:
                ref_path = retained[name].format(s=S.shard(k), k=k)
                old = tables.read(ref_path, "steps")
                pd.testing.assert_frame_equal(steps.reset_index(drop=True), old[steps.columns].reset_index(drop=True),
                                              check_dtype=False, check_exact=True)
                rec.setdefault("retained_exact", []).append(name)
                hashes[ref_path] = A.digest(ref_path)
            if write and name in write:
                d = Path(write[name])
                d.mkdir(parents=True, exist_ok=True)
                dest = d / Path(f).name.replace("fastC.", "fast1b.")
                if dest.exists():
                    raise FileExistsError(dest)
                tables.write(str(dest), steps, parts, gen, tables.read(f, "subruns"), as_ttree=True)
            hc, ic = H.fill(steps[H.STEP_COLS], parts, events, edges)
            H.add(hist[name], hc)
            idx[name] = ic
            H.add_paired(pair[name], H.paired(ir, ic, edges))
            topo[name] = S._merge_views(topo[name], {v: S.accumulate(tr, D.topo_labels(steps, parts, geom), cols)
                                                     for v, cols in D.VIEWS.items()})
            for key, val in info["other_bar"].items():
                why[name][key] = why[name].get(key, 0) + int(val)
            rec["arms"][name] = {"steps": len(steps), "desc_steps": info["desc_steps"], "desc_dropped": info["desc_dropped"],
                                 "showers": info["showers"], "shower_steps": info["shower_steps"]}
            if "templates" in info:
                rec["arms"][name]["templates"] = info["templates"]
            if name.endswith("~reseed"):
                H.add_paired(floor[base], H.paired(idx[base], ic, edges))
        rec["seconds"] = time.monotonic() - t0
        records.append(rec)
        print(json.dumps({"shard": k, "seconds": round(rec["seconds"], 2), "g4": len(rs),
                          **{n: a["steps"] for n, a in rec["arms"].items()}}), flush=True)
    result = {"schema": 1, "library": str(library), "sidecar": sidecar, "templates": templates,
              "arms": {n: o for n, o in arms},
              "reseed": list(reseed), "seed_rule": "[shard,1000]; reseed [shard,1000,7]", "shards": shard_list,
              "hashes": hashes, "records": records, "placement": why,
              "hists": {n: H.to_json(h) for n, h in hist.items()},
              "paired": {n: {k: v.tolist() for k, v in p.items()} for n, p in pair.items()},
              "floor_paired": {n: {k: v.tolist() for k, v in p.items()} for n, p in floor.items()},
              "topology": {n: {v: f.to_dict(orient="list") for v, f in views.items()} for n, views in topo.items()},
              "uncertainty": "event-clustered moments conditional on the fixed library, transport and reference",
              "seconds": time.monotonic() - tick}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, allow_nan=False)
    print(f"wrote {out}; {result['seconds']:.1f} s", flush=True)
    return result


# ------------------------------------------------------------------------------------------------------------ report
def _join(runs):
    """Runs over the same shards with disjoint arms as one run (their G4 histograms must agree)."""
    base = dict(runs[0])
    for r in runs[1:]:
        if r["shards"] != base["shards"] or r["library"] != base["library"]:
            raise ValueError("joined runs differ in shards or library")
        if set(r["arms"]) & set(base["arms"]):
            raise ValueError("joined runs repeat an arm")
        if r["hists"]["g4"] != base["hists"]["g4"]:
            raise ValueError("joined runs differ in the G4 reference")
        for k in ("arms", "hists", "paired", "floor_paired", "topology", "placement"):
            base[k] = {**base[k], **{n: v for n, v in r[k].items() if not (k == "hists" and n == "g4")}}
    return base


def load(paths):
    """Sum runs over disjoint shards (same arms and library); 'A+B' joins runs over the same shards."""
    runs = [_join([json.loads(Path(q).read_text()) for q in p.split("+")]) for p in paths]
    shards = [k for r in runs for k in r["shards"]]
    if len(shards) != len(set(shards)):
        raise ValueError("runs overlap in shards")
    first = runs[0]
    for r in runs[1:]:
        if r["arms"] != first["arms"] or r["library"] != first["library"]:
            raise ValueError("runs differ in arms or library")
    hist = {n: {} for n in first["hists"]}
    pair, floor, topo = {n: {} for n in first["paired"]}, {n: {} for n in first["floor_paired"]}, {}
    for r in runs:
        for n, h in r["hists"].items():
            H.add(hist[n], H.from_json(h))
        for n, p in r["paired"].items():
            H.add_paired(pair[n], {k: np.asarray(v) for k, v in p.items()})
        for n, p in r["floor_paired"].items():
            H.add_paired(floor[n], {k: np.asarray(v) for k, v in p.items()})
        for n, views in r["topology"].items():
            topo[n] = S._merge_views(topo.get(n, {}), {v: pd.DataFrame(f) for v, f in views.items()})
    return first, shards, hist, pair, floor, topo


def summary(paths):
    first, shards, hist, pair, floor, topo = load(paths)
    out = {"shards": sorted(shards), "arms": first["arms"], "vs_g4": {}, "vs_reseed": {}, "topology": {}}
    for n in pair:
        out["vs_g4"][n] = H.compare(hist["g4"], hist[n], pair[n])
    for n in floor:
        out["vs_reseed"][n] = H.compare(hist[n], hist[f"{n}~reseed"], floor[n])
    out["floors"] = {}
    names = list(first["arms"])
    strip = lambda o: {k: v for k, v in o.items() if k != "seed_tag"}
    for i, a in enumerate(names):          # same options, other seed: the sampling floor (independent variance)
        for b in names[i + 1:]:
            if strip(first["arms"][a]) == strip(first["arms"][b]):
                out["floors"][f"{b}_vs_{a}"] = H.compare(hist[a], hist[b])
                out["vs_reseed"].setdefault(a, out["floors"][f"{b}_vs_{a}"])
    for n, views in topo.items():
        out["topology"][n] = A.summarize(views)
    out["components"] = {n: {c: H.compare({k: {"all": v[c]} for k, v in hist["g4"].items() if c in v and k in REPORT_KEYS},
                                         {k: {"all": v[c]} for k, v in hist[n].items() if c in v and k in REPORT_KEYS})
                             for c in H.COMPONENTS[1:]} for n in pair}
    return out


def print_components(sm, keys=("bar", "posx", "all/y", "all/z", "EX/y", "EX/z", "T1/y", "T1/z", "T2/y", "T2/z", "M/y",
                                "M/z")):
    """Exact clustered chi2/ndf of each step origin alone (primary muon, muon descendant, non-muon) against G4."""
    print("\nper origin vs G4, exact clustered chi2/ndf (yield rel. difference of that origin in the histogram range)")
    for n, comps in sm["components"].items():
        print(f"== {n}")
        for k in keys:
            print(f"{k:>8} " + "  ".join(f"{c:>5} {comps[c][k]['exact'] or 0:6.2f}" for c in comps if k in comps[c]))


def _fmt(x, w=7, p=2):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and x is not None else f"{'-':>{w}}"


def print_summary(sm):
    names = list(sm["vs_g4"])
    print(f"shards {sm['shards'][0]}-{sm['shards'][-1]} ({len(sm['shards'])})")
    print("deck chi2/ndf vs G4 (flag > 2), [exact clustered | paired], reseed floor deck:")
    print(f"{'histogram':>12} " + " ".join(f"{n:>28}" for n in names))
    for k in REPORT_KEYS:
        cells = []
        for n in names:
            r = sm["vs_g4"][n][k]
            fl = sm["vs_reseed"].get(n, {}).get(k, {}).get("deck")
            cells.append(f"{_fmt(r['deck'])}{'*' if r['flag'] else ' '}[{_fmt(r['exact'], 5)}|{_fmt(r.get('paired'), 5)}]"
                         f"{(' f' + _fmt(fl, 4, 1)) if fl is not None else '':>7}")
        print(f"{k:>12} " + " ".join(f"{c:>28}" for c in cells))
    if sm.get("floors"):
        print("\nsampling floors (same arm, independent seeds): deck [exact clustered]")
        print(f"{'histogram':>12} " + " ".join(f"{n:>22}" for n in sm["floors"]))
        for k in REPORT_KEYS:
            print(f"{k:>12} " + " ".join(f"{_fmt(f[k]['deck'])} [{_fmt(f[k]['exact'], 5)}]".rjust(22)
                                         for f in sm["floors"].values()))
    print("\nstep yield vs G4: rel diff, clustered independent SE, paired SE (all, then per component)")
    for g in S.POS_GROUPS:
        key = f"_total/{g}"
        for n in names:
            r = sm["vs_g4"][n][key]
            comp = "  ".join(f"{c} {100 * v['rel']:+.2f}+-{100 * v['rel_se_independent']:.2f}"
                             for c, v in r["components"].items() if v["rel"] is not None)
            print(f"{g:>4} {n:>14} {100 * r['rel']:+6.2f} % +- {100 * r['rel_se_independent']:.2f} (ind) "
                  f"+- {100 * r.get('rel_se_paired', float('nan')):.2f} (pair)   {comp}")


def print_census(sm, arms=None):
    """Muon descendants per sector and topology label: G4 steps and energy, arm/G4 energy ratio with SE."""
    names = arms or list(sm["topology"])
    rows = {}
    for n in names:
        for r in sm["topology"][n]["topo"]:
            key = (r["grp"], r["topo"])
            rows.setdefault(key, {"g4": (r["n_ref"], r["e_ref"])})[n] = (r["e_ratio"], r["e_ratio_se"])
    print("\nmuon descendants: sector, label: G4 steps, G4 energy [MeV], arm/G4 energy (SE)")
    for key in sorted(rows, key=lambda k: (k[0], -rows[k]["g4"][1])):
        cells = "  ".join(f"{n} {v[0]:.3f}({v[1]:.3f})" if v[0] is not None else f"{n} -"
                          for n, v in rows[key].items() if n != "g4")
        print(f"{key[0]:>3} {key[1]:>12} {rows[key]['g4'][0]:9.0f} {rows[key]['g4'][1]:10.1f}  {cells}")
    print("other-bar energy (every label but own and no_crossing), arm/G4; no SE (labels share events):")
    for g in sorted({k[0] for k in rows}):
        keys = [k for k in rows if k[0] == g and k[1] not in ("own", "no_crossing")]
        ref = sum(rows[k]["g4"][1] for k in keys)
        cells = "  ".join(f"{n} {sum(rows[k]['g4'][1] * (rows[k][n][0] or 0) for k in keys) / ref:.3f}" for n in names)
        print(f"{g:>3} {ref:10.1f}  {cells}")


def print_layers(sm, arms=None):
    """Muon-descendant steps per sector, layer rank and topology label: G4 count and arm/G4 count ratio with SE."""
    names = arms or list(sm["topology"])
    rows = {}
    for n in names:
        for r in sm["topology"][n]["lay_topo"]:
            key = (r["grp"], int(r["lay"]), r["topo"])
            rows.setdefault(key, {"g4": r["n_ref"]})[n] = (r["n_ratio"], r["n_ratio_se"])
    print("\nmuon descendants: sector, layer rank (0 top), label: G4 steps, arm/G4 (SE)")
    for key in sorted(rows, key=lambda k: (k[0], k[1], -rows[k]["g4"])):
        if rows[key]["g4"] < 500:
            continue
        cells = "  ".join(f"{n} {v[0]:.3f}({v[1]:.3f})" if v[0] is not None else f"{n} -"
                          for n, v in rows[key].items() if n != "g4")
        print(f"{key[0]:>3} {key[1]} {key[2]:>12} {rows[key]['g4']:9.0f}  {cells}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for name in ("library", "transport", "reference", "out"):
        r.add_argument("--" + name, required=True)
    r.add_argument("--sidecar")
    r.add_argument("--templates", help="template sidecar (valhist.descendant_templates) for unvisited / nearmiss arms")
    r.add_argument("--shards", required=True)
    r.add_argument("--arm", action="append", required=True, metavar="NAME=OPT,OPT")
    r.add_argument("--reseed", action="append", default=[], metavar="ARM")
    r.add_argument("--write", action="append", default=[], metavar="ARM=DIR")
    r.add_argument("--retained", action="append", default=[], metavar="ARM=PATTERN",
                   help="require this arm's steps to equal a retained table (exact)")
    r.add_argument("--counters")
    s = sub.add_parser("report")
    s.add_argument("runs", nargs="+")
    s.add_argument("--json", help="also write the summary here (new file)")
    s.add_argument("--layers", help="comma-separated arms: print the descendant census per layer and label")
    a = p.parse_args()
    if a.cmd == "run":
        arms = [parse_arm(x) for x in a.arm]
        if len({n for n, _ in arms}) != len(arms) or any("~" in n or n == "g4" for n, _ in arms):
            raise SystemExit("arm names must be unique, without '~', not 'g4'")
        if set(a.reseed) - {n for n, _ in arms}:
            raise SystemExit("--reseed names an unknown arm")
        run(a.library, a.transport, a.reference, S.shards(a.shards), arms, a.out, a.sidecar, a.reseed,
            dict(x.split("=", 1) for x in a.write), dict(x.split("=", 1) for x in a.retained), a.counters,
            a.templates)
    else:
        sm = summary(a.runs)
        print_summary(sm)
        print_components(sm)
        if a.layers:
            print_census(sm, a.layers.split(","))
            print_layers(sm, a.layers.split(","))
        if a.json:
            if Path(a.json).exists():
                raise FileExistsError(a.json)
            Path(a.json).write_text(json.dumps(sm, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
