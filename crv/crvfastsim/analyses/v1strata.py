"""V1 strata: where a fast/G4 difference in primary-muon bar crossings sits, per sector group and per crossing variable.

Read-only truth diagnostic on persisted tables (no transport is run): gen tables, the G4 reference CrvStep tables and the
fast candidate tables of the same generated events, one file of each per shard (shards disjoint in run/subrun).

Per crossing (event, gen index, bar; CrvSteps summed, earliest step's entry point and direction) and arm:
  sector, layer, module-edge counter, entry face (thickness / width / end), |position along the bar|, incidence on the
  thickness axis |u_t|, |u| along the bar, entry |p|, crossing rank of the muon and whether it crossed another
  module before. Each arm is binned by its own values, so crossings migrate between bins: spectra, not efficiencies.

Per stratum the output is the crossing-count shift, the path-length and visible-energy mean shifts, with standard errors
clustered by generated event (every generated event, zeros included; within-event correlations kept): for the
ratio of means R = (S_a/N_a)/(S_b/N_b), var(log R) = n/(n-1) sum_i z_i^2 with z_i = s_ai/S_a - n_ai/N_a - s_bi/S_b +
n_bi/N_b. Shards add sums and cross products, never errors.

Binned-ratio shapes (--v1-stats: the merged v1compare_stats.json of the same reference): v1compare's path-length and
visible-energy ratios, 20 bins over the reference 5-95 % range, normalised to each arm's histogram total, with the
bin variances clustered by generated event (same formula, value = bin indicator, N = the arm's histogram total) instead
of v1compare's independent-crossing (Poisson) errors; chi2/ndf both ways, and with the full clustered covariance.

Direction diagnostics (histograms per arm): the angle between a crossing's entry direction and its generated direction,
by whether the muon crossed another module before; and the direction change between two consecutive crossings of one
muon in the same module (the scattering inside a module stack; zero for a transport with frozen directions).

    python3 -m crvfastsim.analyses.v1strata --gen 'G/gen.*.root' --reference 'R/crvsteps.*.root' \
        --candidate 'C/*/tables/fastC.*.root' --config run2_v10 --outdir OUT
    python3 -m crvfastsim.analyses.v1strata --merge 'OUT/shard_*.json' --outdir OUT   # summary.json + plots
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import runconfig, tables
from crvfastsim.analyses import transport_validation as V

SCHEMA = "crv-v1-strata-v1"
KEY = tables.KEY + ["genidx", "bar"]
COMPONENTS = ["len_a", "n_a", "len_b", "n_b", "edep_a", "edep_b"]
BINS = {
    "layer": [0, 1, 2, 3, 4],
    "edge_counter": [0, 1, 2],                    # 0 interior counter, 1 first/last counter of a module layer
    "face": [0, 1, 2, 3],                         # entry face: 0 thickness, 1 width, 2 end
    "along": [0.0, 0.5, 0.8, 0.9, 0.95, 1.0001],  # |entry position along the bar| / half length
    "incidence": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001],   # |u . thickness axis|
    "u_long": [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0001],                # |u . bar axis|
    "p": [0.0, 300.0, 600.0, 1000.0, 2000.0, 5000.0, 1e9],
    "rank": [0, 1, 2, 4, 8, 1000],                # crossings of the muon before this one
    "prev_other": [0, 1, 2],                      # 1: the muon crossed a bar of another module before
}
BELOW_BINS = [0.0, 1e-4, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9]   # x/X0 below the MSC plane on the entry ray (--gdml)
BELOW_CLASSES = (0.05, 0.5)                                    # angle histograms: < 0.05, 0.05-0.5, > 0.5 x/X0
ANGLE_EDGES = np.r_[np.linspace(0.0, 100.0, 401), np.inf]   # mrad
FACE_TOL_MM = 0.01


def groups_of(rc, geom):
    cfg = rc.counters_cfg()
    out = {"group:" + name: [i for i, s in enumerate(geom.sector_names) if int(cfg[f"crs.sectorType{s}"]) == t]
           for t, name in runconfig.sector_type_names(geom, cfg).items()}
    out.update({name: [i] for i, name in enumerate(geom.sector_names)})
    return out


def arm_crossings(path, gen, geom, fast_format):
    """Primary-muon crossings with entry geometry and history columns (see module docstring)."""
    x = V.primary_crossings(path, gen, geom, fast_format=fast_format)
    steps, parts = tables.read(path, "steps"), tables.read(path, "particles")
    prim = parts.loc[(parts.isprim == 1) & (parts.pdg.abs() == 13)]
    if "genidx" in prim:
        ident = prim[tables.KEY + ["key", "genidx"]]
    else:
        gm = gen.loc[gen.pdg.abs() == 13].sort_values(tables.KEY + ["idx"], kind="stable").copy()
        gm["key"] = gm.groupby(tables.KEY, sort=False).cumcount() + 1
        ident = prim[tables.KEY + ["key"]].merge(gm[tables.KEY + ["key", "idx"]].rename(columns={"idx": "genidx"}),
                                                 on=tables.KEY + ["key"], validate="one_to_one")
    s = steps.merge(ident.rename(columns={"key": "sp"}), on=tables.KEY + ["sp"], validate="many_to_one")
    first = (s.sort_values(KEY + ["t0"], kind="stable").drop_duplicates(KEY)
             [KEY + ["x0", "y0", "z0", "px0", "py0", "pz0"]])
    x = x.merge(first, on=KEY, validate="one_to_one")
    g = gen.loc[gen.pdg.abs() == 13, tables.KEY + ["idx", "px", "py", "pz"]].rename(
        columns={"idx": "genidx", "px": "gx", "py": "gy", "pz": "gz"})
    x = x.merge(g, on=tables.KEY + ["genidx"], validate="many_to_one")
    mu = tables.KEY + ["genidx"]
    x = x.sort_values(mu + ["t0"], kind="stable").reset_index(drop=True)
    bar = x.bar.to_numpy(int)
    c, h, ax = geom.center[bar], geom.halflen[bar], geom.local_to_world[bar]
    q = x[["x0", "y0", "z0"]].to_numpy() - c
    d = x[["px0", "py0", "pz0"]].to_numpy()
    d = d / np.linalg.norm(d, axis=1)[:, None]
    gd = x[["gx", "gy", "gz"]].to_numpy()
    gd = gd / np.linalg.norm(gd, axis=1)[:, None]
    take = lambda a, k: np.take_along_axis(a, ax[:, k:k + 1], 1)[:, 0]   # noqa: E731
    gap = h - np.abs(q)                                  # distance of the entry point to each face pair
    face_axis = np.argmin(np.where(gap < -FACE_TOL_MM, np.inf, gap), 1)
    x["face"] = np.select([face_axis == ax[:, 0], face_axis == ax[:, 1]], [0, 1], 2)
    x["along"] = np.abs(take(q, 2)) / take(h, 2)
    x["incidence"] = np.abs(take(d, 0))
    x["u_long"] = np.abs(take(d, 2))
    ncnt = pd.Series(geom.counter).groupby(geom.sector).transform("max").to_numpy()
    x["layer"] = geom.layer[bar]
    x["edge_counter"] = ((geom.counter[bar] == 0) | (geom.counter[bar] == ncnt[bar])).astype(int)
    x["module_id"] = geom.sector[bar].astype(np.int64) * 1000 + geom.module[bar]
    x["rank"] = x.groupby(mu, sort=False).cumcount()
    new_module = (~x.duplicated(mu + ["module_id"])).astype(int)
    x["prev_other"] = (new_module.groupby([x[k] for k in mu], sort=False).cumsum() > 1).astype(int).to_numpy()
    x["angle_gen"] = 1e3 * np.arccos(np.clip((d * gd).sum(1), -1.0, 1.0))
    ids = x[mu].to_numpy()
    same = np.r_[False, (ids[1:] == ids[:-1]).all(1) & (x.module_id.to_numpy()[1:] == x.module_id.to_numpy()[:-1])]
    turn = np.full(len(x), np.nan)
    turn[same] = 1e3 * np.arccos(np.clip((d[1:] * d[:-1]).sum(1)[same[1:]], -1.0, 1.0))
    x["turn_in_module"] = turn
    return x


def _event_vectors(xa, xb, ev_a, ev_b, n):
    """[6, n] per-event sums of the COMPONENTS for one stratum."""
    out = np.zeros((len(COMPONENTS), n))
    out[0] = np.bincount(ev_a, weights=xa.length.to_numpy(), minlength=n)
    out[1] = np.bincount(ev_a, minlength=n)
    out[2] = np.bincount(ev_b, weights=xb.length.to_numpy(), minlength=n)
    out[3] = np.bincount(ev_b, minlength=n)
    out[4] = np.bincount(ev_a, weights=xa.edep.to_numpy(), minlength=n)
    out[5] = np.bincount(ev_b, weights=xb.edep.to_numpy(), minlength=n)
    return out


def _moments(v):
    return {"totals": v.sum(1).tolist(), "cross": (v @ v.T).tolist()}


def shape_edges(v1_group, col):
    """The 21 edges of v1compare's 20-bin ratio over the group's reference range (v1compare_stats.json)."""
    from crvfastsim.analyses import v1compare
    fine = v1compare.EDGES[col]
    lo, hi = v1_group[f"{col}_ratio"]["range"]
    lo_i, hi_i = int(np.argmin(np.abs(fine - lo))), int(np.argmin(np.abs(fine - hi)))
    parts = [g for g in np.array_split(np.arange(lo_i, hi_i), 20) if len(g)]
    return np.r_[[fine[g[0]] for g in parts], fine[parts[-1][-1] + 1]], (float(fine[0]), float(fine[-1]))


def _shape_vectors(va, vb, ev_a, ev_b, n, edges, full):
    """[2 (k + 1), n] per-event counts: candidate bins, candidate histogram total, reference bins, reference total."""
    k = len(edges) - 1
    out = np.zeros((2 * (k + 1), n))
    for j, (v, ev) in enumerate(((va, ev_a), (vb, ev_b))):
        inside = (v >= full[0]) & (v < full[1])
        i = np.digitize(v, edges) - 1
        sel = inside & (i >= 0) & (i < k)
        np.add.at(out, (j * (k + 1) + i[sel], ev[sel]), 1.0)
        out[j * (k + 1) + k] = np.bincount(ev[inside], minlength=n)
    return out


def summarise_shape(m, events, min_ref=400):
    """Binned ratio r_k = (A_k / A) / (B_k / B) with Poisson and event-clustered errors, chi2/ndf both ways."""
    t, M = np.asarray(m["totals"]), np.asarray(m["cross"])
    k = len(t) // 2 - 1
    A, B = t[k], t[2 * k + 1]
    f = events / (events - 1.0)
    use = [j for j in range(k) if t[k + 1 + j] >= min_ref and t[j] > 0]
    if not use or A <= 0 or B <= 0:
        return {"valid": False, "nbins_used": len(use)}
    C = np.zeros((len(use), len(t)))
    for r, j in enumerate(use):
        C[r, j], C[r, k], C[r, k + 1 + j], C[r, 2 * k + 1] = 1 / t[j], -1 / A, -1 / t[k + 1 + j], 1 / B
    ratio = np.array([(t[j] / A) / (t[k + 1 + j] / B) for j in use])
    cov = f * C @ M @ C.T                                   # covariance of log r
    se_cl = np.sqrt(np.clip(np.diag(cov), 0, None))
    se_po = np.sqrt(np.array([1 / t[j] + 1 / t[k + 1 + j] for j in use]))
    lr = np.log(ratio)
    try:
        full = float(lr @ np.linalg.solve(cov, lr) / len(use))
    except np.linalg.LinAlgError:
        full = None
    return {"valid": True, "nbins_used": len(use), "ratio": ratio.tolist(), "se_clustered": se_cl.tolist(),
            "se_poisson": se_po.tolist(), "chi2_ndf_poisson": float(np.mean(((ratio - 1) / (ratio * se_po)) ** 2)),
            "chi2_ndf_clustered": float(np.mean(((ratio - 1) / (ratio * se_cl)) ** 2)),
            "chi2_ndf_clustered_cov": full, "design_effect_median": float(np.median((se_cl / se_po) ** 2))}


def below_xx0(x, below, y_plane):
    """x/X0 of the below-plane material (transport.load_below) on the straight ray from the MSC plane to each entry."""
    from mu2efastsim import eloss, materials
    P = x[["x0", "y0", "z0"]].to_numpy()
    u = x[["px0", "py0", "pz0"]].to_numpy()
    u = u / np.linalg.norm(u, axis=1)[:, None]
    down = u[:, 1] < -1e-9
    s = np.where(down, (y_plane - P[:, 1]) / -np.where(down, u[:, 1], -1.0), 0.0)
    s = np.clip(s, 0.0, None)
    out = np.zeros(len(x))
    ok = np.flatnonzero(s > 0)
    if len(ok):
        for m, v in below.column(P[ok] - u[ok] * s[ok, None], u[ok], s[ok]).items():
            out[ok] += v / eloss.radiation_length_mm(materials.BY_NAME.get(m, materials.CONCRETE_MARS))
    return out


def shard(gen_path, reference, candidate, config, gdml=None, v1_stats=None, below_margin=None):
    """Stratum moments and direction histograms of one shard (candidate = arm a, G4 reference = arm b)."""
    rc = runconfig.get(config)
    geom = rc.geometry()
    bins = dict(BINS)
    gen = tables.read(gen_path, "gen")
    universe = gen[tables.KEY].drop_duplicates().reset_index(drop=True)
    universe["ev"] = np.arange(len(universe))
    xa = arm_crossings(candidate, gen, geom, True)
    xb = arm_crossings(reference, gen, geom, False)
    if gdml:
        from crvfastsim import transport as TR
        below = TR.load_below(gdml, geom, **({} if below_margin is None else {"margin": below_margin}))
        y_plane = TR.msc_plane(geom)
        for f in (xa, xb):
            f["below_xx0"] = below_xx0(f, below, y_plane)
            f["below_class"] = np.digitize(f.below_xx0.to_numpy(), BELOW_CLASSES)
        bins["below_xx0"] = BELOW_BINS
    xa = xa.merge(universe, on=tables.KEY, validate="many_to_one")
    xb = xb.merge(universe, on=tables.KEY, validate="many_to_one")
    n = len(universe)
    groups = groups_of(rc, geom)
    strata, angles, shapes = {}, {}, {}
    v1g = json.loads(Path(v1_stats).read_text())["groups"] if v1_stats else {}
    for gname, secs in groups.items():
        a, b = xa[xa.sector.isin(secs)], xb[xb.sector.isin(secs)]
        rows = {"all": _moments(_event_vectors(a, b, a.ev.to_numpy(), b.ev.to_numpy(), n))}
        for var, edges in (bins.items() if gname.startswith("group:") else ()):
            ia = np.digitize(a[var].to_numpy(), edges) - 1
            ib = np.digitize(b[var].to_numpy(), edges) - 1
            for k in range(len(edges) - 1):
                ma, mb = ia == k, ib == k
                rows[f"{var}:{k}"] = _moments(_event_vectors(a[ma], b[mb], a.ev.to_numpy()[ma], b.ev.to_numpy()[mb], n))
        strata[gname] = rows
        if not gname.startswith("group:"):
            continue
        if gname[6:] in v1g:
            shapes[gname] = {}
            for col, var in (("len", "length"), ("edep", "edep")):
                edges, full = shape_edges(v1g[gname[6:]], col)
                shapes[gname][col] = dict(_moments(_shape_vectors(
                    a[var].to_numpy(), b[var].to_numpy(), a.ev.to_numpy(), b.ev.to_numpy(), n, edges, full)),
                    edges=edges.tolist())
        h = {}
        for arm, f in (("candidate", a), ("reference", b)):
            for po in (0, 1):
                h[f"{arm}:angle_gen:prev_other{po}"] = np.histogram(
                    f.angle_gen[f.prev_other == po].to_numpy(), ANGLE_EDGES)[0].tolist()
            if "below_class" in f:
                for bc in range(len(BELOW_CLASSES) + 1):
                    h[f"{arm}:angle_gen:below{bc}"] = np.histogram(
                        f.angle_gen[f.below_class == bc].to_numpy(), ANGLE_EDGES)[0].tolist()
            t = f.turn_in_module.to_numpy()
            h[f"{arm}:turn_in_module"] = np.histogram(t[np.isfinite(t)], ANGLE_EDGES)[0].tolist()
        angles[gname] = h
    return {"schema": SCHEMA, "run_config": config, "events": n,
            "event_ranges": universe.groupby(["run", "subrun"]).event.agg(["min", "max", "size"]).reset_index()
            .to_dict("records"),
            "inputs": {k: V.fingerprint(p) for k, p in (("gen", gen_path), ("reference", reference),
                                                        ("candidate", candidate))},
            "diagnostic_sha256": V.fingerprint(__file__)["sha256"], "bins": {k: list(map(float, v)) for k, v in
                                                                            bins.items()},
            "strata": strata, "angles": angles, "shapes": shapes}


def _add(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            _add(dst.setdefault(k, {}), v)
        else:
            dst[k] = (np.asarray(dst[k]) + np.asarray(v)).tolist() if k in dst else v


def summarise(m, events):
    """Shifts and clustered SEs of one stratum's merged moments (None where an arm is empty)."""
    t, M = np.asarray(m["totals"]), np.asarray(m["cross"])
    La, Na, Lb, Nb, Ea, Eb = t
    f = events / (events - 1.0)
    out = {"n_candidate": int(Na), "n_reference": int(Nb)}

    def ratio(c, value):
        se = float(abs(value) * np.sqrt(max(f * (c @ M @ c), 0.0)))
        return float(value - 1.0), se
    if Nb > 0 and Na > 0:
        out["crossings_rel"], out["crossings_se"] = ratio(np.array([0, 1 / Na, 0, -1 / Nb, 0, 0]), Na / Nb)
        out["len_mean_rel"], out["len_mean_se"] = ratio(np.array([1 / La, -1 / Na, -1 / Lb, 1 / Nb, 0, 0]),
                                                        (La / Na) / (Lb / Nb))
        out["edep_mean_rel"], out["edep_mean_se"] = ratio(np.array([0, -1 / Na, 0, 1 / Nb, 1 / Ea, -1 / Eb]),
                                                          (Ea / Na) / (Eb / Nb))
        out["len_mean_candidate"], out["len_mean_reference"] = float(La / Na), float(Lb / Nb)
        out["len_total_rel"], out["len_total_se"] = ratio(np.array([1 / La, 0, -1 / Lb, 0, 0, 0]), La / Lb)
    return out


def _quantiles(h, qs=(0.5, 0.68, 0.9)):
    h = np.asarray(h, float)
    c = np.cumsum(h)
    if not c[-1]:
        return [None] * len(qs)
    centres = 0.5 * (ANGLE_EDGES[:-2] + ANGLE_EDGES[1:-1])
    return [float(centres[min(np.searchsorted(c, q * c[-1]), len(centres) - 1)]) for q in qs]


def merge(paths, outdir):
    rows = [json.loads(Path(p).read_text()) for p in tables.expand(paths)]
    if not rows or len({(r["schema"], r["run_config"], r["diagnostic_sha256"]) for r in rows}) != 1:
        raise ValueError("merge requires shards of one schema, configuration and diagnostic")
    ids = [(e["run"], e["subrun"]) for r in rows for e in r["event_ranges"]]
    if len(set(ids)) != len(ids):
        raise ValueError("merge requires disjoint run/subrun shards")
    events = sum(r["events"] for r in rows)
    strata, angles, shapes = {}, {}, {}
    for r in rows:
        _add(strata, r["strata"])
        _add(angles, r["angles"])
        _add(shapes, {g: {c: {k: v for k, v in m.items() if k != "edges"} for c, m in x.items()}
                      for g, x in r.get("shapes", {}).items()})
    edges = {g: {c: m["edges"] for c, m in x.items()} for g, x in rows[0].get("shapes", {}).items()}
    if any({g: {c: m["edges"] for c, m in x.items()} for g, x in r.get("shapes", {}).items()} != edges for r in rows):
        raise ValueError("merge requires shards with the same --v1-stats shape edges")
    out = {"schema": SCHEMA, "run_config": rows[0]["run_config"], "events": events, "shards": len(rows),
           "event_ranges": [e for r in rows for e in r["event_ranges"]], "bins": rows[0]["bins"],
           "diagnostic_sha256": rows[0]["diagnostic_sha256"],
           "inputs": [r["inputs"] for r in rows],
           "strata": {g: {s: summarise(m, events) for s, m in v.items()} for g, v in strata.items()},
           "angles_mrad_quantiles_50_68_90": {g: {k: _quantiles(h) for k, h in v.items()} for g, v in angles.items()},
           "shapes": {g: {c: dict(summarise_shape(m, events), edges=edges[g][c]) for c, m in x.items()}
                      for g, x in shapes.items()},
           "angles_hist": angles, "angle_edges_mrad": [float(e) for e in ANGLE_EDGES[:-1]] + ["inf"],
           "moments": strata}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=1, allow_nan=False) + "\n"
    with (outdir / "summary.json").open("x") as f:
        f.write(text)
    print_table(out)
    plot(out, outdir)
    return out


def print_table(out, groups=None):
    groups = groups or [g for g in out["strata"] if g.startswith("group:")]
    for g in groups:
        print(f"== {g}  ({out['run_config']}, {out['events']} generated events)")
        for s, v in out["strata"][g].items():
            if "len_mean_rel" not in v:
                print(f"  {s:16s} nc {v['n_candidate']:8d} nr {v['n_reference']:8d}")
                continue
            print(f"  {s:16s} nc {v['n_candidate']:8d} nr {v['n_reference']:8d}  xing {100 * v['crossings_rel']:+6.2f}"
                  f"+-{100 * v['crossings_se']:4.2f}  len {100 * v['len_mean_rel']:+6.2f}+-{100 * v['len_mean_se']:4.2f}"
                  f"  edep {100 * v['edep_mean_rel']:+6.2f}+-{100 * v['edep_mean_se']:4.2f}"
                  f"  sumlen {100 * v['len_total_rel']:+6.2f}+-{100 * v['len_total_se']:4.2f}")
        for col, sh in out.get("shapes", {}).get(g, {}).items():
            if sh.get("valid"):
                print(f"  ratio {col:5s} chi2/ndf Poisson {sh['chi2_ndf_poisson']:5.2f}  clustered "
                      f"{sh['chi2_ndf_clustered']:5.2f}  clustered cov {sh['chi2_ndf_clustered_cov'] or float('nan'):5.2f}"
                      f"  ({sh['nbins_used']} bins, median design effect {sh['design_effect_median']:.2f})")
        if g in out["angles_mrad_quantiles_50_68_90"]:
            for k, q in out["angles_mrad_quantiles_50_68_90"][g].items():
                print(f"  {k:40s} q50/68/90 mrad {q}")


def v1_table(v1_stats, strata_summary):
    """One row per V1 group: the v1compare verdict quantities next to this diagnostic's event-clustered SEs.

    Crossing shift: v1compare's (event, bar) count per event of the union; the SE is that of the identity-preserving
    count here (the two counts differ by multi-muon events in one bar). Mean SEs from the same clustered moments."""
    v1 = json.loads(Path(v1_stats).read_text())
    st = json.loads(Path(strata_summary).read_text())
    th = v1["thresholds"]
    print(f"V1 {v1_stats}: overall {'PASS' if v1['pass'] else 'FAIL'}; thresholds |xing| {100 * th['crossings_per_event_rel']}%"
          f", |len mean| {100 * th['len_mean_rel']}%, |edep mean|, |MPV| {100 * th['edep_mean_rel']}%, chi2/ndf <="
          f" {th['ratio_chi2_ndf']}")
    print(f"{'group':8s} {'n_G4':>8s} {'xing %':>14s} {'len mean %':>14s} {'edep mean %':>14s} {'MPV %':>14s}"
          f" {'len chi2/ndf':>13s} {'edep chi2/ndf':>13s}  verdict")
    for g, x in v1["groups"].items():
        s = st["strata"].get("group:" + g, {}).get("all", {})
        mpv = x.get("edep_mpv_rel")
        cell = lambda v, e: f"{100 * v:+6.2f}+-{100 * e:4.2f}" if v is not None and e is not None else f"{'n/a':>13s}"  # noqa: E731
        chk = x["checks"]
        fails = [k for k, v in chk.items() if not v]
        print(f"{g:8s} {x['n_ref']:8d} {cell(x['crossings_per_event_rel'], s.get('crossings_se'))}"
              f" {cell(x['len_mean_rel'], s.get('len_mean_se'))} {cell(x['edep_mean_rel'], s.get('edep_mean_se'))}"
              f" {cell(mpv, x.get('edep_mpv_rel_err'))}"
              f" {x['len_ratio']['chi2_ndf'] or float('nan'):8.2f} ({x['len_ratio']['nbins_used']:2d})"
              f" {x['edep_ratio']['chi2_ndf'] or float('nan'):8.2f} ({x['edep_ratio']['nbins_used']:2d})"
              f"  {'PASS' if x['pass'] else 'FAIL ' + ','.join(fails)}")


def plot(out, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [g for g in out["strata"] if g.startswith("group:")]
    for var, edges in out["bins"].items():
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for g in groups:
            rows = [out["strata"][g].get(f"{var}:{k}", {}) for k in range(len(edges) - 1)]
            xs = np.arange(len(rows))
            for ax, key in zip(axs, ("crossings", "len_mean", "len_total")):
                y = np.array([100 * r.get(f"{key}_rel", np.nan) for r in rows])
                e = np.array([100 * r.get(f"{key}_se", np.nan) for r in rows])
                ok = np.isfinite(y) & (e < 20)
                ax.errorbar(xs[ok] + 0.04 * groups.index(g), y[ok], e[ok], fmt="o", ms=3, label=g[6:])
        for ax, key in zip(axs, ("crossings", "path-length mean", "summed path length")):
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(np.arange(len(edges) - 1))
            ax.set_xticklabels([f"{edges[k]:g}-{edges[k + 1]:g}" for k in range(len(edges) - 1)], rotation=45,
                               fontsize=7)
            ax.set_xlabel(var)
            ax.set_ylabel(f"fast/G4 - 1, {key} [%]")
        axs[0].legend(fontsize=7)
        fig.suptitle(f"{out['run_config']}: fast vs G4 by {var} ({out['events']} generated events)")
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"strata_{var}.png", dpi=110)
        plt.close(fig)
    centres = np.asarray(out["angle_edges_mrad"][:-2], float)
    for g, h in out["angles_hist"].items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, v in h.items():
            v = np.asarray(v[:-1], float)
            if v.sum():
                ax.step(centres, v / v.sum(), where="post", label=k, lw=1)
        ax.set_xlim(0, 60)
        ax.set_yscale("log")
        ax.set_xlabel("angle [mrad]")
        ax.legend(fontsize=6)
        ax.set_title(f"{out['run_config']} {g}: entry direction vs generated; turn inside a module")
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"angles_{g[6:]}.png", dpi=110)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir")
    p.add_argument("--merge", nargs="+")
    p.add_argument("--gen", nargs="+")
    p.add_argument("--reference", nargs="+")
    p.add_argument("--candidate", nargs="+")
    p.add_argument("--config")
    p.add_argument("--gdml", help="GDML world: adds the below-plane x/X0 strata (transport.load_below)")
    p.add_argument("--v1-stats", help="merged v1compare_stats.json of the same reference: adds the binned-ratio shapes")
    p.add_argument("--below-margin", type=float, default=None, help="with --gdml: load_below margin [mm]")
    p.add_argument("--summary", help="print the table of an existing summary.json")
    p.add_argument("--v1-table", nargs=2, metavar=("V1_STATS", "STRATA_SUMMARY"),
                   help="V1 verdict table: v1compare_stats.json and this diagnostic's merged summary.json")
    a = p.parse_args(argv)
    if a.v1_table:
        v1_table(*a.v1_table)
        return
    if a.summary:
        print_table(json.loads(Path(a.summary).read_text()))
        return
    if a.merge:
        merge(a.merge, a.outdir)
        return
    gens, refs, cands = (tables.expand(v) for v in (a.gen, a.reference, a.candidate))
    if not (len(gens) == len(refs) == len(cands)) or not gens:
        raise SystemExit("--gen, --reference and --candidate need the same number of files (one per shard)")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for g, r, c in zip(gens, refs, cands):
        res = shard(g, r, c, a.config, a.gdml, a.v1_stats, a.below_margin)
        sub = "_".join(f"{e['subrun']}" for e in res["event_ranges"])
        text = json.dumps(res, allow_nan=False) + "\n"
        with (outdir / f"shard_{sub}.json").open("x") as f:
            f.write(text)
        print(f"shard {sub}: {res['events']} events -> {outdir}/shard_{sub}.json", flush=True)


if __name__ == "__main__":
    main()
