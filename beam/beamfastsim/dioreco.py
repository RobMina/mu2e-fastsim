"""DIO electrons through stock reconstruction, fast tracker steps against Geant4 steps on the same events (opt-in).

Both arms are the same stock DIO dts events. Each arm's first-passage tracker steps are injected into those events
and run through the same stock digitisation, LoopHelix reconstruction and EventNtuple, as for conversion electrons
(`cetracker` route):
- fast: cechain -> fieldsteps from the event's generated electron;
- Geant4: its steps from the first tracker-entrance crossing.

Subcommands:

- `single`: copy StrawGasStepTableDumper tables without the events that hold more than one primary electron (a DIO
  event has one electron per stopped muon; about 0.08 % of events have two, and the one-primary hand-off and injector
  take one). Both arms then start from the same tables.
- `compare`: per dts event, the first fitted electron track and whether it passes the CE track selection (electron,
  downstream at the tracker front, fit consistency, tan(dip), d0 and r_max at the front); its reco momentum at the
  front. Per arm: the fraction of events with a fitted / selected track, the fraction with a selected track in each
  momentum bin and above P_CUTS. Every quantity is paired per event (difference of per-event indicators); yield
  ratios carry the paired error. Denominator: every event of the tables.
- `reduce`: the per-event frame of other EventNtuple files (e.g. the production reco of the same dts events), for use
  as an arm of `compare`;
  `shift_check` recounts the reference with its selected momenta shifted by the same-event mean difference, to test
  whether that shift alone explains the yield ratios on a steep spectrum;
- `plot`: the reco momentum spectrum above P_PLOT with a fast / Geant4 ratio panel.
"""
import argparse
import glob
import json
import os

import numpy as np

KEY = ["run", "subrun", "event"]
TREE = "EventNtuple/ntuple"
FRONT_SID = 0                                  # SurfaceId TT_Front
CE_CUTS = dict(tandip=(0.5774, 1.0), d0=(-80.0, 105.0), rmax=(450.0, 680.0), fitcon=1e-3)
P_EDGES = np.round(np.arange(95.0, 106.0 + 1e-9, 0.25), 3)
P_CUTS = (95.0, 100.0, 102.0, 103.0, 104.0)
P_PLOT = 95.0
PDG_E = 11


# ---- single -------------------------------------------------------------------------------------------------------

def multi_primary_events(particles):
    """KEY rows of the events with more than one primary electron (particles tree of a dumper table)."""
    p = particles[(particles.pdg == PDG_E) & ((particles.isprim == 1) | (particles.ismcprim == 1))]
    n = p.groupby(KEY).size()
    return n[n > 1].reset_index()[KEY]


def drop_events(frame, events):
    """frame without the rows of `events` (frames without event columns are returned unchanged)."""
    if not set(KEY) <= set(frame.columns) or events.empty:
        return frame
    m = frame.merge(events.assign(_drop=True), on=KEY, how="left")["_drop"].fillna(False).to_numpy(bool)
    return frame.loc[~m].reset_index(drop=True)


def single(src, dst):
    """Copy the dump/ trees of one dumper table without its multi-primary events; returns (events dropped, trees)."""
    import uproot
    from mu2efastsim import tables
    with uproot.open(src) as f:
        names = sorted({k.split("/")[1].split(";")[0] for k in f.keys() if k.startswith(f"{tables.DIR}/") and
                        k.count("/") == 1})
    frames = {n: tables.read(src, n) for n in names}
    bad = multi_primary_events(frames["particles"])
    tables.write(dst, {n: drop_events(df, bad) for n, df in frames.items()}, as_ttree=True)
    return len(bad), names


# ---- compare ------------------------------------------------------------------------------------------------------

def _vec(r):
    import awkward as ak
    c = r.fCoordinates
    return np.stack([ak.to_numpy(c.fX), ak.to_numpy(c.fY), ak.to_numpy(c.fZ)], 1).astype(float)


def read_ntuple(path):
    """Per ntuple entry: KEY, whether a fitted electron track exists, and the first one's front momentum, pz, pdg,
    fit consistency and LoopHelix parameters at the front (NaN without an in-bounds front crossing)."""
    import awkward as ak
    import pandas as pd
    import uproot
    with uproot.open(path) as f:
        t = f[TREE]
        a = t.arrays([f"evtinfo/{k}" for k in KEY] + [f"trk/trk.{k}" for k in ("status", "pdg", "fitcon",
                                                                                  "nactive")], library="ak")
        seg = t["trksegs"].array()
        pars = t["trksegpars_lh"].array()
    good = (a["trk/trk.status"] > 0) & (a["trk/trk.pdg"] == PDG_E)   # first fitted electron hypothesis
    has = ak.to_numpy(ak.any(good, axis=1))
    i = ak.argmax(good, axis=1, keepdims=True)
    s = ak.fill_none(ak.firsts(seg[i], axis=1), [], axis=0)
    pp = ak.fill_none(ak.firsts(pars[i], axis=1), [], axis=0)
    front = (s.sid == FRONT_SID) & s.inbounds
    j = ak.argmin(ak.where(front, s.time, np.inf), axis=1, keepdims=True)
    s0, p0 = ak.firsts(s[j]), ak.firsts(pp[j])
    infront = ak.to_numpy(ak.fill_none(ak.firsts(front[j]), False)) & has
    nan3 = {"fCoordinates": {"fX": np.nan, "fY": np.nan, "fZ": np.nan}}
    mom = _vec(ak.fill_none(s0.mom, nan3))
    df = pd.DataFrame({k: ak.to_numpy(a[f"evtinfo/{k}"]).astype(np.uint64) for k in KEY})
    df["fit"] = has
    df["p_reco"] = np.where(infront, np.linalg.norm(mom, axis=1), np.nan)
    df["pz"] = np.where(infront, mom[:, 2], np.nan)
    for k, src in (("tandip", "tanDip"), ("d0", "d0"), ("rmax", "maxr")):
        df[k] = np.where(infront, ak.to_numpy(ak.fill_none(p0[src], np.nan)).astype(float), np.nan)
    for k in ("pdg", "fitcon", "nactive"):
        df[k] = np.where(has, ak.to_numpy(ak.fill_none(ak.firsts(a[f"trk/trk.{k}"][i], axis=1), np.nan)), np.nan)
    if df.duplicated(KEY).any():
        raise ValueError(f"duplicate events in {path}")
    return df


def selected(df, cuts=CE_CUTS):
    c = df.fit & (df.pdg == PDG_E) & (df.pz > 0) & (df.fitcon > cuts["fitcon"])
    for k in ("tandip", "d0", "rmax"):
        lo, hi = cuts[k]
        c &= (df[k] >= lo) & (df[k] <= hi)
    return c.fillna(False).to_numpy(bool)


def read_arm(directory):
    """Per-event frame of every nts.*.root under a directory, or of a csv.gz written by `reduce`."""
    import pandas as pd
    if directory.endswith(".csv.gz"):
        d = pd.read_csv(directory)
        for k in KEY:
            d[k] = d[k].astype(np.uint64)
        d["fit"] = d["fit"].astype(bool)
        d["sel"] = selected(d)
        return d, int(d.attrs.get("files", 0))
    files = sorted(glob.glob(os.path.join(directory, "**", "nts.*.root"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no nts.*.root under {directory}")
    d = pd.concat([read_ntuple(f) for f in files], ignore_index=True)
    d["sel"] = selected(d)
    return d, len(files)


def reduce(paths, subruns=None):
    """read_ntuple over EventNtuple files (e.g. a production sample), keeping only the given subruns."""
    import pandas as pd
    parts = []
    for p in paths:
        d = read_ntuple(p)
        if subruns is not None:
            d = d[d.subrun.isin(np.asarray(subruns, np.uint64))]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    if d.duplicated(KEY).any():
        raise ValueError("duplicate events across the files")
    return d


def paired(a, b):
    """Paired comparison of per-event indicators (or values) a (arm) and b (reference) on the same events."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a.size
    d = a - b
    ma, mb = a.mean(), b.mean()
    r = ma / mb if mb > 0 else np.nan
    se = lambda x: float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return dict(n=int(n), arm=float(ma), arm_err=se(a), ref=float(mb), ref_err=se(b), diff=float(d.mean()),
                diff_err=se(d), ratio=float(r), ratio_err=se(a - r * b) / mb if mb > 0 else np.nan,
                arm_count=float(a.sum()), ref_count=float(b.sum()))


def compare(den, arm, ref, drop_outside=False):
    """den: DataFrame of KEY (every event); arm, ref: read_arm frames. Returns the report dict.

    drop_outside: drop (and count) arm rows outside den, e.g. a production sample that still holds the events `single`
    removed; otherwise they raise."""
    d = den[KEY].copy()
    outside = {}
    for name, f in (("arm", arm), ("ref", ref)):
        inside = f.merge(d[KEY], on=KEY)
        outside[name] = len(f) - len(inside)
        if outside[name] and not drop_outside:
            raise ValueError(f"{outside[name]} {name} events outside the denominator")
        f = inside
        d = d.merge(f.rename(columns={c: f"{name}_{c}" for c in f if c not in KEY}), on=KEY, how="left",
                    validate="one_to_one")
    fa, fr = (d[f"{n}_fit"].fillna(False).to_numpy(bool) for n in ("arm", "ref"))
    sa, sr = (d[f"{n}_sel"].fillna(False).to_numpy(bool) for n in ("arm", "ref"))
    pa = np.where(sa, d["arm_p_reco"].to_numpy(float), np.nan)
    pr = np.where(sr, d["ref_p_reco"].to_numpy(float), np.nan)
    out = dict(events=int(len(d)), cuts={k: list(v) if isinstance(v, tuple) else v for k, v in CE_CUTS.items()},
               efficiency_fit=paired(fa, fr), efficiency_selected=paired(sa, sr),
               gained=int((sa & ~sr).sum()), lost=int((~sa & sr).sum()), dropped_outside=outside, above={}, bins={})
    za, zr = np.nan_to_num(pa, nan=-1.0), np.nan_to_num(pr, nan=-1.0)
    for c in P_CUTS:
        out["above"][f"{c:g}"] = paired(za > c, zr > c)
    ia, ir = np.digitize(za, P_EDGES) - 1, np.digitize(zr, P_EDGES) - 1
    nb = P_EDGES.size - 1
    rows = [paired(ia == k, ir == k) for k in range(nb)]
    pull = [r["diff"] / r["diff_err"] if r["diff_err"] > 0 else 0.0 for r in rows]
    out["bins"] = dict(edges=P_EDGES.tolist(), arm=[r["arm"] for r in rows], ref=[r["ref"] for r in rows],
                       arm_err=[r["arm_err"] for r in rows], ref_err=[r["ref_err"] for r in rows],
                       arm_count=[r["arm_count"] for r in rows], ref_count=[r["ref_count"] for r in rows],
                       ratio=[r["ratio"] for r in rows], ratio_err=[r["ratio_err"] for r in rows], pull=pull)
    for lo in (95.0, 100.0):
        m = (P_EDGES[:-1] >= lo) & np.array([r["diff_err"] > 0 for r in rows])
        out["bins"][f"chi2_paired_above_{lo:g}"] = dict(chi2=float((np.array(pull)[m] ** 2).sum()), ndf=int(m.sum()))
    both = sa & sr & (za > P_PLOT) & (zr > P_PLOT)
    dp = pa[both] - pr[both]
    out["same_event_dp"] = dict(n=int(both.sum()), mean=float(dp.mean()) if dp.size else np.nan,
                                err=float(dp.std(ddof=1) / np.sqrt(dp.size)) if dp.size > 1 else np.nan,
                                median=float(np.median(dp)) if dp.size else np.nan)
    # does that mean shift alone explain the yield ratios? the reference's selected momenta shifted by it, recounted
    delta = out["same_event_dp"]["mean"]
    out["shift_check"] = dict(shift=delta, ratio_predicted={
        f"{c:g}": float((zr + delta > c).sum() / max((zr > c).sum(), 1)) for c in P_CUTS})
    return out


def numbers(rep, prod=None):
    """Headline numbers of a compare report (and of the Geant4-vs-production report, if given)."""
    k = ("arm", "arm_err", "ref", "ref_err", "diff", "diff_err", "ratio", "ratio_err", "arm_count", "ref_count")
    out = dict(events=rep["events"], dropped_outside=rep.get("dropped_outside"),
               efficiency_fit={x: rep["efficiency_fit"][x] for x in k},
               efficiency_selected={x: rep["efficiency_selected"][x] for x in k},
               above={c: {x: v[x] for x in k} for c, v in rep["above"].items()},
               same_event_dp=rep["same_event_dp"], shift_check=rep.get("shift_check"),
               chi2_paired={lo: rep["bins"][f"chi2_paired_above_{lo}"] for lo in ("95", "100")})
    if prod:
        out["g4_vs_production"] = dict(events=prod["events"], dropped_outside=prod.get("dropped_outside"),
                                       efficiency_selected={x: prod["efficiency_selected"][x] for x in k},
                                       above={c: {x: prod["above"][c][x] for x in k} for c in ("95", "100")})
    return out


# ---- plot ---------------------------------------------------------------------------------------------------------

def plot(rep, path, lo=P_PLOT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = rep["bins"]
    e = np.array(b["edges"])
    i0 = int(np.searchsorted(e, lo - 1e-9))
    k = slice(i0, None)
    ek = e[i0:]
    c = 0.5 * (ek[1:] + ek[:-1])
    w = ek[1:] - ek[:-1]
    fa, fr = np.array(b["arm"])[k], np.array(b["ref"])[k]
    er = np.array(b["ref_err"])[k]
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True, gridspec_kw=dict(height_ratios=(3, 1), hspace=0.05))
    ax[0].errorbar(c, np.where(fr > 0, fr, np.nan), yerr=er, fmt="o", ms=3, color="black", label="Geant4 tracker steps")
    ax[0].stairs(np.where(fa > 0, fa, np.nan), ek, color="#E69F00", lw=1.6,
                 label="fast tracker steps")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("selected tracks per event / 0.25 MeV/c")
    y = rep["above"]["100"]
    s = rep["efficiency_selected"]
    ax[0].set_title(f"DIO, stock digitisation + LoopHelix reco ({rep['events']:,} events)\n"
                    f"selected: fast / G4 = {s['ratio']:.4f} +- {s['ratio_err']:.4f};  above 100 MeV/c: "
                    f"{y['ratio']:.3f} +- {y['ratio_err']:.3f}", fontsize=9.5)
    x1, x2 = b["chi2_paired_above_95"], b["chi2_paired_above_100"]
    ax[0].text(0.03, 0.04, rf"paired $\chi^2$/ndf above 95: {x1['chi2']:.1f}/{x1['ndf']}, above 100: "
               rf"{x2['chi2']:.1f}/{x2['ndf']}", transform=ax[0].transAxes, fontsize=8.5)
    ax[0].legend(fontsize=9, loc="upper right")
    rr, re = np.array(b["ratio"], float)[k], np.array(b["ratio_err"], float)[k]
    ok = (fr > 0) & (fa > 0)
    ax[1].errorbar(c[ok], rr[ok], yerr=re[ok], xerr=w[ok] / 2, fmt="o", ms=3, color="#0072B2")
    ax[1].axhline(1, color="black", lw=0.8)
    ax[1].axvline(100.0, color="grey", lw=0.8, ls=":")
    ax[0].axvline(100.0, color="grey", lw=0.8, ls=":")
    ax[1].set_ylim(0.0, 2.0)
    ax[1].set_ylabel("fast / G4")
    sc = rep.get("shift_check")
    if sc:
        ax[0].text(0.03, 0.12, f"same-event shift {sc['shift']:+.3f} MeV/c predicts\n"
                   f"{sc['ratio_predicted']['95']:.3f} above 95, {sc['ratio_predicted']['100']:.3f} above 100",
                   transform=ax[0].transAxes, fontsize=8.5)
    ax[1].set_xlabel("reco momentum at the tracker front [MeV/c]")
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.dioreco", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("single", help="dumper tables without multi-primary events")
    s.add_argument("tables", nargs="+")
    s.add_argument("--outdir", required=True)
    c = sub.add_parser("compare", help="paired reco comparison of two arms on the same events")
    c.add_argument("--den", nargs="+", required=True, help="the tables both arms started from (dump/gen: events)")
    c.add_argument("--arm", required=True, help="dir with the fast arm's nts.*.root (recursive), or a reduce csv.gz")
    c.add_argument("--ref", required=True, help="dir with the Geant4 arm's nts.*.root (recursive), or a reduce csv.gz")
    c.add_argument("--drop-outside", action="store_true", help="drop arm events outside the denominator (counted)")
    c.add_argument("--out", required=True)
    r = sub.add_parser("reduce", help="per-event frame of EventNtuple files (csv.gz), for a --ref of compare")
    r.add_argument("nts", nargs="+")
    r.add_argument("--subruns", type=int, nargs="+", help="keep only these subruns")
    r.add_argument("--out", required=True)
    q = sub.add_parser("numbers", help="headline numbers of a compare report")
    q.add_argument("report")
    q.add_argument("out")
    q.add_argument("--production", help="the Geant4-vs-production compare report")
    p = sub.add_parser("plot")
    p.add_argument("report")
    p.add_argument("png")
    a = ap.parse_args(argv)
    if a.cmd == "single":
        rep = {}
        for t in a.tables:
            n, _ = single(t, os.path.join(a.outdir, os.path.basename(t)))
            rep[os.path.basename(t)] = n
        print(json.dumps(dict(files=len(rep), dropped_events=sum(rep.values()), per_file=rep)))
    elif a.cmd == "compare":
        from mu2efastsim import tables
        den = tables.read_many(a.den, "gen", KEY).drop_duplicates(KEY)
        for k in KEY:
            den[k] = den[k].astype(np.uint64)
        (fa, na), (fr, nr) = read_arm(a.arm), read_arm(a.ref)
        rep = compare(den, fa, fr, a.drop_outside)
        rep["ntuple_files"] = dict(arm=na, ref=nr)
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1, default=float)
        print(json.dumps({k: rep[k] for k in ("events", "efficiency_fit", "efficiency_selected")} |
                         {"above_100": rep["above"]["100"]}, default=float))
    elif a.cmd == "reduce":
        d = reduce(a.nts, a.subruns)
        d.to_csv(a.out, index=False)
        print(json.dumps(dict(files=len(a.nts), events=len(d), fit=int(d.fit.sum()), selected=int(selected(d).sum()))))
    elif a.cmd == "numbers":
        prod = json.load(open(a.production)) if a.production else None
        with open(a.out, "w") as fh:
            json.dump(numbers(json.load(open(a.report)), prod), fh, indent=1, default=float)
    else:
        plot(json.load(open(a.report)), a.png)


if __name__ == "__main__":
    main()
