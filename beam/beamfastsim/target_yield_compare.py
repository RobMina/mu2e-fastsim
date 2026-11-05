"""Exposure-corrected DIO/internal-RPC electron diagnostics; downstream filtering remains uncertified."""
import argparse
import glob
import json

import numpy as np

from . import cechain as C
from . import cecompare as CC


def denominators(reference, exposure=None):
    """Use a certified parent-window census, never retained-event multiplicities, as exposure."""
    if exposure is None:
        identity = np.column_stack([reference[k] for k in ("ifile", "subrun", "evt")])
        if np.unique(identity, axis=0).shape[0] != len(identity):
            raise ValueError("multiple DIO electrons per event: gen.count is not an electron denominator; audit parent multiplicity")
        raise ValueError("exact parent-window electron exposure is required even if kept rows have one electron per event")
    report = exposure.get("production_exposure", exposure)
    if not report.get("exact_production_electron_exposure_known", False):
        raise ValueError("parent audit must certify exact production electron exposure")
    ng = np.asarray(reference["gencount"], dtype=np.int64)
    records = report.get("files", [])
    if sorted(r["ifile"] for r in records) != list(range(ng.size)):
        raise ValueError("exposure must identify each reference file exactly once")
    records = sorted(records, key=lambda r: r["ifile"])
    counts = np.asarray([r["generated_electrons"] for r in records])
    events = np.asarray([r["generated_events"] for r in records])
    matched = np.asarray([r["matched_electrons"] for r in records])
    kept = np.bincount(reference["ifile"], minlength=ng.size)
    if (not np.array_equal(events, ng) or not np.array_equal(matched, kept)
            or np.any(counts < matched) or np.any(counts != np.floor(counts))):
        raise ValueError("electron exposure does not match reference event/row counts")
    return counts.astype(np.int64)


def exposure_ratio(counts, electrons, fast_selected, fast_generated):
    """Independent fast-row error plus file-cluster G4 exposure-normalized influence."""
    counts, electrons = np.asarray(counts, float), np.asarray(electrons, float)
    if counts.shape != electrons.shape or counts.ndim != 1 or not counts.size or np.any(electrons <= 0):
        raise ValueError("positive per-file electron exposures required")
    rate = counts.sum() / electrons.sum()
    influence = counts - rate * electrons
    n = counts.size
    ge = np.sqrt(n / (n - 1) * np.sum(influence ** 2)) / electrons.sum() if n > 1 else None
    fast = fast_selected / fast_generated
    fe = np.sqrt(fast * (1 - fast) / fast_generated)
    ratio = fast / rate if rate > 0 else None
    error = np.hypot(fe / rate, fast * ge / rate ** 2) if ge is not None and rate > 0 else None
    return dict(g4=float(rate), g4_clustered_err=float(ge) if ge is not None else None,
                fast=float(fast), fast_err=float(fe), ratio=float(ratio) if ratio is not None else None,
                ratio_err=float(error) if error is not None else None, clusters=int(n),
                method="G4 file-cluster sandwich on counts minus rate times exposure; independent fast generated-electron binomial error; no filter-efficiency uncertainty included")


def compare(ref, fast_glob, geom, field, exposure=None):
    g, f = CC.g4_plane(ref), CC.fast_plane(glob.glob(fast_glob))
    with np.load(ref) as r:
        code = int(r["creation_code"]) if "creation_code" in r else None
        if code not in (166, 179):
            raise ValueError("audited DIO or internal RPC primary-electron reference required")
        if code == 179 and (exposure or {}).get("mode") != "exact-RPC-one-pion-parent-window-population":
            raise ValueError("RPC requires an exact one-pion parent-window audit, not an event-count assumption")
        group, ng = r["ifile"], denominators(r, exposure)
        from .target_parent_audit import checksum
        if exposure.get("production_exposure", exposure).get("reference_sha256") != checksum(ref):
            raise ValueError("exposure reference hash does not match input reference")
    go, fo = CC.observables(geom, field, g), CC.observables(geom, field, f)
    out = dict(mode=("dio" if code == 166 else "rpc-internal-electron-marginal") + "-independent-exposure-corrected-diagnostic", g4_generated_events=g["gen"],
               g4_generated_electrons=int(ng.sum()), fast_generated=int(f["fate"].size),
               g4_files=len(ng), filter_implication_certified=False, generated_yield_validated=False,
               caveat="production filter must be implied by the VD selection; stability across cuts is necessary, not proof",
               cuts={})
    for rc in CC.RMAX_CUTS:
        gs = (go["p"] > CC.P_MIN) & (go["rmax"] > rc)
        fs = (fo["p"] > CC.P_MIN) & (fo["rmax"] > rc)
        counts = np.bincount(group[g["ok"]][gs], minlength=len(ng))
        rates = counts / ng
        c = dict(yield_per_generated=CC._ratio(int(gs.sum()), int(ng.sum()), int(fs.sum()), f["fate"].size),
                 exposure_clustered_yield=exposure_ratio(counts, ng, int(fs.sum()), int(f["fate"].size)),
                 g4_file_yields=rates.tolist(),
                 g4_file_yield_err=float(rates.std(ddof=1) / np.sqrt(len(ng))) if len(ng) > 1 else None,
                 g4_selected=int(gs.sum()), fast_selected=int(fs.sum()), shapes={})
        for k in ("p", "cos", "r", "rmax", "tof", "t"):
            a, b = go[k][gs], fo[k][fs]
            if not a.size or not b.size:
                c["shapes"][k] = None
                continue
            lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
            edges = CC.quantile_edges(a, CC.NBINS, lo=lo, hi=hi) if lo != hi else np.array([lo - 0.5, hi + 0.5])
            chi, nd = CC.chi2(a, b, edges)
            c["shapes"][k] = dict(g4=CC.summary(a), fast=CC.summary(b), chi2=float(chi), ndf=nd - 1,
                                  chi2_ndf=float(chi / max(nd - 1, 1)), errors="independent rows; descriptive only")
        out["cuts"][str(rc)] = c
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--fast", required=True)
    ap.add_argument("--offline-root", required=True)
    ap.add_argument("--geom", default="Offline/Mu2eG4/geom/geom_run1_a.txt")
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exposure", required=True, help="target_parent_audit JSON with exact production windows")
    a = ap.parse_args(argv)
    with open(a.exposure) as stream:
        exposure = json.load(stream)
    r = compare(a.ref, a.fast, C.from_offline(a.geom, a.offline_root), C.field_maps(a.maps), exposure)
    with open(a.out, "w") as stream:
        json.dump(r, stream, indent=2)
    print(json.dumps({k: v for k, v in r.items() if k != "cuts"}))
    for rc, c in r["cuts"].items():
        print(json.dumps(dict(rmax_cut=rc, yield_per_generated=c["yield_per_generated"],
                              exposure_clustered_yield=c["exposure_clustered_yield"],
                              g4_file_yield_err=c["g4_file_yield_err"])))


if __name__ == "__main__":
    main()
