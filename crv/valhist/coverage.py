"""Read-only ROOT coverage and descriptive rate/shape audit; never a significance test."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import uproot

from .compare import CRV_ONLY


LIMITATIONS = [
    "Inputs must be original unmerged ROOT files; flow/Sumw2 lost by old merges cannot be recovered or detected reliably.",
    "No p-values, pass/fail threshold, event-level covariance, or multiple-testing interpretation is provided.",
    "Source Sumw2 describes fill-weight squares, not covariance between paired arms or multiple fills from an event.",
    "Profiles are inventoried but unsupported; ambiguous DQM objects are conservatively treated as non-count maps.",
    "No paired-shard or bootstrap mode: event/shard identity and independence must be established before uncertainty work.",
]


def _aliases(specs):
    pairs = {}
    for spec in specs:
        if spec.count("=") != 1:
            raise ValueError("alias must be OLD=NEW")
        old, new = spec.split("=")
        if not old or not new or any(c in old + new for c in "/;"):
            raise ValueError("alias must contain nonempty Validation child prefixes without '/' or ';'")
        if old in pairs and pairs[old] != new:
            raise ValueError("conflicting alias: " + old)
        pairs[old] = new
    return sorted(pairs.items(), key=lambda p: (-len(p[0]), p[0]))


def _canonical(path, aliases):
    parts = path.split("/")
    if len(parts) > 1 and parts[0] == "Validation":
        for old, new in aliases:
            if parts[1] == old or parts[1].startswith(old + "_"):
                parts[1] = new + parts[1][len(old):]
                break
    return "/".join(parts)


def _inventory(root, select, aliases):
    latest = {}
    for key, classname in root.classnames(recursive=True, cycle=True).items():
        path, cycle = key.rsplit(";", 1)
        cycle = int(cycle)
        if path not in latest or cycle > latest[path]["cycle"]:
            latest[path] = {"path": path, "key": key, "cycle": cycle, "class": classname}
    selected = {}
    excluded = 0
    for path, item in latest.items():
        if item["class"].startswith("TDirectory"):
            continue
        canonical = _canonical(path, aliases)
        if select and not select.search(canonical):
            excluded += 1
            continue
        selected.setdefault(canonical, []).append(item)
    return selected, excluded


def _classification(path, title, classname):
    if classname.startswith("TProfile"):
        return "profile", "Profile means/errors require dedicated semantics."
    if not classname.startswith(("TH1", "TH2", "TH3")):
        return "unsupported", "Not a supported histogram family."
    text = (path + " " + title).lower()
    if any(word in text for word in ("rate", "mpv", "efficien", "occupancy", "fraction", "ratio", "mean", "rms", "sigma", "pedestal", "gain")):
        return "non_count", "Name/title indicates a derived quantity; no count normalization or Poisson model."
    if path.split("/")[0] in ("crvDigiDQM", "crvRecoDQM", "dqmCrv"):
        return "non_count", "Conservative DQM classification: count semantics are not certified by ROOT class."
    return "count_like", "TH histogram outside derived/DQM heuristics; descriptive sums only, not certified independent counts."


def _read(hist, classification):
    values = np.asarray(hist.values(flow=True), dtype=float)
    edges = [np.asarray(axis.edges(), dtype=float) for axis in hist.axes]
    labels = [axis.labels() for axis in hist.axes]
    raw = np.asarray(hist.member("fSumw2"), dtype=float)
    variance = raw.reshape(values.shape, order="F") if raw.size == values.size else None
    problems = []
    if raw.size not in (0, values.size):
        problems.append("malformed_sumw2")
    if not np.isfinite(values).all():
        problems.append("nonfinite_values")
    if any(not np.isfinite(e).all() or not (np.diff(e) > 0).all() for e in edges):
        problems.append("invalid_edges")
    if variance is not None and (not np.isfinite(variance).all() or (variance < 0).any()):
        problems.append("invalid_variances")
    visible = np.zeros(values.shape, dtype=bool)
    visible[tuple(slice(1, -1) for _ in edges)] = True
    regions = {"visible": visible, "flow": ~visible, "inclusive": np.ones(values.shape, dtype=bool)}
    meta = {
        "title": hist.title, "dimensions": len(edges), "flow_array_shape": list(values.shape),
        "edges": edges, "axis_labels": labels, "sumw2_present": bool(raw.size),
        "variance_source": "source_sumw2" if variance is not None else "unavailable",
        "issues": problems,
        "regions": {name: {"bins": int(mask.sum()), "value_sum": float(values[mask].sum()),
                           "variance_sum": float(variance[mask].sum()) if variance is not None else None}
                    for name, mask in regions.items()},
        "sum_interpretation": "fill weight sum, not event count" if classification == "count_like" else "arithmetic diagnostic only; not a count or rate total",
    }
    return values, variance, edges, labels, regions, meta


def _difference(a, b):
    delta = b - a
    return {"equal": bool(np.array_equal(a, b)), "differing_bins": int(np.count_nonzero(a != b)),
            "max_abs_difference": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "l1_difference": float(np.abs(delta).sum())}


def _shape(a, b):
    if (a < 0).any() or (b < 0).any():
        return {"status": "undefined_signed_weights"}
    ta, tb = float(a.sum()), float(b.sum())
    if ta <= 0 or tb <= 0:
        return {"status": "undefined_zero_total"}
    if not np.isfinite(ta) or not np.isfinite(tb):
        return {"status": "undefined_nonfinite_total"}
    p, q = a / ta, b / tb
    return {"status": "descriptive", "total_variation": float(np.abs(p - q).sum() / 2),
            "max_abs_fraction_difference": float(np.max(np.abs(p - q))) if p.size else 0.0}


def _weighted(a, b, va, vb):
    if va is None or vb is None:
        return {"status": "unavailable_source_sumw2"}
    denom = va + vb
    positive = denom > 0
    delta = b - a
    conflict = (~positive) & (delta != 0)
    terms = delta[positive] ** 2 / denom[positive]
    finite = bool(np.isfinite(terms).all() and np.isfinite(denom).all())
    finite = finite and bool(np.isfinite(terms.sum()))
    status = "numeric_overflow" if not finite else "undefined_zero_variance_difference" if conflict.any() else "descriptive_only"
    return {"status": status,
            "positive_variance_bins": int(positive.sum()), "zero_variance_bins": int((~positive).sum()),
            "zero_variance_different_bins": int(conflict.sum()),
            "positive_variance_bin_contribution": float(terms.sum()) if finite else None,
            "sum_squared_difference_over_variance_sum": float(terms.sum()) if finite and not conflict.any() else None,
            "interpretation": "Positive-variance bins only; zero-variance conflicts are explicit. Not chi-square: paired/event covariance unknown."}


def _compare_pair(root_ref, root_cand, key, ref, cand):
    result = {"ref": ref, "cand": cand, "issues": []}
    if len(ref) > 1 or len(cand) > 1:
        result.update(status="alias_collision", issues=["alias_collision"])
        return result
    if not ref or not cand:
        missing = "missing_ref" if not ref else "missing_cand"
        result.update(status=missing, issues=[missing])
        return result
    r, c = ref[0], cand[0]
    if r["class"] != c["class"]:
        result["issues"].append("type_mismatch")
    try:
        hr, hc = root_ref[r["key"]], root_cand[c["key"]]
        cr, rr = _classification(key, getattr(hr, "title", ""), r["class"])
        cc, rc = _classification(key, getattr(hc, "title", ""), c["class"])
        result["classification"] = {"ref": cr, "cand": cc, "ref_reason": rr, "cand_reason": rc}
        if "profile" in (cr, cc):
            result["issues"].append("unsupported_profile")
        elif "unsupported" in (cr, cc):
            result["issues"].append("unsupported_object")
        else:
            a, va, ea, la, regions, ma = _read(hr, cr)
            b, vb, eb, lb, _, mb = _read(hc, cc)
            result.update(ref_data=ma, cand_data=mb)
            result["issues"].extend("ref_" + issue for issue in ma["issues"])
            result["issues"].extend("cand_" + issue for issue in mb["issues"])
            if len(ea) != len(eb) or any(not np.array_equal(x, y) for x, y in zip(ea, eb)) or la != lb or a.shape != b.shape:
                result["issues"].append("binning_mismatch")
            if not result["issues"]:
                values = _difference(a, b)
                variances = _difference(va, vb) if va is not None and vb is not None else None
                result["closure"] = {"values": values, "source_variances": variances,
                                     "exact_values_and_variances": (values["equal"] and variances["equal"]) if variances is not None else None,
                                     "sumw2_presence_equal": (va is None) == (vb is None),
                                     "scope": "flow-inclusive bin contents and source Sumw2, not ROOT metadata or events"}
                result["bin_differences"] = {name: _difference(a[mask], b[mask]) for name, mask in regions.items()}
                if cr == cc == "count_like":
                    result["normalization"] = {}
                    result["shape"] = {}
                    result["weighted_diagnostics"] = {}
                    for name, mask in regions.items():
                        ta, tb = float(a[mask].sum()), float(b[mask].sum())
                        result["normalization"][name] = {"ref": ta, "cand": tb, "difference": tb - ta,
                                                             "relative_difference": (tb - ta) / ta if ta != 0 else None,
                                                             "relative_status": "defined" if ta != 0 else "undefined_zero_reference"}
                        result["shape"][name] = _shape(a[mask], b[mask])
                        result["weighted_diagnostics"][name] = _weighted(a[mask], b[mask], va[mask] if va is not None else None, vb[mask] if vb is not None else None)
                else:
                    result["normalization"] = {"status": "not_applicable_non_count"}
                    result["shape"] = {"status": "not_applicable_non_count"}
                    result["weighted_diagnostics"] = {"status": "not_applicable_non_count"}
    except Exception as exc:
        result["issues"].append("read_error")
        result["error"] = f"{type(exc).__name__}: {exc}"
    if _has_nonfinite(result) and not any("nonfinite" in issue or "invalid" in issue for issue in result["issues"]):
        result["issues"].append("numeric_overflow")
    result["status"] = result["issues"][0] if result["issues"] else "compared"
    return result


def _has_nonfinite(value):
    if isinstance(value, np.ndarray):
        return not bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(item) for item in value)
    return isinstance(value, (float, np.floating)) and not np.isfinite(value)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def audit(ref, cand, outdir, select=CRV_ONLY, aliases=()):
    """Audit one original file per arm, writing strict JSON to a new directory."""
    alias_pairs = _aliases(aliases)
    selection = re.compile(select) if select else None
    outdir = Path(outdir)
    if outdir.exists():
        raise FileExistsError(f"output directory must be new: {outdir}")
    with uproot.open(ref) as fr, uproot.open(cand) as fc:
        ir, er = _inventory(fr, selection, alias_pairs)
        ic, ec = _inventory(fc, selection, alias_pairs)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            reports = {key: _compare_pair(fr, fc, key, ir.get(key, []), ic.get(key, []))
                       for key in sorted(set(ir) | set(ic))}
    gaps = {key: item["issues"] for key, item in reports.items() if item["issues"]}
    summary = {
        "schema_version": 1, "ref": str(Path(ref).resolve()), "cand": str(Path(cand).resolve()),
        "select": select, "aliases": list(aliases), "limitations": LIMITATIONS,
        "flow_provenance": "Read directly from supplied source objects; original/unmerged provenance is the caller's responsibility.",
        "selected_ref_objects": sum(map(len, ir.values())), "selected_cand_objects": sum(map(len, ic.values())),
        "excluded_ref_objects": er, "excluded_cand_objects": ec, "union_keys": len(reports),
        "statuses": dict(Counter(item["status"] for item in reports.values())),
        "coverage_gaps": gaps, "empty_selection": not bool(reports),
        "only_ref": sorted(set(ir) - set(ic)), "only_cand": sorted(set(ic) - set(ir)),
        "value_differences": [k for k, v in reports.items() if "closure" in v and not v["closure"]["values"]["equal"]],
        "variance_differences": [k for k, v in reports.items() if v.get("closure", {}).get("source_variances") is not None and not v["closure"]["source_variances"]["equal"]],
        "variance_coverage_gaps": [k for k, v in reports.items() if "closure" in v and v["closure"]["source_variances"] is None],
        "nonfinite_json_policy": "Nonfinite numbers become null; invalid source arrays are explicit issues and not compared.",
    }
    outdir.mkdir(parents=True, exist_ok=False)
    for filename, content in (("histograms.json", reports), ("summary.json", summary)):
        (outdir / filename).write_text(json.dumps(_json_safe(content), indent=2, allow_nan=False) + "\n")
    return _json_safe(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref")
    parser.add_argument("cand")
    parser.add_argument("--outdir", required=True, help="new output directory; existing paths are refused")
    parser.add_argument("--select", default=CRV_ONLY, help="regex on canonical paths; empty string selects all")
    parser.add_argument("--alias", action="append", default=[], help="Validation child prefix OLD=NEW; repeatable")
    args = parser.parse_args()
    try:
        summary = audit(args.ref, args.cand, args.outdir, args.select, args.alias)
    except (ValueError, OSError, re.error) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps({"union_keys": summary["union_keys"], "statuses": summary["statuses"],
                      "coverage_gap_count": len(summary["coverage_gaps"])}, allow_nan=False))


if __name__ == "__main__":
    main()
