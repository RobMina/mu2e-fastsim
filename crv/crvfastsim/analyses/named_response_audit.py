"""Paired, all-sector diagnostics of saved stock/fast response products; not V2 certification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from .. import named_conditions as NC, runconfig

KEY = ("run", "subrun", "event")


def event_ids(root):
    """Full-width event identities, sorted without packed-key aliases."""
    a = root["crvdump/events"].arrays(list(KEY), library="np")
    ids = np.column_stack([NC._integers(a[k], k) for k in KEY])
    if len(ids) == 0 or len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError("require nonempty unique event identities")
    return ids[np.lexsort(ids.T[::-1])]


def row_events(rows, ids):
    """Resolve row identities to a common event universe and reject foreign rows."""
    keys = np.column_stack([NC._integers(ak.to_numpy(rows[k]), k) for k in KEY])
    combined = np.vstack((ids, keys))
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    if not np.array_equal(inverse[:len(ids)], np.arange(len(ids))):
        raise ValueError("rows contain identities outside event universe")
    pos = inverse[len(ids):]
    if np.any(pos >= len(ids)) or not np.array_equal(ids[pos], keys):
        raise ValueError("rows contain identities outside event universe")
    return pos


def paired_ratio(candidate, reference):
    """Ratio of event means with paired delta-method SE, including zero rows."""
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if c.shape != r.shape or c.ndim != 1 or not np.isfinite(c).all() or not np.isfinite(r).all():
        raise ValueError("paired finite one-dimensional event values required")
    n = len(c)
    cm, rm = (float(c.mean()), float(r.mean())) if n else (0.0, 0.0)
    out = {"events": n, "candidate_mean": cm, "reference_mean": rm, "relative": None, "relative_se": None}
    if n < 2 or rm <= 0:
        out["status"] = "unsupported"
        return out
    ratio = cm / rm
    out.update(relative=ratio - 1, relative_se=float(np.std(c - ratio * r, ddof=1) / np.sqrt(n) / rm),
               status="diagnostic")
    return out


def _read(path, ids, geom, cond):
    ns = len(geom.sector_names)
    with uproot.open(path) as root:
        if not np.array_equal(event_ids(root), ids):
            raise ValueError("candidate/reference event universes differ")
        d = root["crvdump/digis"].arrays([*KEY, "bar", "sipm", "adc"])
        p = root["crvdump/pulses"].arrays([*KEY, "bar", "sipm", "pe", "seq", "time"])
    result = {}
    spectra = {}
    for label, rows in (("digi", d), ("pulse", p)):
        bar = NC._integers(ak.to_numpy(rows.bar), "bar")
        sipm = NC._integers(ak.to_numpy(rows.sipm), "sipm")
        if np.any(bar >= geom.nbars) or np.any(sipm >= 4):
            raise ValueError("channel outside named geometry")
        channel = bar * 4 + sipm
        slot = row_events(rows, ids) * (ns * 4) + geom.sector[bar] * 4 + sipm
        weights = {label + "_count": np.ones(len(rows))}
        if label == "digi":
            adc = ak.values_astype(rows.adc, np.float64)
            if not np.isfinite(ak.to_numpy(ak.flatten(adc))).all():
                raise ValueError("nonfinite ADC")
            area = ak.to_numpy(ak.sum(adc - cond.pedestal[channel], axis=1))
            weights.update(adc_pedestal_subtracted=area, adc_area_gain_normalized=area / cond.pulsearea[channel],
                           adc_samples=ak.to_numpy(ak.num(adc)))
        else:
            pe, times = ak.to_numpy(rows.pe), ak.to_numpy(rows.time)
            if not np.isfinite(pe).all() or not np.isfinite(times).all():
                raise ValueError("nonfinite pulse values")
            s0 = ak.to_numpy(rows.seq) == 0
            weights.update(pulse_pe=pe, seq0_count=s0.astype(float), seq0_pe=pe * s0,
                           pulse_outside_offspill=((times < 0) | (times > 100000)).astype(float))
            spectra = {"pe": pe, "seq0": s0, "sector": geom.sector[bar], "sipm": sipm}
        for name, values in weights.items():
            result[name] = np.bincount(slot, weights=values, minlength=len(ids) * ns * 4).reshape(len(ids), ns, 4)
    return result, spectra


def audit(reference, candidate, conditions_manifest, run_config, outdir):
    """Audit existing products without simulation, tuning or producer-provenance admission."""
    rc = runconfig.get(run_config) if isinstance(run_config, str) else run_config
    geom = rc.geometry()
    cond, manifest = NC.load_conditions(conditions_manifest, rc, geom)
    paths = {"reference": Path(reference), "candidate": Path(candidate), "conditions_manifest": Path(conditions_manifest)}
    hashes = {k: NC._record(p, str(p.resolve())) for k, p in paths.items()}
    with uproot.open(reference) as root:
        ids = event_ids(root)
    r, rp = _read(reference, ids, geom, cond)
    c, cp = _read(candidate, ids, geom, cond)
    groups = {}
    for sector, name in enumerate(geom.sector_names):
        groups[name] = {}
        for side in ("all", "0", "1", "2", "3"):
            metrics = {}
            for metric in r:
                rv = r[metric][:, sector].sum(axis=1) if side == "all" else r[metric][:, sector, int(side)]
                cv = c[metric][:, sector].sum(axis=1) if side == "all" else c[metric][:, sector, int(side)]
                metrics[metric] = paired_ratio(cv, rv)
            median = {}
            for arm, a in (("reference", rp), ("candidate", cp)):
                mask = a["seq0"] & (a["sector"] == sector)
                if side != "all":
                    mask &= a["sipm"] == int(side)
                values = a["pe"][mask]
                median[arm] = {"count": len(values), "median": float(np.median(values)) if len(values) else None}
            groups[name][side] = {"paired_event_metrics": metrics, "seq0_pe_median": median}
    if any(NC._sha(paths[k]) != h["sha256"] for k, h in hashes.items()):
        raise ValueError("input changed during audit")
    result = {"schema": "crv-named-response-audit-v1", "run_config": rc.name, "events": len(ids),
              "event_identity_sha256": NC._digest(ids.tolist()), "inputs": hashes,
              "conditions_config_sha256": manifest["config_sha256"], "sectors": groups,
              "status": "diagnostic_only_saved_products_no_producer_certification",
              "uncertainty": "paired event delta-method SE; no independent-seed floor; medians descriptive only",
              "adc_semantics": "sum of saved zero-suppressed samples minus per-channel pedestal; not deposited energy",
              "scope": "conditional on exact saved stock event identities, including zero-response events"}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = ["# Named response saved-product audit", "", f"{rc.name}; {len(ids)} identical event identities.",
             "Paired-event relative differences ± SE (%); descriptive diagnostics, not closure.", "",
             "| Sector | Digi count | ADC pedestal-subtracted | Pulse count | Pulse PE sum | Seq0 PE sum |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, sides in groups.items():
        cells = []
        for metric in ("digi_count", "adc_pedestal_subtracted", "pulse_count", "pulse_pe", "seq0_pe"):
            m = sides["all"]["paired_event_metrics"][metric]
            cells.append("unsupported" if m["relative"] is None else f"{100*m['relative']:+.3f} ± {100*m['relative_se']:.3f}")
        lines.append("| " + " | ".join([name, *cells]) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--conditions-manifest", required=True)
    ap.add_argument("--run-config", required=True, choices=sorted(NC.GEOMETRIES))
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    result = audit(a.reference, a.candidate, a.conditions_manifest, a.run_config, a.outdir)
    print(f"{result['run_config']}: {result['events']} paired events; saved-product diagnostics written to {a.outdir}")


if __name__ == "__main__":
    main()
