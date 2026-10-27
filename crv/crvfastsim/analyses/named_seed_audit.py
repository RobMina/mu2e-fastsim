"""Conditional fast-seed spread over saved named-response audits, not response certification."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
import statistics

from .. import named_conditions as NC, readout, runconfig
from ..chain import ChainConfig

AUDIT_SCHEMA = "crv-named-response-audit-v1"
AUDIT_STATUS = "diagnostic_only_saved_products_no_producer_certification"
SIDES = ("all", "0", "1", "2", "3")
METRICS = ("digi_count", "adc_pedestal_subtracted", "adc_area_gain_normalized", "adc_samples",
           "pulse_count", "pulse_pe", "seq0_count", "seq0_pe", "pulse_outside_offspill")
COMMON_METADATA = ("schema", "run_config", "stage", "noise", "window_applied", "configured_window_ns",
                   "effective_window_ns", "window_selection", "conditions", "finder", "timing_policy",
                   "onspill_equivalence", "geometry_readout")


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(value, label):
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value), f"invalid {label} SHA256")
    return value


def _number(value, label):
    _require(type(value) in (int, float), f"{label}: finite number required")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, f"{label}: finite number required")
    return value


def _record(path):
    path = Path(path).resolve(strict=True)
    _require(path.is_file(), f"not a regular file: {path}")
    return NC._record(path, str(path))


def _input(record, base, label):
    _require(isinstance(record, dict), f"missing {label} input record")
    _require(isinstance(record.get("path"), str) and record["path"], f"missing {label} path")
    _digest(record.get("sha256"), label)
    _require(type(record.get("size")) is int and record["size"] >= 0, f"invalid {label} size")
    path = Path(record["path"])
    current = _record(path if path.is_absolute() else base / path)
    _require(all(current[k] == record[k] for k in ("sha256", "size")), f"{label} hash/size mismatch")
    return current


def _json(path):
    value = NC._json(path)
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    NC._canonical(value)
    return value


def _metadata(meta, audit):
    _require("diagnostic_override" not in meta, "diagnostic_override forbidden: declared yields only")
    _require(all(k in meta for k in COMMON_METADATA), "missing producer metadata fields")
    _require(meta["schema"] == "crv-fasttables-named-v1", "unsupported producer metadata schema")
    _require(meta["run_config"] == audit["run_config"], "metadata run_config mismatch")
    _require(meta["stage"] == "chain" and meta["window_applied"] is True, "require named chain step mode")
    _require(type(meta.get("seed")) is int and meta["seed"] >= 0, "invalid producer seed")
    cfg = meta.get("chain_config")
    _require(isinstance(cfg, dict) and {f.name for f in dataclasses.fields(ChainConfig)} <= set(cfg),
             "missing full chain_config")
    _require(all(type(cfg[k]) is bool for k in ("jax", "fused_response", "noise")), "invalid backend/noise flags")
    _require(type(meta.get("jax")) is bool and meta["jax"] == cfg["jax"], "metadata jax mismatch")
    _require(type(meta["noise"]) is bool and meta["noise"] == cfg["noise"], "metadata noise mismatch")
    if "fused_response" in meta:
        _require(type(meta["fused_response"]) is bool and meta["fused_response"] == cfg["fused_response"],
                 "metadata fused_response mismatch")
    _require(not cfg["fused_response"] or cfg["jax"], "fused_response requires jax")
    rc = runconfig.get(audit["run_config"])
    _require(meta["configured_window_ns"] == list(rc.window), "configured window mismatch")
    expected_window = {"offspill": list(runconfig.OFFSPILL_WINDOW), "onspill": list(runconfig.ONSPILL_WINDOW)}
    _require(isinstance(meta["window_selection"], str) and meta["window_selection"] in expected_window,
             "invalid window selection")
    _require(meta["effective_window_ns"] == cfg["window"] == expected_window[meta["window_selection"]],
             "effective window mismatch")
    _require(meta["finder"] == cfg["finder"], "finder metadata mismatch")
    _require(meta["timing_policy"] == "fixed_absolute_crop_no_pbtime_rebase" and
             meta["onspill_equivalence"] is False, "unsupported timing policy")
    response = cfg["response"]
    declared = dataclasses.asdict(rc.response())
    _require(isinstance(response, dict) and response.get("sectors") == declared["sectors"] and
             response.get("photon_yield_scale") == declared["photon_yield_scale"], "require declared named yields")
    conditions = meta["conditions"]
    _require(isinstance(conditions, dict) and conditions.get("run_config") == audit["run_config"] and
             conditions.get("config_sha256") == audit["conditions_config_sha256"], "conditions config mismatch")
    _require(isinstance(conditions.get("config"), dict) and
             NC._digest(conditions["config"]) == conditions["config_sha256"], "conditions config digest mismatch")
    readout.validate_merge([meta])
    _require(meta["geometry_readout"] is not None, "geometry_readout required")
    backend = "jax_fused" if cfg["fused_response"] else "jax_nonfused" if cfg["jax"] else "numpy"
    normalized = {k: v for k, v in cfg.items() if k not in ("jax", "fused_response")}
    return backend, {**{k: meta[k] for k in COMMON_METADATA}, "chain_config": normalized}


def _metrics(audit):
    _require(audit.get("schema") == AUDIT_SCHEMA and audit.get("status") == AUDIT_STATUS,
             "unsupported audit schema/status")
    _require(isinstance(audit.get("run_config"), str) and audit["run_config"] in NC.GEOMETRIES,
             "unsupported named run_config")
    n = audit.get("events")
    _require(type(n) is int and n > 0, "nonempty event count required")
    _digest(audit.get("event_identity_sha256"), "event identity")
    _digest(audit.get("conditions_config_sha256"), "conditions config")
    sectors = audit.get("sectors")
    names = runconfig.get(audit["run_config"]).sectors
    _require(isinstance(sectors, dict) and set(sectors) == set(names), "missing/unknown audit sectors")
    reference = {}
    for sector, sides in sectors.items():
        _require(isinstance(sides, dict) and set(sides) == set(SIDES), f"{sector}: all SiPM groups required")
        for side, group in sides.items():
            label = f"{sector}/{side}"
            _require(isinstance(group, dict), f"{label}: malformed metric group")
            metrics = group.get("paired_event_metrics")
            _require(isinstance(metrics, dict) and set(metrics) == set(METRICS), f"{label}: missing/unknown metrics")
            for name, m in metrics.items():
                tag = f"{label}/{name}"
                _require(isinstance(m, dict) and type(m.get("events")) is int and m["events"] == n,
                         f"{tag}: metric event count mismatch")
                cm = _number(m.get("candidate_mean"), tag + " candidate_mean")
                rm = _number(m.get("reference_mean"), tag + " reference_mean")
                _require("relative" in m and "relative_se" in m, f"{tag}: missing relative metrics")
                supported = n >= 2 and rm > 0
                if supported:
                    relative = _number(m["relative"], tag + " relative")
                    ratio = _number(cm / rm - 1, tag + " recomputed relative")
                    _require(m.get("status") == "diagnostic" and
                             math.isclose(relative, ratio, rel_tol=1e-12, abs_tol=1e-12),
                             f"{tag}: stored relative mismatch")
                    _require(_number(m["relative_se"], tag + " paired SE") >= 0, f"{tag}: negative paired SE")
                else:
                    _require(m.get("status") == "unsupported" and m["relative"] is None and m["relative_se"] is None,
                             f"{tag}: unsupported reference requires null relative metrics")
                reference[(sector, side, name)] = rm
            medians = group.get("seq0_pe_median")
            _require(isinstance(medians, dict) and set(medians) == {"candidate", "reference"},
                     f"{label}: missing descriptive medians")
            for arm, median in medians.items():
                _require(isinstance(median, dict) and type(median.get("count")) is int and median["count"] >= 0
                         and "median" in median, f"{label}: invalid median count")
                if median["count"]:
                    _number(median["median"], label + " median")
                else:
                    _require(median["median"] is None, f"{label}: empty median must be null")
            reference[(sector, side, "seq0_pe_median")] = medians["reference"]
    return reference


def _spread(values, k):
    if values is None:
        return dict(K=k, mean=None, sample_sd=None, se_mean=None, min=None, max=None)
    sd = statistics.stdev(values)
    result = dict(K=k, mean=statistics.mean(values), sample_sd=sd, se_mean=sd / math.sqrt(k),
                  min=min(values), max=max(values))
    for name, value in result.items():
        _number(value, "aggregate " + name)
    return result


def _aggregate(rows):
    k = len(rows)
    groups = {}
    for sector, sides in rows[0]["audit"]["sectors"].items():
        groups[sector] = {}
        for side in sides:
            summaries = {}
            for metric in METRICS:
                values = [r["audit"]["sectors"][sector][side]["paired_event_metrics"][metric] for r in rows]
                rm = values[0]["reference_mean"]
                candidate = [v["candidate_mean"] for v in values]
                supported = values[0]["status"] == "diagnostic"
                relative = [c / rm - 1 for c in candidate] if supported else None
                summaries[metric] = {"reference_mean": rm, "reference_zero": rm == 0,
                                     "status": "conditional_seed_diagnostic" if supported else "unsupported_relative",
                                     "relative": _spread(relative, k), "candidate_mean": _spread(candidate, k),
                                     "candidate_all_zero": all(c == 0 for c in candidate)}
            medians = [{"seed": r["seed"], **r["audit"]["sectors"][sector][side]["seq0_pe_median"]} for r in rows]
            groups[sector][side] = {"metrics": summaries, "seq0_pe_median_by_seed": medians}
    return groups


def aggregate(audits, outdir):
    """Validate saved files and declarations, then create an exclusive conditional seed report."""
    out = Path(outdir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"output must be new: {out}")
    _require(bool(audits), "at least three audits per backend required")
    rows, snapshots, seen_paths, seen_hashes, seen_seeds = [], {}, set(), set(), set()
    baseline = None
    for audit_path in audits:
        audit_record = _record(audit_path)
        path = Path(audit_record["path"])
        audit = _json(path)
        reference_metrics = _metrics(audit)
        inputs = audit.get("inputs")
        _require(isinstance(inputs, dict), "missing audit inputs")
        reference = _input(inputs.get("reference"), path.parent, "reference")
        candidate = _input(inputs.get("candidate"), path.parent, "candidate")
        manifest = _input(inputs.get("conditions_manifest"), path.parent, "conditions_manifest")
        _require(candidate["path"] not in seen_paths and candidate["sha256"] not in seen_hashes,
                 "duplicate candidate resolved path or hash")
        seen_paths.add(candidate["path"])
        seen_hashes.add(candidate["sha256"])
        metadata_record = _record(candidate["path"] + ".metadata.json")
        metadata = _json(metadata_record["path"])
        backend, common_metadata = _metadata(metadata, audit)
        seed = metadata["seed"]
        _require((backend, seed) not in seen_seeds, "duplicate seed within backend")
        seen_seeds.add((backend, seed))
        identity = {k: audit[k] for k in ("schema", "status", "run_config", "events", "event_identity_sha256",
                                         "conditions_config_sha256")}
        common = (identity, reference, common_metadata, reference_metrics, manifest)
        if baseline is None:
            baseline = common
        else:
            _require(common == baseline and NC._canonical(common_metadata) == NC._canonical(baseline[2]),
                     "audit identity/reference/config/metadata/reference metrics drift")
        row = dict(backend=backend, seed=seed, audit=audit, audit_file=audit_record, candidate=candidate,
                   metadata_file=metadata_record, metadata=metadata)
        rows.append(row)
        for record in (audit_record, reference, candidate, manifest, metadata_record):
            old = snapshots.setdefault(record["path"], record)
            _require(old == record, "input changed during validation")
    backends = {}
    for backend in sorted({r["backend"] for r in rows}):
        members = sorted((r for r in rows if r["backend"] == backend), key=lambda r: r["seed"])
        k = len(members)
        _require(k >= 3, f"{backend}: at least three distinct seeds required")
        sectors = _aggregate(members)
        forbidden = []
        if baseline[0]["run_config"] == "run2_v10":
            for side in ("1", "3"):
                m = sectors["R4"][side]["metrics"]
                forbidden.append({"sector": "R4", "sipm": side,
                                  "basis": "documented Run2 R4 absent side-1; metadata is not execution proof",
                                  "candidate_all_zero": all(v["candidate_all_zero"] for v in m.values()),
                                  "metrics": m})
        backends[backend] = {"K": k, "seeds": [r["seed"] for r in members], "small_ensemble": k < 30,
                             "sectors": sectors, "documented_forbidden_channels": forbidden}
    result = {"schema": "crv-named-seed-audit-v1", "status": "conditional_seed_diagnostic_no_execution_certification",
              **{k: baseline[0][k] for k in ("run_config", "events", "event_identity_sha256", "conditions_config_sha256")},
              "reference": baseline[1], "common_metadata": baseline[2], "backends": backends,
              "inputs": [{k: v for k, v in r.items() if k != "audit"} for r in rows],
              "scope": "Single-replay conditional fast-seed spread at fixed G4 and fixed stock sample; fresh fast seeds only. "
                       "Not a full independent-seed floor, systematic error, spectral test, V2 or physics certification.",
              "uncertainty": "Sample SD (ddof=1) across seed ratios; SEmean=SD/sqrt(K). Paired-event SE is validated "
                             "but not combined, averaged, or added in quadrature. Backend labels are metadata declarations.",
              "small_ensemble_policy": "K < 30 is flagged, including K=8; this heuristic is not an admission threshold.",
              "median_semantics": "Per-seed saved seq0 medians are descriptive only; no pooled median or error estimate.",
              "adc_semantics": "Saved zero-suppressed sample areas, not deposited energy or calibrated PE."}
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    lines = ["# Conditional named fast-seed spread", "", f"{result['run_config']}; {result['events']} fixed events.",
             "Pulse PE relative mean ± single-replay seed sample SD (%); K seeds per backend.",
             "Fixed G4 and fixed stock sample; not an independent stock-seed floor, spectra, V2 or certification.",
             "Backend names are declarations, not traced execution certificates. K < 30: small ensemble (including 8).", "",
             "| Sector | " + " | ".join(backends) + " |", "|---|" + "---:|" * len(backends)]
    for sector in baseline[2]["chain_config"]["response"]["sectors"]:
        cells = []
        for group in backends.values():
            m = group["sectors"][sector]["all"]["metrics"]["pulse_pe"]["relative"]
            text = "unsupported" if m["mean"] is None else f"{100*m['mean']:+.3f} ± {100*m['sample_sd']:.3f}%"
            cells.append(f"{text} (K={m['K']})")
        lines.append("| " + " | ".join([sector, *cells]) + " |")
    for record in snapshots.values():
        _require(_record(record["path"]) == record, "input changed before output creation")
    out.mkdir(parents=True, exist_ok=False)
    (out / "seed-audit.json").write_text(payload)
    (out / "table.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", required=True, nargs="+", metavar="AUDIT.json")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.audits, args.outdir)
    print(f"{result['run_config']}: conditional seed diagnostics written to {args.outdir}")


if __name__ == "__main__":
    main()
