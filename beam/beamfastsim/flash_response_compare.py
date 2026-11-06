"""Saved-NPZ anchor diagnostics conditional on fixed G4 inputs, NOT physics closure.

Both anchors are development designs. Resolved-subset energy response is NOT
inclusive; neither residual normalization nor a fitted correction is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .flash_inclusive_check import keys, join

EVENT_FIELDS = ("event_run", "event_subrun", "event_event", "event_ordinal")
ENERGY_EDGES_MEV = np.array([.001, .01, .1, 1., 10., 100., 1000.])
BIN_LABELS = ("underflow", "[.001,.01)", "[.01,.1)", "[.1,1)",
              "[1,10)", "[10,100)", "[100,1000)", "overflow")
TOTAL_FIELDS = {"raw": "tracker", "owned": "owned", "residual": "residual", "unknown": "unknown"}


def _array(data, name, n, *, identity=False, boolean=False, finite=True):
    x = np.asarray(data[name])
    if x.shape != (n,) or x.dtype.kind not in ("iu" if identity else "biuf"):
        raise ValueError(f"{name}: invalid shape or numeric dtype")
    if np.any(x < 0) or (finite and not np.isfinite(x).all()) or np.isinf(x).any():
        raise ValueError(f"{name}: negative or nonfinite values")
    if boolean and not np.isin(x, [0, 1]).all():
        raise ValueError(f"{name}: expected boolean values")
    return x if identity else x.astype(bool if boolean else float)


def _identities(data, fields, n):
    return keys({k: _array(data, k, n, identity=True) for k in fields}, fields)


def _ratio(low, high):
    """Ratio of sums; event residuals retain all within-event correlations."""
    n, a, b = len(low), float(low.sum()), float(high.sum())
    r = b / a if a else None
    se = None
    if a and n > 1:
        z = high - r * low
        se = float(np.sqrt(n / (n - 1) * np.sum((z - z.mean()) ** 2)) / a)
    return r, se


def _concentration(x, event):
    order = np.argsort(-x, kind="stable")[:10]
    total, sq = float(x.sum()), float(x @ x)
    return dict(largest_event_share=float(x[order[0]] / total) if total else None,
                top5_share=float(x[order[:5]].sum() / total) if total else None,
                top10_share=float(x[order].sum() / total) if total else None,
                effective_n=total * total / sq if sq else None,
                top10_events={**{k: event[str(i)][order].tolist() for i, k in enumerate(EVENT_FIELDS)},
                              "ionizing_mev": x[order].tolist()})


def _totals(low, high, event, group, clusters):
    out = {}
    for label in TOTAL_FIELDS:
        l, h = low[label], high[label]
        ratio, se = _ratio(l, h)
        sl, sh = (np.bincount(group, weights=x, minlength=clusters) for x in (l, h))
        item = dict(high_over_low=ratio, paired_delta_ratio_se=se,
                    subrun_cluster_ratio_se=_ratio(sl, sh)[1],
                    ratio_status="zero low total; undefined" if ratio is None else "ok")
        for arm, x in (("low", l), ("high", h)):
            item[arm] = dict(sum_mev=float(x.sum()), mean_mev_per_event=float(x.mean()),
                             event_mean_se=float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
                             nonzero_events=int(np.count_nonzero(x)), concentration=_concentration(x, event))
        out[label] = item
    return out


def _arm(data):
    n = len(np.asarray(data["event_run"]))
    if not n:
        raise ValueError("Empty event ledger: no exposure denominator")
    event = _identities(data, EVENT_FIELDS, n)
    join(event, event)
    ledger = _identities(data, EVENT_FIELDS[:3], n)
    join(ledger, ledger)
    if "input_event_count" in data:
        declared = np.asarray(data["input_event_count"])
        if declared.shape != () or declared.dtype.kind not in "iu" or int(declared) != n:
            raise ValueError("input_event_count disagrees with full event ledger")
    totals = {k: _array(data, "event_" + v + "_ionizing_mev", n) for k, v in TOTAL_FIELDS.items()}
    if not np.allclose(totals["raw"], totals["owned"] + totals["residual"] + totals["unknown"],
                       rtol=1e-9, atol=1e-12):
        raise ValueError("Raw event totals do not equal owned + residual + unknown")
    m = len(np.asarray(data["run"]))
    photon = _identities(data, ("run", "subrun", "event", "product", "key"), m)
    join(photon, photon)
    pe = join(_identities(data, ("run", "subrun", "event"), m), ledger)
    if np.any(pe < 0):
        raise ValueError("Photon exposure missing from event ledger")
    known = _array(data, "photon_response_known", m, boolean=True)
    status = _array(data, "frontier_status", m, identity=True)
    if not np.isin(status, [0, 1, 2, 3]).all() or np.any(known & (status != 1)):
        raise ValueError("Invalid frontier status or known response outside frontier")
    energy = _array(data, "pre_energy_mev", m, finite=False)
    if not np.isfinite(energy[known]).all():
        raise ValueError("Resolved-subset photon energy must be finite")
    response = _array(data, "photon_assigned_ionizing_mev", m)
    if np.any((response > 0) & (status != 1)):
        raise ValueError("Assigned response outside frontier")
    assigned = np.bincount(pe, weights=response, minlength=n)
    if not np.allclose(assigned, totals["owned"], rtol=1e-9, atol=1e-12):
        raise ValueError("Photon exposure mapping disagrees with event owned response")
    nb = len(BIN_LABELS)
    flat = pe[known] * nb + np.searchsorted(ENERGY_EDGES_MEV, energy[known], side="right")
    count = np.bincount(flat, minlength=n * nb).reshape(n, nb)
    edep = np.bincount(flat, weights=response[known], minlength=n * nb).astype(float).reshape(n, nb)
    positive = np.bincount(flat[response[known] > 0], minlength=nb * n).reshape(n, nb).sum(axis=0)
    categories = {}
    for label, use in (("resolved_known", known), ("frontier_response_unknown", (status == 1) & ~known),
                       ("unresolved_frontier", status == 3), ("covered", status == 2), ("non_exit", status == 0)):
        categories[label] = dict(photons=int(use.sum()), events=int(np.unique(pe[use]).size),
                                 assigned_ionizing_mev=float(response[use].sum()))
    return event, totals, count, edep, positive, categories


def _bins(count, edep, positive):
    out = []
    for b, label in enumerate(BIN_LABELS):
        c, e = count[:, b], edep[:, b]
        r, se = _ratio(c, e)
        out.append(dict(bin=label, photon_count=int(c.sum()), photon_events=int(np.count_nonzero(c)),
                        response_sum_mev=float(e.sum()), nonzero_response_events=int(np.count_nonzero(e)),
                        nonzero_response_photons=int(positive[b]), mean_edep_mev_per_photon=r,
                        event_cluster_ratio_se=se,
                        status="unsupported: no photons" if r is None else
                        "zero observed response; unvalidated, not evidence of true zero" if r == 0 else "observed"))
    return out


def _prediction(donor_count, donor_edep, target_count, target_edep):
    c, e, t = donor_count.sum(axis=0), donor_edep.sum(axis=0), target_count.sum(axis=0)
    support = c > 0
    gaps, unvalidated = (~support) & (t > 0), support & (e == 0) & (t > 0)
    response = np.divide(e, c, out=np.zeros_like(e), where=support)
    predicted = t * response
    coefficient = np.divide(t, c, out=np.zeros_like(e), where=support)
    influence = ((donor_edep - donor_count * response) * coefficient + target_count * response).sum(axis=1)
    target = target_edep[:, support].sum(axis=1)
    difference = influence - target
    n = len(influence)
    sum_se = lambda x: float(np.sqrt(n * np.var(x, ddof=1))) if n > 1 else None
    delta = float(predicted.sum() - target.sum())
    delta_se = sum_se(difference)
    return dict(scope="Energy-only resolved-subset cross-design diagnostic; NOT physics closure or design holdout",
                uncertainty="Paired event delta method including donor response and target exposure fluctuations; fragile for rare showers",
                supported_prediction_event_se_mev=sum_se(influence),
                supported_target_observed_mev=float(target.sum()),
                supported_difference_mev=delta, supported_difference_event_se_mev=delta_se,
                supported_difference_pull=delta / delta_se if delta_se and delta_se > 0 else None,
                predicted_mev_by_bin=np.where(gaps, None, predicted).tolist(),
                supported_prediction_sum_mev=float(predicted.sum()),
                full_subset_prediction_sum_mev=None if gaps.any() else float(predicted.sum()),
                observed_target_subset_sum_mev=float(target_edep.sum()),
                support_gap_bins=np.flatnonzero(gaps).tolist(),
                unsupported_target_photons=int(t[gaps].sum()),
                unsupported_target_photons_by_bin=np.where(gaps, t, 0).tolist(),
                zero_observed_response_unvalidated_bins=np.flatnonzero(unvalidated).tolist(),
                zero_observed_response_target_photons=int(t[unvalidated].sum()),
                warning="Zero donor response is an unvalidated plug-in zero, never evidence of true zero; no residual fit")


def compare(lowdict, highdict, *, cross_predict=True):
    """Compare two saved response mappings with identical ordered full event keys.

    Event SEs use sample variance (ddof=1). Ratio SEs use the paired delta
    method; subrun clusters are (run, subrun), with G/(G-1) correction.
    Undefined ratios/SEs are JSON null, never fabricated zeros. Photon rows
    have no ordinal, so a nonunique run/subrun/event ledger is rejected.
    """
    low, high = _arm(lowdict), _arm(highdict)
    event = low[0]
    if not np.array_equal(event, high[0]):
        raise ValueError("Exact ordered (run, subrun, event, ordinal) alignment required; no missing-event zeros")
    cluster = _identities(lowdict, EVENT_FIELDS[:2], len(event))
    unique, group = np.unique(cluster, return_inverse=True)
    result = dict(scope="Flash development-anchor resolved-subset diagnostics; NOT inclusive response or physics closure",
                  uncertainty="Conditional on fixed G4 inputs; event/cluster sampling SEs, NOT full systematics",
                  caveats=["Both anchors informed development; this is NOT design holdout.",
                           "Few-subrun cluster SEs are fragile; fewer than two units gives null SE.",
                           "No missing events imputed; no tuning or fitting to residual totals.",
                           "Unknown, covered and non-exit photons excluded from resolved-subset matrices, not declared zero.",
                           "Zero observed bin response does not establish a true-zero response."],
                  events=len(event), subrun_clusters=len(unique), event_key_fields=list(EVENT_FIELDS),
                  totals=_totals(low[1], high[1], event, group, len(unique)),
                  energy_edges_mev=ENERGY_EDGES_MEV.tolist(), energy_bin_labels=list(BIN_LABELS),
                  energy_convention="pre_energy_mev; left-closed/right-open; underflow < .001; overflow >= 1000 MeV",
                  resolved_subset={}, exposure_categories={})
    for name, arm in (("low", low), ("high", high)):
        result["resolved_subset"][name] = _bins(arm[2], arm[3], arm[4])
        result["exposure_categories"][name] = arm[5]
    result["photon_energy_accounting_diagnostic"] = {}
    for name, data in (("low", lowdict), ("high", highdict)):
        energy = np.asarray(data["pre_energy_mev"], float)
        deposit = np.asarray(data["photon_assigned_ionizing_mev"], float)
        use = (deposit > 0) & np.isfinite(energy) & (energy > 0)
        ratio = np.divide(deposit, energy, out=np.zeros_like(deposit), where=use)
        exceed = use & (deposit > energy * (1 + 1e-5) + 1e-9)
        result["photon_energy_accounting_diagnostic"][name] = dict(
            max_assigned_edep_over_exit_energy=float(ratio.max(initial=0.)),
            exceeding_photons=int(exceed.sum()), assigned_mev_on_exceeding=float(deposit[exceed].sum()),
            note="Sanity diagnostic only; investigate excess before inferring an energy-conserving response")
    if cross_predict:
        result["energy_only_prediction"] = dict(
            low_to_high=_prediction(low[2], low[3], high[2], high[3]),
            high_to_low=_prediction(high[2], high[3], low[2], low[3]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("low", "high", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--no-cross-predict", action="store_true")
    args = parser.parse_args(argv)
    start = perf_counter()
    with np.load(args.low, allow_pickle=False) as low, np.load(args.high, allow_pickle=False) as high:
        result = compare(low, high, cross_predict=not args.no_cross_predict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['events']} paired events; {(perf_counter() - start) / 2:.3f} s/file; NOT physics closure; {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
