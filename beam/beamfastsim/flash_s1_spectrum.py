"""Saved-only absolute matched S1 spectra and event-tail diagnostics, NOT closure.

Run ``python -m beamfastsim.flash_s1_spectrum --fast DIR --g4 A.npz B.npz
--out spectrum.json``. DIR needs event_totals.npz, flux.json, births.npz and
foil_exit.npz; G4 inputs are disjoint flash_g4_primary outputs. No transport,
normalization fit, event cut, timing cut, or removal of loopers/unknowns occurs.
Only identity/flag columns are loaded from the potentially large births archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .flash_attenuation import al_mu_rho
from .flash_inclusive_check import EVENT, PRODUCT, join, keys
from .flash_s1_reference import compare as compare_reference

# Fixed coarse bins include an open-ended final bin; no fitted count scale.
ENERGY_LOWER_EDGES_MEV = np.array([0., .001, .01, .1, 1., 3., 10., 30., 100.])
TAIL_THRESHOLDS_MEV = np.array([0., .001, .1, 1., 10., 30., 100.])
CONCENTRATION_COUNTS = np.array([1, 5, 10, 20])
EVENT_TAIL_THRESHOLDS = {"photons": np.array([1., 2., 5., 10.]),
                         "energy_mev": np.array([1., 10., 30., 100.])}
SOURCE_COLUMNS = tuple("source_" + k for k in (*EVENT, "particle_key")) + ("loopers", "overflow", "capped")
FAST_COLUMNS = ("parent", "energy", "weight", "foil_exit", "attenuation_supported", "path_in_foil_mm")
G4_COLUMNS = (*PRODUCT, "key", "pre_energy_mev", "birth_energy_mev", "primary_brems",
              "uncollided_exit", "exit", "unresolved", "unknown_ancestry", "unknown_history",
              "ambiguous", "birth_inside_geometry", "no_recorded_exit", "prior_non_geometry_steps",
              *("event_" + k for k in EVENT), "evt_primary_exits", "evt_primary_energy", "report_json")


def _column(table, name, n, *, kind=None):
    value = np.asarray(table[name])
    if value.shape != (n,) or (kind is not None and value.dtype.kind not in kind):
        raise ValueError(f"{name}: invalid column shape or dtype")
    return value


def _finite_nonnegative(value, name):
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name}: values must be finite and nonnegative")


def event_histograms(event_index, energy, weight, n):
    """Vectorized (event, common energy bin) count and MeV-sum matrices.

    All bins are [lower, upper), including exactly 100 MeV in [100, infinity).
    Event rows with no selected photons are explicitly zero. Inputs have already
    passed the identity/selection gates in compare; no photon-level SE is used.
    """
    b = np.searchsorted(ENERGY_LOWER_EDGES_MEV, energy, side="right") - 1
    nb = len(ENERGY_LOWER_EDGES_MEV)
    flat = event_index * nb + b
    return {"photons": np.bincount(flat, weights=weight, minlength=n * nb).reshape(n, nb),
            "energy_mev": np.bincount(flat, weights=weight * energy, minlength=n * nb).reshape(n, nb)}


def paired_columns(fast, g4):
    """Absolute column sums and paired event influence SEs; no fitted scale.

    Total variance = n/(n-1) sum_i (d_i - mean(d))**2, d_i = fast_i - g4_i.
    Computing tails BEFORE this reduction retains cross-bin/event correlations.
    """
    fast, g4 = np.asarray(fast, float), np.asarray(g4, float)
    if fast.ndim != 2 or fast.shape != g4.shape or len(fast) < 1:
        raise ValueError("paired matrices must share a nonempty full event ledger")
    _finite_nonnegative(fast, "fast")
    _finite_nonnegative(g4, "g4")
    n = len(fast)
    difference = fast - g4
    out = {}
    for name, values in (("fast", fast), ("g4", g4), ("difference", difference)):
        total = values.sum(axis=0)
        out[name + "_total"] = total.tolist()
        out[name + "_per_event"] = (total / n).tolist()
        se = (np.sqrt(n / (n - 1) * np.sum((values - values.mean(axis=0)) ** 2, axis=0))
              if n > 1 else None)
        out[name + "_total_event_se"] = se.tolist() if se is not None else [None] * fast.shape[1]
    out["fast_nonzero_events"] = np.count_nonzero(fast, axis=0).tolist()
    out["g4_nonzero_events"] = np.count_nonzero(g4, axis=0).tolist()
    out["both_zero_events"] = np.count_nonzero((fast == 0) & (g4 == 0), axis=0).tolist()
    return out


def _ranked_events(events, fast, g4, looper_events, top_events):
    """Columnar [threshold][rank] diagnostics, never one JSON row per event."""
    difference = fast - g4
    out = {}
    for name, score in (("fast", fast), ("g4", g4), ("absolute_difference", np.abs(difference))):
        order = np.argsort(-score, axis=0, kind="stable")
        ordered = np.take_along_axis(score, order, axis=0)
        cumulative = np.cumsum(ordered, axis=0)
        total = score.sum(axis=0)
        k = np.minimum(CONCENTRATION_COUNTS, len(events))
        mass = cumulative[k - 1]
        fractions = np.divide(mass, total, out=np.zeros_like(mass, dtype=float), where=total > 0)
        chosen = order[:top_events]
        positive = np.take_along_axis(score, chosen, axis=0) > 0
        # Bounded object conversion preserves full uint64 IDs for nullable ranks.
        identities = np.where(positive[:, :, None], events[chosen].astype(object), None)
        out[name] = dict(
            event_ids=identities.transpose(1, 0, 2).tolist(),
            fast=np.where(positive, np.take_along_axis(fast, chosen, axis=0).astype(object), None).T.tolist(),
            g4=np.where(positive, np.take_along_axis(g4, chosen, axis=0).astype(object), None).T.tolist(),
            difference=np.where(positive, np.take_along_axis(difference, chosen, axis=0).astype(object), None).T.tolist(),
            fast_looper_event=(looper_events[chosen] & positive).T.tolist(),
            score_total=total.tolist(),
            top_k_actual=k.tolist(),
            top_k_score_sum=mass.T.tolist(),
            top_k_score_fraction=np.where(total[None, :] > 0, fractions.astype(object), None).T.tolist(),
            top_k_signed_difference=np.cumsum(np.where(ordered > 0, np.take_along_axis(difference, order, axis=0), 0.), axis=0)[k - 1].T.tolist())
    return out


def _same_totals(actual, saved, name):
    saved = np.asarray(saved, float)
    if saved.shape != actual.shape or not np.allclose(actual, saved, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name}: photon reconstruction disagrees with saved event totals")


def _raw_summary(mask, energy, event, n):
    # Unknown G4 energies are birth energies, never guessed exit energies.
    valid = mask & np.isfinite(energy) & (energy >= 0)
    return dict(rows=int(mask.sum()), events=int(np.count_nonzero(np.bincount(event[mask], minlength=n))),
                known_energy_mev=float(energy[valid].sum()), unknown_energy_rows=int((mask & ~valid).sum()))


def compare(fast, fast_report, source, photons, references, *, top_events=5):
    """Compare saved arrays, retaining the entire exact integer event universe.

    ``fast``/``fast_report`` are event_totals.npz/flux.json; ``source`` is the
    SOURCE_COLUMNS subset of births.npz; ``photons`` is FAST_COLUMNS from
    foil_exit.npz. ``references`` are flash_g4_primary NPZ dictionaries. Returns
    JSON-ready columnar spectra and cumulative tails, plus the unchanged reference
    comparison. No synthetic zero event IDs are manufactured from a count alone.
    """
    if type(top_events) is not int or not 1 <= top_events <= 20:
        raise ValueError("top_events must be an integer from 1 through 20")
    event = np.asarray(fast["event"])
    if event.ndim != 2 or event.shape[1] != 3 or not references:
        raise ValueError("full event triples and nonempty G4 references required")
    # Reuse the established exact-event, disjoint-shard and source-truncation gates.
    reference = compare_reference(fast, fast_report, references)
    n = len(event)
    ek = keys({k: event[:, i] for i, k in enumerate(EVENT)}, EVENT)
    ns = len(source["source_run"])
    src = {k: _column(source, "source_" + k, ns, kind="iu") for k in (*EVENT, "particle_key")}
    sk = keys(src, (*EVENT, "particle_key"))
    join(sk, sk)
    parent_event = join(keys(src, EVENT), ek)
    if np.any(parent_event < 0):
        raise ValueError("fast source event absent from full event universe")
    loopers = _column(source, "loopers", ns, kind="b")
    if np.any(_column(source, "overflow", ns, kind="iu")) or np.any(_column(source, "capped", ns, kind="b")):
        raise ValueError("saved fast source is truncated")
    if int(loopers.sum()) != fast_report["loopers"]:
        raise ValueError("saved looper flags disagree with flux report")
    looper_events = np.bincount(parent_event[loopers], minlength=n) > 0

    nf = len(photons["parent"])
    parent = _column(photons, "parent", nf, kind="iu")
    if np.any(parent < 0) or np.any(parent >= ns):
        raise ValueError("invalid fast photon parent index")
    fe, fw = (_column(photons, k, nf).astype(float) for k in ("energy", "weight"))
    foil, supported = (_column(photons, k, nf, kind="b") for k in ("foil_exit", "attenuation_supported"))
    distance = _column(photons, "path_in_foil_mm", nf)
    _finite_nonnegative(fe, "fast photon energy")
    _finite_nonnegative(fw, "fast photon weight")
    expected_support = foil & np.isfinite(al_mu_rho(fe)) & np.isfinite(distance) & (distance >= 0)
    if (np.any(fe <= 0) or np.any(fw > 1) or np.any(fw[~supported] != 0)
            or not np.array_equal(expected_support, supported)):
        raise ValueError("invalid saved fast attenuation support/weight convention")
    fi = parent_event[parent]
    fh = event_histograms(fi, fe, fw, n)
    for name, saved in (("photons", "uncollided_photons"), ("energy_mev", "uncollided_energy_mev")):
        _same_totals(fh[name].sum(axis=1), fast[saved], "fast " + saved)
    for name, weight in (("transparent_photons", foil.astype(float)), ("transparent_energy_mev", foil * fe)):
        if name in fast:
            _same_totals(np.bincount(fi, weights=weight, minlength=n), fast[name], "fast " + name)

    # Only flat comparison columns are merged; shard-local row pointers are not.
    for r in references:
        rk = keys({k: r["event_" + k] for k in EVENT}, EVENT)
        rows = len(r["key"])
        for column in G4_COLUMNS:
            if column != "report_json":
                _column(r, column, len(rk) if column.startswith(("event_", "evt_")) else rows)
        if np.any(join(keys(r, EVENT), rk) < 0):
            raise ValueError("G4 photon absent from its own shard event universe")
    g = {k: np.concatenate([r[k] for r in references]) for k in G4_COLUMNS if k != "report_json"}
    ng = len(g["key"])
    for key in (*PRODUCT, "key"):
        _column(g, key, ng, kind="iu")
    pk = keys(g, (*PRODUCT, "key"))
    join(pk, pk)
    gi = join(keys(g, EVENT), ek)
    if np.any(gi < 0):
        raise ValueError("G4 photon absent from full event universe")
    for name in ("primary_brems", "uncollided_exit", "exit", "unresolved", "unknown_ancestry", "unknown_history",
                 "ambiguous", "birth_inside_geometry", "no_recorded_exit"):
        _column(g, name, ng, kind="b")
    prior = _column(g, "prior_non_geometry_steps", ng, kind="iu")
    unresolved = g["unknown_ancestry"] | g["unknown_history"] | g["ambiguous"] | ~g["birth_inside_geometry"]
    expected_selection = g["primary_brems"] & g["exit"] & ~unresolved & (prior == 0)
    if (not np.array_equal(g["unresolved"], unresolved)
            or not np.array_equal(g["no_recorded_exit"], ~g["exit"])
            or not np.array_equal(g["uncollided_exit"], expected_selection)):
        raise ValueError("G4 saved selection disagrees with primary/history/spatial gates")
    selected = g["uncollided_exit"]
    ge = _column(g, "pre_energy_mev", ng).astype(float)
    birth_energy = _column(g, "birth_energy_mev", ng).astype(float)
    _finite_nonnegative(ge[selected], "G4 selected energy")
    if np.any(ge[selected] <= 0):
        raise ValueError("G4 selected energy must be positive")
    gh = event_histograms(gi[selected], ge[selected], np.ones(selected.sum()), n)
    gledger = keys({k: g["event_" + k] for k in EVENT}, EVENT)
    index = join(ek, gledger)
    for name, saved in (("photons", "evt_primary_exits"), ("energy_mev", "evt_primary_energy")):
        _same_totals(gh[name].sum(axis=1), g[saved][index], "G4 " + saved)

    unsupported = foil & ~supported
    tail_bin = np.searchsorted(ENERGY_LOWER_EDGES_MEV, TAIL_THRESHOLDS_MEV)
    spectra, tails, event_tails = {}, {}, {}
    for name in ("photons", "energy_mev"):
        spectra[name] = paired_columns(fh[name], gh[name])
        ft = np.cumsum(fh[name][:, ::-1], axis=1)[:, ::-1][:, tail_bin]
        gt = np.cumsum(gh[name][:, ::-1], axis=1)[:, ::-1][:, tail_bin]
        tails[name] = dict(**paired_columns(ft, gt),
                           ranked_events=_ranked_events(event, ft, gt, looper_events, top_events))
        thresholds = EVENT_TAIL_THRESHOLDS[name]
        ftotal, gtotal = fh[name].sum(axis=1)[:, None], gh[name].sum(axis=1)[:, None]
        fabove, gabove = ftotal >= thresholds, gtotal >= thresholds
        event_tails[name] = dict(
            event_total_thresholds=thresholds.tolist(),
            events_above=paired_columns(fabove.astype(float), gabove.astype(float)),
            contribution_above=paired_columns(ftotal * fabove, gtotal * gabove),
            both_above_events=np.count_nonzero(fabove & gabove, axis=0).tolist(),
            ranked_events=_ranked_events(event, ftotal * fabove, gtotal * gabove, looper_events, top_events))

    reports = [json.loads(str(r["report_json"])) for r in references]
    provenance_keys = ("input", "geometry_file", "offline_root", "expected_events", "input_events",
                       "boundary_tolerance_mm", "geometry_process_ids")
    spatial_keys = ("boundary_tolerance_mm", "continuity_direction_tolerance", "continuity_energy_rtol", "reconstruction")
    provenance = []
    for r, report in zip(references, reports):  # file loop, not event/photon loop
        spatial = json.loads(str(r["spatial_report_json"])) if "spatial_report_json" in r else {}
        if any(k in report and report[k] != len(r["event_run"]) for k in ("expected_events", "input_events")):
            raise ValueError("G4 shard expected-event count mismatch")
        provenance.append(dict(primary={k: report[k] for k in provenance_keys if k in report},
                               spatial={k: spatial[k] for k in spatial_keys if k in spatial}))
    result = dict(
        schema="flash_s1_spectrum/v1", reference_comparison=reference, input_events=n,
        energy_bin_lower_edges_mev=ENERGY_LOWER_EDGES_MEV.tolist(), energy_bin_final_upper_mev=None,
        tail_thresholds_mev=TAIL_THRESHOLDS_MEV.tolist(), spectra=spectra, tails=tails, event_tails=event_tails,
        fast_unmodelled={name: _raw_summary(mask, fe, fi, n) for name, mask in (
            ("below_1kev_foil_births", unsupported & (fe < .001)),
            ("other_unsupported_foil_births", unsupported & (fe >= .001)),
            ("nonfoil_births", ~foil))},
        g4_unknown={name: _raw_summary(mask, birth_energy, gi, n) for name, mask in (
            ("unresolved_rows", g["unresolved"]),
            ("primary_unresolved_rows", g["primary_brems"] & g["unresolved"]),
            ("primary_no_recorded_exit_rows", g["primary_brems"] & g["no_recorded_exit"]))},
        provenance=dict(
            bins="Fixed coarse decades with 3/30 MeV subdivisions; [lower, upper), last [100, infinity). No data-tuned edges.",
            tails="Inclusive E >= threshold, cumulative within each event BEFORE computing paired SE. No selection is changed.",
            event_tails="Estimator concentration only: fast weighted conditional expectations versus realized G4 totals. Thresholding these is NOT a comparison of physical event multiplicity, survival probability or detector acceptance. ALL input event rows retained; never used to cut/renormalize the inclusive comparison.",
            production_thresholds="docs/flash-attenuation.md documents current fast brems cutoff 0.00099 MeV, below XCOM support; docs/flash-inclusive.md documents G4 0.010 mm range production cut (NOT an equal photon-energy threshold). Neither is imposed here. Archived reports do not certify production thresholds; consult the original frozen run configuration for exact provenance.",
            normalization="Absolute saved sums and per-full-input-event means; no fitted normalization or count matching.",
            fast_energy="Saved birth energy: unchanged for the weighted uncollided original photon at first foil exit.",
            g4_energy="Saved pre_energy_mev on accepted first-exit segment; existing primary/history/spatial selection unchanged.",
            attenuation="Saved foil_exit & attenuation_supported gates. Al XCOM [0.001, 100000] MeV; open K-edge gap (0.0015595, 0.0015596) unsupported. Zero weight outside support is a placeholder, NOT proven absorption.",
            uncertainty="Paired event influence: SE(total difference) = sqrt(n/(n-1) * sum((fast_i-g4_i-mean(fast-g4))^2)). No independent-photon or independent-bin assumption. n=1 SE is null; empirical zero SE is not validated certainty.",
            rankings="Threshold-major, then descending score; ties retain fast ledger order. Null IDs mean zero score, not a missing input event. absolute_difference concentration divides by sum(abs(event difference)), NOT the cancellation-prone signed excess.",
            event_id_fields=list(EVENT), concentration_top_k=CONCENTRATION_COUNTS.tolist(), top_events=top_events,
            g4_unknown="Overlapping inventories, using known BIRTH energy only, not inferred exit energy or absorption. All event rows retained.",
            fast={k: fast_report[k] for k in ("source", "geometry", "offline_root", "seed", "h_mm", "s_max_mm", "max_steps", "capacity", "transport_parameters") if k in fast_report},
            g4_shards=provenance))
    return result


def _read(path, columns=None):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in (archive.files if columns is None else columns)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--g4", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-events", type=int, default=5, help="1..20 ranked IDs per tail/estimator; default 5")
    args = parser.parse_args(argv)
    start = perf_counter()
    fast = _read(args.fast / "event_totals.npz")
    source = _read(args.fast / "births.npz", SOURCE_COLUMNS)
    photons = _read(args.fast / "foil_exit.npz", FAST_COLUMNS)
    references = []
    for path in args.g4:
        with np.load(path, allow_pickle=False) as archive:
            columns = (*G4_COLUMNS, "spatial_report_json") if "spatial_report_json" in archive else G4_COLUMNS
            references.append({k: archive[k] for k in columns})
    result = compare(fast, json.loads((args.fast / "flux.json").read_text()), source, photons, references,
                     top_events=args.top_events)
    result["provenance"]["inputs"] = dict(fast=str(args.fast), g4=list(map(str, args.g4)))
    result["provenance"]["elapsed_seconds"] = perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.out), input_events=result["input_events"],
                          seconds=result["provenance"]["elapsed_seconds"],
                          metrics=result["reference_comparison"]["metrics"]), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
