"""Export track-tagged EX crossings from an extracted, field-off EventNtuple with the track-CRV association.

Tracks and crossings come from the collaboration's track-CRV association (Offline ``TrkCrvMatch``, EventNtuple
``<trk>crv`` and ``<trk>crvxings``): the crossing state on the EX sector mid-plane and, per layer, the layer probe's
predicted bar, position along the bar and track time. This module does not extrapolate tracks itself; the only local
geometry is the straight chord through each predicted bar (from the association's crossing position and direction),
for the path length and the across-bar edge margin.

The association does not carry per-SiPM pulses: its layer probe sums the selected bar's in-window pulses over both
ends and stores one time difference. The per-SiPM charge and time therefore come from ``crvpulses`` on the
association's predicted bar (exact channel, sequence 0, frozen PE and time cuts); this is the labelled fallback
(``pulse_match = "crvpulses_exact_sipm"``). The probe's selected-bar offset (``probe_dbar``) flags crossings whose
pulse the matcher found on a neighbouring bar (pointing mismatch); they are ineligible by default.

Inputs need every CRV pulse (``crv.keepUnclusteredPulses: true``): with clustered pulses only, a missed pulse is
confounded with a pulse outside a coincidence, and the export refuses unless told otherwise.
"""

import argparse

import numpy as np

from .. import geometry, readout

# EventNtuple detector frame (tracker centre) to Mu2e coordinates, extracted geometry (same as reco.vr4eff).
DET_TO_MU2E = np.array([-3904.0, 0.0, 24171.0])
KKLINE = 3
TRACK_CUTS = {"nactive": 20, "fitcon": 0.01, "chi2": 3.0, "nplanes": 0}
NO_PULSE_DBAR = 127
PULSE_MATCH = "crvpulses_exact_sipm"


def _leaf(record, prefix, name):
    for key in (f"{prefix}.{name}", name):
        if key in record.fields:
            return record[key]
    raise KeyError(f"EventNtuple record {prefix} has no field {name}")


def _np(array, regular=False):
    import awkward as ak

    if regular:
        array = ak.to_regular(array, axis=-1)
    return ak.to_numpy(array)


def _select_tracks(trk, cuts, prefix):
    """Flat (all-event) index of one quality KKLine track per event (-1: none). Ranked without CRV data."""
    import awkward as ak

    def leaf(name):
        return _np(ak.flatten(_leaf(trk, prefix, name)))

    counts = _np(ak.num(_leaf(trk, prefix, "status"), axis=1))
    event = np.repeat(np.arange(len(counts)), counts)
    nactive, fitcon, ndof, chisq = (leaf(k) for k in ("nactive", "fitcon", "ndof", "chisq"))
    chi2ndof = np.divide(chisq, ndof, out=np.full(len(ndof), np.inf), where=ndof > 0)
    quality = ((leaf("fitalg") == KKLINE) & (leaf("status") > 0) & (leaf("goodfit") > 0)
               & (nactive >= cuts["nactive"]) & (leaf("nplanes") >= cuts["nplanes"])
               & np.isfinite(fitcon) & (fitcon >= cuts["fitcon"]) & (chi2ndof >= 0) & (chi2ndof <= cuts["chi2"]))
    chosen = np.full(len(counts), -1)
    use = np.flatnonzero(quality)
    if use.size:
        # Most active hits, then lowest chi2/ndof, then input order.
        order = use[np.lexsort((use, chi2ndof[use], -nactive[use].astype(np.int64), event[use]))]
        first = np.r_[True, event[order][1:] != event[order][:-1]]
        chosen[event[order][first]] = order[first]
    return counts, chosen


def association_rows(ntuple, fit="trk", trk_cuts=None):
    """Read the selected track's association crossings and all CRV pulses from one EventNtuple.

    Returns flat arrays: per event (``run``, ``subrun``, ``event``, ``trk_selected``, ``trk_probe``), per crossing of
    the selected track (``x_*``, detector frame; ``x_probe_*`` are (n, 4) per layer) and per pulse (``p_*``).
    """
    import awkward as ak
    import uproot

    cuts = dict(TRACK_CUTS, **(trk_cuts or {}))
    with uproot.open(ntuple) as root:
        t = root["EventNtuple/ntuple"]
        branches = set(t.keys(recursive=False))
        missing = [b for b in (f"{fit}crv", f"{fit}crvxings") if b not in branches and b + "." not in branches]
        if missing:
            raise ValueError(f"EventNtuple has no track-CRV association branches {missing}: produce the ntuple with "
                             f"trkCrvMatchTags for the {fit} branch (EventNtuple >= v06_17_00)")

        def record(name):
            # A record branch (TTree split object or RNTuple field), or flat top-level '<name>.<leaf>' branches.
            for key in (name, name + "."):
                if key in branches:
                    return t[key].array()
            flat = [b for b in branches if b.startswith(name + ".")]
            if not flat:
                raise KeyError(f"EventNtuple has no {name} branch")
            return t.arrays(flat)

        evt, trk = record("evtinfo"), record(fit)
        crv, xings, pulses = record(f"{fit}crv"), record(f"{fit}crvxings"), record("crvpulses")
    counts, chosen = _select_tracks(trk, cuts, fit)
    if (not np.array_equal(_np(ak.num(_leaf(crv, f"{fit}crv", "matched"), axis=1)), counts)
            or not np.array_equal(_np(ak.num(xings, axis=1)), counts)):
        raise ValueError("EventNtuple track and track-CRV association counts do not match")
    matched = _np(ak.flatten(_leaf(crv, f"{fit}crv", "matched"))).astype(bool)
    probe = _np(ak.flatten(_leaf(crv, f"{fit}crv", "probe"))).astype(np.int64)
    sel = np.flatnonzero(chosen >= 0)
    gtrk = chosen[sel]
    trk_probe = np.full(len(counts), -1)
    trk_probe[sel] = np.where(matched[gtrk], probe[gtrk], -1)
    good = matched[gtrk] & (probe[gtrk] >= 1)
    sel, gtrk = sel[good], gtrk[good]
    xt = ak.flatten(xings, axis=1)[gtrk]
    nx = _np(ak.num(xt, axis=1))
    X = ak.flatten(xt)
    pre = f"{fit}crvxings"

    def xl(name, regular=False):
        return _np(_leaf(X, pre, name), regular)

    def vec(name):
        v = _leaf(X, pre, name)["fCoordinates"]
        return np.column_stack([_np(v[f"f{k}"]) for k in "XYZ"]).astype(float).reshape(-1, 3)

    out = dict(run=_np(_leaf(evt, "evtinfo", "run")), subrun=_np(_leaf(evt, "evtinfo", "subrun")),
               event=_np(_leaf(evt, "evtinfo", "event")), trk_selected=chosen >= 0, trk_probe=trk_probe,
               x_ev=np.repeat(sel, nx))
    if len(X):
        out.update(x_sector=xl("sector").astype(np.int64), x_inbounds=xl("inbounds").astype(bool),
                   x_dedgeu=xl("dedgeu"), x_dedgev=xl("dedgev"), x_sigu=xl("sigu"), x_sigv=xl("sigv"),
                   x_time=xl("time"), x_pos=vec("pos"), x_dir=vec("dir"))
        for k in ("bar", "dbar", "upred", "tpred", "pe"):
            out[f"x_probe_{k}"] = xl(f"probe_{k}", regular=True).reshape(len(X), -1)
    else:
        out.update(x_sector=np.zeros(0, np.int64), x_inbounds=np.zeros(0, bool),
                   **{f"x_{k}": np.zeros(0) for k in ("dedgeu", "dedgev", "sigu", "sigv", "time")},
                   x_pos=np.zeros((0, 3)), x_dir=np.zeros((0, 3)),
                   **{f"x_probe_{k}": np.zeros((0, 4)) for k in ("bar", "dbar", "upred", "tpred", "pe")})
    n = _np(ak.num(_leaf(pulses, "crvpulses", "barId")))
    out["p_ev"] = np.repeat(np.arange(len(n)), n)
    for key, leaf in (("p_bar", "barId"), ("p_sipm", "SiPMId"), ("p_pe", "PEs"), ("p_time", "time"),
                      ("p_seq", "sequenceIndex"), ("p_hit", "crvHitIndex")):
        out[key] = _np(ak.flatten(_leaf(pulses, "crvpulses", leaf)))
    return out


def observations(geom, present, status, rows, *, run, margin_mm, threshold_pe, time_window_ns, pe_per_mm,
                 attenuation_mm, speed_mm_ns, delay_ns, exclude_neighbour=True, allow_clustered_only=False):
    """One row per (association EX crossing, layer probe bar, SiPM). Side 0 is the low length-axis end."""
    positive = (threshold_pe, time_window_ns, pe_per_mm, attenuation_mm, speed_mm_ns)
    if (run < 0 or not np.isfinite((*positive, margin_mm, delay_ns)).all()
            or min(positive) <= 0 or margin_mm < 0):
        raise ValueError("invalid run, nominal response, or selection")
    present, status = np.asarray(present), np.asarray(status)
    nchan = 4 * geom.nbars
    if present.shape != (nchan,) or present.dtype.kind != "b" or status.ndim != 1 or len(status) < nchan:
        raise ValueError("physical mask and status must cover all geometry channels")
    if status.dtype.kind not in "iu" or np.any(status < 0):
        raise ValueError("status must contain nonnegative integer bit masks")
    names = list(geom.sector_names)
    if "EX" not in names:
        raise ValueError("extracted geometry must contain the EX sector")
    ex = names.index("EX")
    if len(rows["event"]) == 0 or np.any(np.asarray(rows["run"]) != run):
        raise ValueError("input EventNtuple must contain only the specified run")
    hit = np.asarray(rows.get("p_hit", np.zeros(0)))
    if len(hit) and not (hit < 0).any() and not allow_clustered_only:
        raise ValueError("crvpulses holds clustered pulses only (no crvHitIndex = -1): a miss would include pulses "
                         "outside coincidences; produce the ntuple with crv.keepUnclusteredPulses: true")

    # (crossing, layer) pairs with an EX predicted bar.
    xs = np.flatnonzero((np.asarray(rows["x_sector"]) == ex) & np.asarray(rows["x_inbounds"], bool))
    pbar = np.asarray(rows["x_probe_bar"]).astype(np.int64)
    ix, il = np.nonzero(pbar[xs] >= 0)
    ix = xs[ix]
    bar = pbar[ix, il]
    if np.any(bar >= geom.nbars):
        raise ValueError("association probe bar outside the geometry")
    use = geom.sector[bar] == ex
    ix, il, bar = ix[use], il[use], bar[use]
    p = np.asarray(rows["x_pos"], float)[ix] + DET_TO_MU2E
    d = np.asarray(rows["x_dir"], float)[ix]
    upred = np.asarray(rows["x_probe_upred"], float)[ix, il]
    tpred = np.asarray(rows["x_probe_tpred"], float)[ix, il]
    dbar = np.asarray(rows["x_probe_dbar"]).astype(np.int64)[ix, il]
    norm = np.linalg.norm(d, axis=1)
    axes = geom.local_to_world[bar]
    k = np.arange(len(bar))
    thick, width, along = axes.T
    half = geom.halflen[bar]
    if np.any(margin_mm >= np.minimum(half[k, width], half[k, along])):
        raise ValueError("edge margin leaves no interior in an EX bar")
    with np.errstate(divide="ignore", invalid="ignore"):
        d = d / norm[:, None]
        dthick = d[k, thick]
        rel = p - geom.center[bar]
        mid = -rel[k, thick] / dthick
        a = rel + mid[:, None] * d
        dh = half[k, thick] / np.abs(dthick)
        # Enter and leave through the two broad faces, inside the margin across and along the bar.
        keep = (np.isfinite(p).all(axis=1) & np.isfinite(norm) & (norm > 0) & (np.abs(dthick) > 1e-12)
                & np.isfinite(upred) & np.isfinite(tpred)
                & (np.abs(a[k, width]) + dh * np.abs(d[k, width]) < half[k, width] - margin_mm)
                & (np.abs(upred) + dh * np.abs(d[k, along]) < half[k, along] - margin_mm))
        du_line = a[k, along] - upred
    half_along = half[k, along]
    ix, il, bar, upred, tpred, dbar, dh, du_line, half_along = (
        v[keep] for v in (ix, il, bar, upred, tpred, dbar, dh, du_line, half_along))
    if not len(bar):
        raise ValueError("no central EX crossings")
    ie = np.asarray(rows["x_ev"]).astype(np.int64)[ix]
    sign = np.array([1., -1., 1., -1.])
    chan = (4 * bar[:, None] + np.arange(4)).ravel()
    rep = lambda v: np.repeat(v, 4)
    distance = (half_along[:, None] + upred[:, None] * sign).ravel()
    length, tref, ie4, dbar4 = rep(2 * dh), rep(tpred), rep(ie), rep(dbar)
    nominal_time = tref + delay_ns + distance / speed_mm_ns
    neighbour = (dbar4 != 0) & (dbar4 != NO_PULSE_DBAR)
    pe, tm = np.full(len(chan), np.nan), np.full(len(chan), np.nan)
    enabled = present[chan] & ((status[chan] & 0b111) == 0)
    if exclude_neighbour:
        enabled &= ~neighbour
    pp, pt = np.asarray(rows["p_pe"], float), np.asarray(rows["p_time"], float)
    p_ev, p_bar, p_sipm = (np.asarray(rows[k]).astype(np.int64) for k in ("p_ev", "p_bar", "p_sipm"))
    candidates = (np.asarray(rows["p_seq"]) == 0) & np.isfinite(pp) & np.isfinite(pt) & (pp >= threshold_pe)
    if np.any(candidates & ((p_sipm < 0) | (p_sipm > 3) | (p_bar < 0) | (p_bar >= geom.nbars)
                            | (p_ev < 0) | (p_ev >= len(rows["event"])))):
        raise ValueError("invalid stock pulse event, bar or SiPM index")
    pulse_by_key = {}
    for i in np.flatnonzero(candidates):
        pulse_by_key.setdefault((p_ev[i], 4 * p_bar[i] + p_sipm[i]), []).append(i)
    ambiguous = np.zeros(len(chan), bool)
    for i in np.flatnonzero(enabled):
        near = [j for j in pulse_by_key.get((ie4[i], chan[i]), ()) if abs(pt[j] - nominal_time[i]) <= time_window_ns]
        if len(near) == 1:
            pe[i], tm[i] = pp[near[0]], pt[near[0]]
        elif len(near) > 1:
            ambiguous[i] = True
    sigv = rep(np.asarray(rows["x_sigv"], float)[ix])
    return dict(run=np.full(len(chan), run), subrun=np.asarray(rows["subrun"])[ie4],
                event=np.asarray(rows["event"])[ie4], layer=geom.layer[chan // 4], channel=chan,
                present=present[chan], status=status[chan], eligible=enabled & ~ambiguous, ambiguous=ambiguous,
                neighbour=neighbour, seen=np.isfinite(pe), path_mm=length, distance_mm=distance, track_time_ns=tref,
                threshold_pe=np.full(len(chan), threshold_pe),
                nominal_log_pe=np.log(pe_per_mm * length) - distance / attenuation_mm, nominal_time_ns=nominal_time,
                time_half_width_ns=np.full(len(chan), time_window_ns), pe=pe, time_ns=tm,
                probe_dbar=dbar4, sigv_mm=sigv, du_line_mm=rep(du_line),
                pulse_match=np.array(PULSE_MATCH))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntuple")
    parser.add_argument("counters", help="extracted_v04 counters file")
    parser.add_argument("calib", help="per-channel CRVCalib CSV used to build the event sample")
    parser.add_argument("out", help="output NPZ (contains detector data; keep private)")
    parser.add_argument("--run", type=int, required=True, help="run identifier for the single-run input file")
    parser.add_argument("--fit-branch", default="trk", help="EventNtuple track branch holding the KKLine fits")
    parser.add_argument("--keep-neighbour", action="store_true",
                        help="keep crossings whose association probe pulse is on a neighbouring bar")
    parser.add_argument("--allow-clustered-only", action="store_true",
                        help="accept an ntuple without unclustered pulses (biases the misses)")
    for knob in ("margin-mm", "threshold-pe", "time-window-ns", "pe-per-mm", "attenuation-mm", "speed-mm-ns", "delay-ns"):
        parser.add_argument(f"--{knob}", type=float, required=True)
    args = vars(parser.parse_args(argv))
    counters = geometry.find_counters(args.pop("counters"))
    geom = geometry.from_offline_config(counters)
    with open(counters) as stream:
        present = readout.sensor_mask(geom, geometry.parse_simpleconfig(stream.read()))
    calib = np.loadtxt(args.pop("calib"), delimiter=",", ndmin=2)
    if calib.shape[1] < 6 or not len(calib):
        raise ValueError("calibration CSV requires channel and status columns")
    ids, flags = calib[:, [0, 5]].T
    fields = np.stack((ids, flags))
    if (not np.isfinite(fields).all() or np.any(fields < 0) or np.any(fields >= 2.**63)
            or np.any(fields != np.floor(fields))):
        raise ValueError("calibration CSV requires integer channel IDs and status flags")
    chan = ids.astype(np.int64)
    if len(np.unique(chan)) != len(chan) or not np.isin(np.arange(4 * geom.nbars), chan).all():
        raise ValueError("calibration CSV must uniquely cover geometry channels")
    status = np.zeros(4 * geom.nbars, dtype=np.int64)
    use = chan < len(status)
    status[chan[use]] = flags[use].astype(np.int64)
    out = args.pop("out")
    rows = association_rows(args.pop("ntuple"), fit=args.pop("fit_branch"))
    obs = observations(geom, present, status, rows, exclude_neighbour=not args.pop("keep_neighbour"),
                       allow_clustered_only=args.pop("allow_clustered_only"), **args)
    np.savez_compressed(out, **obs)
    print(f"wrote {out}: {int(rows['trk_selected'].sum())} selected tracks, {len(obs['channel'])} sensor rows, "
          f"{int(obs['eligible'].sum())} eligible, {int(obs['seen'].sum())} seen, {int(obs['neighbour'].sum())} "
          f"neighbour, {int(obs['ambiguous'].sum())} ambiguous")


if __name__ == "__main__":
    main()
