"""Export track-tagged EX crossings from an extracted, field-off EventNtuple."""

import argparse

import numpy as np

from .. import geometry, readout


def observations(geom, present, status, rows, *, run, margin_mm, threshold_pe, time_window_ns, pe_per_mm,
                 attenuation_mm, speed_mm_ns, delay_ns):
    """One row per central track/channel crossing. Side 0 is the low length-axis end."""
    from ..reco.vr4eff import DET_Z0, SOLENOID_OFFSET

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
    ex = np.flatnonzero(np.asarray(geom.sector_names)[geom.sector] == "EX")
    if not len(ex):
        raise ValueError("extracted geometry must contain the EX sector")

    if len(rows["event"]) == 0 or np.any(rows["run"] != run):
        raise ValueError("input EventNtuple must contain only the specified run")
    p = np.column_stack([rows[f"trk_{k}"] for k in "xyz"]) + [-SOLENOID_OFFSET, 0, DET_Z0]
    d = np.column_stack([rows[f"trk_d{k}"] for k in "xyz"])
    track_time = np.asarray(rows["trk_time"])
    norm = np.linalg.norm(d, axis=1)
    good = np.isfinite(p).all(axis=1) & np.isfinite(norm) & (norm > 0) & np.isfinite(track_time)
    d = np.divide(d, norm[:, None], out=np.zeros_like(d, dtype=float), where=good[:, None])
    idx, bars, distance, length, tref = [], [], [], [], []
    for b in ex:
        thick, width, along = geom.local_to_world[b]
        half = geom.halflen[b]
        if margin_mm >= min(half[width], half[along]):
            raise ValueError("edge margin leaves no interior in an EX bar")
        # Require the track to enter and leave through the two broad faces, not a side or end.
        use = np.flatnonzero(good & (np.abs(d[:, thick]) > 1e-12))
        rel = p[use] - geom.center[b]
        mid = -rel[:, thick] / d[use, thick]
        dh = half[thick] / np.abs(d[use, thick])
        a = rel + mid[:, None] * d[use]
        farthest = np.abs(a) + dh[:, None] * np.abs(d[use])
        keep = (farthest[:, width] < half[width] - margin_mm) & (farthest[:, along] < half[along] - margin_mm)
        use = use[keep]
        idx.extend(np.repeat(use, 4))
        bars.extend(np.repeat(b, 4 * len(use)))
        distance.extend((half[along] + a[keep, along, None] * [1, -1, 1, -1]).ravel())
        length.extend(np.repeat(2 * dh[keep], 4))
        tref.extend(np.repeat(track_time[use] + mid[keep] / 299.792458, 4))
    ie, bar = np.asarray(idx, dtype=np.int64), np.asarray(bars, dtype=np.int64)
    if not len(ie):
        raise ValueError("no central EX crossings")
    chan = 4 * bar + np.tile(np.arange(4), len(bar) // 4)
    distance, length, tref = map(np.asarray, (distance, length, tref))
    nominal_time = tref + delay_ns + distance / speed_mm_ns
    pe, tm = np.full(len(chan), np.nan), np.full(len(chan), np.nan)
    enabled = present[chan] & ((status[chan] & 0b111) == 0)
    pp, pt = np.asarray(rows["p_pe"]), np.asarray(rows["p_time"])
    candidates = (rows["p_seq"] == 0) & np.isfinite(pp) & np.isfinite(pt) & (pp >= threshold_pe)
    pulse_by_key = {}
    for i in np.flatnonzero(candidates):
        sipm, pb, evt = rows["p_sipm"][i], rows["p_bar"][i], rows["p_ev"][i]
        if not (0 <= sipm < 4 and 0 <= pb < geom.nbars and 0 <= evt < len(p)):
            raise ValueError("invalid stock pulse event, bar or SiPM index")
        pulse_by_key.setdefault((evt, 4 * pb + sipm), []).append(i)
    ambiguous = np.zeros(len(chan), bool)
    for i in np.flatnonzero(enabled):
        near = [j for j in pulse_by_key.get((ie[i], chan[i]), ()) if abs(pt[j] - nominal_time[i]) <= time_window_ns]
        if len(near) == 1:
            pe[i], tm[i] = pp[near[0]], pt[near[0]]
        elif len(near) > 1:
            ambiguous[i] = True
    return dict(run=np.full(len(chan), run), subrun=rows["subrun"][ie], event=rows["event"][ie],
                layer=geom.layer[bar], channel=chan, present=present[chan], status=status[chan],
                eligible=enabled & ~ambiguous, ambiguous=ambiguous, seen=np.isfinite(pe),
                path_mm=length, distance_mm=distance, track_time_ns=tref, threshold_pe=np.full(len(chan), threshold_pe),
                nominal_log_pe=np.log(pe_per_mm * length) - distance / attenuation_mm, nominal_time_ns=nominal_time,
                time_half_width_ns=np.full(len(chan), time_window_ns), pe=pe, time_ns=tm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ntuple")
    parser.add_argument("counters", help="extracted_v04 counters file")
    parser.add_argument("calib", help="per-channel CRVCalib CSV used to build the event sample")
    parser.add_argument("out", help="output NPZ (contains detector data; keep private)")
    parser.add_argument("--run", type=int, required=True, help="run identifier for the single-run input file")
    for knob in ("margin-mm", "threshold-pe", "time-window-ns", "pe-per-mm", "attenuation-mm", "speed-mm-ns", "delay-ns"):
        parser.add_argument(f"--{knob}", type=float, required=True)
    args = vars(parser.parse_args(argv))
    from ..reco.vr4eff import track_pulse_rows

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
    rows = observations(geom, present, status, track_pulse_rows(args.pop("ntuple"), per_sensor=True), **args)
    np.savez_compressed(out, **rows)


if __name__ == "__main__":
    main()
