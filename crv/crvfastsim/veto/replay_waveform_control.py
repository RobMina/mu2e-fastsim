"""Diagnostic waveform ledger with FEB-keyed phase and channel/absolute-TDC-keyed noise."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..reco import pulsefit, waveform as WF
from .replay_sipm_control import keyed_uniform

MULT = 32768
PHASE_DOMAIN = "feb-phase-v1"
NOISE_U1_DOMAIN = "sample-noise-radius-v1"
NOISE_U2_DOMAIN = "sample-noise-angle-v1"


def labelled(frame):
    """Unambiguous selected-event/channel labels, with no run or block mixing."""
    ev, ch = np.asarray(frame.event), np.asarray(frame.chan)
    if np.any(~np.isfinite(ev)) or np.any(~np.isfinite(ch)) or np.any(ev != np.floor(ev)) or np.any(ch != np.floor(ch)):
        raise ValueError("nonintegral event/channel")
    if np.any((ev < 0) | (ev >= 2**20) | (ch < 0) | (ch >= MULT)):
        raise ValueError("event/channel out of label domain")
    return ev.astype(np.int64) * MULT + ch.astype(np.int64)


def phase_table(channels, seed):
    """Each event/ROC/FEB receives one immutable uniform phase, including quiet channels."""
    if channels.duplicated(["event", "chan"]).any():
        raise ValueError("duplicate channel conditions")
    phase = keyed_uniform(seed, PHASE_DOMAIN, channels.event.to_numpy(), channels.roc.to_numpy(), channels.feb.to_numpy()) * 12.5
    return channels.assign(label=labelled(channels), phase=phase)


def sample_noise(seed, event, chan, tdc, sigma):
    """Box-Muller normals keyed by physical channel and clock index, never cluster ordinal."""
    u1 = keyed_uniform(seed, NOISE_U1_DOMAIN, event, chan, tdc)
    u2 = keyed_uniform(seed, NOISE_U2_DOMAIN, event, chan, tdc)
    return sigma * np.sqrt(-2 * np.log(np.maximum(u1, np.finfo(float).tiny))) * np.cos(2 * np.pi * u2)


class FixedNoise:
    """Adapter for the existing synthesize implementation; exactly one bounded noise call."""
    def __init__(self, values):
        self.values = np.asarray(values)
        self.calls = 0

    def normal(self, loc, scale, size):
        if loc != 0 or tuple(size) != self.values.shape or self.calls:
            raise ValueError("unexpected waveform noise call/batching")
        self.calls += 1
        return self.values.copy()


def synthesize(charges, channels, seed, cfg=None, window=(200., 1500.)):
    """Existing fast synthesis plus independently checked full sample/threshold ledgers."""
    cfg = cfg or WF.WaveformConfig()
    ch = phase_table(channels, seed).set_index("label")
    if not ch.index.is_unique:
        raise ValueError("duplicate channel label")
    q = charges.copy()
    q["label"] = labelled(q)
    q = q.sort_values(["label", "time"], kind="stable").reset_index(drop=True)
    if not q.label.isin(ch.index).all() or not np.isfinite(q[["time", "charge"]]).all().all() or (q.charge < 0).any():
        raise ValueError("invalid charge/conditions input")
    if len(q) > 2_000_000:
        raise ValueError("diagnostic single-batch charge limit exceeded")
    label, time, charge = (q[k].to_numpy() for k in ("label", "time", "charge"))
    tmpl = WF.single_pe_template(cfg.single_pe_file, cfg.single_pe_precision, cfg.single_pe_stretch, cfg.single_pe_max_time)
    cid = WF.charge_clusters(label, time, cfg)
    C = int(cid[-1]) + 1 if len(cid) else 0
    first = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]]) if C else np.zeros(0, np.int64)
    last = np.r_[first[1:] - 1, len(cid) - 1] if C else np.zeros(0, np.int64)
    labs = label[first]
    phase = ch.loc[labs, "phase"].to_numpy()
    start = np.ceil((time[first] - 12.5 - phase) / 12.5) * 12.5 + phase
    starttdc = np.rint((start - phase) / 12.5).astype(np.int64)
    count = np.bincount(cid, minlength=C)
    prekeep = count * tmpl.max() >= cfg.min_voltage
    estimate = np.floor((time[last] - time[first] + cfg.single_pe_max_time) / 12.5).astype(np.int64)
    reach = int(np.ceil(cfg.single_pe_max_time / 12.5)) + 2
    j0 = np.maximum(np.ceil((time - start[cid]) / 12.5), 0).astype(np.int64)
    J = j0[:, None] + np.arange(reach)[None, :]
    elapsed = J * 12.5 + start[cid, None] - time[:, None]
    ti = np.rint(elapsed / cfg.single_pe_precision).astype(np.int64)
    ok = (elapsed >= 0) & (ti < len(tmpl)) & prekeep[cid, None]
    used = np.zeros(C, np.int64)
    if ok.any():
        np.maximum.at(used, np.repeat(cid, reach)[ok.ravel()], J.ravel()[ok.ravel()] + 1)
    length = np.maximum(estimate, used)
    L = int(length.max()) + cfg.samples_zs if C else 0
    if L > 1024:
        raise ValueError("diagnostic waveform length bound exceeded")
    noiseless = np.zeros((C, L))
    value = tmpl[np.clip(ti, 0, len(tmpl) - 1)] * (charge / cfg.single_pe_reference_charge)[:, None]
    np.add.at(noiseless, (np.repeat(cid, reach)[ok.ravel()], J.ravel()[ok.ravel()]), value.ravel()[ok.ravel()])
    valid = np.arange(L)[None, :] < length[:, None]
    tdc = starttdc[:, None] + np.arange(L)[None, :]
    noise = sample_noise(seed, labs[:, None] // MULT, labs[:, None] % MULT, tdc, cfg.noise)
    fixed_noise = FixedNoise(noise)
    phase_lookup = lambda labels: ch.loc[labels, "phase"].to_numpy()
    D = WF.synthesize(label, time, charge, phase_lookup, cfg, window, rng=fixed_noise)
    if fixed_noise.calls != 1:
        raise ValueError("waveform synthesis did not consume fixed noise")
    full = noiseless + noise * valid
    if not np.array_equal(D["cluster_chan"], labs) or not np.array_equal(D["cluster_tstart"], start):
        raise ValueError("independent waveform cluster ledger disagrees")
    hit = (full > cfg.min_voltage) & valid
    trigger = np.zeros_like(hit)
    for delta in (-2, -1, 0, 1, 2):
        dest = np.arange(L)
        source = dest + delta
        inside = (source >= 0) & (source < L)
        trigger[:, dest[inside]] |= hit[:, source[inside]]
    trigger &= valid & prekeep[:, None]
    physical = start[:, None] + 12.5 * np.arange(L)[None, :]
    trigger &= (physical >= window[0]) & (physical <= window[1])
    next_allowed = np.zeros(C, np.int64)
    expected_channel, expected_tdc = [], []
    for sample in range(L):
        choose = np.flatnonzero(trigger[:, sample] & (next_allowed <= sample))
        expected_channel.append(labs[choose])
        expected_tdc.append(starttdc[choose] + sample)
        next_allowed[choose] = sample + cfg.samples_zs
    ec = np.concatenate(expected_channel) if expected_channel else np.zeros(0, np.int64)
    et = np.concatenate(expected_tdc) if expected_tdc else np.zeros(0, np.int64)
    order = np.lexsort((et, ec))
    if not np.array_equal(D["chan"], ec[order]) or not np.array_equal(D["starttdc"], et[order]):
        raise ValueError("independent zero-suppression start list disagrees")
    r = np.repeat(np.arange(C), L)
    samples = pd.DataFrame({"cluster": r, "event": np.repeat(labs // MULT, L), "chan": np.repeat(labs % MULT, L),
        "tdc": tdc.ravel(), "time_relative": (tdc * 12.5 + phase[:, None]).ravel(),
        "valid": valid.ravel(), "cluster_prekeep": np.repeat(prekeep, L),
        "voltage_noiseless": noiseless.ravel(), "noise": noise.ravel(), "voltage": full.ravel()})
    samples["above_zs"] = samples.valid & samples.cluster_prekeep & (samples.voltage > cfg.min_voltage)
    samples["zs_margin_v"] = samples.voltage - cfg.min_voltage
    samples["precast_adc"] = np.floor(samples.voltage * cfg.adc_conversion + cfg.pedestal + 0.5)
    if len(D["chan"]):
        valid_samples = samples[samples.valid]
        sample_index = pd.MultiIndex.from_frame(valid_samples[["event", "chan", "tdc"]])
        if sample_index.has_duplicates:
            raise ValueError("overlapping valid sample cluster ledger")
        starts_requested = pd.MultiIndex.from_arrays([D["chan"] // MULT, D["chan"] % MULT, D["starttdc"]])
        start_loc = sample_index.get_indexer(starts_requested)
        if np.any(start_loc < 0):
            raise ValueError("digi starts outside full sample ledger")
        dc = valid_samples.cluster.to_numpy()[start_loc]
        di = D["starttdc"] - starttdc[dc]
        ix = di[:, None] + np.arange(cfg.samples_zs)[None, :]
        voltage = np.where(ix < length[dc, None], full[dc[:, None], ix], 0)
        np.testing.assert_array_equal(D["volt"], voltage)
        safe_adc = np.clip(np.floor(voltage * cfg.adc_conversion + cfg.pedestal + .5), cfg.min_adc, cfg.max_adc).astype(np.int64)
        if not np.array_equal(D["adc"], safe_adc):
            raise ValueError("fast ADC narrowing differs from fixed-stock safe clamp")
    clusters = pd.DataFrame({"cluster": np.arange(C), "event": labs // MULT, "chan": labs % MULT,
        "phase": phase, "first_charge": time[first], "last_charge": time[last], "charges": count,
        "starttdc": starttdc, "length": length, "prekeep": prekeep})
    return D, samples, clusters, ch.reset_index()


def reconstruct(D, conditions, full=True):
    """Full sample concatenation and pulse fit, with diagnostic sample margins."""
    ev, chan = D["chan"] // MULT, D["chan"] % MULT
    order = np.lexsort((D["starttdc"], chan, ev))
    ev, chan, tdc, adc = ev[order], chan[order], D["starttdc"][order], D["adc"][order]
    lens = np.full(len(tdc), adc.shape[1], np.int64)
    _, mat, n, start, first = pulsefit.concat_waveforms(ev, chan // 4, chan % 4, tdc, adc.ravel(), lens)
    cc = conditions.set_index(["event", "chan"])
    if not cc.index.is_unique:
        raise ValueError("duplicate event/channel conditions")
    index = pd.MultiIndex.from_arrays([ev[first], chan[first]])
    c = cc.loc[index]
    cfg = pulsefit.RecoConfig()
    fit = pulsefit.reconstruct_full if full else pulsefit.reconstruct
    P = fit(mat, n, start, c.pedestal.to_numpy(), c.pulsearea.to_numpy(), c.pulseheight.to_numpy(), cfg)
    near = P.pop("wf_near_threshold")
    p = pd.DataFrame(P)
    if len(p):
        wi = p.wf.to_numpy().astype(int)
        p["event"], p["chan"] = ev[first][wi], chan[first][wi]
        p["time_relative"] = p.time + c.timeoffset.to_numpy()[wi]
        p["time_reco"] = p.time_relative + c.w.to_numpy()[wi]
        p["channel_ignored"] = (c.status.to_numpy()[wi].astype(int) & 2) != 0
    else:
        for key in ("event", "chan", "time_relative", "time_reco", "channel_ignored"):
            p[key] = pd.Series(dtype=bool if key == "channel_ignored" else float)
    valid = np.arange(mat.shape[1])[None, :] < n[:, None]
    pedestal = np.where(c.pedestal.to_numpy() - (mat[:, 0] if len(n) else []) > cfg.pedestal_undershoot_threshold,
                        mat[:, 0] if len(n) else [], c.pedestal.to_numpy())
    ymax = np.max(np.where(valid, mat - pedestal[:, None], -np.inf), axis=1) if len(n) else np.zeros(0)
    waveforms = pd.DataFrame({"wf": np.arange(len(n)), "event": ev[first], "chan": chan[first], "starttdc": start,
        "n": n, "pedestal_used": pedestal, "max_adc_minus_pedestal": ymax,
        "peak_margin_adc": ymax - cfg.min_adc_difference, "near_threshold": near})
    sample_rows = np.repeat(np.arange(len(n)), n)
    column = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    concat_samples = pd.DataFrame({"wf": sample_rows, "event": ev[first][sample_rows], "chan": chan[first][sample_rows],
        "tdc": start[sample_rows] + column, "adc": mat[sample_rows, column]})
    return p, waveforms, concat_samples
