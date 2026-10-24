"""The fast CRV chain without art or files: CRY primaries -> transport -> tier 1b -> photons -> SiPM charges ->
waveforms/digis -> pulses -> coincidence clusters (interface notes: wiki sim/crv-fastsim-project, "Sim-to-reco
interface").

Stages and the modules that own them:
  transport.transport (tier 1a muons), tier1b.sample (library secondaries + non-muon showers),
  response.photons[_jax] (photons at the SiPMs), response.sipm[_jax] (SiPM charges),
  reco.waveform[_jax] (waveforms -> 12-sample digis), reco.pulsefit + reco.chain_jax (pulses, coincidences).
Glue added here: the SiPM-status mask (stock CrvSiPMChargeGenerator skips status bits 0 and 2), one TDC0 per
(event, FEB) uniform in a 12.5 ns clock period (stock CrvWaveformsGenerator), digis grouped into reco waveforms with
the CRVCalib constants of their channel.

Event key: (subrun << 20) | event (fits the reco's int64 event keys); a SiPM label is event key * CHAN_MULT + channel
(CHAN_MULT 32768: the Run 2 CRV has 21,376 channels; any power of two above the channel count keeps the label order).
"""
import time as _time
from dataclasses import dataclass, field

import numpy as np

from . import geometry, tables
from .response import photons as PH
from .response import sipm as SI

CHAN_MULT = 32768
PERIOD = 12.5


@dataclass
class Conditions:
    """Per-channel CRVCalib constants and channel map (CSV: channel,pedestal,pulsearea,pulseheight,timeoffset,status,
    roc,feb,febch, e.g. dumped from the reference job's crvdump/calib) and the CRVPhoton yield deviations."""
    pedestal: np.ndarray
    pulsearea: np.ndarray
    pulseheight: np.ndarray
    timeoffset: np.ndarray
    status: np.ndarray
    roc: np.ndarray
    feb: np.ndarray
    deviation: np.ndarray

    @staticmethod
    def from_csv(calib_csv, photon_csv=None):
        a = np.loadtxt(calib_csv, delimiter=",", comments="#")
        n = int(a[:, 0].max()) + 1
        col = lambda j, dt=float: np.zeros(n, dt) if a.size == 0 else _dense(a[:, 0], a[:, j], n, dt)
        dev = np.zeros(n)
        if photon_csv:
            p = np.loadtxt(photon_csv, delimiter=",", comments="#")
            dev[p[:, 0].astype(int)] = p[:, 1]
        return Conditions(col(1), col(2), col(3), col(4), col(5, np.int64), col(6, np.int64), col(7, np.int64), dev)


def _dense(ch, v, n, dt):
    out = np.zeros(n, dt)
    out[ch.astype(int)] = v
    return out


@dataclass
class ChainConfig:
    response: PH.ResponseConfig = field(default_factory=PH.ResponseConfig)
    finder: object = None               # reco.coinc.FinderConfig of the run configuration (None: extracted v04 cuts)
    sipm: SI.SiPMConfig = field(default_factory=SI.SiPMConfig)
    window: tuple = (0.0, 1e5)          # offspill extracted event window [ns]
    noise: bool = True                  # electronics noise in the waveforms
    jax: bool = False                   # JAX twins for photons, SiPM charges and waveforms
    rounds: int = 3                     # pulse-reco kernel rounds (chain_jax)
    unbucketed_synth: bool = False      # JAX waveforms in one call padded to the largest cluster (OOM on real files)
    fused_response: bool = False        # JAX: photons -> SiPM charges on the device in one pass (response.device, as
                                        # reco-r3 and the P2 bench; statistically equal to the two-stage twins, V4)
    waveform: object = None             # reco.waveform.WaveformConfig (None: the stock prolog_v12 values)
    background: object = None           # reco.background.BackgroundModel: out-of-time deposits added before clustering
    background_scale: float = 1.0       # multiplies the background model's net rate (knob background_scale)


# light-propagation keys (extracted v04): sectors, TAG = every muon tagger, and modules (photons.bar_keys)
PROP_KEYS = (("EX", "T1", "T2", "TAG") + tuple(f"EX{i}" for i in range(4)) + ("T10", "T11", "T20", "T21")
             + tuple(f"M{i}" for i in range(1, 9)))
# V-R5 / BO knobs: name -> (config block, field). Stock FHiCL names in KNOB_FCL (run_crvonly_knob.sh arm).
KNOBS = {"photon_yield_scale": ("response", "photon_yield_scale"), "time_constant": ("sipm", "time_constant"),
         "cross_talk": ("sipm", "cross_talk"), "min_voltage": ("waveform", "min_voltage"),
         "wf_noise": ("waveform", "noise"), "pe_stretch": ("waveform", "single_pe_stretch"),
         "background_scale": (None, "background_scale"),
         **{f"yield_{n}": ("sector_yield", n) for n in ("EX", "T1", "T2", "TAG")},
         **{f"speed_{n}": ("speed_ratio", n) for n in PROP_KEYS},
         **{f"atten_{n}": ("atten_ratio", n) for n in PROP_KEYS}}
KNOB_FCL = {"photon_yield_scale": "physics.producers.CrvPhotons.photonYieldScaleFactor",
            "time_constant": "physics.producers.CrvSiPMCharges.timeConstant",
            "cross_talk": "physics.producers.CrvSiPMCharges.CrossTalkProb",
            "min_voltage": "physics.producers.CrvWaveforms.minVoltage",
            "wf_noise": "physics.producers.CrvWaveforms.noise",
            "pe_stretch": "physics.producers.CrvWaveforms.singlePEWaveformStretchFactor",
            "background_scale": None,     # fast chain only: no stock counterpart
            # per-sector scintillation yield [photons/MeV]: element of CrvPhotons.scintillationYields (sector order EX, T1,
            # T2, M1..M8; TAG sets all eight taggers). Fibre diameter and scintillator batch enter here (DocDB 56425:
            # CRV-T 1.8 mm 39794, CRV-T 1.4 mm 30482, batch-2 modules 28610)
            **{f"yield_{n}": "physics.producers.CrvPhotons.scintillationYields" for n in ("EX", "T1", "T2", "TAG")},
            # light propagation relative to the lookup tables: no FHiCL parameter, the stock arm takes a CRVPropagation
            # DbService text table (KNOB_TABLE; scripts/write_crv_propagation.py --knob)
            **{f"{q}_{n}": None for q in ("speed", "atten") for n in PROP_KEYS}}
# knob -> (CRVPropagation column, key) for the stock arm (Offline PR crv-fiber-atten, prodition CRVLightPropagation)
KNOB_TABLE = {**{f"speed_{n}": ("signalSpeedRatio", n) for n in PROP_KEYS},
              **{f"atten_{n}": ("attenuationLengthRatio", n) for n in PROP_KEYS}}


def apply_knobs(cfg, knobs):
    """ChainConfig with knob overrides {name: value} (KNOBS). Unknown names raise."""
    import dataclasses
    from .reco import waveform as WF
    if cfg.waveform is None:
        cfg = dataclasses.replace(cfg, waveform=WF.WaveformConfig())
    for k, v in (knobs or {}).items():
        if k not in KNOBS:
            raise KeyError(f"unknown knob {k}; known: {sorted(KNOBS)}")
        blk, fld = KNOBS[k]
        if blk == "sector_yield":
            names = [f"M{i}" for i in range(1, 9)] if fld == "TAG" else [fld]
            secs = dict(cfg.response.sectors)
            for nm in names:
                secs[nm] = dataclasses.replace(secs[nm], yield_per_mev=float(v))
            cfg = dataclasses.replace(cfg, response=dataclasses.replace(cfg.response, sectors=secs))
            continue
        if blk in ("speed_ratio", "atten_ratio"):
            if not float(v) > 0:
                raise ValueError(f"knob {k} must be positive, got {v}")
            d = {**getattr(cfg.response, blk), fld: float(v)}
            cfg = dataclasses.replace(cfg, response=dataclasses.replace(cfg.response, **{blk: d}))
            continue
        if blk is None:
            cfg = dataclasses.replace(cfg, **{fld: float(v)})
        else:
            cfg = dataclasses.replace(cfg, **{blk: dataclasses.replace(getattr(cfg, blk), **{fld: float(v)})})
    return cfg


def event_key(df):
    return (df.subrun.to_numpy().astype(np.int64) << 20) | df.event.to_numpy().astype(np.int64)


def charges(steps, parts, geom, cond, cfg, rng, key=None, tabs=None, pmap=None, timers=None, pdg=None):
    """CrvSteps (+ particles for pdg, or pdg per step given) -> SiPM charges. Returns dict chan (label), time, charge,
    pe, photon (step row)."""
    timers = timers if timers is not None else {}
    if pdg is None:
        pdg = steps.merge(parts[tables.KEY + ["key", "pdg"]].rename(columns={"key": "sp"}), on=tables.KEY + ["sp"],
                          how="left").pdg.fillna(0).to_numpy()
    t0 = _time.perf_counter()
    if cfg.jax and cfg.fused_response:
        import jax
        from .response import device as DV
        key = key if key is not None else jax.random.PRNGKey(0)
        rs = DV.response_batched(steps, pdg, geom, cfg.response, cfg.sipm, tabs, cond.deviation, cond.status,
                                 event_key(steps), rng, key, pmap)
        if any(bool(r["overflow"]) for r in rs):
            timers["response_overflow_batches"] = timers.get("response_overflow_batches", 0) + 1
        Q = DV.to_host_batches(rs) if rs else {k: np.zeros(0) for k in ("chan", "time", "charge", "pe", "photon")}
        del rs
        keep = Q["time"] <= cfg.window[1]
        Q = {k: v[keep] for k, v in Q.items()}
        timers["response_s"] = timers.get("response_s", 0.0) + _time.perf_counter() - t0
        timers["charges"] = timers.get("charges", 0) + len(Q["time"])
        return Q
    if cfg.jax:
        import jax
        from .response import photons_jax, sipm_jax
        key = key if key is not None else jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        s, c, t = photons_jax.make_photons(steps, pdg, geom, cfg.response, tabs, cond.deviation, rng, k1)
    else:
        s, c, t = PH.make_photons(steps, pdg, geom, cfg.response, tabs, cond.deviation, rng)
    timers["photons_s"] = timers.get("photons_s", 0.0) + _time.perf_counter() - t0
    # stock SiPM stage: no charges on channels flagged not connected (bit 0) or no data (bit 2)
    keep = ((cond.status[c] & 0b101) == 0)
    s, c, t = s[keep], c[keep], t[keep]
    if len(c) and int(c.max()) >= CHAN_MULT:
        raise ValueError(f"channel {int(c.max())} >= CHAN_MULT {CHAN_MULT}: SiPM labels would overlap the next event")
    label = event_key(steps)[s] * CHAN_MULT + c
    t0 = _time.perf_counter()
    if cfg.jax:
        Q = sipm_jax.simulate(label, t, s, pmap, cfg.sipm, rng, k2, t_end=cfg.window[1])
    else:
        Q = SI.simulate(label, t, s, pmap, cfg.sipm, rng, t_end=cfg.window[1])
    timers["sipm_s"] = timers.get("sipm_s", 0.0) + _time.perf_counter() - t0
    timers["photons"] = timers.get("photons", 0) + len(s)
    timers["charges"] = timers.get("charges", 0) + len(Q["time"])
    return Q


def tdc0_per_feb(labels, cond, rng):
    """TDC0 [ns] per SiPM label: one uniform draw in [0, 12.5) per (event, ROC, FEB), as CrvWaveformsGenerator."""
    ev, ch = labels // CHAN_MULT, labels % CHAN_MULT
    feb = cond.roc[ch] * 1000 + cond.feb[ch]
    k = ev * 1_000_000 + feb
    u, inv = np.unique(k, return_inverse=True)
    return rng.random(len(u))[inv.ravel()] * PERIOD


def digis(Q, cond, cfg, rng, key=None, timers=None):
    """SiPM charges -> zero-suppressed digis (reco.waveform[_jax]); per digi chan (label), starttdc, adc [n, 12]."""
    from .reco import waveform as WF
    timers = timers if timers is not None else {}
    labels = np.unique(Q["chan"])
    tdc0 = tdc0_per_feb(labels, cond, rng)
    lut = lambda lab: tdc0[np.searchsorted(labels, lab)]
    wcfg = cfg.waveform if cfg.waveform is not None else WF.WaveformConfig()
    t0 = _time.perf_counter()
    if cfg.jax:
        import jax
        from .reco import waveform_jax
        from .reco.coinc_jax import is_oneapi
        if is_oneapi() or not cfg.unbucketed_synth:   # one call padded to the largest cluster needs tens of GiB
            from .reco.cli import _synth_jax_bucketed    # (OOM on A100-40/80, UVA jobs 21732754-60): bucketed calls
            seed = int(rng.integers(2 ** 31))
            D = _synth_jax_bucketed(waveform_jax, {"row": Q["chan"], "time": Q["time"], "charge": Q["charge"]},
                                    lut, wcfg, noise=wcfg.noise if cfg.noise else 0.0, seed=seed, window=cfg.window)
        else:
            D = waveform_jax.synthesize(Q["chan"], Q["time"], Q["charge"], lut, wcfg, window=cfg.window,
                                        rng_key=(key if cfg.noise else None))
    else:
        D = WF.synthesize(Q["chan"], Q["time"], Q["charge"], lut, wcfg, window=cfg.window,
                          rng=(rng if cfg.noise else None))
    timers["waveforms_s"] = timers.get("waveforms_s", 0.0) + _time.perf_counter() - t0
    timers["digis"] = timers.get("digis", 0) + len(D["chan"])
    return D


def reco_waveforms(D, cond):
    """Digis -> reco waveform rows (reco.pulsefit.concat_waveforms grouping) with calibration, sorted by (ev, chan,
    start), the input of reco.chain_jax.run."""
    from .reco import pulsefit
    ev, ch = D["chan"] // CHAN_MULT, D["chan"] % CHAN_MULT
    o = np.lexsort((D["starttdc"], ch, ev))
    ev, ch, st = ev[o], ch[o], D["starttdc"][o]
    adc = np.asarray(D["adc"])[o]
    lens = np.full(len(o), adc.shape[1], np.int64)
    gid, mat, n, start, first = pulsefit.concat_waveforms(ev, ch // 4, ch % 4, st, adc.ravel(), lens)
    c0 = ch[first]
    return {"ev": ev[first], "bar": c0 // 4, "sipm": c0 % 4, "chan": c0, "adc": mat, "n": n, "start": start,
            "ped": cond.pedestal[c0], "area": cond.pulsearea[c0], "height": cond.pulseheight[c0]}


def background_sampler(cfg, rng):
    """Callable(events) -> background pulses for chain_jax.run / the numpy path, or None without a model."""
    if cfg.background is None or cfg.background_scale <= 0:
        return None
    return lambda events: cfg.background.sample(events, cfg.window, rng, rate_scale=cfg.background_scale)


def clusters(W, cond, geom, counters_cfg, cfg, timers=None, detail=False, rng=None):
    """Reco waveforms -> pulses -> coincidence clusters. JAX: reco.chain_jax.run (pipelined pulses + coincidences);
    numpy: reco.pulsefit.reconstruct then reco.coinc.find_event per event (the validated references; no JAX needed).
    Returns (list of cluster dicts with cl_event, cl_type, cl_pe, ..., timers). detail: each dict also carries cl_pos,
    the member table (mem_cl, mem_idx, mem_event) and its flat pulses under "pulses" (reco.ntview input)."""
    from .reco import coinc as C
    timers = timers if timers is not None else {}
    G = C.sector_geom(geom, counters_cfg)
    fcfg = cfg.finder if cfg.finder is not None else C.FinderConfig()
    if cfg.jax:
        from .reco import chain_jax
        return chain_jax.run(W, G, cond.status, n_chunks=max(1, min(8, len(np.unique(W["ev"])) // 1000)),
                             rounds=cfg.rounds, L=12, timers=timers, keep_pulses=detail, with_pos=detail, coinc_cfg=fcfg,
                             add_pulses=background_sampler(cfg, rng if rng is not None else np.random.default_rng()))
    from .reco import pulsefit
    t0 = _time.perf_counter()
    rcfg = pulsefit.RecoConfig()
    P = pulsefit.reconstruct(W["adc"], W["n"], W["start"], W["ped"], W["area"], W["height"], rcfg)
    if rcfg.use_time_offset_db:  # CrvRecoPulsesFinder adds the DB time offset (offspill: TDC0time = 0)
        toff = W["toff"] if "toff" in W else cond.timeoffset[W["chan"]]   # W["toff"]: DB offset (+ time calibration)
        P["time"] = P["time"] + toff[P["wf"]] * rcfg.time_offset_scale
    timers["pulses_s"] = timers.get("pulses_s", 0.0) + _time.perf_counter() - t0
    wf = P["wf"]
    ev = W["ev"][wf]
    o = np.lexsort((P["seq"], wf, ev))
    ev, wf = ev[o], wf[o]
    idx = np.arange(len(ev)) - np.searchsorted(ev, ev)
    bar, sipm, time, pe, seq = W["bar"][wf], W["sipm"][wf], P["time"][o], P["pe"][o], P["seq"][o]
    smp = background_sampler(cfg, rng if rng is not None else np.random.default_rng())
    if smp is not None:
        from .reco.background import merge_pulses
        B = smp(np.unique(W["ev"]))
        ev, idx, bar, sipm, time, pe, seq, _ = merge_pulses(ev, idx, bar, sipm, time, pe, seq, B)
        timers["background_pulses"] = timers.get("background_pulses", 0) + len(B["ev"])
    t0 = _time.perf_counter()
    cl = {k: [] for k in ("cl_event", "cl_type", "cl_pe", "cl_start", "cl_end", "cl_slope", "cl_avgtime", "cl_poscalc")}
    mem = {"cl_pos": [], "mem_cl": [], "mem_idx": [], "mem_event": []}
    ue, lo = np.unique(ev, return_index=True)
    hi = np.r_[lo[1:], len(ev)]
    for e, a, b in zip(ue, lo, hi):
        p = dict(idx=idx[a:b], bar=bar[a:b], sipm=sipm[a:b], time=time[a:b], pe=pe[a:b], seq=seq[a:b])
        for r in C.find_event(p, G, cond.status, fcfg):
            cl["cl_event"].append(e)
            for k, src in (("cl_type", "sectortype"), ("cl_pe", "pe"), ("cl_start", "start"), ("cl_end", "end"),
                           ("cl_slope", "slope"), ("cl_avgtime", "avgtime"), ("cl_poscalc", "poscalc")):
                cl[k].append(r[src])
            if detail:
                n = len(cl["cl_event"]) - 1
                mem["cl_pos"].append(np.asarray(r["pos"], float))
                mem["mem_cl"] += [n] * len(r["pulses"])
                mem["mem_idx"] += list(r["pulses"])
                mem["mem_event"] += [e] * len(r["pulses"])
    timers["coinc_s"] = timers.get("coinc_s", 0.0) + _time.perf_counter() - t0
    timers["pulses"] = timers.get("pulses", 0) + len(ev)
    out = {k: np.asarray(v) for k, v in cl.items()}
    if detail:
        out.update({k: np.asarray(v, np.int64) for k, v in mem.items() if k != "cl_pos"})
        out["cl_pos"] = np.asarray(mem["cl_pos"], float).reshape(-1, 3)
        out["pulses"] = dict(ev=ev, idx=idx, bar=bar, sipm=sipm, seq=seq, pe=pe, time=time)
    return [out], timers
