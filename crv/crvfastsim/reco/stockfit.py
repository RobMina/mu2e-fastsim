"""Run Offline's own MakeCrvRecoPulses (ROOT TGraph::Fit / Minuit2) on waveforms through PyROOT: the stock oracle for
pulsefit without an art job. Needs a muse environment whose Offline build has libmu2e_CRVReco.so (GPVM, e.g. kkmain).
"""
import os

import numpy as np

_loaded = None


def _load():
    global _loaded
    if _loaded is not None:
        return _loaded
    import ROOT
    work = os.environ.get("MUSE_WORK_DIR", "")
    build = os.environ.get("MUSE_BUILD_BASE", "")
    inc = [os.path.join(work, "Offline", "CRVReco", "inc"), work]
    for d in inc:
        ROOT.gInterpreter.AddIncludePath(d)
    lib = None
    for root in (os.path.join(work, build, "Offline", "lib"),):
        cand = os.path.join(root, "libmu2e_CRVReco.so")
        if os.path.exists(cand):
            lib = cand
    if lib is None:
        raise RuntimeError("libmu2e_CRVReco.so not found: run inside a built muse area")
    ROOT.gSystem.Load(lib)
    ROOT.gInterpreter.Declare('#include "Offline/CRVReco/inc/MakeCrvRecoPulses.hh"')
    _loaded = ROOT
    return ROOT


def reconstruct(adc, n, start_tdc, pedestal, calib_area, calib_height, cfg, period=12.5):
    """Same interface and output keys as pulsefit.reconstruct (the subset the stock class exposes).

    `pedestal` is the pedestal actually used (the undershoot rule is applied by the caller, as the art module does).
    """
    R = _load()
    mk = R.mu2eCrv.MakeCrvRecoPulses(cfg.min_adc_difference, cfg.default_beta, cfg.min_beta, cfg.max_beta,
                                     cfg.max_time_difference, cfg.min_pulse_height_ratio, cfg.max_pulse_height_ratio,
                                     cfg.le_time_factor, cfg.samples_before, cfg.samples_after)
    out = {k: [] for k in ("wf", "seq", "pe", "peph", "time", "height", "beta", "chi2", "letime", "failed", "zerondf")}
    vec = R.std.vector("short")()
    for w in range(adc.shape[0]):
        vec.clear()
        for v in adc[w, :n[w]]:
            vec.push_back(int(v))
        mk.SetWaveform(vec, int(start_tdc[w]), period, float(pedestal[w]), float(calib_area[w]), float(calib_height[w]))
        m = len(mk.GetPEs())
        for j in range(m):
            out["wf"].append(w)
            out["seq"].append(j)
            out["pe"].append(mk.GetPEs()[j])
            out["peph"].append(mk.GetPEsPulseHeight()[j])
            out["time"].append(mk.GetPulseTimes()[j])
            out["height"].append(mk.GetPulseHeights()[j])
            out["beta"].append(mk.GetPulseBetas()[j])
            out["chi2"].append(mk.GetPulseFitChi2s()[j])
            out["letime"].append(mk.GetLEtimes()[j])
            out["failed"].append(bool(mk.GetFailedFits()[j]))
            out["zerondf"].append(bool(mk.GetZeroNdfs()[j]))
    return {k: np.asarray(v) for k, v in out.items()}
