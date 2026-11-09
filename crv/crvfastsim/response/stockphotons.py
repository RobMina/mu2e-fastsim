"""Offline's own MakeCrvPhotons through PyROOT: the stock oracle for photons.py on the same steps, without an art job.

Needs a muse/Offline setup (GPVM: `muse setup SimJob MDC2025ay`): libmu2e_CRVResponse.so and the Offline headers are
taken from the Offline release behind $MUSE_WORK_DIR (or $CRVFS_OFFLINE_RELEASE). Per step it calls MakePhotons with the
same local coordinates, beta, charge and reflector as CrvPhotonGenerator, and the same per-channel yield deviation.
A Python loop over steps (the oracle is for validation samples, not speed).
"""
import glob
import os

import numpy as np

_R = None


def _release():
    rel = os.environ.get("CRVFS_OFFLINE_RELEASE")
    if rel:
        return rel
    work = os.environ.get("MUSE_WORK_DIR", "")
    back = os.path.join(work, "backing")
    return os.path.realpath(back) if os.path.exists(back) else work


def _load():
    global _R
    if _R is not None:
        return _R
    import ROOT
    rel = _release()
    libs = glob.glob(os.path.join(rel, "build", "*", "Offline", "lib", "libmu2e_CRVResponse.so"))
    if not libs:
        raise RuntimeError(f"libmu2e_CRVResponse.so not found under {rel}: set up a muse/Offline environment")
    ROOT.gInterpreter.AddIncludePath(rel)
    ROOT.gSystem.Load("libCLHEP.so")
    ROOT.gSystem.Load(libs[0])
    ROOT.gInterpreter.Declare('#include "Offline/CRVResponse/inc/MakeCrvPhotons.hh"')
    _R = ROOT
    return ROOT


class StockPhotons:
    def __init__(self, cfg, directory, seed=1):
        R = _load()
        self.R = R
        self.eng = R.CLHEP.MixMaxRng(int(seed))
        self.flat, self.gauss, self.pois = R.CLHEP.RandFlat(self.eng), R.CLHEP.RandGaussQ(self.eng), R.CLHEP.RandPoissonQ(self.eng)
        self.cfg = cfg
        self.makers = {}
        for name, sc in cfg.sectors.items():
            mk = R.mu2eCrv.MakeCrvPhotons(self.flat, self.gauss, self.pois)
            mk.LoadLookupTable(os.path.join(directory, f"LookupTable_{sc.table}"), 0)
            mk.SetScintillationYield(sc.yield_per_mev)
            self.makers[name] = mk

    def run(self, s_loc, e_loc, t0, t1, beta, charge, edep, length, bar, sector_names, deviation):
        """Per step: MakePhotons on the local start/end. Returns (step index, channel, time) of every photon."""
        V = self.R.CLHEP.Hep3Vector
        out_s, out_c, out_t = [], [], []
        for i in range(len(bar)):
            if not (np.isfinite(t0[i]) and np.isfinite(t1[i])):
                continue  # CrvPhotonGenerator skips steps with a NaN time
            name = sector_names[i]
            sc = self.cfg.sectors[name]
            mk = self.makers[name]
            for s in range(4):
                dev = (deviation[bar[i] * 4 + s] * self.cfg.photon_yield_variation_scale + 1.0) * self.cfg.photon_yield_scale
                mk.SetPhotonYieldDeviation(float(dev), s)
            mk.MakePhotons(V(*map(float, s_loc[i])), V(*map(float, e_loc[i])), float(t0[i]), float(t1[i]), float(beta[i]),
                           float(charge[i]), float(edep[i]), float(length[i]), int(sc.reflector))
            for s in range(4):
                times = mk.GetArrivalTimes(s)
                n = times.size()
                if n:
                    out_s.append(np.full(n, i))
                    out_c.append(np.full(n, bar[i] * 4 + s))
                    out_t.append(np.fromiter(times, float, n))
        if not out_s:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        return np.concatenate(out_s), np.concatenate(out_c), np.concatenate(out_t)
