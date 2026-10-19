"""Offline's own MakeCrvSiPMCharges through PyROOT on given photon arrivals: the stock oracle for sipm.py.

A small C++ helper (declared through cling) feeds sorted (channel, time, photon id) arrays to MakeCrvSiPMCharges::Simulate
SiPM by SiPM and collects the responses, so there is no per-photon Python. Needs the same muse/Offline setup as
stockphotons.py. Thermal noise is simulated only on SiPMs that have photons (the stock module visits every SiPM; with
the production ThermalRate 0 this makes no difference).
"""
import os

import numpy as np

from .stockphotons import _load

_CODE = r"""
#include "Offline/CRVResponse/inc/MakeCrvSiPMCharges.hh"
#include "CLHEP/Random/MixMaxRng.h"
namespace crvfs {
struct SiPMOracle {
  CLHEP::MixMaxRng eng; CLHEP::RandFlat flat; CLHEP::RandPoissonQ pois; mu2eCrv::MakeCrvSiPMCharges sim;
  std::vector<long> ochan, oph; std::vector<double> otime, ocharge, ope;
  SiPMOracle(long seed, const std::string &map, int nx, int ny, double vov, double tau, double cap,
             double p1, double p2, double t0p, double t1p, double t0l, double t1l, double thermal, double ct)
    : eng(seed), flat(eng), pois(eng), sim(flat, pois, map) {
    mu2eCrv::MakeCrvSiPMCharges::ProbabilitiesStruct p;
    p._avalancheProbParam1=p1; p._avalancheProbParam2=p2; p._trapType0Prob=t0p; p._trapType1Prob=t1p;
    p._trapType0Lifetime=t0l; p._trapType1Lifetime=t1l; p._thermalRate=thermal; p._crossTalkProb=ct;
    std::vector<std::pair<int,int>> dead;
    for(int x=18; x<22; ++x) for(int y=18; y<22; ++y) dead.emplace_back(x,y);
    sim.SetSiPMConstants(nx, ny, vov, tau, cap, p, dead);
  }
  void run(long n, const long *chan, const double *t, const long *ph, double start, double end) {
    ochan.clear(); oph.clear(); otime.clear(); ocharge.clear(); ope.clear();
    long i=0;
    while(i<n) {
      long j=i; std::vector<std::pair<double,size_t>> photons;
      while(j<n && chan[j]==chan[i]) { photons.emplace_back(t[j], size_t(j-i)); ++j; }
      std::vector<mu2eCrv::SiPMresponse> resp;
      sim.Simulate(photons, resp, start, end);
      for(auto const &r : resp) {
        ochan.push_back(chan[i]); otime.push_back(r._time); ocharge.push_back(r._charge); ope.push_back(r._chargeInPEs);
        oph.push_back(r._darkNoise ? -1 : ph[i + long(r._photonIndex)]);
      }
      i=j;
    }
  }
};
}
"""
_declared = False


def run(chan, time, photon, cfg, photon_map_path, seed=1, t_start=0.0, t_end=1e5):
    """Stock SiPM charges for photon arrivals (any order). Returns dict like sipm.simulate."""
    global _declared
    R = _load()
    if not _declared:
        R.gInterpreter.Declare(_CODE)
        _declared = True
    o = np.lexsort((time, chan))
    c = np.ascontiguousarray(np.asarray(chan)[o], np.int64)
    t = np.ascontiguousarray(np.asarray(time, float)[o])
    p = np.ascontiguousarray(np.asarray(photon)[o], np.int64)
    sim = R.crvfs.SiPMOracle(int(seed), photon_map_path, cfg.n_pixels_x, cfg.n_pixels_y, cfg.overvoltage,
                             cfg.time_constant, cfg.capacitance, cfg.p1, cfg.p2, cfg.trap0_prob, cfg.trap1_prob,
                             cfg.trap0_lifetime, cfg.trap1_lifetime, cfg.thermal_rate, cfg.cross_talk)
    sim.run(len(c), c, t, p, float(t_start), float(t_end))

    def arr(v, dt):
        return np.fromiter(v, dt, v.size()) if v.size() else np.zeros(0, dt)
    return {"chan": arr(sim.ochan, np.int64), "time": arr(sim.otime, float), "charge": arr(sim.ocharge, float),
            "pe": arr(sim.ope, float), "photon": arr(sim.oph, np.int64)}


def photon_map_path():
    from . import lookup
    return os.path.join(lookup.lookup_dir(), "photonMap.root")
