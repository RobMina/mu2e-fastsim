"""Stage 3 (CPU): write plan -> EventNtuple file with the stock split-TTree layout (PyROOT, Mu2e environment).

Why PyROOT. The stock EventNtuple is a TTree of split class branches (``trk.<leaf>``, ``evtinfo/run``,
``std::vector<std::vector<mu2e::TrkSegInfo>>`` object branches). uproot cannot write split classes or nested vectors
of classes, and its RNTuple output is read by pyutils' ``Importer`` (``tree.arrays(["trk", ...])``) as empty
records. ROOT with the EventNtuple dictionaries of the stock release writes the stock classes themselves. The writer
runs where those dictionaries are: the Mu2e environment of the stock release (a GPVM, or a Slurm CPU node through
``container/offline/offline_uva.sh`` and CVMFS), with ``libeventntuple_dict`` loaded.

How. The donor files (``donor.py``) are opened as one TChain; ``CloneTree(0)`` gives an empty output tree with
exactly the stock branches, classes, split levels and trigger-path branches, sharing the branch buffers with the
chain. For every fast event the donor entry is loaded into the buffers, the fast values of the plan are written over
them (C++ helpers compiled by cling: rotation about the tracker axis, time shift, the electron's reco surfaces,
truth, clusters), and the output tree is filled. ``EventNtuple/n_proc_events`` (one bin, the stored events, as
the stock ntuple job of the stored stream counts them), ``EventNtuple/version`` and ``trkqual_metadata`` are copied from the first donor file and
``EventNtuple/subrunNtuple`` gets one entry per subrun with its stored count; the provenance
JSON goes to ``EventNtuple/fastsim_provenance`` (TNamed).

Numbers only: this module imports nothing from JAX and needs NumPy and ROOT.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TREE = "EventNtuple/ntuple"
SID = {"front": 0, "mid": 1, "back": 2}
PROV = "fastsim_provenance"
COPIED = ("version", "trkqual_metadata")        # TH1I bookkeeping of the stock release, copied from a donor file
# the stored stream of the stock OnSpill digitization (Production digitize/OnSpill.fcl, MDC2025ax): outputs.Output
# SelectEvents = Digitize.SignalTriggers + physics.TriggerablePaths
STREAM_PATTERNS = ("cpr_Trk*_80*", "apr_Trk*_80*", "TriggerableLHPath", "TriggerableTwoTrackPath",
                   "TriggerableCaloPath")

CPP = r'''
#include <vector>
#include <cmath>
#include "EventNtuple/inc/EventInfo.hh"
#include "EventNtuple/inc/TrkInfo.hh"
#include "EventNtuple/inc/TrkInfoMC.hh"
#include "EventNtuple/inc/TrkSegInfo.hh"
#include "EventNtuple/inc/LoopHelixInfo.hh"
#include "EventNtuple/inc/SurfaceStepInfo.hh"
#include "EventNtuple/inc/SimInfo.hh"
#include "EventNtuple/inc/MCStepInfo.hh"
#include "EventNtuple/inc/TrkCaloHitInfo.hh"
#include "EventNtuple/inc/CaloClusterInfo.hh"
#include "EventNtuple/inc/CaloClusterInfoMC.hh"
#include "EventNtuple/inc/CaloHitInfo.hh"
#include "EventNtuple/inc/MVAResultInfo.hh"
#include "EventNtuple/inc/HitCount.hh"
namespace cent {
using V3 = ROOT::Math::XYZVectorF;
inline double wrap(double p) { return std::remainder(p, 2 * M_PI); }
template <class V> inline void rot(V& v, double c, double s) {
  double x = v.x(), y = v.y(); v.SetXYZ(c * x - s * y, s * x + c * y, v.z());
}
// rotate about the tracker axis by dphi and shift times: reco quantities by dtr (microbunch folds included), MC by dt
void transform(std::vector<std::vector<mu2e::TrkSegInfo>>& segs,
               std::vector<std::vector<mu2e::LoopHelixInfo>>& lh,
               std::vector<std::vector<mu2e::SurfaceStepInfo>>& smc,
               std::vector<std::vector<mu2e::SimInfo>>& sims,
               std::vector<std::vector<mu2e::MCStepInfo>>& vd,
               std::vector<mu2e::TrkCaloHitInfo>& ch,
               std::vector<mu2e::TrkInfoMC>& tmc, double dphi, double dt, double dtr) {
  double c = std::cos(dphi), s = std::sin(dphi);
  for (auto& v : segs) for (auto& x : v) { rot(x.pos, c, s); rot(x.mom, c, s); x.time += dtr; }
  for (auto& v : lh) for (auto& x : v) {
    double cx = x.cx, cy = x.cy; x.cx = c * cx - s * cy; x.cy = s * cx + c * cy;
    x.phi0 = wrap(x.phi0 + dphi); x.t0 += dtr; }
  for (auto& v : smc) for (auto& x : v) { rot(x.pos, c, s); rot(x.mom, c, s); x.time += dt; }
  for (auto& v : vd) for (auto& x : v) { rot(x.pos, c, s); rot(x.mom, c, s); x.time += dt; }
  for (auto& v : sims) if (!v.empty()) {   // the track's own particle only; its ancestors keep their histories
    auto& x = v[0]; rot(x.pos, c, s); rot(x.mom, c, s); rot(x.endpos, c, s); rot(x.endmom, c, s); x.time += dt; }
  for (auto& x : ch) { rot(x.poca, c, s); rot(x.mom, c, s); x.ptoca += dtr; x.ctime += dtr; }
  for (auto& x : tmc) { double cx = x.cx, cy = x.cy; x.cx = c * cx - s * cy; x.cy = s * cx + c * cy;
                        x.phi0 = wrap(x.phi0 + dphi); x.t0 += dt; }
}
void set_seg(mu2e::TrkSegInfo& x, double px, double py, double pz, double qx, double qy, double qz, double t) {
  x.pos.SetXYZ(px, py, pz); x.mom.SetXYZ(qx, qy, qz); x.time = t; }
void set_lh(mu2e::LoopHelixInfo& x, double maxr, double d0, double tandip, double rad, double lam, double cx,
            double cy, double phi0, double t0) {
  x.maxr = maxr; x.d0 = d0; x.tanDip = tandip; x.rad = rad; x.lam = lam; x.cx = cx; x.cy = cy; x.phi0 = phi0;
  x.t0 = t0; }
void set_step(mu2e::SurfaceStepInfo& x, double px, double py, double pz, double qx, double qy, double qz, double t) {
  x.pos.SetXYZ(px, py, pz); x.mom.SetXYZ(qx, qy, qz); x.time = t; }
void set_sim(mu2e::SimInfo& x, double px, double py, double pz, double qx, double qy, double qz, double t) {
  x.pos.SetXYZ(px, py, pz); x.mom.SetXYZ(qx, qy, qz); x.time = t; }
void set_calohit(mu2e::TrkCaloHitInfo& x, bool on, int did, double e, double t, double size, double px, double py,
                 double pz, double qx, double qy, double qz, double tt) {
  if (!on) { x = mu2e::TrkCaloHitInfo(); return; }
  x.active = true; x.did = did; x.edep = e; x.ctime = t; x.csize = size; x.poca.SetXYZ(px, py, pz);
  x.mom.SetXYZ(qx, qy, qz); x.ptoca = tt; x.dt = tt - t; x.tresid = tt - t; }
// fast clusters: replace caloclusters / caloclustersmc / calohits (one hit per crystal of the cluster)
void set_clusters(std::vector<mu2e::CaloClusterInfo>& cl, std::vector<mu2e::CaloClusterInfoMC>& clmc,
                  std::vector<mu2e::CaloHitInfo>& hits, int n, const double* buf) {
  // buf: 10 columns of n values (energy, time, x, y, z, size, disk, split, etot, tprimary), never empty
  const double *e = buf, *t = buf + n, *x = buf + 2 * n, *y = buf + 3 * n, *z = buf + 4 * n, *sizef = buf + 5 * n,
               *diskf = buf + 6 * n, *splitf = buf + 7 * n, *etot = buf + 8 * n, *tprim = buf + 9 * n;
  mu2e::CaloClusterInfoMC proto = clmc.empty() ? mu2e::CaloClusterInfoMC() : clmc[0];
  mu2e::CaloClusterInfo cproto = cl.empty() ? mu2e::CaloClusterInfo() : cl[0];
  cl.clear(); clmc.clear(); hits.clear();
  for (int i = 0; i < n; ++i) {
    long size_i = std::lround(sizef[i]);
    mu2e::CaloClusterInfo c = cproto;
    c.diskID_ = std::lround(diskf[i]); c.time_ = t[i]; c.energyDep_ = e[i]; c.cog_.SetXYZ(x[i], y[i], z[i]);
    c.size_ = size_i; c.isSplit_ = splitf[i] != 0; c.hits_.clear();
    for (long k = 0; k < size_i; ++k) {
      mu2e::CaloHitInfo h; h.time_ = t[i]; h.eDep_ = e[i] / size_i; h.nSiPMs_ = 2; h.clusterIdx_ = i;
      c.hits_.push_back(hits.size()); hits.push_back(h); }
    cl.push_back(c);
    mu2e::CaloClusterInfoMC m = proto;
    m.nsim = etot[i] > 0 ? 1 : 0; m.etot = etot[i]; m.eprimary = etot[i]; m.tprimary = tprim[i];
    m.tavg = tprim[i];
    m.nhits = size_i; m.hits_.clear();
    clmc.push_back(m);
  }
}
}
'''


def setup_root():
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if ROOT.gSystem.Load("libeventntuple_dict") < 0:
        raise RuntimeError("centuple writer: libeventntuple_dict not found: run in the Mu2e environment of the stock "
                           "release (muse setup of an AnalysisMDC2025 musing)")
    if not ROOT.gInterpreter.Declare(CPP):
        raise RuntimeError("centuple writer: could not compile the C++ helpers (EventNtuple headers not found?)")
    return ROOT


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(plan_path, donor_files, out, *, n_generated=None, extra_provenance=None, check_sha256=True,
          progress=5000, stream=STREAM_PATTERNS, subruns=None):
    """Write the EventNtuple file `out` from a write plan and the donor files (paths, in the donor library's file
    order). Events are written in plan order; an event is kept when one of the stream paths (`stream`, fnmatch
    patterns of path names; None keeps every event) accepted it, as the stock ntuple of the triggered stream holds
    only the events its digitization output selected. subruns (opt-in): write only the plan's events of these
    subruns (one file per input sequencer)."""
    import fnmatch
    ROOT = setup_root()
    plan = dict(np.load(plan_path))
    meta = json.loads(str(plan.pop("meta")))
    donor_files = [str(p) for p in donor_files]
    if check_sha256:
        for p, h in zip(donor_files, meta["donor_sha256"]):
            if _sha256(p) != h:
                raise ValueError(f"centuple writer: {p} is not the donor library's file (sha256)")
    chain = ROOT.TChain(TREE)
    offsets = [0]
    for p in donor_files:
        chain.Add(p)
        f = ROOT.TFile.Open(p)
        offsets.append(offsets[-1] + int(f.Get(TREE).GetEntries()))
        f.Close()
    chain.LoadTree(0)
    chain.GetEntry(0)        # allocate the object buffers, so that the clone shares them (CloneTree copies addresses)
    trig_all = [b.GetName() for b in chain.GetTree().GetListOfBranches() if b.GetName().startswith("trig_")]
    tbuf = np.zeros(len(trig_all), dtype=np.bool_)
    for i, name in enumerate(trig_all):
        chain.SetBranchAddress(name, tbuf[i:i + 1])
    fout = ROOT.TFile(str(out) + ".tmp", "RECREATE")
    d = fout.mkdir("EventNtuple")
    d.cd()
    tout = chain.CloneTree(0)
    for i, name in enumerate(trig_all):
        tout.SetBranchAddress(name, tbuf[i:i + 1])
    names = [str(x) for x in plan["trig_names"]]
    tcol = {"trig_" + p: k for k, p in enumerate(names)}
    tidx = [(i, tcol[name]) for i, name in enumerate(trig_all) if name in tcol]
    uncovered = [name for name in trig_all if name not in tcol]
    sidx = np.array([i for i, name in enumerate(trig_all)
                     if stream is None or any(fnmatch.fnmatch(name[5:], pat) for pat in stream)], np.int64)
    n = plan["event"].size
    gentry = np.asarray(offsets)[plan["donor_file"]] + plan["donor_entry"]
    order = np.arange(n) if subruns is None else np.flatnonzero(np.isin(plan["subrun"], list(subruns)))
    nwritten = 0
    tic = time.perf_counter()
    cl_n = plan["cl_valid"].sum(axis=1)
    stats = dict(events=int(len(order)), written=0, untriggered=0, q=0, f=0, nclass=0)
    kept = []
    for k in order:
        chain.GetEntry(int(gentry[k]))
        ev = chain.evtinfo
        ev.run, ev.subrun, ev.event = int(plan["run"][k]), int(plan["subrun"][k]), int(plan["event"][k])
        if "pbtime_mc" in plan and np.isfinite(plan["pbtime_mc"][k]):
            ev.pbtime = float(plan["pbtime_reco"][k])
            chain.evtinfomc.pbtime = float(plan["pbtime_mc"][k])
        for i, j in tidx:
            tbuf[i] = bool(plan["trig"][k, j])
        if stream is not None and not tbuf[sidx].any():     # not in the stored stream
            stats["untriggered"] += 1
            continue
        trk, segs, lh = chain.trk, chain.trksegs, chain.trksegpars_lh
        smc, sims = chain.trksegsmc, chain.trkmcsim
        ROOT.cent.transform(segs, lh, smc, sims, chain.trkmcvd, chain.trkcalohit, chain.trkmc,
                            float(plan["dphi"][k]), float(plan["dt"][k]),
                            float(plan["dt_reco"][k]) if "dt_reco" in plan else float(plan["dt"][k]))
        p = int(plan["donor_prim"][k])
        cls = int(plan["cls"][k])
        if p >= 0 and bool(plan["has_entry"][k]):
            _truth(ROOT, plan, k, smc, sims, p)
        if cls in (1, 2) and p >= 0 and np.isfinite(plan["fitcon"][k]):
            stats["q" if cls == 2 else "f"] += 1
            _electron(ROOT, plan, k, chain, p)
        elif cls in (1, 2):
            stats["q" if cls == 2 else "f"] += 1
        else:
            stats["nclass"] += 1
        m = int(cl_n[k])
        sel = np.flatnonzero(plan["cl_valid"][k])[:m]
        cols = [plan[c][k][sel] for c in ("cl_energy", "cl_time", "cl_x", "cl_y", "cl_z")]
        cols += [np.maximum(plan["cl_size"][k][sel], 1), plan["cl_disk"][k][sel], plan["cl_split"][k][sel],
                 plan["cl_etot"][k][sel], plan["cl_tprim"][k][sel]]
        buf = np.zeros(max(10 * m, 1), np.float64)
        if m:
            buf[:10 * m] = np.concatenate([np.asarray(c, np.float64) for c in cols])
        ROOT.cent.set_clusters(chain.caloclusters, chain.caloclustersmc, chain.calohits, m, buf)
        hc = chain.hitcount
        nsd = int(plan["nsd"][k])
        if nsd >= 0:
            r = nsd / hc.nsd if hc.nsd > 0 else 1.0
            hc.nesel, hc.nrsel, hc.ntsel, hc.nbkg = (int(round(v * r)) if hc.nsd > 0 else nsd
                                                     for v in (hc.nesel, hc.nrsel, hc.ntsel, hc.nbkg))
            hc.nsd = nsd
        tout.Fill()
        kept.append(k)
        nwritten += 1
        if progress and nwritten % progress == 0:
            print(f"centuple writer: {nwritten} written, {time.perf_counter() - tic:.1f} s", flush=True)
    stats["written"] = nwritten
    d.cd()
    tout.Write()
    gen = int(n_generated if n_generated is not None else n)
    # n_proc_events: the events the stock ntuple job processed = the stored stream it read (its tree entries); the
    # generated count goes to the provenance (in production it is the files' SAM dh.gencount)
    h = ROOT.TH1I("n_proc_events", "processed events", 1, 0, 1)
    h.SetBinContent(1, nwritten)
    h.Write()
    src = ROOT.TFile.Open(donor_files[0])
    for name in COPIED:
        obj = src.Get("EventNtuple/" + name)
        if obj:
            d.cd()
            obj.Write(name)
    # subrunNtuple: one entry per (run, subrun) of the written events with its processed (stored) count
    st = src.Get("EventNtuple/subrunNtuple")
    if st:
        st.GetEntry(0)
        from array import array
        pec = array("q", [0])                 # Long64_t buffer (cppyy does not take a NumPy int64 array here)
        st.SetBranchAddress("procEventCount", pec)
        d.cd()
        so = st.CloneTree(0)
        so.SetBranchAddress("procEventCount", pec)
        keys, counts = np.unique(np.stack([plan["run"][kept], plan["subrun"][kept]], 1), axis=0, return_counts=True) \
            if len(kept) else (np.zeros((0, 2), np.int64), np.zeros(0, np.int64))
        for (r, sr), c in zip(keys, counts):
            st.srinfo.run, st.srinfo.subrun = int(r), int(sr)
            pec[0] = int(c)
            so.Fill()
        d.cd()
        so.Write()
    src.Close()
    prov = dict(writer="beamfastsim.centuple.writer", plan=meta, donor_files=[Path(p).name for p in donor_files],
                uncovered_trigger_paths=uncovered, stream_paths=[trig_all[i][5:] for i in sidx], covered_trigger_paths=names, n_generated=gen,
                stats=stats, root=ROOT.gROOT.GetVersion(), python=sys.version.split()[0],
                **(extra_provenance or {}))
    d.cd()
    ROOT.TNamed(PROV, json.dumps(prov, default=str)).Write(PROV)
    fout.Close()
    Path(str(out) + ".tmp").rename(out)
    stats["wall_s"] = round(time.perf_counter() - tic, 2)
    return stats


def _xyz(a):
    return [float(v) for v in a]


def _truth(ROOT, plan, k, smc, sims, p):
    """The electron's truth: birth (trkmcsim[*][0] of the tracks of the same particle) and its TT surfaces."""
    tid = sims[p][0].id if sims[p].size() else None
    for j in range(sims.size()):
        if sims[j].size() and (tid is None or sims[j][0].id == tid):
            ROOT.cent.set_sim(sims[j][0], *_xyz(plan["birth_x"][k]), *_xyz(plan["birth_p"][k]),
                              float(plan["birth_t"][k]))
    v = smc[p] if p < smc.size() else None
    if v is None:
        return
    for name, sid in SID.items():
        if not bool(plan[f"mc_{name}_ok"][k]):
            continue
        best = None
        for i in range(v.size()):
            if v[i].sid == sid and v[i].mom.z() > 0 and (best is None or v[i].time < v[best].time):
                best = i
        if best is not None:
            ROOT.cent.set_step(v[best], *_xyz(plan[f"mc_{name}_pos"][k]), *_xyz(plan[f"mc_{name}_mom"][k]),
                               float(plan[f"mc_{name}_time"][k]))


def _electron(ROOT, plan, k, chain, p):
    """Overwrite the donor's electron track (index p) with the fast reco values."""
    ti = chain.trk[p]
    dn = int(plan["nactive"][k]) - ti.nactive
    ti.nactive += dn
    ti.nhits += dn
    ti.ndof = max(ti.ndof + dn, 1)
    for f in ("nesel", "nrsel", "ntsel", "nsel"):
        setattr(ti, f, max(getattr(ti, f) + dn, 0))
    ti.fitcon = float(plan["fitcon"][k])
    ti.chisq = float(plan["chi2ndf"][k]) * ti.ndof
    segs, lh = chain.trksegs[p], chain.trksegpars_lh[p]
    for name, sid in SID.items():
        if not bool(plan[f"seg_{name}_ok"][k]):
            continue
        best = None
        for i in range(segs.size()):
            if segs[i].sid == sid and segs[i].mom.z() > 0 and (best is None or segs[i].time < segs[best].time):
                best = i
        if best is None:
            continue
        ROOT.cent.set_seg(segs[best], *_xyz(plan[f"seg_{name}_pos"][k]), *_xyz(plan[f"seg_{name}_mom"][k]),
                          float(plan[f"seg_{name}_time"][k]))
        if best < lh.size():
            ROOT.cent.set_lh(lh[best], *(float(plan[f"lh_{name}_{q}"][k]) for q in
                                         ("maxr", "d0", "tanDip", "rad", "lam", "cx", "cy", "phi0", "t0")))
    if p < chain.trkpid.size() and np.isfinite(plan["pid"][k]):
        chain.trkpid[p].result = float(plan["pid"][k])
    on = bool(plan["calo_match"][k])
    if p < chain.trkcalohit.size():
        j = int(plan["calo_match_cluster"][k])
        cx = plan["calo_contact_x"][k]
        mom = plan["seg_back_mom"][k] if bool(plan["seg_back_ok"][k]) else plan["seg_front_mom"][k]
        args = (on, int(plan["cl_disk"][k][j]), float(plan["cl_energy"][k][j]), float(plan["cl_time"][k][j]),
                float(plan["cl_size"][k][j]), *_xyz(np.nan_to_num(cx)), *_xyz(np.nan_to_num(mom)),
                float(np.nan_to_num(plan["calo_contact_t"][k])))
        ROOT.cent.set_calohit(chain.trkcalohit[p], *args)
