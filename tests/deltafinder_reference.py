"""A literal transcription of the stock DeltaFinder (Offline v13_41_01, CalPatRec: DeltaFinder_module, DeltaFinderAlg,
DeltaFinderAlg_findProtons, DeltaSeed, DeltaCandidate, ProtonCandidate), hit by hit, for the device port's tests.

numpy float32 scalars round every float operation as the stock code does; the steps C++ promotes to double (double
members, double literals, float * double) run in numpy float64 and are rounded where stock stores them in a float.
Every constant is wrapped in its C++ type, so numpy's scalar promotion rules do not matter. The stock containers
become Python lists in their stock order (per-station seed lists in creation order, the delta-candidate vector,
the proton-candidate list), pointers become object references, and the out-of-range time-bin reads of an OnSpill
job are reproduced (`Face.first` / `Face.last`).

`run(hits, geo, cfg)` takes one event's panel hits in collection order and returns the output flags of TrigFlagPH.
"""
import copy
import math

import numpy as np

f = np.float32
d = np.float64
NST, NFACE = 18, 4
BKG = 1 << 6           # StrawHitFlag::bkg
ESEL = 1 << 1          # StrawHitFlag::energysel
NTBMAX = 3000          # kMaxNTimeBins
SIGT2 = f(64)          # DeltaSeed::fSigT2


def i_div(a, b):
    """C++ integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def i_mod(a, b):
    return a - b * i_div(a, b)


def trunc_int(x):
    """C++ float -> int conversion (truncation)."""
    return int(np.trunc(x))


def sortable(t):
    u = int(np.asarray(f(t)).view(np.uint32))
    return (~u) & 0xFFFFFFFF if u & 0x80000000 else u | 0x80000000


class Hit:                                     # HitData_t
    def __init__(self, k, h, zface):
        self.k = k                             # index in the ComboHit collection
        self.zface = zface
        self.seed = None
        self.used = 0
        self.chi2min = f(99999.0)
        sigw = f(np.sqrt(h["uvar"][k]))        # posRes(wire) = sqrt(uvar), float
        self.sigw2 = sigw * sigw
        self.ct = f(h["ctime"][k])
        self.t = f(h["time"][k])
        self.x, self.y = f(h["x"][k]), f(h["y"][k])
        self.wx, self.wy = f(h["udx"][k]), f(h["udy"][k])
        self.nr = self.x * self.wy - self.y * self.wx
        self.nx2, self.nxy, self.ny2 = self.wx * self.wx, self.wx * self.wy, self.wy * self.wy
        self.nxr, self.nyr = self.wx * self.nr, self.wy * self.nr
        self.delta = -1
        self.proton = -1
        self.edep = f(h["edep"][k])
        self.nsh = int(h["nsh"][k])
        sid = int(h["sid"][k])
        self.sid = sid
        self.panel = (sid >> 7) & 7
        self.panel_id = (sid >> 7) & 0x1FF

    def phi(self, geo):
        return geo["atan2"](self.y, self.x)


class Face:                                    # FaceZ_t
    def __init__(self, nbins):
        self.hits = []
        self.nbins = nbins
        self.f = [-1] * nbins
        self.l = [-1] * nbins
        self.proton_hits = []

    def first(self, b):
        if 0 <= b < self.nbins:
            return self.f[b]
        if self.nbins <= b < NTBMAX:
            return 0                           # never reset in an OnSpill job: the constructor's zeros
        raise OverflowError(f"fFirst[{b}] outside the array")

    def last(self, b):
        if 0 <= b < self.nbins:
            return self.l[b]
        if self.nbins <= b < NTBMAX:
            return 0
        if -NTBMAX + self.nbins <= b < 0:
            return 0                           # fLast[b < 0] reads fFirst[3000 + b], never reset
        raise OverflowError(f"fLast[{b}] outside the array")


class Seed:                                    # DeltaSeed
    def __init__(self, station, index):
        self.station, self.index = station, index
        self.good = 0
        self.hd = [None] * NFACE
        self.fp = [0] * NFACE

    def init(self, h0, h1, xc, yc, zc):
        self.snx2 = self.snxy = self.sny2 = self.snxr = self.snyr = d(0)
        self.sumedep = self.sumt = self.sumt2 = f(0)
        self.good = 1
        self.sface = [h0.zface, h1.zface if h1 is not None else -1]
        self.delta = -1
        self.proton = -1
        self.fp = [0] * NFACE
        self.hd = [None] * NFACE
        self.hd[h0.zface] = h0
        self.chi21 = h0.chi2min
        if h1 is not None:
            self.hd[h1.zface] = h1
            self.nhits = 2
            self.nsh = h0.nsh + h1.nsh
            self.fp[h0.zface] = self.fp[h1.zface] = 1
            self.chi22 = h1.chi2min
        else:
            self.nhits = 1
            self.nsh = h0.nsh
            self.fp[h0.zface] = 1
            self.chi22 = f(0)
        self.chi2par = self.chi21 + self.chi22
        self.chi2perp = f(0)
        self.tmin = h0.ct
        self.tmax = self.tmin
        if h1 is not None:
            if h1.ct >= self.tmin:
                self.tmax = h1.ct
            else:
                self.tmin = h1.ct
        self.cx, self.cy = f(xc), f(yc)
        for i in range(2):
            face = self.sface[i]
            if face < 0:
                continue
            hd = self.hd[face]
            self.snx2 += d(hd.nx2); self.snxy += d(hd.nxy); self.sny2 += d(hd.ny2)
            self.snxr += d(hd.nxr); self.snyr += d(hd.nyr)
            self.sumedep += hd.edep * f(hd.nsh)
            self.sumt += hd.ct
            self.sumt2 += hd.ct * hd.ct

    def add_hit(self, hd):
        face = hd.zface
        self.hd[face] = hd
        self.fp[face] = 1
        if hd.ct < self.tmin:
            self.tmin = hd.ct
        if hd.ct > self.tmax:
            self.tmax = hd.ct
        self.nhits += 1
        self.nsh += hd.nsh
        self.snx2 += d(hd.nx2); self.snxy += d(hd.nxy); self.sny2 += d(hd.ny2)
        self.snxr += d(hd.nxr); self.snyr += d(hd.nyr)
        self.sumedep += hd.edep * f(hd.nsh)
        self.sumt += hd.ct
        self.sumt2 += hd.ct * hd.ct

    def replace_first_hit(self, hd):
        self.hd[self.sface[0]] = hd
        self.chi21 = hd.chi2min
        self.nsh = hd.nsh
        self.tmin = hd.ct
        self.tmax = self.tmin
        self.sumedep = hd.edep * f(hd.nsh)
        self.sumt = hd.ct * f(hd.nsh)
        self.sumt2 = hd.ct * hd.ct

    def cog_chi2(self, rcore, sig2):
        if self.nhits == 2 and self.sface[0] >= 0 and self.sface[1] >= 0:
            self.chi2par = self.chi21 + self.chi22
            self.chi2perp = f(0)
            return
        dd = self.snx2 * self.sny2 - self.snxy * self.snxy
        if dd == 0.:
            return
        xc = (self.snyr * self.snx2 - self.snxr * self.snxy) / dd
        yc = (self.snyr * self.snxy - self.snxr * self.sny2) / dd
        self.cx, self.cy = f(xc), f(yc)
        self.chi2par = f(0)
        self.chi2perp = f(0)
        for face in range(NFACE):
            hd = self.hd[face]
            if hd is None:
                continue
            dx = d(hd.x) - xc
            dy = d(hd.y) - yc
            w = f(dx * d(hd.wx) + dy * d(hd.wy))
            n = f(dx * d(hd.wy) - dy * d(hd.wx))
            cpar = (w * w) / (sig2 + hd.sigw2)
            drr = f(max(d(abs(n) - rcore), 0.0))
            cperp = (drr * drr) / sig2
            self.chi2par += cpar
            self.chi2perp += cperp

    def used(self):
        return self.delta >= 0

    def is_good(self):
        return self.good >= 0

    def delta_index(self):
        return i_mod(self.delta, 10000)

    def tmean(self):
        return self.sumt / f(self.nhits) if self.nhits > 0 else f(0)

    def chi2_tot_n(self):
        return (self.chi2par + self.chi2perp) / f(self.nhits) if self.nhits > 0 else f(0)

    def chi2_time(self):
        if self.nhits > 0:
            n = f(self.nhits)
            return (self.sumt2 / n - (self.sumt / n) * (self.sumt / n)) / SIGT2
        return f(0)

    def edep_mean(self):
        return self.sumedep / f(self.nsh) if self.nsh > 0 else f(0)

    def t0min(self):
        return (self.tmin + self.tmax) / f(2) - f(20)

    def t0max(self):
        return (self.tmin + self.tmax) / f(2) + f(20)


class Delta:                                   # DeltaCandidate
    def __init__(self, index, seed, geo):
        self.geo = geo
        self.index = index
        self.mask = 0
        self.first, self.last = 999, -1
        self.seeds = [None] * NST
        self.cx = self.cy = f(0)
        self.nx = self.ny = f(0)
        self.nseeds = self.nhits = self.nsh = 0
        self.sumedep = f(0)
        self.snx2 = self.snxy = self.sny2 = self.snxr = self.snyr = d(0)
        self.st = self.sz = self.st2 = self.stz = self.sz2 = d(0)
        self.t0 = self.dtdz = self.sigt0 = d(0)
        if seed is not None:
            self.add_seed(seed)

    def copy(self):
        c = copy.copy(self)
        c.seeds = list(self.seeds)
        return c

    def _centre(self):
        dd = self.snx2 * self.sny2 - self.snxy * self.snxy
        if dd != 0.:
            xc = (self.snyr * self.snx2 - self.snxr * self.snxy) / dd
            yc = (self.snyr * self.snxy - self.snxr * self.sny2) / dd
            self.cx, self.cy = f(xc), f(yc)
            rho = np.sqrt(xc * xc + yc * yc)
            if rho > 0.:
                self.nx, self.ny = f(xc / rho), f(yc / rho)

    def _time(self, check_denom):
        n = d(self.nseeds)
        tm, zm, tzm, t2m, z2m = self.st / n, self.sz / n, self.stz / n, self.st2 / n, self.sz2 / n
        denom = z2m - zm * zm
        if self.nseeds > 1 and (not check_denom or denom != 0.):
            with np.errstate(all="ignore"):
                self.dtdz = (tzm - tm * zm) / denom
                self.t0 = tm - self.dtdz * zm
                self.sigt0 = np.sqrt((t2m - tm * tm) / (n - d(0.9999)))
        else:
            self.dtdz = d(0)
            self.t0 = tm
            self.sigt0 = d(0)

    def add_seed(self, s):
        st = s.station
        self.seeds[st] = s
        self.nseeds += 1
        self.first = min(self.first, st)
        self.last = max(self.last, st)
        self.nhits += s.nhits
        self.nsh += s.nsh
        self.sumedep += s.sumedep
        self.snx2 += s.snx2; self.snxy += s.snxy; self.sny2 += s.sny2; self.snxr += s.snxr; self.snyr += s.snyr
        self._centre()
        t = d(s.tmean())
        z = d(self.geo["stationZ"][st])
        self.st += t; self.st2 += t * t; self.stz += t * z; self.sz += z; self.sz2 += z * z
        self._time(False)

    def merge(self, o):
        for st in range(o.first, o.last + 1):
            if o.seeds[st] is None:
                continue
            if self.seeds[st] is not None:
                continue
            self.seeds[st] = o.seeds[st]
            self.seeds[st].delta = self.index
            self.nseeds += 1
            self.nhits += self.seeds[st].nhits
            self.nsh += self.seeds[st].nsh
            self.sumedep += self.seeds[st].sumedep
            self.first = min(self.first, st)
            self.last = max(self.last, st)
        self.snx2 += o.snx2; self.snxy += o.snxy; self.sny2 += o.sny2; self.snxr += o.snxr; self.snyr += o.snyr
        self._centre()
        self.st += o.st; self.st2 += o.st2; self.stz += o.stz; self.sz += o.sz; self.sz2 += o.sz2
        self._time(True)

    def active(self):
        return self.index >= 0

    def T0(self, st):
        if self.nseeds == 1:
            return f(self.t0)
        with np.errstate(all="ignore"):
            return f(self.t0 + self.dtdz * d(self.geo["stationZ"][st]))

    def edep_mean(self):
        return self.sumedep / f(self.nsh) if self.nsh > 0 else f(0)

    def mark_hits_used(self):
        for st in range(self.first, self.last + 1):
            s = self.seeds[st]
            if s is None:
                continue
            for hd in s.hd:
                if hd is not None:
                    hd.delta = self.index


class Proton:                                  # ProtonCandidate
    def __init__(self, index, geo):
        self.index = index
        self.geo = geo
        self.init()

    def init(self):
        self.first, self.last = 999, -1
        self.nst = 0
        self.ntot = 0
        self.nshtot = 0
        self.sumedep = f(0)
        self.tmid = f(0)
        self.t0 = self.dtdz = self.sigt0 = f(0)
        self.nhst = [0] * NST
        self.sx = [f(0)] * NST
        self.sy = [f(0)] * NST
        self.phi_ = [f(0)] * NST
        self.hd = [[[] for _ in range(NFACE)] for _ in range(NST)]
        self.pid = [[-1] * NFACE for _ in range(NST)]
        self.st = self.sz = self.st2 = self.stz = self.sz2 = d(0)

    def add_seed(self, seed):
        st = seed.station
        z = f(self.geo["stationZ"][st])
        self.first = min(self.first, st)
        self.last = max(self.last, st)
        for face in range(NFACE):
            hd = seed.hd[face]
            if hd is None:
                continue
            if any(h is hd for h in self.hd[st][face]):
                continue
            if self.nhst[st] == 0:
                self.nst += 1
            self.nhst[st] += 1
            if len(self.hd[st][face]) == 0:
                self.pid[st][face] = hd.panel_id
            if self.pid[st][face] == hd.panel_id:
                self.hd[st][face].append(hd)
                self.ntot += 1
                self.nshtot += hd.nsh
                self.sumedep += hd.edep * f(hd.nsh)
                self.sx[st] += hd.x
                self.sy[st] += hd.y
                t = hd.ct
                self.st += d(t); self.st2 += d(t * t); self.stz += d(t * z); self.sz += d(z); self.sz2 += d(z * z)
        self.phi_[st] = self.geo["atan2"](self.sy[st], self.sx[st])
        self.update_time()
        seed.proton = self.index
        for hd in seed.hd:
            if hd is not None:
                hd.proton = self.index

    def add_hit(self, st, hd, update=True):
        face = hd.zface
        if any(h is hd for h in self.hd[st][face]):
            return
        self.hd[st][face].append(hd)
        if self.nhst[st] == 0:
            self.nst += 1
        self.nhst[st] += 1
        self.first = min(self.first, st)
        self.last = max(self.last, st)
        if self.pid[st][face] == -1:
            self.pid[st][face] = hd.panel_id
        self.ntot += 1
        self.nshtot += hd.nsh
        self.sumedep += hd.edep * f(hd.nsh)
        self.sx[st] += hd.x
        self.sy[st] += hd.y
        self.phi_[st] = self.geo["atan2"](self.sy[st], self.sx[st])
        t, z = d(hd.ct), d(self.geo["stationZ"][st])
        self.st += t; self.st2 += t * t; self.stz += t * z; self.sz += z; self.sz2 += z * z
        if update:
            self.update_time()
        hd.proton = self.index

    def remove_hit(self, st, hd, update=True):
        face = hd.zface
        lst = self.hd[st][face]
        nhf = len(lst)
        for ih in range(nhf):
            if ih >= len(lst) or lst[ih] is not hd:
                continue                       # past the end: the popped slot still holds the old last pointer
            self.nhst[st] -= 1
            if self.nhst[st] == 0:
                self.nst -= 1
                if self.first == st:
                    for s2 in range(st + 1, self.last + 1):
                        if self.nhst[s2] > 0:
                            self.first = s2
                            break
                elif st == self.last:
                    for s2 in range(st - 1, self.first - 1, -1):
                        if self.nhst[s2] > 0:
                            self.last = s2
                            break
            self.ntot -= 1
            self.nshtot -= hd.nsh
            self.sumedep -= hd.edep * f(hd.nsh)
            t, z = d(hd.ct), d(self.geo["stationZ"][st])
            self.st -= t; self.st2 -= t * t; self.stz -= t * z; self.sz -= z; self.sz2 -= z * z
            del lst[ih]
            if nhf == 1:
                self.pid[st][face] = -1
            break
        if update:
            self.update_time()

    def update_time(self):
        n = d(self.ntot)
        with np.errstate(all="ignore"):
            tm, zm, tzm, t2m, z2m = self.st / n, self.sz / n, self.stz / n, self.st2 / n, self.sz2 / n
            if self.nst > 1:
                denom = z2m - zm * zm
                if denom != 0.:
                    self.dtdz = f((tzm - tm * zm) / denom)
                    self.t0 = f(tm - d(self.dtdz) * zm)
                    self.sigt0 = f(np.sqrt((t2m - tm * tm) / (n - d(1))))
                else:
                    self.dtdz, self.t0, self.sigt0 = f(0), f(tm), f(0)
            else:
                self.dtdz, self.t0, self.sigt0 = f(0), f(tm), f(0)
        station = 9
        if self.first <= 9:
            if self.last <= 9:
                station = self.last
        else:
            station = self.first
        self.tmid = self.t0 + self.dtdz * f(self.geo["stationZ"][station])

    def t0_at(self, st):
        return self.t0 + self.dtdz * f(self.geo["stationZ"][st])

    def edep_mean(self):
        return self.sumedep / f(self.nshtot) if self.nshtot > 0 else f(0)

    def predict_phi(self, st):
        """-> (phi, err); phi -100 if no prediction."""
        pi, twopi = d(math.pi), d(2 * math.pi)
        with np.errstate(all="ignore"):
            if st < self.first:
                if self.first - st > 1:
                    return f(-100), f(0)
                if self.nst > 1:
                    phi1, phi2 = self.phi_[self.last], self.phi_[self.first]
                    dphi = phi2 - phi1
                    if d(dphi) < pi:
                        dphi = f(d(dphi) + twopi)
                    if d(dphi) > pi:
                        dphi = f(d(dphi) - twopi)
                    dphi = dphi / f(self.last - self.first)
                    return phi2 + dphi, f(0.5)
                return f(d(self.phi_[self.first]) + d(0.15)), f(0.65)
            if self.nhst[st] > 0:
                return self.phi_[st], f(0.2)
            phi1, phi2 = self.phi_[self.last], self.phi_[self.first]
            dphi = (phi2 - phi1) / f(self.last - self.first)
            return phi2 - dphi * f(st - self.first), f(0.5)

    def nhits(self, st, face):
        return len(self.hd[st][face])


def _wrap(dphi):
    """`if (dphi < -M_PI) dphi += 2*M_PI; if (dphi > M_PI) dphi -= 2*M_PI;` on a float."""
    if d(dphi) < -d(math.pi):
        dphi = f(d(dphi) + d(2 * math.pi))
    if d(dphi) > d(math.pi):
        dphi = f(d(dphi) - d(2 * math.pi))
    return dphi


def order_id(station, plane, panel):
    """ChannelID::orderID -> (station, z-ordered face, panel in face)."""
    face = panel % 2
    if station % 2 == 0:
        of = 1 - face if plane == 0 else face + 2
    else:
        of = face if plane == 0 else 3 - face
    return station, of, panel // 2


class Finder:
    def __init__(self, hits, geo, cfg, spill_on=True):
        self.h = hits
        self.geo = geo
        self.c = {k: (f(v) if isinstance(v, float) else v) for k, v in cfg.items()}
        c = self.c
        self.tbin = f(c["timeBin"])
        self.maxT = f(c["maximumTime"])
        self.sig2 = f(c["sigmaR"]) * f(c["sigmaR"])
        self.rcore = f(c["rCore"])
        self.nbins = min(trunc_int(f(2000.0) / self.tbin), NTBMAX) if spill_on else NTBMAX
        self.faces = [[Face(self.nbins) for _ in range(NFACE)] for _ in range(NST)]
        self.seeds = [[] for _ in range(NST)]
        self.compton = [[] for _ in range(NST)]
        self.pseeds = [[] for _ in range(NST)]
        self.deltas = []
        self.protons = []

    # ---------------------------------------------------------------- order
    def order_hits(self):
        n = len(self.h["sid"])
        order = sorted(range(n), key=lambda k: sortable(self.h["time"][k]))   # LSD radix sort: stable
        for k in order:
            sid = int(self.h["sid"][k])
            plane = sid >> 10
            panel = (sid >> 7) & 7
            os_, of, op = order_id(plane // 2, plane % 2, panel)
            if not (0 <= os_ < NST and 0 <= of < NFACE and 0 <= op < 3):
                continue
            fz = self.faces[os_][of]
            loc = len(fz.hits)
            fz.hits.append(Hit(k, self.h, of))
            tb = trunc_int(f(self.h["time"][k]) / self.tbin)
            if tb < self.nbins:
                if tb < 0:
                    raise OverflowError("negative time bin")
                if fz.f[tb] < 0:
                    fz.f[tb] = loc
                fz.l[tb] = loc

    def _bins(self, time_bin):
        max_bin = trunc_int(self.maxT / self.tbin)
        first_tbin, last_tbin = 0, max_bin
        if time_bin > 0:
            first_tbin = time_bin - 1
        if time_bin < max_bin:
            last_tbin = time_bin + 1
        return first_tbin, last_tbin

    def _range(self, fz, ft, lt):
        while ft < lt and fz.first(ft) < 0:
            ft += 1
        while lt > ft and fz.first(lt) < 0:
            lt -= 1
        first = fz.first(ft)
        if first < 0:
            return None
        return first, fz.last(lt)

    def new_seed(self, st):
        s = Seed(st, len(self.seeds[st]))
        self.seeds[st].append(s)
        return s

    # ---------------------------------------------------------------- seeds
    def hit_chi2(self, hd, xc, yc):
        dx = hd.x - xc
        dy = hd.y - yc
        w = dx * hd.wx + dy * hd.wy
        cpar = (w * w) / (self.sig2 + hd.sigw2)
        n = dx * hd.wy - dy * hd.wx
        drr = f(max(d(abs(n) - self.rcore), 0.0))
        return cpar, (drr * drr) / self.sig2

    def seed_chi2(self, s, xc, yc):
        cp, cq = f(0), f(0)
        for hd in s.hd:
            if hd is not None:
                a, b = self.hit_chi2(hd, xc, yc)
                cp += a
                cq += b
        return cp, cq

    def delta_chi2(self, dc, xc, yc):
        cp, cq = f(0), f(0)
        for st in range(dc.first, dc.last + 1):
            s = dc.seeds[st]
            if s is None:
                continue
            for hd in s.hd:
                if hd is not None:
                    a, b = self.hit_chi2(hd, xc, yc)
                    cp += a
                    cq += b
        return cp, cq

    def complete_seed(self, seed):
        c = self.c
        st = seed.station
        ft0, lt0 = self._bins(trunc_int(seed.tmean() / self.tbin))
        xs, ys = seed.cx, seed.cy
        rho = f(np.sqrt(xs * xs + ys * ys))
        snx, sny = xs / rho, ys / rho
        for face in range(NFACE):
            if seed.fp[face] == 1:
                continue
            fz = self.faces[st][face]
            r = self._range(fz, ft0, lt0)
            if r is None:
                continue
            closest, best = None, c["maxChi2Seed"]
            for ih in range(r[0], r[1] + 1):
                hd = fz.hits[ih]
                if hd.used >= 3:
                    continue
                pz = self.geo["pz"][st][face][hd.panel // 2]
                if f(d(snx) * pz["nx"] + d(sny) * pz["ny"]) < f(0.5):
                    continue
                if hd.ct - seed.t0max() > c["maxHitSeedDt"]:
                    break
                if seed.t0min() - hd.ct > c["maxHitSeedDt"]:
                    continue
                snx2 = seed.snx2 + d(hd.nx2); snxy = seed.snxy + d(hd.nxy); sny2 = seed.sny2 + d(hd.ny2)
                snxr = seed.snxr + d(hd.nxr); snyr = seed.snyr + d(hd.nyr)
                dd = snx2 * sny2 - snxy * snxy
                with np.errstate(all="ignore"):
                    xc = f((snyr * snx2 - snxr * snxy) / dd)
                    yc = f((snyr * snxy - snxr * sny2) / dd)
                dx, dy = hd.x - xc, hd.y - yc
                w = f(d(dx) * pz["wx"] + d(dy) * pz["wy"])
                n = f(d(dx) * pz["nx"] + d(dy) * pz["ny"])
                drho = f(max(d(abs(n) - self.rcore), 0.0))
                cpar = (w * w) / (hd.sigw2 + self.sig2)
                cperp = (drho * drho) / self.sig2
                if cpar < c["maxChi2Par"] and cperp < c["maxChi2Perp"]:
                    sp, sq = self.seed_chi2(seed, xc, yc)
                    cpar += sp
                    cperp += sq
                    chi2 = (cpar + cperp) / f(seed.nhits + 1)
                    if chi2 < best:
                        closest, best = hd, chi2
            if closest is not None:
                closest.chi2min = best
                seed.add_hit(closest)
        seed.cog_chi2(self.rcore, self.sig2)
        for hd in seed.hd:
            if hd is not None:
                hd.used = seed.nhits

    def find_seeds_sf(self, st, face):
        c = self.c
        sdt2 = f(64)
        fz1 = self.faces[st][face]
        for hd1 in fz1.hits:
            if hd1.used >= 3:
                continue
            wx1, wy1, x1, y1 = hd1.wx, hd1.wy, hd1.x, hd1.y
            found = 0
            pz1 = self.geo["pz"][st][face][hd1.panel // 2]
            t1 = hd1.t
            ft0, lt0 = self._bins(trunc_int(f(trunc_int(t1)) / self.tbin))
            for f2 in range(face + 1, NFACE):
                fz2 = self.faces[st][f2]
                zc = f((d(self.geo["fz"][st][face]) + d(self.geo["fz"][st][f2])) / d(2))
                r = self._range(fz2, ft0, lt0)
                if r is None:
                    continue
                for h2 in range(r[0], r[1] + 1):
                    hd2 = fz2.hits[h2]
                    if hd2.used >= 3:
                        continue
                    dt = hd2.t - t1
                    if dt < -c["maxDriftTime"]:
                        continue
                    if dt > c["maxDriftTime"]:
                        break
                    dtc = hd1.ct - hd2.ct
                    if abs(dtc) > c["maxDriftTime"]:
                        continue
                    pz2 = self.geo["pz"][st][f2][hd2.panel // 2]
                    if f(pz1["nx"] * pz2["nx"] + pz1["ny"] * pz2["ny"]) < d(-0.5):
                        continue
                    x2, y2 = hd2.x, hd2.y
                    wx2, wy2 = d(hd2.wx), d(hd2.wy)
                    w1w2 = d(wx1) * wx2 + d(wy1) * wy2
                    q12 = d(1.0) - w1w2 * w1w2
                    r12n1 = d((x1 - x2) * wx1 + (y1 - y2) * wy1)
                    r12n2 = d(x1 - x2) * wx2 + d(y1 - y2) * wy2
                    wd1 = -(r12n2 * w1w2 - r12n1) / q12
                    xc = f(d(x1) - d(wx1) * wd1)
                    yc = f(d(y1) - d(wy1) * wd1)
                    wd2 = -(r12n2 - w1w2 * r12n1) / q12
                    c1 = f(wd1 * wd1 / d(hd1.sigw2))
                    c2 = f(wd2 * wd2 / d(hd2.sigw2))
                    if c1 > c["maxChi2Par"] or c2 > c["maxChi2Par"]:
                        continue
                    ctime = (dtc * dtc) / sdt2
                    ctot = ctime + (c1 + c2) / f(2)
                    if ctot > c["maxChi2Seed"]:
                        continue
                    if any(s.hd[face] is hd1 and s.hd[f2] is hd2 for s in self.seeds[st]):
                        continue
                    hd1.chi2min, hd2.chi2min = c1, c2
                    seed = self.new_seed(st)
                    seed.init(hd1, hd2, xc, yc, zc)
                    hd1.seed = hd2.seed = seed
                    self.complete_seed(seed)
                    if seed.chi2_tot_n() > c["maxChi2Seed"]:
                        seed.good = -3000 - seed.index
                    else:
                        if seed.edep_mean() > c["maxSeedEDep"]:
                            seed.good = -2000 - seed.index
                        else:
                            self.compton[st].append(seed)
                        if seed.edep_mean() > c["minProtonSeedEDep"]:
                            self.pseeds[st].append(seed)
                        found = seed.nhits
                    if found >= 3:
                        break
                if found >= 3:
                    break

    def prune_seeds(self, st):
        c = self.c
        ss = self.seeds[st]
        n = len(ss)
        for i1 in range(n - 1):
            ds1 = ss[i1]
            if ds1.good < 0:
                continue
            if ds1.chi2_tot_n() + ds1.chi2_time() > c["maxChi2Seed"]:
                ds1.good = -1000 - i1
                continue
            tm1 = ds1.tmean()
            for i2 in range(i1 + 1, n):
                ds2 = ss[i2]
                if ds2.good < 0:
                    continue
                if ds2.chi2_tot_n() + ds2.chi2_time() > c["maxChi2Seed"]:
                    ds2.good = -1000 - i2
                    continue
                if abs(tm1 - ds2.tmean()) > c["maxSeedDt"]:
                    continue
                nov = sum(1 for face in range(NFACE) if ds1.hd[face] is not None and ds1.hd[face] is ds2.hd[face])
                if nov > 1:
                    if ds1.nhits > ds2.nhits:
                        ds2.good = -1000 - i1
                    elif ds2.nhits > ds1.nhits:
                        ds1.good = -1000 - i2
                        break
                    elif ds1.chi2_tot_n() < ds2.chi2_tot_n():
                        ds2.good = -1000 - i1
                    else:
                        ds1.good = -1000 - i2
                        break
                elif nov > 0:
                    if ds1.nhits == 2:
                        if ds2.nhits > 2:
                            ds1.good = -1000 - i2
                            break
                        if ds1.chi2_tot_n() < ds2.chi2_tot_n():
                            ds2.good = -1000 - i1
                        else:
                            ds1.good = -1000 - i2
                            break
                    elif ds2.nhits == 2:
                        ds2.good = -1000 - i1

    # ---------------------------------------------------------------- deltas
    def _dts(self, base, station, last):
        return f(d(base) + d(self.c["maxDtDs"]) * d(abs(station - last)))

    def recover_seed(self, dc, last, st):
        c = self.c
        td = dc.T0(st)
        xc, yc = dc.cx, dc.cy
        best, closest = c["maxChi2SeedDelta"], None
        dt = self._dts(c["maxSeedDt"], st, last)
        for s in self.compton[st]:
            if not s.is_good():
                continue
            if s.used():
                continue
            if abs(td - s.tmean()) > dt:
                continue
            a, b = self.seed_chi2(s, xc, yc)
            chi2 = (a + b) / f(s.nhits)
            if chi2 < best:
                closest, best = s, chi2
        if closest is not None:
            dc.add_seed(closest)
            closest.delta = dc.index
        return closest is not None

    def recover_station(self, dc, last, st, use_used, recover_seeds):
        c = self.c
        td = dc.T0(st)
        dnx, dny = dc.nx, dc.ny
        with np.errstate(all="ignore"):
            q = td / self.tbin
        if not np.isfinite(q):
            raise OverflowError("non-finite delta time")
        ft0, lt0 = self._bins(trunc_int(q))
        if recover_seeds and self.recover_seed(dc, last, st):
            return 1
        new = None
        mdt = self._dts(c["maxHitDt"], st, last)
        for face in range(NFACE):
            fz = self.faces[st][face]
            r = self._range(fz, ft0, lt0)
            if r is None:
                continue
            for h in range(r[0], r[1] + 1):
                hd = fz.hits[h]
                if hd.delta > 0:
                    continue
                if hd.used >= 3:
                    continue
                if use_used == 0 and hd.used:
                    continue
                ct = hd.ct
                if ct < td - mdt:
                    continue
                if ct > td + mdt:
                    break
                pz = self.geo["pz"][st][face][hd.panel // 2]
                if f(pz["nx"] * d(dnx) + pz["ny"] * d(dny)) < f(0.5):
                    continue
                snx2 = dc.snx2 + d(hd.nx2); snxy = dc.snxy + d(hd.nxy); sny2 = dc.sny2 + d(hd.ny2)
                snxr = dc.snxr + d(hd.nxr); snyr = dc.snyr + d(hd.nyr)
                dd = snx2 * sny2 - snxy * snxy
                with np.errstate(all="ignore"):
                    xc = f((snyr * snx2 - snxr * snxy) / dd)
                    yc = f((snyr * snxy - snxr * sny2) / dd)
                dx, dy = hd.x - xc, hd.y - yc
                w = dx * hd.wx + dy * hd.wy
                n = dx * hd.wy - dy * hd.wx
                drho = f(max(d(abs(n) - self.rcore), 0.0))
                cpar = (w * w) / (hd.sigw2 + self.sig2)
                cperp = (drho * drho) / self.sig2
                chit = cpar + cperp
                if cpar >= c["maxChi2Par"] or cperp >= c["maxChi2Perp"]:
                    continue
                sp, sq = self.delta_chi2(dc, xc, yc)
                cpar = (cpar + sp) / f(dc.nhits + 1)
                cperp = (cperp + sq) / f(dc.nhits + 1)
                dtc = ct - td
                ctime = (dtc * dtc) / f(100)
                if cpar >= c["maxChi2Par"] or cperp >= c["maxChi2Perp"]:
                    continue
                chi2 = cpar + cperp + ctime
                if chi2 >= c["maxChi2Seed"]:
                    continue
                if hd.used:
                    if hd.seed.nhits >= 3 and chit > hd.chi2min:
                        continue
                if new is None:
                    hd.chi2min = chi2
                    new = self.new_seed(st)
                    new.init(hd, None, 0, 0, 0)
                else:
                    if face == new.sface[0]:
                        if chi2 >= new.hd[face].chi2min:
                            continue
                        hd.chi2min = chi2
                        new.replace_first_hit(hd)
                    else:
                        new.add_hit(hd)
                    if ct < new.tmin:
                        new.tmin = ct
                    if ct > new.tmax:
                        new.tmax = ct
            if new is not None:
                new.fp[face] = 1
        if new is not None:
            new.hd[new.sface[0]].seed = new
            dc.add_seed(new)
            new.delta = dc.index
            return 1
        return 0

    def link_delta_seeds(self):
        c = self.c
        for st in range(NST - 1, -1, -1):
            ndelta = len(self.deltas)
            for seed in self.compton[st]:
                if not seed.is_good() or seed.used():
                    continue
                closest, best = None, c["maxChi2SeedDelta"]
                for idc in range(ndelta):
                    dc = self.deltas[idc]
                    if not dc.active() or dc.seeds[st] is not None:
                        continue
                    first = dc.first
                    if first == st:
                        continue
                    gap = first - st
                    if gap > c["maxGap"]:
                        continue
                    dt = dc.T0(st) - seed.tmean()
                    if abs(dt) > c["maxSeedDt"] + c["maxDtDs"] * f(gap):
                        continue
                    snx2 = f(dc.snx2 + seed.snx2); snxy = f(dc.snxy + seed.snxy); sny2 = f(dc.sny2 + seed.sny2)
                    snxr = f(dc.snxr + seed.snxr); snyr = f(dc.snyr + seed.snyr)
                    dd = snx2 * sny2 - snxy * snxy
                    with np.errstate(all="ignore"):
                        xc = (snyr * snx2 - snxr * snxy) / dd
                        yc = (snyr * snxy - snxr * sny2) / dd
                    a, b = self.seed_chi2(seed, xc, yc)
                    chi2 = (a + b) / f(seed.nhits)
                    if chi2 < best:
                        closest, best = dc, chi2
                if closest is not None:
                    closest.add_seed(seed)
                    seed.delta = closest.index
                    continue
                delta = Delta(len(self.deltas), seed, self.geo)
                for st2 in range(st + 1, NST):
                    self.recover_station(delta, delta.last, st2, 1, 1)
                    if st2 - delta.last >= 2:
                        break
                if st > 0:
                    self.recover_station(delta, st, st - 1, 1, 1)
                if delta.nseeds >= c["minNSeeds"]:
                    self.deltas.append(delta.copy())
                else:
                    for s2 in range(delta.first, delta.last + 1):
                        if delta.seeds[s2] is not None:
                            delta.seeds[s2].delta = -1
            for idc in range(ndelta):
                dc = self.deltas[idc]
                first = dc.first
                if first == st or first - st > 3:
                    continue
                self.recover_station(dc, first, st, 1, 0)
        self.merge_deltas()

    def merge_deltas(self):
        c = self.c
        n = len(self.deltas)
        for i1 in range(n - 1):
            dc1 = self.deltas[i1]
            if not dc1.active():
                continue
            x1, y1 = dc1.cx, dc1.cy
            for i2 in range(i1 + 1, n):
                dc2 = self.deltas[i2]
                if not dc2.active():
                    continue
                d2 = (x1 - dc2.cx) * (x1 - dc2.cx) + (y1 - dc2.cy) * (y1 - dc2.cy)
                if d2 > f(400):
                    continue
                t1 = dc1.T0(dc1.last)
                t2 = dc2.T0(dc2.first)
                if abs(t1 - t2) > c["maxDtDc"]:
                    continue
                dds = dc2.first - dc1.last
                if dds < 0 or dds > c["maxGap"]:
                    continue
                dc1.merge(dc2)
                dc2.index = -1000 - dc1.index

    def recover_missing_hits(self):
        c = self.c
        for dc in self.deltas:
            s1, s2 = dc.first, dc.last - 1
            last = -1
            for i in range(s1, s2 + 1):
                if dc.seeds[i] is not None:
                    last = i
                    continue
                self.recover_station(dc, last, i, 1, 0)
            for i in range(dc.first - 1, -1, -1):
                if dc.first - i > c["maxGap"]:
                    break
                self.recover_station(dc, dc.first, i, 1, 0)
            for i in range(dc.last + 1, NST):
                if i - dc.last > c["maxGap"]:
                    break
                self.recover_station(dc, dc.last, i, 1, 0)

    # ---------------------------------------------------------------- protons
    def create_protons(self):
        for st in range(NST - 1, -1, -1):
            for seed in self.pseeds[st]:
                if seed.good < 0 and seed.good > -2000:
                    continue
                if seed.delta_index() >= 0:
                    continue
                t = seed.tmean()
                found = 0
                for pc in list(self.protons):
                    if pc.first - st > 1:
                        continue
                    if abs(pc.t0_at(st) - t) > f(30):
                        continue
                    consistent = 1
                    phi, err = pc.predict_phi(st)
                    if phi > f(-99):
                        consistent = 0
                        dphi = _wrap(phi - self.geo["seed_phi"](seed.cy, seed.cx))
                        if abs(dphi) < err:
                            consistent = 1
                    if consistent:
                        pc.add_seed(seed)
                        found += 1
                if found == 0:
                    pc = Proton(len(self.protons), self.geo)
                    self.protons.append(pc)
                    pc.add_seed(seed)

    def resolve_overlaps(self, pcs):
        novr = {}                              # (station, face) -> (count, hits): stale across pairs, as stock
        nprot = len(pcs)
        for i1 in range(nprot - 1):
            p1 = pcs[i1]
            t1 = p1.tmid
            for i2 in range(i1 + 1, nprot):
                p2 = pcs[i2]
                dt = p2.tmid - t1
                os1 = max(p1.first, p2.first)
                os2 = min(p1.last, p2.last)
                ds = max(os2 - os1, 0)
                if dt > f(60 + 5 * ds):
                    break
                nov = 0
                for st in range(os1, os2 + 1):
                    if p1.nhst[st] == 0 or p2.nhst[st] == 0:
                        continue
                    for face in range(NFACE):
                        novr[(st, face)] = []
                        if p1.nhits(st, face) == 0 or p2.nhits(st, face) == 0:
                            continue
                        for h1 in p1.hd[st][face]:
                            for h2 in p2.hd[st][face]:
                                if h1 is h2:
                                    if len(novr[(st, face)]) < 100:
                                        novr[(st, face)].append(h1)
                                        nov += 1
                                    break
                if nov >= 2:
                    keep, drop = (p2, p1) if p2.ntot >= p1.ntot else (p1, p2)
                    for st in range(os1, os2 + 1):
                        for face in range(NFACE):
                            if (st, face) not in novr:
                                if p1.nhst[st] and p2.nhst[st]:
                                    continue
                                raise OverflowError("resolveProtonOverlaps reads an unset overlap count")
                            for h in list(novr[(st, face)]):
                                drop.remove_hit(st, h, False)
                                h.proton = keep.index
                    drop.update_time()
                    if drop is p1:
                        break

    def merge_protons(self):
        from mpr_reference import std_sort
        pcs = list(self.protons)
        std_sort(pcs, lambda a, b: a.tmid < b.tmid)
        self.resolve_overlaps(pcs)

    def recover_proton_hits(self):
        c = self.c
        for pc in self.protons:
            s1 = max(pc.first - 1, 0)
            s2 = min(pc.last + 1, NST - 1)
            for st in range(s2, s1 - 1, -1):
                pt = pc.t0_at(st)
                phi, err = pc.predict_phi(st)
                for face in range(NFACE):
                    if pc.nhits(st, face) > 0:
                        continue
                    for hd in self.faces[st][face].proton_hits:
                        if abs(hd.ct - pt) > c["maxDriftTime"]:
                            continue
                        if phi > f(-99):
                            dphi = _wrap(phi - hd.phi(self.geo))
                            if abs(dphi) > err:
                                continue
                        pid = pc.pid[st][face]
                        if pid == -1 or hd.panel_id == pid:
                            pc.add_hit(st, hd)

    def find_protons(self):
        c = self.c
        for st in range(NST):
            for face in range(NFACE):
                fz = self.faces[st][face]
                fz.proton_hits = [hd for hd in fz.hits if hd.edep > c["minProtonHitEDep"] and hd.delta < 0]
        self.create_protons()
        self.merge_protons()
        if c["pickupProtonHits"]:
            self.recover_proton_hits()

    # ---------------------------------------------------------------- run
    def run(self):
        c = self.c
        self.order_hits()
        for st in range(NST):
            for face in range(NFACE - 1):
                self.find_seeds_sf(st, face)
            self.prune_seeds(st)
        self.link_delta_seeds()
        self.recover_missing_hits()
        self.merge_deltas()
        for dc in self.deltas:
            if not dc.active():
                continue
            if dc.nhits < c["minDeltaNHits"]:
                dc.mask |= 2
            if dc.edep_mean() > c["maxDeltaEDep"]:
                dc.mask |= 1
            if dc.mask == 0:
                dc.mark_hits_used()
        for st in range(NST):
            for seed in self.seeds[st]:
                if seed.delta_index() >= 0:
                    continue
                did = -1
                for hd in seed.hd:
                    if hd is None:
                        continue
                    if hd.delta < 0:
                        break
                    if did < 0:
                        did = hd.delta
                    elif did != hd.delta:
                        did = -1
                        break
                if did >= 0:
                    seed.delta = 10000 + did
        if c["flagProtonHits"]:
            self.find_protons()

    def flags(self):
        c = self.c
        out = [int(x) & ~BKG for x in self.h["flag"]]
        if c["flagProtonHits"]:
            out = [x | ESEL for x in out]
        for dc in self.deltas:
            if not dc.active() or dc.mask != 0:
                continue
            for st in range(dc.first, dc.last + 1):
                s = dc.seeds[st]
                if s is None:
                    continue
                for hd in s.hd:
                    if hd is not None:
                        out[hd.k] |= BKG
        np15 = 0
        for pc in self.protons:
            if pc.ntot >= 15:
                np15 += 1
            if pc.nst == 1 and pc.edep_mean() < f(0.004):
                continue
            for st in range(pc.first, pc.last + 1):
                for face in range(NFACE):
                    for hd in pc.hd[st][face]:
                        out[hd.k] &= ~ESEL
        return np.array(out, np.uint32), np15


def run(hits, geo, cfg, spill_on=True):
    """One event: panel hits (collection order; sid, flag, time = the early-end time, ctime = the corrected time, x,
    y, udx, udy, uvar, edep, nsh) -> (TrigFlagPH flags, proton candidates with >= 15 hits)."""
    fd = Finder(hits, geo, cfg, spill_on)
    fd.run()
    return fd.flags()


def geo_from(g):
    """`deltafinder.geometry()` arrays -> this module's geometry (float atan2 for the proton azimuths)."""
    at = lambda y, x: f(math.atan2(float(f(y)), float(f(x))))
    return {"pz": [[[{"wx": g["wx"][a, b, c], "wy": g["wy"][a, b, c], "nx": g["nx"][a, b, c], "ny": g["ny"][a, b, c]}
                     for c in range(3)] for b in range(NFACE)] for a in range(NST)],
            "fz": g["fz"], "stationZ": g["stationZ"], "atan2": at,
            "seed_phi": lambda y, x: f(0) if (y == 0 and x == 0) else at(y, x)}
