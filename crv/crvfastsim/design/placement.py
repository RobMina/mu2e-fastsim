"""Automatic placement of a CRV design in the detector envelope: per-sector shifts, sheet moves, thinning and trims.

A design laid out by layout.py (thicker stacks, re-tiled sectors, fillers, staggered joints) no longer fits the stock
sector positions: the stacks grow along their layer direction into the external shielding, the borated-poly sheets
and each other. ``place`` finds, within configured ranges, a placement in which nothing overlaps:

  modules   each sector is the union of its G4 mother parts (one box per layer across all modules, read-out ends
            grown by CMBOffset + CMBHalfThickness, each absorber gap, the strongback and the cover sheet; per length
            segment and filler), plus its FEB boxes; tilted modules use their axis-aligned hull;
  shifts    each stock sector (with its segments and filler) moves along its layer direction by -shift, within
            +-``max_shift``. Its hard obstacles are the fixed envelope (envelope.py), the sectors that face it along
            its layer direction in the stock geometry, and the sheets it does not push. Within the free range it
            takes the shift that needs the least sheet displacement, then the smallest |shift|;
  sheets    a sheet's CRV side is taken from the stock geometry (the facing sector nearest to it); the CRV-side
            sectors push it away from the CRV, it never moves toward the CRV, its thickness grows away from the
            CRV; it must clear the fixed envelope and the other sheets by ``sheet_clearance`` and the sectors on its
            far side by ``clearance``. A sector that reaches a sheet only through its re-layout (it did not face it
            in stock) cuts the sheet's end back, by at most ``max_trim``;
  clearance every pair gets ``clearance`` (sheet pairs ``sheet_clearance``) or the stock gap if smaller, so the stock
            design is always feasible;
  FEBs      a sector whose FEB boxes overlap anything drops them (``drop_febs``; no FEB material in the fast model);
  stagger   the joint stagger is the largest of ``stagger_mm`` that places;
  remedies  when the base placement is infeasible, a remedy is a range: a world move of named sectors along a
            direction up to ``max_mm`` (e.g. the top sectors downstream) or a sheet thinned down to
            ``min_thickness``. Each takes the smallest amount that places. Policy "each" returns the base placement
            (if feasible) and every remedy on its own, so the optimisation loop can score them side by side; "first"
            returns the base placement or the first remedy that places.
The fast model has no beam-background simulation: a thinner or trimmed sheet is recorded (``shield_modified``) and
its deadtime is flagged as uncalibrated (deadtime.py is calibrated on the stock shielding); no shield term is made up.
The result is reported with every remaining overlap when nothing places (``feasible`` False).
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field

import numpy as np

from .. import geometry as G
from . import envelope as E
from . import layout as L
from . import module as M

POLICIES = ("first", "each")
TOL = 1e-6


@dataclass(frozen=True)
class Remedy:
    """One way out of an infeasible placement: a world move of ``sectors`` along ``direction`` by up to ``max_mm``,
    and/or the sheets ``sheets`` (G4 names, ``type<k>`` or label substrings) thinned together down to
    ``min_thickness`` [mm]. A move or a thinning alone takes its smallest amount that places; both together take the
    least thinning, then the smallest move."""
    name: str
    sectors: tuple = ()
    direction: tuple = (0.0, 0.0, 1.0)
    max_mm: float = 0.0
    sheets: tuple = ()
    min_thickness: float | None = None
    step: float = 5.0

    def __post_init__(self):
        if isinstance(self.sheets, str):
            object.__setattr__(self, "sheets", (self.sheets,))
        if not self.sectors and not self.sheets:
            raise ValueError(f"remedy {self.name}: give sectors (a move), sheets (thinning) or both")
        if self.sectors and not self.max_mm > 0:
            raise ValueError(f"remedy {self.name}: max_mm must be > 0")
        if self.sheets and not (self.min_thickness is not None and self.min_thickness > 0):
            raise ValueError(f"remedy {self.name}: min_thickness must be > 0")
        if not self.step > 0:
            raise ValueError(f"remedy {self.name}: step must be > 0")

    @property
    def kind(self):
        return "+".join(k for k, on in (("move", bool(self.sectors)), ("thin", bool(self.sheets))) if on)


@dataclass(frozen=True)
class PlacementConfig:
    layout: L.LayoutConfig = field(default_factory=L.LayoutConfig)
    stagger_mm: tuple = (60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0)   # candidates, largest that places
    clearance: float = 3.0          # sector parts to anything (or the stock gap, if smaller)
    sheet_clearance: float = 1.0    # a sheet to the fixed envelope and to other sheets (or the stock gap)
    max_shift: float = 500.0        # mm, per sector
    max_trim: float = 120.0         # mm a sheet end may be cut back by (0: no trims)
    drop_febs: bool = True
    near: float = 300.0             # mm: sectors farther than this from a sheet or sector do not face it
    remedies: tuple = ()
    policy: str = "each"
    upstream: str | None = E.UPSTREAM
    downstream: str | None = E.DOWNSTREAM
    hall: str | None = E.HALL
    movable_material: str = E.MOVABLE_MATERIAL
    max_iter: int = 40
    tie_chains: bool = True         # sectors that follow each other in stock (layout.sector_chains: T1..T5, ...)
                                    # share one shift, so their plane has no step; False = each sector alone

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if not self.stagger_mm:
            raise ValueError("stagger_mm needs at least one candidate")
        if self.clearance < 0 or self.sheet_clearance < 0 or self.max_trim < 0 or not self.max_shift > 0:
            raise ValueError("clearances and max_trim must be >= 0, max_shift > 0")
        names = [r.name for r in self.remedies]
        if len(set(names)) != len(names):
            raise ValueError("remedy names must be unique")

    def as_dict(self):
        return dataclasses.asdict(self)


# ------------------------------------------------------------------ sector parts (the G4 mother volumes)
@dataclass
class Parts:
    lo: np.ndarray            # [n, 3]
    hi: np.ndarray
    group: np.ndarray         # [n] stock sector index
    volume: list              # Offline sector volume name (segments <S>s<k>, fillers <P>s9<k>)
    feb: np.ndarray           # [n] bool

    def __len__(self):
        return len(self.lo)

    def take(self, m):
        m = np.asarray(m)
        idx = np.flatnonzero(m) if m.dtype == bool else m.astype(np.int64)
        return Parts(self.lo[idx], self.hi[idx], self.group[idx], [self.volume[i] for i in idx], self.feb[idx])

    def moved(self, t):
        """Parts translated by t [n_groups, 3] per group."""
        return Parts(self.lo + t[self.group], self.hi + t[self.group], self.group, self.volume, self.feb)


def _aabb(center, half_world):
    return (center - half_world).min(0), (center + half_world).max(0)


def sector_parts(dg, c, stock):
    """The G4 mother parts and FEB boxes of every sector of a design geometry (module docstring)."""
    lo, hi, grp, vol, feb = [], [], [], [], []
    cmb = float(c.get("crs.CMBOffset", 0.0)) + float(c.get("crs.CMBHalfThickness", 0.0))
    sb, al = float(c.get("crs.strongBackThickness", 0.0)), float(c.get("crs.aluminumSheetThickness", 0.0))
    t = float(c["crs.scintillatorBarThickness"]) if dg.design.thickness is None else float(dg.design.thickness)
    febh = np.atleast_1d(c.get("crs.FEBHalfLengths", [0, 0, 0])).astype(float)
    f_mod, f_edge = float(c.get("crs.FEBDistanceToModule", 0.0)), float(c.get("crs.FEBDistanceToEdge", 0.0))
    f_2w, f_2t = float(c.get("crs.FEBDistanceBetween2FEBsW", 0.0)), float(c.get("crs.FEBDistanceBetween2FEBsT", 0.0))
    hw_all = np.einsum("bj,bji->bi", np.abs(dg.halflen), np.abs(dg.axes[dg.sector]))
    for s, nm in enumerate(dg.sector_names):
        par = L.parent_of(nm, stock)
        g = stock.index(par)
        A = dg.axes[s]
        an, aw, al_ = (int(np.argmax(np.abs(A[i]))) for i in range(3))
        sgn = float(np.sign(A[0][an]))
        b = np.flatnonzero(dg.sector == s)
        if not len(b):
            continue
        nseg = int(dg.segment[b].max()) + 1
        nl = int(dg.n_layers[s])
        only = bool(c.get(f"crs.countersOnly{nm}", False))
        s0, s1 = bool(c.get(f"crs.sipmsAtSide0{nm}", True)), bool(c.get(f"crs.sipmsAtSide1{nm}", True))
        nfeb = (int(c.get(f"crs.FEBBoxesAtSide0{nm}", 0)), int(c.get(f"crs.FEBBoxesAtSide1{nm}", 0)))
        for k in range(nseg):
            name = (f"{par}s{90 + k}" if par != nm else (nm if nseg == 1 else f"{nm}s{k}"))
            bk = b[dg.segment[b] == k]
            for layer in range(nl):
                bl = bk[dg.layer[bk] == layer]
                lo0, hi0 = _aabb(dg.center[bl], hw_all[bl])
                lo1, hi1 = lo0.copy(), hi0.copy()
                if s0:
                    lo1[al_] -= cmb
                if s1:
                    hi1[al_] += cmb
                lo.append(lo1); hi.append(hi1); grp.append(g); vol.append(name); feb.append(False)
                if nl > 1 and not only and (layer == 0 and sb > 0 or layer == nl - 1 and al > 0):
                    th, sd = (sb, -1.0) if layer == 0 else (al, 1.0)
                    c0 = 0.5 * (lo0 + hi0)
                    h0 = 0.5 * (hi0 - lo0)
                    c0[an] += sd * sgn * 0.5 * (t + th)
                    h0[an] = 0.5 * th
                    lo.append(c0 - h0); hi.append(c0 + h0); grp.append(g); vol.append(name); feb.append(False)
                if layer == nl - 1 and nl > 1 and not only and any(nfeb) and febh.any():
                    for m in np.unique(dg.module[bl]):
                        bm = bl[dg.module[bl] == m]
                        llo, lhi = _aabb(dg.center[bm], hw_all[bm])
                        pos, half = 0.5 * (llo + lhi), 0.5 * (lhi - llo)
                        for side, nb in enumerate(nfeb):
                            if nb == 0:
                                continue
                            for fl in range(2):
                                for wk in ((0.0,) if nb == 1 else (-0.5, 0.5)):
                                    cen = pos.copy()
                                    cen[an] = pos[an] + sgn * (0.5 * t + f_mod + fl * f_2t)
                                    cen[aw] = pos[aw] + wk * f_2w
                                    cen[al_] = pos[al_] - half[al_] + f_edge if side == 0 else \
                                        pos[al_] + half[al_] - f_edge
                                    h = np.zeros(3)
                                    h[an], h[aw], h[al_] = febh
                                    lo.append(cen - h); hi.append(cen + h); grp.append(g)
                                    vol.append(f"{name}:FEB"); feb.append(True)
        a = np.flatnonzero(dg.abs_sector == s)
        if len(a) and not only:
            ha = np.abs(dg.abs_halflen[a] @ A)
            key = np.round((dg.abs_center[a] @ A[0]) * 10).astype(np.int64)
            for kk in np.unique(key):
                aa = a[key == kk]
                h = ha[key == kk]
                lo.append((dg.abs_center[aa] - h).min(0)); hi.append((dg.abs_center[aa] + h).max(0))
                grp.append(g); vol.append(nm if nseg == 1 or par != nm else f"{nm}s0"); feb.append(False)
    return Parts(np.array(lo), np.array(hi), np.array(grp, np.int64), vol, np.array(feb, bool))


def _box_sep(alo, ahi, blo, bhi):
    """Signed separation [na, nb] of axis-aligned boxes (max over axes of the gap)."""
    gap = np.maximum(blo[None] - ahi[:, None], alo[:, None] - bhi[None])
    return gap.max(2)


def _near(lo, hi, olo, ohi, reach):
    """Mask [n_other]: other boxes within reach of the AABB of lo/hi."""
    blo, bhi = lo.min(0) - reach, hi.max(0) + reach
    return np.all((olo < bhi) & (ohi > blo), axis=1)


def _intervals_union(t0, t1):
    """Sorted union of open intervals (nan rows dropped)."""
    ok = np.isfinite(t0) | np.isfinite(t1)
    ok &= ~(np.isnan(t0) | np.isnan(t1))
    iv = sorted(zip(t0[ok].tolist(), t1[ok].tolist()))
    out = []
    for a, b in iv:
        if out and a < out[-1][1] - TOL:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# ------------------------------------------------------------------ the world one design is placed in
class _World:
    """A laid-out design (zero shifts) and what its placement is checked against."""

    def __init__(self, design, base_text, env, cfg, stagger, stock=None):
        self.design, self.cfg, self.env = design, cfg, env
        self.lcfg = dataclasses.replace(cfg.layout, stagger_mm=float(stagger))
        self.c0 = G.parse_simpleconfig(base_text)
        self.base_text = base_text
        self.names = list(self.c0["crs.sectorNames"])
        self.ldir = np.array([np.asarray(self.c0[f"crs.layerDirection{n}"], float) for n in self.names])
        text = L.counters_text(design, L.Placement(), base_text, self.lcfg)
        self.dg = L.build(design, text, None, self.lcfg)
        self.parts = sector_parts(self.dg, G.parse_simpleconfig(text), self.names)
        self.stock = stock if stock is not None else self
        if stock is None:                       # the stock world: stock separations and sheet sides
            self._stock_tables()

    def _stock_tables(self):
        """Stock-geometry relations: group hull boxes, part-level separations between groups and to the fixed and
        sheet pieces, which groups face each other along their layer direction, and each sheet's CRV side."""
        p, ng = self.parts.take(~self.parts.feb), len(self.names)
        self.hull = np.array([[p.lo[p.group == g].min(0), p.hi[p.group == g].max(0)] for g in range(ng)])
        sep = _box_sep(p.lo, p.hi, p.lo, p.hi)
        self.gsep = np.full((ng, ng), np.inf)
        for g in range(ng):
            for h in range(ng):
                if g != h:
                    self.gsep[g, h] = sep[np.ix_(p.group == g, p.group == h)].min()
        self.faces = np.zeros((ng, ng), bool)          # faces[g, h]: h is an obstacle of g along g's layer axis
        for g in range(ng):
            a = int(np.argmax(np.abs(self.ldir[g])))
            for h in range(ng):
                if g == h:
                    continue
                (glo, ghi), (hlo, hhi) = self.hull[g], self.hull[h]
                ax = np.maximum(hlo - ghi, glo - hhi)
                lat = [i for i in range(3) if i != a]
                if all(ax[i] < 0 for i in lat) and ax[a] >= -1.0 and ax[a] < self.cfg.near:
                    self.faces[g, h] = True
        # a pair that faces both ways: the sector that rests on the other (the other on its -layerDirection side)
        # yields; the other ignores it (the hand rule: T rests on R, E rests on the sheet above T)
        cen = self.hull.mean(1)
        below = lambda g, h: float((cen[h] - cen[g]) @ self.ldir[g]) < 0   # noqa: E731
        for g in range(ng):
            for h in range(g + 1, ng):
                if self.faces[g, h] and self.faces[h, g]:
                    bg, bh = below(g, h), below(h, g)
                    if bg != bh:
                        self.faces[g, h], self.faces[h, g] = bg, bh
        fx = self.env.fixed
        self.fsep = np.full((ng, len(fx)), np.inf)      # group vs fixed piece, stock
        for g in range(ng):
            q = p.take(p.group == g)
            m = _near(q.lo, q.hi, fx.lo, fx.hi, self.cfg.near)
            if m.any():
                self.fsep[g, m] = E.separation(q.lo, q.hi, fx.take(np.flatnonzero(m))).min(0)
        sh = self.env.sheets
        slo = np.array([s.center - s.half for s in sh]).reshape(-1, 3)
        shi = np.array([s.center + s.half for s in sh]).reshape(-1, 3)
        self.ssep_g = np.full((ng, len(sh)), np.inf)    # group vs sheet, stock
        if len(sh):
            sp = _box_sep(p.lo, p.hi, slo, shi)
            for g in range(ng):
                self.ssep_g[g] = sp[p.group == g].min(0)
        self.ssep_f = E.separation(slo, shi, fx) if len(sh) else np.zeros((0, len(fx)))   # sheet vs fixed, stock
        self.ssep_s = _box_sep(slo, shi, slo, shi) if len(sh) else np.zeros((0, 0))
        np.fill_diagonal(self.ssep_s, np.inf)
        self.side = []                                   # per sheet: (direction away from the CRV, crv groups)
        for j, s in enumerate(sh):
            a = s.axis
            lat = [i for i in range(3) if i != a]
            face = [g for g in range(ng) if all(min(self.hull[g][1][i], shi[j][i]) - max(self.hull[g][0][i], slo[j][i])
                                                 > 0 for i in lat)
                    and max(self.hull[g][0][a] - shi[j][a], slo[j][a] - self.hull[g][1][a]) < self.cfg.near]
            if not face:
                self.side.append((0.0, [], []))
                continue
            gap = lambda g: max(self.hull[g][0][a] - shi[j][a], slo[j][a] - self.hull[g][1][a])   # noqa: E731
            near = min(face, key=gap)
            d = 1.0 if s.center[a] > 0.5 * (self.hull[near][0][a] + self.hull[near][1][a]) else -1.0
            crv = [g for g in face if d * (s.center[a] - 0.5 * (self.hull[g][0][a] + self.hull[g][1][a])) > 0]
            self.side.append((d, crv, [g for g in face if g not in crv]))

    # pads: the clearance, or the stock gap if smaller (never below a stock overlap)
    def pad_gg(self, g, h):
        return min(self.cfg.clearance, self.stock.gsep[g, h])

    def pad_gf(self, g):
        return np.minimum(self.cfg.clearance, self.stock.fsep[g])

    def pad_gs(self, g, j):
        return min(self.cfg.clearance, self.stock.ssep_g[g, j])

    def pad_sf(self, j):
        return np.minimum(self.cfg.sheet_clearance, self.stock.ssep_f[j])

    def pad_ss(self, j, k):
        return min(self.cfg.sheet_clearance, self.stock.ssep_s[j, k])


@dataclass
class _SheetState:
    lo: np.ndarray
    hi: np.ndarray
    moved: float = 0.0
    thickness: float = 0.0
    trimmed: dict = field(default_factory=dict)


def _sheet_index(env, sel):
    for j, s in enumerate(env.sheets):
        if sel in (s.name, f"type{s.type}") or sel.lower() in s.label.lower():
            return j
    raise KeyError(f"no movable sheet matches {sel!r}: {[(s.name, s.type, s.label) for s in env.sheets]}")


class _Solver:
    def __init__(self, world, moves=None, thickness=None):
        self.w, self.cfg = world, world.cfg
        ng = len(world.names)
        self.T0 = np.zeros((ng, 3))
        for sec, v in (moves or {}).items():
            self.T0[world.names.index(sec)] += np.asarray(v, float)
        self.thick = dict(thickness or {})
        self.shift = np.zeros(ng)
        self.no_feb = set()
        self.has = np.array([np.any(world.parts.group == g) for g in range(ng)])
        self.ties = []                                   # index sets sharing one shift (PlacementConfig.tie_chains)
        if self.cfg.tie_chains:
            root = list(range(ng))

            def find(i):
                while root[i] != i:
                    i = root[i]
                return i
            for b, a in L.sector_chains(world.c0).items():
                root[find(world.names.index(b))] = find(world.names.index(a))
            sets = {}
            for g in range(ng):
                sets.setdefault(find(g), []).append(g)
            self.ties = [v for v in sets.values() if len(v) > 1]

    def tie(self):
        """Every tied set takes the shift of its member that moved furthest from stock."""
        for t in self.ties:
            k = max(t, key=lambda g: abs(self.shift[g]))
            self.shift[t] = self.shift[k]

    def translation(self, shift=None):
        s = self.shift if shift is None else shift
        return self.T0 - s[:, None] * self.w.ldir

    def parts(self, shift=None, febs=False):
        p = self.w.parts.moved(self.translation(shift))
        keep = ~p.feb if not febs else np.array([not f or self.w.names[g] not in self.no_feb
                                                  for f, g in zip(p.feb, p.group)])
        return p.take(keep)

    # ---- sheets
    def need(self, j, parts, groups, lo=None, hi=None):
        """How far (along the sheet's away-from-CRV direction, Mu2e mm x d) the near face of sheet j must be for the
        parts of ``groups`` that overlap it laterally (-inf if none); lo/hi: the sheet's lateral extent."""
        w, s = self.w, self.w.env.sheets[j]
        a, d = s.axis, w.stock.side[j][0]
        lo = s.center - s.half if lo is None else lo
        hi = s.center + s.half if hi is None else hi
        lat = [i for i in range(3) if i != a]
        q = parts.take(np.isin(parts.group, list(groups)))
        if not len(q):
            return -np.inf
        ok = np.all((q.lo[:, lat] < hi[lat]) & (q.hi[:, lat] > lo[lat]), axis=1)
        if not ok.any():
            return -np.inf
        faces = q.hi[ok, a] if d > 0 else -q.lo[ok, a]
        pads = np.array([w.pad_gs(g, j) for g in q.group[ok]])
        return float((faces + pads).max())

    def sheets(self, parts):
        """Sheet boxes for the current sector parts: trims against sectors that reach a sheet only through their
        re-layout, then the push by the CRV-side sectors; thickness from self.thick."""
        w, out = self.w, []
        for j, s in enumerate(w.env.sheets):
            a, (d, crv, _far) = s.axis, w.stock.side[j]
            lo, hi = (s.center - s.half).copy(), (s.center + s.half).copy()
            t0 = 2 * s.half[a]
            t = float(self.thick.get(j, t0))
            st = _SheetState(lo, hi, 0.0, t)
            if d == 0.0:                                   # nothing in front: stock position, thickness about centre
                lo[a], hi[a] = s.center[a] - t / 2, s.center[a] + t / 2
                out.append(st)
                continue
            lat = [i for i in range(3) if i != a]
            # trims: parts of groups that do not face the sheet in stock but overlap its slab laterally
            if self.cfg.max_trim > 0:
                other = np.array([g not in crv and g not in _far for g in parts.group])
                q = parts.take(other)
                for _ in range(4):
                    pads = np.array([w.pad_gs(g, j) for g in q.group]) if len(q) else np.zeros(0)
                    hit = np.all((q.lo - pads[:, None] < hi) & (q.hi + pads[:, None] > lo), axis=1)
                    if not hit.any():
                        break
                    best = None
                    for i in lat:
                        for e in (-1, 1):                  # cut the -i or +i end back
                            cut = (q.hi[hit, i] + pads[hit] - lo[i]).max() if e < 0 else \
                                (hi[i] - (q.lo[hit, i] - pads[hit])).max()
                            if 0 < cut <= self.cfg.max_trim - st.trimmed.get(f"{'-+'[e > 0]}{'xyz'[i]}", 0.0) + TOL \
                                    and (best is None or cut < best[0]):
                                best = (cut, i, e)
                    if best is None:
                        break
                    cut, i, e = best
                    if e < 0:
                        lo[i] += cut
                    else:
                        hi[i] -= cut
                    k = f"{'-+'[e > 0]}{'xyz'[i]}"
                    st.trimmed[k] = st.trimmed.get(k, 0.0) + float(cut)
            near0 = s.center[a] - d * t0 / 2
            need = max(d * near0, self.need(j, parts, crv, lo, hi))
            nf = d * need
            ff = nf + d * t
            lo[a], hi[a] = min(nf, ff), max(nf, ff)
            st.moved = float(d * (nf - near0))
            out.append(st)
        return out

    def sheet_violations(self, sheets, parts):
        """Overlaps of each sheet with the fixed envelope, other sheets and non-pushing sectors."""
        w, out = self.w, []
        fx = w.env.fixed
        for j, st in enumerate(sheets):
            nm = w.env.sheets[j].name
            m = _near(st.lo[None], st.hi[None], fx.lo, fx.hi, 50.0)
            if m.any():
                idx = np.flatnonzero(m)
                sep = E.separation(st.lo, st.hi, fx.take(idx))[0]
                bad = sep < w.pad_sf(j)[idx] - TOL
                for i in idx[bad]:
                    out.append(dict(a=nm, b=fx.volume[i], kind="sheet-envelope",
                                    depth_mm=round(float(w.pad_sf(j)[i] - sep[list(idx).index(i)]), 2)))
            for k, o in enumerate(sheets):
                if k <= j:
                    continue
                sep = _box_sep(st.lo[None], st.hi[None], o.lo[None], o.hi[None])[0, 0]
                if sep < w.pad_ss(j, k) - TOL:
                    out.append(dict(a=nm, b=w.env.sheets[k].name, kind="sheet-sheet",
                                    depth_mm=round(float(w.pad_ss(j, k) - sep), 2)))
            crv = w.stock.side[j][1]
            q = parts.take(~np.isin(parts.group, crv))
            if len(q):
                sep = _box_sep(q.lo, q.hi, st.lo[None], st.hi[None])[:, 0]
                pads = np.array([w.pad_gs(g, j) for g in q.group])
                for i in np.flatnonzero(sep < pads - TOL):
                    out.append(dict(a=q.volume[i], b=nm, kind="sector-sheet", depth_mm=round(float(pads[i] - sep[i]), 2)))
        return out

    def sheet_room(self, j, sheets, parts):
        """How far sheet j can be pushed away from the CRV from its stock face before it meets a hard obstacle."""
        w, s = self.w, self.w.env.sheets[j]
        d, crv, _ = w.stock.side[j]
        if d == 0.0:
            return 0.0
        a = s.axis
        st = sheets[j]
        t = st.hi[a] - st.lo[a]
        lo, hi = st.lo.copy(), st.hi.copy()
        near0 = s.center[a] - d * 2 * s.half[a] / 2
        if d > 0:
            lo[a], hi[a] = near0, near0 + t
        else:
            lo[a], hi[a] = near0 - t, near0
        dv = np.zeros(3)
        dv[a] = d
        starts = []
        fx = w.env.fixed
        m = _near(lo[None], hi[None], fx.lo, fx.hi, self.cfg.max_shift)
        if m.any():
            idx = np.flatnonzero(m)
            for pad in np.unique(w.pad_sf(j)[idx]):
                sel = idx[w.pad_sf(j)[idx] == pad]
                t0, t1 = E.sweep(lo, hi, fx.take(sel), dv, pad)
                starts += [(a_, b_) for a_, b_ in zip(t0[0], t1[0]) if np.isfinite(a_) or np.isfinite(b_)]
        for k, o in enumerate(sheets):
            if k == j:
                continue
            t0, t1 = E.sweep(lo, hi, E.PieceSet.boxes(["s"], o.lo, o.hi), dv, w.pad_ss(j, k))
            starts.append((t0[0, 0], t1[0, 0]))
        q = parts.take(~np.isin(parts.group, list(crv) + list(w.stock.side[j][2])))   # far-side sectors yield
        if len(q):
            mm = _near(lo[None], hi[None], q.lo, q.hi, self.cfg.max_shift)
            for g in np.unique(q.group[mm]):
                qq = q.take(mm & (q.group == g))
                t0, t1 = E.sweep(lo, hi, E.PieceSet.boxes(qq.volume, qq.lo, qq.hi), dv, w.pad_gs(g, j))
                starts += list(zip(t0[0], t1[0]))
        room = np.inf
        for a_, b_ in starts:
            if np.isnan(a_) or np.isnan(b_):
                continue
            if b_ <= TOL:                                   # behind the sheet
                continue
            room = min(room, a_)
        return float(room)

    # ---- one sector
    def group_obstacles(self, g, parts, sheets, frozen=False):
        """(PieceSet, pads) of every hard obstacle of group g, and the soft sheets it pushes (frozen: every sheet is
        hard where it is)."""
        w = self.w
        sets, pads = [], []
        fx = w.env.fixed
        sets.append(fx)
        pads.append(w.pad_gf(g))
        for h in np.flatnonzero(w.stock.faces[g]):
            q = parts.take(parts.group == h)
            if len(q):
                sets.append(E.PieceSet.boxes(q.volume, q.lo, q.hi))
                pads.append(np.full(len(q), w.pad_gg(g, h)))
        soft = []
        for j, st in enumerate(sheets):
            d, crv, _ = w.stock.side[j]
            if g in crv and not frozen:
                room = self.sheet_room(j, sheets, parts)
                s = w.env.sheets[j]
                a = s.axis
                t = st.hi[a] - st.lo[a]
                near0 = s.center[a] - d * s.half[a]
                lim = near0 + d * max(room, 0.0)
                lo, hi = st.lo.copy(), st.hi.copy()
                lo[a], hi[a] = (lim, lim + t) if d > 0 else (lim - t, lim)
                sets.append(E.PieceSet.boxes([s.name], lo, hi))
                pads.append(np.array([w.pad_gs(g, j)]))
                soft.append((j, d, near0, room))
            else:
                sets.append(E.PieceSet.boxes([w.env.sheets[j].name], st.lo, st.hi))
                pads.append(np.array([w.pad_gs(g, j)]))
        return E.PieceSet.cat(sets), np.concatenate(pads), soft

    def solve_group(self, g, parts, sheets, relax=False):
        """(shift, ok): the shift of group g with the others where they are; relax: the sheets frozen, the point of
        g's current free range closest to its stock position."""
        w, cfg = self.w, self.cfg
        own = self.w.parts.take((self.w.parts.group == g) & ~self.w.parts.feb)
        if not len(own):
            return 0.0, True
        base = self.translation()[g] + self.shift[g] * w.ldir[g]        # translation at shift 0
        lo, hi = own.lo + base, own.hi + base
        obs, pads, soft = self.group_obstacles(g, parts, sheets, frozen=relax)
        d = -w.ldir[g]
        reach = cfg.max_shift + cfg.clearance + 1.0
        m = _near(lo, hi, obs.lo, obs.hi, reach)
        t0s, t1s = [], []
        idx = np.flatnonzero(m)
        for pad in np.unique(pads[idx]):
            sel = idx[pads[idx] == pad]
            t0, t1 = E.sweep(lo, hi, obs.take(sel), d, float(pad))
            t0s.append(t0.ravel()); t1s.append(t1.ravel())
        bad = _intervals_union(np.concatenate(t0s) if t0s else np.zeros(0), np.concatenate(t1s) if t1s else np.zeros(0))
        R = cfg.max_shift
        if relax:
            cur = self.shift[g]
            if any(a_ + TOL < cur < b_ - TOL for a_, b_ in bad):
                return cur, False
            lo_b = max([b_ for a_, b_ in bad if b_ <= cur + TOL] + [-R])
            hi_b = min([a_ for a_, b_ in bad if a_ >= cur - TOL] + [R])
            return float(min(max(0.0, lo_b), hi_b)), True
        cands = [0.0, -R, R] + [x for iv in bad for x in iv if -R - TOL <= x <= R + TOL]
        for j, dd, near0, _room in soft:                  # where g stops pushing each sheet
            a = w.env.sheets[j].axis
            k = dd * d[a]
            if abs(k) < 1e-12:
                continue
            lat = [i for i in range(3) if i != a]
            st = sheets[j]
            ok = np.all((lo[:, lat] < st.hi[lat]) & (hi[:, lat] > st.lo[lat]), axis=1)
            if ok.any():
                face0 = (hi[ok, a] if dd > 0 else -lo[ok, a]).max() + w.pad_gs(g, j)
                cands.append(float((dd * near0 - face0) / k))
        free = [x for x in cands if -R - TOL <= x <= R + TOL and not any(a_ + TOL < x < b_ - TOL for a_, b_ in bad)]
        if not free:
            return self.shift[g], False

        def cost(s):
            c = 0.0
            for j, dd, near0, _room in soft:
                a = w.env.sheets[j].axis
                lat = [i for i in range(3) if i != a]
                st = sheets[j]
                x_lo, x_hi = lo + s * d, hi + s * d
                ok = np.all((x_lo[:, lat] < st.hi[lat]) & (x_hi[:, lat] > st.lo[lat]), axis=1)
                if ok.any():
                    face = (x_hi[ok, a] if dd > 0 else -x_lo[ok, a]).max() + w.pad_gs(g, j)
                    c += max(0.0, face - dd * near0)
            return c
        best = min(free, key=lambda s: (round(cost(s), 3), round(abs(s), 3)))
        return float(best), True

    def run(self):
        ng = len(self.w.names)
        ok = np.ones(ng, bool)
        for it in range(self.cfg.max_iter):
            old = self.shift.copy()
            for g in range(ng):
                if not self.has[g]:
                    continue
                parts = self.parts()
                sheets = self.sheets(parts)
                s, ok[g] = self.solve_group(g, parts, sheets)
                self.shift[g] = s
            self.tie()
            if np.max(np.abs(self.shift - old)) < 1e-3:
                break
        self.iterations = it + 1
        for g in range(ng):                               # back toward stock within the free range, sheets frozen
            if self.has[g] and ok[g]:
                parts = self.parts()
                s, _ = self.solve_group(g, parts, self.sheets(parts), relax=True)
                self.shift[g] = s
        self.tie()
        self.unsolved = [self.w.names[g] for g in range(ng) if not ok[g]]
        return self

    # ---- final check
    def violations(self):
        w = self.w
        parts = self.parts()
        sheets = self.sheets(parts)
        out = []
        fx = w.env.fixed
        for g in range(len(w.names)):
            q = parts.take(parts.group == g)
            if not len(q):
                continue
            m = _near(q.lo, q.hi, fx.lo, fx.hi, 50.0)
            if m.any():
                idx = np.flatnonzero(m)
                sep = E.separation(q.lo, q.hi, fx.take(idx))
                pad = w.pad_gf(g)[idx]
                bi, bj = np.nonzero(sep < pad[None] - TOL)
                worst = {}
                for i, k in zip(bi, bj):
                    key = (q.volume[i], fx.volume[idx[k]])
                    worst[key] = max(worst.get(key, 0.0), float(pad[k] - sep[i, k]))
                out += [dict(a=a, b=b, kind="sector-envelope", depth_mm=round(v, 2)) for (a, b), v in worst.items()]
        # sector pairs: facing pairs with the clearance, all others no worse than stock
        sep = _box_sep(parts.lo, parts.hi, parts.lo, parts.hi)
        worst = {}
        for i, k in zip(*np.nonzero(sep < self.cfg.clearance)):
            g, h = parts.group[i], parts.group[k]
            if g >= h:
                continue
            need = w.pad_gg(g, h) if (w.stock.faces[g, h] or w.stock.faces[h, g]) else min(0.0, w.stock.gsep[g, h])
            if sep[i, k] < need - TOL:
                key = (parts.volume[i], parts.volume[k])
                worst[key] = max(worst.get(key, 0.0), float(need - sep[i, k]))
        out += [dict(a=a, b=b, kind="sector-sector", depth_mm=round(v, 2)) for (a, b), v in worst.items()]
        out += self.sheet_violations(sheets, parts)
        return out, sheets

    def febs(self, sheets):
        """Sectors whose FEB boxes overlap anything (fixed, sheets, other sectors' parts or FEBs)."""
        w = self.w
        allp = self.w.parts.moved(self.translation())
        f = allp.take(allp.feb)
        if not len(f):
            return []
        bad = set()
        fx = w.env.fixed
        m = _near(f.lo, f.hi, fx.lo, fx.hi, 50.0)
        if m.any():
            sep = E.separation(f.lo, f.hi, fx.take(np.flatnonzero(m)))
            bad |= {w.names[g] for g in f.group[np.any(sep < self.cfg.clearance - TOL, axis=1)]}
        slo = np.array([s.lo for s in sheets]).reshape(-1, 3)
        shi = np.array([s.hi for s in sheets]).reshape(-1, 3)
        if len(slo):
            sep = _box_sep(f.lo, f.hi, slo, shi)
            bad |= {w.names[g] for g in f.group[np.any(sep < self.cfg.clearance - TOL, axis=1)]}
        sep = _box_sep(f.lo, f.hi, allp.lo, allp.hi)
        other = f.group[:, None] != allp.group[None]
        bad |= {w.names[g] for g in f.group[np.any((sep < 0.0 - TOL) & other, axis=1)]}
        return sorted(bad)


# ------------------------------------------------------------------ results
@dataclass
class Placed:
    """One placement of a design: feasible or not, what moved, and the counters text to evaluate it on."""
    feasible: bool
    placement: L.Placement
    stagger_mm: float
    sheets: dict
    violations: list
    remedy: dict | None
    text: str
    layout: L.LayoutConfig
    iterations: int = 0
    unsolved: list = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def shield_modified(self):
        return any(s["thickness_mm"] != s["stock_thickness_mm"] or s["trimmed_mm"] for s in self.sheets.values())

    @property
    def label(self):
        return "base" if self.remedy is None else self.remedy["name"]

    def summary(self):
        moved = [abs(s["moved_mm"]) for s in self.sheets.values()]
        return dict(
            feasible=self.feasible, variant=self.label, remedy=self.remedy, stagger_mm=self.stagger_mm,
            **self.placement.as_dict(), sheets=self.sheets, n_violations=len(self.violations),
            violations=self.violations[:50], iterations=self.iterations, unsolved=self.unsolved,
            shield_modified=self.shield_modified, deadtime_shield_uncalibrated=self.shield_modified,
            sheet_move_max_mm=max(moved) if moved else 0.0,
            sheet_min_thickness_mm=min([s["thickness_mm"] for s in self.sheets.values()] or [0.0]),
            shift_max_abs_mm=max([abs(v) for v in self.placement.shifts.values()] or [0.0]), wall_s=self.wall_s)

    def shield_overlay(self):
        """Geometry-file lines overriding the moved, thinned or trimmed sheets (SimpleConfig: the last definition
        wins), for the stock geometry of a reference."""
        lines = ["// crvfastsim.design.placement sheets: " + json.dumps(
            {k: dict(moved_mm=v["moved_mm"], thickness_mm=v["thickness_mm"], trimmed_mm=v["trimmed_mm"])
             for k, v in self.sheets.items()})]
        for v in self.sheets.values():
            if v["moved_mm"] == 0 and v["thickness_mm"] == v["stock_thickness_mm"] and not v["trimmed_mm"]:
                continue
            t = v["type"]
            lines.append(f"vector<double> ExtShieldUpstream.dimsType{t} = {{" +
                         ", ".join(f"{x:.10g}" for x in v["dims"]) + "};")
            lines.append(f"vector<double> ExtShieldUpstream.centerType{t}Box1 = {{" +
                         ", ".join(f"{x:.10g}" for x in v["center"]) + "};")
        return lines


def _sheet_report(world, sheets):
    out = {}
    for j, (s, st) in enumerate(zip(world.env.sheets, sheets)):
        R = E.orientation_matrix(s.orientation)
        world_half = 0.5 * (st.hi - st.lo)
        loc = np.abs(R.T) @ world_half                       # back to Offline (u, v, w)
        out[s.name] = dict(label=s.label, type=s.type, axis="xyz"[s.axis], moved_mm=round(st.moved, 3),
                           thickness_mm=round(float(st.hi[s.axis] - st.lo[s.axis]), 3),
                           stock_thickness_mm=round(float(2 * s.half[s.axis]), 3),
                           trimmed_mm={k: round(v, 3) for k, v in st.trimmed.items()},
                           center=[round(float(x), 3) for x in 0.5 * (st.lo + st.hi)],
                           dims=[round(float(x), 3) for x in 2 * loc], side=world.stock.side[j][0],
                           crv_side=[world.names[g] for g in world.stock.side[j][1]])
    return out


def _attempt(design, base_text, env, cfg, stock, stagger, moves=None, thickness=None, remedy=None, worlds=None):
    t0 = time.perf_counter()
    key = float(stagger)
    if worlds is not None and key in worlds:
        w = worlds[key]
    else:
        w = _World(design, base_text, env, cfg, stagger, stock)
        if worlds is not None:
            worlds[key] = w
    sv = _Solver(w, moves, thickness).run()
    viol, sheets = sv.violations()
    febs = sv.febs(sheets)
    if cfg.drop_febs:
        sv.no_feb = set(febs)
    else:
        viol += [dict(a=f"{n}:FEB", b="", kind="feb", depth_mm=0.0) for n in febs]
    pl = L.Placement(shifts={n: round(float(s), 3) for n, s in zip(w.names, sv.shift) if abs(s) > 1e-9},
                     moves={k: [float(x) for x in v] for k, v in (moves or {}).items()}, no_feb=tuple(sorted(sv.no_feb)))
    text = L.counters_text(design, pl, base_text, w.lcfg)
    return Placed(feasible=not viol and not sv.unsolved, placement=pl, stagger_mm=float(stagger),
                  sheets=_sheet_report(w, sheets), violations=sorted(viol, key=lambda v: -v["depth_mm"]),
                  remedy=remedy, text=text, layout=w.lcfg, iterations=sv.iterations, unsolved=sv.unsolved,
                  wall_s=time.perf_counter() - t0)


def _layout_conflicts(p, world):
    """Violations a joint stagger can cause: with the floor, or between sectors that do not face each other (the
    layer overhangs reach sideways), as opposed to the placement conflicts shifts and remedies resolve."""
    f = world.stock.faces
    idx = {n: i for i, n in enumerate(world.names)}

    def group(v):
        base = v.split(":")[0]
        for n in sorted(world.names, key=len, reverse=True):
            if base.startswith(n):
                return idx[n]
        return None
    out = []
    for v in p.violations:
        if "floorSu" in (v["a"], v["b"]):
            out.append(v)
        elif v["kind"] == "sector-sector":
            g, h = group(v["a"]), group(v["b"])
            if g is not None and h is not None and not (f[g, h] or f[h, g]):
                out.append(v)
    return out


def _base(design, base_text, env, cfg, stock, cache):
    """The base placement at the largest joint stagger that places, else (and for the remedies) at the largest one
    without layout conflicts (_layout_conflicts), which remedies do not resolve."""
    stags = sorted(set(cfg.stagger_mm), reverse=True)
    tried = {}
    for st in stags:
        tried[st] = _attempt(design, base_text, env, cfg, stock, st, worlds=cache)
        if tried[st].feasible:
            return tried[st], st
    use = next((st for st in stags if not _layout_conflicts(tried[st], cache[st])), min(stags))
    return tried[use], use


def _with_remedy(design, base_text, env, cfg, stock, cache, r, stagger):
    """The smallest amount of remedy r that places (coarse steps, then bisection to 0.5 mm); for a move with
    thinning, the least thinning over the move steps, then the smallest move."""
    js = [_sheet_index(env, x) for x in r.sheets]
    t0s = {j: 2 * env.sheets[j].half[env.sheets[j].axis] for j in js}

    def at(m, x):
        kw = {}
        if r.sectors:
            v = np.asarray(r.direction, float)
            kw["moves"] = {sct: (v / np.linalg.norm(v) * m).tolist() for sct in r.sectors}
        if js:
            kw["thickness"] = {j: max(t0s[j] - x, r.min_thickness) for j in js}
        rem = dict(name=r.name, kind=r.kind, amount_mm=round(float(m if r.sectors else x), 3))
        if r.sectors:
            rem.update(sectors=list(r.sectors), direction=list(r.direction), move_mm=round(float(m), 3))
        if js:
            rem.update(sheets=list(r.sheets), thin_mm=round(float(x), 3),
                       thickness_mm={env.sheets[j].name: round(float(kw["thickness"][j]), 3) for j in js})
        return _attempt(design, base_text, env, cfg, stock, stagger, worlds=cache, remedy=rem, **kw)

    def smallest(f, top):
        """Smallest x in (0, top] with f(x) feasible: (Placed, found)."""
        prev, x, p = 0.0, 0.0, None
        while True:
            x = min(x + r.step, top)
            p = f(x)
            if p.feasible:
                break
            if x >= top - TOL:
                return p, False
            prev = x
        lo, hi, best = prev, x, p
        while hi - lo > 0.5:
            mid = 0.5 * (lo + hi)
            q = f(mid)
            if q.feasible:
                hi, best = mid, q
            else:
                lo = mid
        return best, True

    thin_top = max([t0s[j] - r.min_thickness for j in js] or [0.0])
    if not js:
        return smallest(lambda m: at(m, 0.0), r.max_mm)[0]
    if not r.sectors:
        return smallest(lambda x: at(0.0, x), thin_top)[0] if thin_top > 0 else at(0.0, 0.0)
    best, last = None, None
    m = 0.0
    while True:
        m = min(m + r.step, r.max_mm)
        last = at(m, thin_top)
        if last.feasible:
            p, _ = smallest(lambda x: at(m, x), thin_top)
            if best is None or p.remedy["thin_mm"] < best.remedy["thin_mm"] - 0.25:
                best = p
            if p.remedy["thin_mm"] <= 0.5:
                break
        if m >= r.max_mm - TOL:
            break
    return best or last


def place(design, counters=M.RUN2_COUNTERS, cfg=PlacementConfig(), env=None):
    """Placements of a design (list of Placed; module docstring): the base placement first, then, when it does not
    place, every remedy (policy "each") or the remedies up to the first that places (policy "first")."""
    base_text = L.read_text(counters)
    c0 = G.parse_simpleconfig(base_text)
    env = env or E.Envelope.load(c0, cfg.upstream, cfg.downstream, cfg.hall, cfg.movable_material)
    stock = _World(M.Design(), base_text, env, cfg, 0.0)
    cache = {}
    base, stagger = _base(design, base_text, env, cfg, stock, cache)
    out = [base]
    if base.feasible:
        return out
    for r in cfg.remedies:
        p = _with_remedy(design, base_text, env, cfg, stock, cache, r, stagger)
        out.append(p)
        if p.feasible and cfg.policy == "first":
            break
    return out


def check(design, placement, counters=M.RUN2_COUNTERS, cfg=PlacementConfig(), stagger_mm=0.0, thickness=None,
          env=None, solve=False):
    """Violations of a given placement (shifts, moves, no_feb; sheets placed by the push rule, thickness {sheet
    selector: mm}): the check alone, no solving. Used to test hand-made placements against the envelope."""
    base_text = L.read_text(counters)
    c0 = G.parse_simpleconfig(base_text)
    env = env or E.Envelope.load(c0, cfg.upstream, cfg.downstream, cfg.hall, cfg.movable_material)
    stock = _World(M.Design(), base_text, env, cfg, 0.0)
    w = _World(design, base_text, env, cfg, stagger_mm, stock)
    th = {_sheet_index(env, k): float(v) for k, v in (thickness or {}).items()}
    sv = _Solver(w, placement.moves, th)
    for n, s in placement.shifts.items():
        sv.shift[w.names.index(n)] = float(s)
    sv.no_feb = set(placement.no_feb)
    viol, sheets = sv.violations()
    febs = sv.febs(sheets)
    return dict(violations=sorted(viol, key=lambda v: -v["depth_mm"]), sheets=_sheet_report(w, sheets),
                feb_conflicts=[n for n in febs if n not in placement.no_feb])


def parse_remedies(text):
    """Remedies from a compact string (kit parameter): ';'-separated items, each
    'move:<name>:<S1>,<S2>,...:<dx>,<dy>,<dz>:<max_mm>', 'thin:<name>:<sheet>|<sheet>...:<min_thickness_mm>' or
    'movethin:<name>:<S1>,...:<dx>,<dy>,<dz>:<max_mm>:<sheet>|...:<min_thickness_mm>'."""
    out = []
    for item in [x.strip() for x in (text or "").split(";") if x.strip()]:
        f = item.split(":")
        vec = lambda t: tuple(float(x) for x in t.split(","))       # noqa: E731
        secs = lambda t: tuple(x for x in t.split(",") if x)        # noqa: E731
        shs = lambda t: tuple(x.strip() for x in t.split("|") if x.strip())   # noqa: E731
        if f[0] == "move" and len(f) == 5:
            out.append(Remedy(f[1], sectors=secs(f[2]), direction=vec(f[3]), max_mm=float(f[4])))
        elif f[0] == "thin" and len(f) == 4:
            out.append(Remedy(f[1], sheets=shs(f[2]), min_thickness=float(f[3])))
        elif f[0] == "movethin" and len(f) == 7:
            out.append(Remedy(f[1], sectors=secs(f[2]), direction=vec(f[3]), max_mm=float(f[4]), sheets=shs(f[5]),
                              min_thickness=float(f[6])))
        else:
            raise ValueError(f"remedy {item!r}: expected move:name:S1,S2:dx,dy,dz:max, thin:name:sheet|sheet:min or "
                             "movethin:name:S1,S2:dx,dy,dz:max:sheet|sheet:min")
    return tuple(out)
