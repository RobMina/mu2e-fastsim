"""Design layouts that can be placed in a detector envelope: the counters text of a design and its geometry.

module.build re-tiles every sector with ceil(stock extent / design pitch) modules, which overshoots into the next
sector; it is the default of fom.evaluate and is unchanged. A placed design (placement.py) instead uses this layout,
the rules of the G1 option-B stock references (user 2026-11-20/21), with every constant read from the counters file:

  tiling    per sector, the even counter count per module (within +-``max_counter_delta`` of stock) and module count
            that cover the stock extent along the gap direction most closely without exceeding it, chosen together
            with the end-strip filler (below); ties go to the count closest to stock;
  close-up  sectors that follow each other along a common gap direction in the stock geometry (same plane and
            directions, one starting where the other's stock extent ends: T1..T5, R1..R6, ...) are closed up: each
            starts where its predecessor's design modules (and filler) end, so the uncovered end strip of every
            sector (stock extent minus cover, < one filler pitch, the same in every layer) does not open a crack
            through the stack at the boundary; the strips add up at the chain's last end (``close_up``);
  fillers   each end strip that holds a module of >= 2 design counters gets a one-module sector ``<parent>F`` after
            the parent's last module (largest even count that fits; Offline's layer box assumes an even count): the
            parent's directions, length, SiPM sides, joint stagger and cuts, no FEB boxes; appended after the stock
            sectors, so the stock sector indices are unchanged;
  offset    crs.layerOffset = min(offset_stock x w / w_stock, offset_stock x (N_stock - 1) / (N - 1)): the stock
            fraction of a counter width, never more total stagger than stock;
  segments  off by default (``split``): every sector keeps its stock counter length, as nominal Run 2, and
            Design.max_length is ignored by the layout. With split=True a sector longer than Design.max_length is
            split along its length; segments read at both ends need
            their counter motherboards at the joint, so they are ``segment_gap`` apart (default 2 x (CMBOffset +
            CMBHalfThickness) + 2 mm = 42 mm on Run 2) and the sector grows by the gaps;
  stagger   in a segmented sector, layer j's joints move by j x ``stagger_mm`` along the length (offsetDirection
            gets a length component; CosmicRayShieldMaker and module.build both add layerOffset x offsetDirection
            per layer), the overhang going to the free end (stagger_anchor);
  FEBs      FEB width and two-FEB spacing scaled so the FEBs of neighbouring design modules do not touch;
            ``no_feb`` sectors drop their FEB boxes;
  shifts    per sector, firstCounter -= shift x layerDirection (a positive shift moves the stack against its layer
            direction); ``moves`` add a world translation [dx, dy, dz] after the shift. Both are what placement.py
            solves for.
The same counters text drives this fast model and the stock geometry of a reference (gen_geom-style overlays), so
the bars are the same in both.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from .. import geometry as G
from . import module as M

FILLER = "F"                    # filler sector name suffix: <parent>F


@dataclass(frozen=True)
class LayoutConfig:
    """The layout rules of a placed design (module docstring). Defaults: the G1 option-B v3 references."""
    segment_gap: float | None = None   # mm between length segments; None = 2 x (CMBOffset + CMBHalfThickness) + 2
    stagger_mm: float = 0.0            # joint stagger per layer [mm] in segmented sectors
    fillers: bool = True               # end-strip filler sectors
    max_counter_delta: int = 6         # counters per module within +- this of stock
    feb_margin: float = 2.0            # mm between the FEBs of neighbouring modules
    split: bool = False                # split sectors longer than Design.max_length into length segments (joints
                                       # the nominal geometry does not have); False = stock counter lengths
    close_up: bool = True              # close up stock-adjacent sectors (module docstring); False = each sector at
                                       # its stock firstCounter, leaving its end strip open at the boundary

    def __post_init__(self):
        if self.segment_gap is not None and not self.segment_gap >= 0:
            raise ValueError("segment_gap must be >= 0")
        if not self.stagger_mm >= 0:
            raise ValueError("stagger_mm must be >= 0")


@dataclass
class Placement:
    """Per-sector shifts along -layerDirection [mm], world moves [mm], sectors without FEB boxes."""
    shifts: dict = field(default_factory=dict)
    moves: dict = field(default_factory=dict)
    no_feb: tuple = ()

    def as_dict(self):
        return dict(shifts={k: float(v) for k, v in self.shifts.items()},
                    moves={k: [float(x) for x in v] for k, v in self.moves.items()}, no_feb=list(self.no_feb))


def read_text(counters):
    """Counters text from a path (Offline-relative or absolute) or the text itself."""
    return counters if "crs.sectorNames" in counters else open(G.find_counters(counters)).read()


def segment_gap(c, cfg=LayoutConfig()):
    if cfg.segment_gap is not None:
        return float(cfg.segment_gap)
    return 2.0 * (float(c.get("crs.CMBOffset", 0.0)) + float(c.get("crs.CMBHalfThickness", 0.0))) + 2.0


def layer_offset(c, design):
    """crs.layerOffset of a design on the stock counters c (module docstring)."""
    off0, w0, n0 = float(c["crs.layerOffset"]), float(c["crs.scintillatorBarWidth"]), int(c["crs.nLayers"])
    w = design.width or w0
    n = design.n_layers or n0
    return round(min(off0 * w / w0, off0 * (n0 - 1) / max(n - 1, 1)), 3)


def module_pitch(c, ncnt, w):
    gs, gl, gm = float(c["crs.gapSmall"]), float(c["crs.gapLarge"]), float(c["crs.gapBetweenModules"])
    return ncnt * w + (ncnt // 2 - 1) * gl + (ncnt // 2) * gs + gm


def stock_extent(c, name):
    """The stock sector's extent along its gap direction (nModules x stock pitch, CosmicRayShieldMaker)."""
    return int(c[f"crs.nModules{name}"]) * module_pitch(c, int(c[f"crs.nCountersPerModule{name}"]),
                                                         float(c["crs.scintillatorBarWidth"]))


def filler_counters(c, strip, w):
    """Largest even counter count of a one-module filler that fits a strip (0 if none)."""
    kf = 0
    for kk in range(2, 64, 2):
        if module_pitch(c, kk, w) <= strip + 1e-6:
            kf = kk
    return kf


def tiling(c, name, w, joint, delta=6):
    """(counters per module, modules, covered / stock extent): the closest cover of the stock extent without
    overshoot; joint: maximise the main modules plus the end-strip filler together."""
    ext, n_s = stock_extent(c, name), int(c[f"crs.nCountersPerModule{name}"])
    best = None
    for k in range(max(2, n_s - delta), n_s + delta + 1, 2):
        n = int(math.floor(ext / module_pitch(c, k, w) + 1e-6))
        if n < 1:
            continue
        cov = n * module_pitch(c, k, w)
        if joint:
            kf = filler_counters(c, ext - cov, w)
            cov += module_pitch(c, kf, w) if kf else 0.0
        key = (round(cov, 3), -abs(k - n_s))
        if best is None or key > best[0]:
            best = (key, k, n, cov / ext)
    if best is None:
        raise ValueError(f"{name}: no module of the design fits the stock sector extent")
    return best[1], best[2], best[3]


def sector_chains(c, tol=5.0):
    """{sector: its predecessor} for stock sectors that follow each other along a common gap direction: same
    gapDirection, layerDirection and offsetDirection, firstCounter offset along the gap direction by the
    predecessor's stock extent (within tol mm) and not laterally."""
    names = list(c["crs.sectorNames"])
    vec = lambda k: np.asarray(c[k], float)                          # noqa: E731
    prev = {}
    for b in names:
        for a in names:
            if a == b or any(not np.allclose(vec(f"crs.{d}{a}"), vec(f"crs.{d}{b}"))
                             for d in ("gapDirection", "layerDirection", "offsetDirection")):
                continue
            g = vec(f"crs.gapDirection{a}")
            g = g / np.linalg.norm(g)
            dv = vec(f"crs.firstCounter{b}") - vec(f"crs.firstCounter{a}")
            along = float(dv @ g)
            if abs(along - stock_extent(c, a)) < tol and np.linalg.norm(dv - along * g) < tol:
                prev[b] = a
    return prev


def end_strip(c, name, w, cfg=LayoutConfig()):
    """The uncovered end strip of a sector [mm]: stock extent minus the design modules and the filler."""
    k, n, _ = tiling(c, name, w, cfg.fillers, cfg.max_counter_delta)
    cov = n * module_pitch(c, k, w)
    if cfg.fillers:
        kf = filler_counters(c, stock_extent(c, name) - cov, w)
        cov += module_pitch(c, kf, w) if kf >= 2 else 0.0
    return stock_extent(c, name) - cov


def close_up_offsets(c, w, cfg=LayoutConfig()):
    """{sector: mm} the sector moves against its gap direction to start where its chain predecessor ends."""
    prev = sector_chains(c)
    out = {}

    def off(nm):
        if nm not in out:
            p = prev.get(nm)
            out[nm] = 0.0 if p is None else off(p) + end_strip(c, p, w, cfg)
        return out[nm]
    for nm in c["crs.sectorNames"]:
        off(nm)
    return {k: v for k, v in out.items() if v > 1e-9}


def _length_axis(c, name):
    ldir = np.asarray(c[f"crs.layerDirection{name}"], float)
    gdir = np.asarray(c[f"crs.gapDirection{name}"], float)
    ell = np.zeros(3)
    ell[[d for d in range(3) if ldir[d] == 0 and gdir[d] == 0][0]] = 1.0
    return ell


def sector_boxes(dg, skins=None):
    """{sector: (lo, hi)} world box of a sector's bars and absorbers; skins (strongback, cover) [mm] grow it on the
    -layerDirection and + faces."""
    out = {}
    for s, nm in enumerate(dg.sector_names):
        A = dg.axes[s]
        b = np.flatnonzero(dg.sector == s)
        hw = np.abs(dg.halflen[b] @ A)
        lo, hi = (dg.center[b] - hw).min(0), (dg.center[b] + hw).max(0)
        a = np.flatnonzero(dg.abs_sector == s)
        if len(a):
            ha = np.abs(dg.abs_halflen[a] @ A)
            lo, hi = np.minimum(lo, (dg.abs_center[a] - ha).min(0)), np.maximum(hi, (dg.abs_center[a] + ha).max(0))
        if skins is not None:
            s_in, s_out = skins
            n = A[0]
            lo = lo + np.minimum(n, 0) * s_out - np.maximum(n, 0) * s_in
            hi = hi + np.maximum(n, 0) * s_out - np.minimum(n, 0) * s_in
        out[nm] = (lo, hi)
    return out


def stagger_anchor(nm, boxes0, ell, total, margin=50.0):
    """Fraction of a staggered stack's overhang moved to the -length end: 0.5 centres it when both ends are free,
    else the end facing another stock sector (lateral overlap, within total + margin) keeps its stock position."""
    a = int(np.argmax(np.abs(ell)))
    lat = [i for i in range(3) if i != a]
    lo, hi = boxes0[nm]
    free_p, free_m = 1e9, 1e9
    for other, (olo, ohi) in boxes0.items():
        if other == nm or not all(min(hi[i], ohi[i]) - max(lo[i], olo[i]) > -margin for i in lat):
            continue
        if olo[a] >= hi[a] - 1.0:
            free_p = min(free_p, olo[a] - hi[a])
        if ohi[a] <= lo[a] + 1.0:
            free_m = min(free_m, lo[a] - ohi[a])
    if free_p >= total / 2 + margin and free_m >= total / 2 + margin:
        return 0.5
    return 0.0 if free_p >= free_m else 1.0


def feb_scale(c, w, cfg=LayoutConfig()):
    """Factor (<= 1) on the FEB width and two-FEB spacing so the FEBs of neighbouring design modules do not touch
    (1 for a counters file without FEBs)."""
    if "crs.FEBHalfLengths" not in c:
        return 1.0
    hl = np.atleast_1d(c["crs.FEBHalfLengths"]).astype(float)
    feb_w, feb_sep = 2 * hl[1], float(c["crs.FEBDistanceBetween2FEBsW"])
    f = 1.0
    for nm in c["crs.sectorNames"]:
        n_side = max(int(c.get(f"crs.FEBBoxesAtSide0{nm}", 0)), int(c.get(f"crs.FEBBoxesAtSide1{nm}", 0)))
        if n_side == 0:
            continue
        k, _, _ = tiling(c, nm, w, cfg.fillers, cfg.max_counter_delta)
        foot = feb_sep + feb_w if n_side == 2 else feb_w
        f = min(f, (module_pitch(c, k, w) - cfg.feb_margin) / foot)
    return round(f, 4)


class _Text:
    """Edits one declaration at a time in a SimpleConfig text."""

    def __init__(self, text):
        self.text = text

    def set(self, key, value, kind):
        if kind == "vec":
            pat = r"(vector<double>\s+%s\s*=\s*)\{[^}]*\}" % re.escape(key)
            rep = "{" + ", ".join(f"{float(x):.10g}" for x in value) + "}"
        else:
            pat = r"((?:double|int)\s+%s\s*=\s*)[-0-9.eE+]+" % re.escape(key)
            rep = f"{float(value):.10g}" if kind == "dbl" else str(int(value))
        self.text, n = re.subn(pat, lambda m: m.group(1) + rep, self.text)
        if n != 1:
            raise KeyError(f"{key} not found once in the counters text")

    def get(self, key):
        return G.parse_simpleconfig(self.text)[key]


def counters_text(design, placement=None, base=M.RUN2_COUNTERS, cfg=LayoutConfig()):
    """The counters text of a placed design (module docstring): stock text with layerOffset, shifts and moves,
    lengthened segmented sectors, joint stagger, FEB scaling, no_feb sectors and filler sectors."""
    placement = placement or Placement()
    base_text = read_text(base)
    c = G.parse_simpleconfig(base_text)
    names = list(c["crs.sectorNames"])
    bad = sorted(set(placement.shifts) | set(placement.moves) | set(placement.no_feb))
    bad = [b for b in bad if b not in names]
    if bad:
        raise KeyError(f"placement names unknown sectors {bad}")
    t = _Text(base_text)
    w = design.width or float(c["crs.scintillatorBarWidth"])
    off = layer_offset(c, design)
    t.set("crs.layerOffset", off, "dbl")
    for sec, s in placement.shifts.items():
        ldir = np.asarray(c[f"crs.layerDirection{sec}"], float)
        t.set(f"crs.firstCounter{sec}", np.asarray(t.get(f"crs.firstCounter{sec}"), float) - float(s) * ldir, "vec")
    for sec, v in placement.moves.items():
        t.set(f"crs.firstCounter{sec}", np.asarray(t.get(f"crs.firstCounter{sec}"), float) + np.asarray(v, float),
              "vec")
    closed = close_up_offsets(c, w, cfg) if cfg.close_up else {}
    for sec, dz in closed.items():
        g = np.asarray(c[f"crs.gapDirection{sec}"], float)
        t.set(f"crs.firstCounter{sec}", np.asarray(t.get(f"crs.firstCounter{sec}"), float) - dz * g / np.linalg.norm(g),
              "vec")
    gap = segment_gap(c, cfg)
    mx = float(design.max_length or 0.0) if cfg.split else 0.0
    lengthened = {}
    if mx > 0:
        for nm in names:
            L = float(c[f"crs.scintillatorBarLength{nm}"])
            n = max(1, math.ceil(L / mx - 1e-9))
            if n > 1:
                L2 = L + (n - 1) * gap
                t.set(f"crs.scintillatorBarLength{nm}", L2, "dbl")
                lengthened[nm] = L2
    staggered = {}
    if cfg.stagger_mm > 0 and lengthened and not off > 0:
        raise ValueError("a joint stagger needs crs.layerOffset > 0: it is a length component of offsetDirection")
    if cfg.stagger_mm > 0 and lengthened:
        nl = int(design.n_layers or c["crs.nLayers"])
        boxes0 = sector_boxes(M.build(M.Design(), base_text))
        for nm in lengthened:
            ell = _length_axis(c, nm)
            odir = np.asarray(c[f"crs.offsetDirection{nm}"], float) + (cfg.stagger_mm / off) * ell
            t.set(f"crs.offsetDirection{nm}", odir, "vec")
            frac = stagger_anchor(nm, boxes0, ell, (nl - 1) * cfg.stagger_mm)
            fc = np.asarray(t.get(f"crs.firstCounter{nm}"), float) - frac * (nl - 1) * cfg.stagger_mm * ell
            t.set(f"crs.firstCounter{nm}", fc, "vec")
            staggered[nm] = frac
    f = feb_scale(c, w, cfg)
    if "crs.FEBHalfLengths" in c:
        hl = list(np.atleast_1d(c["crs.FEBHalfLengths"]).astype(float))
        t.set("crs.FEBHalfLengths", [hl[0], hl[1] * f, hl[2]], "vec")
        t.set("crs.FEBDistanceBetween2FEBsW", float(c["crs.FEBDistanceBetween2FEBsW"]) * f, "dbl")
    for sec in placement.no_feb:
        t.set(f"crs.FEBBoxesAtSide0{sec}", 0, "int")
        t.set(f"crs.FEBBoxesAtSide1{sec}", 0, "int")
    text = t.text
    fillers = {}
    if cfg.fillers:
        for nm in names:
            k, n, _ = tiling(c, nm, w, True, cfg.max_counter_delta)
            kf = filler_counters(c, stock_extent(c, nm) - n * module_pitch(c, k, w), w)
            if kf >= 2:
                fillers[nm + FILLER] = (kf, nm, n * module_pitch(c, k, w))
        text += "\nbool crs.layoutJointTiling = true;\n"
        if fillers:
            text = _write_fillers(text, names, fillers)
    meta = dict(layer_offset=off, segment_gap=gap, close_up={k: round(v, 3) for k, v in closed.items()},
                lengthened=lengthened, stagger_mm=cfg.stagger_mm,
                stagger_anchor=staggered, feb_scale=f, fillers={k: v[0] for k, v in fillers.items()},
                **placement.as_dict())
    return f"// crvfastsim.design.layout: {json.dumps(meta)}\n" + text


def _write_fillers(text, names, fillers):
    """Append the filler sectors: every per-sector key of the parent under the filler's name, with nModules 1, its
    counter count, firstCounter moved along gapDirection, no FEB boxes and the parent as preceding sector."""
    cur = G.parse_simpleconfig(text)
    stems = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\S+(?:\s*<\s*\w+\s*>)?)\s+crs\.(\w+?)\s*=", line)
        if not m:
            continue
        key = m.group(2)
        for nm in sorted(names, key=len, reverse=True):
            if key.endswith(nm) and len(key) > len(nm):
                stems.setdefault(key[:-len(nm)], set()).add(nm)
                break
    stems = {k for k, v in stems.items() if len(v) >= len(names) // 2}
    add = ["", "// crvfastsim.design.layout filler sectors: one module in the end strip of the parent's tiling"]
    for f, (kf, par, shift) in fillers.items():
        for st in sorted(stems):
            key = f"crs.{st}{par}"
            if key not in cur:
                continue
            v = cur[key]
            if st == "nModules":
                v = 1
            elif st == "nCountersPerModule":
                v = kf
            elif st == "firstCounter":
                v = list(np.asarray(v, float) + shift * np.asarray(cur[f"crs.gapDirection{par}"], float))
            elif st.startswith("FEBBoxesAtSide"):
                v = 0
            elif st == "precedingSectorFor":
                v = names.index(par)
            nk = f"crs.{st}{f}"
            if isinstance(v, bool):
                add.append(f"bool {nk} = {'true' if v else 'false'};")
            elif isinstance(v, (list, tuple, np.ndarray)):
                add.append(f"vector<double> {nk} = {{" + ", ".join(f"{x:.10g}" for x in v) + "};")
            elif isinstance(v, str):
                add.append(f'string {nk} = "{v}";')
            elif isinstance(v, int) or st in ("nModules", "nCountersPerModule", "nLayers", "sectorType",
                                              "FEBBoxesAtSide0", "FEBBoxesAtSide1", "precedingSectorFor"):
                add.append(f"int {nk} = {int(v)};")
            else:
                add.append(f"double {nk} = {float(v):.10g};")
    allnames = names + list(fillers)
    text, n1 = re.subn(r"(vector<string>\s+crs\.sectorNames\s*=\s*)\{[^}]*\}",
                       lambda m: m.group(1) + "{" + ", ".join(f'"{x}"' for x in allnames) + "}", text)
    text, n2 = re.subn(r"(int\s+crs\.nSectors\s*=\s*)\d+", lambda m: m.group(1) + str(len(allnames)), text)
    if n1 != 1:
        raise KeyError("crs.sectorNames not found once in the counters text")
    if n2 == 0:
        text += f"\nint crs.nSectors = {len(allnames)};\n"
    return text + "\n".join(add) + "\n"


def parent_of(name, stock_names):
    """The stock sector a (possibly filler) sector belongs to."""
    return name[:-len(FILLER)] if name.endswith(FILLER) and name[:-len(FILLER)] in stock_names else name


def build(design, text, sector_yields=None, cfg=LayoutConfig()):
    """DesignGeometry of a design on a layout counters text (counters_text): module.build with the no-overshoot
    tiling per stock sector and one module per filler sector. sector_yields: {stock sector: yield}; fillers take
    their parent's. With cfg.split False (default) the design's max_length is ignored: stock counter lengths."""
    if not cfg.split:
        design = dataclasses.replace(design, max_length=0.0)
    c = G.parse_simpleconfig(text)
    names = list(c["crs.sectorNames"])
    stock = [n for n in names if not (n.endswith(FILLER) and n[:-len(FILLER)] in names)]
    if sector_yields is not None:
        sector_yields = {n: sector_yields[parent_of(n, stock)] for n in names}
    w = design.width or float(c["crs.scintillatorBarWidth"])
    joint = bool(c.get("crs.layoutJointTiling", False))
    tile = {nm: ((int(c[f"crs.nCountersPerModule{nm}"]), 1, 1.0) if nm not in stock else
                 tiling(c, nm, w, joint, cfg.max_counter_delta)) for nm in names}
    builds = {k: M.build(dataclasses.replace(design, counters_per_module=k, segment_gap=segment_gap(c, cfg)), text,
                         sector_yields) for k in sorted({v[0] for v in tile.values()})}
    first = builds[next(iter(builds))]
    parts = {k: [] for k in ("sector", "module", "layer", "counter", "segment", "center", "halflen")}
    aparts = {k: [] for k in ("abs_sector", "abs_center", "abs_halflen")}
    axes, nl, fib, fib0, pitch = [], [], [], [], []
    for s, nm in enumerate(names):
        k, nkeep, _ = tile[nm]
        dg = builds[k]
        b = (dg.sector == s) & (dg.module < nkeep)
        for key in parts:
            parts[key].append(getattr(dg, key)[b])
        a = np.flatnonzero(dg.abs_sector == s)                # absorbers: appended module by module
        if len(a):
            per = len(a) // (int(dg.module[dg.sector == s].max()) + 1)
            a = a[:nkeep * per]
        for key in aparts:
            aparts[key].append(getattr(dg, key)[a])
        axes.append(dg.axes[s]); nl.append(dg.n_layers[s]); fib.append(dg.fiber_diameter[s])
        fib0.append(dg.stock_fiber[s]); pitch.append(dg.module_pitch[s])
    cat = {k: np.concatenate(v) for k, v in {**parts, **aparts}.items()}
    return dataclasses.replace(first, axes=np.array(axes), n_layers=np.array(nl), fiber_diameter=np.array(fib),
                               stock_fiber=np.array(fib0), module_pitch=np.array(pitch), design=design, **cat)
