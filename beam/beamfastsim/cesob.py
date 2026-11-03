"""Fast stand-in for the approximate conversion-electron sensitivity S/sqrt(B) of the stopping-target design studies.

Chain per design point (opt-in, Geant4-free):

1. target stops of the beam stage (``beamfastsim point --stops-out``) and their rate per generated event
   (stop fraction x muons per event of the start table);
2. conversion electrons (104.97 MeV/c, isotropic) from the mu- stops through foils, IPA (with end rings), OPA and the
   DS field to the tracker entry plane (beamfastsim.cechain);
3. calorimeter acceptance: the helix from the tracker entry plane in the DS field runs at least 17.8 mm inside the
   crystals of the two disks (the CsI path giving 10 MeV at minimum ionisation; the stand-in for a visible
   calorimeter energy above 10 MeV in the event); tracker and support material are not modelled;
4. the sensitivity estimate on the selected electrons, the approximate Run-1A estimate of the design studies: signal =
   energy at the tracker entry smeared by a 0.2 MeV Gaussian, normalised to stops/POT x accepted per generated electron;
   DIO = the Heeck/Szafron spectrum (Offline ConditionsService table) smeared by the measured entry-minus-birth energy
   response and the same resolution; cosmics flat in energy over the on-spill time; the window (0.2 MeV bins, lower
   edge >= 50 MeV) that maximises S/sqrt(B).

The same estimate runs on a Geant4 EdepAna tree (``reduce``, ``refsob``) for validation against full-simulation points.

  python3 -m beamfastsim.cesob point --g4 TABLE --geom G --muons N --electrons M --seed S --procs P --out DIR
  python3 -m beamfastsim.cesob ce --stops STOPS.npz --geom G --electrons M --seed S --procs P --out CE.npz
  python3 -m beamfastsim.cesob sob --ce CE.npz --stops-per-gen R --out metrics.json
  python3 -m beamfastsim.cesob reduce --nts F.root [...] --out REF.npz        (Geant4 EdepAna tree; needs uproot)
  python3 -m beamfastsim.cesob refsob --ref REF.npz --stops-per-gen R --out metrics.json
  python3 -m beamfastsim.cesob hybrid --ref-dir REFS --fast-dir RUNS --g4-stops points.txt --out hybrid.json
  python3 -m beamfastsim.cesob calocheck --ref CEREF.npz --nts EDEP.root [...] --out STEM   (proxy vs Geant4)
Paths: BEAMFS_OFFLINE_ROOT, BEAMFS_MAPS as for beamfastsim; the DIO table from the Offline root (--dio to override).
"""
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from mu2efastsim import ehelix, eioni

from . import cechain

# Assumptions of the sensitivity estimate (as the design studies use it)
CAPTURE = 0.609                        # muon captures per stop on Al
R_MUE = 1.0e-13                        # conversions per capture
BR = R_MUE * CAPTURE                   # conversions per stop
NPOT = 1.0e18
POT_PER_EVENT = 1.6e7
ONSPILL_S_PER_EVENT = 1.695e-6
SIGMA_MEV = 0.2                        # tracker resolution
BOX_MIN_MEV = 50.0
COSMIC_PER_S_PER_MEV = 1.0 / 550.0     # the foils studies' value
UPSTREAM_EFF = 0.01278168              # generated beam events per POT of the foils studies' MuBeam sample
CALO_EDEP_VIS_MEV = 10.0               # Geant4 selection the calorimeter acceptance stands in for
FRONT_BINS = (1500, 0.0, 150.0)        # tracker-entry energy [MeV]
RESP_BINS = (500, -100.0, 0.0)         # entry minus birth energy [MeV]
DIO_BINS = (11000, 0.0, 110.0)
DIO_TABLE = "Offline/ConditionsService/data/heeck_finer_binning_2016_szafron.tbl"
E_START = math.sqrt(cechain.P_CE_AL ** 2 + eioni.ME ** 2)


# --- the sensitivity estimate ----------------------------------------------------------------------------------------

def gaussian_kernel(width, sigma, n_sigma=5.0):
    """(mass per offset bin, first offset) of a Gaussian integrated over bins of `width`, renormalised."""
    half = max(1, int(math.ceil(n_sigma * sigma / width)))
    edges = (np.arange(-half, half + 2) - 0.5) * width
    cdf = np.array([0.5 * math.erf(e / (sigma * math.sqrt(2.0))) for e in edges])
    mass = np.diff(cdf)
    return mass / mass.sum(), -half


def density_kernel(values, xmin, w_in, width):
    """A response density on bins (xmin, w_in) resampled onto offsets of `width`, keeping its total, trimmed."""
    xmax = xmin + values.size * w_in
    lo, hi = int(math.floor(xmin / width)), int(math.ceil(xmax / width))
    offsets = np.arange(lo, hi + 1) * width
    centers = xmin + (np.arange(values.size) + 0.5) * w_in
    mass = np.interp(offsets, centers, values, left=0.0, right=0.0) * width
    total = values.sum() * w_in
    if mass.sum() > 0.0:
        mass = mass * (total / mass.sum())
    filled = np.flatnonzero(mass)
    if filled.size:
        lo += int(filled[0])
        mass = mass[filled[0]:filled[-1] + 1]
    return mass, lo


def smear(values, kernel):
    """Each bin's content moved by the kernel's offsets; content leaving the axis is dropped."""
    mass, first = kernel
    full = np.convolve(values, mass)
    out = np.zeros(values.size)
    lo, hi = max(0, first), min(values.size, first + full.size)
    if hi > lo:
        out[lo:hi] = full[lo - first:hi - first]
    return out


def regrid(values, xmin, width, n, xmin2, width2):
    """Contents on (xmin, width) summed into n bins of (xmin2, width2) by bin centre."""
    idx = np.floor((xmin + (np.arange(values.size) + 0.5) * width - xmin2) / width2).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    return np.bincount(idx[ok], weights=values[ok], minlength=n)


def dio_density(path):
    """DIO spectrum as a probability density on DIO_BINS (each tabulated energy sets the bin it ends)."""
    data = np.loadtxt(path)
    n, lo, hi = DIO_BINS
    w = (hi - lo) / n
    v = np.zeros(n)
    idx = np.floor((data[:, 0] - w / 2.0 - lo) / w).astype(np.int64)
    ok = (idx >= 0) & (idx < n)
    v[idx[ok]] = data[ok, 1]
    return v / (v.sum() * w)


def scan_box(sig, dio, cos, centers, box_min=BOX_MIN_MEV):
    """Window [centers[i], centers[j]] maximising S/sqrt(B); sums run from each lower edge (no prefix differences)."""
    best = None
    for i in range(int(np.searchsorted(centers, box_min)), sig.size):
        s, d, c = np.cumsum(sig[i:]), np.cumsum(dio[i:]), np.cumsum(cos[i:])
        b = d + c
        ok = np.flatnonzero((s > 0.0) & (b > 0.0))
        if ok.size == 0:
            continue
        j = ok[int(np.argmax(s[ok] / np.sqrt(b[ok])))]
        sens = s[j] / np.sqrt(b[j])
        if best is None or sens > best["sensitivity"]:
            best = dict(sensitivity=float(sens), signal_box_low_mev=float(centers[i]),
                        signal_box_high_mev=float(centers[i + j]), signal_rate=float(s[j]),
                        dio_background=float(d[j]), cosmic_background=float(c[j]), total_background=float(b[j]))
    if best is None:
        raise ValueError("no window with signal and background above the box minimum")
    return best


def sensitivity(front, front_entries, resp, resp_entries, sig_eff, dio_pdf, cosmic_per_s_per_mev=COSMIC_PER_S_PER_MEV,
                npot=NPOT, pot_per_event=POT_PER_EVENT):
    """Metrics of the estimate from the selected events' histograms: front (FRONT_BINS counts, fill count
    front_entries), resp (RESP_BINS counts, resp_entries) and the signal efficiency sig_eff (stops/POT x acceptance)."""
    if front_entries <= 0 or resp_entries <= 0:
        raise ValueError("no selected electron reached the tracker entry")
    nf, flo, fhi = FRONT_BINS
    fw = (fhi - flo) / nf
    n2 = nf // 2
    sig = np.asarray(front, float)[:2 * n2].reshape(n2, 2).sum(1) * (npot * BR * sig_eff / front_entries)
    sw = 2 * fw
    nr, rlo, rhi = RESP_BINS
    rw = (rhi - rlo) / nr
    rden = np.asarray(resp, float) * (sig_eff / resp_entries / rw)
    sig = smear(sig, gaussian_kernel(sw, SIGMA_MEV))
    centers = flo + (np.arange(n2) + 0.5) * sw
    peak = int(np.argmax(sig))
    above = np.flatnonzero(sig > sig[peak] / 2.0)
    fwhm = float((above[-1] + 1 - above[0]) * sw) if above.size else 0.0
    nd, dlo, dhi = DIO_BINS
    dw = (dhi - dlo) / nd
    dio = dio_pdf * ((1.0 - CAPTURE) * sig_eff * npot)
    dio = smear(smear(dio, density_kernel(rden, rlo, rw, dw)), gaussian_kernel(dw, SIGMA_MEV))
    dio = regrid(dio, dlo, dw, n2, flo, sw)
    cos = np.full(n2, cosmic_per_s_per_mev * (npot / pot_per_event * ONSPILL_S_PER_EVENT) * sw)
    m = scan_box(sig, dio, cos, centers)
    m.update(signal_mpv_mev=float(centers[peak]), signal_fwhm_mev=fwhm)
    return m


def histograms(e_front, e_start, selected):
    """FRONT and RESP histograms of the selected events with a tracker-entry energy, and their fill count."""
    keep = selected & np.isfinite(e_front)
    v = np.asarray(e_front, float)[keep]
    d = v - np.broadcast_to(np.asarray(e_start, float), np.shape(e_front))[keep]
    nf, flo, fhi = FRONT_BINS
    nr, rlo, rhi = RESP_BINS
    return dict(front=np.histogram(v, nf, (flo, fhi))[0].astype(float), resp=np.histogram(d, nr, (rlo, rhi))[0].astype(float),
                entries=float(keep.sum()))


def sob_metrics(h, n_selected, n_gen, stops_per_gen, dio_pdf, upstream_eff=UPSTREAM_EFF,
                cosmic_per_s_per_mev=COSMIC_PER_S_PER_MEV):
    """Sensitivity and its factors: stops/POT, accepted per generated electron, signal per stop in the box, B."""
    acc = n_selected / n_gen
    stops_per_pot = stops_per_gen * upstream_eff
    sig_eff = stops_per_pot * acc
    m = sensitivity(h["front"], h["entries"], h["resp"], h["entries"], sig_eff, dio_pdf, cosmic_per_s_per_mev)
    m.update(stops_per_gen_event=stops_per_gen, stops_per_pot=stops_per_pot, ce_acceptance=acc, sig_eff=sig_eff,
             box_eff_per_stop=m["signal_rate"] / (NPOT * BR * stops_per_pot),
             box_frac_selected=m["signal_rate"] / (NPOT * BR * sig_eff), n_selected=float(n_selected),
             n_gen=float(n_gen))
    return m


def jackknife(fn, groups):
    """Jackknife standard error of fn(mask) over the distinct values of `groups` (mask keeps all but one group)."""
    g = np.unique(groups)
    if g.size < 2:
        return float("nan")
    v = np.array([fn(groups != k) for k in g])
    return float(np.sqrt((g.size - 1) / g.size * ((v - v.mean()) ** 2).sum()))


# --- calorimeter acceptance ------------------------------------------------------------------------------------------

CSI_MIP_MEV_PER_MM = 1.243 * 4.51 / 10.0   # minimum-ionising loss in CsI [MeV/mm] (PDG: 1.243 MeV cm2/g, 4.51 g/cm3)
CSI_PATH_MIN = CALO_EDEP_VIS_MEV / CSI_MIP_MEV_PER_MM   # 17.8 mm: CsI path that gives 10 MeV at minimum ionisation


@dataclass(frozen=True)
class CaloGeometry:
    axis_x: float
    z_front: tuple     # crystal front faces of the two disks [mm]
    z_back: tuple
    r_in: float        # crystal annulus (diskCrystalRIn, diskCrystalROut)
    r_out: float
    cell: float = 34.3            # crystal pitch (crystal + 2 wrappers)
    half: float = 17.0            # crystal half width
    grid: np.ndarray = None       # valid[n + rows, j + cols]: cell at x = j cell / 2, y = n cell (j = n mod 2)


def crystal_grid(cell, r_in, r_out, removed=((257.25, 583.1),)):
    """Crystal cells of one disk (Offline Disk::fillCrystalsIdeal with the square-shift map): cells of side `cell`
    whose four edges lie within [r_in, r_out], minus the four corner cells removed by hand. Returns the lookup."""
    nmax = int(np.ceil(r_out / cell)) + 1
    n = np.arange(-nmax, nmax + 1)[:, None]
    j = np.arange(-2 * nmax, 2 * nmax + 1)[None, :]
    x, y = j * cell / 2, n * cell * np.ones_like(j)
    ok = (j - n) % 2 == 0
    ax = np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * cell
    ay = np.array([-0.5, -0.5, 0.5, 0.5, -0.5]) * cell
    for i in range(1, 5):
        p0x, p0y, p1x, p1y = x + ax[i - 1], y + ay[i - 1], x + ax[i], y + ay[i]
        vx, vy = p1x - p0x, p1y - p0y
        t = -(p0x * vx + p0y * vy) / (vx * vx + vy * vy)
        cx, cy = np.where(t < 0, p0x, np.where(t > 1, p1x, p0x + t * vx)), np.where(t < 0, p0y, np.where(
            t > 1, p1y, p0y + t * vy))
        ok &= (np.hypot(cx, cy) >= r_in) & (np.maximum(np.hypot(p0x, p0y), np.hypot(p1x, p1y)) <= r_out)
    for rx, ry in removed:
        ok &= ~((np.abs(np.abs(x) - rx) < 1.0) & (np.abs(np.abs(y) - ry) < 1.0))
    return ok


def calo_from_config(c):
    """Crystal volumes of the disk calorimeter from a SimpleConfig dict (the arithmetic of DiskCalorimeterMaker)."""
    g = lambda k: c[f"calorimeter.{k}"]
    vd = g("vdThickness")
    crys_h, cap_h = g("crystalZLength") / 2, g("crystalCapZLength") / 2
    fp_h = (g("FPCarbonZLength") + g("FPFoamZLength") - g("pipeRadius") + g("FPCoolPipeRadius")) / 2
    bp_h = g("BPHoleZLength") / 2 + g("FEEZLength") / 2 + 2 * g("FEEBoxThickness") + g("BPPipeZOffset") / 2 + \
        g("BPPipeRadiusHigh")
    disk_h = fp_h + crys_h + cap_h + bp_h + vd
    disp = -disk_h + vd + 2 * fp_h + cap_h + cap_h
    z0, z1 = g("caloMotherZ0"), g("caloMotherZ1")
    centre = 0.5 * (z0 + z1)
    fronts = tuple(centre - 0.5 * (z1 - z0) + disk_h + s + g("FEBToDiskZOffset") + vd + disp
                   for s in c["calorimeter.diskZMotherShift"])
    cell = g("crystalXYLength") + 2 * g("wrapperThickness")
    return CaloGeometry(-c["mu2e.solenoidOffset"], fronts, tuple(f + 2 * crys_h for f in fronts),
                        g("diskCrystalRIn"), g("diskCrystalROut"), cell, g("crystalXYLength") / 2,
                        crystal_grid(cell, g("diskCrystalRIn"), g("diskCrystalROut")))


def in_crystal(calo, dx, dy):
    """True where (dx, dy) about the disk axis lies inside a crystal (not its wrapper or a gap)."""
    nr = (calo.grid.shape[0] - 1) // 2
    nc = (calo.grid.shape[1] - 1) // 2
    n = np.rint(dy / calo.cell).astype(int)
    j = n + 2 * np.rint((2 * dx / calo.cell - n) / 2).astype(int)
    ok = (np.abs(n) <= nr) & (np.abs(j) <= nc)
    v = np.zeros(dx.shape, bool)
    v[ok] = calo.grid[n[ok] + nr, j[ok] + nc]
    return v & (np.abs(dx - j * calo.cell / 2) <= calo.half) & (np.abs(dy - n * calo.cell) <= calo.half)


def calo_accept(field, calo, x, u, T, h_far=50.0, h_near=10.0, sub=5, margin=60.0, r_wall=950.0, s_max=30000.0,
                path_min=CSI_PATH_MIN, annulus=False):
    """Calorimeter acceptance of e- helices from (x, u, kinetic energy T): True where the path inside crystals
    reaches path_min (CSI_PATH_MIN: 10 MeV at minimum ionisation); annulus=True: where the helix enters the crystal
    annulus at all (the coarser first form). Field-map helix steps (field at the step midpoint), h_far until margin
    before the first disk, then h_near sampled at `sub` points. Tracker and support material are not modelled."""
    x, u = np.array(x, float), np.array(u, float)
    p = np.sqrt(T * (T + 2 * eioni.ME))
    n = x.shape[0]
    path = np.zeros(n)
    live = np.ones(n, bool)
    s = np.zeros(n)
    zf, zb = np.array(calo.z_front), np.array(calo.z_back)
    z_lo = x[:, 2].min() - 100.0 if n else 0.0
    goal = 1e-9 if annulus else path_min
    while live.any():
        i = np.flatnonzero(live)
        xi, ui, pi = x[i], u[i], p[i]
        near = xi[:, 2] > zf.min() - margin
        h = np.where(near, h_near, h_far)
        B0 = ehelix.field_at(field, xi)
        Bm = ehelix.field_at(field, ehelix.helix(xi, ui, pi, -1.0, B0, 0.5 * h)[0])
        for k in range(1, sub + 1):
            xk = ehelix.helix(xi, ui, pi, -1.0, Bm, h * k / sub)[0]
            dx, dy = xk[:, 0] - calo.axis_x, xk[:, 1]
            zin = ((xk[:, 2, None] >= zf) & (xk[:, 2, None] <= zb)).any(1)
            r = np.hypot(dx, dy)
            hit = zin & ((r >= calo.r_in) & (r <= calo.r_out) if annulus else in_crystal(calo, dx, dy))
            path[i] += np.where(hit, h / sub, 0.0)
        xn, un = ehelix.helix(xi, ui, pi, -1.0, Bm, h)
        x[i], u[i] = xn, un
        s[i] += h
        rn = np.hypot(xn[:, 0] - calo.axis_x, xn[:, 1])
        done = (path[i] >= goal) | (xn[:, 2] > zb.max()) | (xn[:, 2] < z_lo) | (rn > r_wall) | (s[i] > s_max)
        live[i[done]] = False
    return path >= goal


# --- fast chain ------------------------------------------------------------------------------------------------------

def _paths(geom, offline_root=None, maps=None, dio=None):
    from .__main__ import GEOM_FILE, MAPS, OFFLINE_ROOT
    root = offline_root or OFFLINE_ROOT
    mp = maps or MAPS
    return (geom or GEOM_FILE, root, [f"{mp}/DSMap.header", f"{mp}/TSdMap.header"], dio or f"{root}/{DIO_TABLE}")


def _ce_chunk(args):
    """One chunk of electrons from the stop table: transport to the tracker entry and the calorimeter flag."""
    (stops, geom_file, root, maps, n, seed, chunk, backend) = args
    from . import simpleconfig
    rng = np.random.default_rng(np.random.SeedSequence([seed, chunk]))
    geom = cechain.from_offline(geom_file, root)
    path = geom_file if os.path.isabs(geom_file) else f"{root}/{geom_file}"
    calo = calo_from_config(simpleconfig.read(path, root))
    field = cechain.field_maps(maps)
    x0, u0, p0, t0, idx = cechain.from_stops(rng, stops["x"], stops["t"], n)
    res = cechain.transport(rng, geom, field, x0, u0, p0, t0, backend=backend)
    plane = res.fate == cechain.PLANE
    acc, ann = np.zeros(n, bool), np.zeros(n, bool)
    if plane.any():
        acc[plane] = calo_accept(field, calo, res.x[plane], res.u[plane], res.T[plane])
        ann[plane] = calo_accept(field, calo, res.x[plane], res.u[plane], res.T[plane], annulus=True)
    e_front = np.where(plane, res.T + eioni.ME, np.nan)
    return dict(fate=res.fate.astype(np.int8), e_front=e_front, calo=acc, calo_annulus=ann,
                chunk=np.full(n, chunk, np.int16), r0=np.hypot(x0[:, 0] - geom.beam.axis_x, x0[:, 1]), z0=x0[:, 2],
                x_plane=res.x.astype(np.float32), u_plane=res.u.astype(np.float32))


def _single_thread():
    """Pool initializer: one thread per CE process (set before the process first uses JAX)."""
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                               " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


def run_ce(stops_npz, geom=None, electrons=100_000, seed=1, procs=1, chunks=None, offline_root=None, maps=None,
           backend="jax", mu_minus_only=True):
    """Electrons from the mu- stops of a stop table (``point --stops-out``), in `chunks` chunks over `procs`
    processes (spawned: the JAX backend does not survive fork). Returns per-electron arrays."""
    import time
    st = np.load(stops_npz)
    keep = st["q"] < 0 if mu_minus_only and (st["q"] < 0).any() else np.ones(st["q"].size, bool)
    stops = dict(x=st["x"][keep].astype(float), t=st["t"][keep].astype(float))
    if stops["t"].size == 0:
        raise ValueError(f"{stops_npz}: no stops")
    geom_file, root, mp, _ = _paths(geom, offline_root, maps)
    k = chunks or max(16, procs)
    sizes = np.full(k, electrons // k) + (np.arange(k) < electrons % k)
    jobs = [(stops, geom_file, root, mp, int(sizes[c]), seed, c, backend) for c in range(k) if sizes[c] > 0]
    t0 = time.time()
    if procs > 1:
        import multiprocessing as mp_
        with mp_.get_context("spawn").Pool(procs, initializer=_single_thread) as pool:
            parts = pool.map(_ce_chunk, jobs, chunksize=1)
    else:
        parts = [_ce_chunk(j) for j in jobs]
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    out["wall_s"] = np.array(time.time() - t0)
    out["stops_used"] = np.array(stops["t"].size)
    return out


def ce_metrics(ce, stops_per_gen, dio_pdf, upstream_eff=UPSTREAM_EFF, cosmic_per_s_per_mev=COSMIC_PER_S_PER_MEV,
               errors=True):
    """sob metrics of a fast CE sample, with jackknife errors over its chunks (sob, acceptance, box efficiency)."""
    n = ce["fate"].size

    def one(mask):
        h = histograms(ce["e_front"][mask], E_START, ce["calo"][mask])
        return sob_metrics(h, int(ce["calo"][mask].sum()), int(mask.sum()), stops_per_gen, dio_pdf, upstream_eff,
                           cosmic_per_s_per_mev)
    m = one(np.ones(n, bool))
    plane = ce["fate"] == cechain.PLANE
    m.update(electrons=float(n), plane_frac=float(plane.mean()), stopped_frac=float((ce["fate"] == cechain.STOPPED).mean()))
    if "calo_annulus" in ce:
        m["ce_acceptance_annulus"] = float(ce["calo_annulus"].mean())
    if errors:
        for key in ("sensitivity", "ce_acceptance", "box_eff_per_stop", "total_background"):
            m[f"{key}_err"] = jackknife(lambda mk, key=key: one(mk)[key], ce["chunk"])
    return m


def cmd_ce(a):
    out = run_ce(a.stops, a.geom, a.electrons, a.seed, a.procs, a.chunks, backend=a.backend)
    np.savez_compressed(a.out, **out)
    print(json.dumps(dict(electrons=int(out["fate"].size), accepted=int(out["calo"].sum()),
                          plane=int((out["fate"] == cechain.PLANE).sum()), wall_s=float(out["wall_s"]))))


def cmd_sob(a):
    ce = dict(np.load(a.ce))
    m = ce_metrics(ce, a.stops_per_gen, dio_density(_paths(None, dio=a.dio)[3]), a.upstream_eff, a.cosmic)
    json.dump(m, open(a.out, "w"), indent=1)
    print(json.dumps(m))


def cmd_point(a):
    """beam point (stops) + CE + sob in one: OUT/metrics.json holds the beam_point metrics and the sob metrics;
    OUT/ce_hists.npz the selected electrons' histograms (as `reduce` gives for Geant4) and all plane energies."""
    import time
    from .__main__ import main as beam_main
    os.makedirs(a.out, exist_ok=True)
    stops = os.path.join(a.out, "stops.npz")
    t0 = time.time()
    argv = ["point", "--g4", a.g4, "--out", a.out, "--n", str(a.muons), "--seed", str(a.seed), "--repeat", "0",
            "--batch", str(a.batch), "--chunk", "400", "--stops-out", stops]
    if a.geom:
        argv += ["--geom", a.geom]
    if a.z_up is not None:
        argv += ["--z-up", str(a.z_up)]
    beam_main(argv)
    t_beam = time.time() - t0
    bm = json.load(open(os.path.join(a.out, "metrics.json")))
    events = np.load(a.g4).get("events")
    if events is None or not np.ndim(events) == 0 or int(events) <= 0:
        raise ValueError(f"{a.g4}: the start table records no generated events (g4table --events)")
    muons_per_event = int(np.load(a.g4)["evt"].size) / float(events)
    stops_per_gen = bm["stop_frac"] * muons_per_event
    ce = run_ce(stops, a.geom, a.electrons, a.seed, a.procs, a.chunks, backend=a.backend)
    t_ce = float(ce["wall_s"])
    m = ce_metrics(ce, stops_per_gen, dio_density(_paths(a.geom, dio=a.dio)[3]), a.upstream_eff, a.cosmic)
    h = histograms(ce["e_front"], E_START, ce["calo"])
    np.savez_compressed(os.path.join(a.out, "ce_hists.npz"), n_selected=int(ce["calo"].sum()), n_gen=ce["fate"].size,
                        plane=np.histogram(ce["e_front"][ce["fate"] == cechain.PLANE], *FRONT_BINS[:1],
                                           range=FRONT_BINS[1:])[0], **h)
    m.update({f"beam_{k}" if k in m else k: v for k, v in bm.items()})
    m.update(muons_per_event=muons_per_event, beam_s=t_beam, ce_s=t_ce, electrons_per_s=m["electrons"] / t_ce)
    m = {k: float(v) for k, v in m.items()}
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"))
    if a.keep:
        np.savez_compressed(os.path.join(a.out, "ce.npz"), **ce)
    else:
        os.remove(stops)
    print(json.dumps(m))


# --- Geant4 reference (EdepAna tree) ---------------------------------------------------------------------------------

def reduce_edep(paths, tree="EDepAna/tree"):
    """Histograms and counts of the sensitivity estimate from EdepAna file(s): selection event_calo_edep_vis > 10,
    first primary's tracker-front energy (0 = none), birth energy; ngen = the largest running count per file."""
    import uproot
    fr, rs, ent, nsel, ngen, nev, est = [], [], 0.0, 0.0, 0.0, 0, []
    for p in paths:
        with uproot.open(p) as f:
            t = f[tree]
            b = t.arrays(["event_calo_edep_vis", "weight", "nprimaries", "ngen", "primary_trk_front_p",
                          "primary_trk_front_energy", "primary_start_e"], library="np")
        first = lambda name: np.array([e[0] if len(e) else np.nan for e in b[name]], float)
        fp, fe, se = first("primary_trk_front_p"), first("primary_trk_front_energy"), first("primary_start_e")
        fe = np.where(fp > 0.0, fe, np.nan)
        sel = np.asarray(b["event_calo_edep_vis"], float) > CALO_EDEP_VIS_MEV
        h = histograms(fe, se, sel)
        fr.append(h["front"])
        rs.append(h["resp"])
        ent += h["entries"]
        nsel += float(np.asarray(b["weight"], float)[sel].sum())
        ngen += float(np.max(b["ngen"])) if len(b["ngen"]) else 0.0
        nev += len(sel)
        est.append(se[np.isfinite(se)])
    est = np.concatenate(est)
    return dict(front=np.sum(fr, 0), resp=np.sum(rs, 0), entries=ent, n_selected=nsel, n_gen=ngen, n_events=nev,
                e_start_mean=float(est.mean()), e_start_min=float(est.min()), e_start_max=float(est.max()))


def calo_check(ref, nts, geom=None, offline_root=None, maps=None, tree="EDepAna/tree"):
    """The calorimeter proxy on Geant4 tracker-entry states (cereference table) against EdepAna's visible calorimeter
    energy of the same events (joined on subrun, event): per-electron flags and summary rates."""
    import uproot
    from . import simpleconfig
    r = np.load(ref)
    ev = {}
    for p in nts:
        with uproot.open(p) as f:
            b = f[tree].arrays(["subrun", "event", "event_calo_edep_vis"], library="np")
        ev.update(zip(zip(b["subrun"].tolist(), b["event"].tolist()), b["event_calo_edep_vis"].tolist()))
    key = list(zip(r["subrun"].tolist(), r["evt"].tolist()))
    have = np.array([k in ev for k in key])
    vis = np.array([ev.get(k, np.nan) for k in key])
    plane = have & r["isvd_front"]
    geom_file, root, mp, _ = _paths(geom, offline_root, maps)
    path = geom_file if os.path.isabs(geom_file) else f"{root}/{geom_file}"
    calo = calo_from_config(simpleconfig.read(path, root))
    field = cechain.field_maps(mp)
    x = np.stack([r["xvd_front"], r["yvd_front"], r["zvd_front"]], 1)[plane].astype(float)
    pv = np.stack([r["pxvd_front"], r["pyvd_front"], r["pzvd_front"]], 1)[plane].astype(float)
    p = np.linalg.norm(pv, axis=1)
    u = pv / p[:, None]
    T = np.sqrt(p ** 2 + eioni.ME ** 2) - eioni.ME
    proxy = calo_accept(field, calo, x, u, T)
    proxy_annulus = calo_accept(field, calo, x, u, T, annulus=True)
    from dataclasses import replace
    disk = [calo_accept(field, replace(calo, z_front=(zf,), z_back=(zb,)), x, u, T)
            for zf, zb in zip(calo.z_front, calo.z_back)]
    g4 = vis[plane] > CALO_EDEP_VIS_MEV
    bz = ehelix.field_at(field, x)[:, 2]
    rmax = cechain.helix_rmax(cechain.from_offline(geom_file, root), x, u, p, bz)[0]
    edges = np.arange(300.0, 760.0, 20.0)
    k = np.digitize(rmax, edges)
    bins = [dict(rmax_lo=float(edges[i - 1]), n=int((k == i).sum()), g4=float(g4[k == i].mean()),
                 proxy=float(proxy[k == i].mean()), annulus=float(proxy_annulus[k == i].mean()))
            for i in range(1, edges.size) if (k == i).sum()]
    t = np.asarray(r["tvd_front"], float)[plane]
    tedges = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 1e9])
    j = np.digitize(t, tedges)
    tbins = [dict(t_lo=float(tedges[i - 1]), n=int((j == i).sum()), g4=float(g4[j == i].mean()),
                  proxy=float(proxy[j == i].mean())) for i in range(1, tedges.size) if (j == i).sum()]
    summary = dict(rows=int(r["x"].size), joined=int(have.sum()), plane=int(plane.sum()), g4_sel=float(g4.mean()),
                   proxy_sel=float(proxy.mean()), annulus_sel=float(proxy_annulus.mean()),
                   by_disk={k: dict(n=int(m.sum()), g4=float(g4[m].mean()) if m.any() else -1.0) for k, m in (
                       ("disk0_only", disk[0] & ~disk[1]), ("disk1_only", disk[1] & ~disk[0]),
                       ("both_disks", disk[0] & disk[1]), ("none", ~disk[0] & ~disk[1]))},
                   both=int((g4 & proxy).sum()), g4_only=int((g4 & ~proxy).sum()),
                   proxy_only=int((proxy & ~g4).sum()), neither=int((~g4 & ~proxy).sum()),
                   g4_sel_no_plane=float(np.mean(vis[have & ~r["isvd_front"]] > CALO_EDEP_VIS_MEV))
                   if (have & ~r["isvd_front"]).any() else -1.0, rmax_bins=bins, t_bins=tbins)
    return summary, dict(rmax=rmax, g4=g4, proxy=proxy, annulus=proxy_annulus, vis=vis[plane], p=p, x=x, u=u, t=t)


def cmd_calocheck(a):
    summary, rows = calo_check(a.ref, a.nts, a.geom)
    np.savez_compressed(a.out + ".npz", **rows)
    json.dump(summary, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rmax_bins", "t_bins")}))


def cmd_reduce(a):
    r = reduce_edep(a.nts)
    np.savez_compressed(a.out, **r)
    print(json.dumps({k: float(v) for k, v in r.items() if np.ndim(v) == 0}))


def cmd_refsob(a):
    r = dict(np.load(a.ref))
    m = sob_metrics(dict(front=r["front"], resp=r["resp"], entries=float(r["entries"])), float(r["n_selected"]),
                    float(r["n_gen"]), a.stops_per_gen, dio_density(_paths(None, dio=a.dio)[3]), a.upstream_eff,
                    a.cosmic)
    json.dump(m, open(a.out, "w"), indent=1)
    print(json.dumps(m))


def cmd_hybrid(a):
    """Per point: sob from each pair of (CE side, stops) taken from Geant4 or the fast chain -> one JSON."""
    import glob
    dio = dio_density(_paths(None, dio=a.dio)[3])
    g4_stops = {c: float(v) for c, v, *_ in (l.split() for l in open(a.g4_stops) if l.strip())}
    out = {}
    for mpath in sorted(glob.glob(os.path.join(a.fast_dir, "*", "metrics.json"))):
        c = os.path.basename(os.path.dirname(mpath))
        ref, hist = os.path.join(a.ref_dir, f"{c}.npz"), os.path.join(os.path.dirname(mpath), "ce_hists.npz")
        if c not in g4_stops or not os.path.exists(ref) or not os.path.exists(hist):
            continue
        sides = {"g4": dict(np.load(ref)), "fast": dict(np.load(hist))}
        stops = {"g4": g4_stops[c], "fast": json.load(open(mpath))["stops_per_gen_event"]}
        out[c] = {}
        for ce_side, r in sides.items():
            h = dict(front=r["front"], resp=r["resp"], entries=float(r["entries"]))
            for st_side, spg in stops.items():
                m = sob_metrics(h, float(r["n_selected"]), float(r["n_gen"]), spg, dio, a.upstream_eff, a.cosmic)
                out[c][f"ce_{ce_side}_stops_{st_side}"] = {k: m[k] for k in (
                    "sensitivity", "signal_box_low_mev", "signal_box_high_mev", "signal_rate", "total_background",
                    "ce_acceptance", "box_frac_selected", "signal_mpv_mev", "signal_fwhm_mev")}
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"{len(out)} points -> {a.out}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.cesob", description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--dio", default=None, help="DIO table (default: the Offline root's ConditionsService table)")
        s.add_argument("--upstream-eff", dest="upstream_eff", type=float, default=UPSTREAM_EFF)
        s.add_argument("--cosmic", type=float, default=COSMIC_PER_S_PER_MEV, help="cosmics per s per MeV")

    def ce_args(s):
        s.add_argument("--geom", default=None)
        s.add_argument("--electrons", type=int, default=100_000)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--procs", type=int, default=1)
        s.add_argument("--chunks", type=int, default=None, help="chunks (jackknife groups); default max(16, procs)")
        s.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    s = sp.add_parser("point")
    s.add_argument("--g4", required=True, help="start table with its generated events (g4table --events)")
    s.add_argument("--muons", type=int, default=0)
    s.add_argument("--z-up", dest="z_up", type=float, default=None)
    s.add_argument("--batch", type=int, default=131072)
    s.add_argument("--keep", action="store_true", help="keep OUT/stops.npz and the per-electron OUT/ce.npz")
    s.add_argument("--out", required=True)
    ce_args(s)
    common(s)
    s = sp.add_parser("ce")
    s.add_argument("--stops", required=True)
    s.add_argument("--out", required=True)
    ce_args(s)
    s = sp.add_parser("sob")
    s.add_argument("--ce", required=True)
    s.add_argument("--stops-per-gen", dest="stops_per_gen", type=float, required=True)
    s.add_argument("--out", required=True)
    common(s)
    s = sp.add_parser("reduce")
    s.add_argument("--nts", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s = sp.add_parser("refsob")
    s.add_argument("--ref", required=True)
    s.add_argument("--stops-per-gen", dest="stops_per_gen", type=float, required=True)
    s.add_argument("--out", required=True)
    common(s)
    s = sp.add_parser("hybrid", help="sob for (CE side, stops) from Geant4 or fast, per point")
    s.add_argument("--ref-dir", dest="ref_dir", required=True, help="<config>.npz from reduce (Geant4)")
    s.add_argument("--fast-dir", dest="fast_dir", required=True, help="<config>/metrics.json, ce_hists.npz from point")
    s.add_argument("--g4-stops", dest="g4_stops", required=True, help="lines: config, Geant4 stops per generated event")
    s.add_argument("--out", required=True)
    common(s)
    s = sp.add_parser("calocheck", help="calorimeter proxy on Geant4 tracker-entry states vs EdepAna (diagnostic)")
    s.add_argument("--ref", required=True, help="cereference table of the dts files")
    s.add_argument("--nts", nargs="+", required=True, help="EdepAna trees of the same dts files")
    s.add_argument("--geom", default=None)
    s.add_argument("--out", required=True, help="output stem (.json summary, .npz per electron)")
    a = ap.parse_args(argv)
    {"calocheck": cmd_calocheck, "hybrid": cmd_hybrid, "point": cmd_point, "ce": cmd_ce, "sob": cmd_sob, "reduce": cmd_reduce, "refsob": cmd_refsob}[a.cmd](a)


if __name__ == "__main__":
    main()
