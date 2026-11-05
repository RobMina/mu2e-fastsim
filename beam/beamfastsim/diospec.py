"""Decay-in-orbit (DIO) electrons from muon stops, as the stock primary production makes them (opt-in, Geant4-free).

The stock DIO samples (e.g. DIOtail95) run `SingleProcessGenerator` with the `DIOGenerator` tool on stops resampled
from a MuminusStopsCat file:

- every stopped mu- of a resampled stops event (end code muMinusCaptureAtRest) gives one electron (`selectOne` false);
- the electron starts at the stop position, at the stop time plus an exponential decay time (mu- lifetime in Al);
- its direction is isotropic;
- its total energy is drawn from a tabulated spectrum (`spectrumShape tabulated`), restricted to `elow`..`ehi` the way
  Offline `BinnedSpectrum` restricts a table, and uniform inside a table bin (CLHEP RandGeneral with interpolation).

The stops events are read sequentially from a random start, one per generated event, wrapping at the end of the file.
`recover_window` finds that start for one stock file from its kept electrons, so a fast shard can use the same stops
events as the stock file it is paired with.

Subcommands: `stops` reduces a ReadVirtualDetector tree of the stops file to a stop table; `starts` writes start
states for `cechain --ref` (one shard per stock reference file, or explicit windows). Mu2e frame, mm, ns, MeV.
"""
import argparse
import json

import numpy as np

ME = 0.510998910               # MeV: electron mass (Offline ParticleList)
TAU_MU_AL = 864.0              # ns: physicsParams.Al.decayTime
DIO_CODE = 166                 # ProcessCode mu2eMuonDecayAtRest
STOP_CODE = 32                 # ProcessCode muMinusCaptureAtRest (Geant4 end code of a stopped mu-, decay or capture)
TABLE = "Offline/EventGenerator/data/heeck_finer_binning_2016_szafron.tbl"


def event_key(run, subrun, evt):
    """One int64 per art EventID (run < 2^21, subrun < 2^21, event < 2^21)."""
    return (np.asarray(run, np.int64) << 42) | (np.asarray(subrun, np.int64) << 21) | np.asarray(evt, np.int64)


class Spectrum:
    """A tabulated spectrum restricted as Offline BinnedSpectrum::initialize(Table, binCentered, elo, ehi).

    rows: (x, y) table rows; x are left bin edges unless bin_centered. Rows whose bin starts below elow are skipped,
    and the table ends at the first bin reaching above ehi (elow / ehi <= 0: no cut)."""

    def __init__(self, x, y, elow=0.0, ehi=0.0, bin_centered=False):
        x, y = np.asarray(x, float), np.asarray(y, float)
        w = x[1] - x[0]
        c = x if bin_centered else x + w / 2.0
        keep = np.ones(c.size, bool)
        if elow > 0:
            keep &= ~(c - w / 2.0 < elow)
        if ehi > 0:
            over = np.flatnonzero(c + w / 2.0 > ehi)
            if over.size:
                keep[over[0]:] = False
        self.centres, self.pdf = c[keep], y[keep]
        if self.centres.size < 2:
            raise ValueError("Spectrum: fewer than two bins left after the cut")
        self.width = (self.centres[-1] - self.centres[0]) / (self.centres.size - 1)
        self.xmin = self.centres[0] - self.width / 2.0
        self.xmax = self.centres[-1] + self.width / 2.0
        self.elow, self.ehi = elow, ehi

    @classmethod
    def from_file(cls, path, elow=95.0, ehi=1000.0, bin_centered=False):
        t = np.loadtxt(path, usecols=(0, 1))
        return cls(t[:, 0], t[:, 1], elow, ehi, bin_centered)

    @property
    def edges(self):
        """Sampling bin edges: nbins equal bins from xmin to xmax (what the RandGeneral draw maps onto)."""
        return self.xmin + (self.xmax - self.xmin) * np.arange(self.pdf.size + 1) / self.pdf.size

    def probabilities(self):
        return self.pdf / self.pdf.sum()

    def sample(self, rng, n):
        """Total energies [MeV]: bin by the table weights, uniform inside the bin."""
        cdf = np.cumsum(self.pdf)
        r = rng.random(n) * cdf[-1]
        i = np.minimum(np.searchsorted(cdf, r, side="right"), self.pdf.size - 1)
        lo = np.where(i > 0, cdf[np.maximum(i - 1, 0)], 0.0)
        f = (r - lo) / self.pdf[i]
        return self.xmin + (self.xmax - self.xmin) * (i + f) / self.pdf.size


def momentum(E, m=ME):
    """|p| from total energy, as DIOGenerator: E sqrt(1 - (m/E)^2)."""
    return E * np.sqrt(1.0 - (m / E) ** 2)


def isotropic(rng, n):
    ct = rng.uniform(-1.0, 1.0, n)
    ph = rng.uniform(0.0, 2 * np.pi, n)
    st = np.sqrt(1 - ct * ct)
    return np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)


def stops_table(paths, tree="mutree/ntpart"):
    """Stop table from ReadVirtualDetector trees of a stops file (savePDG mu_minus), in file order.

    Returns dict: iev (ordinal of the stops event, 0-based, in reading order), x, y, z, t of every stopped mu- (end code
    STOP_CODE), nev (events seen) and the run / subrun / evt of each stop."""
    import uproot
    cols = {k: [] for k in ("run", "subrun", "evt", "pdg", "codestop", "xstop", "ystop", "zstop", "tstop")}
    for p in paths:
        with uproot.open(p) as f:
            cyc = max((k for k in f.keys() if k.split(";")[0] == tree), key=lambda k: int(k.split(";")[1]))
            a = f[cyc].arrays(list(cols), library="np")
        for k in cols:
            cols[k].append(a[k])
    a = {k: np.concatenate(v) for k, v in cols.items()}
    key = event_key(a["run"], a["subrun"], a["evt"])
    new = np.ones(key.size, bool)
    new[1:] = key[1:] != key[:-1]
    iev = np.cumsum(new) - 1
    sel = (a["pdg"] == 13) & (a["codestop"] == STOP_CODE)
    return dict(iev=iev[sel], x=a["xstop"][sel].astype(float), y=a["ystop"][sel].astype(float),
                z=a["zstop"][sel].astype(float), t=a["tstop"][sel].astype(float), run=a["run"][sel],
                subrun=a["subrun"][sel], evt=a["evt"][sel], nev=int(iev[-1] + 1) if iev.size else 0)


AUX_BYTES = 78        # art::EventAuxiliary (class version 10) entry in an art Events tree, no compression of members
AUX_ID_AT = 47        # its EventID: run, subRun, event as big-endian uint32


def art_event_ids(path):
    """(run, subrun, event) of every entry of an art file's Events tree, in tree-entry order (the order a sequential
    ResamplingMixer reads, which differs from RootInput's sorted order in a concatenated file)."""
    import uproot
    with uproot.open(path) as f:
        br = f["Events"]["EventAuxiliary"]
        parts = []
        for i in range(br.num_baskets):
            b = br.basket(i)
            if b.byte_offsets is None or np.any(np.diff(b.byte_offsets) != AUX_BYTES):
                raise ValueError("art_event_ids: unexpected EventAuxiliary entry layout")
            parts.append(np.frombuffer(b.data, np.uint8)[:b.num_entries * AUX_BYTES])
        n = br.num_entries
    raw = np.concatenate(parts).reshape(n, AUX_BYTES)
    ids = raw[:, AUX_ID_AT:AUX_ID_AT + 12].copy().view(">u4").astype(np.int64)
    return ids[:, 0], ids[:, 1], ids[:, 2]


def entry_order(stops, run, subrun, event):
    """Replace the stop table's event ordinals by tree-entry ordinals (from art_event_ids); keep the read order in
    iev_read."""
    ek = event_key(run, subrun, event)
    order = np.argsort(ek)
    sk = event_key(stops["run"], stops["subrun"], stops["evt"])
    pos = np.clip(np.searchsorted(ek[order], sk), 0, ek.size - 1)
    if not np.array_equal(ek[order][pos], sk):
        raise ValueError("entry_order: stops not found among the art file's events")
    out = dict(stops)
    out["iev_read"], out["iev"], out["nev"] = stops["iev"], order[pos], int(ek.size)
    return out


def window(stops, start, nev):
    """Row indices of the stops in nev consecutive stops events from ordinal `start`, wrapping at the end."""
    n = stops["nev"]
    rel = (stops["iev"] - start) % n
    rows = np.flatnonzero(rel < nev)
    return rows[np.argsort(rel[rows], kind="stable")]


def recover_window(stops, x, y, z, evt):
    """Start ordinal of the stops window a stock file read: its kept primaries (start x, y, z as float32 and art event
    number, 1-based) matched to the stop table. Returns (start, matched, consistent): consistent is the fraction of
    matched electrons whose stop ordinal equals start + evt - 1 (mod the event count)."""
    u32 = lambda a: np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    sk = (u32(stops["x"]) << np.uint64(32)) | u32(stops["y"])
    order = np.argsort(sk, kind="stable")
    q = (u32(x) << np.uint64(32)) | u32(y)
    pos = np.clip(np.searchsorted(sk[order], q), 0, sk.size - 1)
    hit = (sk[order][pos] == q) & (np.asarray(stops["z"], np.float32)[order][pos] == np.asarray(z, np.float32))
    if not hit.any():
        return None, 0, 0.0
    iev = stops["iev"][order][pos][hit]
    off = (iev - (np.asarray(evt)[hit] - 1)) % stops["nev"]
    vals, cnt = np.unique(off, return_counts=True)
    start = int(vals[np.argmax(cnt)])
    return start, int(hit.sum()), float(cnt.max() / hit.sum())


def generate(rng, spectrum, stops, rows, tau=TAU_MU_AL):
    """One DIO electron per stop row: dict x (n, 3), u (n, 3), p, E, t, row."""
    n = rows.size
    E = spectrum.sample(rng, n)
    x = np.stack([stops["x"][rows], stops["y"][rows], stops["z"][rows]], axis=1)
    return dict(x=x, u=isotropic(rng, n), p=momentum(E), E=E, t=stops["t"][rows] + rng.exponential(tau, n), row=rows)


def _starts_cli(a):
    st = dict(np.load(a.stops))
    st["nev"] = int(st["nev"])
    spec = Spectrum.from_file(a.table, a.elow, a.ehi)
    shards = []
    if a.ref:
        r = np.load(a.ref)
        for i in range(r["gencount"].size):
            m = r["ifile"] == i
            s0, nm, cons = recover_window(st, r["x"][m], r["y"][m], r["z"][m], r["evt"][m])
            shards.append(dict(shard=i, start=s0, events=int(r["gencount"][i]), matched=nm, rows=int(m.sum()),
                               consistent=cons))
    else:
        for i, s0 in enumerate(a.window_starts):
            shards.append(dict(shard=i, start=int(s0), events=a.events, matched=0, rows=0, consistent=1.0))
    parts = []
    for sh in shards:
        if sh["start"] is None:
            sh["generated"] = 0
            continue
        rows = window(st, sh["start"], sh["events"])
        sh["generated"] = int(rows.size)
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, sh["shard"]]))
        g = generate(rng, spec, st, rows, a.tau)
        g["shard"] = np.full(rows.size, sh["shard"])
        parts.append(g)
    if not parts:
        raise SystemExit("diospec starts: no shard has a stops window (no stock electron matched the stop table)")
    cat = {k: np.concatenate([g[k] for g in parts]) for k in parts[0]}
    pv = cat["u"] * cat["p"][:, None]
    np.savez_compressed(a.out, x=cat["x"][:, 0], y=cat["x"][:, 1], z=cat["x"][:, 2], px=pv[:, 0], py=pv[:, 1],
                        pz=pv[:, 2], time=cat["t"], E=cat["E"], shard=cat["shard"], stop_row=cat["row"],
                        shard_id=np.array([s["shard"] for s in shards]),
                        shard_generated=np.array([s["generated"] for s in shards]),
                        shard_start=np.array([-1 if s["start"] is None else s["start"] for s in shards]),
                        shard_events=np.array([s["events"] for s in shards]), elow=a.elow, ehi=a.ehi, tau=a.tau)
    bad = [s for s in shards if s["start"] is None or s["consistent"] < 0.999]
    print(json.dumps(dict(shards=len(shards), electrons=int(cat["p"].size), spectrum=dict(
        bins=int(spec.pdf.size), xmin=spec.xmin, xmax=spec.xmax), inconsistent=bad[:10], n_inconsistent=len(bad)),
        indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m beamfastsim.diospec", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stops", help="stop table from ReadVirtualDetector trees of a stops file")
    s.add_argument("roots", nargs="+")
    s.add_argument("--tree", default="mutree/ntpart")
    s.add_argument("--art", help="the stops art file itself: order the events as its Events tree (mixer order)")
    s.add_argument("--out", required=True)
    g = sub.add_parser("starts", help="DIO start states (cechain --ref format), one shard per stops window")
    g.add_argument("--stops", required=True, help="stop table from the stops subcommand")
    g.add_argument("--table", required=True, help=f"spectrum table, e.g. <Offline root>/{TABLE}")
    g.add_argument("--elow", type=float, default=95.0)
    g.add_argument("--ehi", type=float, default=1000.0, help="as the MDC2025ba DIOtail95 job configuration (config_dumper)")
    g.add_argument("--tau", type=float, default=TAU_MU_AL)
    g.add_argument("--events", type=int, default=4000,
                   help="stops events per shard with --window-starts (with --ref: each file's generated count)")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref", help="stock reference table (cereference, DIO code): one shard per file, same window")
    src.add_argument("--window-starts", type=int, nargs="+", help="explicit window start ordinals, one per shard")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "stops":
        t = stops_table(a.roots, a.tree)
        if a.art:
            t = entry_order(t, *art_event_ids(a.art))
        np.savez_compressed(a.out, **t)
        print(json.dumps(dict(stops=int(t["iev"].size), events=t["nev"],
                              multi=int((np.bincount(t["iev"], minlength=t["nev"]) > 1).sum()),
                              empty=int((np.bincount(t["iev"], minlength=t["nev"]) == 0).sum()))))
    else:
        _starts_cli(a)


if __name__ == "__main__":
    main()
