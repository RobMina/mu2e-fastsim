"""On-spill response of the device trigger paths, per event class and birth time (opt-in, ``assemble --trigger-cal``).

The device trigger (``trigger.device``) decides the tracker paths from the fast device digis, while the fast track
comes from the tracker response layer, which does not see those digis. In stock both come from the same hits: an
event with a good offline track nearly always fires the track paths. The fast chain loses that correlation, and the
device paths were calibrated (KalSeedFilter / AgnosticHelix factors) on off-spill CE replays.

This table restores both on the on-spill chain. From development events with both arms (the same births):

* target  T[path, class, bin] = stock pass fraction of the path among stock events of that class,
* device  D[path, class, bin] = device pass fraction (no response layer) among fast events of that class,

with class Q / F / N (quality e- fit, failing fit, no fit: the donor classes, stock from its fit, fast from the
response layer) and bin = the generated electron's time (the digitization-window turn-on and turn-off). A fast event
keeps a device pass with probability f = min(1, T / D) and turns a device fail into a pass with probability
g = max(0, (T - D) / (1 - D)), so the pass fraction per class and bin is T while the device decision still orders
the events. One uniform per event, shared by every path, keeps the paths' mutual correlation. Cells with fewer than
``min_count`` events on either side use the class's time-integrated factors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import donor as D

NAME = "trigger.centuple_onspill"
SHIPPED = Path(__file__).with_name("data") / "trigcal_run1a_onspill.npz"
CLASSES = (D.CLASS_N, D.CLASS_F, D.CLASS_Q)
EDGES = (-1e9, 300, 400, 450, 500, 550, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
         2000, 2200, 2500, 1e9)
MIN_COUNT = 50


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def stock_events(paths, path_names):
    """Per stock event: run, subrun, event, class (donor rule) and the trig_ bits of path_names."""
    import awkward as ak
    import uproot
    out = []
    for p in paths:
        d = D.index_file(p, list(path_names))
        t = uproot.open(p)["EventNtuple/ntuple"]
        names = set(t.keys())
        a = t.arrays([("trig_" + q) for q in path_names if "trig_" + q in names], library="ak")
        trig = np.zeros((d["run"].size, len(path_names)), bool)
        for j, q in enumerate(path_names):
            if "trig_" + q in names:
                trig[:, j] = ak.to_numpy(a["trig_" + q]).astype(bool)
        out.append(dict(run=d["run"], subrun=d["subrun"], event=d["event"], cls=d["cls"], trig=trig))
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def fast_events(records, triggers):
    """Per fast event: run, subrun, event, class (response layer), birth time and the raw device decisions."""
    from .assemble import classes
    out, names = [], None
    for r, t in zip(records, triggers):
        rec, trg = dict(np.load(r)), np.load(t)
        pn = sorted(k[5:] for k in trg.files if k.startswith("trig_"))
        if names is None:
            names = pn
        if pn != names:
            raise ValueError(f"{t}: other trigger paths than the first file")
        if not np.array_equal(np.asarray(trg["event"]), rec["event"]):
            raise ValueError(f"{t}: events differ from {r}")
        out.append(dict(run=rec["run"], subrun=rec["subrun"], event=rec["event"], cls=classes(rec),
                        tbirth=np.asarray(rec["birth_t"], float),
                        trig=np.stack([np.asarray(trg["trig_" + q], bool) for q in names], 1)))
    return names, {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def factors(T, D_):
    """keep f and rescue g from target and device pass fractions."""
    f = np.where(D_ > 0, np.minimum(1.0, T / np.where(D_ > 0, D_, 1.0)), 1.0)
    g = np.where((T > D_) & (D_ < 1), (T - D_) / np.where(D_ < 1, 1 - D_, 1.0), 0.0)
    return f, g


def counts(paths, stock, fast, edges=EDGES):
    """Paired counts: stock (n, pass) per stock class and fast (n, pass) per fast class, in birth-time bins of the
    common events (the births are shared, so the stock event's birth time is the fast one's)."""
    ks = {k: i for i, k in enumerate(zip(stock["run"].tolist(), stock["subrun"].tolist(), stock["event"].tolist()))}
    i_f, i_s = [], []
    for i, k in enumerate(zip(fast["run"].tolist(), fast["subrun"].tolist(), fast["event"].tolist())):
        j = ks.get(k)
        if j is not None:
            i_f.append(i)
            i_s.append(j)
    i_f, i_s = np.asarray(i_f, np.int64), np.asarray(i_s, np.int64)
    nb, npth = len(edges) - 1, len(paths)
    b = np.clip(np.searchsorted(edges, fast["tbirth"][i_f], side="right") - 1, 0, nb - 1)
    c = {}
    for side, cls, trig in (("stock", stock["cls"][i_s], stock["trig"][i_s]), ("fast", fast["cls"][i_f],
                                                                                fast["trig"][i_f])):
        n = np.zeros((len(CLASSES), nb))
        s = np.zeros((npth, len(CLASSES), nb))
        for ci, cv in enumerate(CLASSES):
            m = cls == cv
            n[ci] = np.bincount(b[m], minlength=nb)
            for j in range(npth):
                s[j, ci] = np.bincount(b[m], weights=trig[m, j].astype(float), minlength=nb)
        c[side] = (n, s)
    return c, int(i_f.size)


def table(paths, c, min_count=MIN_COUNT):
    (ns, ss), (nf, sf) = c["stock"], c["fast"]
    T = ss / np.maximum(ns, 1)[None]
    Dv = sf / np.maximum(nf, 1)[None]
    Ti = ss.sum(2) / np.maximum(ns.sum(1), 1)[None]
    Di = sf.sum(2) / np.maximum(nf.sum(1), 1)[None]
    sparse = (np.minimum(ns, nf) < min_count)[None] | np.zeros_like(T, bool)
    T = np.where(sparse, Ti[:, :, None], T)
    Dv = np.where(sparse, Di[:, :, None], Dv)
    f, g = factors(T, Dv)
    return dict(keep=f, rescue=g, target=T, device=Dv, sparse=sparse[0])


def build(stock_paths, records, triggers, out, *, edges=EDGES, min_count=MIN_COUNT, commit=None, stock_release=None,
          stock_config=None):
    """Development stock EventNtuples (every digitized event kept) + fast records and raw device decisions of the
    same births -> the table (.npz with a library manifest)."""
    from mu2efastsim.library import store
    from mu2efastsim.library.axes import Axis
    paths, fast = fast_events(records, triggers)
    stock = stock_events(stock_paths, paths)
    c, npair = counts(paths, stock, fast, edges)
    tb = table(paths, c, min_count)
    ax = Axis("tbirth", "ns", list(edges), frame="generated electron time (MC, from the proton time)",
              scale="linear", policy="clip", description="birth-time bins")
    arr = dict(paths=np.array(paths), classes=np.array(CLASSES, np.int8), keep=tb["keep"], rescue=tb["rescue"],
               target=tb["target"], device=tb["device"], sparse=tb["sparse"], stock_n=c["stock"][0],
               stock_pass=c["stock"][1], fast_n=c["fast"][0], fast_pass=c["fast"][1])
    arr[ax.array] = ax.edges
    files = [dict(name=Path(p).name, sha256=_sha256(p)) for p in stock_paths]
    fields = dict(
        schema=1, name=NAME + ".run1a", subsystem="trigger", version=1, kind="binned",
        description=("On-spill response of the device trigger paths for the fast CE EventNtuple: per path, event class "
                     "(Q/F/N) and birth time, the probability to keep a device pass and to turn a device fail into a "
                     "pass, so that the pass fraction per class and bin is stock's (development events with both "
                     "arms, the same births)."),
        axes=[ax.describe()],
        outputs=[dict(name="keep", unit="1", dtype="float64", description="P(keep a device pass) [path, class, bin]"),
                 dict(name="rescue", unit="1", dtype="float64", description="P(pass | device fail) [path, class, bin]")],
        source=dict(samples=files, description=f"stock EventNtuples, every digitized event kept; {npair} events paired "
                                               f"with fast records of the same births"),
        generator=dict(name="stock Mu2e chain", release=stock_release or "AnalysisMDC2025 v02_02_02 (SimJob MDC2025ax)",
                       physics_list=None, configuration=stock_config or "Mu2eBO ce_chain dig/mcs/nts FCLs, "
                                                                        "outputs.Output.SelectEvents: []"),
        release=dict(repository="mu2e-fastsim", commit=commit, entry_point="beamfastsim.centuple.trigcal"),
        seeds=dict(apply="per event: numpy SeedSequence([seed, 2, run, subrun, event]), one uniform for every path"),
        geometry=dict(tag="run1a", hash="Offline/Mu2eG4/geom/geom_run1_a.txt"),
        config_hash=hashlib.sha256(json.dumps(dict(edges=list(edges), min_count=min_count)).encode()).hexdigest(),
        split=dict(training="the files above", held_out="disjoint files", rule="by file"),
        dtype=dict(storage={}, compute=["float32", "float64"], notes="host-side NumPy float64; the device decisions "
                                                                     "come from the record's dtype"),
        rng=dict(host="numpy.random.Generator", device=None, streams={"trigcal": "per event"}, draws_per_row=1),
        domain=dict(coverage=f"{npair} paired events; sparse cells (< {min_count} events on a side) use the class's "
                             f"time-integrated factors",
                    fallbacks=[dict(name="time-integrated", levels=1, description="the class's factors over all bins")]),
        validation=dict(status="not validated", record="beam/docs/eventntuple.md (validation)"),
        unrecorded={})
    if commit is None:
        fields["release"]["commit"] = "unknown"
    fields["generator"]["physics_list"] = "stock (the dts files' Geant4)"
    store.save(out, arr, fields)
    return dict(paired=npair, paths=len(paths))


def load(path):
    from mu2efastsim.library import store
    lib = store.load(Path(path))
    return {k: np.asarray(lib.arrays[k]) for k in lib.arrays}


def apply(lib, names, trig, cls, tbirth, run, subrun, event, seed):
    """Calibrated decisions: trig [n, len(names)] raw device decisions -> new bool array (paths not in the table
    unchanged)."""
    paths = [str(p) for p in lib["paths"]]
    edges = lib["tbirth_edges"]
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, np.nan_to_num(tbirth, nan=1e8), side="right") - 1, 0, nb - 1)
    ci = np.searchsorted(np.asarray(lib["classes"]), cls)
    u = np.array([np.random.default_rng(np.random.SeedSequence([seed, 2, int(r), int(s), int(e)])).random()
                  for r, s, e in zip(run, subrun, event)])
    out = np.array(trig, bool, copy=True)
    for j, q in enumerate(names):
        if q not in paths:
            continue
        k = paths.index(q)
        f, g = lib["keep"][k, ci, b], lib["rescue"][k, ci, b]
        out[:, j] = np.where(trig[:, j], u < f, u < g)
    return out
