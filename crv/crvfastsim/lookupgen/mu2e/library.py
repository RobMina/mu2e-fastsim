"""Library-core manifests of generated lookup tables (mu2efastsim.library; core/docs/libraries.md).

The version-6 file stays the payload Offline reads; its manifest is the sidecar ``<file>.manifest.json``
(store.sidecar_path). The arrays are the decoded table, ``crvfastsim.response.lookup.arrays``, so the manifest's
content hash is the hash every pinned load already checks (``lookup.load(name, pin={"content_hash": ...})``): a run
configuration or a benchmark references a generated table by that hash.

``write`` builds the manifest of an assembled table from its pieces' provenance; ``verify`` checks a table against
its sidecar (schema, arrays, content hash, axes, and any expected field such as a pinned content hash).
"""
import datetime
import json
import os

from mu2efastsim.library import Axis, store
from mu2efastsim.library import manifest as LM
from mu2efastsim.library.hashing import digest

from ..core import optics as O
from ..core import transport as T
from . import sections as G
from . import writer as W

REPOSITORY = "RobMina/mu2e-fastsim"
TABLES = {"scint": 0, "ckscint": 1, "ckfiber": 2}
AXES = {"x": ("mm", "|x| along the thickness, counter centre (the table is mirror symmetric in x)"),
        "y": ("mm", "along the width, counter centre; SiPM 0 reads the fiber at y < 0"),
        "z": ("mm", "along the length, counter centre; SiPM 0 is at -z"),
        "beta": ("1", "particle speed / c (Cerenkov tables)"),
        "theta": ("rad", "photon polar angle to the fiber axis (fiber Cerenkov)"),
        "phi": ("rad", "photon azimuth about the fiber axis (fiber Cerenkov)"),
        "r": ("mm", "distance from the fiber axis (fiber Cerenkov)")}
MODEL_SOURCES = {"current": "CRVStandalone since July 2021 (crvfastsim.lookupgen.core.materials, sources cited there)",
                 "v6": "Offline CRVResponse/standalone of January 2019, tuned to Mu2e test-beam data "
                       "(crvfastsim.lookupgen.mu2e.v6)"}
# what the table can claim per optical model (docs/lookupgen.md, "Validation")
VALIDATION = {"v6": ("tested, not closed",
                     "crv/docs/lookupgen.md: the transport matches the released v6_0 tables for scintillation and "
                     "scintillator Cerenkov light; fiber Cerenkov light differs (open)"),
              "current": ("not validated",
                          "crv/docs/lookupgen.md: comparison with the CRVStandalone build of the same model pending")}


def table_name(c):
    """crv.lookup.<length>_<reflector> for the stock cross-section, crv.lookup.<section tag>_<...> otherwise."""
    return f"crv.lookup.{G.name(c)}" if G.is_stock(c) else f"crv.lookup.{G.section_tag(c)}_{G.name(c)}"


def _load(path):
    from crvfastsim.response import lookup
    path = os.path.abspath(path)
    L = lookup.load(os.path.basename(path), directory=os.path.dirname(path))
    return lookup.arrays(L)


def fields(c, model, pieces, arrays, notes=(), commit=None, created_utc=None):
    """Manifest fields (without arrays and content_hash) of a table of counter c made with ``model`` from ``pieces``:
    {sim type: [piece provenance (writer.piece_provenance + photons_per_bin, bins, file)]}."""
    unrecorded = {}
    flat = [p for st in sorted(pieces) for p in pieces[st]]
    per_st = {G.SIM_TYPES[st]: [{k: p.get(k) for k in ("photons_per_bin", "bins", "dtype", "max_steps", "seed")}
                                for p in v] for st, v in sorted(pieces.items())}
    dtypes = sorted({p.get("dtype") for p in flat if p.get("dtype")})
    steps = sorted({p.get("max_steps") for p in flat if p.get("max_steps")})
    seeds = {}
    for st, v in sorted(pieces.items()):
        s = sorted({p.get("seed") for p in v})
        seeds[G.SIM_TYPES[st]] = s[0] if len(s) == 1 else s
    if any(p.get("seed") is None for p in flat):
        seeds = None
        unrecorded["seeds"] = "a piece does not record its seed (made before seeds were recorded)"
    piece_commits = sorted({p.get("commit") for p in flat if p.get("commit")})
    release = dict(repository=REPOSITORY, commit=commit, entry_point="python -m crvfastsim.lookupgen assemble")
    if commit is None:
        release = None
        unrecorded["release"] = "assemble was not given --commit"
    status, record = VALIDATION[model]
    axes = [Axis(name=k, unit=u, edges=arrays[f"edges_{k}"], frame="counter", array=f"edges_{k}",
                 policy="refuse", description=d + "; outside the edges the stock findBin has no bin: the step makes "
                 "no photons from that table").describe() for k, (u, d) in AXES.items()]
    outputs = []
    for t, st in TABLES.items():
        outputs += [dict(name=f"{t}_prob", unit="1", dtype="float32",
                         description=f"{G.SIM_TYPES[st]}: probability that a photon made in the bin reaches SiPM 0"),
                    dict(name=f"{t}_delays", unit="1", dtype="uint8",
                         description="arrival-time weights per ns (255 = all arrivals)"),
                    dict(name=f"{t}_emis", unit="1", dtype="uint8",
                         description="weights per number of fiber (Y11) emissions")]
    config = dict(counter=dict(c.__dict__), model=model, writer=W.WRITER[model], pieces=per_st)
    return dict(
        name=table_name(c), subsystem="crv", version=1, kind="binned",
        description=(f"CRV photon lookup table {G.name(c)} ({'stock' if G.is_stock(c) else G.section_tag(c)} "
                     f"cross-section), optical model {model!r}: per bin of each photon source (scintillation and "
                     "Cerenkov light in the scintillator, Cerenkov light in the fiber), the probability that a photon "
                     "made there reaches SiPM 0, with its arrival-time and fiber-emission distributions. Generated "
                     "by crvfastsim.lookupgen; the payload is the version-6 file Offline MakeCrvPhotons reads."),
        axes=axes, outputs=outputs,
        source=dict(samples=[], description="no input sample: photons generated per bin by the generator"),
        generator=dict(name="crvfastsim.lookupgen", release=piece_commits or None,
                       physics_list="Geant4 10.7 optical physics of CRVStandalone table mode, ported to JAX "
                                    "(G4OpBoundaryProcess, G4OpAbsorption, G4OpWLS, scintillation, G4CerenkovNew)",
                       configuration=dict(model=model, dtypes=dtypes, max_steps=steps, pieces=per_st)),
        release=release, seeds=seeds,
        geometry=dict(tag=f"{'stock' if G.is_stock(c) else G.section_tag(c)}_{G.name(c)}",
                      hash=digest(dict(c.__dict__))),
        config_hash=digest(config),
        split=dict(training=None, held_out=None, rule="not applicable: a generated table, nothing is fitted"),
        dtype=dict(storage={k: str(v.dtype) for k, v in sorted(arrays.items())}, compute=list(LM.COMPUTE_DTYPES),
                   notes=f"generated in {', '.join(dtypes) or '?'} on the device; a lookup reads the stored float32 "
                         "probabilities and uint8 weights in either compute dtype"),
        rng=dict(host=None, device="squares32 counter-based generator keyed by (seed, sim type); counter "
                                   "(bin, photon, step, draw)",
                 streams=dict(transport="key(seed, sim type)", source="key(seed, sim type + 16)"),
                 draws_per_row=None),
        domain=dict(coverage="the scintillator and fiber volumes of the counter; a bin is a box of the axes",
                    fallbacks=[]),
        validation=dict(status=status, record=record),
        unrecorded=unrecorded,
        legacy=dict(schema=0, format="CRV LookupTable version %d.%d (binary, Offline CRVResponse MakeCrvPhotons); "
                                     "this manifest is its sidecar" % W.VERSION),
        interpolation=dict(method="nearest", axes=list(AXES)),
        assumptions=[dict(name="step limit", tested=True,
                          description=f"photon chains stopped at {', '.join(map(str, steps))} steps (exact limit "
                                      f"{1 << T.STEP_BITS}); stopped chains are counted, see docs/lookupgen.md"),
                     dict(name="optical model", tested=status != "not validated",
                          description=f"{model!r}: {MODEL_SOURCES[O.get(model).name]}")]
        + [dict(name=f"stock quirk {i + 1}", description=q, tested=True) for i, q in enumerate(W.QUIRKS)],
        created_utc=created_utc or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes="\n".join(notes) if notes else None,
    )


def write(path, c, model, pieces, notes=(), commit=None):
    """Write the sidecar manifest of the table at ``path``; returns it."""
    arrays = _load(path)
    f = fields(c, model, pieces, arrays, notes, commit)
    if f["notes"] is None:
        del f["notes"]
    m = store.finalize(arrays, f)
    LM.write(store.sidecar_path(path), m)
    return m


def verify(path, expect=None):
    """Check the table at ``path`` against its sidecar manifest; returns the manifest (LibraryError on a problem)."""
    side = store.sidecar_path(path)
    if not os.path.exists(side):
        raise store.LibraryError(path, [f"no sidecar manifest {side}"])
    m = json.loads(open(side).read())
    return store.verify(m, _load(path), where=str(path), expect=expect)
