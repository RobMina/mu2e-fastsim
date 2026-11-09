"""v2origin -- which deposit origin makes the reco pulses of a V2 arm, split by pulse PE (diagnostic for V2).

For each arm: CrvRecoTableDumper output (crvdump/charges, crvdump/pulses) plus the CrvStep table whose SimParticle keys
the charges point to (G4 crvsteps.*.root for the reference; the injected table for a candidate). Each SiPM charge is
classed by its particle's primary (census.primary_of): primary muon, muon descendant, non-muon, or none (no step:
noise). Pulses are taken on channels with exactly one sequence-0 pulse, so all of the channel's charge belongs to it.
Per sector group and pulse-PE bin: pulses per generated event, and the mean PE fraction of each origin; plus counts by
dominant origin (the class with the largest charge). Also the light yield per origin and group: all SiPM charge PE of
that origin over the visible energy of its CrvSteps (steps in G4-accepted events only on the reference side, so the
yield is the comparable number, not the totals). Comparing the arms locates a spectrum gap in one origin.

    v2origin FAST_RECO.root [...] --steps FAST_TABLE.root [...] --compare REF_RECO.root [...]
             --compare-steps G4_TABLE.root [...] --generated N [--compare-generated M]
"""
import awkward as ak
import numpy as np

from mu2efastsim import match, plots

from .. import geometry, tables
from ..reco import reftables
from .census import SECTOR_GROUP, primary_of

ORIGINS = ("mu", "desc", "nonmu", "none")
PE_BINS = np.array([5.0, 20.0, 50.0, 100.0, 150.0, 1e9])
GROUPS = ("EX", "T1", "T2", "M")
DEFAULT_COUNTERS = "Offline/Mu2eG4/geom/crv_counters_extracted_v04.txt"


def _origin_lut(step_paths):
    """int64 key (event key, particle key) -> origin index, from the particles tables."""
    parts = tables.read_many(step_paths, "particles", tables.KEY + ["key", "parent", "pdg"])
    p = primary_of(parts)
    mu = p.prim_pdg.abs().to_numpy() == tables.PRIMARY_MUON_PDG
    own = p.key.to_numpy() == p.prim_key.to_numpy()
    org = np.where(mu & own, 0, np.where(mu, 1, 2))
    k = np.stack([reftables.event_key(parts), parts.key.to_numpy().astype(np.int64)], 1)
    return k, org


def arm(reco_paths, step_paths, geom):
    """Per pulse on single-pulse channels: group index, PE, and PE fraction per origin -> dict of arrays."""
    q = reftables.read(reco_paths, "charges", ["run", "subrun", "event", "bar", "sipm", "pe", "sim"])
    ev = reftables.event_key(q)
    n = ak.to_numpy(ak.num(q.pe))
    qev = np.repeat(ev, n)
    qch = np.repeat(ak.to_numpy(q.bar).astype(np.int64) * 4 + ak.to_numpy(q.sipm), n)
    qpe = ak.to_numpy(ak.flatten(q.pe)).astype(float)
    qsim = ak.to_numpy(ak.flatten(q.sim)).astype(np.int64)
    lk, lorg = _origin_lut(step_paths)
    org = match.lookup(np.stack([qev, qsim], 1), lk, lorg, missing=3).astype(np.int64)
    org[qsim < 0] = 3
    # charge per (event, channel, origin)
    ck = np.stack([qev, qch], 1)
    u, inv = np.unique(ck, axis=0, return_inverse=True)
    inv = inv.ravel()
    per = np.zeros((len(u), len(ORIGINS)))
    np.add.at(per, (inv, org), qpe)
    p = reftables.read(reco_paths, "pulses", ["run", "subrun", "event", "bar", "sipm", "seq", "pe"])
    pev = reftables.event_key(p)
    pch = ak.to_numpy(p.bar).astype(np.int64) * 4 + ak.to_numpy(p.sipm)
    pk = np.stack([pev, pch], 1)
    uk, cnt = np.unique(pk, axis=0, return_counts=True)
    single = match.lookup(pk, uk, cnt, missing=0) == 1
    s0 = (ak.to_numpy(p.seq) == 0) & single
    pk, ppe = pk[s0], ak.to_numpy(p.pe)[s0].astype(float)
    ia, ib = match.join(pk, u)
    frac = np.zeros((len(pk), len(ORIGINS)))
    tot = per[ib].sum(1, keepdims=True)
    frac[ia] = per[ib] / np.maximum(tot, 1e-12)
    grp = SECTOR_GROUP[geom.sector[pk[:, 1] // 4]]
    # light yield: charge PE and step edep per (group, origin), over the events the reco saw
    seen = np.unique(qev)
    qg = SECTOR_GROUP[geom.sector[qch // 4]]
    st = tables.read_many(step_paths, "steps", tables.KEY + ["bar", "sp", "edep"])
    sev = reftables.event_key(st)
    keep = np.isin(sev, seen)
    sorg = match.lookup(np.stack([sev[keep], st.sp.to_numpy()[keep].astype(np.int64)], 1), lk, lorg, missing=3)
    sg = SECTOR_GROUP[geom.sector[st.bar.to_numpy()[keep]]]
    se = st.edep.to_numpy()[keep]
    yld = {g: {o: [float(qpe[(qg == g) & (org == j)].sum()), float(se[(sg == g) & (sorg == j)].sum())]
               for j, o in enumerate(ORIGINS[:3])} for g in GROUPS}
    return {"group": grp, "pe": ppe, "frac": frac, "matched": np.isin(np.arange(len(pk)), ia),
            "n_single": int(s0.sum()), "n_seq0": int((ak.to_numpy(p.seq) == 0).sum()), "yield": yld}


def summarise(A, nev):
    out = {"n_seq0": A["n_seq0"], "single_pulse_channel_frac": A["n_single"] / max(A["n_seq0"], 1),
           "charge_matched_frac": float(A["matched"].mean()) if len(A["matched"]) else None, "groups": {},
           "pe_per_MeV": {g: {o: v[0] / v[1] if v[1] else None for o, v in d.items()} for g, d in A["yield"].items()}}
    b = np.searchsorted(PE_BINS, A["pe"], "right") - 1
    dom = np.argmax(A["frac"], 1)
    for g in GROUPS:
        G = {}
        for i in range(len(PE_BINS) - 1):
            m = (A["group"] == g) & (b == i) & A["matched"]
            G[f"{PE_BINS[i]:.0f}-{PE_BINS[i + 1]:.0f}"] = {
                "pulses_per_1e6_gen": float(m.sum() / nev * 1e6),
                "mean_frac": {o: float(A["frac"][m, j].mean()) if m.any() else None for j, o in enumerate(ORIGINS)},
                "dominant_per_1e6_gen": {o: float((m & (dom == j)).sum() / nev * 1e6) for j, o in enumerate(ORIGINS)}}
        out["groups"][g] = G
    return out


def run(inputs, outdir, steps=None, compare=None, compare_steps=None, generated=None, compare_generated=None,
        counters=None, **kw):
    if not (steps and compare and compare_steps and generated):
        raise SystemExit("v2origin needs --steps, --compare, --compare-steps and --generated")
    geom = geometry.from_offline_config(geometry.find_counters(counters or DEFAULT_COUNTERS))
    fs = summarise(arm(inputs, steps, geom), int(generated))
    rs = summarise(arm(compare, compare_steps, geom), int(compare_generated or generated))
    plots.write_json(outdir, "v2origin_stats.json", {"fast": fs, "ref": rs})
    print(f"single-pulse channels: fast {fs['single_pulse_channel_frac']:.3f}, ref {rs['single_pulse_channel_frac']:.3f};"
          f" charge matched: fast {fs['charge_matched_frac']:.4f}, ref {rs['charge_matched_frac']:.4f}")
    print("light yield [PE/MeV of visible energy], ref -> fast (mu desc nonmu):")
    for g in GROUPS:
        print(f"  {g:3s} " + "  ".join(f"{o} {rs['pe_per_MeV'][g][o] or 0:.2f}->{fs['pe_per_MeV'][g][o] or 0:.2f}"
                                       for o in ORIGINS[:3]))
    for g in GROUPS:
        print(f"== {g}   per PE bin: pulses fast/ref | dominant-origin pulses fast/ref (mu desc nonmu none) | "
              f"mean PE fraction ref -> fast (mu desc nonmu)")
        for bn, F in fs["groups"][g].items():
            R = rs["groups"][g][bn]
            if not R["pulses_per_1e6_gen"]:
                continue
            dom = " ".join(f"{F['dominant_per_1e6_gen'][o] / max(R['dominant_per_1e6_gen'][o], 1e-9):.3f}"
                           if R["dominant_per_1e6_gen"][o] > 0 else "  -  " for o in ORIGINS)
            fr = " ".join(f"{R['mean_frac'][o]:.3f}->{F['mean_frac'][o]:.3f}" for o in ORIGINS[:3])
            print(f"  {bn:>9s} PE: {F['pulses_per_1e6_gen'] / R['pulses_per_1e6_gen']:.3f} "
                  f"(ref {R['pulses_per_1e6_gen']:.0f}/1e6) | {dom} | {fr}")
    return {"fast": fs, "ref": rs}
