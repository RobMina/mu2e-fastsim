"""Build the thin-target interaction bank (`ptgt_device` library file) from `beam/g4/thin_target` outputs (opt-in).

A campaign directory holds `grid.json` ({"log10_ke_edges": [...], "materials": [...], "physics_list": ...,
"geant4": ...}) and, per projectile species and material, the generator outputs of bin ranges starting at bin B:

    xs_<species>_<material>_b<B>.txt          macroscopic cross sections, stopping power, radiation length
    inel_<species>_<material>_b<B>_s<seed>.bin single inelastic interactions (projectile frame)
    el_<species>_<material>_b<B>_s<seed>.bin   elastic cos(theta)

Species names are the Geant4 ones (proton neutron pi+ pi- kaon+ kaon- mu+ mu-). Muons have xs files only.

    python3 -m beamfastsim.ptgt_bank DIR --out BANK.npz --commit SHA [--xs-only]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from beamfastsim import ptgt_device as P
from mu2efastsim import library as L

G4NAME = {2212: "proton", 2112: "neutron", 211: "pi+", -211: "pi-", 321: "kaon+", -321: "kaon-", 13: "mu-", -13: "mu+"}
HADRONS = (2212, 2112, 211, -211, 321, -321)


def read_inel(path):
    """Records of one inel file: (bin, ke, edep, escaping) per event and its secondaries (pdg, p)."""
    raw = np.fromfile(path, dtype="<i4")
    f = raw.view("<f4")
    starts, pos = [], 0
    n = raw.size
    while pos < n:
        starts.append(pos)
        pos += 5 + 4 * int(raw[pos + 1])
    if pos != n:
        raise ValueError(f"{path}: truncated record")
    s = np.asarray(starts, dtype=np.int64)
    nsec = raw[s + 1].astype(np.int64)
    rows = np.repeat(s + 5, nsec) + 4 * (np.arange(nsec.sum()) - np.repeat(np.cumsum(nsec) - nsec, nsec))
    return dict(bin=raw[s].astype(np.int64), ke=f[s + 2].astype(np.float64), edep=f[s + 3].astype(np.float64),
                escaping=f[s + 4].astype(np.float64), nsec=nsec, sec_pdg=raw[rows].astype(np.int32),
                sec_p=np.stack([f[rows + 1], f[rows + 2], f[rows + 3]], axis=1))


def read_el(path):
    raw = np.fromfile(path, dtype="<i4").reshape(-1, 3)
    return raw[:, 0].astype(np.int64), raw[:, 2].view("<f4").astype(np.float64)


def read_xs(path):
    a = np.loadtxt(path, comments="#", ndmin=2)
    return a[:, 0].astype(np.int64), a[:, 2:6]


def _files(d, kind, species, material):
    pat = re.compile(rf"{kind}_{re.escape(species)}_{re.escape(material)}_b(\d+)(?:_s(\d+))?\.(bin|txt)$")
    out = []
    for p in sorted(glob.glob(os.path.join(d, f"{kind}_*"))):
        m = pat.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return out


def build(d, xs_only=False):
    """Bank of one campaign directory. xs_only: macroscopic tables only (a passive material crossed as a thin slab,
    where an inelastic interaction absorbs the lane): no inel / el files are read, sigma_inel is kept."""
    grid = json.load(open(os.path.join(d, "grid.json")))
    edges = np.asarray(grid["log10_ke_edges"], float)
    mats = tuple(grid["materials"])
    nk = edges.size - 1
    shape = (len(P.SPECIES), len(mats), nk)
    cells = {k: np.zeros(shape) for k in P.CELL_ARRAYS}
    filled = np.zeros(shape, bool)
    el_samples = {}
    tc, edep, nsec, pdg, mom = [], [], [], [], []
    stats = {}
    for si, sp in enumerate(P.SPECIES):
        name = G4NAME[sp]
        for mi, mat in enumerate(mats):
            for b0, path in _files(d, "xs", name, mat):
                b, v = read_xs(path)
                k = b0 + b
                for j, key in enumerate(("sigma_inel", "sigma_el", "dedx", "x0")):
                    cells[key][si, mi, k] = v[:, j]
                filled[si, mi, k] = True
            if sp not in HADRONS or xs_only:
                continue
            for b0, path in _files(d, "inel", name, mat):
                r = read_inel(path)
                flat = (si * len(mats) + mi) * nk + b0 + r["bin"]
                tc.append(flat); edep.append(r["edep"]); nsec.append(r["nsec"])
                pdg.append(r["sec_pdg"]); mom.append(r["sec_p"])
                st = stats.setdefault(f"{name}/{mat}", dict(interactions=0, escaping_mev=0.0))
                st["interactions"] += int(r["bin"].size)
                st["escaping_mev"] += float(r["escaping"].sum())
            for b0, path in _files(d, "el", name, mat):
                b, c = read_el(path)
                for k in np.unique(b):
                    el_samples.setdefault((si, mi, b0 + int(k)), []).append(c[b == k])
    if not filled.all():
        missing = [(G4NAME[P.SPECIES[i]], mats[j], int(k)) for i, j, k in zip(*np.nonzero(~filled))]
        raise ValueError(f"cells without cross sections: {missing[:5]} ... ({len(missing)})")
    charged = np.asarray([sp != 2112 for sp in P.SPECIES])
    if np.any(cells["dedx"][charged] <= 0):
        raise ValueError("charged species with zero stopping power: xs files made with the EM processes inactive?")
    icdf = np.ones(shape + (P.N_EL,))
    u = np.linspace(0.0, 1.0, P.N_EL)
    for key, chunks in el_samples.items():
        c = np.clip(np.concatenate(chunks), -1.0, 1.0)
        icdf[key] = np.maximum.accumulate(np.quantile(c, u))
    if xs_only:
        empty = dict(trial_cell=np.zeros(0, np.int64), sec_offsets=np.zeros(1, np.int64), edep_local=np.zeros(0))
        return dict(edges=edges, materials=mats, cells=dict(cells, el_cos_icdf=icdf), trials=empty,
                    secondaries=dict(sec_pdg=np.zeros(0, np.int32), sec_p=np.zeros((0, 3))), stats=dict(xs_only=True),
                    grid=grid, el_cells=0)
    if not tc:
        raise ValueError(f"{d}: no inel files")
    tc, edep, nsec = np.concatenate(tc), np.concatenate(edep), np.concatenate(nsec)
    # hadron cells without stored interactions (pi+ below the Coulomb barrier, bins below the nucleon cut): sigma_inel
    # = 0 there, so a lane never draws from an empty cell; counted in the build statistics
    have = np.bincount(tc, minlength=int(np.prod(shape))).reshape(shape) > 0
    had = np.isin(np.asarray(P.SPECIES), HADRONS)[:, None, None]
    zero = had & ~have & (cells["sigma_inel"] > 0)
    cells["sigma_inel"][zero] = 0.0
    stats["_cells_zeroed_no_interactions"] = [(G4NAME[P.SPECIES[i]], mats[j], int(k)) for i, j, k in zip(*np.nonzero(zero))]
    pdg, mom = np.concatenate(pdg), np.concatenate(mom)
    off = np.concatenate([[0], np.cumsum(nsec)])
    order = np.argsort(tc, kind="stable")                      # trials sorted by cell, secondaries carried along
    n_new = nsec[order]
    rows = np.repeat(off[:-1][order], n_new) + (np.arange(n_new.sum()) - np.repeat(np.cumsum(n_new) - n_new, n_new))
    trials = dict(trial_cell=tc[order], sec_offsets=np.concatenate([[0], np.cumsum(n_new)]), edep_local=edep[order])
    secs = dict(sec_pdg=pdg[rows], sec_p=mom[rows])
    return dict(edges=edges, materials=mats, cells=dict(cells, el_cos_icdf=icdf), trials=trials, secondaries=secs,
                stats=stats, grid=grid, el_cells=len(el_samples))


def manifest_fields(d, grid, xs_only, release, split, name_prefix="beam.thin_target"):
    """Complete manifest fields of a bank built from campaign directory d (nothing unrecorded): the source files, the
    generator, the source seeds (from the file names; the build itself draws no random numbers), and config_hash =
    digest of the campaign configuration (grid.json and the build mode). release: {repository, commit, entry_point}
    of the code that built it; split: {training, held_out, rule}."""
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*_b*")))
    seeds = sorted({int(m.group(1)) for f in files for m in [re.search(r"_s(\d+)\.bin$", f)] if m})
    lst = grid.get("physics_list", "QGSP_BERT_HP")
    config = dict(grid=grid, xs_only=bool(xs_only))
    return dict(name=f"{name_prefix}.{lst}{'.xs' if xs_only else ''}",
                source=dict(samples=files, description="thin_target generator outputs (Geant4 single interactions)"),
                generator=dict(name="thin_target", release=grid.get("geant4", "unrecorded"), physics_list=lst,
                               configuration=json.dumps({k: v for k, v in grid.items() if k != "log10_ke_edges"})),
                release=dict(release),
                seeds=dict(source=seeds, rule="one Geant4 seed per generator output file (_s<seed>)",
                           build="deterministic: the builder draws no random numbers"),
                config_hash=L.hashing.digest(config), split=dict(split), unrecorded={})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True, help="commit of the code that built the bank")
    ap.add_argument("--repository", default="mu2e-fastsim")
    ap.add_argument("--xs-only", action="store_true", help="macroscopic tables only (passive slab material)")
    a = ap.parse_args(argv)
    b = build(a.dir, xs_only=a.xs_only)
    fields = manifest_fields(a.dir, b["grid"], a.xs_only,
                             release=dict(repository=a.repository, commit=a.commit, entry_point="beam/g4/thin_target"),
                             split=dict(training="all", held_out="none",
                                        rule="no held-out bank sample: validated downstream against full-simulation "
                                             "references (the chain's benchmark)"))
    m = P.save_bank(a.out, b["materials"], b["edges"], b["cells"], b["trials"], b["secondaries"], fields)
    print(json.dumps(dict(out=a.out, content_hash=m.get("content_hash"), trials=int(b["trials"]["trial_cell"].size),
                          secondaries=int(b["secondaries"]["sec_pdg"].size), el_cells=b["el_cells"],
                          stats=b["stats"]), indent=1))


if __name__ == "__main__":
    main()
