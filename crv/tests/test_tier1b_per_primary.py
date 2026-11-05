"""Per-primary library identity for named run configurations (tier1b.build per_primary, sidecars, near-miss draws)."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from multiprimary_fixtures import _frames, _tiny_counters, _write
from crvfastsim import runconfig, tables, tier1b
from crvfastsim.analyses import tier1b as T1B
from valhist import descendant_templates, descendant_topology

KEY = tables.KEY


@pytest.fixture
def rc(tmp_path, monkeypatch):
    monkeypatch.setenv("CRVFS_NO_FETCH", "1")
    counters = _tiny_counters(tmp_path / "tiny-run1a-counters.txt")
    r = replace(runconfig.get("run1a_v01"), counters=str(counters.resolve()))
    monkeypatch.setitem(runconfig.REGISTRY, r.name, r)
    return r


def _two_muons(frames, samebar=True):
    """Event 1 gets a second primary muon (key 10, its own gen row) crossing bar 0 (or bar 1) with a delta ray (key 11)
    whose step lies on the first muon's bar: the historical (event, bar) build merges or misattaches it."""
    f = deepcopy(frames)
    p0 = f["particles"].iloc[[0]]
    f["particles"] = pd.concat([f["particles"], p0.assign(key=np.int64(10), genidx=np.int64(1)),
                                p0.assign(key=np.int64(11), parent=np.int64(10), parent_product_id=np.uint32(202),
                                          isprim=np.int32(0), pdg=np.int32(11), genidx=np.int64(0))],
                               ignore_index=True)
    s0 = f["steps"].iloc[[0]]
    mu2 = s0.assign(sp=np.int64(10), t0=5., edep=3.)
    if not samebar:
        mu2 = mu2.assign(bar=np.int64(1))
    f["steps"] = pd.concat([f["steps"], mu2, f["steps"].iloc[[1]].assign(sp=np.int64(11), edep=.7)],
                           ignore_index=True)
    g1 = f["gen"].iloc[[0]].assign(idx=np.int64(1), py=f["gen"].py.iloc[0] * 0.5, px=f["gen"].px.iloc[0] * 0.5 + 1.0)
    f["gen"] = pd.concat([f["gen"], g1], ignore_index=True)
    ev1 = f["events"].event == 1
    f["events"].loc[ev1, ["n_particles", "n_steps"]] = 4
    f["events"].loc[ev1, "n_gen"] = 2
    return f


def _donor(tmp_path, rc, samebar=True):
    donor, full = _frames(rc.geometry())
    donor = _two_muons(donor, samebar)
    full = deepcopy(full)
    full["gen"] = pd.concat([full["gen"], donor["gen"].iloc[[2]]], ignore_index=True)
    full["events"].loc[full["events"].event == 1, "n_gen"] = 2
    return (str(_write(tmp_path / "crvsteps.001400_00000101.root", donor)),
            str(_write(tmp_path / "gen.001400_00000101.root", full)))


@pytest.mark.parametrize("samebar", [True, False])
def test_crossings_per_primary_and_own_primary_attachment(tmp_path, rc, samebar):
    steps, gen = _donor(tmp_path, rc, samebar)
    geom = rc.geometry()
    old, _ = tier1b.build([steps], [gen], geom, roots=True)
    new, info = tier1b.build([steps], [gen], geom, roots=True, per_primary=True)
    assert len(new["libxing"]) == 2
    assert len(old["libxing"]) == (1 if samebar else 2)
    # every descendant row sits on a crossing of its own primary (roots: the step-tree root's primary)
    x = tier1b._crossings(T1B.tier1b.classify(tables.read(steps, "steps"), tables.read(steps, "particles"),
                                                 geom.sector).query("cls == 'mu_primary'"), per_primary=True)
    own = {8: 7, 11: 10}
    for sp, xid in zip(new["libdesc"].sp, new["libdesc"].xid):
        assert x.sp.iloc[xid] == own[sp]
    assert info["identity"]["overall"]["parentless"]["steps"] == 0
    assert info["desc_steps_unattached"] == 0


def test_parentless_step_goes_to_the_unattached_bin(tmp_path, rc):
    donor, full = _frames(rc.geometry())
    p0 = donor["particles"].iloc[[1]]
    donor["particles"] = pd.concat([donor["particles"], p0.assign(key=np.int64(20), parent=np.int64(99))],
                                   ignore_index=True)
    donor["steps"] = pd.concat([donor["steps"], donor["steps"].iloc[[1]].assign(sp=np.int64(20), edep=.4)],
                               ignore_index=True)
    donor["events"].loc[donor["events"].event == 1, ["n_particles", "n_steps"]] = 3
    steps = str(_write(tmp_path / "crvsteps.001400_00000101.root", donor))
    gen = str(_write(tmp_path / "gen.001400_00000101.root", full))
    with pytest.raises(ValueError, match="missing referenced identity"):   # the strict donor gate refuses it
        tier1b.build([steps], [gen], rc.geometry(), per_primary=True)
    geom = rc.geometry()
    s = tier1b.classify(pd.DataFrame(donor["steps"]), pd.DataFrame(donor["particles"]), geom.sector)
    c = tier1b._identity_summary(tier1b._identity_census(s, s.iloc[:0], geom, pd.DataFrame(donor["particles"])))
    c = c["overall"]
    assert c["parentless"]["steps"] == 1 and c["parentless"]["edep"] == pytest.approx(.4)
    assert c["parentless"]["edep_fraction"] == pytest.approx(.4 / 3.1)


def test_noncrossing_primary_descendants_are_nearmiss_not_attached(tmp_path, rc):
    donor, full = _frames(rc.geometry())
    donor = _two_muons(donor)
    donor["steps"] = donor["steps"][donor["steps"].sp != 10].reset_index(drop=True)   # muon 10 never crosses
    donor["events"].loc[donor["events"].event == 1, "n_steps"] = 3
    full = deepcopy(full)
    full["gen"] = pd.concat([full["gen"], donor["gen"].iloc[[2]]], ignore_index=True)
    full["events"].loc[full["events"].event == 1, "n_gen"] = 2
    steps = str(_write(tmp_path / "crvsteps.001400_00000101.root", donor))
    gen = str(_write(tmp_path / "gen.001400_00000101.root", full))
    geom = rc.geometry()
    old, _ = tier1b.build([steps], [gen], geom)
    new, info = tier1b.build([steps], [gen], geom, per_primary=True)
    assert len(old["libdesc"]) == 2                     # historical: muon 10's delta ray joins muon 7's crossing
    assert len(new["libdesc"]) == 1 and info["desc_steps_unattached"] == 1
    assert info["identity"]["overall"]["no_own_crossing"]["edep"] == pytest.approx(.7)
    # the template census: muon 10's generated row has no crossing of its own (exact gen join), muon 7's does
    s = tier1b.classify(tables.read(steps, "steps"), tables.read(steps, "particles"), geom.sector)
    c_old, _, n = descendant_templates.census_pass(str(steps), str(gen), geom)
    c_new, _, _ = descendant_templates.census_pass(str(steps), str(gen), geom, per_primary=True)
    assert n == 3 and len(c_old) == 1 and len(c_new) == 2
    rows = descendant_templates.crossed_generator_rows(str(steps), s, tier1b.muon_rays(tables.read(gen, "gen")))
    assert rows.tolist() == [0]


def test_named_build_sidecars_and_per_primary_sampling(tmp_path, rc):
    steps, gen = _donor(tmp_path, rc)
    out = tmp_path / "lib" / "library.root"
    out.parent.mkdir()
    T1B.build([str(steps)], str(tmp_path), gen=[str(gen)], out=str(out), roots=True, run_config=rc.name,
              generated=6)
    man = json.loads(out.with_suffix(".json").read_text())
    assert man["per_primary"] is True and man["run_config"] == rc.name
    descendant_topology.sidecar(out.with_suffix(".json"), out.parent / "topology.npz")
    descendant_templates.sidecar(out.with_suffix(".json"), [str(tmp_path / "crvsteps.{s}.root")],
                                 [str(tmp_path / "gen.{s}.root")], [[101]], out.parent / "templates_v3.npz")
    geom = rc.geometry()
    lib = tier1b.read_library(out)
    lib = tier1b.attach_topology(lib, *T1B.read_topology_sidecar(out.parent / "topology.npz", out), geom)
    lib = tier1b.attach_templates(lib, T1B.read_template_sidecar(out.parent / "templates_v3.npz", out), geom)
    lib = tier1b.attach_anchors(lib, geom)
    st, pa = tables.read(steps, "steps"), tables.read(steps, "particles")
    s = tier1b.classify(st, pa, geom.sector)
    fast = st[s.cls.eq("mu_primary").to_numpy()].reset_index(drop=True)
    fparts = pa[pa.isprim.eq(1) & pa.pdg.abs().eq(13)].reset_index(drop=True)
    fast = fast.assign(sp=np.where(fast.sp == 10, 2, 1))
    fparts = fparts.assign(key=np.where(fparts.key == 10, 2, 1))
    opts = dict(context=True, contain=True, point=True, topology=True, unvisited=True, nearmiss=True, chord=True,
                anchor=True)
    g = tables.read(gen, "gen")
    fast = fast[fast.sp != 2].reset_index(drop=True)    # event 1's second muon misses the CRV in the fast arm
    for per, n in ((False, 1), (True, 2)):
        out_steps, _, info = tier1b.sample(fast, fparts, g, lib, geom, np.random.default_rng(1), per_primary=per,
                                           **opts)
        assert len(out_steps) >= len(fast)
        # event 5's muon; per primary also event 1's uncrossed second muon (its event has a crossing)
        assert info["templates"]["nearmiss"]["candidates"] == n


def test_nearmiss_candidates_per_muon():
    gen = pd.DataFrame({"run": 1, "subrun": 1, "event": [1, 1, 2], "idx": [0, 1, 0], "pdg": [13, -13, 13],
                        "x": 0., "y": 0., "z": 0., "px": 0., "py": -1., "pz": 0., "e": [500., 600., 700.], "t": 0.})
    fast = pd.DataFrame({"run": [1], "subrun": [1], "event": [1], "sp": [1], "bar": [0]})
    empty = lambda cols: pd.DataFrame({c: np.zeros(0) for c in cols})
    lib = {"nmreach": empty(["cell", "prob"]), "nmtmpl": empty(["tid", "cell"]),
           "nmsteps": empty(["tid", "sp", "bar", "pdg", "dt0", "dt1"] + tier1b.UV_STEP_COLS
                            + ["px0", "py0", "pz0", "p1", "len", "edep"])}

    class G:
        nbars = 2
        center = np.array([[0., -1000., 0.], [0., -1000., 100.]])
        halflen = np.array([[10., 10., 10.]] * 2)
        sector = np.zeros(2, np.int64)
        lib_group = np.array(["EX"], dtype=object)

    for per, n in ((False, 1), (True, 2)):
        _, info = tier1b._nearmiss_for_gen(fast, gen, lib, G, np.random.default_rng(1), per_primary=per)
        assert info["candidates"] == n and info["reached"] == 0


def test_group_boxes_with_empty_groups(rc):
    geom = rc.geometry()
    lo, hi = tier1b.group_boxes(geom)
    assert np.all(lo[0] <= hi[0]) and np.all(lo[1:] > hi[1:])
    g, dist, _ = tier1b.closest_approach(np.array([[0., 5000., 0.]]), np.array([[0., -1., 0.]]), geom)
    assert g.tolist() == [0] and np.isfinite(dist).all()


def test_apply_takes_per_primary_from_the_library_manifest(tmp_path):
    lib = tmp_path / "library.root"
    assert T1B.library_per_primary(lib) is False                     # no manifest
    (tmp_path / "library.json").write_text(json.dumps({"run_config": "run1a_v01"}))
    assert T1B.library_per_primary(lib) is False                     # a default build on a named geometry
    (tmp_path / "library.json").write_text(json.dumps({"run_config": "run1a_v01", "per_primary": True}))
    assert T1B.library_per_primary(lib) is True
