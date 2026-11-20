"""crvfastsim.lookupgen: Geant4 property-vector semantics, geometry and binning, the v6 writer, the device transport
(conservation, both dtypes, the exact helical shortcut), the core/adapter split (the core imports nothing
experiment-specific and runs as a package of its own), the library-core manifests, and, when a released table is
available (CRVFS_LOOKUP_DIR), the table header against it byte for byte."""
import ast
import os
import struct

import numpy as np
import pytest

from crvfastsim.lookupgen.core import materials as M
from crvfastsim.lookupgen.core import optics as O
from crvfastsim.lookupgen.mu2e import sections as G
from crvfastsim.lookupgen.mu2e import writer as W

CORE = os.path.join(os.path.dirname(M.__file__))


# ------------------------------------------------------------------------------------------------ materials
def test_g4vector_sorts_on_insertion_like_geant4():
    v = O.G4Vector([1.0, 3.0, 2.0, 2.0], [10.0, 30.0, 20.0, 21.0])
    assert list(v.e) == [1.0, 2.0, 2.0, 3.0]
    assert list(v.v) == [10.0, 21.0, 20.0, 30.0]      # a later equal energy is inserted before the earlier one


def test_psppo_out_of_order_energy_is_moved():
    e = M.ABS_PSPPO.e
    assert np.all(np.diff(e) >= 0)
    i = np.nonzero(e == 2.32)[0]
    assert len(i) == 2 and M.ABS_PSPPO.v[i[0]] == 290.0 and M.ABS_PSPPO.v[i[1]] == 3960.0
    # 2.30 eV interpolates towards the moved point (29 cm) as Geant4 does
    assert M.ABS_PSPPO.value(2.30) == pytest.approx(0.5 * (4210.0 + 290.0))


def test_value_clamps_and_interpolates():
    v = O.G4Vector([2.0, 3.0, 4.0], [1.0, 3.0, 2.0])
    assert v.value(1.0) == 1.0 and v.value(5.0) == 2.0
    assert v.value(2.5) == pytest.approx(2.0) and v.value(3.5) == pytest.approx(2.5)
    cum = O.G4Vector([2.0, 3.0, 4.0], [1.0, 3.0, 5.0])        # GetEnergy inverts a non-decreasing integral
    assert cum.get_energy(2.0) == pytest.approx(2.5) and cum.get_energy(0.5) == 2.0 and cum.get_energy(9.0) == 4.0


def test_group_velocity_is_below_phase_velocity_in_polystyrene():
    for e in (2.4, 2.9, 3.3):
        vg, n = M.VG_PS.value(e), M.RINDEX_PS.value(e)
        assert vg < O.C_LIGHT / n
        assert 1.6 < O.C_LIGHT / vg < 1.9
    assert O.VG_AIR.value(2.5) == pytest.approx(O.C_LIGHT)


def test_material_tables_have_stock_lengths():
    assert (len(M.RINDEX_PS), len(M.ABS_Y11), len(M.EM_Y11), len(M.EM_PPO), len(M.ABS_PSPPO), len(M.EM_POPOP),
            len(M.ABS_POPOP), len(M.RINDEX_PMMA)) == (61, 325, 43, 70, 45, 63, 26, 48)
    assert M.ABS_Y11.value(2.3) > 1000.0 and M.ABS_Y11.value(3.0) < 2.0     # m-scale below 2.53 eV, mm above
    assert M.REFL_TIO2.value(2.5) == pytest.approx(1 - 0.75 * 0.09)
    v6, cur = O.get("v6"), O.get("current")
    assert v6.refl_tio2.value(2.5) == pytest.approx(1 - 0.9 * 0.09)
    assert (len(v6.em_ppo), len(v6.em_popop), v6.tau_y11, v6.refl_mppc, v6.fiber_end) == (100, 58, 8.7, 0.30, "contact")
    assert (cur.fiber_end, cur.rough_fiber_air, v6.rough_fiber_air) == ("gap", True, False)
    assert v6.abs_y11.value(2.3) / cur.abs_y11.value(2.3) == pytest.approx(0.80 / 0.85)


def test_facet_alpha_distribution():
    a = O.facet_alpha_icdf(0.02)
    assert a[0] == 0 and np.all(np.diff(a) >= 0) and 0.02 < np.median(a) < 0.04
    b = O.facet_alpha_icdf(1.5)
    assert b[-1] <= np.pi / 2 + 1e-12


def test_cerenkov_map_shape():
    ck = O.cerenkov_map(M.RINDEX_PS)
    assert ck[0, 1] == 0 and ck[-1, 0] == 1.0 and np.all(np.diff(ck[:, 1]) >= 0)
    assert 250 < ck[-1, 1] < 280                            # photons/mm at beta = 1


# ------------------------------------------------------------------------------------------------ geometry
def test_stock_counter_and_bins():
    c = G.stock(3200, 0)
    assert G.is_stock(c) and (c.hx, c.hy, c.rc) == pytest.approx((9.64, 25.42, 1.75))
    assert c.r_core == pytest.approx(0.616) and c.r_clad1 == pytest.approx(0.658)
    e = G.edges(c)
    assert (len(e["x"]), len(e["y"]), len(e["z"])) == (11, 42, 57)
    assert G.n_bins(c, 0) == 10 * 41 * 56 and G.n_bins(c, 1) == 5 * G.n_bins(c, 0)
    assert G.n_bins(c, 2) == 56 * 5 * 5 * 11 * 5             # phi: 5 bins (0, pi/8, ..., 7pi/8, pi)
    assert len(G.z_edges(G.stock(4550))) - 1 == 70         # lrint(39.5) = 40 main bins


@pytest.mark.parametrize("t,w", [(24.71, 25.0), (21.99, 35.93), (10.0, 25.0)])
def test_design_counters_get_valid_edges(t, w):
    c = G.design(t, w, 3000.0, 1)
    assert not G.is_stock(c)
    assert c.fiber_separation == pytest.approx(w * 26 / 51.34)
    assert (c.hole_rx, c.hole_ry, c.r_clad2) == pytest.approx((2.3, 1.3, 1.0))
    e = G.edges(c)
    for k in ("x", "y"):
        assert np.all(np.diff(e[k]) > 0)
    assert e["x"][-1] == pytest.approx(t / 2) and e["y"][-1] == pytest.approx(w / 2)
    assert np.allclose(e["y"], -e["y"][::-1])
    yc = c.fiber_separation / 2
    assert np.any(np.isclose(e["y"], yc - 2.5)) and np.any(np.isclose(e["y"], yc + 2.5))


def test_bin_boxes_follow_the_stock_index_order():
    c = G.stock(3200)
    e = G.edges(c)
    nz, ny = len(e["z"]) - 1, len(e["y"]) - 1
    b = G.bin_boxes(c, 0, [3 * nz * ny + 7 * nz + 11])
    assert b["x"][0][0] == e["x"][3] and b["y"][0][0] == e["y"][7] and b["z"][0][0] == e["z"][11]


# ------------------------------------------------------------------------------------------------ writer
def _counts(ht, he, nall=None, hta=None, hea=None):
    return {"nall": int(ht.sum()) if nall is None else nall, "hta": ht if hta is None else hta,
            "hea": he if hea is None else hea, "ht": ht, "he": he}


def test_bin_record_encoding_matches_stock_rounding():
    ht = np.zeros(150, np.int64)
    ht[[3, 4, 10]] = [1, 2, 3]
    he = np.zeros(10, np.int64)
    he[[1, 2]] = [5, 1]
    rec = W.bin_record(7, 1000, _counts(ht, he), "current")
    num, p = struct.unpack_from("<If", rec)
    assert num == 7 and p == np.float32(6) / np.float32(1000)
    nd = struct.unpack_from("<Q", rec, 8)[0]
    d = np.frombuffer(rec, np.uint8, nd, 16)
    assert nd == 11 and d[3] == int(255 * np.float32(1 / 6) + 0.5) and d[10] == 128
    zero = _counts(np.zeros(150, int), np.zeros(10, int))
    assert W.bin_record(0, 100, zero, "current") == struct.pack("<IfQQ", 0, 0.0, 0, 0)
    assert W.bin_record(0, 100, zero, "v6") == struct.pack("<IfQQ", 0, 0.0, 0, 0)


def test_v6_convention_counts_late_arrivals():
    """January 2019 writer: probability over all arrivals, histograms normalised to all arrivals."""
    ht = np.zeros(150, np.int64)
    ht[5] = 2
    he = np.zeros(10, np.int64)
    he[1] = 2
    hea = he.copy()
    hea[1] = 3                                   # the late arrival is in the emission histogram, not the time one
    rec = W.bin_record(1, 100, _counts(ht, he, nall=3, hta=ht, hea=hea), "v6")
    _, p = struct.unpack_from("<If", rec)
    assert p == np.float32(3) / np.float32(100)
    d = np.frombuffer(rec, np.uint8, 6, 16)
    assert d[5] == int(np.float32(255) * (np.float32(2) / np.float32(3)) + 0.5)
    only_late = W.bin_record(2, 100, _counts(np.zeros(150, int), np.zeros(10, int), nall=1,
                                             hea=np.eye(10, dtype=int)[1]), "v6")
    assert struct.unpack_from("<Q", only_late, 8)[0] == 1       # one (zero) entry, as the stock writer leaves it


def test_header_layout():
    h = W.header_bytes(G.stock(2370, 1))
    v1, v2, refl = struct.unpack_from("<3i", h)
    assert (v1, v2, refl) == (6, 0, 1)
    c = struct.unpack_from("<10d", h, 16)
    assert c[:8] == pytest.approx((9.64, 25.42, 1185.0, 26.0, 2.0, 1.0, 0.7, 1.75))


def _ref_dir():
    d = os.environ.get("CRVFS_LOOKUP_DIR")
    if not d or not os.path.exists(os.path.join(d, "LookupTable_3200_0")):
        pytest.skip("no released lookup table (CRVFS_LOOKUP_DIR)")
    return d


def test_header_identical_to_released_table():
    d = _ref_dir()
    b = open(os.path.join(d, "LookupTable_3200_0"), "rb").read()
    h = W.header_bytes(G.stock(3200, 0))
    assert b[:12] == h[:12] and b[16:96] == h[16:96]             # bytes 12-16: uninitialised struct padding
    n = struct.unpack_from("<Q", h, 96)[0]
    end_ck = 96 + 2 * (8 + 16 * n)
    assert b[end_ck:len(h)] == h[end_ck:]                        # bin definitions
    for o in (104, 104 + 8 + 16 * n):                            # Cerenkov maps: equal to the last few ulps
        assert np.allclose(np.frombuffer(b, "<f8", 2 * n, o), np.frombuffer(h, "<f8", 2 * n, o), rtol=1e-13, atol=0)


# ------------------------------------------------------------------------------------------------ transport
def _gen(simtype, bins, nphot, dtype="float64", helix=True, length=1045, reflector=1, seed=3, model="v6"):
    pytest.importorskip("jax")
    from crvfastsim.lookupgen.core.run import generate
    c = G.stock(length, reflector)
    return generate(c, G.binning(c), simtype, np.asarray(bins), nphot, dtype=dtype, seed=seed, lanes=512,
                    chunk_iters=400, helix=helix, model=model)


@pytest.mark.parametrize("dtype,model", [("float64", "v6"), ("float32", "v6"), ("float32", "current")])
def test_transport_runs_and_conserves(dtype, model):
    r = _gen(0, [100, 2000], 600, dtype, model=model)
    s = r["stats"]
    assert s["q"] == 1200 and s["capped"] == 0 and s["stuck"] == 0 and s["badsrc"] == 0
    assert np.all(r["gen"] <= 600) and r["gen"].sum() > 0
    assert np.all(r["ht"].sum(1) == r["he"].sum(1))
    assert np.all(r["ht"].sum(1) <= r["nall"]) and np.all(r["nall"] <= r["gen"])
    assert np.all(r["hta"].sum(1) <= r["nall"]) and np.all(r["hea"].sum(1) <= r["nall"])
    # chain-length histograms: one entry per generated photon, one per SiPM-0 arrival
    assert len(s["len_end"]) == len(s["len_arr"]) == 19
    assert sum(s["len_end"]) == r["gen"].sum() and sum(s["len_arr"]) == r["nall"].sum()
    # the device the kernel ran on is recorded (the manifests name it per piece)
    import jax
    assert s["device"] == jax.devices()[0].device_kind and s["platform"] == jax.devices()[0].platform


def test_bins_outside_the_scintillator_generate_nothing():
    c = G.stock(1045)
    e = G.edges(c)
    nz, ny = len(e["z"]) - 1, len(e["y"]) - 1
    yb = int(np.searchsorted(e["y"], -13.0)) - 1           # x in [0, 0.5], y next to the fiber centre: in the hole
    r = _gen(0, [0 * nz * ny + yb * nz + 5], 200)
    assert r["gen"][0] == 0 and r["ht"].sum() == 0


def test_fiber_cerenkov_generates_every_photon():
    r = _gen(2, [0, 50, 3000], 300)
    assert np.all(r["gen"] == 300)


@pytest.mark.slow
def test_helix_shortcut_matches_bounce_by_bounce():
    """Same seeds with and without the shortcut: arrival fraction and time/emission distributions agree."""
    bins = [40, 900, 2200]
    a = _gen(0, bins, 4000, helix=True, length=2370)
    b = _gen(0, bins, 4000, helix=False, length=2370)
    na, nb = a["ht"].sum(), b["ht"].sum()
    assert a["stats"]["helix"] > 0 and b["stats"]["helix"] == 0
    assert abs(na - nb) < 4 * np.sqrt(na + nb)
    ta = (a["ht"].sum(0) * np.arange(150)).sum() / na
    tb = (b["ht"].sum(0) * np.arange(150)).sum() / nb
    assert abs(ta - tb) < 1.0


def test_cli_resume_reruns_only_incomplete_bins_with_the_same_streams(tmp_path):
    pytest.importorskip("jax")
    from crvfastsim.lookupgen.__main__ import main
    base = ["run", "--length", "1045", "--reflector", "1", "--simtype", "0", "--bins", "100,2000,2100",
            "--photons", "300", "--dtype", "float32", "--lanes", "512", "--chunk-iters", "400", "--checkpoint"]
    full, part = str(tmp_path / "full.npz"), str(tmp_path / "part.npz")
    assert main(base + ["--out", full]) == 0
    z = dict(np.load(full))
    i = int(np.nonzero(z["bins"] == 2000)[0][0])
    z["complete"][i] = False
    for k in W.KEYS[1:]:
        z[k][i] = 0
    np.savez(part, **z)
    assert main(base + ["--out", part, "--resume"]) == 0
    a, b = np.load(full), np.load(part)
    assert b["complete"].all() and sorted(b["bins"]) == sorted(a["bins"])
    oa, ob = np.argsort(a["bins"]), np.argsort(b["bins"])
    for k in W.KEYS:
        assert np.array_equal(a[k][oa], b[k][ob]), k
    assert main(base + ["--out", part, "--resume"]) == 0      # nothing left: no rerun


def test_lower_step_limit_stops_and_counts_long_chains():
    pytest.importorskip("jax")
    from crvfastsim.lookupgen.core.run import generate
    c = G.stock(1045, 1)
    r = generate(c, G.binning(c), 0, np.array([100, 2000]), 300, dtype="float32", seed=3, lanes=512,
                 chunk_iters=400, max_steps=64)
    s = r["stats"]
    assert s["capped"] > 0 and s["max_steps"] == 64 and s["seed"] == 3
    assert sum(s["len_end"]) == r["gen"].sum() and sum(s["len_end"][6:]) == 0      # no chain reaches 64 steps
    with pytest.raises(ValueError):
        generate(c, G.binning(c), 0, np.array([100]), 10, max_steps=(1 << 19) + 1)


def test_assemble_writes_the_table_and_its_provenance(tmp_path):
    import json
    from crvfastsim.lookupgen.__main__ import main
    c = G.stock(1045, 1)
    pieces = []
    for st in (0, 1, 2):
        n = G.n_bins(c, st)
        f = str(tmp_path / f"s{st}.npz")
        z = {k: np.zeros(n, np.int32) for k in ("gen", "nall")}
        z.update(bins=np.arange(n), ht=np.zeros((n, 150), np.int32), he=np.zeros((n, 10), np.int32),
                 hta=np.zeros((n, 150), np.int32), hea=np.zeros((n, 10), np.int32))
        stats = {"dtype": "float32", "model": "v6", "max_steps": 65536, "capped": 2, "device": "gpu B",
                 "resumed_from": {"dtype": "float32", "max_steps": 1 << 19, "capped": 0, "device": "gpu A"}}
        np.savez(f, **z, complete=np.ones(n, bool), nphot=10000 if st == 2 else 100000, simtype=st, model="v6",
                 stats=json.dumps(stats), counter=json.dumps(c.__dict__))
        pieces.append(f)
    out = str(tmp_path / "LookupTable_1045_1")
    assert main(["assemble", "--length", "1045", "--reflector", "1", "--section", "stock", "--fiber-diameter", "1.4",
                 "--out", out, "--model", "v6", "--note", "scintillator Cerenkov at 2e4 photons per bin"] + pieces) == 0
    assert os.path.getsize(out) == len(W.table_bytes(c, {st: {"bins": np.arange(G.n_bins(c, st)),
                                                              **{k: np.zeros((G.n_bins(c, st),) + s, np.int32)
                                                                 for k, s in (("gen", ()), ("nall", ()),
                                                                              ("ht", (150,)), ("he", (10,)),
                                                                              ("hta", (150,)), ("hea", (10,)))}}
                                                         for st in (0, 1, 2)}))
    p = json.load(open(out + ".provenance.json"))
    assert p["model"] == "v6" and p["name"] == "1045_1" and len(p["quirks"]) == 3
    part = p["parts"]["Cerenkov in fiber"][0]
    assert part["photons_per_bin"] == 10000 and part["max_steps"] == 65536
    assert part["resumed_from"]["max_steps"] == 1 << 19
    assert p["notes"] == ["scintillator Cerenkov at 2e4 photons per bin"]
    # the library-core manifest: the content hash pinned loads check, the gaps of these pieces named
    from crvfastsim.lookupgen.mu2e import library as LIB
    from crvfastsim.response import lookup
    from mu2efastsim.library.hashing import content_hash
    from mu2efastsim.library.store import LibraryError
    m = json.load(open(str(tmp_path / "LookupTable_1045_1.manifest.json")))
    assert m["name"] == "crv.lookup.1045_1" and m["subsystem"] == "crv" and m["kind"] == "binned"
    assert m["content_hash"] == content_hash(lookup.arrays(lookup.load("LookupTable_1045_1", str(tmp_path))))
    assert set(m["unrecorded"]) == {"seeds", "release"} and m["validation"]["status"] == "tested, not closed"
    assert [a["name"] for a in m["axes"]] == ["x", "y", "z", "beta", "theta", "phi", "r"]
    assert m["generator"]["configuration"]["max_steps"] == [65536]
    # devices per piece, newest run first (this piece was resumed on another device)
    assert m["generator"]["configuration"]["devices"] == ["gpu A", "gpu B"]
    assert m["generator"]["configuration"]["pieces"]["Cerenkov in fiber"][0]["devices"] == ["gpu B", "gpu A"]
    assert part["device"] == "gpu B" and part["resumed_from"]["device"] == "gpu A"
    assert LIB.verify(out)["content_hash"] == m["content_hash"]
    assert main(["verify", out, "--content-hash", m["content_hash"]]) == 0
    with pytest.raises(LibraryError):
        LIB.verify(out, expect={"content_hash": "0" * 64})
    # given the seed and the commits, the manifest is complete
    assert main(["assemble", "--length", "1045", "--reflector", "1", "--out", out, "--model", "v6", "--assume-seed",
                 "1", "--piece-commit", "abc1234", "--commit", "def5678"] + pieces) == 0
    m2 = json.load(open(str(tmp_path / "LookupTable_1045_1.manifest.json")))
    assert m2["unrecorded"] == {} and m2["content_hash"] == m["content_hash"]
    assert m2["release"]["commit"] == "def5678" and m2["generator"]["release"] == ["abc1234"]
    assert m2["seeds"] == {"scintillation in scintillator": 1, "Cerenkov in scintillator": 1, "Cerenkov in fiber": 1}
    # a piece made before devices were recorded: None in its list (not a manifest gap field)
    z = dict(np.load(pieces[2]))
    z["stats"] = json.dumps({"dtype": "float32", "model": "v6", "max_steps": 65536, "capped": 0})
    np.savez(pieces[2], **z)
    assert main(["assemble", "--length", "1045", "--reflector", "1", "--out", out, "--model", "v6", "--assume-seed",
                 "1", "--piece-commit", "abc1234", "--commit", "def5678"] + pieces) == 0
    m3 = json.load(open(str(tmp_path / "LookupTable_1045_1.manifest.json")))
    assert m3["unrecorded"] == {} and m3["content_hash"] == m["content_hash"]
    assert m3["generator"]["configuration"]["pieces"]["Cerenkov in fiber"][0]["devices"] == [None]
    assert m3["generator"]["configuration"]["devices"] == ["gpu A", "gpu B"]


def test_validation_status_of_each_model():
    """The status a new manifest is written with (core/docs/libraries.md): the current model was measured against a
    CRVStandalone build of the same optics, without criteria declared before the comparison."""
    from crvfastsim.lookupgen.mu2e import library as LIB
    assert LIB.VALIDATION["current"][0] == "tested, not closed"
    assert LIB.VALIDATION["v6"][0] == "tested, not closed"


def test_design_tables_are_named_by_their_cross_section():
    from crvfastsim.lookupgen.mu2e import library as LIB
    assert LIB.table_name(G.design(24.71, 25.0, 3200.0, 0)) == "crv.lookup.t24p71_w25p00_f2p0_s12p66_3200_0"
    assert LIB.table_name(G.stock(6000, 0)) == "crv.lookup.6000_0"


# ------------------------------------------------------------------------------------------------ core / adapter
_STDLIB_OK = {"bisect", "dataclasses", "math", "time", "numpy", "jax"}


@pytest.mark.parametrize("name", sorted(
    f for f in os.listdir(CORE) if f.endswith(".py") and f != "__init__.py"
))
def test_core_imports_nothing_experiment_specific(name):
    """The scint-fibre-fastsim core imports only stdlib, NumPy, JAX and its own modules (level-1 relative)."""
    tree = ast.parse(open(os.path.join(CORE, name)).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in _STDLIB_OK, (name, a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert node.level == 1, (name, node.module)
            else:
                assert node.module.split(".")[0] in _STDLIB_OK, (name, node.module)


def test_mu2e_numbering_is_the_stock_row_major_order():
    c = G.stock(3200)
    e = G.edges(c)
    n = {k: len(v) - 1 for k, v in e.items()}
    b = 2 * n["theta"] * n["phi"] * n["r"] * n["z"] + 7 * n["phi"] * n["r"] * n["z"] + 3 * n["r"] * n["z"] + 4 * n["z"] + 9
    box = G.bin_boxes(c, 2, [b])
    assert [box[k][0][0] for k in ("beta", "theta", "phi", "r", "z")] == \
        [e["beta"][2], e["theta"][7], e["phi"][3], e["r"][4], e["z"][9]]
    b1 = ((3 * n["y"] + 5) * n["z"] + 11) * n["beta"] + 4
    box = G.bin_boxes(c, 1, [b1])
    assert [box[k][0][0] for k in ("x", "y", "z", "beta")] == [e["x"][3], e["y"][5], e["z"][11], e["beta"][4]]


def test_v6_model_is_registered_by_the_adapter_only():
    import subprocess
    import sys
    code = ("from crvfastsim.lookupgen.core import optics as O, materials\n"
            "assert sorted(O.MODELS) == ['current'], O.MODELS\n"
            "import crvfastsim.lookupgen.mu2e\n"
            "assert sorted(O.MODELS) == ['current', 'v6']\n")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_core_runs_as_a_package_of_its_own():
    """scintfibrefastsim.core generates a table for a strip and bins that are not Mu2e's."""
    pytest.importorskip("jax")
    from scintfibrefastsim.core.binning import Binning
    from scintfibrefastsim.core.counter import Counter
    from scintfibrefastsim.core.run import generate

    c = Counter(length=1000.0, thickness=10.0, width=40.0, fiber_separation=10.0, fiber_diameter=1.2,
                clad_fraction=0.03, hole_rx=1.0, hole_ry=0.8, coating=0.2, corner_radius=1.0, guide_bar=5.0,
                sipm_width=1.3, sipm_window=0.1, sipm_length=0.2, air_gap=0.001)
    b = Binning({"x": [0.0, 2.0, 4.8], "y": np.linspace(-19.8, 19.8, 7), "z": np.linspace(-500.0, 500.0, 5)},
                {0: ("z", "y", "x")})
    r = generate(c, b, 0, np.arange(b.n_bins(0)), 50, dtype="float32", lanes=256, chunk_iters=400)
    assert r["gen"].sum() > 0 and np.all(r["ht"].sum(1) == r["he"].sum(1)) and r["ht"].sum() > 0, r["stats"]
