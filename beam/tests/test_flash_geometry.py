"""Synthetic SimpleConfig checks of scoped flash geometry, not full Offline coverage."""
import numpy as np
import pytest

from beamfastsim import cechain as C
from beamfastsim import flash_geometry as F
from beamfastsim import geometry as G


@pytest.fixture
def config():
    return {
        "mu2e.solenoidOffset": 3904.0, "stoppingTarget.radii": [75.0] * 3,
        "stoppingTarget.halfThicknesses": [0.3, 0.05], "stoppingTarget.zVars": [1.0, 3.0],
        "stoppingTarget.deltaZ": 22.0, "stoppingTarget.z0InMu2e": 5871.0,
        "stoppingTarget.holeRadius": 21.5, "hasTarget": True,
        "hasVirtualDetector": True, "vd.halfLength": 0.01,
        "protonabsorber.distFromTargetEnd": 625.0, "hasTSdA": True,
        "tsda.z0": 4235.0, "tsda.halfLength4": 50.0, "tsda.rin": 170.0, "tsda.r4": 400.0,
        "protonabsorber.OutRadius0": 300.5, "protonabsorber.thickness": 0.5,
        "protonabsorber.halfLength": 500.0, "protonabsorber.outerPAInnerRadius0": 436.0,
        "protonabsorber.outerPAInnerRadius1": 720.0, "protonabsorber.outerPAThickness": 10.0,
        "protonabsorber.outerPAZCenter": 7000.0, "protonabsorber.outerPAHalfLength": 1000.0,
        "ts.rTorus": 2934.8, "ts.ts5.halfLength": 625.0, "ts.ts5.rVac": 248.0, "ts.ts5in.rOut": 250.0,
        "ts.coll5.sOffset": -125.0, "ts.coll5.halfLength": 500.0, "ts.coll5.Radius1": 128.0,
    }


def _write(path, config):
    lines = []
    for key, value in config.items():
        if isinstance(value, list):
            typ, value = "vector<double>", "{" + ",".join(map(str, value)) + "}"
        elif isinstance(value, bool):
            typ, value = "bool", str(value).lower()
        elif isinstance(value, str):
            typ, value = "string", '"' + value + '"'
        else:
            typ = "double"
        lines.append(f"{typ} {key} = {value};")
    path.write_text("\n".join(lines))
    return path


def _load(tmp_path, config, **kwargs):
    return F.from_offline(_write(tmp_path / "geom.txt", config), tmp_path, **kwargs)


@pytest.mark.parametrize("n,zvars,expected", [
    (1, [2.0], [5873.0]), (2, [2.0], [5862.0, 5884.0]),
    (3, [1.0, 3.0], [5850.0, 5874.0, 5896.0]),
    (4, [1.0, 3.0], [5839.0, 5863.0, 5885.0, 5907.0]),
])
def test_foil_parity_and_repeated_last_entries(tmp_path, config, n, zvars, expected):
    config.update({"stoppingTarget.radii": [75.0] * n, "stoppingTarget.zVars": zvars,
                   "stoppingTarget.halfThicknesses": [0.05]})
    b = _load(tmp_path, config).beam
    np.testing.assert_array_equal(b.foil_z, expected)
    np.testing.assert_array_equal(b.foil_ht, [0.05] * n)
    np.testing.assert_array_equal(b.foil_rin, [21.5] * n)


@pytest.mark.parametrize("vd", [True, False])
def test_relative_ipa_uses_target_mother_end_not_foil_thickness(tmp_path, config, vd):
    config["hasVirtualDetector"] = vd
    if not vd:
        del config["vd.halfLength"]
    b = _load(tmp_path, config).beam
    np.testing.assert_array_equal(b.foil_ht, [0.3, 0.05, 0.05])
    assert b.ipa_z0 == pytest.approx(5896.0 + 5.0 + (0.02 if vd else 0.0) + 625.0)
    assert b.ipa_z1 == b.ipa_z0 + 1000.0
    assert b.ipa_z0 != b.foil_z.max() + b.foil_ht.max() + 625.0


def test_explicit_ipa_wins_over_target_distance_and_vd(tmp_path, config):
    config["protonabsorber.zStartInMu2e"] = 7123.0
    first = _load(tmp_path, config).beam
    config["stoppingTarget.z0InMu2e"] += 100.0
    config["protonabsorber.distFromTargetEnd"] = 123.0
    config["hasVirtualDetector"] = False
    second = _load(tmp_path, config).beam
    assert first.ipa_z0 == second.ipa_z0 == 7123.0
    assert first.ipa_z1 == second.ipa_z1 == 8123.0
    del config["protonabsorber.distFromTargetEnd"]
    assert _load(tmp_path, config).beam.ipa_z0 == 7123.0


def test_tsda_off_does_not_move_ts5_or_collimator(tmp_path, config):
    on = _load(tmp_path, config)
    config["hasTSdA"] = False
    off = _load(tmp_path, config)
    assert on.beam.tsda_hl == 50.0 and off.beam.tsda_hl == 0.0
    assert on.z_ds == off.z_ds == 4184.8
    assert on.z_up == off.z_up == 2934.8
    assert on.coll5_z0 == off.coll5_z0 == pytest.approx(2934.8)
    assert on.coll5_z1 == off.coll5_z1 == pytest.approx(3934.8)
    assert on.r_ts == off.r_ts == 248.0
    config["tsda.z0"] += 123.0
    config["tsda.halfLength4"] *= 2
    moved = _load(tmp_path, config)
    assert moved.z_ds == off.z_ds and moved.coll5_z0 == off.coll5_z0


@pytest.mark.parametrize("holes", [[], [20.0, 21.0, 22.0]])
def test_hole_vector_precedence_and_empty_scalar_fallback(tmp_path, config, holes):
    config["stoppingTarget.holeRadii"] = holes
    b = _load(tmp_path, config).beam
    np.testing.assert_array_equal(b.foil_rin, holes or [21.5] * 3)


@pytest.mark.parametrize("key,value,match", [
    ("stoppingTarget.radii", [], "radii"),
    ("stoppingTarget.radii", [75.0, np.nan, 75.0], "radii"),
    ("stoppingTarget.halfThicknesses", [], "halfThicknesses"),
    ("stoppingTarget.halfThicknesses", [0.1] * 4, "halfThicknesses"),
    ("stoppingTarget.halfThicknesses", [0.0], "half thickness"),
    ("stoppingTarget.halfThicknesses", [15.0], "overlap"),
    ("stoppingTarget.zVars", [], "zVars"),
    ("stoppingTarget.zVars", [0.0] * 4, "zVars"),
    ("stoppingTarget.zVars", [np.inf], "zVars"),
    ("stoppingTarget.zVars", [0.0, -22.0, 0.0], "increasing"),
    ("stoppingTarget.zVars", [0.0, -23.0, 0.0], "increasing"),
    ("stoppingTarget.holeRadii", [1.0], "length"),
    ("stoppingTarget.holeRadii", [1.0] * 4, "length"),
    ("stoppingTarget.holeRadii", [0.0, np.nan, 0.0], "holeRadii"),
    ("stoppingTarget.holeRadius", -1.0, "radii"),
    ("stoppingTarget.holeRadius", 75.0, "radii"),
    ("stoppingTarget.deltaZ", 0.0, "deltaZ"),
    ("stoppingTarget.z0InMu2e", np.nan, "z0InMu2e"),
    ("stoppingTarget.overrideMaterial", "G4_W", "overrideMaterial"),
    ("stoppingTarget.materials", [], "materials"),
    ("hasTarget", False, "target type"),
])
def test_reject_invalid_or_unrepresented_foils(tmp_path, config, key, value, match):
    config[key] = value
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, config)


@pytest.mark.parametrize("name", ["xVars", "yVars", "xCos", "yCos"])
def test_offsets_and_normal_cosines_are_not_silently_ignored(tmp_path, config, name):
    key = "stoppingTarget." + name
    config[key] = [0.0]
    _load(tmp_path, config)
    config[key] = [0.0, 0.001]
    with pytest.raises(ValueError, match=name):
        _load(tmp_path, config)


@pytest.mark.parametrize("material", ["", "G4_Al", "StoppingTarget_Al"])
def test_known_aluminium_override(tmp_path, config, material):
    config["stoppingTarget.overrideMaterial"] = material
    _load(tmp_path, config)


def test_includes_provenance_and_shared_loaders_unchanged(tmp_path, config):
    base = _write(tmp_path / "base.txt", config)
    overlay = tmp_path / "overlay.txt"
    overlay.write_text('#include "base.txt"\nbool hasTSdA = false;\n')
    sources = []
    scoped = F.from_offline("overlay.txt", tmp_path, z_plane=9000.0, source_files=sources)
    assert sources == [str(overlay), str(base)]
    assert scoped.z_plane == 9000.0
    shared = G.from_offline(str(overlay), str(tmp_path))
    ce = C.from_offline(str(overlay), str(tmp_path))
    assert shared.foil_z[-1] == 5893.0
    assert scoped.beam.foil_z[-1] == 5896.0
    assert ce.z_ds == 4235.0 and scoped.z_ds == 4184.8


@pytest.mark.parametrize("present", [False, True], ids=["absent", "installed"])
def test_navigator_absent_tsda_has_no_absorbing_plane(tmp_path, config, present):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from beamfastsim import flash_navigator_jax as FN

    config["hasTSdA"] = present
    geom = _load(tmp_path, config)
    init, run = FN._build(geom, np.float64, 4.0, 1000.0, False, True, True, False,
                         (0.0, 0.0, 0.0), photon_capacity=1)
    x = jnp.array([[geom.beam.axis_x + 200.0, 0.0, 4286.0]], dtype=jnp.float64)
    u = jnp.array([[0.0, 0.0, -1.0]], dtype=jnp.float64)
    st = init((), x, u, jnp.array([105.0]), jnp.zeros(1), jnp.full(1, -1, jnp.int32), jnp.zeros(1))
    _, st = run((), jax.random.PRNGKey(204), st, jnp.int32(16), 1, 1)
    assert int(st["fate"][0]) == (C.ABSORBED if present else C.LIVE)
    assert float(st["x"][0, 2]) == pytest.approx(4285.0 if present else 4222.0)
    assert not np.asarray(st["cnt"]).any()
    assert not np.asarray(st["count"]).any()


def test_axial_and_radial_mother_containment(tmp_path, config):
    config["stoppingTarget.halfThicknesses"] = [6.0]
    with pytest.raises(ValueError, match="axial extent"):
        _load(tmp_path, config)
    config["stoppingTarget.halfThicknesses"] = [0.1]
    config["stoppingTarget.radii"] = [300.] * 3
    with pytest.raises(ValueError, match="mother cylinder"):
        _load(tmp_path, config)
    config["stoppingTarget.foilTarget_supportStructure_endAtOPA"] = True
    config["stoppingTarget.foilTarget_supportStructure_radius"] = 0.1
    _load(tmp_path, config)
    config["stoppingTarget.radii"] = [800.] * 3
    with pytest.raises(ValueError, match="mother cylinder"):
        _load(tmp_path, config)


@pytest.mark.parametrize("distance", [-1., 2500.01])
def test_out_of_range_ipa_distance_rejected_not_silently_reset(tmp_path, config, distance):
    config["protonabsorber.distFromTargetEnd"] = distance
    with pytest.raises(ValueError, match="outside"):
        _load(tmp_path, config)


@pytest.mark.parametrize("distance", [0., 2500.])
def test_ipa_distance_endpoints(tmp_path, config, distance):
    config["protonabsorber.distFromTargetEnd"] = distance
    assert _load(tmp_path, config).beam.ipa_z0 == pytest.approx(5896 + 5 + .02 + distance)
