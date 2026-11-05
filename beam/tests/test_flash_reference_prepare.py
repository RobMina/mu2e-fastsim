"""Synthetic reference preparation controls; no art, G4 or site inputs required."""
from pathlib import Path
import hashlib
import json

import pytest

from beamfastsim.flash_reference_prepare import flatten_geometry, prepare


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "runtime"
    (root / "Offline").mkdir(parents=True)
    (root / "Offline" / "base.txt").write_text("double target.r = 1;\n")
    geom = tmp_path / "overlay.txt"
    geom.write_text('#include "Offline/base.txt"\ndouble target.r = 2;\n')
    source, field, template = [tmp_path / name for name in ("source.art", "field.txt", "raw.fcl")]
    source.write_bytes(b"synthetic input")
    field.write_text("string map = \"fixed\";\n")
    template.write_text("source: { module_type: RootInput maxEvents: 0 }\n")
    return dict(input_path=source, geom=geom, offline_root=root, field=field, template=template,
                events=1000, seed=51021, physics_list="FTFP_BERT", out=tmp_path / "prepared")


def test_bundle_is_not_a_simulation_or_normalization_certificate(inputs):
    report = prepare(**inputs)
    saved = json.loads((inputs["out"] / "preparation.json").read_text())
    assert saved == report and report["transport_executed"] is False
    assert "no POT" in report["normalization"]
    assert "no resampling" in report["source_assignment"]
    text = (inputs["out"] / "geometry.txt").read_text()
    assert text.index("target.r = 1") < text.index("target.r = 2")
    assert "#include" not in text and "int g4.stepsSizeLimit = 0;" in text
    assert len(report["geometry_sources"]) == 2
    for name, expected in report["prepared_sha256"].items():
        assert hashlib.sha256((inputs["out"] / name).read_bytes()).hexdigest() == expected
    job = (inputs["out"] / "job.fcl").read_text()
    assert "source.maxEvents: 1000" in job
    assert job.startswith(inputs["template"].read_text())
    assert '#include "raw.fcl"' not in job


def test_exclusive_output(inputs):
    prepare(**inputs)
    with pytest.raises(FileExistsError):
        prepare(**inputs)


@pytest.mark.parametrize("events", [True, 0, -1, 1001, 2.0])
def test_invalid_event_allowance(inputs, events):
    with pytest.raises(ValueError, match="events"):
        prepare(**dict(inputs, events=events))
    assert not inputs["out"].exists()


@pytest.mark.parametrize("seed", [False, 0, -1, 900000000, 2.0])
def test_invalid_seed(inputs, seed):
    with pytest.raises(ValueError, match="Seed"):
        prepare(**dict(inputs, seed=seed))


def test_explicit_physics_only(inputs):
    with pytest.raises(ValueError, match="reference"):
        prepare(**dict(inputs, physics_list="default"))


def test_include_cycle_refuses(inputs):
    (inputs["offline_root"] / "Offline" / "base.txt").write_text('#include "Offline/base.txt"\n')
    with pytest.raises(ValueError, match="Cyclic"):
        prepare(**inputs)


@pytest.mark.parametrize("include", ["../outside.txt", "/absolute.txt"])
def test_geometry_escape_refuses(inputs, include):
    inputs["geom"].write_text(f'#include "{include}"\n')
    with pytest.raises(ValueError, match="under offline_root"):
        prepare(**inputs)


def test_symlink_include_escape_refuses(inputs):
    base = inputs["offline_root"] / "Offline" / "base.txt"
    base.unlink()
    base.symlink_to(inputs["field"])
    with pytest.raises(ValueError, match="escapes"):
        prepare(**inputs)


def test_repeated_include_not_silently_removed(inputs):
    inputs["geom"].write_text('#include "Offline/base.txt"\ndouble target.r = 2;\n#include "Offline/base.txt"\n')
    text, sources = flatten_geometry(inputs["geom"], inputs["offline_root"])
    assert text.count("target.r = 1") == 2 and len(sources) == 2
    assert text.rindex("target.r = 1") > text.index("target.r = 2")


def test_nested_include_order_and_unsupported_syntax(inputs):
    root = inputs["offline_root"]
    (root / "Offline" / "nested.txt").write_text('double target.r = 3;\n')
    (root / "Offline" / "base.txt").write_text('#include "Offline/nested.txt"\ndouble target.r = 1;\n')
    text, sources = flatten_geometry(inputs["geom"], root)
    assert text.index("target.r = 3") < text.index("target.r = 1") < text.index("target.r = 2")
    assert len(sources) == 3
    inputs["geom"].write_text('#include <Offline/base.txt>\n')
    with pytest.raises(ValueError, match="Unsupported"):
        prepare(**inputs)
    assert not inputs["out"].exists()


def test_missing_include_refuses(inputs):
    (inputs["offline_root"] / "Offline" / "base.txt").unlink()
    with pytest.raises(FileNotFoundError):
        prepare(**inputs)


def test_fcl_template_static_safety_contract():
    text = (Path(__file__).parents[1] / "fcl" / "flash_reference_raw.fcl").read_text(encoding="ascii")
    assert 'source: { module_type: RootInput maxEvents: 0 }' in text
    assert 'inputPhysVolumeMultiInfo: "compressPVBeam"' in text
    assert 'primaryTag: "BeamFilter:Beam"' in text
    assert 'maxSimParticleCollectionSize: 1000000' in text
    assert 'value: false write: "r5PhotonSteps"' in text
    assert text.index("{ type: pdgId pars: [22] }") < text.index('write: "r5PhotonSteps"')
    assert 'transport: [g4run]' in text and 'outputCommands: ["keep *_*_*_*"]' in text
    for forbidden in ("SelectEvents", "beamResampler", "genCounter", "protonTimeOffset", "compressDetStepMCs"):
        assert forbidden not in text
