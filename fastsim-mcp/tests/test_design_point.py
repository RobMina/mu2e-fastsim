import hashlib

import pytest

from fastsim_tools import design_point as dp

geometry = pytest.importorskip("mu2efastsim.endpoints.geometry")

TARGET = {"radii": [75.0, 75.0, 70.0], "halfThicknesses": [0.05, 0.05, 0.05], "zVars": [0.0, 22.2, 44.4],
          "holeRadii": [21.5, 21.5, 21.5], "deltaZ": 22.2, "distFromTargetEnd": 250}


def test_render_is_admitted_by_the_kit():
    text = dp.render_target(TARGET)
    _, keys = geometry.validate_text(text.encode("ascii"))
    assert keys == sorted(["stoppingTarget.radii", "stoppingTarget.halfThicknesses", "stoppingTarget.zVars",
                           "stoppingTarget.holeRadii", "stoppingTarget.deltaZ", "protonabsorber.distFromTargetEnd"])
    assert text.startswith('#include "Offline/Mu2eG4/geom/geom_common.txt"\n')
    assert dp.render_target(dict(reversed(list(TARGET.items())))) == text   # order-free, so the sha is stable


def test_full_names_and_absorber_switch():
    text = dp.render_target({"stoppingTarget.radii": [75], "hasTSdA": False})
    assert "vector<double> stoppingTarget.radii = {75};" in text and "bool hasTSdA = false;" in text
    geometry.validate_text(text.encode("ascii"))


@pytest.mark.parametrize("bad, match", [
    ({}, "at least one key"),
    ({"radius": [1.0]}, "unknown key"),
    ({"radii": [75.0, float("nan")]}, "finite"),
    ({"radii": [75.0], "zVars": [0.0, 1.0]}, "differ in length"),
    ({"radii": [-1.0]}, "positive"),
    ({"holeRadius": 1.0, "holeRadii": [1.0]}, "not both"),
    ({"hasTSdA": True}, "only false"),
    ({"deltaZ": True}, "finite numbers"),
    ({"radii": "75"}, "list"),
])
def test_render_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        dp.render_target(bad)


def test_file_ref_passes_the_kit_descriptor_check(tmp_path):
    text = dp.render_target(TARGET)
    path = tmp_path / "g.txt"
    path.write_text(text)
    ref = dp.file_ref(path.resolve().as_uri(), text)
    assert ref["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert geometry.reference([ref], [], {"task": "beam_edep_cpu"})["sha256"] == ref["sha256"]
    bound = geometry.resolve(ref)
    assert bound["text"] == text


def test_requests_one_job_per_dtype():
    reqs = dp.requests("pt1", "edep", ["float64", "float32"], 5000, 0, 7, "nominal")
    assert [r["handle"] for r in reqs] == ["pt1.edep.float64", "pt1.edep.float32"]
    assert reqs[1]["params"] == {"task": "beam_edep_cpu", "muons": 0, "electrons": 5000, "seed": 7,
                                 "geom": "nominal", "dtype": "float32"}
    no_geom = dp.requests("pt1", "edep", ["float64", "float64"], 5000, 0, 7, None)
    assert len(no_geom) == 1 and "geom" not in no_geom[0]["params"]
    sob = dp.requests("pt1", "sob", ["float64", "float32"], 100_000, 0, 7, "nominal")
    assert sob == [{"handle": "pt1.sob", "dtype": None, "params": {"task": "beam_sob", "muons": 0,
                                                                   "electrons": 100_000, "seed": 7, "geom": "nominal"}}]


def test_stage_table():
    assert {k: v.task for k, v in dp.STAGES.items()} == {
        ("edep", "cpu"): "beam_edep_cpu", ("sob", "cpu"): "beam_sob", ("edep", "gpu"): "beam_edep",
        ("stops", "gpu"): "beam_point", ("centuple", "gpu"): "beam_centuple"}
    assert all((v.gpus > 0) == (k[1] == "gpu") for k, v in dp.STAGES.items())
    with pytest.raises(ValueError, match="no gpu task"):
        dp.stage("sob", "gpu")
    with pytest.raises(ValueError, match="no cpu task"):
        dp.stage("stops", "cpu")


@pytest.mark.parametrize("args, match", [
    (("bad name", "edep", ["float64"], 5000, 0, 1, "nominal"), "name"),
    (("p", "gpu", ["float64"], 5000, 0, 1, "nominal"), "stage"),
    (("p", "edep", ["float64"], 60_000, 0, 1, "nominal"), "electrons"),
    (("p", "edep", ["float16"], 5000, 0, 1, "nominal"), "dtypes"),
    (("p", "edep", [], 5000, 0, 1, "nominal"), "dtypes"),
    (("p", "edep", ["float64"], 5000, -1, 1, "nominal"), "muons"),
    (("p", "edep", ["float64"], 5000, 0, 1, "/abs/path.txt"), "geometry"),
])
def test_request_refusals(args, match):
    with pytest.raises(ValueError, match=match):
        dp.requests(*args)


def test_summarise_edep():
    res = {"metrics": {"stops_per_pot": 1.2e-3, "ce_acceptance": 0.52},
           "files": [{"name": "geom", "uri": "file:///a/out/geometry.txt", "kind": "geom"},
                     {"name": dp.EDEP_FILE, "uri": f"file:///a/out/{dp.EDEP_FILE}", "kind": "root"}],
           "metadata": {"job_id": "123"}}
    s = dp.summarise("edep", "float64", res)
    assert s["edepana_file"] == f"/a/out/{dp.EDEP_FILE}" and s["job_id"] == "123"
    assert s["edep_run_record"].endswith(".edep.json") and s["edep_summary"] == "/a/out/edep_summary.txt"
    call = s["score_with"]["arguments"]
    assert call["analysis"] == "approx_ce_sensitivity" and call["parameters"]["stops_per_pot"] == 1.2e-3
    assert call["data_file"] == s["edepana_file"]


def test_summarise_sob_and_missing_file():
    assert dp.summarise("sob", None, {"metrics": {"sensitivity": 1.1}})["sensitivity"] == 1.1
    assert "no " in dp.summarise("edep", "float32", {"metrics": {}, "files": []})["note"]


def test_overall_state():
    assert dp.overall_state(["completed", "completed"]) == "completed"
    assert dp.overall_state(["completed", "working"]) == "working"
    assert dp.overall_state(["completed", "failed"]) == "failed"
    assert dp.overall_state([]) == "unknown"


def test_expected_minutes_grow_with_electrons():
    assert dp.expected_minutes("edep", 50_000, "float64") > dp.expected_minutes("edep", 5000, "float64")
    assert 15 < dp.expected_minutes("edep", 5000, "float32") < dp.expected_minutes("edep", 5000, "float64") < 25
