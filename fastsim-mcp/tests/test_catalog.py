import pytest

from fastsim_tools import approximations_doc, catalog
from fastsim_tools.tools import repo_root

ROOT = repo_root()


def test_keys_unique_and_findable():
    assert len(catalog.KEYS) == len(set(catalog.KEYS))
    for key in catalog.KEYS:
        assert catalog.find(key).key == key
        assert catalog.find(key.lower()).key == key
    for alias in ("W2", "W3", "w4"):
        assert catalog.find(alias).key == "W2-W4"
    with pytest.raises(ValueError, match="unknown workflow"):
        catalog.find("W9")


def test_brief_workflows_present():
    for key in ("W1", "W5", "W7", "W8", "ce_chain", "edepana_writer", "eventntuple_writer"):
        assert key in catalog.BY_KEY


def test_approximation_sets_are_documented():
    documented = {r["workflow"] for r in approximations_doc.parse_table(approximations_doc.read(ROOT))}
    for w in catalog.WORKFLOWS:
        assert w.approximations, w.key
        assert set(w.approximations) <= documented, (w.key, w.approximations)


def test_documents_exist():
    for w in catalog.WORKFLOWS:
        for doc in w.docs:
            assert (ROOT / doc).exists(), (w.key, doc)


def test_packages_exist():
    dirs = {"beamfastsim": "beam/beamfastsim", "trkfastsim": "tracker/trkfastsim", "crvfastsim": "crv/crvfastsim",
            "calofastsim": "calo/calofastsim", "mixing": "mixing", "trigger": "trigger"}
    for w in catalog.WORKFLOWS:
        for pkg in w.packages:
            assert (ROOT / dirs[pkg]).is_dir(), (w.key, pkg)


def test_kit_tasks_are_registered_names():
    beam_tasks = {"beam_point", "beam_sob", "beam_edep", "beam_edep_cpu", "beam_centuple"}
    other = {"chain", "chain_point", "crv_response"}
    for w in catalog.WORKFLOWS:
        assert set(w.kit_tasks) <= beam_tasks | other, w.key
    src = (ROOT / "beam" / "beamfastsim" / "kit_task.py").read_text()
    for name in beam_tasks:
        assert f'tasks["{name}"]' in src


def test_describe_shape():
    d = catalog.find("edepana_writer").describe()
    assert set(d["boundary"]) == {"input", "output"}
    assert "beam_edep_cpu" in d["kit_tasks"]
    assert d["run_design_point"]
