"""A successful no-MSC driver checks the particle name, not an environment variable name."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from beamfastsim import foilclosure


@pytest.mark.parametrize("pdg,name", [(13, "mu-"), (-13, "mu+")])
def test_no_msc_completion_uses_charge_name(tmp_path, monkeypatch, pdg, name):
    monkeypatch.setenv("MUSE_WORK_DIR", foilclosure.RELEASE)
    monkeypatch.setenv("BEAMFS_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(foilclosure.shutil, "which", lambda _: "/mock/mu2e")
    monkeypatch.setattr(foilclosure, "_identity", lambda p: dict(path=str(p), bytes=1, sha256="a" * 64))

    def execute(args, **kwargs):
        if "--debug-config" in args:
            (Path(kwargs["cwd"]) / "expanded.fcl").write_text("expanded\n")
        else:
            kwargs["stdout"].write(f"/process/dump msc {name}\n/process/dump CoulombScat {name}\n"
                                   "Art has completed and will exit with status 0.\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(foilclosure.subprocess, "run", execute)
    receipt = foilclosure.run_case(tmp_path / "run", energy=2.02, thickness=.2, pdg=pdg, no_msc=True)
    assert receipt["complete"] and receipt["case"]["pdg"] == pdg
