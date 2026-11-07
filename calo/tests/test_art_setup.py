"""Area preparation must register a discoverable muse repository without building."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


ART = Path(__file__).resolve().parents[1] / "art"


@pytest.mark.parametrize("marker", [True, False])
def test_prepare_requires_repository_marker(tmp_path, marker):
    area, backing = tmp_path / "art", tmp_path / "backing"
    package = area / "CaloFastSim"
    (package / "src").mkdir(parents=True)
    shutil.copy2(ART / "setup_area.sh", area / "setup_area.sh")
    shutil.copy2(ART / "CaloFastSim/src/SConscript", package / "src/SConscript")
    if marker:
        shutil.copy2(ART / "CaloFastSim/.muse", package / ".muse")
    (backing / "Offline").mkdir(parents=True)
    (backing / "Offline/.muse").write_text("ENVSET p109\n")
    (backing / "build/al9-prof-e29-p109").mkdir(parents=True)
    env = {**os.environ, "CALO_BACKING": str(backing), "CALO_ENVSET": "p109"}
    result = subprocess.run(["bash", str(area / "setup_area.sh")], env=env,
                            capture_output=True, text=True)
    assert not (area / "build").exists()
    if marker:
        assert result.returncode == 0, result.stderr
        assert (area / "backing").resolve() == backing
        assert (area / ".muse").read_text() == "ENVSET p109\n"
        assert "muse build -j 32" in result.stdout
        assert [p.parent.name for p in area.glob("*/.muse")] == ["CaloFastSim"]
    else:
        assert result.returncode != 0
        assert "Missing CaloFastSim/.muse" in result.stderr
        assert not (area / "backing").exists()
        assert not (area / ".muse").exists()
