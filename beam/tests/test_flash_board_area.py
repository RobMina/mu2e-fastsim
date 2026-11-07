"""Source-layout guards for Muse discovery; these tests do not compile plugins."""
from pathlib import Path
import shutil
import subprocess

import pytest


ART = Path(__file__).resolve().parents[1] / "art"


def test_muse_discovers_board_repository_and_plugins():
    repos = {marker.parent.name: marker.parent for marker in ART.glob("*/.muse")}
    assert "FlashBoard" in repos
    repo = repos["FlashBoard"]
    assert (ART / ".muse").read_text().strip() == "ENVSET p107"
    assert (repo / "src/SConscript").is_file()
    assert {source.name for source in (repo / "src").glob("*_module.cc")} == {
        "FlashBoardExporter_module.cc", "FlashBoardFixture_module.cc",
    }


@pytest.mark.parametrize("missing", [".muse", "src/SConscript"])
def test_prepare_rejects_missing_build_discovery_file(tmp_path, missing):
    area = tmp_path / "art"
    (area / "FlashBoard/src").mkdir(parents=True)
    script = area / "prepare_board_area.sh"
    shutil.copyfile(ART / script.name, script)
    for relative in (".muse", "src/SConscript"):
        if relative != missing:
            (area / "FlashBoard" / relative).write_text("")
    result = subprocess.run(
        ["bash", str(script), str(tmp_path / "unused-backing")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "repository .muse marker and src/SConscript" in result.stderr
    assert not (area / "backing").is_symlink()
