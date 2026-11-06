"""No-wrap same-input shards for arbitrary explicit design labels and geometries."""
import json
from pathlib import Path

import pytest

from beamfastsim.flash_reference_campaign import manifest, main


def args():
    return dict(events=100, shards=3, skip=10, pool_events=1000, seed_base=100, source="ts3")


def test_new_designs_have_same_nonoverlap_ranges_and_distinct_seeds():
    rows = [x.split("\t") for x in manifest([("unseen_A", "/geom/a.txt"), ("unseen_B", "/geom/b.txt")], **args()).splitlines()]
    assert [int(x[3]) for x in rows] == [10, 110, 210, 10, 110, 210]
    assert [int(x[5]) for x in rows] == list(range(100, 106))
    assert {x[0] for x in rows} == {"unseen_A", "unseen_B"}


@pytest.mark.parametrize("key,value", [("shards", 0), ("skip", -1), ("events", True),
                                       ("events", 0), ("pool_events", 309), ("seed_base", 900000000)])
def test_bad_ranges_or_seed_fail(key, value):
    kw = args()
    kw[key] = value
    with pytest.raises(ValueError):
        manifest([("new", "/g")], **kw)


@pytest.mark.parametrize("designs", [[], [("bad name", "/g")], [("new", "relative")],
                                     [("new", "/g space")], [("new", "/g"), ("new", "/h")]])
def test_bad_design_labels_paths_and_duplicates(designs):
    with pytest.raises(ValueError):
        manifest(designs, **args())


def test_cli_writes_once_without_submit(tmp_path):
    out = tmp_path / "manifest.tsv"
    argv = ["--design", "custom=/g.txt", "--events", "100", "--shards", "3", "--skip", "10",
            "--pool-events", "1000", "--seed-base", "100", "--source", "ts3", "--out", str(out)]
    assert main(argv) == 0
    assert len(out.read_text().splitlines()) == 3
    sidecar = out.with_suffix(".json")
    original = out.read_bytes(), sidecar.read_bytes()
    note = json.loads(original[1])
    assert note["jobs"] == 3
    assert note["range_end_exclusive"] == 310
    with pytest.raises(FileExistsError):
        main(argv)
    assert (out.read_bytes(), sidecar.read_bytes()) == original



def one_event_args(out):
    return ["--design", "custom=/geometry.txt", "--events", "1", "--shards", "1",
            "--skip", "0", "--pool-events", "1", "--seed-base", "100", "--source", "ts3",
            "--out", str(out)]


@pytest.mark.parametrize("existing", [False, True])
def test_json_manifest_alias_rejected_without_changes(tmp_path, existing, capsys):
    out = tmp_path / "nested" / "manifest.json"
    if existing:
        out.parent.mkdir()
        out.write_text("previous campaign metadata\n")
    with pytest.raises(SystemExit) as exc:
        main(one_event_args(out))
    assert exc.value.code == 2
    assert "paths must be distinct" in capsys.readouterr().err
    if existing:
        assert out.read_text() == "previous campaign metadata\n"
    else:
        assert not out.parent.exists()


@pytest.mark.parametrize("existing", ["manifest.tsv", "manifest.json"])
def test_existing_artifact_preserved_without_orphan(tmp_path, existing):
    out = tmp_path / "manifest.tsv"
    previous = tmp_path / existing
    previous.write_text("previous campaign artifact\n")
    with pytest.raises(FileExistsError):
        main(one_event_args(out))
    assert previous.read_text() == "previous campaign artifact\n"
    assert list(tmp_path.iterdir()) == [previous]


@pytest.mark.parametrize("existing", ["manifest.tsv", "manifest.json"])
def test_output_symlink_is_not_overwritten_or_removed(tmp_path, existing):
    out = tmp_path / "manifest.tsv"
    target = tmp_path / "previous"
    target.write_text("keep this\n")
    link = tmp_path / existing
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        main(one_event_args(out))
    assert link.is_symlink()
    assert target.read_text() == "keep this\n"
    assert set(tmp_path.iterdir()) == {target, link}


def test_sidecar_reservation_failure_rolls_back_unwritten_manifest(tmp_path, monkeypatch):
    out = tmp_path / "manifest.tsv"
    sidecar = out.with_suffix(".json")
    original_open = Path.open

    def fail_sidecar(path, mode="r", *args, **kwargs):
        if path == sidecar and mode == "x":
            assert out.exists() and out.read_bytes() == b""
            raise PermissionError("sidecar reservation failed")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_sidecar)
    with pytest.raises(PermissionError, match="sidecar reservation"):
        main(one_event_args(out))
    assert not list(tmp_path.iterdir())


def test_sidecar_created_during_reservation_is_preserved(tmp_path, monkeypatch):
    out = tmp_path / "manifest.tsv"
    sidecar = out.with_suffix(".json")
    original_open = Path.open

    def competing_sidecar(path, mode="r", *args, **kwargs):
        if path == sidecar and mode == "x":
            assert out.read_bytes() == b""
            with original_open(sidecar, "x") as stream:
                stream.write("other campaign metadata\n")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", competing_sidecar)
    with pytest.raises(FileExistsError):
        main(one_event_args(out))
    assert not out.exists()
    assert sidecar.read_text() == "other campaign metadata\n"


def test_write_failure_rolls_back_both_new_files(tmp_path, monkeypatch):
    out = tmp_path / "manifest.tsv"
    sidecar = out.with_suffix(".json")
    original_open = Path.open

    class FailingWriter:
        def __enter__(self):
            self.stream = original_open(sidecar, "x")
            return self

        def write(self, text):
            assert out.exists() and sidecar.exists()
            raise OSError("metadata write failed")

        def __exit__(self, *args):
            self.stream.close()

    def fail_write(path, mode="r", *args, **kwargs):
        return FailingWriter() if path == sidecar and mode == "x" else original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_write)
    with pytest.raises(OSError, match="metadata write"):
        main(one_event_args(out))
    assert not list(tmp_path.iterdir())



def test_resolved_sidecar_alias_is_rejected_before_reservation(tmp_path, capsys):
    out = tmp_path / "manifest.tsv"
    sidecar = out.with_suffix(".json")
    sidecar.symlink_to(out)
    with pytest.raises(SystemExit) as exc:
        main(one_event_args(out))
    assert exc.value.code == 2
    assert "paths must be distinct" in capsys.readouterr().err
    assert not out.exists()
    assert sidecar.is_symlink()
