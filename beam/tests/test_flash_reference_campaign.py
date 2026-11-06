"""No-wrap same-input shards for arbitrary explicit design labels and geometries."""
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
    with pytest.raises(FileExistsError):
        main(argv)
