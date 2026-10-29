"""valhist.deck_rerun: arm lists, comparison pairs and the merge refusals (synthetic, no detector data)."""
import pytest

from valhist import deck_rerun as R


def test_arms_and_pairs():
    assert R.arms_of(["new", "legacy"]) == ["a0", "a0p", "a0g", "a1_new", "e2e_new", "a1_legacy", "e2e_legacy"]
    pairs = {n: (r, c, f) for n, r, c, f in R.comparisons(["new"])}
    assert pairs["floor_a0p_vs_a0"] == ("a0", "a0p", None)
    assert pairs["a1_new_vs_a0g"] == ("a0g", "a1_new", ("a0", "a0p"))
    assert pairs["e2e_new_vs_a1_new"] == ("a1_new", "e2e_new", ("a0", "a0p"))


def test_merge_refuses_missing_shards(tmp_path):
    with pytest.raises(SystemExit, match="missing 2 shard files"):
        R.merge(tmp_path, [22, 24], "t", tmp_path / "m", ["new"])
