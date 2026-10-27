"""Tests for certifying the actual dumper loaded by runtime smoke jobs."""
from pathlib import Path

import pytest

from valhist.genassoc_smoke import loaded_plugin
from valhist.genassoc_report import aggregate, MAX_METRICS, SUM_METRICS


def test_report_aggregates_counts_and_keeps_worst_residual():
    a = {**dict.fromkeys(SUM_METRICS, 2), **dict.fromkeys(MAX_METRICS, 8.0)}
    b = {**dict.fromkeys(SUM_METRICS, 3), **dict.fromkeys(MAX_METRICS, 1.0)}
    assert aggregate([a, b]) == {**dict.fromkeys(SUM_METRICS, 5), **dict.fromkeys(MAX_METRICS, 8.0)}


PLUGIN = Path("/area/build/stub/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so")


def test_loader_requires_initialization_not_search():
    trace = f"42: trying file={PLUGIN}\n42: calling init: {PLUGIN}\n42: calling fini: {PLUGIN} [0]\n"
    assert loaded_plugin(trace, PLUGIN) == str(PLUGIN)


@pytest.mark.parametrize("trace", ["", f"trying file={PLUGIN}",
    "calling init: /other/libcrvfastsim_CrvStepTableDumper_module.so",
    f"calling init: {PLUGIN}\ncalling init: /other/libcrvfastsim_CrvStepTableDumper_module.so"])
def test_loader_rejects_missing_wrong_or_ambiguous_plugin(trace):
    with pytest.raises(ValueError, match="unexpected loaded dumper"):
        loaded_plugin(trace, PLUGIN)
