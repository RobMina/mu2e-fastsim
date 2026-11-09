"""runchain refuses --calib nominal for configurations that name their stock conditions (RunConfig.conditions)."""
import pytest

from crvfastsim import runconfig
from crvfastsim.analyses import runchain


@pytest.mark.parametrize("name", ["run1a_v01", "run1b_v01"])
def test_nominal_refused_for_run1a(name):
    with pytest.raises(SystemExit, match="--calib nominal is refused"):
        runchain.check_conditions(runconfig.get(name), "nominal")


@pytest.mark.parametrize("name", ["run1a_v01", "run1b_v01"])
def test_resolve_refuses_before_loading_assets(name):
    with pytest.raises(SystemExit, match="conditions of its stock job"):
        runchain._resolve(None, "nominal", None, None, False, None, name, "offspill")


def test_converted_csv_accepted_for_run1a():
    runchain.check_conditions(runconfig.get("run1a_v01"), "crvcalib_run1a_v01.csv")


@pytest.mark.parametrize("name", ["extracted_v04", "run2_v10"])
def test_nominal_allowed_without_stock_conditions(name):
    assert runconfig.get(name).conditions == ""
    runchain.check_conditions(runconfig.get(name), "nominal")


def test_default_configuration_keeps_nominal():
    runchain.check_conditions(runconfig.get(None), "nominal")
