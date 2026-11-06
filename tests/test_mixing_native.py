"""Native mixer wiring preserves stock laws and uses detector steps, not digis."""
import pytest

from mixing.native import render_mix
from mixing.sampling import STREAMS


def donors():
    return {stream: [f"{stream}.art"] for stream in STREAMS}


def test_native_stock_configuration():
    text = render_mix(donors(), parent_fcl="campaign.fcl", bunch_fcl="OneBB.fcl")
    assert text.startswith('#include "campaign.fcl"\n#include "OneBB.fcl"')
    assert text.count("readMode: sequential") == 4
    assert text.count("wrapFiles: true") == 4
    assert text.count("writeEventIDs: true") == 4
    assert "simTimeOffset" not in text
    assert "meanEventsPerProton" not in text
    assert "simStageEfficiencyTags" not in text
    assert "StrawGasStepModules" in text
    assert "CrvPhotons.crvStepModuleLabels" in text
    assert "MuStopPileupFilter, EleBeamFlashFilter, MuBeamFlashFilter, NeutralsFlashFilter" in text
    assert "CaloShowerROMaker" not in text


def test_fast_signal_only_changes_tracker_crv_and_truth_inputs():
    text = render_mix(donors(), parent_fcl="campaign.fcl", bunch_fcl="OneBB.fcl",
                      signal_tag="fastSteps", max_skip=30)
    assert "StrawGasStepModules: [fastSteps," in text
    assert 'crvStepModuleLabels: ["fastSteps",' in text
    assert "simParticleTags: [compressDetStepMCs, fastSteps," in text
    assert text.count("MaxEventsToSkip: 30") == 4


@pytest.mark.parametrize("kwargs", [
    {"parent_fcl": "/absolute.fcl"}, {"bunch_fcl": "bad\n.fcl"},
    {"signal_tag": "bad:tag"}, {"max_skip": -1},
])
def test_invalid_config(kwargs):
    args = {"parent_fcl": "campaign.fcl", "bunch_fcl": "OneBB.fcl"} | kwargs
    with pytest.raises(ValueError):
        render_mix(donors(), **args)


def test_missing_stream_not_silently_zero():
    incomplete = donors()
    del incomplete[STREAMS[0]]
    with pytest.raises(ValueError, match="all four"):
        render_mix(incomplete, parent_fcl="campaign.fcl", bunch_fcl="OneBB.fcl")
