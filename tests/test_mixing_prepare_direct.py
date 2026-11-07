"""Explicit campaign export parsing for direct-frame sampling."""
import numpy as np
import pytest

from mixing.prepare_direct import campaign
from mixing.sampling import CHAINS, STREAMS


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "eff.csv"
    tags = sorted({tag for chain in CHAINS.values() for tag in chain})
    path.write_text("\n".join(tag + ",123,100000,0.00123456789" for tag in tags) + "\n")
    physics = {"producers": {"PBISim": {"module_type": "ProtonBunchIntensityLogNormal",
                "PrimaryParticle": "FindMCPrimary", "SDF": .8, "extendedMean": 1.58e7, "cutMax": 9.48e7}},
               "filters": {stream + "Mixer": {"module_type": "MixBackgroundFrames", "readMode": "sequential",
                   "wrapFiles": True, "mu2e": {"simStageEfficiencyTags": list(CHAINS[stream]),
                       "meanEventsPerPOTFactors": [1000] if stream == STREAMS[0] else [],
                       "MaxEventsToSkip": 2000., "products": {}}} for stream in STREAMS}}
    physics["producers"].update({s + "Filter": {"module_type": "MixingFilter", "MixingModule": s + "Mixer"}
                                for s in STREAMS})
    return {"physics": physics}, path


def test_efficiencies_use_exported_eff_with_stock_float_parse_not_counts(config):
    expanded, path = config
    result = campaign(expanded, path, beam_primary=True)
    assert set(result["efficiencies"].values()) == {float(np.float32("0.00123456789"))}
    assert result["efficiencies"]["MuBeamCat"] != 123 / 100000
    assert result["max_skip"] == dict.fromkeys(STREAMS, 2000)
    path.write_text("TABLE SimEfficiencies2\n" + path.read_text())
    assert campaign(expanded, path, beam_primary=True) == result


@pytest.mark.parametrize("change", ["duplicate", "missing", "nan", "wrongtable", "columns"])
def test_efficiency_export_fails_closed(config, change):
    expanded, path = config
    text = path.read_text()
    if change == "duplicate":
        text += text.splitlines()[0] + "\n"
    elif change == "missing":
        text = "\n".join(text.splitlines()[1:])
    elif change == "nan":
        text = text.replace("0.00123456789", "nan")
    elif change == "wrongtable":
        text = "TABLE Wrong\n" + text
    else:
        text = "tag,value\n"
    path.write_text(text)
    with pytest.raises(ValueError):
        campaign(expanded, path, beam_primary=True)


@pytest.mark.parametrize("key,value", [("readMode", "randomReplace"), ("wrapFiles", False),
                                       ("module_type", "Other")])
def test_mixer_contract_rejected(config, key, value):
    expanded, path = config
    expanded["physics"]["filters"]["MuStopPileupMixer"][key] = value
    with pytest.raises(ValueError):
        campaign(expanded, path, beam_primary=True)


@pytest.mark.parametrize("key,value", [("meanEventsPerPOTFactors", [1]), ("simStageEfficiencyTags", ["Other"]),
                                       ("MaxEventsToSkip", 1.5), ("timeOffset", 10.)])
def test_campaign_details_not_silently_defaulted(config, key, value):
    expanded, path = config
    expanded["physics"]["filters"]["MuStopPileupMixer"]["mu2e"][key] = value
    with pytest.raises(ValueError):
        campaign(expanded, path, beam_primary=True)


def test_real_stock_simtimeoffset_setting_rejected(config):
    expanded, path = config
    expanded["physics"]["filters"]["MuStopPileupMixer"]["mu2e"]["products"]["simTimeOffset"] = {}
    with pytest.raises(ValueError, match="timing"):
        campaign(expanded, path, beam_primary=True)


def test_lower_intensity_cut_preserved_and_beam_declaration_required(config):
    expanded, path = config
    expanded["physics"]["producers"]["PBISim"]["cutMin"] = 1e6
    with pytest.raises(ValueError, match="beam-primary"):
        campaign(expanded, path)
    assert campaign(expanded, path, beam_primary=True)["cut_min"] == 1e6


def test_nonuniversal_postmix_filter_rejected(config):
    expanded, path = config
    expanded["physics"]["producers"]["MuStopPileupFilter"]["selection"] = {"tool_type": "Other"}
    with pytest.raises(ValueError, match="universal"):
        campaign(expanded, path, beam_primary=True)
