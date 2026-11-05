"""Static library-path contracts complement released-runtime preflight."""
from pathlib import Path

import pytest


FCL = Path(__file__).resolve().parents[1] / 'fcl'


@pytest.mark.parametrize('name', ['pot', 'stops', 'read'])
def test_library_configs_ascii_and_no_unbounded_default(name):
    raw = (FCL / f'target_library_{name}.fcl').read_bytes()
    text = raw.decode('ascii')
    assert 'RunSubrunEvent' in text and 'PrintModule' in text
    assert 'geom_run1_a.txt' in text
    if name != 'read':
        assert 'source.maxEvents : 0' in text
        assert 'SelectEvents : []' in text
        assert 'outputCommands : [ "keep *_*_*_*" ]' in text
        assert 'PrimaryPath :' in text


def test_library_pot_has_full_stock_upstream_field_and_muon_splitter():
    text = (FCL / 'target_library_pot.fcl').read_text()
    assert 'Production/JobConfig/beam/POT.fcl' in text
    assert 'bFieldFile' not in text
    assert 'mu2eKillerVolume' in text
    assert 'BeamFilter, compressPVBeam, muonSelector' in text


def test_library_stops_preserves_stock_time_mixer_and_negative_muon_capture():
    text = (FCL / 'target_library_stops.fcl').read_text()
    assert 'Production/JobConfig/pileup/MuBeamResampler.fcl' in text
    assert 'protonTimeOffset, beamResampler, g4run, g4consistentFilter' in text
    assert 'TargetMuonFinder, TargetStopFilter, compressPVTargetStops, muminusSelector' in text
    assert '[13, "uninitialized", "muMinusCaptureAtRest"]' in text
    assert 'TargetStopPrescaleFilter' not in text
    assert 'MaxEventsToSkip : 0' in text
