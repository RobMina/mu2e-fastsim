"""Zero-event seed probe contract tests."""
from pathlib import Path

import pytest

from valhist import generator_seed_probe as S


def test_configuration_changes_no_physics():
    text, base = S.configuration("crygen", 2001, Path("/data/probe"))
    assert base == 20263000 and 'verbosity: 1' in text
    assert 'gen_only_extracted.fcl' in text and 'PrimaryOutput' not in text
    text, base = S.configuration("reference", 2, Path("/data/probe"))
    assert base == 20261017 and 'PrimaryOutput' in text
    for mode, subrun in (("crygen", 2000), ("reference", 221), ("unknown", 2)):
        with pytest.raises(ValueError):
            S.configuration(mode, subrun, Path("/data"))


def test_zero_event_exit_guard():
    text = ('Summary of seeds computed by the SeedService.\nBase Seed: 20\n'
            'Seed Value     ModuleLabel.InstanceName\n20 cosmicTimeOffset\n21 g4run\n22 generate\n'
            '=============================================\n'
            'TrigReport Events total = 0 passed = 0 failed = 0\n'
            'Art has completed and will exit with status 0.\n')
    assert S.verify_zero(text) == (20, dict(cosmicTimeOffset=20, g4run=21, generate=22))
    for bad in (text.replace('total = 0', 'total = 1'), text.replace('status 0', 'status 1'),
                text.replace('22 generate', 'no assignments'), text + 'Art has completed and will exit with status 2.\n',
                text.replace('21 g4run', '21 generate'), text.replace('Base Seed: 20', 'Base Seed: 20\nBase Seed: 21')):
        with pytest.raises(ValueError):
            S.verify_zero(bad)
