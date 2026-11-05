"""Portable refusal checks; never execute Linux child probes on the test host."""

from pathlib import Path

import pytest

from beamfastsim.flash_reference_apptainer_probe import main


@pytest.mark.parametrize("argv", [[], ["--outer-pid-namespace", "pid:[1]"],
                                  ["--persistent-target", "/never-write"]])
def test_explicit_targets_and_external_namespace_required(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_probe_is_explicitly_not_certificate_and_checks_actual_handoff():
    source = Path(__file__).parents[1] / "beamfastsim/flash_reference_apptainer_probe.py"
    text = source.read_text()
    for contract in ('"verified": False', '"transport_events": 0', '"other_visible_processes"',
                     '"unexpected_descriptors"', '"mount_propagation_unresolved"',
                     '"ordinary_threads"', '"inherited_blocked_signals"', 'path.open("xb")',
                     '"memory_open_allowed"', '"syscall_not_denied:"'):
        assert contract in text
