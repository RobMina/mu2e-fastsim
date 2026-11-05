"""Status census rejects missing/failure records masquerading as valid zeros."""
import pytest

from test_controls import load_script


def text(event, code):
    return (f"PrintModule Run/Subrun/Event 1430 6000 {event}\n"
            f"ProductPrint mu2e::StatusG4_g4run__CaloLibraryCE\n"
            f"  status: {code} nG4Tracks: 50 overflowSimP: {int(code == 10)} nStepLimit: {int(code == 1)}\n")


def test_export_identity_order_and_status_compared_to_independent_printer(tmp_path):
    import sys
    import numpy as np
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    script = load_script("audit_export")
    log = tmp_path / "status.txt"
    log.write_text(text(1, 0) + text(2, 1) + text(3, 10))
    events = {"run": np.full(3, 1430), "subrun": np.full(3, 6000), "event": np.arange(1, 4),
              "g4_status": np.array([0, 1, 10])}
    script.reconcile_status(events, log, 3)
    with pytest.raises(ValueError, match="identities/order/status"):
        script.reconcile_status({**events, "g4_status": np.zeros(3, dtype=int)}, log, 3)
    with pytest.raises(ValueError, match="identities/order/status"):
        script.reconcile_status({name: value[::-1] for name, value in events.items()}, log, 3)


class File:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __getitem__(self, name):
        assert name == "Events"
        return type("Tree", (), {"num_entries": 3})()


def test_status_zero_distinguished_from_accepted_warning_and_overflow(monkeypatch, tmp_path):
    script = load_script("audit_status")
    monkeypatch.setattr(script.uproot, "open", lambda _: File())
    log = tmp_path / "status.txt"
    log.write_text(text(1, 0) + text(2, 1) + text(3, 10))
    result = script.audit("unused", log, 3)
    assert result["status_counts"] == {"0": 1, "1": 1, "10": 1}
    assert result["status_zero_events"] == 1 and result["nonzero_status_events"] == 2
    assert result["overflow_events"] == 1
    assert result["zero_response_certified"] is False


@pytest.mark.parametrize("content", [text(1, 0) + text(2, 0),
                                     text(1, 0) + text(1, 0) + text(3, 0),
                                     "PrintModule Run/Subrun/Event 1430 6000 1\n" + text(2, 0) + text(3, 0)])
def test_missing_or_duplicate_status_refused(monkeypatch, tmp_path, content):
    script = load_script("audit_status")
    monkeypatch.setattr(script.uproot, "open", lambda _: File())
    log = tmp_path / "status.txt"
    log.write_text(content)
    with pytest.raises(ValueError):
        script.audit("unused", log, 3)
