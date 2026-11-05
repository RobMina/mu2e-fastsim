"""Validate a saved stock PrintModule text census without reading art products."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


_FLAGS = re.MULTILINE
_EVENT_MARKER = re.compile(r"^[^\r\n]*\bPrintModule Run/Subrun/Event\b[^\r\n]*$", _FLAGS)
_BOUNDARY = re.compile(r"^[^\r\n]*\bPrintModule Run/Subrun(?:/Event)?\b[^\r\n]*$", _FLAGS)
_HEADER = re.compile(
    r"^[ \t]*(?:#+[ \t]+)?PrintModule Run/Subrun/Event[ \t]+"
    r"([0-9]+)[ \t]+([0-9]+)[ \t]+([0-9]+)[ \t]*$", _FLAGS)
_PRODUCT = re.compile(r"^[ \t]*ProductPrint\b([^\r\n]*)$", _FLAGS)
_STATUS_MARKER = re.compile(r"^[ \t]*status\b[^\r\n]*$", _FLAGS)
_NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
_STATUS = re.compile(
    r"^[ \t]*status:[ \t]*([0-9]+)[ \t]+nG4Tracks:[ \t]*([0-9]+)"
    r"[ \t]+overflowSimP:[ \t]*([0-9]+)[ \t]+nStepLimit:[ \t]*([0-9]+)"
    rf"(?:[ \t]+CPU:[ \t]*{_NUMBER}[ \t]+Wall:[ \t]*{_NUMBER})?[ \t]*$", _FLAGS)
_TRIGGER_MARKER = re.compile(r"^[ \t]*(?:pass|fail|true|false|error|exception|[01])[ \t]+[^\r\n]*\bPrimaryPath\b[^\r\n]*$", _FLAGS)
_TRIGGER = re.compile(r"^[ \t]*(pass|fail)[ \t]+([0-9]+)[ \t]+PrimaryPath[ \t]*$", _FLAGS)
_STATUS_TYPES = ("mu2e::StatusG4", "mu2eStatusG4", "StatusG4")
_TRIGGER_TYPES = ("art::TriggerResults", "artTriggerResults", "TriggerResults")


def _starts(matches):
    return np.fromiter((m.start() for m in matches), dtype=np.int64, count=len(matches))


def _records(text, marker, pattern, name):
    candidates, matches = list(marker.finditer(text)), list(pattern.finditer(text))
    if not np.array_equal(_starts(candidates), _starts(matches)):
        raise ValueError(f"malformed {name} record")
    return matches


def _integers(rows, width, name):
    try:
        return np.asarray(rows, dtype=np.int64).reshape(-1, width)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain nonnegative int64 integers") from exc


def read_log(path, expected_events, process="R1Inclusive", require_triggers=True):
    """Return pickle-free ndarrays: identities (N,3), status/counts, accepted (-1 absent, 0 fail, 1 pass)."""
    if isinstance(expected_events, (bool, np.bool_)) or not isinstance(expected_events, (int, np.integer)) or expected_events < 0:
        raise ValueError("expected_events must be a nonnegative integer")
    if not isinstance(process, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", process):
        raise ValueError("process must be a nonempty process name")
    if not isinstance(require_triggers, (bool, np.bool_)):
        raise ValueError("require_triggers must be boolean")
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    headers = _records(text, _EVENT_MARKER, _HEADER, "Run/Subrun/Event header")
    if len(headers) != expected_events:
        raise ValueError(f"event count mismatch: found {len(headers)}, expected {expected_events}")
    identities = _integers([m.groups() for m in headers], 3, "event identities")
    if np.unique(identities, axis=0).shape[0] != expected_events:
        raise ValueError("duplicate event identities")
    boundaries = _starts(list(_BOUNDARY.finditer(text)))
    boundary_events = np.full(boundaries.size, -1, dtype=np.int64)
    boundary_events[np.searchsorted(boundaries, _starts(headers))] = np.arange(expected_events)

    def event_owners(positions):
        owner = np.searchsorted(boundaries, positions, side="right") - 1
        result = np.full(positions.size, -1, dtype=np.int64)
        valid = owner >= 0
        result[valid] = boundary_events[owner[valid]]
        if np.any(result < 0):
            raise ValueError("required record outside a PrintModule event")
        return result

    products = list(_PRODUCT.finditer(text))
    product_starts = _starts(products)
    tags = np.asarray([m.group(1).strip() for m in products], dtype=str)

    def selected_products(kind, types, module):
        selected = np.flatnonzero(np.char.find(tags, kind) >= 0)
        allowed = [f"{typ}_{module}__{process}" for typ in types]
        if not np.isin(tags[selected], allowed).all():
            raise ValueError(f"wrong or malformed {kind} product/process; expected {module}__{process}")
        owners = event_owners(product_starts[selected])
        counts = np.bincount(owners, minlength=expected_events)
        if np.any(counts > 1):
            raise ValueError(f"duplicate {kind} product per event")
        return selected, owners, counts

    def row_owners(matches, selected, product_events, name):
        positions = _starts(matches)
        event_indices = event_owners(positions)
        product_indices = np.searchsorted(product_starts, positions, side="right") - 1
        if not np.isin(product_indices, selected).all():
            raise ValueError(f"{name} record outside its expected ProductPrint block")
        indices = np.searchsorted(selected, product_indices)
        if not np.array_equal(event_indices, product_events[indices]):
            raise ValueError(f"{name} record crosses event boundary")
        if not np.all(np.bincount(indices, minlength=selected.size) == 1):
            raise ValueError(f"exactly one {name} record required per product")
        return event_indices

    status_products, status_events, status_counts = selected_products("StatusG4", _STATUS_TYPES, "g4run")
    if np.any(status_counts != 1):
        raise ValueError("exactly one StatusG4 product required per event")
    status_rows = _records(text, _STATUS_MARKER, _STATUS, "StatusG4")
    status_indices = row_owners(status_rows, status_products, status_events, "StatusG4")
    values = _integers([m.groups() for m in status_rows], 4, "StatusG4 values")
    if not np.isin(values[:, 0], [0, 1, 10]).all():
        raise ValueError("unknown StatusG4 status code")
    if not np.isin(values[:, 2], [0, 1]).all():
        raise ValueError("overflowSimP must be 0 or 1")
    ordered = np.empty((expected_events, 4), dtype=np.int64)
    ordered[status_indices] = values
    trigger_products, trigger_events, trigger_counts = selected_products("TriggerResults", _TRIGGER_TYPES, "TriggerResults")
    if require_triggers and np.any(trigger_counts != 1):
        raise ValueError("exactly one TriggerResults product required per event")
    trigger_rows = _records(text, _TRIGGER_MARKER, _TRIGGER, "PrimaryPath trigger")
    trigger_indices = row_owners(trigger_rows, trigger_products, trigger_events, "PrimaryPath trigger")
    _integers([(m.group(2),) for m in trigger_rows], 1, "trigger path IDs")
    accepted = np.full(expected_events, -1, dtype=np.int8)
    accepted[trigger_indices] = np.asarray([m.group(1) == "pass" for m in trigger_rows], dtype=np.int8)
    if require_triggers and np.any(accepted < 0):
        raise ValueError("missing PrimaryPath trigger")
    return dict(identities=identities, accepted=accepted, status=ordered[:, 0],
                nG4Tracks=ordered[:, 1], overflowSimP=ordered[:, 2], nStepLimit=ordered[:, 3],
                source_sha256=np.asarray(hashlib.sha256(raw).hexdigest()), process=np.asarray(process),
                expected_events=np.asarray(expected_events, dtype=np.int64),
                require_triggers=np.asarray(require_triggers, dtype=bool))


def summary(ledger):
    """Produce JSON-native counts; absent triggers are neither passes nor failures."""
    ids, status, accepted = ledger["identities"], ledger["status"], ledger["accepted"]
    codes, counts = np.unique(status, return_counts=True)
    return dict(events=int(ids.shape[0]), expected_events=int(ledger["expected_events"]),
                process=str(ledger["process"].item()), source_sha256=str(ledger["source_sha256"].item()),
                require_triggers=bool(ledger["require_triggers"]),
                status_counts={str(int(code)): int(count) for code, count in zip(codes, counts)},
                status_zero_events=int(np.count_nonzero(status == 0)),
                nonzero_status_events=int(np.count_nonzero(status != 0)),
                overflow_events=int(np.count_nonzero(ledger["overflowSimP"])),
                nG4Tracks=sum(map(int, ledger["nG4Tracks"])),
                killed_step_limit_tracks=sum(map(int, ledger["nStepLimit"])),
                accepted_events=int(np.count_nonzero(accepted == 1)),
                rejected_events=int(np.count_nonzero(accepted == 0)),
                missing_trigger_events=int(np.count_nonzero(accepted < 0)),
                event_id_min=ids.min(axis=0).tolist() if ids.size else None,
                event_id_max=ids.max(axis=0).tolist() if ids.size else None,
                scope="stock text event/status/trigger census; no physics completeness certification",
                complete_incident_census=False, zero_response_certified=False,
                generation_census_reconciled='generation_source_sha256' in ledger)


def main(argv=None):
    """Write an exclusive NPZ/JSON pair; --out is a prefix or an .npz filename."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-events", type=int, required=True)
    parser.add_argument("--process", default="R1Inclusive")
    parser.add_argument("--generation-log", type=Path, help="independent completed generation status census")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    npz = args.out if args.out.suffix == ".npz" else Path(str(args.out) + ".npz")
    report = npz.with_suffix(".json")
    for path in (npz, report):
        if path.exists():
            raise FileExistsError(f"refusing existing output: {path}")
    ledger = read_log(args.log, args.expected_events, process=args.process)
    if args.generation_log:
        original = read_log(args.generation_log, args.expected_events, process=args.process, require_triggers=False)
        original_order = np.lexsort(original['identities'].T[::-1])
        read_order = np.lexsort(ledger['identities'].T[::-1])
        for key in ('identities', 'status', 'nG4Tracks', 'overflowSimP', 'nStepLimit'):
            if not np.array_equal(original[key][original_order], ledger[key][read_order]):
                raise ValueError('saved-art read differs from independent generation census: ' + key)
        ledger['generation_source_sha256'] = original['source_sha256']
    encoded = json.dumps(summary(ledger), indent=2, allow_nan=False) + "\n"
    with npz.open("xb") as arrays:
        try:
            with report.open("x") as stream:
                np.savez_compressed(arrays, **ledger)
                stream.write(encoded)
        except BaseException:
            npz.unlink()
            raise
    print(encoded, end="")


if __name__ == "__main__":
    main()
