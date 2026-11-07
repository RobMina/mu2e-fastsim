"""Validate opt-in numeric source EventIDs without inferring file or particle identity."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import uproot

from .export import EVENT, columns, join
from .source_split import _columns, _keys

EVENT_FIELDS = EVENT + ("source_present", "source_product_id", "source_count", "source_tag", "source_branch")
SOURCE_FIELDS = EVENT + ("product_id", "index", "source_valid", "source_run", "source_subrun", "source_event")
PRODUCT_FIELDS = EVENT + ("product_id", "kind", "size", "present", "branch_name")


def _integers(table, fields, label):
    result = dict(zip(fields, _columns(table, fields)))
    for field in fields:
        if field in ("product_id", "source_product_id"):
            if np.any(result[field] > 2 ** 32 - 1):
                raise ValueError(f"{label}: invalid or overflowing art identity {field}")
    return result


def _booleans(table, field, length):
    value = np.asarray(table[field])
    if value.dtype.kind != "b" or value.shape != (length,):
        raise ValueError(f"{field} requires event-aligned booleans")
    if not np.all(value):
        raise ValueError(f"{field} is false; missing or invalid source provenance")
    return value


def _text(table, field, length, expected):
    value = np.asarray(table[field])
    if value.ndim != 1 or len(value) != length or value.dtype.kind not in "UO":
        raise ValueError(f"{field} requires an aligned string column")
    if not np.all(value == expected):
        raise ValueError(f"{field} differs from explicitly expected source product {expected!r}; observed {np.unique(value)[:4]!r}")


def validate(events, sources, products, expected_events, *, source_tag, source_branch):
    """Check integer identities/counts/registry against an independent ordered event census.

    No status, incident-crossing or shower cut is applied. Source IDs refer only to
    immediate secondary events; file resolution remains a separate verified gate.
    """
    tag = source_tag.split(":")
    if len(tag) != 3 or not tag[0] or not tag[2] or any(c.isspace() for c in source_tag):
        raise ValueError("source_tag must explicitly specify label:instance:process")
    if source_branch != "art::EventIDs_" + "_".join(tag) + ".":
        raise ValueError("expected branch does not describe the explicit source_tag")
    evt = _integers(events, EVENT + ("source_product_id", "source_count"), "events")
    exp = _integers(expected_events, EVENT, "expected events")
    ids, expected = _keys(evt, EVENT), _keys(exp, EVENT)
    n = len(ids)
    if not n:
        raise ValueError("empty export cannot validate source provenance")
    if len(np.unique(ids)) != n or len(np.unique(expected)) != len(expected):
        raise ValueError("duplicate output event identity")
    if not np.array_equal(ids, expected):
        raise ValueError("ordered full event census mismatch")
    _booleans(events, "source_present", n)
    if np.any(evt["source_count"] != 1):
        raise ValueError("CE source provenance requires exactly one secondary per output event")
    if np.any(evt["source_product_id"] == 0):
        raise ValueError("invalid source ProductID")
    _text(events, "source_tag", n, source_tag)
    _text(events, "source_branch", n, source_branch)

    src = _integers(sources, EVENT + ("product_id", "index", "source_run", "source_subrun", "source_event"), "source rows")
    source_ids = _keys(src, EVENT)
    rows = join(ids, source_ids)
    if not np.array_equal(np.bincount(rows, minlength=n), evt["source_count"]):
        raise ValueError("source row counts disagree with full event census")
    if not np.array_equal(source_ids, ids):
        raise ValueError("source rows not in exported event order")
    _booleans(sources, "source_valid", n)
    if np.any(src["index"] != 0) or not np.array_equal(src["product_id"], evt["source_product_id"]):
        raise ValueError("source sequence index or ProductID mismatch")

    kind = np.asarray(products["kind"])
    if kind.ndim != 1 or kind.dtype.kind not in "iu":
        raise ValueError("invalid product registry kind")
    chosen = kind == -6
    for field in PRODUCT_FIELDS:
        if np.asarray(products[field]).shape != kind.shape:
            raise ValueError("product registry column lengths differ")
    prod = {field: np.asarray(products[field])[chosen] for field in PRODUCT_FIELDS}
    p = _integers(prod, EVENT + ("product_id", "size"), "product registry")
    product_ids = _keys(p, EVENT)
    if len(product_ids) != n or len(np.unique(product_ids)) != n:
        raise ValueError("source product registry requires one row per event")
    order = join(product_ids, ids)
    _booleans(prod, "present", n)
    _text(prod, "branch_name", n, source_branch)
    if not np.array_equal(p["product_id"][order], evt["source_product_id"]) or not np.array_equal(p["size"][order], evt["source_count"]):
        raise ValueError("source product registry ID/count mismatch")
    result = {field: evt[field] for field in EVENT + ("source_count", "source_product_id")}
    result.update({field: src[field] for field in ("source_run", "source_subrun", "source_event")})
    result.update(source_present=np.ones(n, bool), source_file_resolved=np.zeros(n, bool))
    report = dict(events=n, source_rows=len(source_ids), source_tag=source_tag, source_branch=source_branch,
                  ordered_event_census_exact=True, source_file_resolution="not_performed",
                  source_independence_certified=False,
                  scope="immediate secondary EventIDs only; no original-file or stopped-muon-key inference")
    return result, report


def read(path, expected_events, *, source_tag, source_branch):
    with uproot.open(path) as source:
        events = columns(source, "events", EVENT_FIELDS)
        sources = columns(source, "source_events", SOURCE_FIELDS)
        products = columns(source, "products", PRODUCT_FIELDS)
    return validate(events, sources, products, expected_events, source_tag=source_tag, source_branch=source_branch)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--expected-events", type=Path, required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    expected_bytes = args.expected_events.read_bytes()
    with np.load(io.BytesIO(expected_bytes), allow_pickle=False) as archive:
        expected = dict(archive)
    before = file_digest(args.tables)
    result, report = read(args.tables, expected, source_tag=args.source_tag, source_branch=args.source_branch)
    if file_digest(args.tables) != before:
        raise ValueError("tables changed during source-event validation")
    report.update(input_sha256=dict(tables=before, expected_events=hashlib.sha256(expected_bytes).hexdigest()),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.mkdir(parents=True)
    np.savez(args.out / "source_events.npz", **result)
    (args.out / "summary.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
