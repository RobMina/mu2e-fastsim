"""Bounded post-build source-export check against a previously certified baseline.

Reads numeric tables and a stock EventIDSequencePrinter text log only. This does
not build/run art, decode art objects, select events, or certify file identity.
The strict printer grammar matches the checked stock InputTag/EventID rendering;
unsupported formats fail closed rather than being guessed or partially parsed.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re

import awkward as ak
import numpy as np
import uproot

from . import source_events
from .export import EVENT, columns
from .source_split import _keys

SOURCE_TAG = "TargetStopResampler::Primary"
SOURCE_BRANCH = "art::EventIDs_TargetStopResampler__Primary."
SOURCE_IDS = ("source_run", "source_subrun", "source_event")
ADDED_EVENT_FIELDS = tuple(field for field in source_events.EVENT_FIELDS if field not in EVENT)
COMPLETION = "Art has completed and will exit with status 0."


def _new(path):
    # exists() alone misses dangling symlinks.
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)


def _census(events):
    ids = _keys(events, EVENT)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("baseline event census must be nonempty and unique")
    # Preserve the original integer dtypes, not the normalized join-key dtype.
    return {field: np.asarray(events[field]) for field in EVENT}


def read_census(path):
    """Full ordered dump/events IDs; baseline certification is an external premise."""
    with uproot.open(path) as root:
        return _census(columns(root, "events", EVENT))


def _inventory(root):
    classes = root.classnames(recursive=True, cycle=False)
    # Multiple cycles could hide an old schema or additional rows.
    if len(root.keys(recursive=True, cycle=True)) != len(classes):
        raise ValueError("duplicate ROOT key cycles")
    if any(kind not in ("TDirectory", "TTree") for kind in classes.values()):
        raise ValueError("only numeric TTree tables and directories are supported")
    return classes


def _equal(left, right, label):
    form_a, length_a, buffers_a = ak.to_buffers(ak.to_packed(left))
    form_b, length_b, buffers_b = ak.to_buffers(ak.to_packed(right))
    if form_a != form_b or length_a != length_b or buffers_a.keys() != buffers_b.keys():
        raise ValueError(f"{label}: dtype/shape mismatch")
    for key, a in buffers_a.items():
        b = buffers_b[key]
        if a.dtype != b.dtype or a.shape != b.shape:
            raise ValueError(f"{label}: dtype/shape mismatch")
        equal = np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
        if not equal:
            raise ValueError(f"{label}: ordered values mismatch")


def compare_tables(baseline, tables):
    """Compare every old column, filtering only new kind==-6 product rows."""
    with uproot.open(baseline) as old, uproot.open(tables) as new:
        before, after = _inventory(old), _inventory(new)
        if "dump/source_events" in before:
            raise ValueError("baseline already contains source_events")
        if after != {**before, "dump/source_events": "TTree"}:
            raise ValueError("ROOT table/directory schema mismatch")
        expected = _census(columns(old, "events", EVENT))
        if set(old["dump/events"].keys()) & set(ADDED_EVENT_FIELDS):
            raise ValueError("baseline already contains source event fields")
        if set(new["dump/source_events"].keys()) != set(source_events.SOURCE_FIELDS):
            raise ValueError("source_events column schema mismatch")
        old_kind = old["dump/products/kind"].array(library="np")
        new_kind = new["dump/products/kind"].array(library="np")
        if old_kind.dtype.kind not in "iu" or new_kind.dtype.kind not in "iu":
            raise ValueError("products.kind requires integer dtype")
        if np.any(old_kind == -6):
            raise ValueError("baseline contains kind==-6 source products")
        retained = new_kind != -6
        rows, count = {}, 0
        for name, kind in before.items():
            if kind != "TTree":
                continue
            left, right = old[name], new[name]
            fields = list(left.keys())
            added = set(ADDED_EVENT_FIELDS) if name == "dump/events" else set()
            if set(right.keys()) != set(fields) | added:
                raise ValueError(f"{name}: column schema mismatch")
            # New branches may be interleaved, but no old branch may move.
            if [field for field in right.keys() if field not in added] != fields:
                raise ValueError(f"{name}: old column order mismatch")
            rows[name] = int(left.num_entries)
            for field in fields:
                label = name + "/" + field
                if left[field].typename != right[field].typename:
                    raise ValueError(f"{label}: ROOT dtype mismatch")
                a, b = left[field].array(library="ak"), right[field].array(library="ak")
                if name == "dump/products":
                    b = b[retained]
                _equal(a, b, label)
                count += 1
    return expected, dict(old_table_rows=rows, old_columns_compared=count,
                          all_old_columns_exact=True, equal_nan=True,
                          filtered_product_kind=-6)


def _parse(text, pattern, fields):
    # Bounded decimal capture -> uint64 -> checked int64; never via float.
    rows = np.fromregex(io.StringIO(text), pattern,
                        dtype=[(field, np.uint64) for field in fields])
    if any(np.any(rows[field] > np.iinfo(np.int64).max) for field in fields):
        raise ValueError("printer integer identity/count overflow")
    return {field: rows[field].astype(np.int64) for field in fields}


def parse_printer(text, expected_events, *, source_tag):
    """Decode singleton source IDs independently, requiring complete ordered logs.

    ArtReport headers (%MSG ... / %MSG) and other unrelated log lines may occur
    between records. Every Begin-processing/sequence marker must match a complete
    supported line, alternate in order, and precede the sole event TrigReport.
    No Python loop is performed over events or text lines.
    """
    expected = _census(expected_events)
    n = len(expected["event"])
    h, number = r"[ \t]*", r"([0-9]{1,19})"
    art_pattern = (r"(?m)^[ \t]*Begin processing the " + number +
                   r"(?:st|nd|rd|th) record\. run:" + h + number +
                   r"[ \t]+subRun:" + h + number + r"[ \t]+event:" + h + number +
                   r"[ \t]+at[ \t]+\S[^\r\n]*\r?$")
    tag = source_tag.split(":")
    if len(tag) != 3 or not tag[0] or not tag[2]:
        raise ValueError("printer source_tag requires explicit label:instance:process")
    rendered = f"InputTag: label = '{tag[0]}', instance = '{tag[1]}', process = '{tag[2]}'"
    sequence_pattern = (r"(?m)^[ \t]*EventIDSequence[ \t]+" + re.escape(rendered) +
                        h + ":" + h + r"\(" + h + "run:" + h + number +
                        r"[ \t]+subRun:" + h + number + r"[ \t]+event:" + h + number +
                        h + r"\)" + h + r"\r?$")
    trig_pattern = (r"(?m)^[ \t]*TrigReport[ \t]+Events[ \t]+total" + h + "=" + h + number +
                    r"[ \t]+passed" + h + "=" + h + number +
                    r"[ \t]+failed" + h + "=" + h + number + h + r"\r?$")
    art = _parse(text, art_pattern, ("ordinal",) + EVENT)
    sources = _parse(text, sequence_pattern, SOURCE_IDS)
    trig = _parse(text, trig_pattern, ("total", "passed", "failed"))
    # Broad markers reject malformed, duplicate, orphan, unsupported and misordered rows.
    tokens = np.fromregex(io.StringIO(text),
                         r"(Begin[ \t]+processing\b|\bEventIDSequence\b|\([ \t]*run\b|TrigReport[ \t]+Events\b)",
                         dtype=[("token", "S32")])["token"].astype("S1")
    wanted = np.frombuffer(b"BE(" * n + b"T", dtype="S1")
    if not np.array_equal(tokens, wanted):
        raise ValueError("printer orphan/duplicate/malformed rows or ArtReport/sequence order")
    if len(art["ordinal"]) != n or len(sources["source_event"]) != n:
        raise ValueError("printer malformed/incomplete ArtReport or singleton sequence")
    if not np.array_equal(art["ordinal"], np.arange(1, n + 1)):
        raise ValueError("ArtReport record ordinal mismatch")
    if not np.array_equal(_keys(art, EVENT), _keys(expected, EVENT)):
        raise ValueError("ArtReport full ordered output event census mismatch")
    if (len(trig["total"]) != 1 or trig["total"][0] != n or
            trig["passed"][0] != n or trig["failed"][0] != 0):
        raise ValueError("TrigReport event counts mismatch")
    if (text.count("Art has completed and will exit") != 1 or
            re.search(r"(?m)^[ \t]*" + re.escape(COMPLETION) + r"\s*\Z", text) is None):
        raise ValueError("printer missing terminal successful art completion")
    return sources, dict(printer_events=n, printer_source_rows=n,
                         artreport_ordered_census_exact=True,
                         trigreport=dict(total=n, passed=n, failed=0),
                         terminal_success=True,
                         printer_format="stock InputTag/EventID text with per-event ArtReport; no art binary decoding")


def validate(baseline, tables, printer_log, *, source_tag, source_branch):
    """Validate all inputs before the CLI creates any output directory."""
    if (source_tag, source_branch) != (SOURCE_TAG, SOURCE_BRANCH):
        raise ValueError("this production check requires explicit TargetStopResampler::Primary and branch")
    expected, comparison = compare_tables(baseline, tables)
    result, report = source_events.read(tables, expected, source_tag=source_tag,
                                       source_branch=source_branch)
    decoded, printer_report = parse_printer(Path(printer_log).read_text(encoding="utf-8"),
                                           expected, source_tag=source_tag)
    for field in SOURCE_IDS:
        if not np.array_equal(decoded[field], result[field]):
            raise ValueError(f"stock printer/exported {field} mismatch")
    report.update(comparison=comparison, printer=printer_report,
                  printer_source_ids_exact=True,
                  unique_immediate_source_event_ids=int(len(np.unique(_keys(decoded, SOURCE_IDS)))),
                  unique_count_scope="diagnostic immediate (run,subrun,event) count only; no FID or independence inference",
                  baseline_certification="external prerequisite; not repeated by this check")
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    census = commands.add_parser("census", help="save the baseline full ordered integer event census")
    census.add_argument("--baseline", type=Path, required=True)
    census.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("validate", help="check numeric tables against baseline and stock printer")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--tables", type=Path, required=True)
    check.add_argument("--printer-log", type=Path, required=True)
    check.add_argument("--source-tag", required=True)
    check.add_argument("--source-branch", required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _new(args.out)
    paths = {"baseline": args.baseline}
    if args.command == "validate":
        paths.update(tables=args.tables, printer_log=args.printer_log)
    before = {name: source_events.file_digest(path) for name, path in paths.items()}
    if args.command == "census":
        if args.out.suffix != ".npz":
            raise ValueError("census output must have .npz suffix")
        result = read_census(args.baseline)
        report = dict(events=len(result["event"]), ordered_full_baseline_census=True,
                      baseline_certification="external prerequisite; not repeated by this check")
    else:
        result, report = validate(args.baseline, args.tables, args.printer_log,
                                  source_tag=args.source_tag, source_branch=args.source_branch)
    for name, path in paths.items():
        if source_events.file_digest(path) != before[name]:
            raise ValueError(f"{name} changed during source-event validation")
    report.update(input_sha256=before, source_sha256=source_events.file_digest(__file__))
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.command == "census":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("xb") as stream:
            np.savez(stream, **result)
    else:
        args.out.mkdir(parents=True)
        with (args.out / "source_events.npz").open("xb") as stream:
            np.savez(stream, **result)
        with (args.out / "summary.json").open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
