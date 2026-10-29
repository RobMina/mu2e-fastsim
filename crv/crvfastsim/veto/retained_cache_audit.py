"""Read-only real replay-cache admission and whole-plan source-row coverage measurement."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot

from . import batch_artifacts as A, frame_registry as F, indexed_replay as I
from . import retained_plan_report as PR, keyed_optical_smoke as M
from . import replay, exposure
from .replay import IDS, sha256

SCHEMA = "retained-indexed-cache-audit-v1"
OUTPUT_CAP = 8 * 1024**2
LOADER_SHA = "8b3b4ad3c1c35adb99149c4d0e76f662d032b4daab30f2f61d05c165a95a5d63"
LEDGER_COLUMNS = [*IDS, "bar", "t0", "t1", "edep", "step_row"]
SCOPE = ("Complete planned raw-cache selection coverage only. No optical/response generation, "
         "coincidences, track coverage, F, independent exposure or finite-library uncertainty. "
         "RSS high-water marks are cumulative for this process, including saved-plan verification.")


def code_paths():
    return [Path(m.__file__) for m in (I, A, F, PR, PR.P, M, replay, exposure)] + [Path(__file__)]


def memory():
    """Normalize process high-water RSS to bytes; Linux current RSS is a separate observation."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    current = None
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
                break
    return {"peak_rss_bytes_cumulative": peak, "current_rss_bytes": current}


def merge_bindings(*mappings):
    """Reject conflicting digests before a single deliberate unique-path hash pass."""
    merged = {}
    for mapping in mappings:
        for path, digest in mapping.items():
            I._digest(digest)
            if path in merged and merged[path] != digest:
                raise ValueError("conflicting input hash binding")
            merged[path] = digest
    return merged


def parquet_admit(path, expected_rows=None):
    """Bound saved-table metadata before any pandas/Arrow table materialization."""
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("saved Parquet file byte cap exceeded")
    with pq.ParquetFile(path) as parquet:
        meta = parquet.metadata
        if (meta.num_rows > 2_000_000 or meta.num_columns > 64 or meta.num_row_groups > 4096
                or (expected_rows is not None and meta.num_rows != expected_rows)):
            raise ValueError("saved Parquet row/schema admission failed")
        groups = [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)]
        if sum(groups) > 512 * 1024**2 or any(n < 0 or n > 256 * 1024**2 for n in groups):
            raise ValueError("saved Parquet uncompressed byte cap exceeded")


def preflight_saved(plan, report):
    """Preflight every bound Parquet, including inputs read by the frozen saved-plan verifier."""
    prior = report["admission_bindings"]
    if any(Path(a["path"]).name != a["path"] for a in report["artifacts"].values()):
        raise ValueError("unsafe plan artifact")
    mappings = [report["inputs"], report["code_sha256"], prior["assets"], prior["inputs"], prior["code"], *prior["blocks"]]
    artifacts = {str(plan / a["path"]): a["sha256"] for a in report["artifacts"].values()}
    bindings = merge_bindings(*mappings, artifacts)
    expected = {str(plan / a["path"]): a["rows"] for a in report["artifacts"].values()}
    for path, digest in bindings.items():
        if Path(path).suffix == ".parquet":
            parquet_admit(M.P.bound(path, digest), expected.get(path))


def _table(directory, artifact):
    if Path(artifact["path"]).name != artifact["path"]:
        raise ValueError("unsafe plan artifact")
    table = pd.read_parquet(M.P.bound(Path(directory) / artifact["path"], artifact["sha256"]))
    if len(table) != artifact["rows"]:
        raise ValueError("plan artifact row disagreement")
    return table


def admission(plan, expected_plan_sha256):
    """Repeat saved-only plan verification and bind the exact original timing row ledgers."""
    plan = Path(plan)
    path = M.P.bound(plan / "summary.json", expected_plan_sha256)
    report = json.loads(path.read_text())
    preflight_saved(plan, report)
    with tempfile.TemporaryDirectory(prefix="crv-cache-admission-") as tmp:
        verified = PR.verify(plan, Path(tmp) / "verified.json")
    registry, frames, batches = (_table(plan, report["artifacts"][n]) for n in ("registry", "frames", "batches"))
    if F.digest(registry) != verified["registry_sha256"]:
        raise ValueError("registry binding disagreement")
    smoke_path = Path(report["smoke"]) / "summary.json"
    smoke = json.loads(M.P.bound(smoke_path, report["inputs"][str(smoke_path)]).read_text())
    census_path = Path(smoke["paths"]["audit"]) / "summary.json"
    census = json.loads(M.P.bound(census_path, report["admission_bindings"]["inputs"][str(census_path)]).read_text())
    records = []
    for source in census["inputs"]:
        artifact = source["artifacts"]["steps"]
        ledger = census_path.parent / artifact["path"]
        if report["inputs"].get(str(ledger)) != artifact["sha256"]:
            raise ValueError("missing original row ledger binding")
        records.append({"path": source["path"], "input_sha256": source["sha256"],
                        "manifest_sha256": source["manifest_sha256"], "ledger": str(ledger),
                        "ledger_sha256": artifact["sha256"]})
    digests = [r["input_sha256"] for r in records]
    if len(set(digests)) != len(digests) or set(digests) != set(registry.input_sha256):
        raise ValueError("incomplete or duplicate input domain")
    bindings = {str(path): expected_plan_sha256}
    mappings = [report["inputs"], report["code_sha256"],
                {str(plan / a["path"]): a["sha256"] for a in report["artifacts"].values()}]
    prior = report["admission_bindings"]
    mappings += [prior["assets"], prior["inputs"], prior["code"], *prior["blocks"]]
    bindings = merge_bindings(bindings, *mappings)
    M.check_bindings(bindings)
    return registry, frames, batches, records, bindings, verified


def _integers(values, upper, label):
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= upper):
        raise ValueError(f"invalid {label}")
    return values.astype(np.int64, copy=False)


def fingerprint(table, name):
    """Hash typed source-order values, including original row and qualified frame identity."""
    h = hashlib.sha256((SCHEMA + ":" + name).encode())
    columns = {**I.TYPES[name], "source_row": "int64", "frame_id": "int64"}
    if name == "steps":
        columns["pdg"] = "int64"
    for column, dtype in columns.items():
        a = table[column].to_numpy(dtype=np.dtype(dtype).newbyteorder("<"), copy=True)
        if a.dtype.kind == "f":
            a[np.isnan(a)] = np.nan
            a[a == 0] = 0
        h.update(column.encode() + b"\0" + a.tobytes())
    h.update("".join(table.input_sha256).encode("ascii"))
    return h.hexdigest()


class Coverage:
    """Vectorized per-batch validation with one byte per source row for exact coverage."""

    def __init__(self, counts, registry, frames, ledger, digest):
        I._admit(counts, I.Limits())
        self.counts, self.registry, self.frames, self.ledger, self.digest = counts, registry, frames, ledger, digest
        self.registry_digest = F.digest(registry)
        self.seen = {name: np.zeros(counts[name], np.bool_) for name in I.TYPES}
        self.local = registry[registry.input_sha256 == digest]
        if (len(self.local) != counts["events"] or len(ledger) != counts["steps"]
                or not np.array_equal(ledger.step_row, np.arange(counts["steps"]))):
            raise ValueError("source census/ledger size or ordering disagreement")
        self.first_event = int(self.local.event.min())
        self.done_batches = set()

    def consume(self, selection, selected, batch):
        """Check whole-frame membership, independent saved step fields and particle joins."""
        number = int(batch["batch"])
        if number in self.done_batches or batch["input_sha256"] != self.digest:
            raise ValueError("duplicate or foreign batch")
        if selection.input_sha256 != self.digest or selection.registry_sha256 != self.registry_digest:
            raise ValueError("selection binding disagreement")
        expected = self.frames[self.frames.batch == number]
        pd.testing.assert_frame_equal(selected.reset_index(drop=True),
                                      self.registry.iloc[expected.frame_id][list(F.COLUMNS)].reset_index(drop=True), check_exact=True)
        if not len(selected) or len(selected) != int(batch["frames"]):
            raise ValueError("selected whole-frame batch disagreement")
        target = pd.MultiIndex.from_frame(selected[list(F.FRAME_KEYS)])
        positions, hashes, sizes = {}, {}, {}
        for name in I.TYPES:
            table = getattr(selection, name)
            required = {*I.TYPES[name], "input_sha256", "frame_id", "source_row"}
            if name == "steps":
                required.add("pdg")
            if set(table) != required or any(table[c].dtype != np.dtype(t) for c, t in I.TYPES[name].items()):
                raise ValueError(f"selection {name} schema disagreement")
            rows = _integers(table.source_row, self.counts[name], f"{name} source rows")
            if len(np.unique(rows)) != len(rows) or self.seen[name][rows].any():
                raise ValueError(f"duplicate {name} source rows")
            if name != "events" and np.any(rows[1:] <= rows[:-1]):
                raise ValueError("selection lost original source order")
            loc = target.get_indexer(pd.MultiIndex.from_frame(table[list(F.FRAME_KEYS)]))
            ids = _integers(table.frame_id, len(self.registry), "frame IDs")
            if np.any(loc < 0) or not np.array_equal(ids, selected.frame_id.to_numpy()[loc]):
                raise ValueError("selection full-tuple/frame disagreement")
            positions[name] = loc
            hashes[name], sizes[name] = fingerprint(table, name), len(table)
        events, steps, parts = selection.events, selection.steps, selection.particles
        pd.testing.assert_frame_equal(events[list(F.COLUMNS)].astype({c: "int64" for c in IDS}).reset_index(drop=True),
                                      selected.reset_index(drop=True), check_exact=True)
        if not np.array_equal(events.source_row, events.event.to_numpy(dtype=np.int64) - self.first_event):
            raise ValueError("event source-row mapping disagreement")
        pd.testing.assert_frame_equal(steps[LEDGER_COLUMNS[:-1]].reset_index(drop=True),
                                      self.ledger.iloc[steps.source_row][LEDGER_COLUMNS[:-1]].reset_index(drop=True), check_exact=True)
        particle_index = pd.MultiIndex.from_frame(parts[[*IDS, "key"]])
        if particle_index.has_duplicates:
            raise ValueError("duplicate selected particle identity")
        join = particle_index.get_indexer(pd.MultiIndex.from_frame(steps[[*IDS, "sp"]]))
        if (np.any(join < 0) or steps.pdg.dtype != np.dtype("int64")
                or not np.array_equal(steps.pdg, parts.pdg.to_numpy()[join])):
            raise ValueError("selected particle join disagreement")
        nsteps = np.bincount(positions["steps"], minlength=len(selected))
        finite = np.isfinite(steps.t0) & np.isfinite(steps.t1)
        nf = np.bincount(positions["steps"][finite], minlength=len(selected))
        if (not np.array_equal(nsteps, expected.steps) or not np.array_equal(nf, expected.finite_steps)
                or len(steps) != int(batch["steps"])):
            raise ValueError("per-frame raw/finite step census disagreement")
        typed = sum(sizes[n] * (I.DTYPES[n].itemsize + 272 + (8 if n == "steps" else 0)) for n in I.TYPES)
        if selection.typed_payload_bytes != typed:
            raise ValueError("selected typed byte accounting disagreement")
        for name in I.TYPES:
            self.seen[name][getattr(selection, name).source_row.to_numpy()] = True
        self.done_batches.add(number)
        frames = selected.copy()
        frames["batch"], frames["event_source_row"] = number, events.source_row.to_numpy()
        frames["steps"], frames["finite_steps"] = nsteps, nf
        frames["particles"] = np.bincount(positions["particles"], minlength=len(selected))
        metrics = {"batch": number, "input_sha256": self.digest, "frames": len(selected),
                   "steps": len(steps), "particles": len(parts), "typed_payload_bytes": typed,
                   **{f"{n}_sha256": h for n, h in hashes.items()}}
        return frames, metrics

    def finish(self):
        expected = set(self.frames.loc[self.frames.input_sha256 == self.digest, "batch"].unique())
        if self.done_batches != expected or any(not a.all() for a in self.seen.values()):
            raise ValueError("incomplete batch or source-row coverage")
        return {"counts": self.counts, "batches": len(expected), "all_source_rows_once": True,
                "coverage_bitmap_bytes": sum(a.nbytes for a in self.seen.values())}


def compare_snapshot(cache, selection):
    """Check every selected raw field against its immutable source-row slot, without ROOT rereads."""
    for name in I.TYPES:
        table = getattr(selection, name)
        raw = cache._tables[name]
        rows = _integers(table.source_row, len(raw), f"{name} source rows")
        for column, dtype in I.TYPES[name].items():
            actual = table[column].to_numpy()
            if actual.dtype != np.dtype(dtype) or not np.array_equal(actual, raw[column][rows], equal_nan=True):
                raise ValueError(f"selected raw snapshot disagreement: {name}.{column}")


def audit_input(record, registry, frames, batches):
    """Open one cache, walk its frozen batches, release every selection before the next."""
    started = time.perf_counter()
    before = memory()
    path = Path(record["path"])
    manifest = json.loads(M.P.bound(path.with_suffix(".json"), record["manifest_sha256"]).read_text())
    I._validate_manifest(manifest)
    counts = manifest["entries"]
    I._admit(counts, I.Limits())
    parquet_admit(M.P.bound(record["ledger"], record["ledger_sha256"]), counts["steps"])
    digest = F.digest(registry)
    metrics, ledgers = [], []
    with I.load_input(path, registry, expected_registry_sha256=digest,
                      expected_input_sha256=record["input_sha256"], expected_manifest_sha256=record["manifest_sha256"]) as cache:
        loaded_at, loaded_memory = time.perf_counter(), memory()
        ledger = pd.read_parquet(M.P.bound(record["ledger"], record["ledger_sha256"]), columns=LEDGER_COLUMNS)
        coverage = Coverage(counts, registry, frames, ledger, record["input_sha256"])
        for batch in batches[batches.input_sha256 == record["input_sha256"]].to_dict("records"):
            selected = registry.iloc[frames.loc[frames.batch == batch["batch"], "frame_id"]]
            tick = time.perf_counter()
            selection = cache.select(selected, expected_registry_sha256=digest)
            selected_at = time.perf_counter()
            compare_snapshot(cache, selection)
            checked, receipt = coverage.consume(selection, selected, batch)
            receipt.update(select_seconds=selected_at - tick, validate_seconds=time.perf_counter() - selected_at)
            ledgers.append(checked)
            metrics.append(receipt)
            del selection
        result = {**record, **coverage.finish(), "resident_array_bytes": cache.resident_array_bytes,
                  "load_seconds": loaded_at - started, "selection_audit_seconds": time.perf_counter() - loaded_at,
                  "memory_before": before, "memory_loaded": loaded_memory, "memory_audited": memory()}
        del coverage, ledger
    gc.collect()
    result["memory_released"] = memory()
    result["elapsed_seconds"] = time.perf_counter() - started
    return pd.concat(ledgers, ignore_index=True), pd.DataFrame(metrics), result


def run(plan, out, *, expected_plan_sha256):
    out = Path(out)
    if out.exists():
        raise ValueError("choose a new cache-audit directory")
    M.P.bound(I.__file__, LOADER_SHA)
    code = {str(path): sha256(path) for path in code_paths()}
    started, initial = time.perf_counter(), memory()
    registry, frames, batches, records, bindings, verified = admission(plan, expected_plan_sha256)
    admitted = memory()
    context = {"registry_sha256": F.digest(registry), "batch_identity": {"plan_sha256": expected_plan_sha256,
               "contract": SCHEMA}, "arms": ["cache_audit"], "seeds": [0]}
    names = ["registry.parquet", "frames.parquet", "batches.parquet", "admission.json", "summary.json"]
    names += [f"code{i}.txt" for i in range(len(code))]
    frame_ledgers, batch_ledgers, results = [], [], []
    with A.BatchArtifacts(out, max_bytes=OUTPUT_CAP, context=context, expected_artifacts=names) as tx:
        tx.write_json("admission.json", verified)
        tx.write_parquet("registry.parquet", registry)
        for i, path in enumerate(code):
            with tx.open_artifact(f"code{i}.txt") as stream:
                stream.write(Path(path).read_bytes())
        for record in records:
            f, b, result = audit_input(record, registry, frames, batches)
            frame_ledgers.append(f)
            batch_ledgers.append(b)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        f = pd.concat(frame_ledgers, ignore_index=True).sort_values("frame_id").reset_index(drop=True)
        b = pd.concat(batch_ledgers, ignore_index=True).sort_values("batch").reset_index(drop=True)
        pd.testing.assert_frame_equal(f[list(F.COLUMNS)], registry, check_exact=True)
        if not np.array_equal(b.batch, batches.batch) or int(f.steps.sum()) != int(frames.steps.sum()):
            raise ValueError("global retained coverage disagreement")
        tx.write_parquet("frames.parquet", f)
        tx.write_parquet("batches.parquet", b)
        M.check_bindings(bindings, code)
        result = {"schema": SCHEMA, "scope": SCOPE, "plan": str(Path(plan)), "plan_sha256": expected_plan_sha256,
                  "registry_sha256": F.digest(registry), "inputs": bindings, "code_sha256": code,
                  "limits": asdict(I.Limits()), "output_cap": OUTPUT_CAP, "records": results,
                  "frames": len(f), "batches": len(b), "steps": int(f.steps.sum()), "particles": int(f.particles.sum()),
                  "all_source_rows_once": True, "whole_frames_exact": True, "execution_admitted": False,
                  "response_executed": False, "optics_executed": False, "F": None, "runtime_projection": None,
                  "finite_library_uncertainty": None, "memory_initial": initial, "memory_after_admission": admitted,
                  "memory_final": memory(), "elapsed_seconds_before_commit": time.perf_counter() - started,
                  "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                  "pandas": pd.__version__, "uproot": uproot.__version__}}
        tx.write_json("summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    a = parser.parse_args(argv)
    print(json.dumps(run(a.plan, a.out, expected_plan_sha256=a.expected_plan_sha256), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
