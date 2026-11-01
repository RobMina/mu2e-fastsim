"""Prospective fasttable integrity receipts; never stock-producer or named V2 certification.

A live input snapshot must precede output creation. Validation reopens all assets,
checks the current implementation and strict conditions, and reports only bounded
conditional-response evidence. There is deliberately no retroactive receipt CLI.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
import hashlib
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

SCHEMA = "crvfastsim.response_receipt.v1"
METADATA_SCHEMA = "crv-fasttables-named-v1"
KEYS = ("run", "subrun", "event")
LIMITATIONS = ("stock_source_configuration_not_certified", "no_transport_or_e2e_admission",
               "not_full_named_v2_certification", "integrity_not_authenticity")
_COLUMNS = {
    "digis": {"int": "idx bar sipm starttdc nzs roc feb febch", "float": "", "jagged": "adc"},
    "pulses": {"int": "idx bar sipm seq flags pedfromdb", "float": "pe peph height beta chi2 ped time letime",
               "jagged": "wfidx"},
    "clusters": {"int": "idx sectortype poscalc sidehits0 sidehits1",
                 "float": "start end pe slope avgtime x y z sidepe0 sidepe1 sidetime0 sidetime1",
                 "jagged": "layers pulses"},
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_canonical(value))


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))


def _record(path):
    p = Path(path).resolve(strict=True)
    _require(p.is_file(), f"not a file: {p}")
    before = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = p.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
             (after.st_size, after.st_mtime_ns, after.st_ctime_ns), f"file changed while hashing: {p}")
    return {"path": str(p), "sha256": digest.hexdigest(), "size": after.st_size}


def _sources():
    root = Path(__file__).resolve().parent
    names = ["response_receipt.py", "chain.py", "readout.py", "geometry.py", "tables.py", "runconfig.py", "named_conditions.py",
             "reco/pulsefit.py", "reco/coinc.py", "reco/cli.py", "reco/coinc_jax.py", "reco/chain_jax.py",
             "reco/waveform.py", "reco/waveform_jax.py", "reco/pulsefit_jax.py",
             "response/photons.py", "response/photons_jax.py", "response/sipm.py", "response/sipm_jax.py",
             "response/lookup.py", "response/stockphotons.py", "response/stocksipm.py",
             "response/device.py", "response/contract.py", "response/shared_yield.py", "response_config.py"]
    return {**{name: _record(root / name) for name in names},
            "valhist/fasttables.py": _record(root.parent / "valhist/fasttables.py")}


class _InputSnapshot(dict):
    """A process-local guard against accidentally issuing receipts from saved hashes."""


def snapshot_inputs(mapping):
    """Hash named input paths before production; retain this live mapping for write_receipt.

    Reserve ``input`` for the producer input and ``event_reference`` for an optional
    stock population file. Additional named assets are permitted and all are rehashed.
    """
    _require(isinstance(mapping, Mapping) and bool(mapping), "input mapping must be nonempty")
    _require(all(isinstance(k, str) and k for k in mapping), "input names must be nonempty strings")
    result = _InputSnapshot({name: _record(path) for name, path in mapping.items()})
    result._original = _canonical(result)
    result._sources = _sources()
    result._completed_ns = time.time_ns()
    result._used = False
    return result


def _verify_records(records, label):
    _require(isinstance(records, dict) and bool(records), f"missing {label} records")
    for name, record in records.items():
        _require(isinstance(record, dict) and set(record) == {"path", "sha256", "size"},
                 f"invalid {label} record: {name}")
        _require(isinstance(record["path"], str) and Path(record["path"]).is_absolute(),
                 f"invalid {label} path: {name}")
        _require(_record(record["path"]) == record, f"{label} drift: {name}")


def _integers(a, label, maximum=None, minimum=0):
    a = np.asarray(a)
    _require(a.ndim == 1 and a.dtype.kind in "iu", f"{label}: integer column required")
    _require(np.all(a >= minimum) and (maximum is None or np.all(a <= maximum)),
             f"{label}: integer range exceeded")
    return a


def _matrix(tree):
    _require(set(KEYS).issubset(tree.keys()), "missing full event identity columns")
    cols = tree.arrays(list(KEYS), library="np")
    return np.column_stack([_integers(cols[k], k, 2**32 - 1).astype("<u8") for k in KEYS])


def _sorted_ids(tree):
    ids = _matrix(tree)
    _require(len(ids) > 0, "empty event universe")
    order = np.lexsort((ids[:, 2], ids[:, 1], ids[:, 0]))
    ids = np.ascontiguousarray(ids[order], dtype="<u8")
    _require(not np.any(np.all(ids[1:] == ids[:-1], axis=1)), "duplicate event identity")
    return ids


def _identity(ids):
    return {"encoding": "sorted_run_subrun_event_uint64_le_matrix_v1", "count": len(ids),
            "sha256": hashlib.sha256(ids.tobytes(order="C")).hexdigest()}


def event_identity(root_path):
    """Digest unique sorted full-width uint32 run/subrun/event, encoded as little-endian uint64.

    This standalone identity helper accepts all uint32 keys; producer receipts apply
    the narrower single-run, 20-bit subrun/event packing contract separately.
    """
    with uproot.open(root_path) as f:
        _require("crvdump/events" in f, "missing crvdump/events")
        return _identity(_sorted_ids(f["crvdump/events"]))


def _packed(ids):
    _require(len(np.unique(ids[:, 0])) <= 1, "fasttable requires one run")
    _require(np.all(ids[:, 1:] < 2**20), "fasttable packed event-key capacity exceeded")


def _rows(ids):
    return np.ascontiguousarray(ids, dtype="<u8").view(np.dtype([("r", "<u8"), ("s", "<u8"),
                                                               ("e", "<u8")])).reshape(-1)


def _membership(rows, ids, label):
    index = np.searchsorted(_rows(ids), _rows(rows))
    _require(np.all(index < len(ids)), f"{label}: row event outside events tree")
    _require(np.array_equal(ids[index], rows), f"{label}: row event outside events tree")
    return index


def _tables(root_path, metadata):
    stage, nbars = metadata["stage"], metadata["conditions"]["geometry"]["nbars"]
    names = ("digis",) if stage == "digis" else ("digis", "pulses", "clusters")
    with uproot.open(root_path) as f:
        _require("crvdump/events" in f, "missing events tree")
        tree = f["crvdump/events"]
        ids = _sorted_ids(tree)
        _packed(ids)
        event_order = _membership(_matrix(tree), ids, "events")
        counts = {}
        tables = {}
        for name in names:
            _require(f"crvdump/{name}" in f, f"missing stage table: {name}")
            t = f[f"crvdump/{name}"]
            spec = _COLUMNS[name]
            required = set(KEYS) | set(" ".join(spec.values()).split())
            _require(required.issubset(t.keys()), f"{name}: missing required columns")
            a = t.arrays(sorted(required), library="ak")
            row_ids = _matrix(t)
            row_index = _membership(row_ids, ids, name)
            count = np.bincount(row_index, minlength=len(ids))
            counts[name] = count
            for col in spec["int"].split():
                minimum = -(2**31) if col == "starttdc" else -1 if col in ("roc", "feb", "febch") else 0
                _integers(ak.to_numpy(a[col]), f"{name}.{col}", maximum=2**31 - 1, minimum=minimum)
            for col in spec["float"].split():
                values = ak.to_numpy(a[col])
                _require(values.ndim == 1 and values.dtype.kind in "iuf" and np.all(np.isfinite(values)),
                         f"{name}.{col}: nonfinite/non-numeric values")
            _require(np.all(row_index[1:] >= row_index[:-1]), f"{name}: rows not in event order")
            idx = ak.to_numpy(a.idx)
            starts = np.cumsum(count) - count
            _require(np.array_equal(idx, np.arange(len(idx)) - starts[row_index]),
                     f"{name}: indices not in collection order")
            _require(np.all(idx < count[row_index]), f"{name}: row index range")
            pairs = np.column_stack((row_index, idx))
            _require(len(np.unique(pairs, axis=0)) == len(idx), f"{name}: duplicate row index")
            if name != "clusters":
                _require(np.all(ak.to_numpy(a.bar) < nbars) and np.all(ak.to_numpy(a.sipm) < 4),
                         f"{name}: bar/SiPM range")
            for col in spec["jagged"].split():
                try:
                    flat = ak.to_numpy(ak.flatten(a[col], axis=1))
                    lengths = ak.to_numpy(ak.num(a[col], axis=1))
                except (ValueError, TypeError, IndexError) as exc:
                    raise ValueError(f"{name}.{col}: jagged integer column required") from exc
                _integers(flat, f"{name}.{col}", minimum=-32768 if col == "adc" else 0)
                if col == "adc":
                    _require(np.all(flat <= 32767) and np.all(lengths > 0), "invalid ADC samples")
                if col in ("wfidx", "pulses"):
                    target = "digis" if col == "wfidx" else "pulses"
                    _require(np.all(flat < np.repeat(counts[target][row_index], lengths)),
                             f"{name}.{col}: collection index range")
            tables[name] = len(row_ids)
        for name, col in (("digis", "ndigi"), ("pulses", "npulse"), ("clusters", "ncluster")):
            _require(col in tree.keys(), f"events: missing {col}")
            actual = _integers(tree[col].array(library="np"), col)
            expected = counts.get(name, np.zeros(len(ids), dtype=int))[event_order]
            _require(np.array_equal(actual, expected), f"events: {col} count mismatch")
        if stage == "digis":
            _require(not any(f"crvdump/{n}" in f for n in ("pulses", "clusters")), "digis stage has reco tables")
    return ids, tables


def _window(value, label):
    _require(isinstance(value, (list, tuple)) and len(value) == 2 and
             all(type(x) in (int, float) and np.isfinite(x) for x in value) and value[0] < value[1],
             f"invalid {label}")


def _metadata(metadata):
    from . import named_conditions
    m = _copy(metadata)
    _require(isinstance(m, dict) and m.get("schema") == METADATA_SCHEMA, "unsupported named metadata schema")
    _require(m.get("stage") in ("reco", "chain", "digis"), "invalid metadata stage")
    _require("diagnostic_override" not in m and "geometry_readout" not in m and "response_execution" not in m,
             "diagnostic/readout/execution overrides are not normal receipt admission")
    _require(isinstance(m.get("run_config"), str) and m["run_config"] in named_conditions.GEOMETRIES,
             "unsupported metadata run_config")
    c = m.get("conditions")
    _require(isinstance(c, dict) and c.get("schema") == named_conditions.SCHEMA, "missing embedded conditions")
    _require(c.get("run_config") == m["run_config"] and isinstance(c.get("config"), dict),
             "metadata/conditions run_config mismatch")
    from .runconfig import RunConfig
    config = c["config"]
    _require(set(config) == {f.name for f in dataclasses.fields(RunConfig)}, "incomplete embedded run configuration")
    _require(config.get("name") == m["run_config"] and
             hashlib.sha256(_canonical(config).encode()).hexdigest() == c.get("config_sha256"),
             "invalid embedded config identity")
    _require(isinstance(c.get("geometry"), dict) and type(c["geometry"].get("nbars")) is int and
             0 < c["geometry"]["nbars"] <= 8192, "invalid embedded geometry/channel capacity")
    for key in ("configured_window_ns", "effective_window_ns"):
        _window(m.get(key), key)
    _require(m["configured_window_ns"] == config.get("window"), "configured window/config mismatch")
    expected = {"offspill": [0.0, 100000.0], "onspill": [200.0, 1500.0]}
    _require(isinstance(m.get("window_selection"), str) and m["window_selection"] in expected and
             m["effective_window_ns"] == expected[m["window_selection"]], "effective window/selection mismatch")
    _require(m.get("timing_policy") == "fixed_absolute_crop_no_pbtime_rebase" and
             m.get("onspill_equivalence") is False, "unsupported timing policy/onspill equivalence")
    _require(m.get("window_applied") is (m["stage"] != "reco"), "stage/window_applied mismatch")
    _require(type(m.get("seed")) is int and m["seed"] >= 0 and
             type(m.get("jax")) is bool and type(m.get("noise")) is bool, "invalid producer seed/backend/noise")
    _require(isinstance(m.get("conditions_manifest"), str) and bool(m["conditions_manifest"]),
             "missing conditions_manifest")
    _require(isinstance(m.get("finder"), dict) and bool(m["finder"]), "missing finder settings")
    if "chain_config" in m:
        from .chain import ChainConfig
        cfg = m["chain_config"]
        _require(isinstance(cfg, dict) and set(cfg) == {f.name for f in dataclasses.fields(ChainConfig)},
                 "incomplete producer chain_config")
        _require(cfg["window"] == m["effective_window_ns"] and cfg["finder"] == m["finder"] and
                 cfg["jax"] is m["jax"] and cfg["noise"] is m["noise"], "producer chain_config mismatch")
    return m


def _population(inputs, metadata, ids):
    _require("input" in inputs, "snapshot requires producer input named 'input'")
    stage = metadata["stage"]
    if stage != "reco":
        with uproot.open(inputs["input"]["path"]) as f:
            for name in ("steps", "particles"):
                _require(f"dump/{name}" in f, f"missing input dump/{name}")
                rows = _matrix(f[f"dump/{name}"])
                _packed(rows)
                _require(not len(rows) or np.all(rows[:, 0] == ids[0, 0]), "input/output run mismatch")
    name = "input" if stage == "reco" else "event_reference"
    if name not in inputs:
        return {"status": "unverified_population_no_event_reference", "window_status": "unverified_no_stock_events"}
    with uproot.open(inputs[name]["path"]) as f:
        _require("crvdump/events" in f, f"{name}: missing stock events")
        tree = f["crvdump/events"]
        source = _sorted_ids(tree)
        _packed(source)
        _require(np.array_equal(source, ids), f"{name}: output event universe mismatch")
        if stage == "reco":
            _require("crvdump/digis" in f, "reco input missing digis")
            _membership(_matrix(f["crvdump/digis"]), source, "input digis")
        columns = ("spill", "evlen", "pbtime")
        present = [col in tree.keys() for col in columns]
        for col, value, exists in zip(columns, (0, 100000, 0), present):
            if exists:
                a = tree[col].array(library="np")
                _require(a.ndim == 1 and a.dtype.kind in "iuf" and np.all(np.isfinite(a)),
                         f"stock {col}: nonfinite/non-numeric window")
                if metadata["window_selection"] == "offspill":
                    _require(np.all(a == value), f"stock {col}: not fixed offspill 0..100000/pbtime=0")
        window = ("verified_fixed_offspill" if all(present) and metadata["window_selection"] == "offspill"
                  else "unsupported_onspill" if metadata["window_selection"] == "onspill"
                  else "unverified_missing_stock_window_columns")
    return {"status": "verified_exact_stock_event_universe", "source": name, "events": _identity(source),
            "window_status": window}


def write_receipt(output, metadata, inputs_before, *, path=None):
    """Exclusively publish OUT.receipt.json from a live, pre-production snapshot; return Path.

    Output and inputs must be stable. Metadata is bundled unchanged (JSON-normalized),
    including the complete producer chain_config when supplied. No sidecar is inferred.
    """
    output = Path(output).resolve(strict=True)
    destination = Path(path) if path is not None else Path(str(output) + ".receipt.json")
    _require(not destination.exists() and not destination.is_symlink(), "receipt already exists")
    _require(isinstance(inputs_before, _InputSnapshot) and not inputs_before._used,
             "a live unused prospective snapshot_inputs result is required")
    _require(_canonical(inputs_before) == inputs_before._original, "input snapshot was modified")
    stat = output.stat()
    _require(min(stat.st_mtime_ns, stat.st_ctime_ns) > inputs_before._completed_ns,
             "output predates input snapshot; historical output cannot gain a receipt")
    _require(str(output) not in [r["path"] for r in inputs_before.values()], "output aliases an input")
    m = _metadata(metadata)
    _verify_records(inputs_before, "input")
    _require(_sources() == inputs_before._sources, "producer source drift since snapshot")
    out = _record(output)
    ids, tables = _tables(output, m)
    population = _population(inputs_before, m, ids)
    receipt = {"schema": SCHEMA, "scope": "prospective_fast_output_integrity_only", "metadata": m,
               "inputs": dict(inputs_before), "output": out, "sources": inputs_before._sources,
               "events": _identity(ids), "tables": tables, "population": population,
               "snapshot_completed_ns": inputs_before._completed_ns, "limitations": list(LIMITATIONS)}
    _require(_record(output) == out, "output drift during receipt construction")
    _verify_records(inputs_before, "input")
    _require(_sources() == receipt["sources"], "producer source drift during receipt construction")
    text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with destination.open("x") as stream:
        stream.write(text)
    inputs_before._used = True
    return destination


def validate_receipt(output, receipt_path, *, run_config, geom, conditions_manifest):
    """Revalidate bytes and requested assets; return compact evidence, never a V2 pass.

    Paths are fixed (no relocation). Missing stock event/window evidence is explicit;
    onspill is never admitted, even when a metadata flag claims equivalence.
    """
    from . import named_conditions, runconfig
    rc = runconfig.get(run_config) if isinstance(run_config, str) else run_config
    r = _json(receipt_path)
    _require(isinstance(r, dict) and r.get("schema") == SCHEMA and
             r.get("scope") == "prospective_fast_output_integrity_only", "unsupported receipt schema/scope")
    m = _metadata(r.get("metadata"))
    _require(m["run_config"] == rc.name, "requested run_config mismatch")
    _, conditions = named_conditions.load_conditions(conditions_manifest, rc, geom)
    _require(_canonical(m["conditions"]) == _canonical(conditions), "embedded/external conditions metadata mismatch")
    _require(m["conditions"]["config"] == _copy(dataclasses.asdict(rc)), "requested config mismatch")
    _require(m["conditions"]["geometry"] == named_conditions._geometry(geom), "requested geometry mismatch")
    _require(m["configured_window_ns"] == list(rc.window), "requested configured window mismatch")
    _require(m["finder"] == _copy(dataclasses.asdict(rc.finder())), "unsupported finder settings")
    if "chain_config" in m:
        cfg = dataclasses.replace(rc.chain_config(jax=m["jax"], noise=m["noise"]),
                                  window=tuple(m["effective_window_ns"]))
        _require(m["chain_config"] == _copy(dataclasses.asdict(cfg)), "unsupported producer chain_config")
    _verify_records(r.get("inputs"), "input")
    _verify_records(r.get("sources"), "source")
    _require(r["sources"] == _sources(), "receipt/current producer source mismatch")
    out = _record(output)
    _require(out == r.get("output"), "output hash/path/size drift")
    _require(type(r.get("snapshot_completed_ns")) is int and r["snapshot_completed_ns"] > 0,
             "missing prospective snapshot marker")
    ids, tables = _tables(output, m)
    _require(_identity(ids) == r.get("events") and tables == r.get("tables"), "output event/table identity mismatch")
    population = _population(r["inputs"], m, ids)
    _require(population == r.get("population"), "input population evidence mismatch")
    limitations = list(LIMITATIONS)
    limitations.append("runtime_dependencies_not_certified")
    if m["stage"] != "reco":
        limitations.append("lookup_photon_map_and_waveform_payloads_not_automatically_bound")
    if "chain_config" not in m:
        limitations.append("full_producer_chain_configuration_not_recorded")
    conditional = (m["window_selection"] == "offspill" and
                   population["window_status"] == "verified_fixed_offspill" and "chain_config" in m)
    _verify_records(r["inputs"], "input")
    _require(_record(output) == out and _json(receipt_path) == r, "output/receipt drift during validation")
    _require(_sources() == r["sources"], "producer source drift during validation")
    return {"status": "bound_fast_output_and_assets", "stage": m["stage"], "run_config": rc.name,
            "events": r["events"], "population": population, "conditional_response_supported": conditional,
            "timing_status": "unsupported_onspill" if m["window_selection"] == "onspill" else population["window_status"],
            "named_v2_pass": False, "transport_admitted": False, "e2e_admitted": False,
            "limitations": limitations}
