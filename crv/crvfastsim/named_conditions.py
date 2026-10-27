"""Strict stock-job conditions snapshots; these are text overlays, not database IOVs."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import uproot

SCHEMA = "crvfastsim.named_conditions.v1"
SCOPE = "explicit text-file all-run overlay on Sim_best/v1_5; not a database IOV lookup"
INPUTS = ("crvreco.root", "resolved.fcl", "manifest.txt", "calib_source.txt", "status_source.txt",
          "photonyield_source.txt", "ordinal_source.txt", "stock.log")
CALIB = ("channel", "pedestal", "pulsearea", "pulseheight", "timeoffset", "status", "roc", "feb", "febch")
EVENTS = ("run", "subrun", "event", "spill", "evlen", "pbtime")
GEOMETRIES = {"run1a_v01": "run1a_v01", "run2_v10": "nominal_v10"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            _require(key not in out, f"duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"invalid JSON {s}")))


def _integers(values, label, minimum=0):
    a = np.asarray(values)
    _require(a.dtype.kind in "iuf" and np.all(np.isfinite(a)), f"{label}: nonfinite/non-numeric integers")
    _require(np.all(a >= minimum) and np.all(a < 2**53) and np.all(a == np.floor(a)),
             f"{label}: invalid integer/range")
    return a.astype(np.int64)


def _channels(rows, n, label, dense=True):
    ch = _integers(rows[:, 0], label + " channel")
    order = np.argsort(ch)
    ch = ch[order]
    _require(np.all(ch < n) and np.all(np.diff(ch) > 0), f"{label}: duplicate/out-of-range channel")
    if dense:
        _require(np.array_equal(ch, np.arange(n)), f"{label}: exact channel coverage 0..{n - 1} required")
    return rows[order]


def _text_tables(path, schemas):
    tables, active = {}, None
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("TABLE"):
            words = line.split()
            _require(len(words) == 2 and words[0] == "TABLE" and words[1] in schemas,
                     f"{path}:{number}: unexpected table/directive")
            active = words[1]
            _require(active not in tables, f"{path}: repeated table {active}")
            tables[active] = []
        else:
            _require(active is not None, f"{path}:{number}: row outside table")
            words = line.split(",")
            _require(len(words) == schemas[active], f"{path}:{number}: wrong column count")
            integer_columns = (0, 1) if active == "CRVBadChan" else (0,)
            _require(all(re.fullmatch(r"[+-]?\d+", words[j].strip()) for j in integer_columns),
                     f"{path}:{number}: expected literal integer")
            tables[active].append([float(v) for v in words])
    _require(set(tables) == set(schemas), f"{path}: missing tables")
    out = {k: np.asarray(v, dtype=np.float64).reshape(-1, schemas[k]) for k, v in tables.items()}
    _require(all(np.all(np.isfinite(v)) for v in out.values()), f"{path}: nonfinite source values")
    return out


def _ordinal(path, n):
    lines = [s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith("#")]
    _require(bool(lines) and lines[0].lower().split() == ["channel", "roc", "feb", "febchannel"],
             "ordinal: unexpected header")
    rows = []
    for line in lines[1:]:
        words = line.split()
        _require(len(words) == 4, "ordinal: wrong column count")
        _require(all(re.fullmatch(r"[+-]?\d+", v) for v in words), "ordinal: expected literal integer")
        rows.append([float(v) for v in words])
    a = _channels(np.asarray(rows, dtype=float).reshape(-1, 4), n, "ordinal", dense=False)
    _integers(a, "ordinal")
    _hardware(a[:, 1:], allow_missing=False)
    result = np.full((n, 3), -1, dtype=np.int64)
    result[a[:, 0].astype(np.int64)] = a[:, 1:].astype(np.int64)
    return result


def _hardware(a, allow_missing=True):
    a = _integers(a, "hardware", -1 if allow_missing else 0)
    missing = np.all(a == -1, axis=1)
    _require(np.all(missing | np.all(a >= 0, axis=1)), "hardware: partial unmapped sentinel")
    mapped = a[~missing]
    _require(np.all((mapped[:, 0] >= 1) & (mapped[:, 0] <= 18) &
                    (mapped[:, 1] >= 1) & (mapped[:, 1] <= 24) & (mapped[:, 2] < 64)),
             "hardware: invalid ROC/FEB/FEBchannel range")
    _require(len(np.unique(mapped, axis=0)) == len(mapped), "hardware: duplicate mapped address")
    return a


def _validate_calib(a, n):
    _require(a.ndim == 2 and a.shape[1] == 9, "calib: expected nine columns")
    a = _channels(a, n, "calib")
    _require(np.all(np.isfinite(a)), "calib: nonfinite values")
    _require(np.all(a[:, 2:4] > 0), "calib: gains must be positive")
    _integers(a[:, 5], "status")
    _hardware(a[:, 6:9])
    return a


def _source_payload(source, n):
    tables = _text_tables(source / "calib_source.txt", {"CRVTime": 2, "CRVSiPM": 4})
    t = _channels(tables["CRVTime"], n, "CRVTime")
    s = _channels(tables["CRVSiPM"], n, "CRVSiPM")
    b = _channels(_text_tables(source / "status_source.txt", {"CRVBadChan": 2})["CRVBadChan"],
                  n, "CRVBadChan", dense=False)
    _integers(b[:, 1], "source status")
    status = np.zeros(n)
    status[b[:, 0].astype(np.int64)] = b[:, 1]
    with np.errstate(over="ignore"):
        floats = np.column_stack((s[:, 1], s[:, 3], s[:, 2], t[:, 1])).astype(np.float32).astype(np.float64)
    a = _validate_calib(np.column_stack((np.arange(n), floats, status, _ordinal(source / "ordinal_source.txt", n))), n)
    p = _channels(_text_tables(source / "photonyield_source.txt", {"CRVPhoton": 2})["CRVPhoton"], n, "CRVPhoton")
    with np.errstate(over="ignore"):
        p[:, 1] = p[:, 1].astype(np.float32).astype(np.float64)
    _require(np.all(np.isfinite(p)), "CRVPhoton: nonfinite float32 values")
    return a, p


def _fcl_values(path):
    """Read resolved scalar/list/table FHiCL only; refuse unresolved references and duplicates."""
    text = Path(path).read_text()
    token = re.compile(r'\s+|\#[^\n]*|//[^\n]*|"(?:\\.|[^"\\])*"|[{}\[\]:,]|[^\s{}\[\]:,]+')
    tokens = [m.group() for m in token.finditer(text) if not (m.group().isspace() or m.group().startswith(("#", "//")))]
    index = 0
    def value():
        nonlocal index
        _require(index < len(tokens), "resolved.fcl: missing value")
        t = tokens[index]
        index += 1
        if t == "{":
            return table("}")
        if t == "[":
            items = []
            while index < len(tokens) and tokens[index] != "]":
                items.append(value())
                if index < len(tokens) and tokens[index] == ",":
                    index += 1
            _require(index < len(tokens), "resolved.fcl: unclosed list")
            index += 1
            return items
        _require(t not in "{}[]:," and not t.startswith("@"), "resolved.fcl: unresolved/invalid value")
        return json.loads(t) if t.startswith('"') else t
    def table(end=None):
        nonlocal index
        result = {}
        while index < len(tokens) and tokens[index] != end:
            key = tokens[index]
            index += 1
            if key.startswith('"'):
                key = json.loads(key)
            _require(index < len(tokens) and tokens[index] == ":", "resolved.fcl: expected key:value")
            index += 1
            _require(key not in result, f"resolved.fcl: duplicate key {key}")
            result[key] = value()
        if end:
            _require(index < len(tokens), "resolved.fcl: unclosed table")
            index += 1
        return result
    root = table()
    flat = {}
    def flatten(obj, prefix=""):
        for key, val in obj.items():
            name = prefix + key
            if isinstance(val, dict):
                flatten(val, name + ".")
            else:
                _require(name not in flat, f"resolved.fcl: duplicate path {name}")
                flat[name] = val
    flatten(root)
    return flat


def _bindings(source, rc):
    f = _fcl_values(source / "resolved.fcl")
    _require(f.get("services.DbService.purpose") == "Sim_best" and
             f.get("services.DbService.version") == "v1_5", "expected Sim_best/v1_5 base")
    for name in ("crvCalib", "crvStatus", "crvPhotonYield"):
        _require(f.get(f"services.ProditionsService.{name}.useDb") == "true", f"{name}.useDb must be true")
    _require(rc.name in GEOMETRIES, f"unsupported named geometry {rc.name}")
    geo = GEOMETRIES[rc.name]
    names = {"calib_source.txt": f"calib_{geo}.txt", "status_source.txt": f"status_{geo}.txt",
             "photonyield_source.txt": f"photonyield_{geo}.txt"}
    refs = f.get("services.DbService.textFile")
    _require(isinstance(refs, list) and len(refs) == 3 and len(set(refs)) == 3,
             "exactly three explicit DbService text overlays required")
    configured = {}
    for snapshot, name in names.items():
        matches = [r for r in refs if Path(r).name == name]
        _require(len(matches) == 1, f"resolved.fcl: missing/ambiguous {name}")
        configured[snapshot] = matches[0]
    ordinal_dir = f.get("services.ProditionsService.crvOrdinal.filePath")
    _require(isinstance(ordinal_dir, str) and bool(ordinal_dir), "missing ordinal filePath")
    configured["ordinal_source.txt"] = str(Path(ordinal_dir) / (geo + ".txt"))
    records = re.findall(r"^([a-fA-F0-9]{64})\s+\*?(/[^\r\n]+)$", (source / "manifest.txt").read_text(), re.M)
    bindings = {}
    for snapshot, reference in configured.items():
        digest = _sha(source / snapshot)
        matches = [(h.lower(), p) for h, p in records if p == reference or p.endswith("/" + reference.lstrip("/"))]
        _require(len(matches) == 1 and matches[0][0] == digest,
                 f"{snapshot}: manifest.txt must bind configured reference to one backing-path SHA256")
        bindings[snapshot] = {"configured": reference, "backing_path": matches[0][1], "sha256": digest}
    return bindings


def _geometry(geom):
    payload = {key: np.asarray(getattr(geom, key)).tolist() for key in
               ("sector", "module", "layer", "counter", "center", "halflen", "local_to_world")}
    payload["sector_names"] = list(geom.sector_names)
    return {"nbars": int(geom.nbars), "nchannels": 4 * int(geom.nbars), "sha256": _digest(payload)}


def _identity(rc, geom):
    from .geometry import from_offline_config
    counters = Path(rc.counters_path())
    g = _geometry(geom)
    _require(g["nbars"] > 0 and g == _geometry(from_offline_config(counters)), "geometry/counters mismatch")
    config = json.loads(_canonical(dataclasses.asdict(rc)))
    return {"run_config": rc.name, "config": config, "config_sha256": _digest(config), "geometry": g,
            "counters": {"configured": rc.counters, "sha256": _sha(counters)}}


def _root(source, n):
    with uproot.open(source / "crvreco.root") as root:
        cal = root["crvdump/calib"].arrays(["run", *CALIB], library="np")
        ev = root["crvdump/events"].arrays(list(EVENTS), library="np")
    _require(len(ev["run"]) > 0, "stock events must not be empty")
    ids = np.column_stack([_integers(ev[k], k) for k in ("run", "subrun", "event")])
    rs = np.unique(ids[:, :2], axis=0)
    _require(len(rs) == 1 and int(rs[0, 0]) == 1400, "require one stock run/subrun at reference run 1400")
    _require(len(np.unique(ids, axis=0)) == len(ids), "duplicate stock event identity")
    _require(np.all(_integers(cal["run"], "calib run") == 1400), "calib run differs from stock events")
    for key in ("spill", "evlen", "pbtime"):
        _require(np.all(np.isfinite(ev[key])), f"events: nonfinite {key}")
    a = _validate_calib(np.column_stack([cal[k] for k in CALIB]), n)
    evidence = {"run": 1400, "subrun": int(rs[0, 1]), "event_count": len(ids),
                "first_event": int(ids[0, 2]), "event_min": int(ids[:, 2].min()), "event_max": int(ids[:, 2].max()),
                "identities_sha256": _digest(ids.tolist()),
                "events_sha256": _digest({k: np.asarray(ev[k]).tolist() for k in EVENTS}),
                "calib_capture": "first stock event; conditions read once per job"}
    return a, evidence


def _record(path, relative):
    return {"path": relative, "sha256": _sha(path), "size": Path(path).stat().st_size}


def export_conditions(source, rc, outdir, geom=None):
    """Export a new portable bundle and return its manifest path; never overwrite an output."""
    source, outdir = Path(source).resolve(), Path(outdir).absolute()
    _require(not outdir.exists() and not outdir.is_symlink(), "output already exists")
    for name in INPUTS:
        _require((source / name).is_file(), f"missing input {name}")
    before = {name: _record(source / name, f"inputs/{name}") for name in INPUTS}
    geom = rc.geometry() if geom is None else geom
    identity = _identity(rc, geom)
    n = identity["geometry"]["nchannels"]
    bindings = _bindings(source, rc)
    expected, photons = _source_payload(source, n)
    calib, events = _root(source, n)
    _require(np.array_equal(calib[:, [0, 5, 6, 7, 8]], expected[:, [0, 5, 6, 7, 8]]),
             "stock/source channel/status/ordinal mismatch")
    _require(np.array_equal(calib[:, 1:5].astype(np.float32).astype(float), expected[:, 1:5]),
             "stock/source calibration mismatch at float32 precision")
    _require(all(_sha(source / name) == item["sha256"] for name, item in before.items()), "inputs changed during export")
    outdir.mkdir(parents=False, exist_ok=False)
    (outdir / "inputs").mkdir()
    for name in INPUTS:
        shutil.copyfile(source / name, outdir / "inputs" / name)
        _require(_sha(outdir / "inputs" / name) == before[name]["sha256"], f"input changed while copying {name}")
    shutil.copyfile(rc.counters_path(), outdir / "counters.txt")
    _require(_sha(outdir / "counters.txt") == identity["counters"]["sha256"], "counters changed during export")
    np.savetxt(outdir / "calib.csv", expected, delimiter=",", fmt="%.17g", header=",".join(CALIB))
    np.savetxt(outdir / "photon.csv", photons, delimiter=",", fmt="%.17g", header="channel,deviation")
    manifest = {"schema": SCHEMA, "scope": SCOPE, **identity, "stock_event_identity": events,
                "provenance": {"source_directory": str(source), "bindings": bindings,
                               "exporter_sha256": _sha(__file__),
                               "photon_evidence": "configured text overlay; photon values not dumped in stock ROOT"},
                "inputs": before,
                "assets": {name: _record(outdir / name, name) for name in ("calib.csv", "photon.csv", "counters.txt")}}
    _require(_identity(rc, geom) == identity, "configuration/counters changed during export")
    _require(all(_sha(source / name) == item["sha256"] for name, item in before.items()), "inputs changed during export")
    path = outdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def load_conditions(manifest_path, rc, geom):
    """Verify bundle bytes, run configuration, geometry and source closure before returning Conditions, metadata."""
    from .chain import Conditions
    path = Path(manifest_path).resolve()
    m = _json(path)
    _require(m.get("schema") == SCHEMA and m.get("scope") == SCOPE, "unsupported conditions manifest/scope")
    _require(set(m["inputs"]) == set(INPUTS) and set(m["assets"]) == {"calib.csv", "photon.csv", "counters.txt"},
             "manifest input/asset inventory mismatch")
    for group, prefix in ((m["inputs"], "inputs/"), (m["assets"], "")):
        for name, item in group.items():
            _require(item["path"] == prefix + name, "unexpected manifest asset path")
            p = path.parent / item["path"]
            _require(p.is_file() and p.stat().st_size == item["size"] and _sha(p) == item["sha256"],
                     f"hash/size mismatch: {name}")
    current = _identity(rc, geom)
    _require(all(m.get(key) == val for key, val in current.items()), "run configuration/geometry/counters mismatch")
    _require(m["assets"]["counters.txt"]["sha256"] == current["counters"]["sha256"], "counters snapshot mismatch")
    source = path.parent / "inputs"
    _require(m["provenance"]["bindings"] == _bindings(source, rc), "source binding identity mismatch")
    n = current["geometry"]["nchannels"]
    expected, photon = _source_payload(source, n)
    stock, events = _root(source, n)
    _require(events == m["stock_event_identity"], "stock event identity mismatch")
    _require(np.array_equal(stock[:, [0, 5, 6, 7, 8]], expected[:, [0, 5, 6, 7, 8]]) and
             np.array_equal(stock[:, 1:5].astype(np.float32).astype(float), expected[:, 1:5]), "stock/source mismatch")
    a = _validate_calib(np.loadtxt(path.parent / "calib.csv", delimiter=",", ndmin=2), n)
    p = np.loadtxt(path.parent / "photon.csv", delimiter=",", ndmin=2)
    _require(p.ndim == 2 and p.shape[1] == 2, "photon CSV column count")
    p = _channels(p, n, "photon CSV")
    _require(np.array_equal(a, expected) and np.array_equal(p, photon), "CSV/source payload mismatch")
    cond = Conditions(a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5].astype(np.int64),
                      a[:, 6].astype(np.int64), a[:, 7].astype(np.int64), p[:, 1])
    for group in (m["inputs"], m["assets"]):
        for name, item in group.items():
            _require(_sha(path.parent / item["path"]) == item["sha256"], f"input changed during load: {name}")
    _require(_identity(rc, geom) == current and _json(path) == m, "identity changed during load")
    return cond, m


def main(argv=None):
    from . import runconfig
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export an exclusive, verified conditions bundle")
    export.add_argument("--source", required=True)
    export.add_argument("--run-config", required=True, choices=sorted(GEOMETRIES))
    export.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    print(export_conditions(args.source, runconfig.get(args.run_config), args.outdir))


if __name__ == "__main__":
    main()
