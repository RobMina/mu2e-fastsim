"""Opt-in, census-bound Cat extraction and tuple-keyed pools; legacy modules are unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

SCHEMA = "crv-cat-v2"
CONTENT_HASH_FORMAT = "arrow-ipc-stream-v2"
IDS = ["run", "subrun", "event"]
IDENTITY = ["source_identity", *IDS]
F32 = ["edep", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "p1", "len"]
STEP_COLUMNS = [*IDS, "bar", "sp", "t0", "t1", *F32, "pdg"]
CENSUS_COLUMNS = [*IDS, "ngen", "nsteps"]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _locator(value, label):
    _text(value, label)
    if not (Path(value).is_absolute() or "://" in value):
        raise ValueError(f"{label} must be an absolute locator, not a basename")


def _provenance(source_art, extraction):
    if not isinstance(source_art, dict) or not isinstance(extraction, dict):
        raise ValueError("source_art and extraction must be objects")
    _text(source_art.get("identity"), "source_art.identity")
    _locator(source_art.get("uri"), "source_art.uri")
    _digest(source_art.get("sha256"), "source_art.sha256")
    for key in ("id", "command", "software_version"):
        _text(extraction.get(key), f"extraction.{key}")
    for key in ("fcl_sha256", "dumper_sha256"):
        _digest(extraction.get(key), f"extraction.{key}")
    if extraction.get("write_steps") is not True or extraction.get("event_selection") != "all":
        raise ValueError("extraction must declare write_steps=true and event_selection='all'")
    if type(extraction.get("source_count")) is not int or extraction["source_count"] != 1:
        raise ValueError("exactly one source art input is required per extraction")
    _json({"source_art": source_art, "extraction": extraction})


def _require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing or not frame.columns.is_unique:
        raise ValueError(f"{label}: missing columns {sorted(missing)} or duplicate columns")


def _integer(frame, name, label, low=0, high=2**63 - 1):
    _require(frame, [name], label)
    values = frame[name].to_numpy()
    if values.dtype.kind not in "iu" or np.any(values < low) or np.any(values > high):
        raise ValueError(f"{label}.{name}: invalid integer IDs/counts")
    frame[name] = values.astype(np.int64)


def _event_ids(frame, label):
    for name in IDS:
        _integer(frame, name, label, 0 if name == "subrun" else 1, 2**32 - 2)


def _unique(frame, columns, label):
    if frame.duplicated(columns).any():
        raise ValueError(f"{label}: duplicate keys {columns}")


def _membership(frame, census, label):
    if not pd.MultiIndex.from_frame(frame[IDS]).isin(pd.MultiIndex.from_frame(census[IDS])).all():
        raise ValueError(f"{label}: event ID missing from census")


def _counts(frame, census, count, label):
    actual = frame.groupby(IDS, sort=False).size().reindex(pd.MultiIndex.from_frame(census[IDS]), fill_value=0)
    if not np.array_equal(actual.to_numpy(), census[count].to_numpy()):
        raise ValueError(f"{label}: {count} disagrees with census")


def _census(frame):
    frame = frame.copy()
    _require(frame, CENSUS_COLUMNS, "events")
    _event_ids(frame, "events")
    for name in ("ngen", "nsteps"):
        _integer(frame, name, "events")
    _unique(frame, IDS, "events")
    if not len(frame):
        raise ValueError("events census must not be empty")
    return frame[CENSUS_COLUMNS].sort_values(IDS, kind="stable").reset_index(drop=True)


def _step_values(steps):
    _require(steps, STEP_COLUMNS, "steps")
    _event_ids(steps, "steps")
    for name in ("bar", "sp"):
        _integer(steps, name, "steps", 0, 2**31 - 1)
    _integer(steps, "pdg", "steps", -(2**31), 2**31 - 1)
    if (steps.pdg == 0).any():
        raise ValueError("steps: unknown PDG zero")
    for name in F32:
        values = steps[name].to_numpy()
        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError(f"steps.{name}: nonfinite or nonnumeric kinematics")
        if np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError(f"steps.{name}: float32 overflow")
        steps[name] = values.astype(np.float32)
    for name in ("t0", "t1"):
        values = steps[name].to_numpy()
        if values.dtype.kind not in "fiu" or np.isinf(values).any():
            raise ValueError(f"steps.{name}: infinite or nonnumeric times")
        steps[name] = values.astype(np.float64)
    keys = [*IDS, "sp"]
    if (steps.groupby(keys, sort=False).pdg.nunique() > 1).any():
        raise ValueError("steps: inconsistent PDGs for particle key")
    return steps[STEP_COLUMNS].reset_index(drop=True)


def _validated_tables(events, gen, steps, particles):
    census = _census(events)
    gen, steps, particles = gen.copy(), steps.copy(), particles.copy()
    for label, frame in (("gen", gen), ("steps", steps), ("particles", particles)):
        _event_ids(frame, label)
        _membership(frame, census, label)
    _integer(gen, "idx", "gen", 0, 2**31 - 1)
    _integer(gen, "pdg", "gen", -(2**31), 2**31 - 1)
    _unique(gen, [*IDS, "idx"], "gen")
    _integer(particles, "key", "particles", 0, 2**31 - 1)
    _integer(particles, "pdg", "particles", -(2**31), 2**31 - 1)
    _unique(particles, [*IDS, "key"], "particles")
    if (particles.pdg == 0).any() or (gen.pdg == 0).any():
        raise ValueError("gen/particles: unknown PDG zero")
    _counts(gen, census, "ngen", "gen")
    indexed = gen.merge(census[[*IDS, "ngen"]], on=IDS, how="left", validate="many_to_one")
    if (indexed.idx >= indexed.ngen).any():
        raise ValueError("gen: index outside census generator count")
    _counts(steps, census, "nsteps", "steps")
    if "genidx" in particles:
        _integer(particles, "genidx", "particles", -1, 2**31 - 1)
        # Compression may attach an ancestor's GenParticle pointer to a descendant
        # of a different species. Validate the link, not particle/generator PDG equality.
        linked = particles.loc[particles.genidx >= 0, [*IDS, "genidx"]]
        joined = linked.merge(gen[[*IDS, "idx"]], left_on=[*IDS, "genidx"],
                              right_on=[*IDS, "idx"], how="left", validate="many_to_one")
        if joined.idx.isna().any():
            raise ValueError("particles: missing generator key")
    _integer(steps, "sp", "steps", 0, 2**31 - 1)
    if "nstep" in steps:
        _integer(steps, "nstep", "steps")
        joined = steps.merge(census[[*IDS, "nsteps"]], on=IDS, how="left", validate="many_to_one")
        if not np.array_equal(joined.nstep, joined.nsteps):
            raise ValueError("steps.nstep disagrees with census")
    joined = steps.merge(particles[[*IDS, "key", "pdg"]].rename(columns={"key": "sp", "pdg": "particle_pdg"}),
                         on=[*IDS, "sp"], how="left", validate="many_to_one", sort=False)
    if joined.particle_pdg.isna().any():
        raise ValueError("steps: missing particle key")
    if "pdg" in joined and not np.array_equal(joined.pdg, joined.particle_pdg):
        raise ValueError("steps: inconsistent PDGs")
    joined["pdg"] = joined.particle_pdg.astype(np.int64)
    return census, _step_values(joined)


def _table_hash(frame):
    """Hash typed numeric rows, not pandas indices or undefined Arrow null bytes.

    NaN sign/payload and signed zero are not distinct logical content. Normalize
    only the hash buffers: persisted kinematics and finite nonzero bits stay intact.
    """
    import pyarrow as pa
    arrays = []
    for name in frame.columns:
        values = frame[name].to_numpy(copy=True)
        if values.dtype.kind not in "iuf":
            raise ValueError(f"content hash requires numeric column: {name}")
        validity, null_count = None, 0
        if values.dtype.kind == "f":
            missing = np.isnan(values)
            null_count = int(missing.sum())
            values[missing | (values == 0)] = 0
            if null_count:
                # Initialize null data and bitmap padding for deterministic Arrow IPC buffers.
                validity = pa.py_buffer(np.packbits(~missing, bitorder="little"))
        arrays.append(pa.Array.from_buffers(
            pa.from_numpy_dtype(values.dtype), len(values),
            [validity, pa.py_buffer(np.ascontiguousarray(values))], null_count=null_count))
    table = pa.Table.from_arrays(arrays, names=list(frame.columns))
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _payload_hash(census, steps):
    ordered = steps.sort_values(STEP_COLUMNS, kind="stable").reset_index(drop=True)
    return hashlib.sha256(_json({"events": _table_hash(census), "steps": _table_hash(ordered)})).hexdigest()


def _root_frame(root, name):
    if f"dump/{name}" not in root:
        raise ValueError(f"required dump/{name} tree is missing; generator IDs are not an event census")
    tree = root[f"dump/{name}"]
    required = {"events": CENSUS_COLUMNS, "gen": [*IDS, "idx", "pdg"],
                "steps": [column for column in STEP_COLUMNS if column != "pdg"],
                "particles": [*IDS, "key", "pdg"]}[name]
    optional = {"events": [], "gen": [], "steps": ["nstep", "pdg"], "particles": ["genidx"]}[name]
    missing = set(required) - set(tree.keys())
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    arrays = tree.arrays(required + [column for column in optional if column in tree.keys()], library="np")
    return pd.DataFrame(arrays)


def _companion_path(path, name):
    if (not isinstance(name, str) or not name.endswith(".events.parquet")
            or any(char in name for char in ("/", "\\", "\0", ":"))
            or Path(name).name != name):
        raise ValueError("event census companion must be a relative sibling filename")
    companion = Path(path).parent / name
    if companion == Path(path):
        raise ValueError("event census companion path collision")
    return companion


def convert(path, dataset, out, *, source_art, extraction):
    """Write step/census siblings and return compact, hash-bound provenance metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import uproot
    _text(dataset, "dataset")
    _provenance(source_art, extraction)
    path, out = Path(path), Path(out)
    events_path = _companion_path(out, out.with_suffix(".events.parquet").name)
    for target in (out, events_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    dump_hash = sha256(path)
    with uproot.open(path) as root:
        census, steps = _validated_tables(*(_root_frame(root, name) for name in ("events", "gen", "steps", "particles")))
    if sha256(path) != dump_hash:
        raise ValueError("dump changed during extraction")
    meta = {"schema": SCHEMA, "dataset": dataset, "source_art": source_art, "extraction": extraction,
            "dump": {"uri": str(path.resolve()), "sha256": dump_hash},
            "converter_sha256": sha256(__file__), "content_hash_format": CONTENT_HASH_FORMAT,
            "frames": len(census), "stepped_frames": int((census.nsteps > 0).sum()), "steps": len(steps),
            "census_sha256": _table_hash(census),
            "content_sha256": _payload_hash(census, steps)}
    steps.insert(0, "source_identity", source_art["identity"])
    table = pa.Table.from_pandas(steps, preserve_index=False)
    events_table = pa.Table.from_pandas(census, preserve_index=False).replace_schema_metadata(None)
    # Exclusive creation and owned-file cleanup protect both sibling outputs.
    created = []
    try:
        with out.open("xb") as stream:
            created.append(out)
            with events_path.open("xb") as events_stream:
                created.append(events_path)
                pq.write_table(events_table, events_stream, compression="zstd")
                events_stream.flush()
                meta["events"] = {"path": events_path.name, "sha256": sha256(events_path)}
                table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"crvfastsim": _json(meta)})
                pq.write_table(table, stream, compression="zstd")
    except BaseException:
        for target in reversed(created):
            target.unlink()
        raise
    return meta


def read_meta(path):
    """Read the version tag; PoolV2 additionally validates payload and provenance bindings."""
    import pyarrow.parquet as pq
    raw = (pq.read_schema(path).metadata or {}).get(b"crvfastsim")
    if raw is None:
        raise ValueError("missing crvfastsim metadata")
    meta = json.loads(raw)
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
        raise ValueError(f"expected metadata schema {SCHEMA}")
    return meta


def _read_census(path, meta):
    """Resolve only a nonsymlink sibling; bind physical bytes and canonical rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    reference = meta.get("events")
    if "event_census" in meta or not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("missing or malformed event census companion reference")
    companion = _companion_path(path, reference["path"])
    if companion.is_symlink():
        raise ValueError("event census companion must not be a symlink")
    _digest(reference["sha256"], "events.sha256")
    if not companion.is_file():
        raise ValueError("missing event census companion")
    if sha256(companion) != reference["sha256"]:
        raise ValueError("event census companion file hash disagrees with metadata")
    table = pq.read_table(companion)
    if table.column_names != CENSUS_COLUMNS or any(field.type != pa.int64() for field in table.schema):
        raise ValueError("event census companion dtype/columns disagree with v2 schema")
    if sha256(companion) != reference["sha256"]:
        raise ValueError("event census companion changed during pool loading")
    return _census(table.to_pandas())


def _read_validated(path):
    meta = read_meta(path)
    _text(meta.get("dataset"), "dataset")
    _provenance(meta.get("source_art"), meta.get("extraction"))
    dump = meta.get("dump", {})
    if not isinstance(dump, dict):
        raise ValueError("dump provenance must be an object")
    _locator(dump.get("uri"), "dump.uri")
    for label, value in (("dump.sha256", dump.get("sha256")), ("converter_sha256", meta.get("converter_sha256")),
                         ("census_sha256", meta.get("census_sha256")), ("content_sha256", meta.get("content_sha256"))):
        _digest(value, label)
    if meta.get("content_hash_format") != CONTENT_HASH_FORMAT:
        raise ValueError("unsupported content hash format; reconvert the ROOT dump to a new output pair")
    census = _read_census(path, meta)
    steps = pd.read_parquet(path)
    _require(steps, ["source_identity", *STEP_COLUMNS], "steps")
    if set(steps.columns) != {"source_identity", *STEP_COLUMNS}:
        raise ValueError("unexpected v2 step columns")
    if not steps.source_identity.eq(meta["source_art"]["identity"]).all():
        raise ValueError("source identity disagrees with metadata")
    for name in STEP_COLUMNS:
        expected = np.dtype("float32" if name in F32 else "float64" if name in ("t0", "t1") else "int64")
        if steps[name].dtype != expected:
            raise ValueError(f"steps.{name}: dtype disagrees with v2 schema")
    payload = _step_values(steps.copy())
    _membership(payload, census, "steps")
    _counts(payload, census, "nsteps", "steps")
    for name, actual in (("frames", len(census)), ("stepped_frames", int((census.nsteps > 0).sum())), ("steps", len(payload))):
        if type(meta.get(name)) is not int or meta[name] != actual:
            raise ValueError(f"metadata {name} disagrees with payload/census")
    if _table_hash(census) != meta["census_sha256"] or _payload_hash(census, payload) != meta["content_sha256"]:
        raise ValueError("content/census hash disagrees with metadata")
    payload.insert(0, "source_identity", meta["source_art"]["identity"])
    census.insert(0, "source_identity", meta["source_art"]["identity"])
    return meta, census, payload


class PoolV2:
    """Strict single-dataset pool: no duplicate-source acceptance or hidden reweighting."""

    def __init__(self, files):
        files = list(files)
        if not files:
            raise ValueError("PoolV2 requires at least one census-bound file")
        parts, censuses, self.manifest = [], [], []
        self.hash_groups = {name: {} for name in ("file_sha256", "dump_sha256", "source_sha256", "content_sha256")}
        seen = {name: set() for name in ("source_identity", "source_uri", "extraction_id")}
        dataset, contract = None, None
        for path in files:
            file_hash = sha256(path)
            meta, census, steps = _read_validated(path)
            if sha256(path) != file_hash:
                raise ValueError("input parquet changed during pool loading")
            if dataset is not None and dataset != meta["dataset"]:
                raise ValueError("pool dataset metadata disagreement")
            dataset = meta["dataset"]
            current = {name: meta["extraction"][name] for name in
                       ("software_version", "fcl_sha256", "dumper_sha256", "write_steps", "source_count", "event_selection")}
            current["converter_sha256"] = meta["converter_sha256"]
            if contract is not None and contract != current:
                raise ValueError("pool extraction metadata disagreement")
            contract = current
            values = {"source_identity": meta["source_art"]["identity"], "source_uri": meta["source_art"]["uri"],
                      "extraction_id": meta["extraction"]["id"]}
            hashes = {"file_sha256": file_hash, "dump_sha256": meta["dump"]["sha256"],
                      "source_sha256": meta["source_art"]["sha256"], "content_sha256": meta["content_sha256"]}
            for name, value in values.items():
                if value in seen[name]:
                    raise ValueError(f"duplicate {name}: {value}; explicit provenance review required")
                seen[name].add(value)
            for name, value in hashes.items():
                if value in self.hash_groups[name]:
                    raise ValueError(f"duplicate {name}: {value}; explicit provenance review required")
                self.hash_groups[name][value] = [str(Path(path).resolve())]
            self.manifest.append({"path": str(Path(path).resolve()), **hashes, "metadata": meta})
            parts.append(steps)
            censuses.append(census)
        self.dataset = dataset
        self.extraction_contract = contract
        self.events = pd.concat(censuses, ignore_index=True)
        _unique(self.events, IDENTITY, "pool events")
        self.frames = len(self.events)
        self.steps = pd.concat(parts, ignore_index=True).sort_values(IDENTITY, kind="stable").reset_index(drop=True)
        keys = pd.MultiIndex.from_frame(self.steps[IDENTITY])
        fid, self.ids = pd.factorize(keys, sort=False)
        self.ids = self.ids.set_names(IDENTITY)
        self.steps["fid"] = fid.astype(np.int64)
        self.stepped = len(self.ids)
        self.count = np.bincount(fid, minlength=self.stepped).astype(np.int64)
        self.start = np.cumsum(self.count, dtype=np.int64) - self.count
        if self.stepped != int((self.events.nsteps > 0).sum()):
            raise ValueError("pool stepped denominator disagreement")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provenance", required=True, help="JSON object with source_art and extraction")
    args = parser.parse_args(argv)
    with open(args.provenance) as stream:
        provenance = json.load(stream)
    meta = convert(args.input, args.dataset, args.out, source_art=provenance["source_art"], extraction=provenance["extraction"])
    print(json.dumps({name: meta[name] for name in ("schema", "dataset", "frames", "stepped_frames", "steps")}))


if __name__ == "__main__":
    main()
