"""Direct flat-frame mixing; NumPy distributions, not C++ RNG replay.

Changing max_events changes RNG consumption and the resulting realization; it
is not prefix replay of a longer run. All source pools are loaded in memory:
max_events limits signal/output events, not donor-pool memory. Selected-row
estimates are diagnostic only; no memory cap or chunked loading is imposed.

No timing is generated or loaded here. Before detector response, the caller must
load complete, explicit MC/reco timing matching mixed.event_ids, including empty
events. Times remain unwrapped; only the detector response may fold them.
"""
import hashlib
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .frames import CRV_FIELDS, IDS, TRACKER_FIELDS, FrameBatch, overlay, read_event_log, read_frames
from .sampling import CHAINS, STREAMS, BunchLaw, draw_multiplicities, select_frames, stock_rates


_CAMPAIGN = {"extended_mean", "sdf", "cut_max", "beam_primary", "efficiencies",
             "stop_prescale", "max_skip", "mode", "postmix_selection", "additional_time_offset_ns"}
_FACTORS = {tag for chain in CHAINS.values() for tag in chain}


def _keys(value, required, label, optional=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing, extra = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{label}: missing {sorted(missing)}; unsupported {sorted(extra)}")


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def _source(spec, base, label):
    _keys(spec, {"path"}, label, {"event_log"})
    for key, value in spec.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}.{key} must be a nonempty path")
    return {key: str((base / value).resolve()) for key, value in spec.items()}


def _sha256(path):
    """Hash source bytes with bounded IO buffers, independently of ROOT size."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_native_products(path):
    """Validate declared native links before read_frames discards product fields.

    A missing pointer-product branch is legacy/unavailable (reported explicitly),
    not proof of identity. This includes current legacy tracker dumpers alongside
    modern CRV trees. A declared pointer branch without its target-product branch
    is partial data and fails. Null parent/gen pointers use product ID zero.
    Joins always include full event IDs, the local key, and native product ID.
    """
    specs = {"particles": ("crv/particles", ("key", "parent", "genidx"),
                            ("sim_product_id", "parent_product_id", "gen_product_id")),
             "gen": ("crv/gen", ("idx",), ("gen_product_id",)),
             "tracker": ("tracker/sgsteps", ("sp",), ("sp_product_id",)),
             "crv": ("crv/steps", ("sp",), ("sp_product_id",)),
             "events": ("crv/events", (), ("sim_product_id", "gen_product_id"))}
    tables = {}
    with uproot.open(path) as root:
        for name, (tree_name, keys, products) in specs.items():
            if name == "events" and tree_name not in root:
                tables[name] = pd.DataFrame()
                continue
            tree = root[tree_name]
            present = [field for field in products if field in tree.keys()]
            if not present:
                tables[name] = pd.DataFrame()
                continue
            arrays = tree.arrays([*IDS, *keys, *present], library="np")
            tables[name] = pd.DataFrame({field: arrays[field] for field in [*IDS, *keys, *present]})
            for field in present:
                values = tables[name][field].to_numpy()
                if (values.dtype.kind not in "iu" or np.any(values < 0)
                        or (name in ("gen", "tracker", "crv") and np.any(values == 0))
                        or (name == "particles" and field == "sim_product_id" and np.any(values == 0))):
                    raise ValueError(f"invalid native product ID: {name}.{field}")

    checks = {}

    def link(label, source_name, source_key, source_product, target_name, target_key,
             target_product, nullable=False):
        source, target = tables[source_name], tables[target_name]
        if source_product not in source:
            checks[label] = ("unavailable: pointer product branch absent; "
                             + ("target product branch present (partial coverage)"
                                if target_product in target else "target product branch absent"))
            return
        if target_product not in target:
            raise ValueError(f"partial native product data for {label}: missing {target_name}.{target_product}")
        if nullable:
            null = source[source_key] == -1
            if (source.loc[null, source_product] != 0).any():
                raise ValueError(f"native product ID mismatch for null {label}")
            source = source.loc[~null]
        if (source[source_product] == 0).any() or (target[target_product] == 0).any():
            raise ValueError(f"invalid zero native product ID for {label}")
        left_keys = [*IDS, *([source_key] if source_key else []), source_product]
        right_keys = [*IDS, *([target_key] if target_key else []), target_product]
        # Keep integer IDs in merge keys to avoid unmatched-payload float conversion.
        columns = [*IDS, *(["_key"] if source_key else []), "_product"]
        left = source[left_keys].set_axis(columns, axis=1)
        right = target[right_keys].set_axis(columns, axis=1)
        joined = left.merge(right, on=columns, how="left", sort=False,
                            validate="m:1", indicator=True)
        if (joined._merge != "both").any():
            raise ValueError(f"native product ID/key mismatch for {label}")
        checks[label] = "checked"

    for name in ("tracker", "crv"):
        link(name, name, "sp", "sp_product_id", "particles", "key", "sim_product_id")
    link("parent", "particles", "parent", "parent_product_id",
         "particles", "key", "sim_product_id", nullable=True)
    link("generator", "particles", "genidx", "gen_product_id",
         "gen", "idx", "gen_product_id", nullable=True)
    # Available census product identities must agree even for empty detector events.
    for name, product in (("particles", "sim_product_id"), ("gen", "gen_product_id")):
        if product in tables["events"]:
            if product not in tables[name]:
                raise ValueError(f"partial native product data: missing {name}.{product}")
            link(f"{name}_census", name, None, product, "events", None, product)
    return checks


def _read(source, *, hashes=None):
    fingerprint = dict(hashes) if hashes is not None else {"source_sha256": _sha256(source["path"])}
    if "event_log" in source and "event_log_sha256" not in fingerprint:
        fingerprint["event_log_sha256"] = _sha256(source["event_log"])
    fingerprint["product_id_checks"] = _validate_native_products(source["path"])
    ids = read_event_log(source["event_log"]) if "event_log" in source else None
    batch = read_frames(source["path"], event_ids=ids)
    # Hash the full ordered empty-inclusive census as C-order little-endian uint64 IDs.
    census = np.ascontiguousarray(batch.event_ids, dtype="<u8")
    fingerprint["census_sha256"] = hashlib.sha256(memoryview(census.reshape(-1)).cast("B")).hexdigest()
    for name, target in (("path", "source_sha256"), ("event_log", "event_log_sha256")):
        if name in source and _sha256(source[name]) != fingerprint[target]:
            raise ValueError("source changed while reading validated frames")
    return batch, fingerprint


def _table(batch, name, fields=None):
    rows = batch.collections[name]
    ids = np.repeat(batch.event_ids, ak.to_numpy(ak.num(rows)), axis=0)
    flat = ak.flatten(rows, axis=1)
    return pd.DataFrame({**dict(zip(IDS, ids.T)),
                         **{field: ak.to_numpy(flat[field]) for field in (fields or ak.fields(flat))}})


def to_detector_inputs(batch: FrameBatch) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (tracker, crv, pdg), with pdg aligned to CRV rows by full ID and sp.

    All detector fields and original event/step order are retained. Truth remains
    on the unmodified FrameBatch. NaN CRV endpoint times are allowed, not infinity.
    """
    for name, fields in (("tracker", TRACKER_FIELDS), ("crv", CRV_FIELDS),
                         ("particles", ("key", "pdg"))):
        missing = set(fields) - set(ak.fields(batch.collections[name]))
        if missing:
            raise ValueError(f"missing {name} fields: {sorted(missing)}")
    references = batch.validate_references()
    tracker, crv = (_table(batch, name) for name in ("tracker", "crv"))
    payload = [field for field in CRV_FIELDS if field not in ("t0", "t1")]
    if not np.isfinite(crv[payload]).all().all() or np.isinf(crv[["t0", "t1"]]).any().any():
        raise ValueError("nonfinite CRV step payload or infinite endpoint time")
    pdg = ak.to_numpy(ak.flatten(batch.collections["particles"].pdg))
    if (pdg.dtype.kind not in "iuf" or not np.isfinite(pdg).all()
            or np.any(pdg != np.floor(pdg))):
        raise ValueError("invalid integer particle pdg")
    if ((pdg.dtype.kind == "u" and np.any(pdg > np.iinfo(np.int64).max))
            or (pdg.dtype.kind == "f" and (np.any(pdg < -(2.0**63)) or np.any(pdg >= 2.0**63)))):
        raise ValueError("particle pdg exceeds int64")
    return tracker, crv, pdg[references["crv"]].astype(np.int64, copy=False)


def _load_manifest_config(path, max_events=None):
    """Resolve and validate v1 source/campaign configuration without reading frames."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    _keys(manifest, {"version", "signal", "pools", "campaign", "seed"}, "manifest")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("unsupported manifest version; expected 1")
    seed = _integer(manifest["seed"], "seed")
    if max_events is not None:
        max_events = _integer(max_events, "max_events")
    campaign = manifest["campaign"]
    _keys(campaign, _CAMPAIGN, "campaign", {"cut_min"})
    for key, expected in (("beam_primary", True), ("mode", "sequential"),
                          ("postmix_selection", "universal")):
        if type(campaign[key]) is not type(expected) or campaign[key] != expected:
            raise ValueError(f"unsupported {key}; expected {expected!r}")
    if _number(campaign["additional_time_offset_ns"], "additional_time_offset_ns") != 0:
        raise ValueError("unsupported additional_time_offset_ns; expected 0")
    for key in ("extended_mean", "sdf", "cut_max", "stop_prescale"):
        _number(campaign[key], key)
    cut_min = _number(campaign.get("cut_min", 0.0), "cut_min")
    law = BunchLaw.from_sdf(campaign["extended_mean"], campaign["sdf"], campaign["cut_max"], cut_min)
    _keys(campaign["efficiencies"], _FACTORS, "efficiencies")
    for key, value in campaign["efficiencies"].items():
        _number(value, key)
    rates = stock_rates(campaign["efficiencies"], stop_prescale=campaign["stop_prescale"])
    _keys(campaign["max_skip"], STREAMS, "max_skip")
    for stream in STREAMS:
        _integer(campaign["max_skip"][stream], f"max_skip.{stream}")
    _keys(manifest["pools"], STREAMS, "pools")
    sources = {"signal": _source(manifest["signal"], path.parent, "signal"),
               "pools": {s: _source(manifest["pools"][s], path.parent, f"pools.{s}") for s in STREAMS}}
    return path, seed, max_events, campaign, law, rates, sources


def load_manifest(path, max_events=None, *, source_cache=None) -> tuple[FrameBatch, pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """Return (mixed, tracker, crv, pdg, provenance) from an explicit JSON v1 manifest.

    Pools are loaded in full before truncating the signal census. Only ordinary,
    beam-primary, sequential/wrapping, universal/no-offset mixing is supported.
    Provenance arrays follow signal census order and STREAMS column order; each
    selection records donor indices, source full IDs and modulo-pool cursors.
    source_fingerprints mirrors sources, with source_sha256 over file bytes,
    census_sha256 over the full ordered little-endian uint64 Nx3 census, optional
    event_log_sha256, and explicit product_id_checks coverage. Row estimates are
    computed from selected per-frame lengths before overlay (including signal,
    repeated donors, and empty frames); actual_rows counts the result. Optional
    FrameSourceCache reuses owned source snapshots after byte-hash verification;
    it does not skip reference validation or alter random-number consumption.
    """
    path, seed, max_events, campaign, law, rates, sources = _load_manifest_config(path, max_events)
    start = time.perf_counter()
    reader = _read if source_cache is None else source_cache.read
    signal, signal_fingerprint = reader(sources["signal"])
    pools, pool_fingerprints = {}, {}
    for stream in STREAMS:
        pools[stream], pool_fingerprints[stream] = reader(sources["pools"][stream])
    read_s = time.perf_counter() - start
    pool_counts = {stream: len(pool.event_ids) for stream, pool in pools.items()}
    if not all(pool_counts.values()):
        raise ValueError("all four full donor pools must have a nonempty event census")
    signal_events_full = len(signal.event_ids)
    if max_events is not None:
        signal = FrameBatch(signal.event_ids[:max_events],
                            {name: rows[:max_events] for name, rows in signal.collections.items()})
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    intensity = law.draw(rng, len(signal.event_ids), beam_primary=True)
    counts = draw_multiplicities(rng, intensity, rates)
    backgrounds, selections = [], {}
    estimated_rows = {name: int(ak.sum(ak.num(rows))) for name, rows in signal.collections.items()}
    for column, stream in enumerate(STREAMS):
        selection, next_cursor = select_frames(
            rng, counts[:, column], pool_counts[stream], cursor=None,
            mode=campaign["mode"], max_events_to_skip=campaign["max_skip"][stream])
        backgrounds.append((pools[stream], selection))
        for name, rows in pools[stream].collections.items():
            estimated_rows[name] += int(ak.sum(ak.num(rows)[selection.donors]))
        selections[stream] = {"counts": selection.counts.copy(), "donors": selection.donors.copy(),
                              "source_event_ids": pools[stream].event_ids[selection.donors].copy(),
                              "event_index": selection.event_index,
                              "initial_cursor": (next_cursor - len(selection.donors)) % pool_counts[stream],
                              "next_cursor": next_cursor}
    sampling_s = time.perf_counter() - start
    start = time.perf_counter()
    mixed = overlay(signal, backgrounds)
    overlay_s = time.perf_counter() - start
    start = time.perf_counter()
    tracker, crv, pdg = to_detector_inputs(mixed)
    adapter_s = time.perf_counter() - start
    provenance = {"version": 1, "manifest": str(path), "sources": sources, "campaign": campaign,
                  "source_fingerprints": {"signal": signal_fingerprint, "pools": pool_fingerprints},
                  "estimated_selected_rows": estimated_rows,
                  "actual_rows": {name: int(ak.sum(ak.num(rows))) for name, rows in mixed.collections.items()},
                  "seed": seed, "stream_order": STREAMS, "rates": rates, "intensity": intensity,
                  "multiplicities": counts, "pool_event_counts": pool_counts,
                  "signal_events_full": signal_events_full, "signal_event_ids": mixed.event_ids.copy(),
                  "selections": selections, "timing_s": {"read_validate": read_s, "sample": sampling_s,
                                                          "overlay_validate": overlay_s, "adapter_validate": adapter_s},
                  "rng": "NumPy default_rng; distributional agreement, not C++ RNG replay"}
    return mixed, tracker, crv, pdg, provenance
