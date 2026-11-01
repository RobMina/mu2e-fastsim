"""Read-only, paired-shard bootstrap of count-histogram shapes and total ratios.

The resampling unit is an original shard pair, never a histogram bin or event.
See docs/valhist-paired-shapes.md for the input and interpretation contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import uproot

from .compare import CRV_ONLY
from .pilot import digest
from .coverage import _aliases, _classification, _has_nonfinite, _inventory, _json_safe, _read


LIMITATIONS = [
    "Assumes exchangeable independent shard pairs, conditional on the fixed secondary library, conditions and model; shared-library and systematic uncertainty are not estimated.",
    "Filename pairing verifies run/subrun labels, not event populations. Histogram-only A0g does not establish event pairing, generated-event completeness or exposure equality.",
    "Inputs must be original unmerged ROOT files. Lost flow/Sumw2 or misleading filenames cannot be detected reliably; provenance remains the caller's responsibility.",
    "No independent-bin or Poisson assumption is used. Source Sumw2 is validated but not used as a covariance estimate.",
    "Centered bootstrap TV tails are approximate null-calibration diagnostics, not exact randomization p-values or acceptance criteria; small shard samples and sparse support can be poorly calibrated.",
    "Intervals and bin RMS diagnostics are pointwise, not simultaneous. Histograms, regions and bins are correlated; no multiplicity-adjusted discovery or blanket agreement claim is provided.",
    "Zero-denominator replicas are explicitly counted. Summaries of surviving replicas are conditional on being defined, not unconditional confidence coverage.",
]


def _finite(a, context):
    if not np.isfinite(a).all():
        raise ValueError(f"nonfinite input or numeric overflow: {context}")


def _options(replicas, batch_size, confidence, seed):
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 2:
        raise ValueError("replicas must be an integer >= 2")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def bootstrap_region(ref, cand, *, replicas=10000, seed=12345, batch_size=256, confidence=0.95):
    """Bootstrap (shards, bins) arrays with identical multinomial weights per arm.

    No per-event/bin/replica Python loops: only bounded matrix batches. Bin RMS
    is sqrt(mean((D* - D_observed)**2)), not an independent-bin standard error.
    """
    _options(replicas, batch_size, confidence, seed)
    a, b = np.asarray(ref, dtype=float), np.asarray(cand, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 2 or a.shape[1] < 1:
        raise ValueError("need matching (>=2 shards, >=1 bins) arrays")
    _finite(a, "reference values")
    _finite(b, "candidate values")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("negative bin values are not probability shapes")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            return _bootstrap(a, b, replicas, seed, batch_size, confidence)
        except FloatingPointError as exc:
            raise ValueError("numeric overflow in bootstrap") from exc


def _bootstrap(a, b, replicas, seed, batch_size, confidence):
    n, bins = a.shape
    total_a, total_b = a.sum(axis=0), b.sum(axis=0)
    sa, sb = float(total_a.sum()), float(total_b.sum())
    _finite(np.array([sa, sb]), "aggregate totals")
    observed_ratio = sb / sa if sa > 0 else None
    delta = total_b / sb - total_a / sa if sa > 0 and sb > 0 else None
    observed_tv = float(np.abs(delta).sum() / 2) if delta is not None else None
    ratios, tvs, null_tvs = np.empty(replicas), np.empty(replicas), np.empty(replicas)
    nr = ns = zero_a = zero_b = zero_both = 0
    squares, shifts = np.zeros(bins), np.zeros(bins)
    rng = np.random.default_rng(seed)
    for start in range(0, replicas, batch_size):
        size = min(batch_size, replicas - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=size)
        ar, br = weights @ a, weights @ b
        ta, tb = ar.sum(axis=1), br.sum(axis=1)
        _finite(ta, "bootstrap reference totals")
        _finite(tb, "bootstrap candidate totals")
        za, zb = ta == 0, tb == 0
        zero_a += int(za.sum())
        zero_b += int(zb.sum())
        zero_both += int((za & zb).sum())
        valid_r, valid_s = ~za, ~(za | zb)
        cr, cs = int(valid_r.sum()), int(valid_s.sum())
        ratios[nr:nr + cr] = tb[valid_r] / ta[valid_r]
        nr += cr
        if delta is not None and cs:
            dstar = br[valid_s] / tb[valid_s, None] - ar[valid_s] / ta[valid_s, None]
            centered = dstar - delta
            tvs[ns:ns + cs] = np.abs(dstar).sum(axis=1) / 2
            null_tvs[ns:ns + cs] = np.abs(centered).sum(axis=1) / 2
            squares += np.square(centered).sum(axis=0)
            shifts += centered.sum(axis=0)
            ns += cs
    alpha = (1 - confidence) / 2

    def interval(values):
        return np.quantile(values, [alpha, 1 - alpha]).tolist() if len(values) else None

    def status(valid, observed):
        if observed is None:
            return "undefined_observed_zero_total"
        if valid == 0:
            return "undefined_all_replicas"
        return "conditional_on_defined_replicas" if valid < replicas else "estimated"

    ratio = dict(status=status(nr, observed_ratio), value=observed_ratio,
                 relative_difference=observed_ratio - 1 if observed_ratio is not None else None,
                 percentile_ci=interval(ratios[:nr]), valid_replicas=nr,
                 undefined_zero_reference_replicas=zero_a)
    shape = dict(status=status(ns, observed_tv), total_variation=observed_tv,
                 percentile_tv_ci=interval(tvs[:ns]), valid_replicas=ns,
                 undefined_zero_either_total_replicas=zero_a + zero_b - zero_both,
                 ref_fraction=total_a / sa if sa else None,
                 cand_fraction=total_b / sb if sb else None,
                 fraction_difference=delta,
                 fraction_difference_bootstrap_rms=np.sqrt(squares / ns) if ns else None,
                 fraction_difference_bootstrap_bias=shifts / ns if ns else None,
                 centered_tv_null=None)
    if ns:
        exceedances = int(np.count_nonzero(null_tvs[:ns] >= observed_tv))
        tail = (exceedances + 1) / (ns + 1)
        shape["centered_tv_null"] = dict(
            status="conditional_on_defined_replicas" if ns < replicas else "approximate_diagnostic",
            definition="T0*=0.5*sum(abs((q*-p*)-(q-p))); compare to observed TV",
            quantile_probabilities=[0.5, 0.9, 0.95, 0.99],
            quantiles=np.quantile(null_tvs[:ns], [0.5, 0.9, 0.95, 0.99]),
            mean=float(null_tvs[:ns].mean()), rms=float(np.sqrt(np.square(null_tvs[:ns]).mean())),
            exceedances=exceedances, valid_replicas=ns,
            tail_probability_plus_one=tail,
            monte_carlo_se_approx=float(np.sqrt(tail * (1 - tail) / ns)),
            minimum_tail_resolution=1 / (ns + 1),
            interpretation="Unadjusted centered-bootstrap diagnostic, not exact p-value; no acceptance threshold.")
    result = dict(shards=n, bins=bins, total_ref=sa, total_cand=sb,
                  total_difference=sb - sa, rate_ratio=ratio, shape=shape,
                  zero_total_replicas=dict(ref=zero_a, cand=zero_b, both=zero_both),
                  shard_totals_ref=a.sum(axis=1), shard_totals_cand=b.sum(axis=1))
    if _has_nonfinite(result):
        raise ValueError("nonfinite arithmetic in bootstrap summaries")
    return _json_safe(result)


def _shard_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+[_:][0-9]+", value):
        raise ValueError("shard_id must be a run_subrun string, e.g. 001400_00000011")
    run, subrun = map(int, re.split("[_:]", value))
    return run, subrun


def _filename_identity(path):
    # Only the basename, never a caller-created directory label, is evidence.
    matches = re.findall(r"(?<![0-9])([0-9]{4,})_([0-9]+)(?![0-9])", path.name)
    ids = {(int(r), int(s)) for r, s in matches}
    if len(ids) > 1:
        raise ValueError(f"ambiguous filename identity: {path}")
    return next(iter(ids)) if ids else None


def _identity(root, path, expected_id, expected_events):
    filename_id = _filename_identity(path)
    if filename_id is not None and filename_id != expected_id:
        raise ValueError(f"filename shard identity mismatch: {path}")
    inventory, _ = _inventory(root, None, [])
    items = inventory.get("crvdump/events", [])
    result = dict(filename_run_subrun=list(filename_id) if filename_id is not None else None,
                  event_census=None, expected_events=expected_events,
                  expected_events_status="not_supplied" if expected_events is None else "unverified_histogram_only")
    keys = None
    if items:
        if len(items) != 1:
            raise ValueError(f"ambiguous event census: {path}")
        item = items[0]
        tree = root[item["key"]]
        if not hasattr(tree, "arrays"):
            raise ValueError(f"highest event-tree cycle is not a tree: {path}")
        try:
            cols = tree.arrays(["run", "subrun", "event"], library="np")
            arrays = [np.asarray(cols[k]) for k in ("run", "subrun", "event")]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid event census: {path}") from exc
        if any(v.ndim != 1 or v.dtype.kind not in "iu" or (v < 0).any() or
               (v > np.iinfo(np.int64).max).any() for v in arrays):
            raise ValueError(f"event identities must be nonnegative int64 values: {path}")
        keys = np.stack(arrays, axis=1).astype("<i8")
        if not len(keys) or not np.all(keys[:, :2] == expected_id):
            raise ValueError(f"event-tree shard identity mismatch or empty census: {path}")
        keys = keys[np.lexsort(keys.T[::-1])]
        if np.any(np.all(keys[1:] == keys[:-1], axis=1)):
            raise ValueError(f"duplicate event identity: {path}")
        if expected_events is not None and len(keys) != expected_events:
            raise ValueError(f"expected event count mismatch: {path}")
        result["event_census"] = dict(cycle=item["cycle"], rows=len(keys),
                                     sorted_ids_sha256=hashlib.sha256(keys.tobytes()).hexdigest())
        result["expected_events_status"] = "verified" if expected_events is not None else "not_supplied"
    if filename_id is None and keys is None:
        raise ValueError(f"no actual shard identity: need run_subrun filename or crvdump/events: {path}")
    return result, keys


def _load_pairs(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("pairs JSON must be an explicit list of at least two shard pairs")
    seen_ids, seen_files, result = set(), set(), []
    for row in rows:
        if not isinstance(row, dict) or not {"shard_id", "ref", "cand"} <= row.keys():
            raise ValueError("every pair needs shard_id, ref and cand")
        if row.keys() - {"shard_id", "ref", "cand", "expected_events"}:
            raise ValueError(f"unknown pair fields: {sorted(row.keys() - {'shard_id', 'ref', 'cand', 'expected_events'})}")
        identity = _shard_id(row["shard_id"])
        if identity in seen_ids:
            raise ValueError("duplicate shard identity")
        seen_ids.add(identity)
        count = row.get("expected_events")
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count <= 0):
            raise ValueError("expected_events must be a positive integer")
        item = dict(shard_id=f"{identity[0]:06d}_{identity[1]:08d}", identity=identity, expected_events=count)
        pair_files = set()
        for arm in ("ref", "cand"):
            name = row[arm]
            if not isinstance(name, str) or not name or "://" in name or any(c in name for c in "*?[]"):
                raise ValueError("ROOT paths must be explicit local paths, not URLs or globs")
            source = (path.parent / name).resolve()
            if not source.is_file():
                raise ValueError(f"missing ROOT file: {source}")
            stat = source.stat()
            file_id = (stat.st_dev, stat.st_ino)
            if file_id in seen_files:
                raise ValueError(f"file reused across shard pairs: {source}")
            pair_files.add(file_id)
            item[arm] = source
        # Same file in both arms is useful for exact-closure controls, not another shard.
        seen_files.update(pair_files)
        result.append(item)
    return sorted(result, key=lambda r: r["identity"])


def _metadata(root, inventory):
    return {key: dict(name=root[items[0]["key"]].member("fName"),
                      title=root[items[0]["key"]].member("fTitle"))
            for key, items in inventory.items() if len(items) == 1 and items[0]["class"] == "TNamed"}


def _histogram(root, item, key, count_keys):
    hist = root[item["key"]]
    classification, reason = _classification(key, getattr(hist, "title", ""), item["class"], count_keys)
    if classification != "count_like":
        raise ValueError(f"{key}: {classification}; use source-reviewed exact --count-key for count maps, or --exclude-key")
    values, _, edges, labels, masks, meta = _read(hist, classification)
    if meta["issues"] or _has_nonfinite(meta):
        raise ValueError(f"{key}: invalid source: {meta['issues']}; nonfinite metadata/totals={_has_nonfinite(meta)}")
    if (values < 0).any():
        raise ValueError(f"{key}: negative bin values")
    stats = ["fEntries", "fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"]
    if values.ndim >= 2:
        stats += ["fTsumwy", "fTsumwy2", "fTsumwxy"]
    if values.ndim >= 3:
        stats += ["fTsumwz", "fTsumwz2", "fTsumwxz", "fTsumwyz"]
    for field in stats:
        value = float(hist.member(field))
        _finite(np.array([value]), f"{key} {field}")
        if field in ("fEntries", "fTsumw", "fTsumw2") and value < 0:
            raise ValueError(f"{key}: negative {field}")
    _finite(np.array([values.sum()]), f"{key} total")
    booking = _json_safe(dict(classname=item["class"], title=hist.title,
                             flow_array_shape=list(values.shape), edges=edges, axis_labels=labels,
                             axis_titles=[axis.member("fTitle") for axis in hist.axes]))
    provenance = dict(**item, classification_reason=reason, sumw2_present=meta["sumw2_present"],
                      entries=float(hist.member("fEntries")))
    return values, masks, booking, provenance


def run(pairs, out, *, select=CRV_ONLY, aliases=(), count_keys=(), exclude_keys=(),
        replicas=10000, seed=12345, batch_size=256, confidence=0.95):
    """Validate every shard and write one strict JSON report to a NEW local path."""
    _options(replicas, batch_size, confidence, seed)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    rows = _load_pairs(pairs)
    selection = re.compile(select) if select else None
    alias_pairs = _aliases(aliases)
    count_keys, exclude_keys = set(count_keys), set(exclude_keys)
    if count_keys & exclude_keys:
        raise ValueError("count-key and exclude-key must not overlap")
    records, data, bookings, regions, sources = [], {}, {}, {}, {}
    expected_keys = expected_metadata = None
    excluded_seen, metadata_sources = set(), []
    for row in rows:
        pair_record = dict(shard_id=row["shard_id"], expected_events=row["expected_events"])
        event_keys = {}
        for arm in ("ref", "cand"):
            path = row[arm]
            source_hash = digest(path)
            with uproot.open(path) as root:
                identity, event_keys[arm] = _identity(root, path, row["identity"], row["expected_events"])
                pair_record[arm] = dict(path=str(path), sha256=source_hash, identity=identity)
                inventory, _ = _inventory(root, None, alias_pairs)
                if any(len(items) != 1 for items in inventory.values()):
                    raise ValueError(f"alias collision: {path}")
                metadata = _metadata(root, inventory)
                if expected_metadata is None:
                    expected_metadata = metadata
                elif metadata != expected_metadata:
                    raise ValueError(f"missing or mismatched TNamed metadata: {path}")
                metadata_sources.append(dict(shard_id=row["shard_id"], arm=arm,
                    objects={k: inventory[k][0] for k in metadata}))
                selected = {k for k in inventory if selection is None or selection.search(k)}
                excluded_seen.update(selected & exclude_keys)
                selected -= exclude_keys | metadata.keys()
                if not selected:
                    raise ValueError(f"empty histogram selection: {path}")
                if expected_keys is None:
                    expected_keys = selected
                elif selected != expected_keys:
                    raise ValueError(f"missing/different selected histogram keys: {path}; missing={sorted(expected_keys - selected)}, extra={sorted(selected - expected_keys)}")
                for key in sorted(selected):
                    values, masks, booking, source = _histogram(root, inventory[key][0], key, count_keys)
                    if key in bookings and booking != bookings[key]:
                        raise ValueError(f"class/axes/title metadata mismatch: {key} in {path}")
                    bookings[key], regions[key] = booking, masks
                    data.setdefault(key, {"ref": [], "cand": []})[arm].append(values)
                    sources.setdefault(key, []).append(dict(shard_id=row["shard_id"], arm=arm, **source))
            if digest(path) != source_hash:
                raise ValueError(f"input changed during reading: {path}")
        kr, kc = event_keys["ref"], event_keys["cand"]
        if kr is not None and kc is not None:
            if not np.array_equal(kr, kc):
                raise ValueError(f"paired event census mismatch: {row['shard_id']}")
            pair_record["pairing_evidence"] = "matching_persisted_event_ids; resampling_unit_remains_shard"
        else:
            pair_record["pairing_evidence"] = "run_subrun_only; event_pairing_and_completeness_unverified"
        records.append(pair_record)
    if count_keys - expected_keys:
        raise ValueError(f"unused count-key: {sorted(count_keys - expected_keys)}")
    if exclude_keys - excluded_seen:
        raise ValueError(f"unused exclude-key: {sorted(exclude_keys - excluded_seen)}")
    histograms = {}
    for key in sorted(data):
        a, b = np.stack(data[key]["ref"]), np.stack(data[key]["cand"])
        report = dict(booking=bookings[key], sources=sources[key], regions={})
        for region in ("inclusive", "visible"):
            mask = regions[key][region]
            result = bootstrap_region(a[:, mask], b[:, mask], replicas=replicas, seed=seed,
                                      batch_size=batch_size, confidence=confidence)
            result["flat_indices_in_flow_array_C_order"] = np.flatnonzero(mask.ravel()).tolist()
            report["regions"][region] = result
        flow = regions[key]["flow"]
        report["flow_totals"] = dict(ref=float(a[:, flow].sum()), cand=float(b[:, flow].sum()),
                                    definition="complement of all-axes-visible; corners counted once")
        histograms[key] = report
    report = dict(schema_version=1, pairs_json=str(Path(pairs).resolve()),
                  pairs_json_sha256=hashlib.sha256(Path(pairs).read_bytes()).hexdigest(),
                  shards=records, select=select, aliases=list(aliases),
                  explicit_count_keys=sorted(count_keys), excluded_keys=sorted(exclude_keys),
                  metadata=expected_metadata, metadata_sources=metadata_sources,
                  bootstrap=dict(unit="paired_shard", scheme="n draws with replacement; same multinomial weights for both arms and all histograms/regions",
                                 replicas=replicas, seed=seed, rng="numpy.default_rng/PCG64", numpy_version=np.__version__,
                                 batch_size=batch_size, confidence=confidence, interval="pointwise percentile",
                                 bin_uncertainty="RMS of centered bootstrap fraction difference; no independent-bin model"),
                  multiplicity=dict(histograms=len(histograms), region_comparisons=2 * len(histograms),
                                    adjustment="none", interpretation="Dependent exploratory diagnostics; no family-wise coverage or blanket acceptance"),
                  rate_interpretation="candidate/reference fill-weight total ratio; not an independently exposure-corrected physical rate",
                  limitations=LIMITATIONS, histograms=histograms)
    # Exclusive creation prevents accidental overwrite; validation failures write nothing.
    text = json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    with out.open("x") as stream:
        stream.write(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", help="explicit JSON list of shard_id/ref/cand/[expected_events]")
    parser.add_argument("--out", required=True, help="new JSON file; never overwrite")
    parser.add_argument("--select", default=CRV_ONLY, help="canonical-key regex; empty selects all objects")
    parser.add_argument("--alias", action="append", default=[], help="Validation prefix OLD=NEW")
    parser.add_argument("--count-key", action="append", default=[], help="exact source-reviewed canonical count histogram")
    parser.add_argument("--exclude-key", action="append", default=[], help="exact canonical omission; recorded, never silently dropped")
    parser.add_argument("--replicas", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    try:
        report = run(args.pairs, args.out, select=args.select, aliases=args.alias,
                     count_keys=args.count_key, exclude_keys=args.exclude_key, replicas=args.replicas,
                     seed=args.seed, batch_size=args.batch_size, confidence=args.confidence)
    except (ValueError, OSError, KeyError, TypeError, re.error) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(dict(shards=len(report["shards"]), histograms=len(report["histograms"]), out=args.out)))


if __name__ == "__main__":
    main()
