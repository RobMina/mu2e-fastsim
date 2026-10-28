"""Read-only POST-capture conditional species/KE support diagnostic; never admit a bank.

Requires the completed 94M aggregate, the ORIGINAL uncertainty_v2/production
cache, and the existing independent heldout_certificate (shards 21--220).
Processes one bank/evaluation file at a time; no 94M-event identity concatenation.
Only one exclusively-created JSON is written. No capture, art, library/cache
builder, remote discovery, or downstream propagation is invoked.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

# Do not create import caches when invoked as a read-only command.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from crvfastsim import genassoc, geometry, tables
from . import corrected_denominator as D
from . import descendant_audit as A
from . import generator_bank_campaign as G
from . import heldout_certificate as E
from . import shower_heldout as H
from . import shower_uncertainty as U

GENERATED = 10_000_000
BANK_EVENTS = 94_000_000
REPLICAS = 1000
SEED = 20261027
CATEGORIES = ("absent_bank", "present_but_rounded_zero", "no_template", "saturated")


class Bindings:
    """Conflict-checked content AND resolved-path bindings, verified before consumption."""

    def __init__(self):
        self.hashes = {}
        self.targets = {}

    def add(self, hashes):
        for name, expected in hashes.items():
            path = Path(name)
            if not path.is_absolute() or str(path).startswith("/pnfs/"):
                raise ValueError("require exact local absolute evidence path: " + str(path))
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(c not in "0123456789abcdef" for c in expected)):
                raise ValueError("invalid SHA256 binding")
            if name in self.hashes and self.hashes[name] != expected:
                raise ValueError("conflicting hash binding: " + name)
            target = str(path.resolve(strict=True))
            if target.startswith("/pnfs/"):
                raise ValueError("remote evidence target forbidden")
            if name in self.targets and self.targets[name] != target:
                raise ValueError("path target drift: " + name)
            if name not in self.hashes:
                if A.digest(path) != expected:
                    raise ValueError("input hash drift: " + name)
                self.hashes[name], self.targets[name] = expected, target

    def freeze(self, path):
        # Keep the lexical path too, so a symlink retarget cannot be hidden by resolve().
        name = os.path.abspath(path)
        if name.startswith("/pnfs/") or str(Path(name).resolve(strict=True)).startswith("/pnfs/"):
            raise ValueError("remote evidence target forbidden")
        if name not in self.hashes:
            self.add({name: A.digest(name)})
        return name

    def read(self, path, required=True):
        name = os.path.abspath(path)
        if required and name not in self.hashes:
            raise ValueError("unbound evidence path: " + name)
        self.freeze(name)
        return json.loads(Path(name).read_text())

    def verify(self):
        for name, expected in self.hashes.items():
            if str(Path(name).resolve(strict=True)) != self.targets[name] or A.digest(name) != expected:
                raise ValueError("input/code/path drift: " + name)
        return dict(self.hashes)


def completed_contract(report):
    """An in-progress capture receipt is not an input to this post-capture gate."""
    yes = ("conditional_capture_evidence_complete", "strict_event_census_certified",
           "observed_producer_pairs_match")
    no = ("bank_admitted_for_propagation", "downstream_authorized", "selection_independence_certified")
    if (report.get("schema") != G.SCHEMA or report.get("scope") != G.SCOPE
            or report.get("assumptions") != G.ASSUMPTIONS or report.get("limits") != G.LIMITS
            or any(report.get(k) is not True for k in yes)
            or any(report.get(k) is not False for k in no)
            or report.get("files") != 94 or report.get("events") != BANK_EVENTS
            or not report.get("input_sha256")
            or report.get("input_sha256") != report.get("input_sha256_post")):
        raise ValueError("require completed conditional 94M campaign contract; no live capture input")


def integer_array(value, label, ndim=None):
    a = np.asarray(value)
    if a.dtype.kind not in "iu" or (a < 0).any() or (a > np.iinfo(np.int64).max).any():
        raise ValueError(label + " must be nonnegative int64-compatible integers")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(label + " has wrong dimension")
    return a.astype(np.int64, copy=False)


def cell_union(*frames):
    """Union native cells only, never primary/event rows; stable sorted integer keys."""
    checked = []
    for frame in frames:
        a = integer_array(frame[U.BIN].to_numpy(), "cells", 2)
        if a.shape[1] != 2 or (a[:, 0] == 0).any() or (a[:, 0] == 13).any():
            raise ValueError("invalid nonmuon cell identity")
        checked.append(pd.DataFrame(a, columns=U.BIN))
    if not checked:
        return pd.DataFrame({k: pd.Series(dtype=np.int64) for k in U.BIN})
    return pd.concat(checked, ignore_index=True).drop_duplicates().sort_values(U.BIN).reset_index(drop=True)


def cell_counts(gen):
    return U.bin_frame(gen).groupby(U.BIN).size().astype(np.int64)


def masks(raw, denominator, showers):
    raw, denominator, showers = [integer_array(x, "support", 1) for x in (raw, denominator, showers)]
    if raw.shape != denominator.shape or raw.shape != showers.shape or ((raw == 0) & (denominator > 0)).any():
        raise ValueError("inconsistent support arrays")
    return {"absent_bank": raw == 0,
            "present_but_rounded_zero": (raw > 0) & (denominator == 0),
            "no_template": showers == 0,
            "saturated": (denominator > 0) & (showers > denominator)}


def exact_partition(census, expected=None):
    """Prove a single exact run/subrun partition, including zero-generator events."""
    ids = U.event_index(census)
    parts = census[U.KEY[:2]].drop_duplicates()
    if len(census) == 0 or not ids.is_unique or len(parts) != 1:
        raise ValueError("empty/duplicate/multipartition census")
    part = tuple(int(v) for v in parts.iloc[0])
    if expected is not None:
        run, subrun, n = expected
        if part != (run, subrun) or not np.array_equal(np.sort(census.event.to_numpy()), np.arange(1, n + 1)):
            raise ValueError("census exact partition/range mismatch")
    return part


def claim_partition(census, seen, expected=None):
    part = exact_partition(census, expected)
    if part in seen:
        raise ValueError("overlapping donor/training/bank/evaluation partition")
    seen.add(part)
    return part


def count_strict_file(old, new, run, subrun, expected_events):
    """Full schema-v2, aliases, exact legacy payload and COMPLETE census, one file."""
    report = G.B.compare_file(old, new, run, subrun, expected_events)
    return cell_counts(tables.read(new, "gen")), report


def candidate_counts_match(series, bins, expected):
    cells = pd.DataFrame(bins)
    if cells.duplicated(U.BIN).any():
        raise ValueError("duplicate candidate cells")
    cell_union(cells)
    actual = series.reindex(pd.MultiIndex.from_frame(cells[U.BIN]), fill_value=0).to_numpy()
    expected = integer_array(expected, "candidate counts", 1)
    if not np.array_equal(actual, expected):
        raise ValueError("fresh strict counts differ from legacy candidate within candidate bins")


def evaluation_pairs(certificate, report, cache, cache_sha):
    """Read the actual heldout_certificate schema (it intentionally has no schema tag)."""
    expected = list(range(21, 221))
    if (certificate.get("complete_expected_id_range") is not True
            or certificate.get("train_eval_overlap") != 0 or certificate.get("cross_shard_overlap") != 0
            or certificate.get("strict_art_role_certified") is not False
            or certificate.get("events") != GENERATED or report.get("events_inferred") != GENERATED
            or report.get("cache") != str(cache) or report.get("cache_sha256") != cache_sha
            or [r["shard"] for r in certificate["shards"]] != expected
            or [r["shard"] for r in report["shards"]] != expected
            or any(r.get("events") != 50000 or r.get("reference_generator_exact") is not True
                   for r in certificate["shards"])
            or any(r.get("events_inferred") != 50000 or r.get("reference_gen_exact") is not True
                   for r in report["shards"])):
        raise ValueError("independent 200-shard evaluation certificate/report contract mismatch")
    files = report["input_sha256"]
    pairs = []
    for k in expected:
        pair = []
        for prefix in ("gen", "crvsteps"):
            found = [p for p in files if Path(p).name == f"{prefix}.001400_{k:08d}.root"]
            if len(found) != 1:
                raise ValueError("ambiguous/unbound evaluation input pair")
            pair.append(found[0])
        pairs.append((k, *pair))
    if len(files) != 400:
        raise ValueError("evaluation certificate report must bind exactly 400 original inputs")
    return pairs


def tail_bound(primary_rows, selected, features):
    """Whole-event upper bound; each selected event counted once, even with many primaries."""
    primary_rows = integer_array(primary_rows, "primary rows", 1)
    selected = np.asarray(selected)
    features = np.asarray(features, dtype=float)
    if (selected.dtype.kind != "b" or selected.shape != primary_rows.shape
            or features.ndim != 2 or features.shape[1] != 8 or not np.isfinite(features).all()
            or (features < 0).any() or (primary_rows >= len(features)).any()):
        raise ValueError("invalid whole-event bound input")
    rows = np.unique(primary_rows[selected])
    return {"primaries": int(selected.sum()), "events": len(rows),
            "whole_event_nonmu_steps_energy": features[rows].sum(axis=0).tolist()}


def add_bound(total, value):
    total["primaries"] += value["primaries"]
    total["events"] += value["events"]
    total["whole_event_nonmu_steps_energy"] = (
        np.asarray(total["whole_event_nonmu_steps_energy"]) + value["whole_event_nonmu_steps_energy"]).tolist()


def empty_bound():
    return {"primaries": 0, "events": 0, "whole_event_nonmu_steps_energy": [0.0] * 8}


def bootstrap_tails(counts, events, evaluated):
    """Same 1000 whole-file weights as support_report; keep ALL tail-loss draws."""
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(events), np.full(len(events), 1 / len(events)), size=REPLICAS)
    raw = weights @ counts
    den = np.rint(raw * GENERATED / (weights @ events)[:, None])
    absent, rounded = raw == 0, (raw > 0) & (den == 0)
    relevant = evaluated > 0
    return {"absent_bank_replicas_per_cell": absent.sum(0).tolist(),
            "present_but_rounded_zero_replicas_per_cell": rounded.sum(0).tolist(),
            "any_evaluated_zero_denominator_replicas": int(((den == 0) & relevant).any(1).sum()),
            "replicas": REPLICAS, "seed": SEED,
            "scope": "Evaluated tail support, including cells with zero templates; no draws discarded."}


def _campaign(report_path, bound):
    report = bound.read(report_path, required=False)
    completed_contract(report)  # Fail before opening any campaign worker output.
    bound.add(report["input_sha256"])
    manifest = report["manifest"]
    if manifest not in bound.hashes:
        raise ValueError("aggregate does not bind manifest")
    plan = bound.read(manifest)
    # Register the actual historical closure before the read-only existing loader
    # consults it; no new snapshot may override a recorded checksum claim.
    bound.add(plan["input_sha256"])
    bound.add(plan["code_sha256"])
    _, loaded = G.load_plan(manifest, bound.hashes[manifest])  # READ ONLY, never aggregate/run.
    if loaded != plan:
        raise ValueError("campaign manifest changed during loading")
    if os.path.abspath(report_path) != str(Path(plan["outdir"]) / "aggregate/report.json"):
        raise ValueError("campaign report is not its manifest's aggregate/report.json")
    bound.add(plan["input_sha256"])
    bound.add(plan["code_sha256"])
    candidate = bound.read(plan["candidate"])
    checked = bound.read(Path(plan["outdir"]) / "aggregate/capture_verification.json")
    if (checked.get("schema") != "valhist.generator_capture_verify.v1"
            or checked.get("event_count") != BANK_EVENTS or len(checked["captures"]) != 94
            or checked.get("downstream_authorized") is not False
            or checked["input_sha256"] != checked["input_sha256_post"]
            or checked["code_sha256"] != checked["code_sha256_post"]):
        raise ValueError("incomplete aggregate capture verification")
    bound.add(checked["input_sha256"])
    bound.add(checked["code_sha256"])
    expected_producers = [str(Path(plan["outdir"]) / f"producer_{s}/report.json")
                          for s in sorted(r["subrun"] for r in candidate["files"])]
    if report["producers"] != expected_producers:
        raise ValueError("incomplete producer inventory")
    return report, plan, candidate, checked


def _cache(cache, bound, seen):
    root = Path(cache).absolute()
    if root.name != "production" or root.parent.name != "uncertainty_v2":
        raise ValueError("require original uncertainty_v2/production cache")
    meta = bound.read(root / "cache.json", required=False)
    bound.add(meta["input_sha256"])
    bound.add(meta["output_sha256"])
    expected_outputs = {str(root / p) for p in ("index.npz", "templates.npz", "file_cell_features.npz")}
    if set(meta["output_sha256"]) != expected_outputs:
        raise ValueError("cache projection output paths mismatch")
    loaded, index, cells, x, file_x = U.load_cache(root)  # Original API; NEVER rebuild/rewrite.
    if (loaded != meta or meta["generated"] != GENERATED or len(meta["donors"]) != 20
            or Path(meta["campaign"]).name != "strict_v2"):
        raise ValueError("strict production cache exposure/donor mismatch")
    sidecar = bound.read(Path(meta["campaign"]) / "library/library.json")
    if (meta["library"] != str(Path(meta["campaign"]) / "library/library.root")
            or sidecar["generated_scaled_to"] != GENERATED or sidecar["inputs"] != [d["path"] for d in meta["donors"]]
            or sidecar["gen"] != meta["gen"] or sidecar["geometry"] != meta["geometry"]
            or sidecar["output_sha256"] != bound.hashes[meta["library"]]
            or meta["geometry_sha256"] != bound.hashes[meta["geometry"]]):
        raise ValueError("cache/library/donor/geometry binding mismatch")
    plan = bound.read(sidecar["plan"])
    bound.add({sidecar["plan"]: sidecar["plan_sha256"],
               plan["source_metadata"]: plan["source_metadata_sha256"]})
    bound.add(plan["gen_sha256"])
    U.verify_library(meta["campaign"])
    if (plan["generated"] != GENERATED or meta["generated_per_file"] != 500000
            or plan["generated_per_file"] != 500000
            or [r["new_table"] for r in plan["donors"]] != sidecar["inputs"]):
        raise ValueError("strict donor plan/exposure mismatch")
    if (cells.duplicated().any() or len(index["cell"]) != x.shape[0]
            or file_x.shape != (20 * len(cells), x.shape[1])
            or len(index["file_index"]) != x.shape[0]
            or x.shape[1] != len(meta["names"])):
        raise ValueError("invalid cached projection dimensions")
    reach = tables.read(meta["library"], "libgen").sort_values(U.BIN).reset_index(drop=True)
    sh = tables.read(meta["library"], "libshower")
    ns = np.bincount(U.cell_rows(sh, cells), minlength=len(cells))
    if (not np.array_equal(reach[U.BIN], cells) or not np.array_equal(ns, reach.nshower)
            or not np.array_equal(ns, np.bincount(index["cell"], minlength=len(cells)))
            or not np.array_equal(index["ngen"], reach.ngen)):
        raise ValueError("persisted shower projection/cache mismatch")
    # The cache's exact projection outputs are historical evidence, not recomputed features.
    donor_rows, offset = [], 0
    for donor, record in zip(meta["donors"], plan["donors"]):
        validation = donor["validation"]
        if (validation["schema_version"] != 2 or validation["associations_checked"] is not True
                or validation["ancestry_cycles"] != 0):
            raise ValueError("cache lacks strict donor certification")
        done = bound.read(Path(meta["campaign"]) / "logs" / f"dump.{record['sequence']}.done.json")
        if (done["exit"] != 0 or done["output_sha256"] != bound.hashes[donor["path"]]
                or done["bound"].get(sidecar["plan"]) != sidecar["plan_sha256"]
                or done["bound"] != sidecar["dumper_binding"]
                or donor["showers"] != int((index["file_index"] == len(donor_rows)).sum())):
            raise ValueError("donor completion/projection binding mismatch")
        # Retired donor-build paths remain historical claims, never current-code bindings.
        bound.add({sidecar["plan"]: done["bound"][sidecar["plan"]]})
        census = tables.read(donor["path"], "events", U.KEY)
        part = claim_partition(census, seen)
        if not np.array_equal(census[U.KEY].to_numpy(), index["donor_events"][offset:offset + len(census)]):
            raise ValueError("donor exact census differs from original projection cache")
        offset += len(census)
        donor_rows.append({"path": donor["path"], "partition": list(part), "census_events": len(census),
                           "historical_dumper_binding": done["bound"], "historical_dumper_reexecuted": False})
    if offset != len(index["donor_events"]):
        raise ValueError("extra cached donor identities")
    old_rows = []
    if len(meta["gen"]) != 10:
        raise ValueError("require unchanged ten-file original denominator")
    for i, path in enumerate(meta["gen"]):
        genassoc.validate_generator_stream(path)
        census = genassoc.generator_event_keys(path)
        part = claim_partition(census, seen, (1400, i + 1, 50000))
        counts = cell_counts(tables.read(path, "gen"))
        if not counts.index.isin(pd.MultiIndex.from_frame(cells)).all():
            raise ValueError("original bank contains uncached cell")
        candidate_counts_match(counts, cells.to_dict("records"), index["gen_counts"][i])
        if len(census) != index["gen_events"][i]:
            raise ValueError("original generator exposure drift")
        old_rows.append({"path": path, "partition": list(part), "events": len(census)})
    nominal_old = np.rint(index["gen_counts"].sum(axis=0) * GENERATED / index["gen_events"].sum())
    if not np.array_equal(nominal_old, index["ngen"]):
        raise ValueError("unchanged original denominator does not reproduce persisted reach")
    return meta, cells, pd.Series(ns, index=pd.MultiIndex.from_frame(cells)), donor_rows, old_rows


def _bank(plan, candidate, checked, bound, seen):
    counts, records = [], []
    verified = {r["certificate"]: r for r in checked["captures"]}
    if len(verified) != 94:
        raise ValueError("duplicate capture verification entries")
    legacy = integer_array(candidate["per_file_counts"], "legacy candidate counts", 2)
    if legacy.shape != (94, len(candidate["bins"])):
        raise ValueError("candidate count matrix dimensions")
    for i, row in enumerate(candidate["files"]):
        subrun = row["subrun"]
        cp = str(G.certificate_path(plan, subrun))
        cert = bound.read(cp)
        if cert != G.check_certificate(plan, subrun) or cp not in verified:
            raise ValueError("capture not in aggregate verification")
        if (cert["input_sha256"] != cert["input_sha256_post"]
                or cert["code_sha256"] != cert["code_sha256_post"] or cert["art_run"]["exit"] != 0):
            raise ValueError("capture recorded drift/failure")
        bound.add(cert["input_sha256"])
        bound.add(cert["artifact_sha256"])
        # Historical Python hashes remain historical. Approved external dumper/configs must still match.
        configs = {p: h for p, h in cert["code_sha256"].items() if p.endswith(".fcl")}
        G.B.approved_configs(configs)
        bound.add(configs)
        area = Path(cert["area"])
        approved = {str(area / "CrvFastSim/src/CrvStepTableDumper_module.cc"): G.B.SOURCE_SHA256,
                    str(area / "build/al9-prof-e29-p107/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so"): G.B.PLUGIN_SHA256}
        if any(cert["code_sha256"].get(p) != h for p, h in approved.items()):
            raise ValueError("unapproved historical dumper")
        bound.add(approved)
        producer = bound.read(Path(plan["outdir"]) / f"producer_{subrun}/report.json")
        for item in producer["inputs"].values():
            if item["sha256_before"] != item["sha256_after"]:
                raise ValueError("producer evidence drift")
            bound.add({item["path"]: item["sha256_before"]})
        current = G.P.compare(Path(cp).parent / "original_processes.fcl",
                              Path(plan["outdir"]) / f"producer_{subrun}/generation.log.gz",
                              plan["reference"]["config"], plan["reference"]["log"])
        if producer != current or producer["configurations"]["crygen"]["subrun"] != subrun:
            raise ValueError("producer pair differs from completed evidence")
        part = (row["run"], subrun)
        if part in seen:
            raise ValueError("overlapping bank partition")
        dump_log = str(Path(cp).parent / "dump.log")
        if dump_log not in cert["artifact_sha256"]:
            raise ValueError("unbound capture completion log")
        G.F.parse_dump_log(Path(dump_log).read_text(), cert["source"]["art"])
        traces = [p for p in cert["artifact_sha256"] if Path(p).name.startswith("loader.")]
        if len(traces) != 1:
            raise ValueError("ambiguous loader evidence")
        plugin = area / "build/al9-prof-e29-p107/CrvFastSim/lib/libcrvfastsim_CrvStepTableDumper_module.so"
        G.B.S.loaded_plugin(Path(traces[0]).read_text(), plugin)
        strict = str(Path(cp).parent / "fullgen.root")
        if strict not in bound.hashes or bound.hashes.get(row["path"]) != row["sha256"]:
            raise ValueError("unbound strict/legacy bank input")
        c, report = count_strict_file(row["path"], strict, *part, 1_000_000)
        if (report != cert["report"] or report != verified[cp]["report"]
                or verified[cp]["source"] != cert["source"]):
            raise ValueError("fresh strict per-file schema/census differs from capture")
        seen.add(part)
        candidate_counts_match(c, candidate["bins"], legacy[i])
        counts.append(c)
        records.append({"legacy": row["path"], "strict": strict, "certificate": cp,
                        "partition": list(part), "legacy_candidate_counts_exact": True, **report})
    if set(verified) != {r["certificate"] for r in records}:
        raise ValueError("aggregate capture inventory mismatch")
    return counts, records


def _evaluation(pairs, certificate, heldout, bank, showers, geom, seen):
    counts, records, cell_bounds = [], [], {}
    categories = {k: empty_bound() for k in (*CATEGORIES, "any_tail")}
    for (k, gp, rp), certified, original in zip(pairs, certificate["shards"], heldout["shards"]):
        genassoc.validate_generator_stream(gp)
        gen = tables.read(gp, "gen")
        census = E.contiguous_census(gen, k, 50000)
        claim_partition(census, seen, (1400, k, 50000))
        refgen = tables.read(rp, "gen")
        if refgen.duplicated(U.KEY + ["idx"]).any():
            raise ValueError("duplicate reference generator identity")
        pos = pd.MultiIndex.from_frame(gen[U.KEY + ["idx"]]).get_indexer(
            pd.MultiIndex.from_frame(refgen[U.KEY + ["idx"]]))
        if (pos < 0).any():
            raise ValueError("reference generators outside evaluation population")
        pd.testing.assert_frame_equal(refgen.reset_index(drop=True), gen.iloc[pos][refgen.columns].reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
        steps, parts = tables.read(rp, "steps"), tables.read(rp, "particles")
        U.event_rows(steps, census)
        U.event_rows(parts, census)
        _, ancestry = A.annotate(steps, parts, geom)
        if ancestry != certified["ancestry"] or len(gen) != original["generator_rows"]:
            raise ValueError("evaluation differs from independent certificate/report")
        # classify joins full event/particle ancestry; no position/time or primary truth matching.
        features = H.scalar_features(steps, parts, census, geom).toarray()
        pop = U.bin_frame(gen)
        rows = U.event_rows(pop, census)
        local = cell_counts(gen)
        counts.append(local)
        raw = bank.reindex(local.index, fill_value=0).to_numpy(dtype=np.int64)
        den = np.rint(raw * GENERATED / BANK_EVENTS).astype(np.int64)
        ns = showers.reindex(local.index, fill_value=0).to_numpy(dtype=np.int64)
        labels = masks(raw, den, ns)
        cell = local.index.get_indexer(pd.MultiIndex.from_frame(pop[U.BIN]))
        for j, key in enumerate(local.index):  # Cell loops only, never event loops.
            value = tail_bound(rows, cell == j, features)
            add_bound(cell_bounds.setdefault(key, empty_bound()), value)
        selected = np.zeros(len(pop), dtype=bool)
        for name in CATEGORIES:
            selection = labels[name][cell]
            add_bound(categories[name], tail_bound(rows, selection, features))
            selected |= selection
        add_bound(categories["any_tail"], tail_bound(rows, selected, features))
        records.append({"shard": k, "generator": gp, "reference": rp, "events": len(census),
                        "generator_rows": len(gen), "reference_generator_exact": True,
                        "ancestry": ancestry, "all_event_nonmu_steps_energy": features.sum(0).tolist()})
    return counts, records, cell_bounds, categories


def _current_code():
    # Import closure already loaded above: include indirect local/core implementation,
    # not guessed historical source locations from basename-only certificate fields.
    paths = [__file__, sys.executable]
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name.split(".")[0] in {"valhist", "crvfastsim", "mu2efastsim"}:
            paths.append(path)
    return U.fingerprint(paths)


def write_new_json(out, result):
    path = Path(out)
    if path.suffix != ".json":
        raise ValueError("output must be a NEW JSON file")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    # O_EXCL also rejects broken symlinks. No directory creation or other writes.
    with path.open("x") as stream:
        stream.write(payload)


def run(campaign_report, cache, evaluation_certificate, out):
    output = Path(out).absolute()
    if str(output).startswith("/pnfs/") or str(output.resolve()).startswith("/pnfs/"):
        raise ValueError("output must be local, not pnfs")
    if os.path.lexists(output):
        raise FileExistsError(output)
    if output.suffix != ".json" or not output.parent.is_dir():
        raise ValueError("output must be a NEW JSON in an existing directory")
    start = time.monotonic()
    bound, code = Bindings(), Bindings()
    code.add(_current_code())
    report, plan, candidate, checked = _campaign(campaign_report, bound)
    seen = set()
    meta, _, showers, donors, oldbank = _cache(cache, bound, seen)
    for protected in (Path(cache), Path(meta["campaign"]), Path(plan["outdir"])):
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("output cannot be inside protected cache/library/capture campaign")
    cert = bound.read(evaluation_certificate, required=False)
    bound.add({cert["report"]: cert["report_sha256"]})
    heldout = bound.read(cert["report"])
    root = Path(cache).absolute()
    pairs = evaluation_pairs(cert, heldout, root, bound.hashes[str(root / "cache.json")])
    bound.add(heldout["input_sha256"])
    if heldout.get("candidate_bank_sha256") not in (None, bound.hashes[plan["candidate"]]):
        raise ValueError("evaluation sensitivity arm binds a different candidate bank")
    bank_counts, bank_records = _bank(plan, candidate, checked, bound, seen)
    if (sum(r["event_count"] for r in bank_records) != BANK_EVENTS
            or sum(r["genrows"] for r in bank_records) != report["genrows"]):
        raise ValueError("fresh aggregate census/row exposure mismatch")
    bank = pd.concat(bank_counts, axis=1).fillna(0).astype(np.int64).sum(axis=1)
    geom = geometry.from_offline_config(meta["geometry"])
    eval_counts, eval_records, cell_bounds, categories = _evaluation(
        pairs, cert, heldout, bank, showers, geom, seen)
    cells = cell_union(bank.index.to_frame(index=False), showers[showers > 0].index.to_frame(index=False),
                       *(s.index.to_frame(index=False) for s in eval_counts))
    keys = pd.MultiIndex.from_frame(cells)
    counts = np.stack([s.reindex(keys, fill_value=0).to_numpy(dtype=np.int64) for s in bank_counts])
    events = np.asarray([r["event_count"] for r in bank_records], dtype=np.int64)
    ns = showers.reindex(keys, fill_value=0).to_numpy(dtype=np.int64)
    evaluated = np.stack([s.reindex(keys, fill_value=0).to_numpy(dtype=np.int64) for s in eval_counts]).sum(0)
    support = D.support_report(counts, events, ns, GENERATED, REPLICAS, SEED)
    raw = counts.sum(0)
    denominator = np.asarray(support["nominal_scaled_denominator"], dtype=np.int64)
    labels = masks(raw, denominator, ns)
    tails = []
    for j, key in enumerate(keys):
        tails.append({**cells.iloc[j].to_dict(), "raw_bank_primaries": int(raw[j]),
                      "rounded_denominator": int(denominator[j]), "showers": int(ns[j]),
                      **{k: bool(labels[k][j]) for k in CATEGORIES},
                      **cell_bounds.get(key, empty_bound())})
    evaluated_bootstrap = bootstrap_tails(counts, events, evaluated)
    nominal_evaluated = not bool(((evaluated > 0) & (denominator == 0)).any())
    result = {"schema": "valhist.generator_bank_support.v1", "diagnostic_only": True,
              "status": "POST_CAPTURE_CONDITIONAL_DIAGNOSTIC_ONLY",
              "nominal_evaluated_bank_support_pass": nominal_evaluated,
              "ensemble_evaluated_bank_support_pass": nominal_evaluated and
                  evaluated_bootstrap["any_evaluated_zero_denominator_replicas"] == 0,
              "all_evaluated_cells_have_template": not bool(((evaluated > 0) & (ns == 0)).any()),
              "scope": G.SCOPE, "assumptions": G.ASSUMPTIONS,
              "campaign_report": os.path.abspath(campaign_report), "cache": str(root),
              "evaluation_certificate": os.path.abspath(evaluation_certificate),
              "generated_exposure": GENERATED, "bank_events": int(events.sum()),
              "evaluation_events": sum(r["events"] for r in eval_records),
              "bank_admitted_for_propagation": False, "downstream_authorized": False,
              "physical_zero_certified": False, "selection_independence_certified": False,
              "strict_event_census_certified": True, "evaluation_strict_art_role_certified": False,
              "donor_bank_evaluation_partition_overlap": 0,
              "bins": cells.to_dict("records"), "per_file_counts": counts.tolist(),
              "per_file_events": events.tolist(), "evaluated_primaries_per_cell": evaluated.tolist(),
              "support": support, "evaluated_bootstrap_tails": evaluated_bootstrap,
              "cells": tails, "evaluated_tail_categories": categories,
              "feature_order": [f"{sector}_{quantity}" for sector in U.GROUPS for quantity in ("steps", "energy_MeV")],
              "bank_files": bank_records, "donors": donors, "original_bank": oldbank,
              "evaluation_files": eval_records,
              "historical_cache_code_sha256": meta["code_sha256"],
              "historical_evaluation_certificate_source_sha256": cert["source_sha256"],
              "historical_evaluation_report_source_sha256": heldout["source_sha256"],
              "limitations": [
                  "Conditional diagnostic only; neither support pass nor zero observed activity authorizes propagation or certifies a physical zero.",
                  "Host-related missingness and distinct-CRY-stream assumptions are accepted, not demonstrated independence.",
                  "Whole-event nonmuon G4 bounds use identity ancestry, not attribution to a particular tail primary. Categories overlap; category unions deduplicate events.",
                  "Per-cell bounds may overlap in events and must not be added as disjoint contributions. All four sectors retained.",
                  "All 1000 whole-file draws retained, without clipping repair, dropping or redrawing invalid support. Fixed strict donors, generated exposure 10M.",
                  "No donor variance, physical unseen-tail bound, nonlinear closure, or joint timing/energy claim.",
                  "Evaluation reuses the exact independently certified legacy 21--220 population; no strict art stream/producer proof is invented.",
                  "Original cache projection outputs and donor certificates reused by exact hashes; retired donor dumper files are historical provenance, not present-day execution evidence.",
                  "No cache, library, old denominator or historical report is rewritten.",
                  "Historical source hashes without recorded paths are preserved as historical fields, never assigned guessed source metadata."]}
    # No output is created unless every consumed file/path AND current implementation is stable.
    if _current_code() != code.hashes:
        raise ValueError("current code closure drift")
    result.update(input_sha256=dict(bound.hashes), input_sha256_post=bound.verify(),
                  input_resolved_paths=dict(bound.targets), code_sha256=dict(code.hashes),
                  code_sha256_post=code.verify(), seconds=time.monotonic() - start)
    write_new_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-report", "cache", "evaluation-certificate", "out"):
        parser.add_argument("--" + name, required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
