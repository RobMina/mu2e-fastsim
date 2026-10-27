"""Read-only full-support audit of a certified corrected library; never publish ratio variants."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from crvfastsim import tables
from valhist import denominator_audit as A
from valhist import shower_uncertainty as U
from valhist.genassoc_campaign import _new_json, verify_library
from valhist.genassoc_fullgen import compare_payload
from valhist.library_variants import _new_output, _positive_int, sha256


def support_report(counts, events, showers, generated, replicas, seed):
    """Keep all native-acceptance draws, but withhold nonlinear propagation if any is invalid."""
    _positive_int(replicas, "replicas", 2)
    _positive_int(seed, "seed", 0)
    _positive_int(generated, "generated")
    counts, events, showers = [np.asarray(a) for a in (counts, events, showers)]
    for array in (counts, events, showers):
        if array.dtype.kind not in "iu" or np.any(array < 0):
            raise ValueError("counts, exposures and showers must be nonnegative integer arrays")
    result = A.propagate(counts, events, showers, generated, replicas, seed)
    nominal_ok = not any(result["nominal_missing_support"]) and not any(result["nominal_saturated"])
    replicas_ok = result["any_invalid_replicas"] == 0
    files = (counts > 0).sum(axis=0)
    fraction = result["any_invalid_replicas"] / replicas
    z = 1.959963984540054
    center = (fraction + z * z / (2 * replicas)) / (1 + z * z / replicas)
    radius = z * np.sqrt(fraction * (1 - fraction) / replicas + z * z / (4 * replicas ** 2)) / (1 + z * z / replicas)
    result.update(invalid_fraction=fraction, invalid_fraction_mc_wilson95=[center - radius, center + radius],
                  nominal_support_pass=nominal_ok, ensemble_support_pass=nominal_ok and replicas_ok,
                  downstream_propagation_authorized=False,
                  status="SUPPORT_PASS_ONLY" if nominal_ok and replicas_ok else "BLOCKED_SUPPORT",
                  generator_files_per_cell=files.tolist(),
                  exact_zero_support_probability=((1 - files / len(events)) ** len(events)).tolist(),
                  covariance_scope="Native sampler parameter diagnostic over ALL draws, not a valid ratio-ensemble covariance.")
    return result


def certify_generators(paths, certificate, cells, hashes):
    """Reuse the strict full-stream census while independently checking every legacy generator value."""
    cert_path = Path(certificate).resolve(strict=True)
    hashes.update(U.fingerprint([cert_path]))
    cert = json.loads(cert_path.read_text())
    if (cert.get("schema") != "valhist.genassoc_fullgen.v1" or cert.get("cross_file_event_overlap") != 0
            or cert["input_sha256_pre"] != cert["input_sha256_post"]
            or cert["code_sha256_pre"] != cert["code_sha256_post"]):
        raise ValueError("invalid full-generator certificate")
    hashes.update(cert["evidence_sha256"])
    counts, events, populations, rows = [], [], [], []
    for path in paths:
        row = cert["files"][path]
        if (not row["legacy_values_identical"] or sha256(path) != cert["input_sha256_post"][path]
                or sha256(row["v2path"]) != row["digest"]):
            raise ValueError("full-generator certificate payload drift")
        hashes[path] = cert["input_sha256_post"][path]
        hashes[row["v2path"]] = row["digest"]
        report, keys = compare_payload(path, row["v2path"])
        if any(report[k] != row[k] for k in ("event_count", "genrows", "zero_generator_events", "legacy_values_identical")):
            raise ValueError("full-generator certificate census differs")
        g = U.bin_frame(tables.read(path, "gen"))
        counts.append(np.bincount(U.cell_rows(g, cells), minlength=len(cells)))
        events.append(len(keys))
        populations.append(U.event_index(pd.DataFrame(keys, columns=U.KEY)))
        rows.append({"path": path, "schema_v2_census": row["v2path"], **report})
    combined = populations[0].append(populations[1:])
    if not combined.is_unique:
        raise ValueError("overlapping full-generator files")
    return np.asarray(counts), np.asarray(events), combined, rows


def run(cache, certificate, out, replicas=1000, seed=20261025, legacy_library=None):
    start = time.monotonic()
    out = _new_output(out)
    root = Path(cache).resolve(strict=True)
    meta = json.loads((root / "cache.json").read_text())
    hashes = U.fingerprint([root / "cache.json", root / "index.npz"])
    U.verify(meta["output_sha256"])
    hashes.update(meta["input_sha256"])
    hashes.update(meta["output_sha256"])
    campaign = Path(meta["campaign"])
    verify_library(campaign)
    library_meta = json.loads((campaign / "library/library.json").read_text())
    if (str(campaign / "library/library.root") != meta["library"]
            or library_meta["generated_scaled_to"] != meta["generated"]
            or library_meta["gen"] != meta["gen"]
            or library_meta["inputs"] != [d["path"] for d in meta["donors"]]):
        raise ValueError("cache does not describe the completed corrected library")
    for donor in meta["donors"]:
        validation = donor["validation"]
        if (validation["schema_version"] != 2 or validation["associations_checked"] is not True
                or validation["ancestry_cycles"] != 0):
            raise ValueError("cache lacks strict donor validation")
    code = {**U.code_identity(), **U.fingerprint([__file__, A.__file__,
            Path(__file__).with_name("genassoc_fullgen.py"), Path(__file__).with_name("genassoc_campaign.py")])}
    with np.load(root / "index.npz", allow_pickle=False) as saved:
        index = dict(saved)
    cells = pd.DataFrame({k: index[k] for k in U.BIN})
    if cells.duplicated().any():
        raise ValueError("duplicate cached bin")
    counts, events, genids, records = certify_generators(meta["gen"], certificate, cells, hashes)
    if not np.array_equal(counts, index["gen_counts"]) or not np.array_equal(events, index["gen_events"]):
        raise ValueError("fresh denominator census differs from cache")
    donorids = U.event_index(pd.DataFrame(index["donor_events"], columns=U.KEY))
    if not donorids.is_unique or len(donorids.intersection(genids)):
        raise ValueError("donor/denominator event overlap")
    sh = tables.read(meta["library"], "libshower")
    reach = tables.read(meta["library"], "libgen").sort_values(U.BIN).reset_index(drop=True)
    if not np.array_equal(reach[U.BIN].to_numpy(), cells.to_numpy()):
        raise ValueError("library/cache bin mismatch")
    ns = np.bincount(U.cell_rows(sh, cells), minlength=len(cells))
    if not np.array_equal(ns, reach.nshower) or not np.array_equal(ns, np.bincount(index["cell"], minlength=len(cells))):
        raise ValueError("library/cache shower-count mismatch")
    corrected = support_report(counts, events, ns, meta["generated"], replicas, seed)
    if not np.array_equal(corrected["nominal_scaled_denominator"], reach.ngen):
        raise ValueError("fresh nominal denominator differs from libgen")
    result = {"schema": "valhist.corrected_denominator.v1", "library": meta["library"],
              "library_sha256": hashes[meta["library"]], "generated_exposure": meta["generated"],
              "generator_events": int(events.sum()), "donor_events": len(donorids), "donor_generator_overlap": 0,
              "bins": cells.to_dict("records"), "per_file_counts": counts.tolist(), "per_file_events": events.tolist(),
              "generator_census": records, "corrected": corrected}
    if legacy_library is not None:
        legacy = str(Path(legacy_library).resolve(strict=True))
        hashes.update(U.fingerprint([legacy]))
        oldsh = tables.read(legacy, "libshower")
        oldreach = tables.read(legacy, "libgen").sort_values(U.BIN).reset_index(drop=True)
        pd.testing.assert_frame_equal(oldreach[U.BIN + ["ngen"]], reach[U.BIN + ["ngen"]], check_exact=True)
        oldcounts = oldsh.groupby(U.BIN).size()
        bank = pd.MultiIndex.from_frame(cells)
        keys = bank.union(oldcounts.index).sort_values()
        expanded = pd.DataFrame(keys.to_frame(index=False).to_numpy(), columns=U.BIN)
        oldn = oldcounts.reindex(keys, fill_value=0).to_numpy()
        oldblocks = np.zeros((len(events), len(keys)), dtype=np.int64)
        oldblocks[:, keys.get_indexer(bank)] = counts
        old = support_report(oldblocks, events, oldn, meta["generated"], replicas, seed)
        result["legacy"] = {"path": legacy, "sha256": hashes[legacy], "bins": expanded.to_dict("records"), **old}
    U.verify({**hashes, **code})
    result.update(input_sha256=hashes, analysis_source_sha256=code, wall_seconds=time.monotonic() - start,
                  versions={"numpy": np.__version__, "pandas": pd.__version__}, limitations=[
                      "Retained strict donor/cached projection evidence is reused by hashes, not re-dumped or re-certified from art.",
                      "Complete generated exposure comes from strict census/payload equality, never accepted donor events.",
                      "All file-bootstrap draws retained; no clipping repair, discarded draws, redraw or nominal substitution.",
                      "Effective acceptance reproduces the existing sampler, including reject-at-zero and saturate-above-one.",
                      "Nominal shower support is not evaluated-tail support, finite-bank completeness, or physics closure.",
                      "No library, downstream covariance, reconstruction replay or response change is produced.",
                      "Frozen-library comparisons use the same bank and random multiplicities, not an independent statistical A/B."])
    _new_json(out, result)
    for name in ("legacy", "corrected"):
        if name in result:
            value = result[name]
            print(name, "missing cells", sum(value["nominal_missing_support"]),
                  "missing showers", sum(n for n, bad in zip(value["nominal_showers"], value["nominal_missing_support"]) if bad),
                  "invalid", value["any_invalid_replicas"], "/", replicas, flush=True)
    for i, row in enumerate(result["bins"]):
        if corrected["invalid_replicas_per_bin"][i]:
            print(row, "showers", ns[i], "files", corrected["generator_files_per_cell"][i],
                  "invalid", corrected["invalid_replicas_per_bin"][i], flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache")
    p.add_argument("certificate")
    p.add_argument("--out", required=True)
    p.add_argument("--legacy-library")
    p.add_argument("--replicas", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261025)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
