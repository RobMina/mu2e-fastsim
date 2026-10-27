"""Muon loss/range and msc tables with explicit material and source-file provenance.

Build (never replaces an existing file)::
  python3 -m beamfastsim.g4tables EXPORT/tables --material StoppingTarget_Al --out NEW.npz
Read-only comparison with a fresh export::
  python3 -m beamfastsim.g4tables compare EXPORT EXISTING.npz --material StoppingTarget_Al --require-audited-export

Native nodes are in MeV/mm. msc_lambda1 is derived from LambdaMod0.msc.mu+;
using it for both charges is an assumption, not a check of mu- msc equivalence.
Hashes identify consumed bytes, not historical binaries or physics qualification.
Legacy package tables remain readable but are explicitly unqualified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from . import physics_tables as pt

SLAB = Path(__file__).with_name("data") / "g4_al_mu.npz"
TARGET = Path(__file__).with_name("data") / "g4_target_al_mu.npz"
DEFAULT = TARGET
PATHS = {"target": TARGET, "slab": SLAB}
EMAX = 1000.0
CHARGES = ("mu-", "mu+")
MSC = "LambdaMod0.msc.mu+.asc"
REQUIRED_FILES = ("couple.dat", "cut.dat", "material.dat") + tuple(
    f"{name}.muIoni.{charge}.asc" for charge in CHARGES for name in pt.TABLES) + (MSC,)
ARRAYS = ("energy", "msc_energy", "msc_lambda1") + tuple(
    f"{name}_{charge}" for charge in CHARGES for name in ("csda_range", "dedx_nr"))
SCOPE = ("Source-byte traceability and recorded export completion only; no physics validation, "
         "historical binary certification, or retroactive qualification of another NPZ. "
         "The shared msc array is sourced from mu+ only.")


def _identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def _directory(path):
    path = Path(path)
    return path / "tables" if (path / "tables").is_dir() else path


def _verified_records(records, directory, label):
    """Verify the local bundle, never a stale absolute receipt path elsewhere."""
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label}: missing hash records")
    result = {}
    for record in records:
        name = Path(record["path"]).name
        if name in result:
            raise ValueError(f"{label}: duplicate filename {name}")
        actual = _identity(directory / name)
        if actual["sha256"] != record.get("sha256") or actual["bytes"] != record.get("bytes"):
            raise ValueError(f"{label}: hash/size mismatch for {name}")
        result[name] = actual
    return result


def _successful_log(root, wrapper=False):
    log = (root / "job.log").read_text()
    if "Art has completed and will exit with status 0" not in log:
        raise ValueError("successful table-export job required")
    if wrapper and re.findall(r"^exit=(\S+)\s*$", log, re.M) != ["0"]:
        raise ValueError("full export requires exactly one exit=0 record")
    if re.search(r"command not found|illegal application state|command refused|not found in the process table|"
                 r"illegal process|illegal parameter|command aborted|batch is interrupted|G4Exception.*Fatal", log, re.I):
        raise ValueError("table-export log contains a macro/fatal error")
    return log


def _audit_export(directory, files):
    root = directory.parent
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("complete") is not True or type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0:
            raise ValueError("receipt requires complete=true and exit_code=0")
        _successful_log(root)
        artifacts = _verified_records(receipt.get("artifacts"), root, "receipt artifacts")
        if not {"job.log", "expanded.fcl"} <= artifacts.keys():
            raise ValueError("receipt artifacts must cover job.log and expanded.fcl")
        recorded = _verified_records(receipt.get("tables"), directory, "receipt tables")
        if not files.keys() <= recorded.keys():
            raise ValueError("receipt tables omit consumed tables")
        artifacts["receipt.json"] = _identity(receipt_path)
        return dict(audited=True, kind="slab_receipt", artifacts=artifacts,
                    release_observed=receipt.get("release"), export_time_table_hashes=True,
                    scope=SCOPE)
    required = ("provenance.txt", "expanded.fcl", "job.log", "export.fcl", "job.mac")
    present = [name for name in required if (root / name).exists()]
    if not present:
        return dict(audited=False, kind="none", artifacts={}, release_observed=None,
                    export_time_table_hashes=False, scope=SCOPE)
    if len(present) != len(required):
        raise ValueError("incomplete full export bundle: missing " + ", ".join(set(required) - set(present)))
    log = _successful_log(root, wrapper=True)
    text = (root / "provenance.txt").read_text()
    release = re.search(r"^MUSE_WORK_DIR=(.+)$", text, re.M)
    if not release:
        raise ValueError("provenance.txt has no MUSE_WORK_DIR release record")
    records = []
    local = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if match:
            digest, path = match.groups()
            records.append(dict(path=path, sha256=digest.lower()))
            # These are the only output-bundle files hashed by the full-export wrapper.
            name = Path(path).name
            if name in ("export.fcl", "job.mac"):
                if name in local or _identity(root / name)["sha256"] != digest.lower():
                    raise ValueError(f"provenance hash mismatch/duplicate for {name}")
                local[name] = digest.lower()
    if set(local) != {"export.fcl", "job.mac"}:
        raise ValueError("provenance.txt must hash export.fcl and job.mac")
    g4 = re.search(r"Geant4 version Name:\s*([^\n]+)", log)
    return dict(audited=True, kind="full_export_script", artifacts={n: _identity(root / n) for n in required},
                release_observed=release.group(1), geant4_observed=g4.group(1).strip() if g4 else None,
                recorded_input_hashes=records, verified_provenance_files=sorted(local),
                export_time_table_hashes=False,
                scope=SCOPE + " Full-export script has no export-time table hashes; table/config/log hashes "
                "are captured now. External provenance input hashes are recorded, not reverified.")


def _coverage(meta):
    provenance = meta.get("provenance")
    files = provenance.get("files", {}) if isinstance(provenance, dict) else {}
    if not isinstance(files, dict):
        files = {}
    valid = {name for name, item in files.items() if isinstance(item, dict)
             and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
             and type(item.get("bytes")) is int and item["bytes"] > 0}
    return dict(complete=set(REQUIRED_FILES) <= valid, missing=sorted(set(REQUIRED_FILES) - valid),
                metadata={n: n in valid for n in ("couple.dat", "cut.dat", "material.dat")},
                charges={c: {n: f"{n}.muIoni.{c}.asc" in valid for n in pt.TABLES} for c in CHARGES},
                msc={"source_particle": "mu+", "sha256_covered": MSC in valid,
                     "applied_to": list(CHARGES), "charge_equivalence_verified": False})


def _qualification(meta):
    coverage = _coverage(meta)
    provenance = meta.get("provenance", {})
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1 or not coverage["complete"]:
        return dict(status="unqualified_legacy_or_incomplete_provenance", audited_export=False, scope=SCOPE)
    audit = provenance.get("audit", {})
    if not isinstance(audit, dict):
        audit = {}
    required = {"slab_receipt": {"receipt.json", "job.log", "expanded.fcl"},
                "full_export_script": {"provenance.txt", "expanded.fcl", "job.log", "export.fcl", "job.mac"}}
    artifacts = audit.get("artifacts", {})
    valid = {name for name, item in artifacts.items() if isinstance(item, dict)
             and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
             and type(item.get("bytes")) is int and item["bytes"] > 0} if isinstance(artifacts, dict) else set()
    audited = (audit.get("audited") is True and audit.get("kind") in required
               and required[audit["kind"]] <= valid)
    return dict(status="audited_export_record" if audited else "unqualified_hashes_only",
                audited_export=audited, scope=audit.get("scope", SCOPE))


def _validate(out, expected_material=None, require_audited_export=False):
    meta = out.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("material"), str) or not meta["material"].strip():
        raise ValueError("table metadata must name a material")
    if expected_material is not None and meta["material"] != expected_material:
        raise ValueError(f"material mismatch: {meta['material']!r} != {expected_material!r}")
    couples = meta.get("couples", {})
    if not isinstance(couples, dict):
        raise ValueError("invalid material/cuts couples metadata")
    for couple in [meta.get("couple")] + list(couples.values()):
        if couple is not None and (not isinstance(couple, dict) or couple.get("material") != meta["material"]):
            raise ValueError("material disagrees with recorded couple")
    for name in ARRAYS:
        if name not in out:
            raise ValueError(f"missing array {name}")
        value = np.asarray(out[name])
        if (value.ndim != 1 or len(value) < 2 or value.dtype.kind not in "fiu"
                or not np.all(np.isfinite(value)) or np.any(value <= 0)):
            raise ValueError(f"{name}: expected a finite positive 1D array with at least two nodes")
        if name in ("energy", "msc_energy") or name.startswith("csda_range_"):
            if np.any(np.diff(value.astype(float)) <= 0):
                raise ValueError(f"{name}: grid/range must be strictly increasing")
        grid = "msc_energy" if name.startswith("msc_") else "energy"
        if value.shape != np.asarray(out[grid]).shape:
            raise ValueError(f"{name}: array/grid shape mismatch")
    meta["qualification"] = _qualification(meta)
    if require_audited_export and not meta["qualification"]["audited_export"]:
        raise ValueError("audited export required; table is unqualified")
    return out


def _extract(directory, material, require_audited_export=False):
    if not isinstance(material, str) or not material.strip():
        raise ValueError("explicit nonempty material required")
    directory = _directory(directory)
    # Snapshot before reading; reject a changed bundle rather than hash different bytes after parsing.
    files = {name: _identity(directory / name) for name in REQUIRED_FILES}
    arrays, couples = {}, {}
    energy = None
    for charge in CHARGES:
        tables, info, paths = pt.load_target(directory, material=material, particle=charge)
        if {p.name for p in paths} - files.keys():
            raise ValueError("unrecorded consumed physics tables")
        couples[charge] = info
        r, d = tables["CSDARange"], tables["DEDXnr"]
        if not (np.array_equal(r["energy"], d["energy"]) and
                (energy is None or np.array_equal(energy, r["energy"]))):
            raise ValueError("loss/range vectors must share one energy grid")
        energy = r["energy"]
        arrays[f"csda_range_{charge}"] = r["values"]
        arrays[f"dedx_nr_{charge}"] = d["values"]
    if {k: v for k, v in couples["mu-"].items() if k != "particle"} != {
            k: v for k, v in couples["mu+"].items() if k != "particle"}:
        raise ValueError("charges disagree on material/cuts couple")
    vectors = pt.read_vectors(directory / MSC)
    if len(vectors) != len(pt.read_couples(directory / "couple.dat")):
        raise ValueError("msc/couple table count mismatch")
    m = vectors[couples["mu+"]["index"]]
    if np.any(m["energy"] <= 0) or np.any(m["values"] <= 0):
        raise ValueError("nonpositive msc energy/cross section")
    keep = m["energy"] <= EMAX * 1.0001
    arrays.update(energy=energy, msc_energy=m["energy"][keep],
                  msc_lambda1=m["energy"][keep] ** 2 / m["values"][keep])
    audit = _audit_export(directory, files)
    if files != {name: _identity(directory / name) for name in REQUIRED_FILES}:
        raise ValueError("consumed tables changed during extraction")
    arrays["meta"] = dict(material=material, couple=couples["mu-"], couples=couples,
                          export=str(directory.resolve()), release=audit["release_observed"],
                          provenance=dict(schema_version=1, files=files, audit=audit,
                                          builder=_identity(__file__),
                                          array_sources={
                                              **{f"{array}_{charge}": f"{source}.muIoni.{charge}.asc"
                                                 for charge in CHARGES
                                                 for array, source in (("csda_range", "CSDARange"), ("dedx_nr", "DEDXnr"))},
                                              "energy": "shared CSDARange/DEDXnr nodes, both charges",
                                              "msc_energy": MSC, "msc_lambda1": MSC},
                                          msc_derivation=dict(expression="energy**2 / values", source_particle="mu+",
                                                              maximum_energy_MeV=EMAX, grid_limit_factor=1.0001,
                                                              applied_to=list(CHARGES), charge_equivalence_verified=False)))
    return _validate(arrays, material, require_audited_export)


def build(directory, out, material, *, require_audited_export=False):
    """Explicit destination/material; refuse overwrites, including shipped package data."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    arrays = _extract(directory, material, require_audited_export)
    meta = arrays.pop("meta")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, meta=json.dumps(meta), **arrays)
    return out


def load(path=DEFAULT, expected_material=None, *, require_audited_export=False):
    """Validate numerical/material invariants; qualify only recorded provenance.

    Embedded hashes are not signatures. Loading does not re-audit an unavailable
    export or certify the producing executable. Legacy assertions are not evidence.
    """
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files if k != "meta"}
        out["meta"] = json.loads(str(f["meta"])) if "meta" in f else {}
    return _validate(out, expected_material, require_audited_export)


def _summary(tables):
    meta = tables["meta"]
    return dict(material=meta["material"], couples=meta.get("couples", meta.get("couple")),
                charges=list(CHARGES), release_recorded=meta.get("release"),
                qualification=meta["qualification"], hash_coverage=_coverage(meta),
                provenance=meta.get("provenance"))


def compare(directory, existing, material, *, require_audited_export=False):
    """No writes. Exact numerical (not tolerance) comparison; enumerate every changed node."""
    fresh = _extract(directory, material, require_audited_export)
    old = load(existing)
    differences = {}
    for name in sorted((fresh.keys() | old.keys()) - {"meta"}):
        if name not in fresh or name not in old:
            differences[name] = dict(equal=False, missing_from="fresh" if name not in fresh else "existing")
            continue
        a, b = fresh[name], old[name]
        same_shape = a.shape == b.shape
        entry = dict(equal=bool(np.array_equal(a, b)), fresh_shape=list(a.shape), existing_shape=list(b.shape),
                     fresh_dtype=str(a.dtype), existing_dtype=str(b.dtype))
        if same_shape:
            indices = np.flatnonzero(a != b)
            entry.update(differing_nodes=len(indices),
                         max_abs_difference=float(np.max(np.abs(a.astype(float) - b.astype(float)))),
                         max_relative_to_existing=float(np.max(np.abs(a / b - 1))),
                         differences=[dict(index=int(i), fresh=float(a[i]), existing=float(b[i]),
                                           delta=float(a[i] - b[i])) for i in indices])
        else:
            entry.update(fresh_values=a.tolist(), existing_values=b.tolist())
        differences[name] = entry
    return dict(fresh=_summary(fresh), existing=_summary(old), existing_npz=_identity(existing),
                material_equal=fresh["meta"]["material"] == old["meta"]["material"],
                all_arrays_equal=all(v["equal"] for v in differences.values()), arrays=differences,
                scope=SCOPE + " Equality does not upgrade the existing NPZ's provenance.")


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    comparing = bool(argv and argv[0] == "compare")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tables", help="export root or tables/ directory")
    if comparing:
        ap.add_argument("existing", help="existing NPZ (read only)")
    else:
        ap.add_argument("--out", required=True, help="new NPZ path (must not exist)")
    ap.add_argument("--material", required=True, help="exact exported material name")
    ap.add_argument("--require-audited-export", action="store_true", help="require checked export records, not physics certification")
    a = ap.parse_args(argv[1:] if comparing else argv)
    if comparing:
        print(json.dumps(compare(a.tables, a.existing, a.material,
                                 require_audited_export=a.require_audited_export), indent=2))
    else:
        print(build(a.tables, a.out, a.material, require_audited_export=a.require_audited_export))


if __name__ == "__main__":
    main()
