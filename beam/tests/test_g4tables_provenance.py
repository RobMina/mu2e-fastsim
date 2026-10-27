"""Synthetic provenance/numerics checks; no private data or external executable."""
import json
from pathlib import Path

import numpy as np
import pytest

from beamfastsim import g4tables as gt


def _vector(path, energy, values):
    path.write_text("1\n2\n" + f"{energy[0]} {energy[-1]} {len(energy)}\n{len(energy)}\n" +
                    "\n".join(f"{x:.17g} {y:.17g}" for x, y in zip(energy, values)) + "\n")


def _bundle(tmp_path, material="StoppingTarget_Al"):
    directory = tmp_path / "export" / "tables"
    directory.mkdir(parents=True)
    (directory / "couple.dat").write_text(
        f"COUPLE-V3.0\n1\n0\n{material}\nDefaultRegionForTheWorld\n.01\n.01\n.01\n0\n")
    (directory / "material.dat").write_text(f"MATERIAL-V3.0\n1\n{material} 2.699\n")
    (directory / "cut.dat").write_text("CUT-V3.0\n1\n.01 .99\n.01 34\n.01 33\n0 0\n")
    energy = np.array([.001, .01, .1, 1., 10.])
    for charge in gt.CHARGES:
        for name in gt.pt.TABLES:
            values = energy / 2 if name in ("Range", "CSDARange") else np.full(5, 2.)
            _vector(directory / f"{name}.muIoni.{charge}.asc", energy, values)
    _vector(directory / gt.MSC, energy, energy**2 / 4)
    return directory


def _receipt(directory):
    root = directory.parent
    (root / "job.log").write_text("Art has completed and will exit with status 0.\n")
    (root / "expanded.fcl").write_text("synthetic config\n")
    receipt = dict(complete=True, exit_code=0, release="synthetic-release-not-a-claim",
                   artifacts=[gt._identity(root / n) for n in ("job.log", "expanded.fcl")],
                   tables=[gt._identity(directory / n) for n in gt.REQUIRED_FILES])
    # Relocatable receipts must validate these local bytes, not original absolute paths.
    for item in receipt["artifacts"] + receipt["tables"]:
        item["path"] = "/unavailable/original/" + Path(item["path"]).name
    (root / "receipt.json").write_text(json.dumps(receipt))
    return receipt


def _full_export(directory):
    root = directory.parent
    (root / "job.log").write_text("Geant4 version Name: synthetic-test\n"
                                 "Art has completed and will exit with status 0.\nexit=0\n")
    for name in ("expanded.fcl", "export.fcl", "job.mac"):
        (root / name).write_text("synthetic " + name)
    (root / "provenance.txt").write_text(
        "MUSE_WORK_DIR=/synthetic/release\nMUSE_STUB=test\n/bin/synthetic-mu2e\n" +
        "\n".join(f"{gt._identity(root / n)['sha256']}  {n}" for n in ("export.fcl", "job.mac")) +
        "\n" + "a" * 64 + "  /unavailable/reference/job.log\n")


def _save(path, tables):
    np.savez(path, meta=json.dumps(tables["meta"]), **{k: v for k, v in tables.items() if k != "meta"})


def test_build_explicit_and_complete_consumed_hashes(tmp_path):
    directory = _bundle(tmp_path)
    with pytest.raises(TypeError):
        gt.build(directory)
    with pytest.raises(TypeError):
        gt.build(directory, tmp_path / "out.npz")
    out = gt.build(directory.parent, tmp_path / "out.npz", "StoppingTarget_Al")
    tables = gt.load(out, expected_material="StoppingTarget_Al")
    meta = tables["meta"]
    assert meta["release"] is None
    assert meta["qualification"]["status"] == "unqualified_hashes_only"
    assert gt._coverage(meta)["complete"]
    assert set(meta["provenance"]["files"]) == set(gt.REQUIRED_FILES)
    for name in gt.REQUIRED_FILES:
        assert meta["provenance"]["files"][name] == gt._identity(directory / name)
    assert not gt._coverage(meta)["msc"]["charge_equivalence_verified"]
    np.testing.assert_allclose(tables["msc_lambda1"], 4.)
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        gt.build(directory, out, "StoppingTarget_Al")
    assert before == out.read_bytes()
    with pytest.raises(ValueError, match="audited export"):
        gt.load(out, require_audited_export=True)
    with pytest.raises(ValueError, match="material mismatch"):
        gt.load(out, expected_material="G4_Al")
    with pytest.raises(ValueError, match="unique named"):
        gt.build(directory, tmp_path / "wrong.npz", "G4_Al")


@pytest.mark.parametrize("auditor", [_receipt, _full_export])
def test_audited_bundle_records_observed_not_hardcoded_release(tmp_path, auditor):
    directory = _bundle(tmp_path)
    auditor(directory)
    out = gt.build(directory, tmp_path / "out.npz", "StoppingTarget_Al", require_audited_export=True)
    meta = gt.load(out, require_audited_export=True)["meta"]
    assert meta["qualification"]["status"] == "audited_export_record"
    assert "synthetic" in meta["release"]
    audit = meta["provenance"]["audit"]
    assert audit["export_time_table_hashes"] is (auditor is _receipt)
    assert {"job.log", "expanded.fcl"} <= audit["artifacts"].keys()
    if auditor is _full_export:
        assert audit["verified_provenance_files"] == ["export.fcl", "job.mac"]
        assert "not reverified" in audit["scope"]


@pytest.mark.parametrize("change", ["incomplete", "exit", "missing_exit", "missing_msc", "missing_muplus",
                                     "missing_metadata", "table_hash", "artifact_hash", "log", "duplicate"])
def test_receipt_fails_closed_even_without_strict_flag(tmp_path, change):
    directory = _bundle(tmp_path)
    receipt = _receipt(directory)
    if change == "incomplete":
        receipt["complete"] = False
    elif change == "exit":
        receipt["exit_code"] = 1
    elif change == "missing_exit":
        del receipt["exit_code"]
    elif change.startswith("missing_"):
        name = {"missing_msc": gt.MSC, "missing_muplus": "DEDX.muIoni.mu+.asc",
                "missing_metadata": "material.dat"}[change]
        receipt["tables"] = [r for r in receipt["tables"] if Path(r["path"]).name != name]
    elif change == "table_hash":
        (directory / "DEDX.muIoni.mu+.asc").write_text((directory / "DEDX.muIoni.mu+.asc").read_text() + "\n")
    elif change == "artifact_hash":
        (directory.parent / "expanded.fcl").write_text("changed")
    elif change == "log":
        (directory.parent / "job.log").write_text("failed\n")
    else:
        receipt["tables"].append(receipt["tables"][0])
    (directory.parent / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        gt.build(directory, tmp_path / "out.npz", "StoppingTarget_Al")
    assert not (tmp_path / "out.npz").exists()


@pytest.mark.parametrize("change", ["missing", "hash", "exit", "completion", "provenance", "macro"])
def test_full_export_bundle_checks(tmp_path, change):
    directory = _bundle(tmp_path)
    _full_export(directory)
    root = directory.parent
    if change == "missing":
        (root / "expanded.fcl").unlink()
    elif change == "hash":
        (root / "export.fcl").write_text("tampered")
    elif change == "provenance":
        (root / "provenance.txt").write_text("MUSE_WORK_DIR=/synthetic\n")
    else:
        log = (root / "job.log").read_text()
        if change == "exit":
            log = log.replace("exit=0", "exit=1")
        elif change == "completion":
            log = log.replace("Art has completed and will exit with status 0.", "crashed")
        else:
            log += "COMMAND NOT FOUND\n"
        (root / "job.log").write_text(log)
    with pytest.raises(ValueError):
        gt.build(directory, tmp_path / "out.npz", "StoppingTarget_Al")


@pytest.mark.parametrize("change", ["nan", "inf", "zero", "negative", "energy", "range", "msc_grid",
                                     "shape", "dimension", "missing", "material", "couple"])
def test_load_rejects_invalid_arrays_and_material(tmp_path, change):
    directory = _bundle(tmp_path)
    tables = gt._extract(directory, "StoppingTarget_Al")
    if change in ("nan", "inf", "zero", "negative"):
        tables["dedx_nr_mu+"][0] = {"nan": np.nan, "inf": np.inf, "zero": 0, "negative": -1}[change]
    elif change in ("energy", "range", "msc_grid"):
        name = {"energy": "energy", "range": "csda_range_mu-", "msc_grid": "msc_energy"}[change]
        tables[name][1] = tables[name][0]
    elif change == "shape":
        tables["msc_lambda1"] = tables["msc_lambda1"][:-1]
    elif change == "dimension":
        tables["energy"] = tables["energy"].reshape(1, -1)
    elif change == "missing":
        del tables["dedx_nr_mu+"]
    elif change == "material":
        tables["meta"]["material"] = ""
    else:
        tables["meta"]["couple"]["material"] = "wrong"
    path = tmp_path / "bad.npz"
    _save(path, tables)
    with pytest.raises(ValueError):
        gt.load(path)


def test_legacy_comparison_exact_nodes_and_no_retroactive_upgrade(tmp_path, capsys):
    directory = _bundle(tmp_path)
    tables = gt._extract(directory, "StoppingTarget_Al")
    tables["meta"] = dict(material="G4_Al", release="legacy unsupported claim",
                          qualification=dict(status="audited_export_record", audited_export=True))
    tables["dedx_nr_mu+"][2] += .125
    old = tmp_path / "legacy.npz"
    _save(old, tables)
    before = old.read_bytes()
    _full_export(directory)
    gt.main(["compare", str(directory.parent), str(old), "--material", "StoppingTarget_Al", "--require-audited-export"])
    report = json.loads(capsys.readouterr().out)
    assert not report["material_equal"] and not report["all_arrays_equal"]
    assert report["fresh"]["hash_coverage"]["complete"]
    assert not report["existing"]["hash_coverage"]["complete"]
    assert report["existing"]["qualification"]["status"] == "unqualified_legacy_or_incomplete_provenance"
    assert report["existing"]["charges"] == ["mu-", "mu+"]
    diff = report["arrays"]["dedx_nr_mu+"]
    assert diff["differing_nodes"] == 1
    assert diff["differences"] == [dict(index=2, fresh=2., existing=2.125, delta=-.125)]
    assert report["arrays"]["csda_range_mu-"]["equal"]
    assert old.read_bytes() == before
    with pytest.raises(ValueError, match="unqualified"):
        gt.load(old, require_audited_export=True)


def test_missing_provenance_strict_build_and_cli_material_required(tmp_path):
    directory = _bundle(tmp_path)
    with pytest.raises(ValueError, match="audited export"):
        gt.build(directory, tmp_path / "out.npz", "StoppingTarget_Al", require_audited_export=True)
    assert not (tmp_path / "out.npz").exists()
    with pytest.raises(SystemExit):
        gt.main([str(directory), "--out", str(tmp_path / "out.npz")])


def test_compare_equal_legacy_does_not_certify_it(tmp_path):
    directory = _bundle(tmp_path)
    tables = gt._extract(directory, "StoppingTarget_Al")
    tables["meta"] = dict(material="StoppingTarget_Al", files=["old names but no hashes"])
    old = tmp_path / "old.npz"
    _save(old, tables)
    _receipt(directory)
    report = gt.compare(directory, old, "StoppingTarget_Al", require_audited_export=True)
    assert report["all_arrays_equal"] and report["material_equal"]
    assert report["fresh"]["qualification"]["audited_export"]
    assert not report["existing"]["qualification"]["audited_export"]


@pytest.mark.parametrize("change", ["charge", "msc", "metadata", "audit"])
def test_load_strict_requires_complete_hash_coverage(tmp_path, change):
    directory = _bundle(tmp_path)
    _receipt(directory)
    tables = gt._extract(directory, "StoppingTarget_Al")
    provenance = tables["meta"]["provenance"]
    if change == "audit":
        del provenance["audit"]["artifacts"]["expanded.fcl"]
    else:
        name = {"charge": "DEDXnr.muIoni.mu+.asc", "msc": gt.MSC, "metadata": "cut.dat"}[change]
        del provenance["files"][name]
    old = tmp_path / "incomplete.npz"
    _save(old, tables)
    assert not gt.load(old)["meta"]["qualification"]["audited_export"]
    with pytest.raises(ValueError, match="unqualified"):
        gt.load(old, require_audited_export=True)


def test_compare_shape_difference_reports_both_arrays(tmp_path):
    directory = _bundle(tmp_path)
    tables = gt._extract(directory, "StoppingTarget_Al")
    for name in gt.ARRAYS:
        tables[name] = tables[name][:-1]
    old = tmp_path / "coarse.npz"
    _save(old, tables)
    report = gt.compare(directory, old, "StoppingTarget_Al")
    entry = report["arrays"]["energy"]
    assert not entry["equal"]
    assert len(entry["fresh_values"]) == 5 and len(entry["existing_values"]) == 4


@pytest.mark.parametrize("name", ["CSDARange.muIoni.mu+", "DEDXnr.muIoni.mu+", "LambdaMod0.msc.mu+"])
def test_build_rejects_invalid_source_vectors(tmp_path, name):
    directory = _bundle(tmp_path)
    energy = np.array([.001, .01, .1, 1., 10.])
    _vector(directory / (name + ".asc"), energy, np.array([1., 2., np.nan, 4., 5.]))
    with pytest.raises(ValueError, match="nonfinite"):
        gt.build(directory, tmp_path / "bad.npz", "StoppingTarget_Al")
