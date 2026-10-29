"""Campaign preflight and donor-only re-dump contracts without art execution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from valhist import genassoc_campaign as C


def fixture_plan(tmp_path):
    donor_dir = tmp_path / "old"
    (donor_dir / "logs").mkdir(parents=True)
    donor = donor_dir / "crvsteps.001400_00000990.root"
    donor.write_bytes(b"original donor")
    log = donor_dir / "logs/dump.001400_00000990.log"
    log.write_text('Opened input file "/pnfs/known/dts.x.001400_00000990.art"\n'
                   'Art has completed and will exit with status 0.\n')
    gen = tmp_path / "gen.root"
    gen.write_bytes(b"complete generated population")
    meta = tmp_path / "old_library.json"
    meta.write_text(json.dumps({"inputs": [str(donor)], "gen": [str(gen)], "generated_scaled_to": 500000}))
    return meta, donor, gen, log


def test_plan_resolves_saved_paths_and_preserves_full_gen(tmp_path):
    meta, donor, gen, log = fixture_plan(tmp_path)
    out = tmp_path / "campaign"
    C.plan(meta, out, tmp_path / "area", 500000)
    _, plan = C._load_plan(out / "plan.json")
    assert plan["gen"] == [str(gen)] and plan["generated"] == 500000
    assert plan["donors"][0]["art"] == "/pnfs/known/dts.x.001400_00000990.art"
    assert Path(plan["donors"][0]["new_table"]).parent == out / "donors"
    assert Path(plan["donors"][0]["new_table"]) != donor
    assert len((out / "donors.tsv").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        C.plan(meta, out, tmp_path / "area", 500000)
    gen.write_bytes(b"changed")
    with pytest.raises(ValueError, match="denominator drift"):
        C._load_plan(out / "plan.json")


def test_plan_mutation_cannot_change_rebuild_normalization(tmp_path):
    meta, _, _, _ = fixture_plan(tmp_path)
    out = tmp_path / "campaign"
    C.plan(meta, out, tmp_path / "area", 500000)
    path = out / "plan.json"
    plan = json.loads(path.read_text())
    plan["generated"] = 1000000
    plan["generated_per_file"] = 1000000
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="source donor/exposure"):
        C._load_plan(path)


def test_plan_rejects_incomplete_log_and_wrong_exposure(tmp_path):
    meta, _, _, log = fixture_plan(tmp_path)
    out = tmp_path / "campaign"
    with pytest.raises(ValueError, match="exposure"):
        C.plan(meta, out, tmp_path / "area", 499999)
    assert not out.exists()
    log.write_text('Opened input file "/pnfs/known/dts.x.001400_00000990.art"\n')
    with pytest.raises(ValueError, match="completion"):
        C.plan(meta, out, tmp_path / "area", 500000)
    assert not out.exists()


def test_redump_requires_user_built_area_before_launch(tmp_path, monkeypatch):
    meta, _, _, _ = fixture_plan(tmp_path)
    out, area = tmp_path / "campaign", tmp_path / "area"
    C.plan(meta, out, area, 500000)
    monkeypatch.delenv("MUSE_WORK_DIR", raising=False)
    with pytest.raises(ValueError, match="MUSE_WORK_DIR"):
        C.redump(out / "plan.json")
    monkeypatch.setenv("MUSE_WORK_DIR", str(area))
    with pytest.raises(ValueError, match="built dumper"):
        C.redump(out / "plan.json")
    assert not list((out / "donors").iterdir())


def test_summary_retains_same_species_wrong_energy():
    frame = pd.DataFrame(dict(pdg_disagreement=[False, True], gen_join_missing=False,
                              negative_ke=[False, True], nonfinite=False,
                              start_minus_gen_e=[100., 200.], start_minus_gen_t=[.1, .2]))
    result = C.summarize(frame)
    assert result["showers"] == 2 and result["pdg_mismatch"] == 1
    assert result["same_pdg_energy_delta_gt_1MeV"] == 1


def test_cpp_fcl_schema_contract_and_early_validation():
    root = Path(__file__).resolve().parents[1]
    cc = (root / "art/CrvFastSim/src/CrvStepTableDumper_module.cc").read_text()
    for field in ("schema_version", "associations_checked", "full_generator_stream", "gen_product_id",
                  "sim_product_id", "parent_product_id", "sp_product_id", "n_gen", "n_particles", "n_steps"):
        assert '"' + field + '"' in cc
    analyze = cc.split("void CrvStepTableDumper::analyze", 1)[1]
    assert analyze.index("validate(event") < analyze.index("gens_->Fill()")
    assert "gp.id() != gid" in cc and "gp.key() >= gen.size()" in cc
    assert 'associationPolicy_ == "retained_primary" && p.isPrimary() && p.pdgId() != g.pdgId()' in cc
    assert 'fhicl::Name("associationPolicy"), "retained_primary"' in cc
    assert 'associationPolicy_ != "retained_primary" && associationPolicy_ != "staged_cat"' in cc
    assert 'if (associationPolicy_ == "staged_cat") schemaVersion_ = 3;' in cc
    assert 'events_->Branch("association_policy", &associationPolicy_)' in cc
    assert 'gp->pdgId() != g.pdgId() ||' in cc
    cat = (root / "art/CrvFastSim/fcl/dump_crvsteps_cat.fcl").read_text()
    assert 'associationPolicy : "staged_cat"' in cat
    fcl = (root / "art/CrvFastSim/fcl/dump_gen.fcl").read_text()
    assert 'genParticles : "cosmicTimeOffset"' in fcl
    assert "fullGeneratorStream : true" in fcl and "writeSteps : false" in fcl
    assert 'genParticles : "compressDetStepMCs"' in (root / "art/CrvFastSim/fcl/dump_crvsteps.fcl").read_text()
