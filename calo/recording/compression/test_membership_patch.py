"""Source-data checks only: no C++ compilation, art execution, or simulation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import tempfile

import unittest

HERE = Path(__file__).resolve().parent
REL = "CaloMC/src/CaloShowerStepMaker_module.cc"
spec = importlib.util.spec_from_file_location("membership_applicability", HERE / "check_applicability.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class MembershipPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix=".test-copy-", dir=HERE)
        cls.addClassCleanup(cls.temporary.cleanup)
        path = checker.apply_copy(HERE / "snapshot", Path(cls.temporary.name))
        cls.patched = path.read_text()

    def test_snapshot_hashes(self):
        checker.check_hashes(HERE / "snapshot")


    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".pytest-mismatch-", dir=HERE) as temporary:
            root = Path(temporary)
            shutil.copytree(HERE / "snapshot", root, dirs_exist_ok=True)
            target = root / REL
            target.write_text(target.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
                checker.check_hashes(root)


    def test_patch_changes_only_producer_and_no_stock_statement(self):
        patch = (HERE / "membership.patch").read_text().splitlines()
        assert [line for line in patch if line.startswith("--- ")] == ["--- a/" + REL]
        removed = [line[1:] for line in patch if line.startswith("-") and not line.startswith("---")]
        assert removed == ["              if (eDep < eDepThreshold_) continue;"]
        added = [line[1:] for line in patch if line.startswith("+") and not line.startswith("+++")]
        assert "              if (eDep < eDepThreshold_)" in added
        assert "                  continue;" in added
        assert not any("std::sort(" in line or "stable_sort" in line or "buffer.add(" in line
                       or "emplace_back(volId" in line or "produces<" in line
                       or "simsToKeepUnique.insert" in line for line in added)
        assert not any("random" in line.lower() or "TFileService" in line or "Assns<" in line for line in added)


    def test_default_off_and_config_storage_types(self):
        patched = self.patched
        for declaration in ("fhicl::Atom<unsigned>        numZSlices", "fhicl::Atom<float>           deltaTime",
                            "fhicl::Atom<bool>            compressData", "fhicl::Atom<double>          eDepThreshold"):
            assert declaration in patched
        assert 'Comment("Optional exclusive-create text membership ledger; empty disables"), "" }' in patched
        assert 'if (!config().membershipLedger().empty())' in patched
        assert 'std::fopen(config().membershipLedger().c_str(), "wx")' in patched
        for field in ("numZSlices unsigned", "deltaTime float", "compressData bool", "eDepThreshold double"):
            assert field in patched
        assert '"D zSliceSize double " << std::hexfloat << zSliceSize_' in patched


    def test_actual_buffer_order_and_emission_hooks(self):
        patched = self.patched
        body = patched.split("  void CaloShowerStepMaker::compressSteps(", 1)[1].split("  void CaloShowerStepMaker::ledgerBegin", 1)[0]
        assert body.count("caloShowerStepMCs.emplace_back(") == 2
        assert body.count("ledgerEmit(caloShowerStepMCs.size()-1") == 2
        add = body.index("buffer.add(")
        shadow = body.index("ledgerMembers[idx].push_back(step)")
        assert add < shadow
        assert body[add:shadow].count(";") == 1
        first_emplace = body.index("caloShowerStepMCs.emplace_back(")
        emit = body.index("ledgerEmit(")
        clear = body.index("ledgerMembers.at(idx).clear()")
        generation = body.index("++ledgerGeneration[idx]")
        reset = body.index("buffer.reset(idx)")
        assert first_emplace < emit < clear < generation < reset < add
        assert '"time_gap", ledgerMembers.at(idx), buffer.entries(idx)' in body
        assert '"final_bucket", ledgerMembers.at(i), buffer.entries(i)' in body
        assert "std::map<unsigned, std::size_t> ledgerGeneration;" in body


    def test_threshold_is_group_level_before_noncompress_subdivision(self):
        patched = self.patched
        body = patched.split("  void CaloShowerStepMaker::makeCompressedHits(", 1)[1]
        assert body.index("for (const auto* step : steps) eDep += step->totalEDep();") < body.index("ledgerReject(steps)")
        assert body.index("ledgerReject(steps)") < body.index("if (compressData_)") < body.index("newSimStepMap;")
        assert body.count("compressSteps(cal, caloShowerStepMCs") == 2
        assert "below_ancestor_crystal_threshold" in patched
        assert '"G " << ledgerGroup_' in patched
        assert '"S " << output << \' \' << ledgerGroup_' in patched


    def test_multi_collection_empty_collection_and_address_ambiguity(self):
        patched = self.patched
        body = patched.split("  void CaloShowerStepMaker::ledgerBegin(", 1)[1].split("  void CaloShowerStepMaker::ledgerReject", 1)[0]
        assert body.index("products.insert(product)") < body.index("index < handle->size()")
        assert "Duplicate resolved input ProductID, including empty collections" in body
        assert "ledgerRaw_.emplace(&handle->at(index), LedgerRaw{slot, index, product, 0})" in body
        assert "Ambiguous raw source address" in body
        assert '"C " << slot' in body
        assert "ledgerTags_.at(slot)" in body
        assert "handles.size()" in body
        assert "event.getProductID<CaloShowerStepCollection>().value()" in body


    def test_event_completion_and_exhaustive_dispositions(self):
        patched = self.patched
        produce = patched.split("  void CaloShowerStepMaker::produce(", 1)[1].split("  void CaloShowerStepMaker::makeCompressedHits", 1)[0]
        assert produce.index("ledgerBegin(") < produce.index("makeCompressedHits(") < produce.index("ledgerFinish(")
        assert produce.index("event.put(std::move(simsToKeep))") < produce.index("ledgerWrite(ledgerBuffer_.text())")
        assert "ledgerRaw_.clear();" in patched
        assert "ledgerGroup_ = ledgerAccepted_ = ledgerRejected_ = 0;" in patched
        assert "Raw row lacks disposition" in patched
        assert patched.count("Repeated raw disposition") == 2
        assert "Shadow membership differs from stock buffer entries" in patched
        assert '"Z " << ledgerRaw_.size()' in patched
        assert "std::fflush(ledgerFile_.get()) != 0" in patched


    def test_positive_typed_caps_and_header_accounting(self):
        patched = self.patched
        assert 'fhicl::Atom<std::size_t>     membershipMaxRawRowsPerEvent' in patched
        assert 'fhicl::Atom<std::size_t>     membershipMaxBytesPerJob' in patched
        assert 'Comment("Positive raw-row cap when ledger enabled"), 2000000 }' in patched
        assert 'Comment("Positive total ledger byte cap including header"), 268435456 }' in patched
        constructor = patched.split('if (!config().membershipLedger().empty())', 1)[1].split('void CaloShowerStepMaker::produce', 1)[0]
        assert constructor.index('if (!ledgerMaxRawRows_ || !ledgerMaxBytes_)') < constructor.index('std::fopen(')
        assert constructor.index('ledgerBuffer_.reset(ledgerMaxBytes_)') < constructor.index('"H calo-membership 1')
        assert 'P membershipMaxRawRowsPerEvent size_t ' in constructor
        assert 'P membershipMaxBytesPerJob size_t ' in constructor
        assert 'ledgerWrite(ledgerBuffer_.text())' in constructor

    def test_raw_total_preflight_and_sparse_shadow_storage(self):
        body = self.patched.split('void CaloShowerStepMaker::ledgerBegin(', 1)[1].split('void CaloShowerStepMaker::ledgerReject', 1)[0]
        check = body.index('if (handle->size() > ledgerMaxRawRows_ - rawRows)')
        assert body.index('for (const auto& handle : handles)') < check < body.index('rawRows += handle->size()')
        assert check < body.index('ledgerRaw_.clear()') < body.index('ledgerRaw_.emplace(')
        assert 'Membership raw rows per event cap exceeded' in body
        assert 'std::map<unsigned, std::vector<const StepPointMC*>> ledgerMembers;' in self.patched
        assert 'ledgerFile_ ? numZSlices_' not in self.patched

    def test_bounded_text_every_insertion_and_prewrite_job_accounting(self):
        patched = self.patched
        assert 'ostringstream' not in patched
        assert 'std::ostream ledgerText_{&ledgerBuffer_}' in patched
        assert 'ledgerText_.exceptions(std::ios::badbit | std::ios::failbit)' in patched
        buffer = patched.split('class MembershipBuffer', 1)[1].split('// Accumulates', 1)[0]
        assert buffer.index('count < 0 || static_cast<std::size_t>(count) > limit_ - text_.size()') < buffer.index('text_.append(')
        assert 'xsputn(&value, 1)' in buffer
        assert 'const std::string& text() const' in buffer
        assert 'void reset(std::size_t limit) { text_.clear(); limit_ = limit; }' in buffer
        assert 'ledgerBuffer_.reset(ledgerMaxBytes_ - ledgerBytesWritten_)' in patched
        write = patched.split('void CaloShowerStepMaker::ledgerWrite(', 1)[1]
        assert write.index('if (text.size() > ledgerMaxBytes_ - ledgerBytesWritten_)') < write.index('ledgerBytesWritten_ += text.size()') < write.index('std::fwrite(')
        assert patched.count('std::fwrite(') == 1
        assert patched.count('ledgerWrite(ledgerBuffer_.text())') == 2
        # Strict >, not >=, deliberately permits exact-limit and empty writes.
        assert 'text.size() >= ledgerMaxBytes_' not in write
        assert 'handle->size() >= ledgerMaxRawRows_' not in patched

    def test_review_artifact_hashes(self):
        manifest = json.loads((HERE / "source_manifest.json").read_text())
        for relative, expected in manifest["proposal_artifacts"].items():
            assert hashlib.sha256((HERE / relative).read_bytes()).hexdigest() == expected
        assert hashlib.sha256(self.patched.encode()).hexdigest() == manifest["patched_producer_sha256"]

    def test_context_build_files_unchanged(self):
        patched = self.patched
        manifest = json.loads((HERE / "source_manifest.json").read_text())
        for relative in ("CaloMC/CMakeLists.txt", "CaloMC/src/SConscript", "CaloMC/inc/ShowerStepUtil.hh", "CaloMC/src/ShowerStepUtil.cc"):
            assert hashlib.sha256((HERE / "snapshot" / relative).read_bytes()).hexdigest() == manifest["files"][relative]
        assert patched.count("produces<CaloShowerStepCollection>();") == 1
        assert patched.count("produces<SimParticlePtrCollection>();") == 1


    def test_patch_added_comments_are_one_line(self):
        for line in (HERE / "membership.patch").read_text().splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                assert "/*" not in line and "*/" not in line


if __name__ == "__main__":
    unittest.main()
