"""Text guards for an uncompiled proposal; not native lifecycle tests."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "native"


def source():
    return (ROOT / "TargetRecording.cc").read_text()


def test_qualified_header_and_no_physics_mutation():
    code = source()
    assert code.startswith('#include "Offline/Mu2eG4/inc/TargetRecording.hh"')
    for forbidden in ("SetTrackStatus(", "SetWeight(", "LocateGlobalPoint", "G4UniformRand(", "new G4Navigator"):
        assert forbidden not in code
    assert '"library_eligible", "false"' in code


def test_unsupported_continuation_before_cut_and_terminal_end_only():
    code = source()
    step = code.split("void step(G4Step const& s)", 1)[1].split("void cuts(", 1)[0]
    assert "status != fSuspend && status != fSuspendAndWait && status != fPostponeToNextEvent" in step
    end = code.split("void endTrack(G4Track const& t)", 1)[1].split("void end(", 1)[0]
    assert "status == fStopAndKill || status == fKillTrackAndSecondaries" in end
    assert "resumed or revisited tracking is unsupported" in code
    assert '"tracking without a preceding stacking callback"' in code
    assert '"duplicate stacking or track identity"' in code
    assert '"unsupported final track lifecycle"' in code


def test_birth_mapping_consumes_pointer_and_no_birthtime_art_identity():
    code = source()
    association = code.split("void associate(", 1)[1].split("std::string causal(", 1)[0]
    assert "pendingBirths.erase(it)" in association
    assert "birth.parent == t.GetParentID()" in association
    assert "birth.mappedID = t.GetTrackID()" in association
    birth = code.split("for (auto const* child : *secondaries)", 1)[1].split("void cuts(", 1)[0]
    assert "expectedIdentity(*child)" not in birth
    assert "GetSecondaryInCurrentStep" in code
    assert '"unconsumed secondary birth tokens prevent callback completeness"' in code


def test_caps_exclusive_creation_and_completion_after_flush():
    code = source()
    assert "fs::create_directory(eventDirectory)" in code
    assert "rows < maxRows" in code
    assert "line.size() <= maxBytes - bytes" in code
    assert "volumeCount < maxVolumes" in code
    end = code.split("void end(int status", 1)[1]
    assert end.index("ledger.flush()") < end.index("ledger.close()") < end.index("fs::rename(")
    assert end.index("marker.flush()") < end.index("marker.close()") < end.index("fs::rename(")


def test_endpoints_and_inventory_are_native_observations():
    code = source()
    for field in ("p.GetGlobalTime()", "p.GetProperTime()", "p.GetWeight()", "p.GetMomentum()",
                  "p.GetStepStatus()", "p.GetProcessDefinedStep()", "touch.GetHistoryDepth()",
                  "touch.GetCopyNumber(d)", "pv->GetObjectRotationValue()", "pv->GetObjectTranslation()"):
        assert field in code
    assert "!pv->IsReplicated() && !pv->IsParameterised()" in code
    assert "std::size_t const daughters = lv->GetNoDaughters()" in code
    assert "for (std::size_t i = 0; i != daughters; ++i)" in code


def test_native_abort_guard_and_completion_after_sd_handoff():
    code = source()
    end = code.split("void end(int status", 1)[1]
    assert end.index('require(!eventAborted,') < end.index('writeRow("event_end"')
    assert code.count('.add("event_aborted_native", boolean(eventAborted))') == 2
    patch = (ROOT / "target_recording.patch").read_text()
    hook = patch.split("+++ b/Mu2eG4/src/Mu2eG4EventAction.cc", 1)[1].split("--- a/", 1)[0]
    assert "status, evt->IsAborted()," in hook
    assert hook.index("insertSDDataIntoPerThreadStorage") < hook.index("targetRecording->endEvent(")


def test_kill_with_pending_children_refuses_without_fabricating_disposition():
    code = source()
    end = code.split("void endTrack(G4Track const& t)", 1)[1].split("void end(", 1)[0]
    assert "if (status == fKillTrackAndSecondaries)" in end
    assert "birth.second.parent != t.GetTrackID() || birth.second.mappedID != 0" in end
    assert "terminal child disposition unsupported" in end
    assert "pendingBirths.erase" not in end
    assert "birth.mappedID =" not in end


def test_source_preparation_is_bound_before_native_tracking():
    code = source()
    prepare = code.split("void prepare(art::Event const&", 1)[1].split("void begin(unsigned", 1)[0]
    for text in ("artEvent.getValidHandle<GenParticleCollection>(tag)", "source.provenance()",
                 'writeRow("source_begin"', 'writeRow("generated_primary"', 'writeRow("source_end"',
                 "artEvent.processHistoryID()", "provenance->psetIDs()", "g.pdgId() == PDGCode::proton"):
        assert text in prepare
    assert "state = EventState::prepared" in prepare
    patch = (ROOT / "target_recording.patch").read_text()
    hook = patch.split("+++ b/Mu2eG4/src/Mu2eG4_module.cc", 1)[1].split("--- a/", 1)[0]
    assert hook.index("prepareEvent(event,") < hook.index("BeamOnDoOneArtEvent(")
    assert "ionToGenerate.hasValue()" in patch


def test_primary_link_is_actual_input_pointer_and_insert_is_stored_object():
    code = source()
    stack = code.split("void stacking(G4Track const&", 1)[1].split("template<class T>", 1)[0]
    assert "primaryHelper->getEntry(t.GetTrackID())" in stack
    assert "p->get() == g->second.address" in stack
    assert 'writeRow("primary_link"' in stack
    patch = (ROOT / "target_recording.patch").read_text()
    assert "simParticleOmitted(*trk)" in patch
    assert "simParticleInserted(*trk, _transientMap.at(kid))" in patch
    reconcile = code.split("void reconcile(SimParticleCollection const&", 1)[1].split("void beginTrack(", 1)[0]
    assert "daughters == expectedDaughters[key]" in reconcile
    assert 'writeRow("source_reconciliation"' in reconcile
    assert "p.parent()->" not in reconcile
    assert "p.genParticle()->" not in reconcile


def test_direct_source_dependencies_are_explicit_and_scons_bound():
    patch = (ROOT / "target_recording.patch").read_text()
    cmake = patch.split("+++ b/Mu2eG4/CMakeLists.txt", 1)[1].split("--- a/", 1)[0]
    for library in ("art::Framework_Principal", "canvas::canvas", "fhiclcpp::fhiclcpp"):
        assert "+      " + library in cmake
    code = source()
    assert '#include "canvas/Persistency/Provenance/ProcessHistoryID.h"' in code
    assert '#include "fhiclcpp/ParameterSetID.h"' in code
    assert 'tag.label() == provenance->moduleLabel()' in code
