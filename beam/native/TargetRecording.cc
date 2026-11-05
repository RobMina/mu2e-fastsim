#include "Offline/Mu2eG4/inc/TargetRecording.hh"

#include "Offline/Mu2eG4/inc/SimParticleHelper.hh"
#include "Offline/Mu2eG4/inc/SimParticlePrimaryHelper.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Provenance.h"
#include "canvas/Utilities/InputTag.h"
#include "canvas/Persistency/Provenance/ProcessHistoryID.h"
#include "fhiclcpp/ParameterSetID.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "Geant4/G4ClassificationOfNewTrack.hh"
#include "Geant4/G4DynamicParticle.hh"
#include "Geant4/G4LogicalVolume.hh"
#include "Geant4/G4Material.hh"
#include "Geant4/G4Navigator.hh"
#include "Geant4/G4PVPlacement.hh"
#include "Geant4/G4ParticleDefinition.hh"
#include "Geant4/G4ReflectedSolid.hh"
#include "Geant4/G4Step.hh"
#include "Geant4/G4StepPoint.hh"
#include "Geant4/G4StepStatus.hh"
#include "Geant4/G4SystemOfUnits.hh"
#include "Geant4/G4Track.hh"
#include "Geant4/G4TrackStatus.hh"
#include "Geant4/G4TransportationManager.hh"
#include "Geant4/G4VPhysicalVolume.hh"
#include "Geant4/G4VProcess.hh"
#include "Geant4/G4VSolid.hh"
#include "Geant4/G4VTouchable.hh"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

// Uncompiled diagnostic proposal; completion and native observations never establish library eligibility.
// Cross-event postponed continuations with prior steps fail closed; no cross-event token carry is implemented.
// Bool cut inputs follow native short-circuit order; a skipped common cut is explicitly not evaluated.
// Caps count all event-file rows and bytes, including inventory and completion, but not filesystem metadata.
// A failed or abandoned event keeps its directory without completion; destructor never completes an event.
// No generator, art product, original-primary pointer, surface normal, or launch ownership is fabricated.
namespace mu2e {
namespace {
using U64 = std::uint64_t;
namespace fs = std::filesystem;

std::string quote(std::string const& s) {
  std::ostringstream o;
  o.imbue(std::locale::classic());
  o << '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\b': o << "\\b"; break;
      case '\f': o << "\\f"; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if (c < 0x20 || c >= 0x7f) {
          o << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<unsigned>(c) << std::dec;
        } else {
          o << static_cast<char>(c);
        }
    }
  }
  o << '"';
  return o.str();
}

std::string number(double x) {
  if (!std::isfinite(x)) throw std::runtime_error("TargetRecording: nonfinite native value");
  std::ostringstream o;
  o.imbue(std::locale::classic());
  o << std::setprecision(std::numeric_limits<double>::max_digits10) << x;
  return o.str();
}

std::string boolean(bool b) { return b ? "true" : "false"; }
std::string integer(U64 n) { return std::to_string(n); }
std::string signedInteger(int n) { return std::to_string(n); }

struct Object {
  std::string text = "{";
  bool first = true;
  Object& add(std::string const& name, std::string const& json) {
    if (!first) text += ',';
    first = false;
    text += quote(name) + ':' + json;
    return *this;
  }
  std::string str() const { return text + '}'; }
};

std::string vector(CLHEP::Hep3Vector const& v, double unit = 1.) {
  return '[' + number(v.x()/unit) + ',' + number(v.y()/unit) + ',' + number(v.z()/unit) + ']';
}

std::string process(G4VProcess const* p) {
  if (!p) return "null";
  return Object().add("name", quote(p->GetProcessName()))
    .add("type", signedInteger(static_cast<int>(p->GetProcessType())))
    .add("subtype", signedInteger(p->GetProcessSubType())).str();
}

std::string counterObject(std::map<std::string, U64> const& values) {
  Object o;
  for (auto const& item : values) o.add(item.first, integer(item.second));
  return o.str();
}

void increment(U64& n) {
  if (n == std::numeric_limits<U64>::max())
    throw std::runtime_error("TargetRecording: counter overflow");
  ++n;
}
}

class TargetRecording::Impl {
public:
  enum class EventState { idle, prepared, active, complete, failed };
  enum class Life { stacked, active, ended, stackingKilled };

  struct PathPart {
    G4VPhysicalVolume const* volume;
    int copy;
  };
  using Path = std::vector<PathPart>;
  struct PathLess {
    bool operator()(Path const& a, Path const& b) const {
      std::size_t n = a.size() < b.size() ? a.size() : b.size();
      std::less<G4VPhysicalVolume const*> less;
      for (std::size_t i = 0; i != n; ++i) {
        if (less(a[i].volume, b[i].volume)) return true;
        if (less(b[i].volume, a[i].volume)) return false;
        if (a[i].copy != b[i].copy) return a[i].copy < b[i].copy;
      }
      return a.size() < b.size();
    }
  };
  struct Placement {
    U64 id;
    bool target;
    std::string history;
  };
  struct Endpoint {
    U64 id = 0;
    bool target = false;
    bool worldNull = false;
    std::string json;
  };
  struct Birth {
    U64 token;
    int parent;
    int parentStep;
    int pdg;
    int rawChildID;
    int mappedID = 0;
  };
  struct TrackInfo {
    int parent;
    int pdg;
    U64 token = 0;
    Life life = Life::stacked;
    U64 begins = 0;
    U64 ends = 0;
    U64 stackings = 0;
    int lastStep = -1;
    int endStatus = -1;
    bool pendingCuts = false;
    bool insertionObserved = false;
    bool insertionOmitted = false;
  };
  struct Generated {
    int pdg;
    GenParticle const* address;
  };
  struct PrimaryLink {
    U64 generatorKey;
    int classification;
  };
  struct Inserted {
    int trackID;
    std::string identity;
  };

  fs::path outputDirectory;
  fs::path eventDirectory;
  std::set<std::string> selectors;
  std::string materialName;
  U64 maxRows;
  U64 maxVolumes;
  U64 maxBytes;
  CLHEP::Hep3Vector origin;
  std::thread::id owner = std::this_thread::get_id();
  EventState state = EventState::idle;
  SimParticleHelper const* helper = nullptr;
  SimParticlePrimaryHelper const* primaryHelper = nullptr;
  U64 generatorProduct = 0;
  U64 generatorProtons = 0;
  U64 omittedParticles = 0;
  U64 finalParticles = 0;
  std::map<U64, Generated> generated;
  std::map<int, PrimaryLink> primaryLinks;
  std::set<U64> linkedGeneratorKeys;
  std::map<U64, Inserted> inserted;
  std::map<int, U64> insertedTracks;
  std::ofstream ledger;
  unsigned run = 0;
  unsigned subrun = 0;
  U64 event = 0;
  U64 rows = 0;
  U64 attemptedRows = 0;
  U64 bytes = 0;
  U64 volumeCount = 0;
  U64 birthCount = 0;
  U64 associationCount = 0;
  U64 selectedSteps = 0;
  int activeTrack = 0;
  std::string context = "constructor";
  std::map<std::string, U64> kinds;
  std::map<std::string, U64> hooks;
  std::map<Path, Placement, PathLess> placements;
  std::unordered_map<G4Track const*, U64> pendingBirths;
  std::map<U64, Birth> births;
  std::map<int, TrackInfo> tracks;

  Impl(std::string out, std::vector<std::string> names, std::string material,
       U64 rowCap, U64 volumeCap, U64 byteCap, CLHEP::Hep3Vector const& mu2eOrigin)
    : outputDirectory(std::move(out)), materialName(std::move(material)),
      maxRows(rowCap), maxVolumes(volumeCap), maxBytes(byteCap), origin(mu2eOrigin) {
    require(!outputDirectory.empty() && fs::is_directory(outputDirectory),
            "outputDirectory must already be a directory");
    require(maxRows && maxVolumes && maxBytes, "all caps must be positive");
    require(!materialName.empty() && !names.empty(), "empty target configuration");
    for (auto const& name : names)
      require(!name.empty() && selectors.insert(name).second, "empty or duplicate selector");
    vector(origin, mm);
  }

  [[noreturn]] void fail(std::string const& why) const {
    throw std::runtime_error("TargetRecording: " + context + ": " + why
                             + "; event_directory=" + eventDirectory.string());
  }
  void require(bool ok, std::string const& why) const { if (!ok) fail(why); }

  template<class F> void guarded(char const* hook, F&& f) {
    try {
      require(std::this_thread::get_id() == owner, "cross-thread callback");
      context = hook;
      require(state != EventState::failed, "recorder is permanently failed");
      f();
    } catch (...) {
      state = EventState::failed;
      helper = nullptr;
      if (ledger.is_open()) ledger.flush();
      throw;
    }
  }

  void observed(std::string const& hook) {
    require(state == EventState::active, "hook outside begun event");
    increment(hooks[hook]);
  }
  void trackContext(char const* hook, G4Track const& t) {
    context = std::string(hook) + " track=" + signedInteger(t.GetTrackID())
      + " step=" + signedInteger(t.GetCurrentStepNumber());
  }

  std::string position(CLHEP::Hep3Vector const& p) const {
    return Object().add("g4_world_mm", vector(p, mm))
      .add("mu2e_mm", vector(p - origin, mm)).str();
  }

  std::string envelope(std::string const& kind, std::string const& data) const {
    return Object().add("schema", quote("target-recording-proposal-v2"))
      .add("kind", quote(kind)).add("row", integer(rows + 1))
      .add("run", std::to_string(run)).add("subrun", std::to_string(subrun))
      .add("event", integer(event)).add("library_eligible", "false")
      .add("proposal_only", "true")
      .add("geometry_file", quote("records.jsonl"))
      .add("geometry_hash", "null").add("hash_binding", quote("requires_external_postread_file_hash"))
      .add("data", data).str() + '\n';
  }

  void writeRow(std::string const& kind, std::string const& data, std::ostream& stream) {
    increment(attemptedRows);
    require(rows < maxRows, "maxRows exceeded; first_offending_kind=" + kind);
    auto const line = envelope(kind, data);
    require(line.size() <= maxBytes - bytes, "maxBytes exceeded; first_offending_kind=" + kind);
    require(line.size() <= static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()),
            "row exceeds streamsize");
    stream.write(line.data(), static_cast<std::streamsize>(line.size()));
    require(static_cast<bool>(stream), "write failed; first_offending_kind=" + kind);
    bytes += static_cast<U64>(line.size());
    increment(rows);
    increment(kinds[kind]);
  }
  void writeRow(std::string const& kind, std::string const& data) {
    writeRow(kind, data, ledger);
  }

  void inventory() {
    auto* manager = G4TransportationManager::GetTransportationManager();
    require(manager != nullptr, "missing transportation manager");
    auto* navigator = manager->GetNavigatorForTracking();
    require(navigator != nullptr, "missing tracking navigator");
    auto const* world = navigator->GetWorldVolume();
    require(world != nullptr, "missing world volume");
    std::set<std::string> found;
    std::set<G4LogicalVolume const*> ancestors;
    Path path;
    std::vector<std::string> history;
    std::function<void(G4VPhysicalVolume const*, U64, std::size_t, bool)> visit;
    visit = [&](G4VPhysicalVolume const* pv, U64 parent, std::size_t daughter, bool root) {
      require(pv != nullptr, "null inventory placement");
      require(!pv->IsReplicated() && !pv->IsParameterised(), "replica/parameterisation unsupported");
      require(root || pv->GetMotherLogical() != nullptr, "non-root placement has no mother logical volume");
      require(dynamic_cast<G4PVPlacement const*>(pv) != nullptr, "non-placement volume unsupported");
      auto const* lv = pv->GetLogicalVolume();
      require(lv && lv->GetSolid() && lv->GetMaterial(), "missing logical/solid/material");
      auto const* solid = lv->GetSolid();
      require(dynamic_cast<G4ReflectedSolid const*>(solid) == nullptr,
              "reflected placement unsupported");
      require(ancestors.insert(lv).second, "cyclic logical-volume hierarchy");
      require(volumeCount < maxVolumes, "maxVolumes exceeded at " + std::string(pv->GetName()));
      increment(volumeCount);
      U64 const id = volumeCount;
      bool const target = selectors.count(pv->GetName()) != 0;
      if (target) {
        require(lv->GetMaterial()->GetName() == materialName,
                "selected physical name has unexpected material: " + std::string(pv->GetName()));
        found.insert(pv->GetName());
      }
      auto const rotation = pv->GetObjectRotationValue();
      double const det = rotation.xx()*(rotation.yy()*rotation.zz()-rotation.yz()*rotation.zy())
        - rotation.xy()*(rotation.yx()*rotation.zz()-rotation.yz()*rotation.zx())
        + rotation.xz()*(rotation.yx()*rotation.zy()-rotation.yy()*rotation.zx());
      require(std::isfinite(det) && det > 0., "reflected/invalid placement transform");
      std::string const matrix = '[' + number(rotation.xx()) + ',' + number(rotation.xy()) + ','
        + number(rotation.xz()) + ',' + number(rotation.yx()) + ',' + number(rotation.yy()) + ','
        + number(rotation.yz()) + ',' + number(rotation.zx()) + ',' + number(rotation.zy()) + ','
        + number(rotation.zz()) + ']';
      path.push_back({pv, pv->GetCopyNo()});
      history.push_back(Object().add("instance_id", integer(id))
        .add("daughter_index", root ? "null" : std::to_string(daughter))
        .add("copy_no", signedInteger(pv->GetCopyNo()))
        .add("physical_name", quote(pv->GetName())).add("logical_name", quote(lv->GetName())).str());
      std::string fullHistory = "[";
      for (auto const& part : history) {
        if (fullHistory.size() != 1) fullHistory += ',';
        fullHistory += part;
      }
      fullHistory += ']';
      require(placements.emplace(path, Placement{id, target, fullHistory}).second,
              "duplicate full placement/copy path");
      writeRow("volume", Object().add("instance_id", integer(id))
        .add("parent_id", parent ? integer(parent) : "null")
        .add("physical_name", quote(pv->GetName())).add("logical_name", quote(lv->GetName()))
        .add("solid_name", quote(solid->GetName())).add("solid_type", quote(solid->GetEntityType()))
        .add("material", quote(lv->GetMaterial()->GetName())).add("target_member", boolean(target))
        .add("copy_history_root_to_leaf", fullHistory)
        .add("local_transform_convention", quote("object_to_mother; row_major_rotation; translation_mm"))
        .add("local_rotation", matrix).add("local_translation_mm", vector(pv->GetObjectTranslation(), mm))
        .add("rotation_determinant", number(det)).str());
      // GetNoDaughters returns std::size_t in the intended API.
      std::size_t const daughters = lv->GetNoDaughters();
      for (std::size_t i = 0; i != daughters; ++i) {
        auto const* child = lv->GetDaughter(i);
        require(child && child->GetMotherLogical() == lv, "daughter/mother hierarchy mismatch");
        visit(child, id, i, false);
      }
      history.pop_back();
      path.pop_back();
      ancestors.erase(lv);
    };
    visit(world, 0, 0, true);
    require(found == selectors, "one or more exact target selectors are absent");
    writeRow("inventory_end", Object().add("volume_count", integer(volumeCount))
      .add("membership", quote("exact_physical_name_and_material; no_ancestor_membership"))
      .add("hash_policy", quote("postread_hash_actual_file_bytes; no_C++_SHA_claim"))
      .add("identity_scope", quote("event_file_local_root_to_leaf_placement_and_copy_history")).str());
  }

  Placement const& resolve(G4VTouchable const& touch) const {
    int const depth = touch.GetHistoryDepth();
    require(depth >= 0, "negative touchable history depth");
    Path path;
    for (int d = depth; d >= 0; --d) {
      auto const* pv = touch.GetVolume(d);
      require(pv != nullptr, "null volume inside touchable history");
      path.push_back({pv, touch.GetCopyNumber(d)});
    }
    auto const it = placements.find(path);
    require(it != placements.end(), "unknown full root-to-leaf touchable/copy path");
    return it->second;
  }

  std::string touchable(G4VTouchable const* touch) const {
    if (!touch || !touch->GetVolume())
      return Object().add("availability", quote("unavailable_native_track_touchable"))
        .add("instance_id", "null").add("target_member", "null").str();
    auto const& p = resolve(*touch);
    return Object().add("availability", quote("native_track_touchable_not_vertex_membership"))
      .add("instance_id", integer(p.id)).add("target_member", boolean(p.target))
      .add("copy_history_root_to_leaf", p.history).str();
  }

  Endpoint endpoint(G4StepPoint const& p, bool post) const {
    Endpoint result;
    auto const* t = p.GetTouchable();
    Object geometry;
    if (!t || !t->GetVolume()) {
      require(post && p.GetStepStatus() == fWorldBoundary,
              "unavailable endpoint touchable without explicit native world-boundary status");
      result.worldNull = true;
      geometry.add("availability", quote("explicit_null_world_post"))
        .add("instance_id", "null").add("target_member", "false")
        .add("copy_history_root_to_leaf", "null");
    } else {
      require(!post || p.GetStepStatus() != fWorldBoundary,
              "world-boundary post has a non-null volume; unsupported endpoint semantics");
      auto const& placement = resolve(*t);
      require(p.GetMaterial() != nullptr, "missing endpoint material");
      require(post || p.GetMaterial() == t->GetVolume()->GetLogicalVolume()->GetMaterial(),
              "pre-step material differs from inventory placement");
      result.id = placement.id;
      result.target = placement.target;
      geometry.add("availability", quote("native_touchable"))
        .add("instance_id", integer(placement.id)).add("target_member", boolean(placement.target))
        .add("copy_history_root_to_leaf", placement.history);
    }
    result.json = Object().add("position", position(p.GetPosition()))
      .add("momentum_MeV", vector(p.GetMomentum(), MeV))
      .add("momentum_direction", vector(p.GetMomentumDirection()))
      .add("global_time_ns", number(p.GetGlobalTime()/ns))
      .add("local_time_ns", number(p.GetLocalTime()/ns))
      .add("proper_time_ns", number(p.GetProperTime()/ns))
      .add("kinetic_energy_MeV", number(p.GetKineticEnergy()/MeV))
      .add("total_energy_MeV", number(p.GetTotalEnergy()/MeV))
      .add("weight", number(p.GetWeight()))
      .add("polarization", vector(p.GetPolarization()))
      .add("step_status_native", signedInteger(static_cast<int>(p.GetStepStatus())))
      .add("process_defining_step", process(p.GetProcessDefinedStep()))
      .add("material", p.GetMaterial() ? quote(p.GetMaterial()->GetName()) : "null")
      .add("material_semantics", quote(post ? "native_post_material_not_assumed_equal_to_next_placement"
                                          : "native_pre_material_matches_placement"))
      .add("geometry", geometry.str()).str();
    return result;
  }

  std::string nativeState(G4Track const& t) const {
    auto const* definition = t.GetDefinition();
    require(definition && t.GetDynamicParticle(), "missing particle definition/dynamic particle");
    auto const* lv = t.GetLogicalVolumeAtVertex();
    return Object().add("track_id", signedInteger(t.GetTrackID()))
      .add("parent_id", signedInteger(t.GetParentID()))
      .add("pdg", signedInteger(definition->GetPDGEncoding()))
      .add("particle_name", quote(definition->GetParticleName()))
      .add("native_status", signedInteger(static_cast<int>(t.GetTrackStatus())))
      .add("current_step_number", signedInteger(t.GetCurrentStepNumber()))
      .add("position", position(t.GetPosition())).add("momentum_MeV", vector(t.GetMomentum(), MeV))
      .add("momentum_direction", vector(t.GetMomentumDirection()))
      .add("global_time_ns", number(t.GetGlobalTime()/ns))
      .add("local_time_ns", number(t.GetLocalTime()/ns))
      .add("proper_time_ns", number(t.GetProperTime()/ns))
      .add("kinetic_energy_MeV", number(t.GetKineticEnergy()/MeV))
      .add("total_energy_MeV", number(t.GetTotalEnergy()/MeV))
      .add("weight", number(t.GetWeight())).add("track_length_mm", number(t.GetTrackLength()/mm))
      .add("velocity_mm_per_ns", number(t.GetVelocity()/(mm/ns)))
      .add("polarization", vector(t.GetPolarization()))
      .add("dynamic_charge_eplus", number(t.GetDynamicParticle()->GetCharge()/eplus))
      .add("dynamic_mass_MeV", number(t.GetDynamicParticle()->GetMass()/MeV))
      .add("vertex_position", position(t.GetVertexPosition()))
      .add("vertex_momentum_direction", vector(t.GetVertexMomentumDirection()))
      .add("vertex_kinetic_energy_MeV", number(t.GetVertexKineticEnergy()/MeV))
      .add("vertex_logical_volume", lv ? quote(lv->GetName()) : "null")
      .add("vertex_placement_membership", quote("unresolved_no_vertex_touchable_claim"))
      .add("creator_process", process(t.GetCreatorProcess()))
      .add("track_geometry", touchable(t.GetTouchable())).str();
  }

  TrackInfo& info(G4Track const& t) {
    require(t.GetTrackID() > 0 && t.GetParentID() >= 0 && t.GetDefinition(),
            "invalid assigned native track identity");
    int const pdg = t.GetDefinition()->GetPDGEncoding();
    auto const inserted = tracks.emplace(t.GetTrackID(), TrackInfo{t.GetParentID(), pdg});
    auto& i = inserted.first->second;
    require(i.parent == t.GetParentID() && i.pdg == pdg, "duplicate native track ID with different identity");
    return i;
  }

  void associate(G4Track const& t, TrackInfo& i, char const* at) {
    auto const it = pendingBirths.find(&t);
    if (it == pendingBirths.end()) return;
    auto& birth = births.at(it->second);
    require(i.token == 0 && birth.mappedID == 0, "duplicate child-token association");
    require(t.GetParentID() != 0 && birth.parent == t.GetParentID()
            && birth.pdg == t.GetDefinition()->GetPDGEncoding(), "birth-to-track identity mismatch");
    require(birth.rawChildID <= 0 || birth.rawChildID == t.GetTrackID(), "birth raw assigned ID mismatch");
    i.token = birth.token;
    birth.mappedID = t.GetTrackID();
    // Consume the transient pointer immediately; all subsequent linkage uses native ID and token only.
    pendingBirths.erase(it);
    increment(associationCount);
    writeRow("birth_track_association", Object().add("child_token", integer(i.token))
      .add("track_id", signedInteger(t.GetTrackID())).add("parent_id", signedInteger(t.GetParentID()))
      .add("parent_step", signedInteger(birth.parentStep)).add("observed_at", quote(at))
      .add("sim_particle_presence", quote("not_queried")).str());
  }

  std::string causal(TrackInfo const& i) const {
    return Object().add("child_token", i.token ? integer(i.token) : "null")
      .add("parent_id", signedInteger(i.parent))
      .add("birth_association", quote(i.parent == 0 ? "explicit_primary_root"
        : i.token ? "observed_secondary_birth_incidence" : "unresolved_missing_birth_incidence"))
      .add("ownership", quote("unresolved_no_launch_frontier_algorithm")).str();
  }

  std::string expectedIdentity(G4Track const& t) const {
    require(helper != nullptr && t.GetTrackID() > 0, "missing event helper or assigned track ID");
    Object parent;
    if (t.GetParentID() > 0) {
      parent.add("product_id", integer(helper->productID().value()))
        .add("key", integer(helper->particleKeyFromG4TrackID(t.GetParentID()).asUint()));
    }
    // These are expected collection coordinates, never a dereferenced or validated art::Ptr.
    return Object().add("product_id", integer(helper->productID().value()))
      .add("key", integer(helper->particleKeyFromG4TrackID(t.GetTrackID()).asUint()))
      .add("sim_stage", std::to_string(helper->simStage()))
      .add("parent_id", signedInteger(t.GetParentID()))
      .add("expected_parent", t.GetParentID() ? parent.str() : "null")
      .add("product_presence", quote("unverified_no_product_dereference"))
      .add("original_primary_or_generator_link", "null")
      .add("original_parent_art_pointer", "null")
      .add("original_parent_pointer_availability", quote("not_observed; expected_parent_is_not_a_resolved_pointer")).str();
  }

  void prepare(art::Event const& artEvent, art::InputTag const& tag, SimParticleHelper const& h) {
    require(state == EventState::idle || state == EventState::complete, "overlapping prepared event");
    auto const id = artEvent.id();
    run = id.run();
    subrun = id.subRun();
    event = id.event();
    eventDirectory = outputDirectory / ("run_" + std::to_string(run) + "_subrun_"
                     + std::to_string(subrun) + "_event_" + integer(event));
    // create_directory is the exclusive event reservation; never adopt or truncate an existing event.
    require(fs::create_directory(eventDirectory), "event directory already exists");
    rows = attemptedRows = bytes = volumeCount = birthCount = associationCount = selectedSteps = 0;
    activeTrack = 0;
    kinds.clear();
    hooks.clear();
    placements.clear();
    pendingBirths.clear();
    births.clear();
    tracks.clear();
    generated.clear();
    primaryLinks.clear();
    linkedGeneratorKeys.clear();
    inserted.clear();
    insertedTracks.clear();
    generatorProduct = generatorProtons = omittedParticles = finalParticles = 0;
    primaryHelper = nullptr;
    helper = &h;
    require(h.productID().isValid(), "invalid output SimParticle product identity");
    state = EventState::prepared;
    ledger.clear();
    ledger.open(eventDirectory / "records.jsonl", std::ios::binary | std::ios::out);
    require(static_cast<bool>(ledger), "cannot open event ledger");
    increment(hooks["prepareEvent"]);
    auto const source = artEvent.getValidHandle<GenParticleCollection>(tag);
    auto const* provenance = source.provenance();
    require(source.id().isValid() && provenance && provenance->isValid() && provenance->isPresent(),
            "missing generator product/provenance");
    generatorProduct = source.id().value();
    require(provenance->productID() == source.id(), "generator provenance product mismatch");
    require(!tag.label().empty() && tag.label() == provenance->moduleLabel()
            && tag.instance() == provenance->productInstanceName()
            && (tag.process().empty() || tag.process() == provenance->processName()),
            "configured primary tag differs from resolved generator provenance");
    require(source->size() <= maxRows - rows, "generator count exceeds row budget");
    for (auto const& g : *source) {
      if (g.pdgId() == PDGCode::proton) increment(generatorProtons);
    }
    std::ostringstream history;
    history << artEvent.processHistoryID();
    require(!history.str().empty(), "missing process history identity");
    std::string psets = "[";
    for (auto const& pset : provenance->psetIDs()) {
      std::ostringstream text;
      text << pset;
      require(!text.str().empty(), "empty generator parameter-set identity");
      if (psets.size() != 1) psets += ',';
      psets += quote(text.str());
    }
    psets += ']';
    require(!provenance->branchName().empty() && !provenance->moduleLabel().empty()
            && !provenance->processName().empty() && !provenance->friendlyClassName().empty()
            && !provenance->psetIDs().empty(), "incomplete generator provenance");
    writeRow("source_begin", Object().add("generator_product_id", integer(generatorProduct))
      .add("sim_product_id", integer(h.productID().value())).add("sim_stage", integer(h.simStage()))
      .add("generator_size", integer(source->size())).add("generator_protons", integer(generatorProtons))
      .add("configured_primary_tag", quote(tag.encode())).add("process_history_id", quote(history.str()))
      .add("generator_provenance", Object().add("branch_name", quote(provenance->branchName()))
        .add("module_label", quote(provenance->moduleLabel()))
        .add("product_instance_name", quote(provenance->productInstanceName()))
        .add("process_name", quote(provenance->processName()))
        .add("friendly_class_name", quote(provenance->friendlyClassName()))
        .add("parameter_set_ids", psets).str())
      .add("max_rows", integer(maxRows)).add("max_volumes", integer(maxVolumes)).add("max_bytes", integer(maxBytes))
      .add("persistent_output_verified", "false").add("exposure_certified", "false").str());
    for (std::size_t k = 0; k != source->size(); ++k) {
      auto const& g = source->at(k);
      require(generated.emplace(k, Generated{static_cast<int>(g.pdgId()), &g}).second,
              "duplicate generator key");
      auto const& p = g.momentum();
      std::string const p4 = '[' + number(p.px()/MeV) + ',' + number(p.py()/MeV) + ','
        + number(p.pz()/MeV) + ',' + number(p.e()/MeV) + ']';
      writeRow("generated_primary", Object().add("generator_product_id", integer(generatorProduct))
        .add("generator_key", integer(k)).add("pdg", signedInteger(static_cast<int>(g.pdgId())))
        .add("generator_id", signedInteger(static_cast<int>(g.generatorId().id())))
        .add("position_mu2e_mm", vector(g.position(), mm)).add("four_momentum_MeV", p4)
        .add("global_time_ns", number(g.time()/ns)).add("proper_time_ns", number(g.properTime()/ns))
        .add("generator_weight", "null").str());
    }
    writeRow("source_end", Object().add("generator_size", integer(generated.size()))
      .add("generator_protons", integer(generatorProtons)).str());
  }

  void begin(unsigned r, unsigned s, U64 e, SimParticleHelper const& h,
             SimParticlePrimaryHelper const& primary) {
    require(state == EventState::prepared, "native event has no prepared source");
    require(r == run && s == subrun && e == event && helper == &h, "prepared/native event identity mismatch");
    primaryHelper = &primary;
    state = EventState::active;
    observed("beginEvent");
    std::string selected = "[";
    for (auto const& name : selectors) {
      if (selected.size() != 1) selected += ',';
      selected += quote(name);
    }
    selected += ']';
    writeRow("event_begin", Object().add("outcome", quote("begun_not_complete"))
      .add("proposal_only", "true").add("target_physical_names", selected)
      .add("target_material", quote(materialName)).add("mu2e_origin_in_world_mm", vector(origin, mm))
      .add("max_rows", integer(maxRows)).add("max_volumes", integer(maxVolumes))
      .add("max_bytes", integer(maxBytes))
      .add("units", quote("mm,ns,MeV,eplus; native dimensionless weights"))
      .add("string_encoding", quote("byte_preserving_ASCII_JSON_escape; non_ASCII_bytes_use_u00XX"))
      .add("source_runtime_config_hashes", "null").add("job_shard_process_history", "null")
      .add("generated_primary_products_and_exposure", quote("source_begin_and_generated_primary_rows; exposure_unverified"))
      .add("eligibility_reason", quote("proposal_only; unvalidated_runtime; missing_shard_runtime_and_persistence_receipts; unresolved_ownership"))
      .add("hash_policy", quote("external_postread_hash_actual_file_bytes_only"))
      .add("callback_contract", quote("step_before_cuts; cuts_once_per_step; no_mutation; per_thread"))
      .add("native_status_mapping", quote("numeric_observation_not_runtime_certified")).str());
    inventory();
  }

  void stacking(G4Track const& t, int classification, bool stackingCut, bool commonCut) {
    observed("stacking");
    trackContext("stacking", t);
    require(activeTrack == 0, "stacking during active tracking callback sequence");
    require(classification == static_cast<int>(fUrgent) || classification == static_cast<int>(fKill),
            "unsupported stacking classification");
    require(tracks.count(t.GetTrackID()) == 0, "duplicate stacking or track identity");
    auto& i = info(t);
    require(!(stackingCut && commonCut), "common stacking cut was supplied after short-circuit");
    require((!stackingCut && !commonCut) || classification == static_cast<int>(fKill),
            "cut outcome disagrees with native stacking classification");
    associate(t, i, "stacking");
    if (t.GetParentID() == 0) {
      require(primaryHelper != nullptr, "missing authoritative primary helper");
      auto const original = primaryHelper->getEntry(t.GetTrackID());
      auto const* p = std::get_if<art::Ptr<GenParticle>>(&original);
      require(p && p->isNonnull() && p->id().isValid() && p->id().value() == generatorProduct,
              "primary helper does not point to configured generator product");
      auto const key = static_cast<U64>(p->key());
      auto const g = generated.find(key);
      require(g != generated.end() && g->second.pdg == i.pdg && p->isAvailable()
              && p->get() == g->second.address, "primary helper input pointer/PDG mismatch");
      require(linkedGeneratorKeys.insert(key).second
              && primaryLinks.emplace(t.GetTrackID(), PrimaryLink{key, classification}).second,
              "duplicate authoritative primary association");
      writeRow("primary_link", Object().add("track_id", signedInteger(t.GetTrackID()))
        .add("generator_product_id", integer(generatorProduct)).add("generator_key", integer(key))
        .add("pdg", signedInteger(i.pdg)).add("classification_native", signedInteger(classification))
        .add("source_pointer_available", "true").add("native_at_classification", nativeState(t)).str());
    }
    increment(i.stackings);
    i.life = classification == static_cast<int>(fKill) ? Life::stackingKilled : Life::stacked;
    // Killed classifications carry no expected SimParticle coordinates or validity claims.
    writeRow("stacking", Object().add("track_id", signedInteger(t.GetTrackID()))
      .add("parent_id", signedInteger(t.GetParentID())).add("pdg", signedInteger(i.pdg))
      .add("classification_native", signedInteger(classification))
      .add("stacking_cut", boolean(stackingCut))
      .add("common_cut_evaluated", boolean(!stackingCut))
      .add("common_cut", stackingCut ? "null" : boolean(commonCut))
      .add("observation_number", integer(i.stackings)).add("causal", causal(i)).str());
  }

  template<class T> std::string pointerIdentity(art::Ptr<T> const& p) const {
    if (p.isNull()) return "null";
    require(p.id().isValid(), "non-null product pointer has invalid ProductID");
    return Object().add("product_id", integer(p.id().value())).add("key", integer(p.key())).str();
  }

  std::string particleIdentity(int tid, SimParticle const& p) const {
    auto const t = tracks.find(tid);
    require(t != tracks.end() && helper, "SimParticle without observed native track");
    return Object().add("track_id", signedInteger(tid)).add("parent_id", signedInteger(t->second.parent))
      .add("pdg", signedInteger(static_cast<int>(p.pdgId())))
      .add("product_id", integer(helper->productID().value())).add("key", integer(p.id().asUint()))
      .add("sim_stage", integer(p.simStage())).add("creation_code", signedInteger(static_cast<int>(p.creationCode().id())))
      .add("parent", pointerIdentity(p.parent())).add("generator", pointerIdentity(p.genParticle())).str();
  }

  void insertion(G4Track const& t, SimParticle const& p) {
    observed("simParticleInserted");
    trackContext("simParticleInserted", t);
    require(activeTrack == 0 && tracks.count(t.GetTrackID()) == 1,
            "insertion outside accepted pre-tracking sequence");
    auto& i = info(t);
    require(i.life == Life::stacked && i.stackings == 1 && !i.insertionObserved && !i.insertionOmitted,
            "duplicate or invalid insertion outcome");
    require(static_cast<int>(p.pdgId()) == i.pdg && p.simStage() == helper->simStage()
            && p.id() == helper->particleKeyFromG4TrackID(t.GetTrackID()), "inserted SimParticle identity mismatch");
    if (!i.parent) {
      auto const link = primaryLinks.find(t.GetTrackID());
      require(link != primaryLinks.end() && p.parent().isNull() && p.genParticle().isNonnull()
              && p.genParticle().id().value() == generatorProduct
              && p.genParticle().key() == link->second.generatorKey, "inserted primary link mismatch");
    } else {
      auto const parent = insertedTracks.find(i.parent);
      require(parent != insertedTracks.end() && p.parent().isNonnull()
              && p.parent().id() == helper->productID() && p.parent().key() == parent->second
              && p.genParticle().isNull(), "inserted secondary parent link mismatch");
    }
    auto const identity = particleIdentity(t.GetTrackID(), p);
    require(inserted.emplace(p.id().asUint(), Inserted{t.GetTrackID(), identity}).second
            && insertedTracks.emplace(t.GetTrackID(), p.id().asUint()).second, "duplicate inserted key/track");
    i.insertionObserved = true;
    std::string row = identity;
    row.pop_back();
    row += ",\"insertion_observed\":true,\"persistent_output_verified\":false}";
    writeRow("sim_particle_insert", row);
  }

  void omission(G4Track const& t) {
    observed("simParticleOmitted");
    trackContext("simParticleOmitted", t);
    require(activeTrack == 0 && tracks.count(t.GetTrackID()) == 1, "omission without accepted stacking");
    auto& i = info(t);
    require(i.life == Life::stacked && i.stackings == 1 && !i.insertionObserved && !i.insertionOmitted,
            "duplicate or invalid omission outcome");
    i.insertionOmitted = true;
    increment(omittedParticles);
    writeRow("sim_particle_omitted", Object().add("track_id", signedInteger(t.GetTrackID()))
      .add("parent_id", signedInteger(i.parent)).add("pdg", signedInteger(i.pdg))
      .add("reason", quote("sim_particle_size_limit")).add("persistent_output_verified", "false").str());
  }

  void reconcile(SimParticleCollection const& collection) {
    observed("reconcileParticles");
    require(activeTrack == 0 && omittedParticles == 0, "incomplete SimParticle insertion population");
    require(primaryLinks.size() == generated.size() && linkedGeneratorKeys.size() == generated.size(),
            "generated-source/primary-classification population mismatch");
    require(collection.size() == inserted.size(), "assembled SimParticle count mismatch");
    std::map<U64, std::set<U64>> expectedDaughters;
    for (auto const& entry : collection) {
      auto const key = static_cast<U64>(entry.first.asUint());
      auto const record = inserted.find(key);
      require(record != inserted.end() && entry.second.id() == entry.first,
              "assembled particle key has no insertion record");
      auto const& p = entry.second;
      auto const t = tracks.find(record->second.trackID);
      require(t != tracks.end() && t->second.life == Life::ended && t->second.insertionObserved,
              "assembled particle has incomplete native history");
      require(particleIdentity(record->second.trackID, p) == record->second.identity,
              "assembled particle identities/links differ from inserted object");
      if (p.parent().isNonnull()) {
        require(p.parent().id() == helper->productID() && inserted.count(p.parent().key()) == 1,
                "assembled parent pointer outside produced collection");
        expectedDaughters[p.parent().key()].insert(key);
      }
    }
    for (auto const& entry : collection) {
      auto const key = static_cast<U64>(entry.first.asUint());
      auto const& record = inserted.at(key);
      std::set<U64> daughters;
      std::string encoded = "[";
      for (auto const& p : entry.second.daughters()) {
        require(p.isNonnull() && p.id() == helper->productID() && inserted.count(p.key()) == 1
                && daughters.insert(p.key()).second, "invalid/duplicate assembled daughter");
        if (encoded.size() != 1) encoded += ',';
        encoded += pointerIdentity(p);
      }
      encoded += ']';
      require(daughters == expectedDaughters[key], "nonreciprocal assembled parent/daughter links");
      std::string row = particleIdentity(record.trackID, entry.second);
      row.pop_back();
      row += ",\"daughters\":" + encoded + ",\"in_memory_only\":true,\"persistent_output_verified\":false}";
      writeRow("sim_particle_final", row);
      increment(finalParticles);
    }
    require(finalParticles == inserted.size(), "final particle row count mismatch");
    writeRow("source_reconciliation", Object().add("generator_size", integer(generated.size()))
      .add("generator_protons", integer(generatorProtons)).add("primary_links", integer(primaryLinks.size()))
      .add("inserted_particles", integer(inserted.size())).add("omitted_particles", integer(omittedParticles))
      .add("final_particles", integer(finalParticles)).add("in_memory_links_consistent", "true")
      .add("persistent_output_verified", "false").add("exposure_certified", "false").str());
  }

  void beginTrack(G4Track const& t) {
    observed("beginTrack");
    trackContext("beginTrack", t);
    require(activeTrack == 0, "overlapping active track callbacks");
    require(tracks.count(t.GetTrackID()) == 1, "tracking without a preceding stacking callback");
    auto& i = info(t);
    require(i.stackings == 1 && i.life == Life::stacked, "tracking requires exactly one accepted stacking");
    require(i.insertionObserved != i.insertionOmitted, "tracking needs exactly one actual insertion outcome");
    require(i.life != Life::active && i.life != Life::stackingKilled, "invalid beginTrack lifecycle");
    require(!i.pendingCuts, "beginTrack with pending cut observation");
    require(i.begins == 0 && i.ends == 0, "resumed or revisited tracking is unsupported");
    require(t.GetCurrentStepNumber() == 0,
            "first observed visit already has native steps; cross-event continuation unsupported");
    associate(t, i, "beginTrack");
    increment(i.begins);
    require(i.begins == i.ends + 1, "unbalanced lifecycle begin/end");
    i.life = Life::active;
    activeTrack = t.GetTrackID();
    writeRow("track_begin", Object().add("native", nativeState(t)).add("expected_sim_particle", expectedIdentity(t))
      .add("causal", causal(i)).add("lifecycle_visit", integer(i.begins))
      .add("previous_visit_end_status_native", "null")
      .add("callback_meaning", quote("first_observed_tracking_visit_not_a_birth_claim"))
      .add("stacking_observations", integer(i.stackings)).str());
  }

  TrackInfo& active(G4Track const& t) {
    require(activeTrack == t.GetTrackID(), "callback track is not the active track");
    auto& i = info(t);
    require(i.life == Life::active, "callback outside active track visit");
    return i;
  }

  void step(G4Step const& s) {
    observed("step");
    auto const* t = s.GetTrack();
    require(t && s.GetPreStepPoint() && s.GetPostStepPoint(), "incomplete native step");
    trackContext("step", *t);
    auto& i = active(*t);
    auto const status = t->GetTrackStatus();
    require(status != fSuspend && status != fSuspendAndWait && status != fPostponeToNextEvent,
            "suspension/postponement is unsupported before cut evaluation");
    require(!i.pendingCuts, "previous step has no cuts callback");
    int const ordinal = t->GetCurrentStepNumber();
    require(ordinal > 0 && (i.lastStep == -1 ? ordinal == 1 : ordinal - 1 == i.lastStep),
            "duplicate/nonconsecutive native step ordinal");
    i.lastStep = ordinal;
    i.pendingCuts = true;
    auto const pre = endpoint(*s.GetPreStepPoint(), false);
    auto const post = endpoint(*s.GetPostStepPoint(), true);
    bool const selected = pre.target || post.target;
    bool const boundary = s.GetPostStepPoint()->GetStepStatus() == fGeomBoundary
      || s.GetPostStepPoint()->GetStepStatus() == fWorldBoundary || pre.id != post.id;
    std::string const transition = pre.target && post.target ? "internal"
      : pre.target ? "leaving" : post.target ? "entering" : "outside";
    auto const parentStep = Object().add("track_id", signedInteger(t->GetTrackID()))
      .add("step_number", signedInteger(ordinal)).add("lifecycle_visit", integer(i.begins))
      .add("target_step_row_present", boolean(selected)).add("transition_observation", quote(transition))
      .add("boundary_observed", boolean(boundary)).add("pre", pre.json).add("post", post.json).str();
    if (selected) {
      increment(selectedSteps);
      writeRow("target_step", Object().add("parent_step", parentStep)
        .add("target_step_ordinal", integer(selectedSteps)).add("causal", causal(i))
        .add("expected_sim_particle", expectedIdentity(*t))
        .add("track_status_native", signedInteger(static_cast<int>(t->GetTrackStatus())))
        .add("step_length_mm", number(s.GetStepLength()/mm))
        .add("total_energy_deposit_MeV", number(s.GetTotalEnergyDeposit()/MeV))
        .add("nonionizing_energy_deposit_MeV", number(s.GetNonIonizingEnergyDeposit()/MeV))
        .add("zero_length", boolean(s.GetStepLength() == 0.))
        .add("cut_outcome", quote("separate_cuts_row_by_track_and_step"))
        .add("ownership", quote("unresolved_no_launch_frontier_classification"))
        .add("world_post_null", boolean(post.worldNull)).str());
    }
    // Scan every native step, including wholly exterior steps, before any cut-outcome callback.
    auto const* secondaries = s.GetSecondaryInCurrentStep();
    if (secondaries) {
      for (auto const* child : *secondaries) {
        require(child && child != t && child->GetDefinition(), "invalid secondary-in-current-step");
        require(pendingBirths.count(child) == 0, "duplicate unconsumed secondary pointer");
        increment(birthCount);
        U64 const token = birthCount;
        Birth birth{token, t->GetTrackID(), ordinal, child->GetDefinition()->GetPDGEncoding(), child->GetTrackID()};
        require(births.emplace(token, birth).second, "duplicate child token");
        require(pendingBirths.emplace(child, token).second, "duplicate pending child token");
        writeRow("secondary_birth", Object().add("child_token", integer(token))
          .add("parent_step", parentStep).add("parent_causal", causal(i))
          .add("raw_child_track_id", signedInteger(child->GetTrackID()))
          .add("raw_child_parent_id", signedInteger(child->GetParentID()))
          .add("pdg", signedInteger(birth.pdg)).add("native_child_at_birth", nativeState(*child))
          .add("birth_incidence", quote("observed_GetSecondaryInCurrentStep; no_time_or_proximity_inference"))
          .add("ownership", quote(boundary ? "unresolved_same_parentstep_boundary"
                                           : "unresolved_no_launch_frontier_algorithm"))
          .add("pointer_persistence", quote("none; transient_lookup_consumed_at_stacking_or_beginTrack")).str());
      }
    }
  }

  void cuts(G4Track const& t, bool steppingCut, bool commonCut) {
    observed("cuts");
    trackContext("cuts", t);
    auto& i = active(t);
    require(!(steppingCut && commonCut), "common stepping cut was supplied after short-circuit");
    require(i.pendingCuts && i.lastStep == t.GetCurrentStepNumber(), "cuts without exactly one preceding step");
    i.pendingCuts = false;
    writeRow("cuts", Object().add("track_id", signedInteger(t.GetTrackID()))
      .add("parent_id", signedInteger(t.GetParentID())).add("step_number", signedInteger(i.lastStep))
      .add("lifecycle_visit", integer(i.begins)).add("stepping_cut", boolean(steppingCut))
      .add("common_cut_evaluated", boolean(!steppingCut))
      .add("common_cut", steppingCut ? "null" : boolean(commonCut))
      .add("native_status", signedInteger(static_cast<int>(t.GetTrackStatus())))
      .add("causal", causal(i)).str());
  }

  void endTrack(G4Track const& t) {
    observed("endTrack");
    trackContext("endTrack", t);
    auto& i = active(t);
    auto const status = t.GetTrackStatus();
    require(status == fStopAndKill || status == fKillTrackAndSecondaries,
            "nonterminal end or suspension/postponement is unsupported at track end");
    if (status == fKillTrackAndSecondaries) {
      for (auto const& birth : births) {
        require(birth.second.parent != t.GetTrackID() || birth.second.mappedID != 0,
                "kill-with-secondaries leaves unassigned offspring; terminal child disposition unsupported");
      }
    }
    require(!i.pendingCuts, "endTrack before cuts for last observed step");
    require(t.GetCurrentStepNumber() == i.lastStep || (i.lastStep == -1 && t.GetCurrentStepNumber() == 0),
            "unobserved steps before endTrack");
    increment(i.ends);
    require(i.ends == i.begins, "unbalanced lifecycle end");
    i.endStatus = static_cast<int>(t.GetTrackStatus());
    i.life = Life::ended;
    activeTrack = 0;
    writeRow("track_end", Object().add("native", nativeState(t)).add("expected_sim_particle", expectedIdentity(t))
      .add("causal", causal(i)).add("lifecycle_visit", integer(i.begins))
      .add("callback_meaning", quote("native_terminal_callback_not_persisted_product_verification"))
      .add("resume_candidate_native_enum", "false").str());
  }

  std::string sourceCounts() const {
    return ",\"source_primary_count\":" + integer(generated.size())
      + ",\"source_proton_count\":" + integer(generatorProtons)
      + ",\"primary_link_count\":" + integer(primaryLinks.size())
      + ",\"sim_particle_inserted\":" + integer(inserted.size())
      + ",\"sim_particle_omitted\":" + integer(omittedParticles)
      + ",\"sim_particle_final\":" + integer(finalParticles)
      + ",\"source_reconciled_in_memory\":true,\"persistent_output_verified\":false";
  }

  void end(int status, bool eventAborted, bool simOverflow, U64 stepKills, U64 fieldKills,
           SimParticleCollection const& collection) {
    observed("endEvent");
    require(!eventAborted, "native G4 event aborted; normal recorder completion forbidden");
    require(activeTrack == 0, "event ended during active track visit");
    require(pendingBirths.empty(), "unconsumed secondary birth tokens prevent callback completeness");
    U64 unresolved = 0;
    U64 noBegin = 0;
    U64 deferred = 0;
    U64 const suspendedOrPostponed = 0;
    U64 killedInStacking = 0;
    for (auto const& entry : tracks) {
      auto const& i = entry.second;
      require(!i.pendingCuts && i.begins == i.ends && i.life != Life::active,
              "unbalanced track lifecycle at event end");
      require(i.stackings == 1 && ((i.life == Life::stackingKilled && i.begins == 0)
              || (i.life == Life::ended && i.begins == 1)), "unsupported final track lifecycle");
      require(i.token == 0 || births.at(i.token).mappedID == entry.first,
              "track-to-birth reverse association mismatch");
      if (i.parent && !i.token) increment(unresolved);
      if (!i.begins) increment(noBegin);
      if (i.life == Life::stacked) increment(deferred);
      if (i.life == Life::ended) {
        require(i.endStatus == static_cast<int>(fStopAndKill)
                || i.endStatus == static_cast<int>(fKillTrackAndSecondaries), "unsupported final native status");
      }
      if (i.life == Life::stackingKilled) increment(killedInStacking);
    }
    require(unresolved == 0 && deferred == 0, "unresolved ancestry or unprocessed stacking prevents completeness");
    reconcile(collection);
    require(birthCount == static_cast<U64>(births.size()), "birth token count mismatch");
    require(associationCount + static_cast<U64>(pendingBirths.size()) == birthCount,
            "birth association conservation failure");
    require(kinds["secondary_birth"] == birthCount && kinds["birth_track_association"] == associationCount,
            "persisted birth/association count mismatch");
    require(kinds["target_step"] == selectedSteps, "target step count mismatch");
    require(kinds["volume"] == volumeCount, "inventory row count mismatch");
    U64 countedRows = 0;
    for (auto const& item : kinds) {
      require(item.second <= std::numeric_limits<U64>::max() - countedRows, "row-kind sum overflow");
      countedRows += item.second;
    }
    require(countedRows == rows && attemptedRows == rows, "row accounting mismatch before completion");
    require(kinds["track_begin"] == hooks["beginTrack"] && kinds["track_end"] == hooks["endTrack"]
            && kinds["stacking"] == hooks["stacking"] && kinds["cuts"] == hooks["cuts"],
            "observed-hook/row count mismatch");
    require(hooks["step"] == hooks["cuts"], "step/cuts observation count mismatch");
    require(hooks["simParticleInserted"] == inserted.size() && hooks["simParticleOmitted"] == omittedParticles
            && hooks["prepareEvent"] == 1 && hooks["reconcileParticles"] == 1,
            "source/insertion hook count mismatch");
    auto eventEnd = Object().add("status_native", signedInteger(status))
      .add("event_aborted_native", boolean(eventAborted))
      .add("sim_overflow", boolean(simOverflow)).add("step_kills", integer(stepKills))
      .add("field_kills", integer(fieldKills)).add("secondary_births", integer(birthCount))
      .add("associated_births", integer(associationCount))
      .add("unconsumed_birth_tokens", integer(static_cast<U64>(pendingBirths.size())))
      .add("unresolved_nonprimary_tracks", integer(unresolved))
      .add("tracks_without_begin", integer(noBegin)).add("stacked_not_begun_at_end", integer(deferred))
      .add("suspended_or_postponed_ended_visits", integer(suspendedOrPostponed))
      .add("stacking_killed_native_enum", integer(killedInStacking))
      .add("native_status_interpretation", quote("not_certified; event_end_does_not_imply_good_physics_status"))
      .add("zero_target_steps", boolean(selectedSteps == 0))
      .add("zero_exit_or_exposure_claim", "false").str();
    eventEnd.pop_back();
    writeRow("event_end", eventEnd + sourceCounts() + '}');
    ledger.flush();
    require(static_cast<bool>(ledger), "ledger flush failed");
    ledger.close();
    require(!ledger.fail(), "ledger close failed");
    U64 const ledgerBytes = bytes;
    auto finalKinds = kinds;
    increment(finalKinds["completion"]);
    require(rows < maxRows, "maxRows exceeded; first_offending_kind=completion");
    auto summary = Object().add("outcome", quote("endEvent_and_ledger_flush_complete"))
      .add("library_eligible", "false").add("row_count", integer(rows + 1))
      .add("attempted_rows", integer(attemptedRows + 1)).add("per_kind_counts", counterObject(finalKinds))
      .add("observed_hooks", counterObject(hooks)).add("secondary_births", integer(birthCount))
      .add("associated_births", integer(associationCount))
      .add("unconsumed_birth_tokens", integer(static_cast<U64>(pendingBirths.size())))
      .add("unresolved_nonprimary_tracks", integer(unresolved))
      .add("volume_count", integer(volumeCount)).add("target_step_count", integer(selectedSteps))
      .add("ledger_bytes", integer(ledgerBytes))
      .add("stacked_not_begun_at_end", integer(deferred))
      .add("suspended_or_postponed_ended_visits", integer(suspendedOrPostponed))
      .add("byte_cap_scope", quote("records.jsonl_plus_completion.json; filesystem_metadata_excluded"))
      .add("recording_row_overflow", "false").add("recording_byte_overflow", "false")
      .add("recording_volume_overflow", "false")
      .add("manifest_hash", "null").add("manifest_hash_policy", quote("postprocess_actual_records.jsonl_bytes"))
      .add("status_native", signedInteger(status)).add("event_aborted_native", boolean(eventAborted))
      .add("sim_overflow", boolean(simOverflow))
      .add("step_kills", integer(stepKills)).add("field_kills", integer(fieldKills))
      .add("eligibility_reason", quote("unapplied_proposal; missing_shard_runtime_and_persistence_receipts; ownership_and_runtime_unvalidated"))
      .add("completion_not_physics_or_exposure_certification", "true").str();
    summary.pop_back();
    summary += sourceCounts() + '}';
    std::ofstream marker(eventDirectory / "completion.pending", std::ios::binary | std::ios::out);
    require(static_cast<bool>(marker), "cannot open pending completion");
    writeRow("completion", summary, marker);
    marker.flush();
    require(static_cast<bool>(marker), "completion flush failed");
    marker.close();
    require(!marker.fail(), "completion close failed");
    // Only the final rename publishes completion; any earlier failure retains the partial directory.
    fs::rename(eventDirectory / "completion.pending", eventDirectory / "completion.json");
    state = EventState::complete;
    helper = nullptr;
  }
};

TargetRecording::TargetRecording(std::string outputDirectory,
                                 std::vector<std::string> targetPhysicalVolumeNames,
                                 std::string targetMaterialName, U64 maxRows, U64 maxVolumes,
                                 U64 maxBytes, CLHEP::Hep3Vector const& mu2eOrigin)
  : impl_(new Impl(std::move(outputDirectory), std::move(targetPhysicalVolumeNames),
                   std::move(targetMaterialName), maxRows, maxVolumes, maxBytes, mu2eOrigin)) {}
TargetRecording::~TargetRecording() = default;

void TargetRecording::prepareEvent(art::Event const& event, art::InputTag const& tag, SimParticleHelper const& helper) {
  impl_->guarded("prepareEvent", [&] { impl_->prepare(event, tag, helper); });
}
void TargetRecording::beginEvent(unsigned run, unsigned subrun, U64 event, SimParticleHelper const& helper,
                                 SimParticlePrimaryHelper const& primary) {
  impl_->guarded("beginEvent", [&] { impl_->begin(run, subrun, event, helper, primary); });
}
void TargetRecording::simParticleInserted(G4Track const& t, SimParticle const& p) {
  impl_->guarded("simParticleInserted", [&] { impl_->insertion(t, p); });
}
void TargetRecording::simParticleOmitted(G4Track const& t) {
  impl_->guarded("simParticleOmitted", [&] { impl_->omission(t); });
}
void TargetRecording::stacking(G4Track const& t, int classification, bool stackingCut, bool commonCut) {
  impl_->guarded("stacking", [&] { impl_->stacking(t, classification, stackingCut, commonCut); });
}
void TargetRecording::beginTrack(G4Track const& t) {
  impl_->guarded("beginTrack", [&] { impl_->beginTrack(t); });
}
void TargetRecording::step(G4Step const& s) {
  impl_->guarded("step", [&] { impl_->step(s); });
}
void TargetRecording::cuts(G4Track const& t, bool steppingCut, bool commonCut) {
  impl_->guarded("cuts", [&] { impl_->cuts(t, steppingCut, commonCut); });
}
void TargetRecording::endTrack(G4Track const& t) {
  impl_->guarded("endTrack", [&] { impl_->endTrack(t); });
}
void TargetRecording::endEvent(int status, bool eventAborted, bool simOverflow, U64 stepKills, U64 fieldKills,
                               SimParticleCollection const& collection) {
  impl_->guarded("endEvent", [&] { impl_->end(status, eventAborted, simOverflow, stepKills, fieldKills, collection); });
}
}
