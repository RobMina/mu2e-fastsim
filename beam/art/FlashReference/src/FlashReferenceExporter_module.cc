// Read-only typed export of stored raw reference products, including explicit empty ledgers.
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Provenance.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/PhysicalVolumeInfoMultiCollection.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StatusG4.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"

namespace mu2e {
namespace flashref {
using U64 = ULong64_t;
using U32 = UInt_t;
using I32 = Int_t;
using F32 = Float_t;
using F64 = Double_t;
using PIDValue = decltype(std::declval<art::ProductID const&>().value());
using PtrKey = decltype(std::declval<art::Ptr<SimParticle> const&>().key());
static_assert(std::is_integral_v<PtrKey> && std::is_unsigned_v<PtrKey>);
static_assert(sizeof(PtrKey) <= sizeof(U64));
static_assert(std::is_integral_v<PIDValue> && std::is_unsigned_v<PIDValue>);
static_assert(sizeof(PIDValue) <= sizeof(U64));
static_assert(sizeof(std::size_t) <= sizeof(U64));
static_assert(sizeof(decltype(std::declval<cet::map_vector_key const&>().asUint())) <= sizeof(U64));
static_assert(sizeof(StepPointMC::VolumeId_type) <= sizeof(U64));
static_assert(sizeof(F32) == 4 && sizeof(F64) == 8);

struct Identity {
  U32 run{}, subrun{}, event{};
  U64 ordinal{}, product{}, index{};
  void bind(TTree* tree, bool collection = true) {
    tree->Branch("run", &run); tree->Branch("subrun", &subrun); tree->Branch("event", &event);
    tree->Branch("ordinal", &ordinal);
    if (collection) { tree->Branch("product", &product); tree->Branch("index", &index); }
  }
};

struct Pointer {
  U32 state{};
  U64 product{}, key{};
  void bind(TTree* tree, std::string const& name) {
    tree->Branch((name + "_state").c_str(), &state);
    tree->Branch((name + "_product").c_str(), &product);
    tree->Branch((name + "_key").c_str(), &key);
  }
  template<class T> void assign(art::Ptr<T> const& ptr, U64& failures) {
    state = 0; product = static_cast<U64>(ptr.id().value()); key = static_cast<U64>(ptr.key());
    if (ptr.isNull()) return;
    try { state = ptr.isAvailable() && ptr.get() ? 1 : 2; }
    catch (...) { state = 3; }
    if (state != 1) ++failures;
  }
};

struct StepRow {
  Pointer particle;
  U64 volume{};
  F64 x{}, y{}, z{}, post_x{}, post_y{}, post_z{}, px{}, py{}, pz{}, post_px{}, post_py{}, post_pz{}, time{}, proper_time{};
  F32 total_edep{}, nonionizing_edep{}, visible_edep{}, length{};
  I32 end_process{};
  void bind(TTree* t) {
    particle.bind(t, "particle"); t->Branch("volume", &volume);
    for (auto const& [name, ptr] : std::vector<std::pair<char const*, F64*>>{
        {"x", &x}, {"y", &y}, {"z", &z}, {"post_x", &post_x}, {"post_y", &post_y}, {"post_z", &post_z},
        {"px", &px}, {"py", &py}, {"pz", &pz}, {"post_px", &post_px}, {"post_py", &post_py}, {"post_pz", &post_pz},
        {"time", &time}, {"proper_time", &proper_time}}) t->Branch(name, ptr);
    t->Branch("total_edep", &total_edep); t->Branch("nonionizing_edep", &nonionizing_edep);
    t->Branch("visible_edep", &visible_edep); t->Branch("length", &length); t->Branch("end_process", &end_process);
  }
  void assign(StepPointMC const& s, U64& failures) {
    particle.assign(s.simParticle(), failures); volume = static_cast<U64>(s.volumeId());
    x = s.position().x(); y = s.position().y(); z = s.position().z();
    post_x = s.postPosition().x(); post_y = s.postPosition().y(); post_z = s.postPosition().z();
    px = s.momentum().x(); py = s.momentum().y(); pz = s.momentum().z();
    post_px = s.postMomentum().x(); post_py = s.postMomentum().y(); post_pz = s.postMomentum().z();
    time = s.time(); proper_time = s.properTime();
    total_edep = static_cast<F32>(s.totalEDep()); nonionizing_edep = static_cast<F32>(s.nonIonizingEDep());
    visible_edep = static_cast<F32>(s.visibleEDep()); length = static_cast<F32>(s.stepLength());
    end_process = static_cast<I32>(s.endProcessCode().id());
  }
};

struct ParticleRow {
  U64 key{}, id{}, ndaughters{};
  U32 stage{}, start_volume{}, end_volume{}, start_g4status{}, end_g4status{}, end_defined{};
  I32 pdg{}, creation{}, stopping{}, nsteps{}, ion_float_level{};
  Pointer parent, generator;
  F64 x{}, y{}, z{}, end_x{}, end_y{}, end_z{};
  F32 px{}, py{}, pz{}, energy{}, end_px{}, end_py{}, end_pz{}, end_energy{}, time{}, proper_time{},
      end_time{}, end_proper_time{}, pre_last_ke{}, end_ke{}, track_length{}, ion_excitation{};
  void bind(TTree* t) {
    for (auto const& [n, p] : std::vector<std::pair<char const*, U64*>>{
        {"key", &key}, {"id", &id}, {"ndaughters", &ndaughters}}) t->Branch(n, p);
    for (auto const& [n, p] : std::vector<std::pair<char const*, U32*>>{
        {"stage", &stage}, {"start_volume", &start_volume}, {"end_volume", &end_volume},
        {"start_g4status", &start_g4status}, {"end_g4status", &end_g4status}, {"end_defined", &end_defined}}) t->Branch(n, p);
    for (auto const& [n, p] : std::vector<std::pair<char const*, I32*>>{
        {"pdg", &pdg}, {"creation", &creation}, {"stopping", &stopping}, {"nsteps", &nsteps},
        {"ion_float_level", &ion_float_level}}) t->Branch(n, p);
    parent.bind(t, "parent"); generator.bind(t, "generator");
    for (auto const& [n, p] : std::vector<std::pair<char const*, F64*>>{
        {"x", &x}, {"y", &y}, {"z", &z}, {"end_x", &end_x}, {"end_y", &end_y}, {"end_z", &end_z}}) t->Branch(n, p);
    for (auto const& [n, p] : std::vector<std::pair<char const*, F32*>>{
        {"px", &px}, {"py", &py}, {"pz", &pz}, {"energy", &energy}, {"end_px", &end_px}, {"end_py", &end_py},
        {"end_pz", &end_pz}, {"end_energy", &end_energy}, {"time", &time}, {"proper_time", &proper_time},
        {"end_time", &end_time}, {"end_proper_time", &end_proper_time}, {"pre_last_ke", &pre_last_ke},
        {"end_ke", &end_ke}, {"track_length", &track_length}, {"ion_excitation", &ion_excitation}}) t->Branch(n, p);
  }
  void assign(SimParticle const& p, U64 collectionKey, U64& failures) {
    key = collectionKey; id = p.id().asUint(); stage = p.simStage(); ndaughters = p.daughters().size();
    pdg = static_cast<I32>(p.pdgId()); creation = static_cast<I32>(p.creationCode().id()); stopping = static_cast<I32>(p.stoppingCode().id());
    parent.assign(p.parent(), failures); generator.assign(p.genParticle(), failures);
    x = p.startPosXYZ().x(); y = p.startPosXYZ().y(); z = p.startPosXYZ().z();
    end_x = p.endPosXYZ().x(); end_y = p.endPosXYZ().y(); end_z = p.endPosXYZ().z();
    px = p.startMomXYZT().x(); py = p.startMomXYZT().y(); pz = p.startMomXYZT().z(); energy = p.startMomXYZT().t();
    end_px = p.endMomXYZT().x(); end_py = p.endMomXYZT().y(); end_pz = p.endMomXYZT().z(); end_energy = p.endMomXYZT().t();
    time = p.startGlobalTime(); proper_time = p.startProperTime(); end_time = p.endGlobalTime(); end_proper_time = p.endProperTime();
    start_volume = p.startVolumeIndex(); end_volume = p.endVolumeIndex(); start_g4status = p.startG4Status(); end_g4status = p.endG4Status();
    end_defined = p.endDefined(); pre_last_ke = p.preLastStepKineticEnergy(); end_ke = p.endKineticEnergy();
    nsteps = p.nSteps(); track_length = p.trackLength(); ion_excitation = p.ion().excitationEnergy; ion_float_level = p.ion().floatLevelBaseIndex;
  }
};
}

class FlashReferenceExporter : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Sequence<art::InputTag> steps{fhicl::Name("steps"), fhicl::Comment("Every listed StepPointMC collection is required")};
    fhicl::Sequence<art::InputTag> requiredParticles{fhicl::Name("requiredParticles"), fhicl::Comment("Required SimParticles; all available collections are exported")};
    fhicl::Sequence<art::InputTag> requiredGenerators{fhicl::Name("requiredGenerators"), fhicl::Comment("Required generators; all available collections are exported")};
    fhicl::Sequence<art::InputTag> statuses{fhicl::Name("statuses"), fhicl::Comment("Required StatusG4 products, possibly empty for source-only reads")};
    fhicl::Sequence<art::InputTag> volumes{fhicl::Name("volumes"), fhicl::Comment("Required SubRun PhysicalVolumeInfoMultiCollections")};
    fhicl::Atom<ULong64_t> ordinalBase{fhicl::Name("ordinalBase"), fhicl::Comment("First source ordinal in the declared input assignment"), 0};
    fhicl::Atom<ULong64_t> expectedEvents{fhicl::Name("expectedEvents"), fhicl::Comment("Positive exact exported event count required for endJob completion"), 0};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit FlashReferenceExporter(Parameters const&);
  void beginJob() override;
  void analyze(art::Event const&) override;
  void endSubRun(art::SubRun const&) override;
  void endJob() override;
 private:
  using U64 = flashref::U64;
  using U32 = flashref::U32;
  using I32 = flashref::I32;
  using F32 = flashref::F32;
  using F64 = flashref::F64;
  void product(art::Provenance const*, U32 kind, U32 scope, U64 size, std::string const& role);
  void checkedFill(TTree* tree) { if (tree->Fill() < 0) throw cet::exception("FlashReferenceExport") << "TTree write failure: " << tree->GetName(); }
  std::vector<art::InputTag> steps_, requiredParticles_, requiredGenerators_, statuses_, volumes_;
  std::set<std::tuple<U32, U32, U32>> eventIDs_;
  flashref::Identity ids_;
  flashref::StepRow step_;
  flashref::ParticleRow particle_;
  flashref::Pointer daughter_;
  TTree *events_{}, *products_{}, *stepsTree_{}, *particles_{}, *daughters_{}, *generators_{}, *statusesTree_{}, *volumesTree_{}, *stages_{}, *schema_{}, *completion_{}, *requirements_{}, *storage_{};
  U64 ordinalBase_{}, expectedEvents_{}, count_{}, nproducts_{}, nparticles_{}, ngenerators_{}, nsteps_{}, ndaughters_{}, nstatuses_{}, ptrFailures_{};
  U32 schemaVersion_{1}, pidBytes_{sizeof(flashref::PIDValue)}, ptrKeyBytes_{sizeof(flashref::PtrKey)}, complete_{0};
  U32 kind_{}, scope_{}, stage_{}, volumeCopy_{}, requirementKind_{};
  U64 productSize_{}, ownerKey_{}, stageSize_{}, subrunProducts_{0};
  std::string role_, branch_, friendly_, label_, instance_, process_, type_, producedType_, volumeName_, materialName_, requirementTag_, storageGroup_, storageType_, exportType_;
  U32 storageBytes_{}, exportBytes_{};
  I32 pdg_{}, genID_{}, status_{}, ntracks_{}, killedStep_{}, killedField_{};
  U32 overflow_{};
  F32 cpu_{}, real_{};
  F64 gx_{}, gy_{}, gz_{}, gpx_{}, gpy_{}, gpz_{}, ge_{}, gt_{}, gpt_{};
};

FlashReferenceExporter::FlashReferenceExporter(Parameters const& c)
    : art::EDAnalyzer(c), steps_(c().steps()), requiredParticles_(c().requiredParticles()),
      requiredGenerators_(c().requiredGenerators()), statuses_(c().statuses()), volumes_(c().volumes()),
      ordinalBase_(c().ordinalBase()), expectedEvents_(c().expectedEvents()) {
  if (expectedEvents_ == 0 || steps_.empty() || requiredParticles_.empty() || requiredGenerators_.empty() || volumes_.empty())
    throw cet::exception("FlashReferenceExport") << "Positive expectedEvents and nonempty source/volume requirements are mandatory";
  consumesMany<SimParticleCollection>(); consumesMany<GenParticleCollection>();
  auto unique = [](auto const& tags) {
    std::set<std::string> seen;
    for (auto const& tag : tags)
      if (tag.label().empty() || !seen.insert(tag.encode()).second)
        throw cet::exception("FlashReferenceExport") << "Empty or repeated configured InputTag";
  };
  for (auto const* tags : {&steps_, &requiredParticles_, &requiredGenerators_, &statuses_, &volumes_}) unique(*tags);
  for (auto const& tag : steps_) consumes<StepPointMCCollection>(tag);
  for (auto const& tag : requiredParticles_) consumes<SimParticleCollection>(tag);
  for (auto const& tag : requiredGenerators_) consumes<GenParticleCollection>(tag);
  for (auto const& tag : statuses_) consumes<StatusG4>(tag);
  for (auto const& tag : volumes_) consumes<PhysicalVolumeInfoMultiCollection, art::InSubRun>(tag);
}

void FlashReferenceExporter::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  auto make = [&](char const* name, bool collection = true) {
    auto* t = tfs->make<TTree>(name, name); ids_.bind(t, collection); return t;
  };
  events_ = make("events", false);
  for (auto const& [n, p] : std::vector<std::pair<char const*, U64*>>{
      {"nproducts", &nproducts_}, {"nparticles", &nparticles_}, {"ngenerators", &ngenerators_}, {"nsteps", &nsteps_},
      {"ndaughters", &ndaughters_}, {"nstatuses", &nstatuses_}, {"ptr_failures", &ptrFailures_}}) events_->Branch(n, p);
  products_ = make("products");
  products_->Branch("kind", &kind_); products_->Branch("scope", &scope_); products_->Branch("size", &productSize_);
  for (auto const& [n, p] : std::vector<std::pair<char const*, std::string*>>{
      {"role", &role_}, {"branch", &branch_}, {"friendly", &friendly_}, {"label", &label_}, {"instance", &instance_},
      {"process", &process_}, {"type", &type_}, {"produced_type", &producedType_}}) products_->Branch(n, p);
  stepsTree_ = make("steps"); step_.bind(stepsTree_);
  particles_ = make("particles"); particle_.bind(particles_);
  daughters_ = make("daughters"); daughters_->Branch("owner_key", &ownerKey_); daughter_.bind(daughters_, "daughter");
  generators_ = make("generators"); generators_->Branch("pdg", &pdg_); generators_->Branch("gen_id", &genID_);
  for (auto const& [n, p] : std::vector<std::pair<char const*, F64*>>{
      {"x", &gx_}, {"y", &gy_}, {"z", &gz_}, {"px", &gpx_}, {"py", &gpy_}, {"pz", &gpz_}, {"energy", &ge_},
      {"time", &gt_}, {"proper_time", &gpt_}}) generators_->Branch(n, p);
  statusesTree_ = make("statuses"); statusesTree_->Branch("status", &status_); statusesTree_->Branch("ntracks", &ntracks_);
  statusesTree_->Branch("overflow", &overflow_); statusesTree_->Branch("killed_step", &killedStep_);
  statusesTree_->Branch("killed_field", &killedField_); statusesTree_->Branch("cpu", &cpu_); statusesTree_->Branch("real", &real_);
  stages_ = make("stages"); stages_->Branch("stage", &stage_); stages_->Branch("size", &stageSize_);
  volumesTree_ = make("volumes"); volumesTree_->Branch("stage", &stage_); volumesTree_->Branch("copy", &volumeCopy_);
  volumesTree_->Branch("name", &volumeName_); volumesTree_->Branch("material", &materialName_);
  schema_ = tfs->make<TTree>("schema", "persistent storage contract");
  schema_->Branch("version", &schemaVersion_); schema_->Branch("product_id_bytes", &pidBytes_); schema_->Branch("ptr_key_bytes", &ptrKeyBytes_);
  checkedFill(schema_);
  completion_ = tfs->make<TTree>("completion", "endJob marker; external closed-file/job audit still required");
  completion_->Branch("complete", &complete_); completion_->Branch("events", &count_);
  completion_->Branch("ordinal_base", &ordinalBase_); completion_->Branch("subrun_products", &subrunProducts_);
  requirements_ = tfs->make<TTree>("requirements", "explicit configured mandatory products and event allowance");
  requirements_->Branch("kind", &requirementKind_); requirements_->Branch("tag", &requirementTag_);
  requirements_->Branch("expected_events", &expectedEvents_); requirements_->Branch("ordinal_base", &ordinalBase_);
  auto requirements = [&](U32 k, auto const& tags) {
    requirementKind_ = k;
    for (auto const& tag : tags) { requirementTag_ = tag.encode(); checkedFill(requirements_); }
  };
  requirements(1, requiredParticles_); requirements(2, requiredGenerators_); requirements(3, steps_);
  requirements(4, statuses_); requirements(5, volumes_);
  storage_ = tfs->make<TTree>("storage", "persistent source-to-export representation groups");
  storage_->Branch("group", &storageGroup_); storage_->Branch("persistent", &storageType_); storage_->Branch("export", &exportType_);
  storage_->Branch("persistent_bytes", &storageBytes_); storage_->Branch("export_bytes", &exportBytes_);
  auto storage = [&](char const* group, char const* source, U32 bytes, char const* dest, U32 outBytes) {
    storageGroup_ = group; storageType_ = source; storageBytes_ = bytes; exportType_ = dest; exportBytes_ = outBytes; checkedFill(storage_);
  };
  storage("positions;step_vectors;step_times;generator_states", "double", sizeof(double), "Float64", sizeof(F64));
  storage("particle_momenta_times_energies_length_excitation;step_deposits_length;status_times", "float", sizeof(float), "Float32", sizeof(F32));
  storage("particle_ion_float_level", "short", sizeof(short), "Int32", sizeof(I32));
  storage("status_overflow;particle_end_defined", "bool", sizeof(bool), "UInt32", sizeof(U32));
  storage("product_id", "art::ProductID::value_type", sizeof(flashref::PIDValue), "UInt64", sizeof(U64));
  storage("ptr_key", "art::Ptr::key_type", sizeof(flashref::PtrKey), "UInt64", sizeof(U64));
  storage("map_key;particle_id", "cet::map_vector_key", sizeof(decltype(std::declval<cet::map_vector_key const&>().asUint())), "UInt64", sizeof(U64));
  storage("step_volume", "StepPointMC::VolumeId_type", sizeof(StepPointMC::VolumeId_type), "UInt64", sizeof(U64));
}

void FlashReferenceExporter::product(art::Provenance const* p, U32 kind, U32 scope, U64 size, std::string const& role) {
  if (!p) throw cet::exception("FlashReferenceExport") << "Missing product provenance";
  ids_.product = static_cast<U64>(p->productID().value()); ids_.index = 0;
  kind_ = kind; scope_ = scope; productSize_ = size; role_ = role;
  branch_ = p->branchName(); friendly_ = p->friendlyClassName(); producedType_ = p->producedClassName();
  label_ = p->moduleLabel(); instance_ = p->productInstanceName(); process_ = p->processName();
  switch (kind) {
    case 1: type_ = "mu2e::SimParticleCollection"; break;
    case 2: type_ = "mu2e::GenParticleCollection"; break;
    case 3: type_ = "mu2e::StepPointMCCollection"; break;
    case 4: type_ = "mu2e::StatusG4"; break;
    case 5: type_ = "mu2e::PhysicalVolumeInfoMultiCollection"; break;
    default: throw cet::exception("FlashReferenceExport") << "Invalid export kind";
  }
  checkedFill(products_);
  if (scope == 0) ++nproducts_; else ++subrunProducts_;
}

void FlashReferenceExporter::analyze(art::Event const& event) {
  if (count_ >= expectedEvents_)
    throw cet::exception("FlashReferenceExport") << "Event allowance exceeded before " << event.id();
  ids_.run = event.run(); ids_.subrun = event.subRun(); ids_.event = event.event();
  if (!eventIDs_.emplace(ids_.run, ids_.subrun, ids_.event).second)
    throw cet::exception("FlashReferenceExport") << "Duplicate EventID: " << event.id();
  if (count_ > std::numeric_limits<U64>::max() - ordinalBase_)
    throw cet::exception("FlashReferenceExport") << "Source ordinal overflow";
  ids_.ordinal = ordinalBase_ + count_;
  nproducts_ = nparticles_ = ngenerators_ = nsteps_ = ndaughters_ = nstatuses_ = ptrFailures_ = 0;
  for (auto const& tag : requiredParticles_) event.getValidHandle<SimParticleCollection>(tag);
  for (auto const& tag : requiredGenerators_) event.getValidHandle<GenParticleCollection>(tag);
  std::set<U64> seen;
  for (auto const& h : event.getMany<SimParticleCollection>()) {
    if (!h.isValid() || !seen.insert(static_cast<U64>(h.id().value())).second)
      throw cet::exception("FlashReferenceExport") << "Invalid/duplicate SimParticle handle";
    product(h.provenance(), 1, 0, h->size(), "discovered");
    U64 index = 0;
    for (auto const& item : *h) {
      ids_.index = index++; particle_.assign(item.second, item.first.asUint(), ptrFailures_); checkedFill(particles_); ++nparticles_;
      ownerKey_ = item.first.asUint();
      U64 dindex = 0;
      for (auto const& d : item.second.daughters()) {
        ids_.index = dindex++; daughter_.assign(d, ptrFailures_); checkedFill(daughters_); ++ndaughters_;
      }
    }
  }
  seen.clear();
  for (auto const& h : event.getMany<GenParticleCollection>()) {
    if (!h.isValid() || !seen.insert(static_cast<U64>(h.id().value())).second)
      throw cet::exception("FlashReferenceExport") << "Invalid/duplicate GenParticle handle";
    product(h.provenance(), 2, 0, h->size(), "discovered");
    U64 index = 0;
    for (auto const& g : *h) {
      ids_.index = index++; pdg_ = static_cast<I32>(g.pdgId()); genID_ = static_cast<I32>(g.generatorId().id());
      gx_ = g.position().x(); gy_ = g.position().y(); gz_ = g.position().z();
      gpx_ = g.momentum().x(); gpy_ = g.momentum().y(); gpz_ = g.momentum().z(); ge_ = g.momentum().e();
      gt_ = g.time(); gpt_ = g.properTime(); checkedFill(generators_); ++ngenerators_;
    }
  }
  seen.clear();
  for (auto const& tag : steps_) {
    auto h = event.getValidHandle<StepPointMCCollection>(tag);
    if (!seen.insert(static_cast<U64>(h.id().value())).second)
      throw cet::exception("FlashReferenceExport") << "Aliased StepPoint tags: " << tag.encode();
    product(h.provenance(), 3, 0, h->size(), tag.encode());
    U64 index = 0;
    for (auto const& s : *h) {
      ids_.index = index++; step_.assign(s, ptrFailures_); checkedFill(stepsTree_); ++nsteps_;
    }
  }
  seen.clear();
  for (auto const& tag : statuses_) {
    auto h = event.getValidHandle<StatusG4>(tag);
    if (!seen.insert(static_cast<U64>(h.id().value())).second)
      throw cet::exception("FlashReferenceExport") << "Aliased status tags";
    product(h.provenance(), 4, 0, 1, tag.encode());
    status_ = h->status(); ntracks_ = h->nG4Tracks(); overflow_ = h->overflowSimParticles();
    killedStep_ = h->nKilledStepLimit(); killedField_ = h->nKilledByFieldPropagator(); cpu_ = h->cpuTime(); real_ = h->realTime();
    checkedFill(statusesTree_); ++nstatuses_;
  }
  checkedFill(events_); ++count_;
}

void FlashReferenceExporter::endSubRun(art::SubRun const& sr) {
  ids_.run = sr.run(); ids_.subrun = sr.subRun(); ids_.event = 0; ids_.ordinal = 0;
  std::set<U64> seen;
  for (auto const& tag : volumes_) {
    auto h = sr.getValidHandle<PhysicalVolumeInfoMultiCollection>(tag);
    if (!seen.insert(static_cast<U64>(h.id().value())).second)
      throw cet::exception("FlashReferenceExport") << "Aliased volume tags";
    product(h.provenance(), 5, 1, h->size(), tag.encode());
    stage_ = 0;
    for (auto const& collection : *h) {
      ids_.index = stage_; stageSize_ = collection.size(); checkedFill(stages_);
      for (auto const& item : collection) {
        ids_.index = item.first.asUint(); volumeCopy_ = item.second.copyNo();
        volumeName_ = item.second.name(); materialName_ = item.second.materialName(); checkedFill(volumesTree_);
      }
      ++stage_;
    }
  }
}

void FlashReferenceExporter::endJob() {
  if (count_ != expectedEvents_ || subrunProducts_ == 0)
    throw cet::exception("FlashReferenceExport") << "Expected " << expectedEvents_ << " events, got " << count_
        << "; SubRun products=" << subrunProducts_;
  complete_ = 1; checkedFill(completion_);
}
}

DEFINE_ART_MODULE(mu2e::FlashReferenceExporter)
