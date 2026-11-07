#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/IO/ProductMix/MixTypes.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Provenance.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "Offline/CalorimeterGeom/inc/Calorimeter.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/GeometryService/inc/VirtualDetector.hh"
#include "Offline/MCDataProducts/inc/CaloShowerStep.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/PrimaryParticle.hh"
#include "Offline/MCDataProducts/inc/ProcessCode.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StageParticle.hh"
#include "Offline/MCDataProducts/inc/StatusG4.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "TTree.h"
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mu2e {

class CaloLibraryDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<std::string> inputMode{fhicl::Name("inputMode"), "stage"};
    fhicl::Atom<bool> exportSourceEvents{fhicl::Name("exportSourceEvents"), false};
    fhicl::Atom<art::InputTag> sourceEvents{fhicl::Name("sourceEvents"), art::InputTag{}};
    fhicl::Atom<art::InputTag> primaryParticles{fhicl::Name("primaryParticles"), art::InputTag("FindMCPrimary::Primary")};
    fhicl::Atom<art::InputTag> status{fhicl::Name("status"), art::InputTag("g4run::CaloLibraryCE")};
    fhicl::Atom<art::InputTag> generated{fhicl::Name("generated"), art::InputTag("generate::CaloLibraryCE")};
    fhicl::Atom<art::InputTag> transportParticles{fhicl::Name("transportParticles"), art::InputTag("g4run::CaloLibraryCE")};
    fhicl::Atom<art::InputTag> showers{fhicl::Name("showers"), art::InputTag("CaloShowerStepMaker::CaloLibraryCE")};
    fhicl::Sequence<art::InputTag> steps{fhicl::Name("steps")};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit CaloLibraryDumper(Parameters const& config) : art::EDAnalyzer(config), conf_(config()) {
    if (conf_.inputMode() != "stage" && conf_.inputMode() != "primary")
      throw cet::exception("CALOLIBRARY") << "inputMode must be stage or primary\n";
    if (conf_.exportSourceEvents()) {
      auto const tag = conf_.sourceEvents();
      if (tag.label().empty() || tag.process().empty())
        throw cet::exception("CALOLIBRARY") << "sourceEvents requires an explicit label and process\n";
      consumes<art::EventIDSequence>(tag);
    }
  }
  void beginJob() override;
  void analyze(art::Event const& event) override;

 private:
  using Identity = std::pair<std::uint32_t, std::uint64_t>;
  using SimPtr = art::Ptr<SimParticle>;
  struct Link {
    bool valid{false};
    std::uint32_t product_id{0};
    std::uint64_t key{0};
  } sim_, parent_, generator_;
  void ids(TTree* tree);
  void link(TTree* tree, std::string const& prefix, Link& row);
  void vector(TTree* tree, std::string const& prefix, std::array<double, 3>& row);
  void momentum(TTree* tree, std::string const& prefix, std::array<double, 4>& row);
  template <class T> static Link identity(art::Ptr<T> const& ptr) {
    return {ptr.isNonnull(), ptr.id().value(), ptr.isNonnull() ? std::uint64_t(ptr.key()) : 0};
  }
  static std::array<double, 3> xyz(CLHEP::Hep3Vector const& point) { return {point.x(), point.y(), point.z()}; }
  static std::array<double, 4> four(CLHEP::HepLorentzVector const& p) { return {p.px(), p.py(), p.pz(), p.e()}; }
  void requireSim(SimPtr const& ptr, std::set<Identity> const& available);
  void geometry(unsigned run);
  void sourceEvents(art::Event const& event);
  Config conf_;
  TTree *events_{nullptr}, *particles_{nullptr}, *generated_{nullptr}, *steps_{nullptr}, *showers_{nullptr};
  TTree *products_{nullptr}, *genparticles_{nullptr}, *crystals_{nullptr}, *disks_{nullptr}, *vd_{nullptr};
  TTree* primaries_{nullptr};
  TTree* source_events_{nullptr};
  bool source_present_{false}, source_valid_{false};
  std::uint32_t source_product_{0};
  std::uint64_t source_count_{0}, source_index_{0};
  std::uint64_t source_run_{0}, source_subrun_{0}, source_event_{0};
  std::string source_tag_, source_branch_;
  unsigned run_{0}, subrun_{0}, event_{0}, geometry_run_{0};
  int input_mode_{0};
  std::uint32_t transport_product_{0}, primary_product_{0};
  std::uint64_t primary_count_{0};
  bool primary_present_{false}, self_parent_{false}, is_ce_endpoint_{false};
  std::uint32_t product_id_{0}, shower_product_{0}, generated_product_{0}, status_product_{0};
  std::uint64_t row_{0}, particle_count_{0}, generated_count_{0}, shower_count_{0}, step_count_{0};
  std::uint64_t unresolved_links_{0}, n_collection_{0};
  int status_{0}, n_tracks_{0}, killed_steps_{0}, killed_field_{0}, kind_{0}, pdg_{0}, creation_{0}, stopping_{0};
  unsigned stage_{0}, start_volume_{0}, end_volume_{0}, start_g4_{0}, end_g4_{0}, n_compress_{0};
  bool overflow_{false}, shower_present_{false}, end_defined_{false}, collection_present_{false};
  bool transport_particles_present_{false}, generated_present_{false};
  float cpu_{0}, wall_{0}, eg4_{0}, ebirks_{0}, pin_{0}, total_edep_{0}, nonion_edep_{0}, visible_edep_{0}, length_{0};
  float start_time_{0}, end_time_{0}, start_proper_{0}, end_proper_{0}, track_length_{0}, end_ke_{0}, prelast_ke_{0};
  int n_steps_{0};
  std::uint64_t volume_{0};
  int crystal_{0};
  double time_{0}, proper_{0};
  std::array<double, 3> position_{}, post_position_{}, p3_{}, post_p3_{}, start_position_{}, end_position_{};
  std::array<double, 4> p4_{}, start_p4_{}, end_p4_{};
  std::string branch_name_;
  unsigned geo_crystal_{0}, geo_disk_{0}, geo_local_{0};
  std::array<double, 3> geo_position_{}, geo_local_position_{}, geo_size_{}, disk_origin_{}, ff_origin_{};
  std::array<double, 3> disk_x_{}, disk_y_{}, disk_z_{};
  std::vector<unsigned> neighbors_, next_neighbors_;
  double feb_offset_{0}, feb_length_{0}, vd_half_length_{0};
  int vd_id_{0};
  std::array<double, 3> vd_position_{}, vd_x_{}, vd_y_{}, vd_z_{};
};

void CaloLibraryDumper::ids(TTree* tree) {
  tree->Branch("run", &run_); tree->Branch("subrun", &subrun_); tree->Branch("event", &event_);
}

void CaloLibraryDumper::link(TTree* tree, std::string const& prefix, Link& row) {
  tree->Branch((prefix + "_valid").c_str(), &row.valid);
  tree->Branch((prefix + "_product_id").c_str(), &row.product_id);
  tree->Branch((prefix + "_key").c_str(), &row.key);
}

void CaloLibraryDumper::vector(TTree* tree, std::string const& prefix, std::array<double, 3>& row) {
  tree->Branch((prefix + "_x").c_str(), &row[0]);
  tree->Branch((prefix + "_y").c_str(), &row[1]);
  tree->Branch((prefix + "_z").c_str(), &row[2]);
}

void CaloLibraryDumper::momentum(TTree* tree, std::string const& prefix, std::array<double, 4>& row) {
  tree->Branch((prefix + "_px").c_str(), &row[0]); tree->Branch((prefix + "_py").c_str(), &row[1]);
  tree->Branch((prefix + "_pz").c_str(), &row[2]); tree->Branch((prefix + "_e").c_str(), &row[3]);
}

void CaloLibraryDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  events_ = tfs->make<TTree>("events", "every retained input event, including failures and empty responses"); ids(events_);
  events_->Branch("g4_status", &status_); events_->Branch("n_g4_tracks", &n_tracks_);
  events_->Branch("overflow", &overflow_); events_->Branch("killed_step_limit", &killed_steps_);
  events_->Branch("killed_field_propagator", &killed_field_);
  events_->Branch("g4_cpu", &cpu_); events_->Branch("g4_wall", &wall_);
  events_->Branch("showers_present", &shower_present_); events_->Branch("unresolved_links", &unresolved_links_);
  events_->Branch("n_particles", &particle_count_); events_->Branch("n_generated", &generated_count_);
  events_->Branch("n_showers", &shower_count_); events_->Branch("n_steps", &step_count_);
  events_->Branch("shower_product_id", &shower_product_); events_->Branch("generated_product_id", &generated_product_);
  events_->Branch("status_product_id", &status_product_);
  events_->Branch("transport_particles_present", &transport_particles_present_);
  events_->Branch("generated_present", &generated_present_);
  events_->Branch("input_mode", &input_mode_);
  events_->Branch("transport_product_id", &transport_product_);
  events_->Branch("primary_present", &primary_present_);
  events_->Branch("primary_product_id", &primary_product_);
  events_->Branch("n_primaries", &primary_count_);
  if (conf_.exportSourceEvents()) {
    source_tag_ = conf_.sourceEvents().encode();
    events_->Branch("source_present", &source_present_);
    events_->Branch("source_product_id", &source_product_);
    events_->Branch("source_count", &source_count_);
    events_->Branch("source_tag", &source_tag_);
    events_->Branch("source_branch", &source_branch_);
    source_events_ = tfs->make<TTree>("source_events", "ordered immediate secondary EventIDs; no source-file or particle-key inference");
    ids(source_events_);
    source_events_->Branch("product_id", &source_product_);
    source_events_->Branch("index", &source_index_);
    source_events_->Branch("source_valid", &source_valid_);
    source_events_->Branch("source_run", &source_run_);
    source_events_->Branch("source_subrun", &source_subrun_);
    source_events_->Branch("source_event", &source_event_);
  }
  products_ = tfs->make<TTree>("products", "per-event product registry and collection row counts"); ids(products_);
  products_->Branch("product_id", &product_id_); products_->Branch("branch_name", &branch_name_);
  products_->Branch("kind", &kind_); products_->Branch("size", &n_collection_);
  products_->Branch("present", &collection_present_);
  particles_ = tfs->make<TTree>("particles", "all event SimParticle physics states, including ancestors"); ids(particles_);
  link(particles_, "sim", sim_); link(particles_, "parent", parent_); link(particles_, "generator", generator_);
  particles_->Branch("pdg", &pdg_); particles_->Branch("creation", &creation_); particles_->Branch("stopping", &stopping_);
  particles_->Branch("stage", &stage_); particles_->Branch("end_defined", &end_defined_);
  particles_->Branch("self_parent", &self_parent_); particles_->Branch("is_ce_endpoint", &is_ce_endpoint_);
  vector(particles_, "start", start_position_); vector(particles_, "end", end_position_);
  momentum(particles_, "start", start_p4_); momentum(particles_, "end", end_p4_);
  particles_->Branch("start_time", &start_time_); particles_->Branch("end_time", &end_time_);
  particles_->Branch("start_proper_time", &start_proper_); particles_->Branch("end_proper_time", &end_proper_);
  particles_->Branch("start_volume", &start_volume_); particles_->Branch("end_volume", &end_volume_);
  particles_->Branch("start_g4_status", &start_g4_); particles_->Branch("end_g4_status", &end_g4_);
  particles_->Branch("n_steps", &n_steps_); particles_->Branch("track_length", &track_length_);
  particles_->Branch("end_ke", &end_ke_); particles_->Branch("prelast_ke", &prelast_ke_);
  primaries_ = tfs->make<TTree>("primaries", "ordered actual PrimaryParticle SimParticle pointers"); ids(primaries_);
  primaries_->Branch("product_id", &product_id_); primaries_->Branch("index", &row_); link(primaries_, "sim", sim_);
  generated_ = tfs->make<TTree>("generated", "ordered generated StageParticles"); ids(generated_);
  generated_->Branch("product_id", &product_id_); generated_->Branch("index", &row_);
  generated_->Branch("pdg", &pdg_); generated_->Branch("creation", &creation_); link(generated_, "parent", parent_);
  vector(generated_, "position", position_); momentum(generated_, "momentum", p4_); generated_->Branch("time", &time_);
  genparticles_ = tfs->make<TTree>("genparticles", "all event GenParticle states referenced by SimParticles"); ids(genparticles_);
  genparticles_->Branch("product_id", &product_id_); genparticles_->Branch("index", &row_);
  genparticles_->Branch("pdg", &pdg_); genparticles_->Branch("generator_id", &creation_);
  vector(genparticles_, "position", position_); momentum(genparticles_, "momentum", p4_); genparticles_->Branch("time", &time_);
  steps_ = tfs->make<TTree>("steps", "ordered raw StepPointMCs, kind selected by configured collection order"); ids(steps_);
  steps_->Branch("product_id", &product_id_); steps_->Branch("kind", &kind_); steps_->Branch("index", &row_);
  steps_->Branch("volume", &volume_); link(steps_, "sim", sim_);
  vector(steps_, "position", position_); vector(steps_, "post_position", post_position_);
  vector(steps_, "momentum", p3_); vector(steps_, "post_momentum", post_p3_);
  steps_->Branch("time", &time_); steps_->Branch("proper_time", &proper_);
  steps_->Branch("energy_g4", &total_edep_); steps_->Branch("energy_nonion", &nonion_edep_);
  steps_->Branch("energy_visible", &visible_edep_); steps_->Branch("length", &length_); steps_->Branch("end_process", &stopping_);
  showers_ = tfs->make<TTree>("showers", "ordered compressed CaloShowerSteps, crystal-local position"); ids(showers_);
  showers_->Branch("product_id", &product_id_); showers_->Branch("index", &row_); link(showers_, "sim", sim_);
  showers_->Branch("crystal", &crystal_); showers_->Branch("n_compress", &n_compress_);
  showers_->Branch("time", &time_); showers_->Branch("energy_g4", &eg4_); showers_->Branch("energy_birks", &ebirks_);
  showers_->Branch("momentum_in", &pin_); vector(showers_, "position", position_);
  crystals_ = tfs->make<TTree>("crystals", "geometry-service crystal identities, centers, sizes and neighbors");
  crystals_->Branch("run", &run_); crystals_->Branch("crystal", &geo_crystal_); crystals_->Branch("disk", &geo_disk_);
  crystals_->Branch("local_id", &geo_local_); vector(crystals_, "position", geo_position_);
  vector(crystals_, "local_position", geo_local_position_); vector(crystals_, "size", geo_size_);
  crystals_->Branch("neighbors", &neighbors_); crystals_->Branch("next_neighbors", &next_neighbors_);
  disks_ = tfs->make<TTree>("disks", "geometry-service disk frame and envelope, not crystal-front boundary");
  disks_->Branch("run", &run_); disks_->Branch("disk", &geo_disk_); vector(disks_, "origin", disk_origin_);
  vector(disks_, "front_face_origin", ff_origin_); vector(disks_, "size", geo_size_);
  vector(disks_, "axis_x", disk_x_); vector(disks_, "axis_y", disk_y_); vector(disks_, "axis_z", disk_z_);
  disks_->Branch("feb_z_offset", &feb_offset_); disks_->Branch("feb_z_length", &feb_length_);
  vd_ = tfs->make<TTree>("virtualdetectors", "calorimeter VD centers and frames; shell shapes require geometry audit");
  vd_->Branch("run", &run_); vd_->Branch("id", &vd_id_); vd_->Branch("half_length", &vd_half_length_);
  vector(vd_, "position", vd_position_); vector(vd_, "axis_x", vd_x_); vector(vd_, "axis_y", vd_y_); vector(vd_, "axis_z", vd_z_);
}

void CaloLibraryDumper::requireSim(SimPtr const& ptr, std::set<Identity> const& available) {
  if (!ptr.isNonnull() || available.count({ptr.id().value(), std::uint64_t(ptr.key())}) == 0) {
    ++unresolved_links_;
    if (status_ == 0) throw cet::exception("CALOLIBRARY") << "unresolved SimParticle identity in completed transport\n";
  }
}

void CaloLibraryDumper::geometry(unsigned run) {
  if (geometry_run_ == run) return;
  geometry_run_ = run;
  auto const& cal = *GeomHandle<Calorimeter>();
  for (unsigned crystal = 0; crystal < cal.nCrystals(); ++crystal) {
    auto const& c = cal.crystal(crystal);
    geo_crystal_ = c.ID(); geo_disk_ = c.diskID(); geo_local_ = c.localID();
    geo_position_ = xyz(c.position()); geo_local_position_ = xyz(c.localPosition()); geo_size_ = xyz(c.size());
    neighbors_ = c.neighbors(); next_neighbors_ = c.nextNeighbors(); crystals_->Fill();
  }
  for (unsigned disk = 0; disk < cal.nDisks(); ++disk) {
    auto const& info = cal.disk(disk).diskInfo();
    geo_disk_ = disk; disk_origin_ = xyz(cal.diskToMu2e(disk, {0, 0, 0}));
    ff_origin_ = xyz(cal.diskFFToMu2e(disk, {0, 0, 0})); geo_size_ = xyz(info.size());
    auto const origin = cal.diskToMu2e(disk, {0, 0, 0});
    disk_x_ = xyz(cal.diskToMu2e(disk, {1, 0, 0}) - origin);
    disk_y_ = xyz(cal.diskToMu2e(disk, {0, 1, 0}) - origin);
    disk_z_ = xyz(cal.diskToMu2e(disk, {0, 0, 1}) - origin);
    feb_offset_ = info.FEBZOffset(); feb_length_ = info.FEBZLength(); disks_->Fill();
  }
  auto const& vd = *GeomHandle<VirtualDetector>();
  for (int id : {73, 74, 75, 76, 77, 78, 79, 80, 102, 103, 104, 105, 106, 107, 108, 109}) {
    if (!vd.exist(id)) continue;
    vd_id_ = id; vd_half_length_ = vd.getHalfLength(); vd_position_ = xyz(vd.getGlobal(id));
    auto const* rotation = vd.getRotation(id);
    vd_x_ = xyz(rotation ? rotation->inverse() * CLHEP::Hep3Vector(1, 0, 0) : CLHEP::Hep3Vector(1, 0, 0));
    vd_y_ = xyz(rotation ? rotation->inverse() * CLHEP::Hep3Vector(0, 1, 0) : CLHEP::Hep3Vector(0, 1, 0));
    vd_z_ = xyz(rotation ? rotation->inverse() * CLHEP::Hep3Vector(0, 0, 1) : CLHEP::Hep3Vector(0, 0, 1)); vd_->Fill();
  }
}

void CaloLibraryDumper::sourceEvents(art::Event const& event) {
  source_present_ = false; source_product_ = 0; source_count_ = 0; source_branch_.clear();
  source_index_ = 0; source_valid_ = false; source_run_ = source_subrun_ = source_event_ = 0;
  if (!conf_.exportSourceEvents()) return;
  art::Handle<art::EventIDSequence> handle;
  if (!event.getByLabel(conf_.sourceEvents(), handle))
    throw cet::exception("CALOLIBRARY") << "missing required source EventIDSequence " << conf_.sourceEvents() << '\n';
  source_present_ = true; source_product_ = handle.id().value(); source_count_ = handle->size();
  source_branch_ = handle.provenance()->productDescription().branchName();
  product_id_ = source_product_; kind_ = -6; n_collection_ = source_count_; collection_present_ = true;
  branch_name_ = source_branch_; products_->Fill();
  for (auto const& id : *handle) {
    source_valid_ = id.isValid();
    source_run_ = id.run(); source_subrun_ = id.subRun(); source_event_ = id.event();
    source_events_->Fill(); ++source_index_;
  }
}

void CaloLibraryDumper::analyze(art::Event const& event) {
  run_ = event.run(); subrun_ = event.subRun(); event_ = event.event(); geometry(run_);
  auto const stat = event.getValidHandle<StatusG4>(conf_.status());
  status_product_ = stat.id().value(); status_ = stat->status(); n_tracks_ = stat->nG4Tracks();
  overflow_ = stat->overflowSimParticles(); killed_steps_ = stat->nKilledStepLimit();
  killed_field_ = stat->nKilledByFieldPropagator(); cpu_ = stat->cpuTime(); wall_ = stat->realTime();
  unresolved_links_ = particle_count_ = generated_count_ = shower_count_ = step_count_ = 0;
  input_mode_ = conf_.inputMode() == "primary" ? 1 : 0;
  transport_product_ = primary_product_ = generated_product_ = 0;
  primary_count_ = 0; primary_present_ = false; generated_present_ = false;
  sourceEvents(event);
  art::Handle<SimParticleCollection> transport;
  transport_particles_present_ = event.getByLabel(conf_.transportParticles(), transport);
  transport_product_ = transport_particles_present_ ? transport.id().value() : 0;
  if (!transport_particles_present_ && status_ == 0) throw cet::exception("CALOLIBRARY") << "missing transport SimParticles\n";
  auto const collections = event.getMany<SimParticleCollection>();
  auto const genCollections = event.getMany<GenParticleCollection>();
  std::set<Identity> genAvailable;
  for (auto const& handle : genCollections) {
    product_id_ = handle.id().value(); kind_ = -4; n_collection_ = handle->size(); collection_present_ = true;
    branch_name_ = handle.provenance()->productDescription().branchName(); products_->Fill(); row_ = 0;
    for (auto const& g : *handle) {
      genAvailable.insert({product_id_, row_}); pdg_ = g.pdgId(); creation_ = g.generatorId().id();
      position_ = xyz(g.position()); p4_ = four(g.momentum()); time_ = g.time(); genparticles_->Fill(); ++row_;
    }
  }
  std::set<Identity> available;
  for (auto const& handle : collections) {
    for (auto const& item : *handle) available.insert({handle.id().value(), item.first.asUint()});
  }
  for (auto const& handle : collections) {
    product_id_ = handle.id().value(); kind_ = -1; n_collection_ = handle->size(); collection_present_ = true;
    branch_name_ = handle.provenance()->productDescription().branchName(); products_->Fill();
    for (auto const& item : *handle) {
      auto const& p = item.second;
      sim_ = {true, product_id_, item.first.asUint()}; parent_ = identity(p.parent()); generator_ = identity(p.genParticle());
      if (p.parent().isNonnull()) requireSim(p.parent(), available);
      if (generator_.valid && genAvailable.count({generator_.product_id, generator_.key}) == 0) {
        ++unresolved_links_;
        if (status_ == 0) throw cet::exception("CALOLIBRARY") << "unresolved generator identity\n";
      }
      pdg_ = p.pdgId(); stage_ = p.simStage(); creation_ = p.creationCode().id(); stopping_ = p.stoppingCode().id();
      self_parent_ = p.selfParent(); is_ce_endpoint_ = p.creationCode() == ProcessCode::mu2eCeMinusEndpoint;
      end_defined_ = p.endDefined(); start_position_ = xyz(p.startPosition()); end_position_ = xyz(p.endPosition());
      start_p4_ = four(p.startMomentum()); end_p4_ = four(p.endMomentum());
      start_time_ = p.startGlobalTime(); end_time_ = p.endGlobalTime();
      start_proper_ = p.startProperTime(); end_proper_ = p.endProperTime();
      start_volume_ = p.startVolumeIndex(); end_volume_ = p.endVolumeIndex();
      start_g4_ = p.startG4Status(); end_g4_ = p.endG4Status(); n_steps_ = p.nSteps(); track_length_ = p.trackLength();
      end_ke_ = p.endKineticEnergy(); prelast_ke_ = p.preLastStepKineticEnergy(); particles_->Fill(); ++particle_count_;
    }
  }
  if (conf_.inputMode() == "stage") {
    art::Handle<StageParticleCollection> generated;
    generated_present_ = event.getByLabel(conf_.generated(), generated);
    if (!generated_present_ && status_ == 0) throw cet::exception("CALOLIBRARY") << "missing generated StageParticles\n";
    generated_product_ = generated_present_ ? generated.id().value() : 0;
    if (generated_present_) {
      product_id_ = generated_product_; kind_ = -2; n_collection_ = generated->size(); collection_present_ = true;
      branch_name_ = generated.provenance()->productDescription().branchName(); products_->Fill(); row_ = 0;
      for (auto const& g : *generated) {
        parent_ = identity(g.parent()); if (g.parent().isNonnull()) requireSim(g.parent(), available);
        pdg_ = g.pdgId(); creation_ = g.creationCode().id(); position_ = xyz(g.position()); p4_ = four(g.momentum()); time_ = g.time();
        generated_->Fill(); ++row_; ++generated_count_;
      }
    }
  } else {
    art::Handle<PrimaryParticle> primary;
    primary_present_ = event.getByLabel(conf_.primaryParticles(), primary);
    if (!primary_present_ && status_ == 0)
      throw cet::exception("CALOLIBRARY") << "missing PrimaryParticle after completed transport\n";
    primary_product_ = primary_present_ ? primary.id().value() : 0;
    if (primary_present_) {
      auto const& pointers = primary->primarySimParticles();
      product_id_ = primary_product_; kind_ = -5; n_collection_ = pointers.size(); collection_present_ = true;
      branch_name_ = primary.provenance()->productDescription().branchName(); products_->Fill(); row_ = 0;
      for (auto const& ptr : pointers) {
        requireSim(ptr, available);
        if (status_ == 0 && ptr.id() != transport.id())
          throw cet::exception("CALOLIBRARY") << "primary pointer targets a different transport product\n";
        sim_ = identity(ptr); primaries_->Fill(); ++row_; ++primary_count_;
      }
    }
  }
  kind_ = 0;
  for (auto const& tag : conf_.steps()) {
    art::Handle<StepPointMCCollection> handle;
    collection_present_ = event.getByLabel(tag, handle);
    if (!collection_present_) {
      if (status_ == 0) throw cet::exception("CALOLIBRARY") << "missing raw step collection " << tag << '\n';
      product_id_ = 0; n_collection_ = 0; branch_name_ = tag.encode(); products_->Fill(); ++kind_; continue;
    }
    product_id_ = handle.id().value(); branch_name_ = handle.provenance()->productDescription().branchName();
    n_collection_ = handle->size(); products_->Fill(); row_ = 0;
    for (auto const& s : *handle) {
      requireSim(s.simParticle(), available); sim_ = identity(s.simParticle()); volume_ = s.volumeId();
      position_ = xyz(s.position()); post_position_ = xyz(s.postPosition());
      p3_ = xyz(s.momentum()); post_p3_ = xyz(s.postMomentum()); time_ = s.time(); proper_ = s.properTime();
      total_edep_ = s.totalEDep(); nonion_edep_ = s.nonIonizingEDep(); visible_edep_ = s.visibleEDep();
      length_ = s.stepLength(); stopping_ = s.endProcessCode().id(); steps_->Fill(); ++row_; ++step_count_;
    }
    ++kind_;
  }
  art::Handle<CaloShowerStepCollection> showers;
  shower_present_ = event.getByLabel(conf_.showers(), showers); shower_product_ = shower_present_ ? showers.id().value() : 0;
  if (!shower_present_ && status_ == 0) throw cet::exception("CALOLIBRARY") << "missing showers after completed transport\n";
  if (shower_present_) {
    product_id_ = showers.id().value(); branch_name_ = showers.provenance()->productDescription().branchName();
    kind_ = -3; n_collection_ = showers->size(); collection_present_ = true; products_->Fill(); row_ = 0;
    for (auto const& s : *showers) {
      requireSim(s.simParticle(), available); sim_ = identity(s.simParticle()); crystal_ = s.volumeG4ID();
      auto const& cal = *GeomHandle<Calorimeter>();
      if (status_ == 0 && (crystal_ < 0 || unsigned(crystal_) >= cal.nCrystals()))
        throw cet::exception("CALOLIBRARY") << "invalid shower crystal\n";
      n_compress_ = s.nCompress(); position_ = xyz(s.position()); time_ = s.time();
      eg4_ = s.energyDepG4(); ebirks_ = s.energyDepBirks(); pin_ = s.momentumIn(); showers_->Fill(); ++row_; ++shower_count_;
    }
  }
  events_->Fill();
}

} // namespace mu2e

DEFINE_ART_MODULE(mu2e::CaloLibraryDumper)
