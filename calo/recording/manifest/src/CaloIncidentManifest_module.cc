// REVIEW SOURCE ONLY: explicit StageParticle input, never response-derived.
#include "art/Framework/Core/EDFilter.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Provenance.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "Offline/MCDataProducts/inc/StageParticle.hh"
#include "ManifestMetadata.hh"
#include <array>
#include <cstdint>
namespace mu2e {
class CaloIncidentManifest : public art::EDFilter {
 public:
  explicit CaloIncidentManifest(fhicl::ParameterSet const& p)
    : art::EDFilter(p), metadata_(p), tag_(p.get<std::string>("incidentTag")),
      mode_(p.get<std::string>("rootMode")) {
    if (tag_.label().empty() || tag_.process().empty())
      throw cet::exception("MANIFEST_CONFIG") << "incidentTag requires explicit module and process";
    if (mode_ != "record_all" && mode_ != "require_one")
      throw cet::exception("MANIFEST_CONFIG") << "rootMode must be record_all or require_one";
    consumes<StageParticleCollection>(tag_);
    tagText_ = tag_.encode();
  }
  void beginJob() override;
  bool filter(art::Event&) override;
 private:
  void ids(TTree* t) {
    t->Branch("ordinal", &ordinal_); t->Branch("run", &run_);
    t->Branch("subrun", &subrun_); t->Branch("event", &event_);
  }
  manifest_review::Metadata metadata_;
  art::InputTag tag_;
  std::string mode_, tagText_, branchName_, type_{"mu2e::StageParticleCollection"};
  // Native StageParticle coordinates, NOT G4_world. A future adapter must transform.
  std::string coordinateFrame_{"Mu2e"}, positionUnit_{"mm"}, momentumUnit_{"MeV"}, timeUnit_{"ns"};
  // kind=1 reserved for GenParticle; kind=2 is StageParticle. Never conflate IDs.
  int kind_{2}, pdg_{0}, creationCode_{0};
  TTree *events_{nullptr}, *products_{nullptr}, *roots_{nullptr};
  unsigned run_{0}, subrun_{0}, event_{0};
  std::uint64_t ordinal_{0}, count_{0}, index_{0}, parentKey_{0};
  std::uint32_t productId_{0}, parentProductId_{0};
  bool present_{false}, supported_{false}, parentValid_{false};
  double time_{0};
  std::array<double, 3> position_{};
  std::array<double, 4> momentum_{};
};
void CaloIncidentManifest::beginJob() {
  art::ServiceHandle<art::TFileService> fs;
  auto* m = fs->make<TTree>("metadata", "configured claims, not source hash attestation");
  metadata_.branches(m); m->Branch("incident_tag", &tagText_); m->Branch("incident_type", &type_);
  m->Branch("root_mode", &mode_);
  m->Branch("coordinate_frame", &coordinateFrame_); m->Branch("position_unit", &positionUnit_);
  m->Branch("momentum_unit", &momentumUnit_); m->Branch("time_unit", &timeUnit_); m->Fill();
  events_ = fs->make<TTree>("events", "one row per post-generator event, including empty/missing"); ids(events_);
  events_->Branch("present", &present_); events_->Branch("n_roots", &count_);
  events_->Branch("single_root_supported", &supported_); events_->Branch("product_id", &productId_);
  products_ = fs->make<TTree>("products", "designated branch registry, one row per event"); ids(products_);
  products_->Branch("kind", &kind_); products_->Branch("type", &type_);
  products_->Branch("input_tag", &tagText_); products_->Branch("branch_name", &branchName_);
  products_->Branch("product_id", &productId_); products_->Branch("present", &present_);
  products_->Branch("size", &count_);
  roots_ = fs->make<TTree>("roots", "ordered native StageParticle creation states, not SimParticle starts"); ids(roots_);
  roots_->Branch("kind", &kind_); roots_->Branch("product_id", &productId_); roots_->Branch("index", &index_);
  roots_->Branch("pdg", &pdg_); roots_->Branch("creation_code", &creationCode_);
  roots_->Branch("parent_valid", &parentValid_); roots_->Branch("parent_product_id", &parentProductId_);
  roots_->Branch("parent_key", &parentKey_); roots_->Branch("time", &time_);
  for (unsigned i = 0; i < 3; ++i)
    roots_->Branch((std::string("position_") + "xyz"[i]).c_str(), &position_[i]);
  for (unsigned i = 0; i < 4; ++i)
    roots_->Branch((std::string("momentum_") + std::array<char const*,4>{"px","py","pz","e"}[i]).c_str(), &momentum_[i]);
}
bool CaloIncidentManifest::filter(art::Event& e) {
  run_ = e.run(); subrun_ = e.subRun(); event_ = e.event();
  art::Handle<StageParticleCollection> h;
  present_ = e.getByLabel(tag_, h);
  productId_ = present_ ? h.id().value() : 0;
  count_ = present_ ? h->size() : 0;
  branchName_ = present_ ? h.provenance()->productDescription().branchName() : "";
  supported_ = present_ && count_ == 1;
  // Record event and branch even for missing/empty/multiple input. No silent drop.
  events_->Fill(); products_->Fill();
  index_ = 0;
  if (present_) for (auto const& g : *h) {
    pdg_ = g.pdgId(); creationCode_ = g.creationCode().id(); time_ = g.time();
    auto const& p = g.position(); position_ = {p.x(), p.y(), p.z()};
    auto const& q = g.momentum(); momentum_ = {q.px(), q.py(), q.pz(), q.e()};
    auto const parent = g.parent(); parentValid_ = parent.isNonnull();
    parentProductId_ = parentValid_ ? parent.id().value() : 0;
    parentKey_ = parentValid_ ? std::uint64_t(parent.key()) : 0;
    roots_->Fill(); ++index_;
  }
  ++ordinal_;
  // A missing product is not an empty collection. Always fail after attempted rows.
  if (!present_) throw cet::exception("MANIFEST_MISSING") << "designated StageParticle product missing: " << tagText_;
  if (mode_ == "require_one" && !supported_)
    throw cet::exception("MANIFEST_MULTIPLICITY") << "expected exactly one root, got " << count_;
  return true;
}
}
DEFINE_ART_MODULE(mu2e::CaloIncidentManifest)
