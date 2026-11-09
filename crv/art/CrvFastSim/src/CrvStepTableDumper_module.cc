// Dump CrvStep, SimParticle and GenParticle collections to flat ROOT TTrees: the reference table for mu2e-crv-fastsim.
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/SubRun.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"
#include "cetlib_except/exception.h"

#include "Offline/MCDataProducts/inc/CosmicLivetime.hh"
#include "Offline/MCDataProducts/inc/CrvStep.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"

namespace mu2e {

class CrvStepTableDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<art::InputTag> crvSteps{fhicl::Name("crvSteps"), art::InputTag("compressDetStepMCs")};
    fhicl::Atom<art::InputTag> simParticles{fhicl::Name("simParticles"), art::InputTag("compressDetStepMCs")};
    fhicl::Atom<art::InputTag> genParticles{fhicl::Name("genParticles"), art::InputTag("compressDetStepMCs")};
    fhicl::Atom<art::InputTag> livetime{fhicl::Name("livetime"), art::InputTag("generate")};
    fhicl::Atom<bool> writeSteps{fhicl::Name("writeSteps"), true};
    fhicl::Atom<bool> fullGeneratorStream{fhicl::Name("fullGeneratorStream"), false};
    fhicl::Atom<std::string> associationPolicy{fhicl::Name("associationPolicy"), "retained_primary"};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit CrvStepTableDumper(const Parameters& conf);
  void beginJob() override;
  void analyze(const art::Event& event) override;
  void endSubRun(const art::SubRun& subrun) override;

 private:
  Config conf_;
  TTree *steps_{nullptr}, *parts_{nullptr}, *gens_{nullptr}, *subruns_{nullptr}, *events_{nullptr};
  unsigned schemaVersion_{2}, checked_{0}, fullGen_{0}, genProduct_{0}, simProduct_{0};
  unsigned particleGenProduct_{0}, parentProduct_{0}, stepProduct_{0};
  long long nGen_{0}, nParticles_{0}, nSteps_{0};
  std::string genTag_, simTag_, stepTag_, associationPolicy_;
  void validate(const art::Event&, const GenParticleCollection&, art::ProductID,
                const SimParticleCollection&, art::ProductID, const CrvStepCollection&) const;
  // shared event id
  unsigned run_{0}, subrun_{0}, event_{0};
  // All-event census; nsteps=-1 means step extraction was disabled.
  unsigned long long ngen_{0};
  long long nsteps_{-1};
  // CrvStep row
  int bar_{0}, sp_{0}, nstep_{0};
  float edep_{0}, x0_{0}, y0_{0}, z0_{0}, x1_{0}, y1_{0}, z1_{0}, px0_{0}, py0_{0}, pz0_{0}, p1_{0}, len_{0};
  double t0_{0}, t1_{0};
  // SimParticle row
  int key_{0}, parent_{0}, pdg_{0}, ccode_{0}, scode_{0}, isprim_{0}, genidx_{0};
  double sx_{0}, sy_{0}, sz_{0}, ex_{0}, ey_{0}, ez_{0};
  float spx_{0}, spy_{0}, spz_{0}, se_{0}, epx_{0}, epy_{0}, epz_{0}, ee_{0}, st_{0}, et_{0};
  // GenParticle row
  int gidx_{0}, gpdg_{0}, ggen_{0};
  double gx_{0}, gy_{0}, gz_{0}, gpx_{0}, gpy_{0}, gpz_{0}, ge_{0}, gt_{0};
  // subrun row
  double livetime_{0};
  unsigned long long primaries_{0};
};

CrvStepTableDumper::CrvStepTableDumper(const Parameters& conf) : art::EDAnalyzer(conf), conf_(conf()) {
  if (conf_.writeSteps() && conf_.fullGeneratorStream())
    throw cet::exception("CrvStepTableDumper") << "fullGeneratorStream requires writeSteps:false\n";
  associationPolicy_ = conf_.associationPolicy();
  if (associationPolicy_ != "retained_primary" && associationPolicy_ != "staged_cat")
    throw cet::exception("CrvStepTableDumper") << "unknown associationPolicy " << associationPolicy_ << '\n';
  if (associationPolicy_ == "staged_cat" && (!conf_.writeSteps() || conf_.fullGeneratorStream()))
    throw cet::exception("CrvStepTableDumper") << "staged_cat requires step extraction, not a full generator stream\n";
  if (associationPolicy_ == "staged_cat") schemaVersion_ = 3;
  fullGen_ = conf_.fullGeneratorStream();
  genTag_ = conf_.genParticles().encode();
  simTag_ = conf_.simParticles().encode();
  stepTag_ = conf_.crvSteps().encode();
}

void CrvStepTableDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  auto ids = [this](TTree* t) {
    t->Branch("run", &run_);
    t->Branch("subrun", &subrun_);
    t->Branch("event", &event_);
  };
  events_ = tfs->make<TTree>("events", "Dump census and validated association schema");
  ids(events_);
  for (auto [n, v] : std::vector<std::pair<const char*, unsigned*>>{
           {"schema_version", &schemaVersion_}, {"associations_checked", &checked_},
           {"full_generator_stream", &fullGen_}, {"gen_product_id", &genProduct_}, {"sim_product_id", &simProduct_}})
    events_->Branch(n, v);
  events_->Branch("association_policy", &associationPolicy_);
  events_->Branch("ngen", &ngen_);     // census v1 (veto exposure tools)
  events_->Branch("nsteps", &nsteps_);
  events_->Branch("n_gen", &nGen_);
  events_->Branch("n_particles", &nParticles_);
  events_->Branch("n_steps", &nSteps_);
  events_->Branch("gen_tag", &genTag_);
  events_->Branch("sim_tag", &simTag_);
  events_->Branch("step_tag", &stepTag_);
  gens_ = tfs->make<TTree>("gen", "Selected GenParticle collection; stream role in events");
  ids(gens_);
  gens_->Branch("gen_product_id", &genProduct_);
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{{"idx", &gidx_}, {"pdg", &gpdg_}, {"genid", &ggen_}})
    gens_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, double*>>{
           {"x", &gx_}, {"y", &gy_}, {"z", &gz_}, {"px", &gpx_}, {"py", &gpy_}, {"pz", &gpz_}, {"e", &ge_}, {"t", &gt_}})
    gens_->Branch(n, v);
  subruns_ = tfs->make<TTree>("subruns", "CosmicLivetime per subrun");
  subruns_->Branch("run", &run_);
  subruns_->Branch("subrun", &subrun_);
  subruns_->Branch("livetime", &livetime_);
  subruns_->Branch("primaries", &primaries_);
  if (!conf_.writeSteps()) return;
  steps_ = tfs->make<TTree>("steps", "CrvSteps");
  ids(steps_);
  steps_->Branch("nstep", &nstep_);
  steps_->Branch("bar", &bar_);
  steps_->Branch("sp", &sp_);
  steps_->Branch("sp_product_id", &stepProduct_);
  steps_->Branch("t0", &t0_);
  steps_->Branch("t1", &t1_);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{
           {"edep", &edep_}, {"x0", &x0_}, {"y0", &y0_}, {"z0", &z0_}, {"x1", &x1_}, {"y1", &y1_}, {"z1", &z1_},
           {"px0", &px0_}, {"py0", &py0_}, {"pz0", &pz0_}, {"p1", &p1_}, {"len", &len_}})
    steps_->Branch(n, v);
  parts_ = tfs->make<TTree>("particles", "SimParticles");
  ids(parts_);
  parts_->Branch("sim_product_id", &simProduct_);
  parts_->Branch("parent_product_id", &parentProduct_);
  parts_->Branch("gen_product_id", &particleGenProduct_);
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{{"key", &key_}, {"parent", &parent_}, {"pdg", &pdg_},
                                                               {"ccode", &ccode_}, {"scode", &scode_},
                                                               {"isprim", &isprim_}, {"genidx", &genidx_}})
    parts_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, double*>>{
           {"sx", &sx_}, {"sy", &sy_}, {"sz", &sz_}, {"ex", &ex_}, {"ey", &ey_}, {"ez", &ez_}})
    parts_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"spx", &spx_}, {"spy", &spy_}, {"spz", &spz_},
                                                                 {"se", &se_}, {"epx", &epx_}, {"epy", &epy_},
                                                                 {"epz", &epz_}, {"ee", &ee_}, {"st", &st_},
                                                                 {"et", &et_}})
    parts_->Branch(n, v);
}

void CrvStepTableDumper::validate(const art::Event& event, const GenParticleCollection& gen, art::ProductID gid,
                                  const SimParticleCollection& parts, art::ProductID pid,
                                  const CrvStepCollection& steps) const {
  std::set<unsigned long> keys;
  for (auto const& [k, p] : parts) {
    if (k.asUint() > static_cast<unsigned>(std::numeric_limits<int>::max()))
      throw cet::exception("CrvStepTableDumper") << event.id() << " SimParticle key exceeds flat schema\n";
    keys.insert(k.asUint());
  }
  auto simptr = [&](const art::Ptr<SimParticle>& ptr, const char* label) {
    if (ptr.isNull() || ptr.id() != pid || !keys.count(ptr.key()) || !ptr.isAvailable())
      throw cet::exception("CrvStepTableDumper") << event.id() << " invalid " << label << " pointer " << ptr
          << "; selected SimParticles " << conf_.simParticles().encode() << " ProductID " << pid << '\n';
  };
  for (auto const& step : steps) simptr(step.simParticle(), "step");
  for (auto const& [k, p] : parts) {
    if (p.parent().isNonnull()) simptr(p.parent(), "parent");
    auto const& gp = p.genParticle();
    if (gp.isNull()) continue;
    if (gp.id() != gid || gp.key() >= gen.size() || !gp.isAvailable())
      throw cet::exception("CrvStepTableDumper") << event.id() << " SimParticle " << k.asUint()
          << " references GenParticle " << gp << "; selected " << conf_.genParticles().encode()
          << " ProductID " << gid << " size " << gen.size() << '\n';
    auto const& g = gen.at(gp.key());
    if (gp->pdgId() != g.pdgId() ||
        (associationPolicy_ == "retained_primary" && p.isPrimary() && p.pdgId() != g.pdgId()))
      throw cet::exception("CrvStepTableDumper") << event.id() << " SimParticle " << k.asUint()
          << " primary/gen signed PDG mismatch for " << gp << '\n';
    if (!std::isfinite(g.momentum().e()) || !std::isfinite(g.time()))
      throw cet::exception("CrvStepTableDumper") << event.id() << " nonfinite generator energy/time for " << gp << '\n';
  }
}

void CrvStepTableDumper::analyze(const art::Event& event) {
  run_ = event.run();
  subrun_ = event.subRun();
  event_ = event.event();
  auto gh = event.getValidHandle<GenParticleCollection>(conf_.genParticles());
  ngen_ = gh->size();
  nsteps_ = -1;
  if (gh->size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw cet::exception("CrvStepTableDumper") << event.id() << " GenParticle count exceeds flat schema\n";
  genProduct_ = gh.id().value();
  simProduct_ = checked_ = 0;
  nGen_ = gh->size();
  nParticles_ = nSteps_ = 0;
  art::Handle<CrvStepCollection> sh;
  art::Handle<SimParticleCollection> ph;
  if (conf_.writeSteps()) {
    event.getByLabel(conf_.crvSteps(), sh);
    event.getByLabel(conf_.simParticles(), ph);
    if (!sh.isValid() || !ph.isValid())
      throw cet::exception("CrvStepTableDumper") << event.id() << " missing steps or SimParticles\n";
    if (sh->size() > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw cet::exception("CrvStepTableDumper") << event.id() << " step count exceeds flat schema\n";
    validate(event, *gh, gh.id(), *ph, ph.id(), *sh);
    simProduct_ = ph.id().value();
    nParticles_ = ph->size();
    nSteps_ = sh->size();
    checked_ = 1;
  }
  for (size_t i = 0; i < gh->size(); ++i) {
    auto const& g = gh->at(i);
    gidx_ = i;
    gpdg_ = g.pdgId();
    ggen_ = g.generatorId().id();
    gx_ = g.position().x();
    gy_ = g.position().y();
    gz_ = g.position().z();
    gpx_ = g.momentum().px();
    gpy_ = g.momentum().py();
    gpz_ = g.momentum().pz();
    ge_ = g.momentum().e();
    gt_ = g.time();
    gens_->Fill();
  }
  if (!conf_.writeSteps()) {
    events_->Fill();
    return;
  }
  nstep_ = sh->size();
  nsteps_ = sh->size();
  for (auto const& s : *sh) {
    bar_ = s.barIndex().asInt();
    sp_ = int(s.simParticle().key());
    stepProduct_ = s.simParticle().id().value();
    edep_ = s.visibleEDep();
    t0_ = s.startTime();
    t1_ = s.endTime();
    x0_ = s.startPos().x();
    y0_ = s.startPos().y();
    z0_ = s.startPos().z();
    x1_ = s.endPos().x();
    y1_ = s.endPos().y();
    z1_ = s.endPos().z();
    px0_ = s.startMom().x();
    py0_ = s.startMom().y();
    pz0_ = s.startMom().z();
    p1_ = s.endMom();
    len_ = s.pathLength();
    steps_->Fill();
  }
  for (auto const& [k, p] : *ph) {
    key_ = int(k.asUint());
    parent_ = p.parent().isNonnull() ? int(p.parent().key()) : -1;
    pdg_ = p.pdgId();
    ccode_ = p.creationCode().id();
    scode_ = p.stoppingCode().id();
    isprim_ = p.isPrimary();
    genidx_ = p.genParticle().isNonnull() ? int(p.genParticle().key()) : -1;
    particleGenProduct_ = p.genParticle().isNonnull() ? p.genParticle().id().value() : 0;
    parentProduct_ = p.parent().isNonnull() ? p.parent().id().value() : 0;
    sx_ = p.startPosition().x();
    sy_ = p.startPosition().y();
    sz_ = p.startPosition().z();
    ex_ = p.endPosition().x();
    ey_ = p.endPosition().y();
    ez_ = p.endPosition().z();
    spx_ = p.startMomentum().px();
    spy_ = p.startMomentum().py();
    spz_ = p.startMomentum().pz();
    se_ = p.startMomentum().e();
    epx_ = p.endMomentum().px();
    epy_ = p.endMomentum().py();
    epz_ = p.endMomentum().pz();
    ee_ = p.endMomentum().e();
    st_ = p.startGlobalTime();
    et_ = p.endGlobalTime();
    parts_->Fill();
  }
  events_->Fill();
}

void CrvStepTableDumper::endSubRun(const art::SubRun& subrun) {
  run_ = subrun.run();
  subrun_ = subrun.subRun();
  art::Handle<CosmicLivetime> lh;
  subrun.getByLabel(conf_.livetime(), lh);
  livetime_ = lh.isValid() ? lh->liveTime() : -1;
  primaries_ = lh.isValid() ? lh->primaries() : 0;
  subruns_->Fill();
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvStepTableDumper)
