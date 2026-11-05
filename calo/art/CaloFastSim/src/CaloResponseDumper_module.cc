#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/CaloConditions/inc/CalSimParams.hh"
#include "Offline/CalorimeterGeom/inc/Calorimeter.hh"
#include "Offline/DAQConditions/inc/EventTiming.hh"
#include "Offline/DataProducts/inc/CrystalId.hh"
#include "Offline/DataProducts/inc/EventWindowMarker.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/GlobalConstantsService/inc/GlobalConstantsHandle.hh"
#include "Offline/GlobalConstantsService/inc/PhysicsParams.hh"
#include "Offline/MCDataProducts/inc/CaloShowerRO.hh"
#include "Offline/MCDataProducts/inc/CaloShowerSim.hh"
#include "Offline/MCDataProducts/inc/CaloShowerStep.hh"
#include "Offline/MCDataProducts/inc/ProtonBunchTimeMC.hh"
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "TTree.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mu2e {

class CaloResponseDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<art::InputTag> steps{fhicl::Name("steps")};
    fhicl::Atom<art::InputTag> readouts{fhicl::Name("readouts")};
    fhicl::Atom<art::InputTag> summaries{fhicl::Name("summaries")};
    fhicl::Atom<art::InputTag> eventWindowMarker{fhicl::Name("eventWindowMarker")};
    fhicl::Atom<art::InputTag> protonBunchTimeMC{fhicl::Name("protonBunchTimeMC")};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit CaloResponseDumper(Parameters const& config) : art::EDAnalyzer(config), conf_(config()) {}
  void beginJob() override;
  void analyze(art::Event const& event) override;

 private:
  void ids(TTree* tree);
  static std::array<unsigned, 4> interval(DbIoV const& iov) {
    return {iov.startRun(), iov.startSubrun(), iov.endRun(), iov.endSubrun()};
  }
  Config conf_;
  ProditionsHandle<CalSimParams> simConditions_;
  ProditionsHandle<EventTiming> timingConditions_;
  TTree *events_{nullptr}, *steps_{nullptr}, *readouts_{nullptr}, *summaries_{nullptr};
  unsigned run_{0}, subrun_{0}, event_{0};
  std::uint32_t stepProduct_{0}, roProduct_{0}, summaryProduct_{0}, ewmProduct_{0}, pbProduct_{0};
  std::uint32_t linkedProduct_{0}, simProduct_{0};
  std::uint64_t index_{0}, linkedIndex_{0}, simKey_{0}, nSteps_{0}, nReadouts_{0}, nSummaries_{0};
  int crystal_{0}, sipm0_{0}, sipm_{0}, spill_{0};
  float energyG4_{0}, energyBirks_{0}, z_{0}, lru_{0}, length_{0}, pbtime_{0}, mbtime_{0}, marker_{0};
  float energy_{0}, summaryTime_{0};
  double time_{0}, eventLength_{0};
  std::array<float, 2> yields_{};
  std::array<unsigned, 4> simIov_{}, timingIov_{};
  std::vector<int> simCids_, timingCids_;
  std::vector<float> peTimes_;
  std::vector<std::uint32_t> memberProducts_;
  std::vector<std::uint64_t> memberIndices_;
};

void CaloResponseDumper::ids(TTree* tree) {
  tree->Branch("run", &run_); tree->Branch("subrun", &subrun_); tree->Branch("event", &event_);
}

void CaloResponseDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  events_ = tfs->make<TTree>("events", "all input events and conditions fetched for this control"); ids(events_);
  events_->Branch("step_product_id", &stepProduct_); events_->Branch("ro_product_id", &roProduct_);
  events_->Branch("summary_product_id", &summaryProduct_); events_->Branch("ewm_product_id", &ewmProduct_);
  events_->Branch("pb_product_id", &pbProduct_); events_->Branch("n_steps", &nSteps_);
  events_->Branch("n_readouts", &nReadouts_); events_->Branch("n_summaries", &nSummaries_);
  events_->Branch("spill", &spill_); events_->Branch("event_length", &eventLength_);
  events_->Branch("pbtime", &pbtime_); events_->Branch("mbtime", &mbtime_); events_->Branch("marker", &marker_);
  events_->Branch("crystal_length", &length_);
  events_->Branch("sim_iov", simIov_.data(), "sim_iov[4]/i");
  events_->Branch("timing_iov", timingIov_.data(), "timing_iov[4]/i");
  events_->Branch("sim_cids", &simCids_); events_->Branch("timing_cids", &timingCids_);
  steps_ = tfs->make<TTree>("steps", "saved compressed inputs and fetched per-crystal conditions, no response recomputation"); ids(steps_);
  steps_->Branch("product_id", &stepProduct_); steps_->Branch("index", &index_);
  steps_->Branch("sim_product_id", &simProduct_); steps_->Branch("sim_key", &simKey_);
  steps_->Branch("crystal", &crystal_); steps_->Branch("sipm0", &sipm0_);
  steps_->Branch("energy_g4", &energyG4_); steps_->Branch("energy_birks", &energyBirks_);
  steps_->Branch("time", &time_); steps_->Branch("z", &z_); steps_->Branch("lru", &lru_);
  steps_->Branch("yield0", &yields_[0]); steps_->Branch("yield1", &yields_[1]);
  readouts_ = tfs->make<TTree>("readouts", "observed stock readouts with exact input pointers and all PE times"); ids(readouts_);
  readouts_->Branch("product_id", &roProduct_); readouts_->Branch("index", &index_);
  readouts_->Branch("step_product_id", &linkedProduct_); readouts_->Branch("step_index", &linkedIndex_);
  readouts_->Branch("sipm", &sipm_); readouts_->Branch("pe_times", &peTimes_);
  summaries_ = tfs->make<TTree>("summaries", "observed stock corrected-energy/time groups and ordered member pointers"); ids(summaries_);
  summaries_->Branch("product_id", &summaryProduct_); summaries_->Branch("index", &index_);
  summaries_->Branch("energy", &energy_); summaries_->Branch("time", &summaryTime_);
  summaries_->Branch("member_products", &memberProducts_); summaries_->Branch("member_indices", &memberIndices_);
}

void CaloResponseDumper::analyze(art::Event const& event) {
  run_ = event.id().run(); subrun_ = event.id().subRun(); event_ = event.id().event();
  auto const steps = event.getValidHandle<CaloShowerStepCollection>(conf_.steps());
  auto const readouts = event.getValidHandle<CaloShowerROCollection>(conf_.readouts());
  auto const summaries = event.getValidHandle<CaloShowerSimCollection>(conf_.summaries());
  auto const ewm = event.getValidHandle<EventWindowMarker>(conf_.eventWindowMarker());
  auto const pb = event.getValidHandle<ProtonBunchTimeMC>(conf_.protonBunchTimeMC());
  auto const& conditions = simConditions_.get(event.id());
  auto const& timing = timingConditions_.get(event.id());
  stepProduct_ = steps.id().value(); roProduct_ = readouts.id().value(); summaryProduct_ = summaries.id().value();
  ewmProduct_ = ewm.id().value(); pbProduct_ = pb.id().value();
  nSteps_ = steps->size(); nReadouts_ = readouts->size(); nSummaries_ = summaries->size();
  spill_ = ewm->spillType(); eventLength_ = ewm->eventLength(); pbtime_ = pb->pbtime_;
  mbtime_ = GlobalConstantsHandle<PhysicsParams>()->getNominalDRPeriod();
  marker_ = timing.timeFromProtonsToDRMarker();
  length_ = GeomHandle<Calorimeter>()->G4Info().get<double>("crystalZLength");
  simIov_ = interval(simConditions_.iov()); timingIov_ = interval(timingConditions_.iov());
  simCids_.assign(conditions.getCids().begin(), conditions.getCids().end());
  timingCids_.assign(timing.getCids().begin(), timing.getCids().end());
  if ((spill_ != 0 && spill_ != 1) || !std::isfinite(eventLength_) || eventLength_ <= 0 ||
      !std::isfinite(pbtime_) || !std::isfinite(mbtime_) || mbtime_ <= 0 ||
      !std::isfinite(marker_) || !std::isfinite(length_) || length_ <= 0)
    throw cet::exception("CALORESPONSE") << "unsupported timing or geometry conditions\n";
  for (index_ = 0; index_ < steps->size(); ++index_) {
    auto const& step = steps->at(index_);
    crystal_ = step.volumeG4ID();
    if (crystal_ < 0 || crystal_ >= CaloConst::_nCrystal || step.simParticle().isNull())
      throw cet::exception("CALORESPONSE") << "invalid saved crystal or SimParticle link\n";
    CrystalId const crystal(crystal_);
    sipm0_ = crystal.SiPMId(CaloConst::SiPM0);
    auto const& yields = conditions.pePerMeVs(crystal);
    if (yields.size() != 2) throw cet::exception("CALORESPONSE") << "expected two SiPM yields\n";
    yields_ = {yields[0], yields[1]}; lru_ = conditions.LRU(crystal);
    energyG4_ = step.energyDepG4(); energyBirks_ = step.energyDepBirks();
    time_ = step.time(); z_ = step.position().z();
    simProduct_ = step.simParticle().id().value(); simKey_ = step.simParticle().key();
    if (!std::isfinite(energyG4_) || !std::isfinite(energyBirks_) || !std::isfinite(time_) ||
        !std::isfinite(z_) || !std::isfinite(lru_) || !std::isfinite(yields_[0]) || !std::isfinite(yields_[1]))
      throw cet::exception("CALORESPONSE") << "nonfinite saved state or conditions\n";
    steps_->Fill();
  }
  auto requireStep = [&](art::Ptr<CaloShowerStep> const& ptr) {
    if (ptr.isNull() || ptr.id() != steps.id() || ptr.key() >= steps->size())
      throw cet::exception("CALORESPONSE") << "stock output does not reference the configured saved steps\n";
  };
  for (index_ = 0; index_ < readouts->size(); ++index_) {
    auto const& ro = readouts->at(index_);
    auto const& ptr = ro.caloShowerStep(); requireStep(ptr);
    linkedProduct_ = ptr.id().value(); linkedIndex_ = ptr.key(); sipm_ = ro.SiPMID(); peTimes_ = ro.PETime();
    readouts_->Fill();
  }
  for (index_ = 0; index_ < summaries->size(); ++index_) {
    auto const& summary = summaries->at(index_);
    energy_ = summary.energyDep(); summaryTime_ = summary.time();
    memberProducts_.clear(); memberIndices_.clear();
    if (summary.caloShowerSteps().empty()) throw cet::exception("CALORESPONSE") << "empty stock summary\n";
    for (auto const& ptr : summary.caloShowerSteps()) {
      requireStep(ptr); memberProducts_.push_back(ptr.id().value()); memberIndices_.push_back(ptr.key());
    }
    summaries_->Fill();
  }
  events_->Fill();
}

} // namespace mu2e
DEFINE_ART_MODULE(mu2e::CaloResponseDumper)
