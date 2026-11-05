#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "Offline/RecoDataProducts/inc/KalSeed.hh"
#include "Offline/RecoDataProducts/inc/TriggerInfo.hh"
#include "Offline/RecoDataProducts/inc/TrkFitFlag.hh"

namespace mu2e {
class KalSeedSelectionFeatures : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<art::InputTag> kalSeeds{
        fhicl::Name("kalSeeds"), fhicl::Comment("Exact shared replay KalSeed collection"),
        art::InputTag("TrigKSFDe::TriggerReplay")};
    fhicl::Sequence<art::InputTag> selectionInfo{
        fhicl::Name("selectionInfo"), fhicl::Comment("The two exact replay selection TriggerInfo products"),
        std::vector<art::InputTag>{art::InputTag("TprTrkDe80m70pD0200KSFilter::TriggerReplay"),
                                   art::InputTag("TprTrkDe80m70pKSFilter::TriggerReplay")}};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit KalSeedSelectionFeatures(Parameters const& config);
  void beginJob() override;
  void analyze(art::Event const& event) override;

 private:
  art::InputTag kalSeeds_;
  std::vector<art::InputTag> selectionInfo_;
  TrkFitFlag seedOK_{std::string("SeedOK")};
  TTree* events_{nullptr};
  unsigned run_{0}, subrun_{0}, event_{0};
  std::uint64_t sourceOrdinal_{0};
  bool productPresent_{false};
  std::string productID_;
  std::vector<std::uint64_t> trackIndex_;
  std::vector<std::uint32_t> rawStatus_;
  std::vector<unsigned char> validTrack_, statusSeedOK_, hasT0Segment_, supportedFit_;
  std::vector<int> fitParticle_;
  std::vector<unsigned> activeHits_;
  std::vector<double> pz_, momentum_, tanDip_, d0_, momentumError_, chi2Dof_, fitConsistency_, t0_;
  std::vector<std::int64_t> t0SegmentIndex_;
  std::array<bool, 2> infoPresent_{{false, false}};
  std::array<std::vector<std::uint64_t>, 2> acceptedTrackKeys_;
  unsigned schemaVersion_{1};
  std::string release_{"v13_41_01"};
  std::string commit_{"b132ab6567ac021851353e05b7a1feb45bfb7c14"};
  std::string filterSHA256_{"51b6dbfe6f5dcd115afdfb8b9ada7aa187b8610f717cdbbc1527e7b573fabd39"};
  std::string backing_{"/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ba"};
  std::string precision_{"chi2_dof=float(chisquared)/int(nDOF), then double; fit_consistency=float then double; segment kinematics=double"};
  std::string kalSeedsTag_;
  std::array<std::string, 2> selectionTags_;
};

KalSeedSelectionFeatures::KalSeedSelectionFeatures(Parameters const& config)
    : art::EDAnalyzer(config), kalSeeds_(config().kalSeeds()), selectionInfo_(config().selectionInfo()) {
  if (kalSeeds_.label() != "TrigKSFDe" || !kalSeeds_.instance().empty() || kalSeeds_.process() != "TriggerReplay")
    throw cet::exception("R7FeaturesConfig") << "kalSeeds must be TrigKSFDe::TriggerReplay";
  if (selectionInfo_.size() != 2)
    throw cet::exception("R7FeaturesConfig") << "selectionInfo must contain exactly the two supported filters";
  std::set<std::string> expected{"TprTrkDe80m70pD0200KSFilter", "TprTrkDe80m70pKSFilter"};
  for (auto const& tag : selectionInfo_) {
    if (!tag.instance().empty() || tag.process() != "TriggerReplay" || expected.erase(tag.label()) != 1)
      throw cet::exception("R7FeaturesConfig") << "unsupported or duplicate selectionInfo tag " << tag.encode();
  }
  // Branch slots are canonical, independent of configuration sequence order.
  selectionInfo_ = {art::InputTag("TprTrkDe80m70pD0200KSFilter::TriggerReplay"),
                    art::InputTag("TprTrkDe80m70pKSFilter::TriggerReplay")};
  consumes<KalSeedCollection>(kalSeeds_);
  kalSeedsTag_ = kalSeeds_.encode();
  for (std::size_t i = 0; i < selectionInfo_.size(); ++i) {
    consumes<TriggerInfo>(selectionInfo_[i]);
    selectionTags_[i] = selectionInfo_[i].encode();
  }
}

void KalSeedSelectionFeatures::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  events_ = tfs->make<TTree>("events", "R7 measured KalSeed selection inputs, every event");
  events_->Branch("run", &run_);
  events_->Branch("subrun", &subrun_);
  events_->Branch("event", &event_);
  events_->Branch("source_ordinal", &sourceOrdinal_);
  events_->Branch("product_present", &productPresent_);
  events_->Branch("product_id", &productID_);
  events_->Branch("track_index", &trackIndex_);
  events_->Branch("raw_status", &rawStatus_);
  events_->Branch("valid_track", &validTrack_);
  events_->Branch("status_has_SeedOK", &statusSeedOK_);
  events_->Branch("has_t0_segment", &hasT0Segment_);
  events_->Branch("supported_fit", &supportedFit_);
  events_->Branch("fit_particle", &fitParticle_);
  events_->Branch("active_hits", &activeHits_);
  events_->Branch("pz", &pz_);
  events_->Branch("momentum", &momentum_);
  events_->Branch("tan_dip", &tanDip_);
  events_->Branch("d0", &d0_);
  events_->Branch("momentum_error", &momentumError_);
  events_->Branch("chi2_dof", &chi2Dof_);
  events_->Branch("fit_consistency", &fitConsistency_);
  events_->Branch("t0_segment_index", &t0SegmentIndex_);
  events_->Branch("t0", &t0_);
  events_->Branch("info_present_D0200", &infoPresent_[0]);
  events_->Branch("info_present_nominal", &infoPresent_[1]);
  events_->Branch("accepted_track_keys_D0200", &acceptedTrackKeys_[0]);
  events_->Branch("accepted_track_keys_nominal", &acceptedTrackKeys_[1]);

  auto* schema = tfs->make<TTree>("schema", "Declared source binding, not binary attestation");
  schema->Branch("schema_version", &schemaVersion_);
  schema->Branch("offline_release", &release_);
  schema->Branch("offline_commit", &commit_);
  schema->Branch("filter_sha256", &filterSHA256_);
  schema->Branch("required_backing", &backing_);
  schema->Branch("precision", &precision_);
  schema->Branch("kalSeeds", &kalSeedsTag_);
  schema->Branch("selectionInfo_D0200", &selectionTags_[0]);
  schema->Branch("selectionInfo_nominal", &selectionTags_[1]);
  schema->Fill();
}

void KalSeedSelectionFeatures::analyze(art::Event const& event) {
  if (sourceOrdinal_ == std::numeric_limits<std::uint64_t>::max())
    throw cet::exception("R7FeaturesIdentity") << "source ordinal overflow";
  ++sourceOrdinal_;
  run_ = event.run();
  subrun_ = event.subRun();
  event_ = event.event();
  productID_.clear();
  trackIndex_.clear();
  rawStatus_.clear();
  for (auto* values : {&validTrack_, &statusSeedOK_, &hasT0Segment_, &supportedFit_}) values->clear();
  fitParticle_.clear();
  activeHits_.clear();
  t0SegmentIndex_.clear();
  for (auto* values : {&pz_, &momentum_, &tanDip_, &d0_, &momentumError_, &chi2Dof_, &fitConsistency_, &t0_})
    values->clear();
  infoPresent_.fill(false);
  for (auto& keys : acceptedTrackKeys_) keys.clear();

  auto const seeds = event.getHandle<KalSeedCollection>(kalSeeds_);
  productPresent_ = seeds.isValid();
  for (std::size_t i = 0; i < selectionInfo_.size(); ++i) {
    auto const info = event.getHandle<TriggerInfo>(selectionInfo_[i]);
    infoPresent_[i] = info.isValid();
    if (!infoPresent_[i]) continue;
    if (!productPresent_)
      throw cet::exception("R7FeaturesIdentity") << "TriggerInfo present without shared KalSeeds in " << event.id();
    std::set<std::uint64_t> seen;
    for (auto const& ptr : info->tracks()) {
      if (ptr.isNull() || ptr.id() != seeds.id() || ptr.key() >= seeds->size())
        throw cet::exception("R7FeaturesIdentity") << "invalid selected track identity for "
            << selectionInfo_[i].encode() << " in " << event.id();
      auto const key = static_cast<std::uint64_t>(ptr.key());
      if (!seen.insert(key).second)
        throw cet::exception("R7FeaturesIdentity") << "duplicate selected track key " << key << " in " << event.id();
      acceptedTrackKeys_[i].push_back(key);
    }
  }

  if (productPresent_) {
    std::ostringstream id;
    id << seeds.id();
    productID_ = id.str();
    auto const nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t index = 0; index < seeds->size(); ++index) {
      auto const& Ks = (*seeds)[index];
      trackIndex_.push_back(static_cast<std::uint64_t>(index));
      auto const status = std::stoull(Ks.status().hex(), nullptr, 16);
      if (status > std::numeric_limits<std::uint32_t>::max())
        throw cet::exception("R7FeaturesStatus") << "status exceeds uint32 in " << event.id();
      rawStatus_.push_back(static_cast<std::uint32_t>(status));
      validTrack_.push_back(1);
      bool const seedOK = Ks.status().hasAllProperties(seedOK_);
      statusSeedOK_.push_back(static_cast<unsigned char>(seedOK));
      bool const supported = Ks.loopHelixFit() || Ks.centralHelixFit() || Ks.kinematicLineFit();
      supportedFit_.push_back(static_cast<unsigned char>(supported));
      hasT0Segment_.push_back(0);
      fitParticle_.push_back(static_cast<int>(Ks.particle()));
      activeHits_.push_back(Ks.nHits(true));
      t0SegmentIndex_.push_back(-1);
      for (auto* values : {&pz_, &momentum_, &tanDip_, &d0_, &momentumError_, &chi2Dof_, &fitConsistency_, &t0_})
        values->push_back(nan);
      if (!seedOK) continue;
      double t0;
      auto const t0seg = Ks.t0Segment(t0);
      if (t0seg == Ks.segments().end()) continue;
      hasT0Segment_.back() = 1;
      t0SegmentIndex_.back() = static_cast<std::int64_t>(std::distance(Ks.segments().begin(), t0seg));
      t0_.back() = t0;
      auto const momvec = t0seg->momentum3();
      auto const posvec = t0seg->position3();
      double const td = 1.0/tan(momvec.Theta());
      double const mom = momvec.R();
      double d0;
      if (Ks.loopHelixFit()) {
        auto const lhtraj = t0seg->loopHelix();
        d0 = std::copysign(lhtraj.minAxisDist(), posvec.Cross(momvec).Z());
      } else if (Ks.centralHelixFit()) {
        auto const chtraj = t0seg->centralHelix();
        d0 = chtraj.d0();
      } else if (Ks.kinematicLineFit()) {
        auto const kltraj = t0seg->kinematicLine();
        d0 = kltraj.d0();
      } else {
        continue;
      }
      pz_.back() = momvec.Z();
      momentum_.back() = mom;
      tanDip_.back() = td;
      d0_.back() = d0;
      momentumError_.back() = t0seg->momerr();
      // The division is float/int before assignment promotes the result to double.
      chi2Dof_.back() = Ks.chisquared()/Ks.nDOF();
      fitConsistency_.back() = Ks.fitConsistency();
    }
  }
  events_->Fill();
}
}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::KalSeedSelectionFeatures)
