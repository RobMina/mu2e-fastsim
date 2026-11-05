// REVIEW SOURCE ONLY. Schedule first, before any resampler/generator/filter.
#include "art/Framework/Core/EDFilter.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "ManifestMetadata.hh"
#include <cstdint>
namespace mu2e {
class CaloInputEventManifest : public art::EDFilter {
 public:
  explicit CaloInputEventManifest(fhicl::ParameterSet const& p) : art::EDFilter(p), metadata_(p) {}
  void beginJob() override {
    art::ServiceHandle<art::TFileService> fs;
    auto* m = fs->make<TTree>("metadata", "externally verified input artifact claims");
    metadata_.branches(m); m->Fill();
    events_ = fs->make<TTree>("inputEvents", "all EventIDs reaching first scheduled module");
    events_->Branch("ordinal", &ordinal_); events_->Branch("run", &run_);
    events_->Branch("subrun", &subrun_); events_->Branch("event", &event_);
  }
  bool filter(art::Event& e) override {
    run_ = e.run(); subrun_ = e.subRun(); event_ = e.event(); events_->Fill(); ++ordinal_;
    return true;
  }
 private:
  manifest_review::Metadata metadata_;
  TTree* events_{nullptr};
  std::uint64_t ordinal_{0};
  unsigned run_{0}, subrun_{0}, event_{0};
};
}
DEFINE_ART_MODULE(mu2e::CaloInputEventManifest)
