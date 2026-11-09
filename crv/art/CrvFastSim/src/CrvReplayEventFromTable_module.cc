// Supply the saved mixed-frame event window and bunch time to both stock response and reconstruction.
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "TFile.h"
#include "TTree.h"
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/DataProducts/inc/EventWindowMarker.hh"
#include "Offline/MCDataProducts/inc/ProtonBunchTimeMC.hh"
#include "Offline/RecoDataProducts/inc/ProtonBunchTime.hh"

namespace mu2e {

class CrvReplayEventFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> tableFile{fhicl::Name("tableFile")};
    fhicl::Atom<double> eventWindowLength{fhicl::Name("eventWindowLength")};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit CrvReplayEventFromTable(const Parameters& conf);
  void produce(art::Event& event) override;

 private:
  using Key = std::tuple<unsigned, unsigned, unsigned>;
  std::map<Key, double> times_;
  double length_;
};

CrvReplayEventFromTable::CrvReplayEventFromTable(const Parameters& conf)
    : art::EDProducer(conf), length_(conf().eventWindowLength()) {
  if (!std::isfinite(length_) || length_ <= 0)
    throw cet::exception("CRVREPLAY") << "eventWindowLength must be positive and finite";
  produces<EventWindowMarker>();
  produces<ProtonBunchTimeMC>();
  produces<ProtonBunchTime>();
  std::unique_ptr<TFile> file(TFile::Open(conf().tableFile().c_str()));
  if (!file || file->IsZombie()) throw cet::exception("CRVREPLAY") << "cannot open " << conf().tableFile();
  auto* tree = file->Get<TTree>("dump/events");
  if (!tree || tree->GetEntries() == 0) throw cet::exception("CRVREPLAY") << "missing or empty dump/events TTree";
  UInt_t run, subrun, event;
  double pbtime;
  if (tree->SetBranchAddress("run", &run) < 0 || tree->SetBranchAddress("subrun", &subrun) < 0 ||
      tree->SetBranchAddress("event", &event) < 0 || tree->SetBranchAddress("pbtime", &pbtime) < 0)
    throw cet::exception("CRVREPLAY") << "invalid dump/events branch types";
  for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
    if (tree->GetEntry(i) <= 0 || !std::isfinite(pbtime) ||
        !times_.emplace(Key(run, subrun, event), pbtime).second)
      throw cet::exception("CRVREPLAY") << "invalid or duplicate event row " << i;
  }
}

void CrvReplayEventFromTable::produce(art::Event& event) {
  auto it = times_.find(Key(event.run(), event.subRun(), event.event()));
  if (it == times_.end()) throw cet::exception("CRVREPLAY") << "missing event conditions for " << event.id();
  auto marker = std::make_unique<EventWindowMarker>();
  marker->_spillType = EventWindowMarker::onspill;
  marker->_eventLength = length_;
  auto truth = std::make_unique<ProtonBunchTimeMC>();
  truth->pbtime_ = static_cast<float>(it->second);
  auto reco = std::make_unique<ProtonBunchTime>();
  reco->pbtime_ = truth->pbtime_;
  reco->pbterr_ = 0.f;
  event.put(std::move(marker));
  event.put(std::move(truth));
  event.put(std::move(reco));
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvReplayEventFromTable)
