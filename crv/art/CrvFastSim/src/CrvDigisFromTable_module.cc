// Put the digis of a CrvRecoTableDumper-layout table (crvdump/digis) into each art event as a CrvDigiCollection, matched
// by (run, subrun, event). Source: the fast chain's digis (crvfastsim valhist writer) or a stock dump (round trip).
// ROC/FEB/FEB channel come from the table when present (>= 0), else from CRVOrdinal as in CrvDigitizer.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/CRVConditions/inc/CRVOrdinal.hh"
#include "Offline/DataProducts/inc/CRVId.hh"
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/RecoDataProducts/inc/CrvDigi.hh"
#include "Offline/CRVConditions/inc/CRVROC.hh"

#include "CrvRecoTable.hh"

namespace mu2e {

class CrvDigisFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> tableFile{fhicl::Name("tableFile"), fhicl::Comment("table file (CrvRecoTableDumper layout)")};
    fhicl::Atom<std::string> directory{fhicl::Name("directory"), fhicl::Comment("directory of the trees"), "crvdump"};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit CrvDigisFromTable(const Parameters& conf);
  void produce(art::Event& event) override;

 private:
  crvfastsim::RecoTable table_;
  ProditionsHandle<CRVOrdinal> channelMap_h_;
  bool hasOnline_{false};
  Int_t bar_{0}, sipm_{0}, starttdc_{0}, nzs_{0}, roc_{-1}, feb_{-1}, febch_{-1};
  std::unique_ptr<crvfastsim::Jagged<short>> adc_;
};

CrvDigisFromTable::CrvDigisFromTable(const Parameters& conf)
    : art::EDProducer(conf), table_(conf().tableFile(), conf().directory(), "digis") {
  produces<CrvDigiCollection>();
  TTree* t = table_.tree();
  hasOnline_ = t->GetBranch("roc") != nullptr;
  t->SetBranchAddress("bar", &bar_);
  t->SetBranchAddress("sipm", &sipm_);
  t->SetBranchAddress("starttdc", &starttdc_);
  t->SetBranchAddress("nzs", &nzs_);
  adc_ = std::make_unique<crvfastsim::Jagged<short>>(t, "adc");
  if (hasOnline_) {
    t->SetBranchAddress("roc", &roc_);
    t->SetBranchAddress("feb", &feb_);
    t->SetBranchAddress("febch", &febch_);
  }
}

void CrvDigisFromTable::produce(art::Event& event) {
  auto out = std::make_unique<CrvDigiCollection>();
  const auto& rows = table_.rows(crvfastsim::EventKey(event.run(), event.subRun(), event.event()));
  if (!rows.empty()) {
    auto const& channelMap = channelMap_h_.get(event.id());
    TTree* t = table_.tree();
    out->reserve(rows.size());
    for (Long64_t i : rows) {
      t->GetEntry(i);
      int roc = roc_, feb = feb_, febch = febch_;
      if (!hasOnline_ || roc < 0) {
        auto const& on = channelMap.online(static_cast<std::uint16_t>(bar_ * CRVId::nChanPerBar + sipm_));
        roc = on.ROC();
        feb = on.FEB();
        febch = on.FEBchannel();
      }
      auto a = adc_->values();
      std::vector<int16_t> adcs(a.begin(), a.end());
      out->emplace_back(std::move(adcs), static_cast<uint16_t>(starttdc_), nzs_ != 0, false, CRSScintillatorBarIndex(bar_),
                        static_cast<uint8_t>(sipm_), static_cast<uint8_t>(roc), static_cast<uint8_t>(feb),
                        static_cast<uint8_t>(febch));
    }
  }
  event.put(std::move(out));
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvDigisFromTable)
