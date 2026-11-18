// Put the reco pulses of a CrvRecoTableDumper-layout table (crvdump/pulses) into each art event as a
// CrvRecoPulseCollection, matched by (run, subrun, event), in table order (the clusters' pulse indices refer to it).
// Waveform indices (wfidx) refer to the digi collection of the same table; ROC/FEB come from CRVOrdinal.
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/CRVConditions/inc/CRVOrdinal.hh"
#include "Offline/DataProducts/inc/CRVId.hh"
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/RecoDataProducts/inc/CrvRecoPulse.hh"
#include "Offline/CRVConditions/inc/CRVROC.hh"

#include "CrvRecoTable.hh"

namespace mu2e {

namespace {
// CrvRecoPulse gained ringing-artifact fields and a reflected-pulse flag after Offline v13_39_00. The tables carry
// neither, so a table pulse has no ringing artifact removed (height 0) and is not a reflected-pulse candidate. The
// check is on the template parameter, so only the constructor the Offline in use has is compiled.
template <class P>
void emplacePulse(std::vector<P>& out, float pe, float peph, double time, float height, float beta, float chi2,
                  double letime, size_t seq, CrvRecoPulseFlags const& flags, std::vector<size_t> const& wf,
                  CRSScintillatorBarIndex bar, uint8_t sipm, CRVROC const& on, float ped, bool pedfromdb) {
  if constexpr (std::is_constructible_v<P, float, float, double, float, float, float, double, size_t,
                                        CrvRecoPulseFlags const&, double, float, float, std::vector<size_t> const&,
                                        CRSScintillatorBarIndex, uint8_t, uint8_t, uint8_t, uint8_t, float, bool,
                                        bool>) {
    out.emplace_back(pe, peph, time, height, beta, chi2, letime, seq, flags, 0.0, 0.f, 0.f, wf, bar, sipm,
                     static_cast<uint8_t>(on.ROC()), static_cast<uint8_t>(on.FEB()),
                     static_cast<uint8_t>(on.FEBchannel()), ped, pedfromdb, false);
  } else {
    out.emplace_back(pe, peph, time, height, beta, chi2, letime, seq, flags, wf, bar, sipm, on.ROC(), on.FEB(),
                     on.FEBchannel(), ped, pedfromdb);
  }
}
}  // namespace

class CrvRecoPulsesFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> tableFile{fhicl::Name("tableFile"), fhicl::Comment("table file (CrvRecoTableDumper layout)")};
    fhicl::Atom<std::string> directory{fhicl::Name("directory"), fhicl::Comment("directory of the trees"), "crvdump"};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit CrvRecoPulsesFromTable(const Parameters& conf);
  void produce(art::Event& event) override;

 private:
  crvfastsim::RecoTable table_;
  ProditionsHandle<CRVOrdinal> channelMap_h_;
  Int_t bar_{0}, sipm_{0}, seq_{0}, flags_{0}, pedfromdb_{0};
  Float_t pe_{0}, peph_{0}, height_{0}, beta_{0}, chi2_{0}, ped_{0};
  Double_t time_{0}, letime_{0};
  std::unique_ptr<crvfastsim::Jagged<int>> wfidx_;
};

CrvRecoPulsesFromTable::CrvRecoPulsesFromTable(const Parameters& conf)
    : art::EDProducer(conf), table_(conf().tableFile(), conf().directory(), "pulses") {
  produces<CrvRecoPulseCollection>();
  TTree* t = table_.tree();
  t->SetBranchAddress("bar", &bar_);
  t->SetBranchAddress("sipm", &sipm_);
  t->SetBranchAddress("pe", &pe_);
  t->SetBranchAddress("peph", &peph_);
  t->SetBranchAddress("height", &height_);
  t->SetBranchAddress("beta", &beta_);
  t->SetBranchAddress("chi2", &chi2_);
  t->SetBranchAddress("ped", &ped_);
  t->SetBranchAddress("time", &time_);
  t->SetBranchAddress("letime", &letime_);
  t->SetBranchAddress("seq", &seq_);
  t->SetBranchAddress("flags", &flags_);
  t->SetBranchAddress("pedfromdb", &pedfromdb_);
  wfidx_ = std::make_unique<crvfastsim::Jagged<int>>(t, "wfidx");
}

void CrvRecoPulsesFromTable::produce(art::Event& event) {
  auto out = std::make_unique<CrvRecoPulseCollection>();
  const auto& rows = table_.rows(crvfastsim::EventKey(event.run(), event.subRun(), event.event()));
  if (!rows.empty()) {
    auto const& channelMap = channelMap_h_.get(event.id());
    TTree* t = table_.tree();
    out->reserve(rows.size());
    for (Long64_t i : rows) {
      t->GetEntry(i);
      auto const& on = channelMap.online(static_cast<std::uint16_t>(bar_ * CRVId::nChanPerBar + sipm_));
      auto w = wfidx_->values();
      std::vector<size_t> wf(w.begin(), w.end());
      emplacePulse(*out, pe_, peph_, time_, height_, beta_, chi2_, letime_, static_cast<size_t>(seq_),
                   CrvRecoPulseFlags(static_cast<unsigned long>(flags_)), wf, CRSScintillatorBarIndex(bar_),
                   static_cast<uint8_t>(sipm_), on, ped_, pedfromdb_ != 0);
    }
  }
  event.put(std::move(out));
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvRecoPulsesFromTable)
