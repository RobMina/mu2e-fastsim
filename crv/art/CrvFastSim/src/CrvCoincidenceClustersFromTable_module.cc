// Put the coincidence clusters of a CrvRecoTableDumper-layout table (crvdump/clusters) into each art event as a
// CrvCoincidenceClusterCollection, matched by (run, subrun, event). Member pulses (the `pulses` column: indices into
// the event's pulse rows of the same table) become art::Ptr into the CrvRecoPulseCollection of `pulseTag`, normally a
// CrvRecoPulsesFromTable on the same file.
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/DataProducts/inc/CRVId.hh"
#include "Offline/RecoDataProducts/inc/CrvCoincidenceCluster.hh"
#include "cetlib_except/exception.h"

#include "CrvRecoTable.hh"

namespace mu2e {

class CrvCoincidenceClustersFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> tableFile{fhicl::Name("tableFile"), fhicl::Comment("table file (CrvRecoTableDumper layout)")};
    fhicl::Atom<std::string> directory{fhicl::Name("directory"), fhicl::Comment("directory of the trees"), "crvdump"};
    fhicl::Atom<art::InputTag> pulseTag{fhicl::Name("pulseTag"), fhicl::Comment("pulses the member indices refer to")};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit CrvCoincidenceClustersFromTable(const Parameters& conf);
  void produce(art::Event& event) override;

 private:
  crvfastsim::RecoTable table_;
  art::InputTag pulseTag_;
  Int_t sectortype_{0}, poscalc_{0}, sh0_{0}, sh1_{0};
  Double_t start_{0}, end_{0}, avgtime_{0}, st0_{0}, st1_{0};
  Float_t pe_{0}, slope_{0}, x_{0}, y_{0}, z_{0}, spe0_{0}, spe1_{0};
  std::unique_ptr<crvfastsim::Jagged<int>> layers_, members_;
};

CrvCoincidenceClustersFromTable::CrvCoincidenceClustersFromTable(const Parameters& conf)
    : art::EDProducer(conf), table_(conf().tableFile(), conf().directory(), "clusters"), pulseTag_(conf().pulseTag()) {
  produces<CrvCoincidenceClusterCollection>();
  consumes<CrvRecoPulseCollection>(pulseTag_);
  TTree* t = table_.tree();
  t->SetBranchAddress("sectortype", &sectortype_);
  t->SetBranchAddress("start", &start_);
  t->SetBranchAddress("end", &end_);
  t->SetBranchAddress("pe", &pe_);
  t->SetBranchAddress("slope", &slope_);
  layers_ = std::make_unique<crvfastsim::Jagged<int>>(t, "layers");
  members_ = std::make_unique<crvfastsim::Jagged<int>>(t, "pulses");
  t->SetBranchAddress("poscalc", &poscalc_);
  t->SetBranchAddress("avgtime", &avgtime_);
  t->SetBranchAddress("x", &x_);
  t->SetBranchAddress("y", &y_);
  t->SetBranchAddress("z", &z_);
  t->SetBranchAddress("sidepe0", &spe0_);
  t->SetBranchAddress("sidepe1", &spe1_);
  t->SetBranchAddress("sidetime0", &st0_);
  t->SetBranchAddress("sidetime1", &st1_);
  t->SetBranchAddress("sidehits0", &sh0_);
  t->SetBranchAddress("sidehits1", &sh1_);
}

void CrvCoincidenceClustersFromTable::produce(art::Event& event) {
  auto out = std::make_unique<CrvCoincidenceClusterCollection>();
  art::Handle<CrvRecoPulseCollection> pulses;
  if (!event.getByLabel(pulseTag_, pulses))
    throw cet::exception("CRVFASTSIM") << "no CrvRecoPulseCollection " << pulseTag_ << " in " << event.id();
  const auto& rows = table_.rows(crvfastsim::EventKey(event.run(), event.subRun(), event.event()));
  if (!rows.empty()) {
    TTree* t = table_.tree();
    out->reserve(rows.size());
    for (Long64_t i : rows) {
      t->GetEntry(i);
      auto members = members_->values();
      std::vector<art::Ptr<CrvRecoPulse>> ptrs;
      ptrs.reserve(members.size());
      for (int m : members) {
        if (m < 0 || static_cast<size_t>(m) >= pulses->size())
          throw cet::exception("CRVFASTSIM") << "cluster member " << m << " outside " << pulses->size()
                                             << " pulses in " << event.id();
        ptrs.emplace_back(art::Ptr<CrvRecoPulse>(pulses, static_cast<size_t>(m)));
      }
      std::array<size_t, CRVId::nSidesPerBar> sideHits{static_cast<size_t>(sh0_), static_cast<size_t>(sh1_)};
      std::array<float, CRVId::nSidesPerBar> sidePEs{spe0_, spe1_};
      std::array<double, CRVId::nSidesPerBar> sideTimes{st0_, st1_};
      out->emplace_back(sectortype_, start_, end_, pe_, ptrs, slope_, layers_->values(), sideHits, sidePEs, sideTimes, avgtime_,
                        CLHEP::Hep3Vector(x_, y_, z_), poscalc_ != 0);
    }
  }
  event.put(std::move(out));
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvCoincidenceClustersFromTable)
