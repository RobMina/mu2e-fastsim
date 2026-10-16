// Put the fast-simulation CrvSteps of a crvfastsim table file into each art event (path A of docs/plan.md section 6).
// Matched by (run, subrun, event): run it on the G4 dts file of the same events and the stock digitisation / CRV reco
// then see the fast steps instead of G4's (CrvPhotons.crvStepModuleLabels: [ "<this label>" ]).
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "canvas/Persistency/Common/Ptr.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/DataProducts/inc/PDGCode.hh"
#include "Offline/MCDataProducts/inc/CrvStep.hh"
#include "Offline/MCDataProducts/inc/ProcessCode.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"

namespace mu2e {

class CrvStepsFromTable : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<std::string> tableFile{fhicl::Name("tableFile"), fhicl::Comment("crvfastsim table (dump/steps)")};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit CrvStepsFromTable(const Parameters& conf);
  void produce(art::Event& event) override;

 private:
  struct Row {
    int bar, pdg;
    float edep, x0, y0, z0, x1, y1, z1, px0, py0, pz0, p1, len;
    double t0, t1;
  };
  using Key = std::tuple<unsigned, unsigned, unsigned>;
  std::map<Key, std::vector<Row>> rows_;
  std::map<Key, int> pdg_;
};

CrvStepsFromTable::CrvStepsFromTable(const Parameters& conf) : art::EDProducer(conf) {
  produces<SimParticleCollection>();
  produces<CrvStepCollection>();
  std::unique_ptr<TFile> f(TFile::Open(conf().tableFile().c_str()));
  if (!f || f->IsZombie()) throw cet::exception("CRVFASTSIM") << "cannot open " << conf().tableFile();
  auto* dir = f->GetDirectory("dump");
  auto* pt = dir ? dir->Get<TTree>("particles") : nullptr;
  auto* st = dir ? dir->Get<TTree>("steps") : nullptr;
  if (!pt || !st) throw cet::exception("CRVFASTSIM") << "no dump/particles or dump/steps in " << conf().tableFile();
  // column types as written by crvfastsim.tables.write (uproot): keys uint32, pdg int32, bar int64, floats double
  UInt_t run, sub, evt;
  Int_t pdg;
  Long64_t bar;
  pt->SetBranchAddress("run", &run);
  pt->SetBranchAddress("subrun", &sub);
  pt->SetBranchAddress("event", &evt);
  pt->SetBranchAddress("pdg", &pdg);
  for (Long64_t i = 0; i < pt->GetEntries(); ++i) {
    pt->GetEntry(i);
    pdg_[Key(run, sub, evt)] = pdg;
  }
  double v[12], t0, t1;
  const char* names[12] = {"edep", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "p1", "len"};
  st->SetBranchAddress("run", &run);
  st->SetBranchAddress("subrun", &sub);
  st->SetBranchAddress("event", &evt);
  st->SetBranchAddress("bar", &bar);
  st->SetBranchAddress("t0", &t0);
  st->SetBranchAddress("t1", &t1);
  for (int k = 0; k < 12; ++k) st->SetBranchAddress(names[k], &v[k]);
  for (Long64_t i = 0; i < st->GetEntries(); ++i) {
    st->GetEntry(i);
    Key key(run, sub, evt);
    rows_[key].push_back(Row{int(bar), pdg_[key], float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]),
                             float(v[5]), float(v[6]), float(v[7]), float(v[8]), float(v[9]), float(v[10]),
                             float(v[11]), t0, t1});
  }
}

void CrvStepsFromTable::produce(art::Event& event) {
  auto parts = std::make_unique<SimParticleCollection>();
  auto steps = std::make_unique<CrvStepCollection>();
  Key key(event.run(), event.subRun(), event.event());
  auto it = rows_.find(key);
  if (it != rows_.end() && !it->second.empty()) {
    const auto& r0 = it->second.front();
    cet::map_vector_key pkey(1);
    CLHEP::Hep3Vector pos(r0.x0, r0.y0, r0.z0);
    CLHEP::HepLorentzVector mom(r0.px0, r0.py0, r0.pz0, 0);
    (*parts)[pkey] = SimParticle(pkey, 0, art::Ptr<SimParticle>(), PDGCode::type(r0.pdg), art::Ptr<GenParticle>(),
                                 pos, mom, float(r0.t0), 0.f, 0u, 0u, ProcessCode(ProcessCode::mu2ePrimary));
    auto pid = event.getProductID<SimParticleCollection>();
    art::Ptr<SimParticle> ptr(pid, pkey.asUint(), event.productGetter(pid));
    for (const auto& r : it->second)
      steps->emplace_back(CRSScintillatorBarIndex(r.bar), r.edep, r.t0, r.t1, XYZVectorF(r.x0, r.y0, r.z0),
                          XYZVectorF(r.x1, r.y1, r.z1), XYZVectorF(r.px0, r.py0, r.pz0), r.p1, r.len, ptr);
  }
  event.put(std::move(parts));
  event.put(std::move(steps));
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvStepsFromTable)
