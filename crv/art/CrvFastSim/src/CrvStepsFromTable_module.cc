// Put the fast-simulation CrvSteps of a crvfastsim table file into each art event (path A of docs/plan.md section 6).
// Matched by (run, subrun, event): run it on the G4 dts file of the same events and the stock digitisation / CRV reco
// then see the fast steps instead of G4's (CrvPhotons.crvStepModuleLabels: [ "<this label>" ]).
// One SimParticle per particles-table row (key, parent, pdg, isprim); each step points to its row through `sp`.
#include <cstddef>
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
    int bar;
    long sp;
    float edep, x0, y0, z0, x1, y1, z1, px0, py0, pz0, p1, len;
    double t0, t1;
  };
  struct Part {
    long key, parent;
    int pdg;
    bool prim;
  };
  using Key = std::tuple<unsigned, unsigned, unsigned>;
  std::map<Key, std::vector<Row>> rows_;
  std::map<Key, std::vector<Part>> parts_;
};

CrvStepsFromTable::CrvStepsFromTable(const Parameters& conf) : art::EDProducer(conf) {
  produces<SimParticleCollection>();
  produces<CrvStepCollection>();
  std::unique_ptr<TFile> f(TFile::Open(conf().tableFile().c_str()));
  if (!f || f->IsZombie()) throw cet::exception("CRVFASTSIM") << "cannot open " << conf().tableFile();
  auto* dir = f->GetDirectory("dump");
  auto* pt = dir ? dir->Get<TTree>("particles") : nullptr;
  auto* st = dir ? dir->Get<TTree>("steps") : nullptr;
  if (!pt || !st) throw cet::exception("CRVFASTSIM") << "no dump/particles or dump/steps TTree in " << conf().tableFile();
  // column types as written by crvfastsim.tables.write: keys uint32, pdg int32, bar/sp/key/parent/isprim int64, floats double
  UInt_t run, sub, evt;
  Int_t pdg;
  Long64_t key, parent, isprim, bar, sp;
  pt->SetBranchAddress("run", &run);
  pt->SetBranchAddress("subrun", &sub);
  pt->SetBranchAddress("event", &evt);
  pt->SetBranchAddress("key", &key);
  pt->SetBranchAddress("parent", &parent);
  pt->SetBranchAddress("pdg", &pdg);
  pt->SetBranchAddress("isprim", &isprim);
  for (Long64_t i = 0; i < pt->GetEntries(); ++i) {
    pt->GetEntry(i);
    parts_[Key(run, sub, evt)].push_back(Part{long(key), long(parent), int(pdg), isprim != 0});
  }
  double v[12], t0, t1;
  const char* names[12] = {"edep", "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0", "p1", "len"};
  st->SetBranchAddress("run", &run);
  st->SetBranchAddress("subrun", &sub);
  st->SetBranchAddress("event", &evt);
  st->SetBranchAddress("bar", &bar);
  st->SetBranchAddress("sp", &sp);
  st->SetBranchAddress("t0", &t0);
  st->SetBranchAddress("t1", &t1);
  for (int k = 0; k < 12; ++k) st->SetBranchAddress(names[k], &v[k]);
  for (Long64_t i = 0; i < st->GetEntries(); ++i) {
    st->GetEntry(i);
    rows_[Key(run, sub, evt)].push_back(Row{int(bar), long(sp), float(v[0]), float(v[1]), float(v[2]), float(v[3]),
                                            float(v[4]), float(v[5]), float(v[6]), float(v[7]), float(v[8]),
                                            float(v[9]), float(v[10]), float(v[11]), t0, t1});
  }
}

void CrvStepsFromTable::produce(art::Event& event) {
  auto parts = std::make_unique<SimParticleCollection>();
  auto steps = std::make_unique<CrvStepCollection>();
  Key key(event.run(), event.subRun(), event.event());
  auto it = rows_.find(key);
  auto ip = parts_.find(key);
  if (it != rows_.end() && !it->second.empty()) {
    if (ip == parts_.end()) throw cet::exception("CRVFASTSIM") << "steps without particles in event " << event.id();
    auto pid = event.getProductID<SimParticleCollection>();
    auto* getter = event.productGetter(pid);
    std::map<long, const Row*> first;  // first step of each particle: start position, momentum, time
    for (const auto& r : it->second) first.emplace(r.sp, &r);
    std::map<long, bool> known;
    for (const auto& p : ip->second) known[p.key] = true;
    for (const auto& p : ip->second) {
      cet::map_vector_key pkey(static_cast<unsigned long>(p.key));
      auto f = first.find(p.key);
      CLHEP::Hep3Vector pos;
      CLHEP::HepLorentzVector mom;
      float t = 0.f;
      if (f != first.end()) {
        pos = CLHEP::Hep3Vector(f->second->x0, f->second->y0, f->second->z0);
        mom = CLHEP::HepLorentzVector(f->second->px0, f->second->py0, f->second->pz0, 0);
        t = float(f->second->t0);
      }
      art::Ptr<SimParticle> par;
      if (p.parent >= 0 && known.count(p.parent)) par = art::Ptr<SimParticle>(pid, std::size_t(p.parent), getter);
      ProcessCode code(p.prim ? ProcessCode::mu2ePrimary : ProcessCode::unknown);
      (*parts)[pkey] = SimParticle(pkey, 0, par, PDGCode::type(p.pdg), art::Ptr<GenParticle>(), pos, mom, t, 0.f, 0u,
                                   0u, code);
    }
    for (const auto& r : it->second) {
      if (!known.count(r.sp)) throw cet::exception("CRVFASTSIM") << "step with unknown sp " << r.sp << " in " << event.id();
      art::Ptr<SimParticle> ptr(pid, std::size_t(r.sp), getter);
      steps->emplace_back(CRSScintillatorBarIndex(r.bar), r.edep, r.t0, r.t1, XYZVectorF(r.x0, r.y0, r.z0),
                          XYZVectorF(r.x1, r.y1, r.z1), XYZVectorF(r.px0, r.py0, r.pz0), r.p1, r.len, ptr);
    }
  }
  event.put(std::move(parts));
  event.put(std::move(steps));
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvStepsFromTable)
