// Deterministic already-aggregated extraction fixture; never substitute it for transport.
#include <memory>

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/SubRun.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/MCDataProducts/inc/CaloShowerStep.hh"
#include "Offline/MCDataProducts/inc/CrvStep.hh"
#include "Offline/MCDataProducts/inc/GenEventCount.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StrawGasStep.hh"

namespace mu2e {
class FlashBoardFixture : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<bool> badPtr{fhicl::Name("badPtr"), fhicl::Comment("Inject nonexistent SimParticle key 999 in the first tracker step"), false};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit FlashBoardFixture(Parameters const& c) : art::EDProducer(c), badPtr_(c().badPtr()) {
    produces<SimParticleCollection>("particlesA"); produces<SimParticleCollection>("particlesB");
    produces<StrawGasStepCollection>("tracker"); produces<CaloShowerStepCollection>("calo");
    produces<CrvStepCollection>("crv"); produces<GenEventCount, art::InSubRun>();
  }
  void produce(art::Event&) override;
  void endSubRun(art::SubRun&) override;
  void endJob() override;
 private:
  bool badPtr_;
  unsigned ordinal_{0}, inSubRun_{0}, subruns_{0};
};

void FlashBoardFixture::produce(art::Event& e) {
  if (ordinal_ >= 4 || e.run() != 1 || e.subRun() != 1 + ordinal_ / 2)
    throw cet::exception("FlashBoardFixture") << "Fixture requires four ordered events in run 1, SubRuns 1 and 2";
  auto a = std::make_unique<SimParticleCollection>();
  auto b = std::make_unique<SimParticleCollection>();
  auto straw = std::make_unique<StrawGasStepCollection>();
  auto calo = std::make_unique<CaloShowerStepCollection>();
  auto crv = std::make_unique<CrvStepCollection>();
  auto const aid = e.getProductID<SimParticleCollection>("particlesA");
  auto const bid = e.getProductID<SimParticleCollection>("particlesB");
  art::Ptr<SimParticle> const ap(aid, 17, e.productGetter(aid));
  art::Ptr<SimParticle> const bp(bid, 17, e.productGetter(bid));
  art::Ptr<SimParticle> const bad(aid, 999, e.productGetter(aid));
  CLHEP::Hep3Vector const position(0., 0., 0.);
  CLHEP::HepLorentzVector const fourMomentum(0., 0., 1., 2.);
  SimParticle const particle(cet::map_vector_key(17), 0, {}, PDGCode::e_minus, {},
                            position, fourMomentum, 0.f, 0.f, 0, 0, ProcessCode::mu2ePrimary);
  a->insert({cet::map_vector_key(17), particle}); b->insert({cet::map_vector_key(17), particle});
  XYZVectorF const start(0.f, 0.f, 0.f), end(0.f, 0.f, 1.f), momentum(0.f, 0.f, 1.f);
  auto tracker = [&](float energy, art::Ptr<SimParticle> const& ptr) {
    straw->emplace_back(StrawId(0, 0, 0), StrawGasStep::StepType(StrawGasStep::StepType::line, StrawGasStep::StepType::minion),
                        energy, 1.f, 0.f, 0., start, end, momentum, ptr);
  };
  if (ordinal_ == 0) {
    tracker(2e-6f, badPtr_ ? bad : ap); tracker(5e-7f, bp);
    crv->emplace_back(CRSScintillatorBarIndex(0), 2e-6f, 0., 1., start, end,
                      XYZVectorF(3.f, 4.f, 0.f), 5.f, 1.f, bp);
  } else if (ordinal_ == 1) {
    tracker(5e-7f, ap);
    calo->emplace_back(0, bp, 1, 0.f, .02f, .02f, 1.f, position);
  } else if (ordinal_ == 3) {
    tracker(5e-7f, ap);
  }
  e.put(std::move(a), "particlesA"); e.put(std::move(b), "particlesB");
  e.put(std::move(straw), "tracker"); e.put(std::move(calo), "calo"); e.put(std::move(crv), "crv");
  ++ordinal_; ++inSubRun_;
}

void FlashBoardFixture::endSubRun(art::SubRun& sr) {
  if (inSubRun_ != 2 || subruns_ >= 2 || sr.run() != 1 || sr.subRun() != subruns_ + 1)
    throw cet::exception("FlashBoardFixture") << "Fixture requires two events per complete SubRun";
  // Exposure is deliberately independent of the two events supplied to detector selection.
  sr.put(std::make_unique<GenEventCount>(subruns_ == 0 ? 10UL : 20UL), art::fullSubRun());
  inSubRun_ = 0; ++subruns_;
}

void FlashBoardFixture::endJob() {
  if (ordinal_ != 4 || subruns_ != 2 || inSubRun_)
    throw cet::exception("FlashBoardFixture") << "Incomplete four-event/two-SubRun fixture";
}
}

DEFINE_ART_MODULE(mu2e::FlashBoardFixture)
