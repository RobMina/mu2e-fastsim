// Deterministic extraction fixture only; never schedules or substitutes for particle transport.
#include <cmath>
#include <memory>
#include <string>

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/SubRun.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/PhysicalVolumeInfoMultiCollection.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StatusG4.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"

namespace mu2e {
class FlashReferenceFixture : public art::EDProducer {
 public:
  struct Config {
    fhicl::Atom<bool> failStatus{fhicl::Name("failStatus"), fhicl::Comment("Inject a retained field-propagator failure"), false};
    fhicl::Atom<bool> unavailablePtr{fhicl::Name("unavailablePtr"), fhicl::Comment("Inject an unavailable nonnull particle Ptr"), false};
  };
  using Parameters = art::EDProducer::Table<Config>;
  explicit FlashReferenceFixture(Parameters const& c) : art::EDProducer(c), fail_(c().failStatus()), unavailable_(c().unavailablePtr()) {
    produces<GenParticleCollection>();
    produces<SimParticleCollection>("input"); produces<SimParticleCollection>("continued");
    produces<StepPointMCCollection>("source"); produces<StepPointMCCollection>("photon"); produces<StepPointMCCollection>("tracker");
    produces<StatusG4>(); produces<PhysicalVolumeInfoMultiCollection, art::InSubRun>();
  }
  void produce(art::Event&) override;
  void beginSubRun(art::SubRun&) override;
 private:
  bool fail_, unavailable_;
};

void FlashReferenceFixture::beginSubRun(art::SubRun& sr) {
  auto v = std::make_unique<PhysicalVolumeInfoMultiCollection>(3);
  (*v)[0].insert({cet::map_vector_key(7), PhysicalVolumeInfo("upstream", 9, "Vacuum")});
  (*v)[2].insert({cet::map_vector_key(11), PhysicalVolumeInfo("foil", 9, "Aluminum")});
  sr.put(std::move(v), art::fullSubRun());
}

void FlashReferenceFixture::produce(art::Event& e) {
  auto gen = std::make_unique<GenParticleCollection>();
  auto input = std::make_unique<SimParticleCollection>();
  auto continued = std::make_unique<SimParticleCollection>();
  auto source = std::make_unique<StepPointMCCollection>();
  auto photon = std::make_unique<StepPointMCCollection>();
  auto tracker = std::make_unique<StepPointMCCollection>();
  auto gp = e.getProductID<GenParticleCollection>();
  auto ip = e.getProductID<SimParticleCollection>("input");
  auto cp = e.getProductID<SimParticleCollection>("continued");
  art::Ptr<GenParticle> gptr(gp, 0, e.productGetter(gp));
  art::Ptr<SimParticle> inptr(ip, 17, e.productGetter(ip));
  art::Ptr<SimParticle> copied(cp, 17, e.productGetter(cp));
  art::Ptr<SimParticle> primary(cp, 101, e.productGetter(cp));
  art::Ptr<SimParticle> secondPrimary(cp, 103, e.productGetter(cp));
  art::Ptr<SimParticle> gamma(cp, 102, e.productGetter(cp));
  if (e.event() % 2) {
    CLHEP::Hep3Vector const x(1.0 + std::ldexp(1., -40), -0.0, 4179.);
    CLHEP::HepLorentzVector const p(0.125, -0.25, 4., std::sqrt(16.078125 + 0.51099895 * 0.51099895));
    double const time = 10331490143436802.;
    gen->emplace_back(PDGCode::e_minus, GenId::particleGun, x, p, time, 0.125);
    SimParticle a(cet::map_vector_key(17), 0, {}, PDGCode::e_minus, gptr, x, p,
                  static_cast<float>(time), .125f, 7, 0, ProcessCode::mu2ePrimary);
    a.addEndInfo(x, p, static_cast<float>(time), .25f, 7, 2, ProcessCode::Transportation, 3.5f, 2, .5f);
    input->insert({cet::map_vector_key(17), a});
    a.addDaughter(secondPrimary); a.addDaughter(primary); continued->insert({cet::map_vector_key(17), a});
    SimParticle b(cet::map_vector_key(101), 2, copied, PDGCode::e_minus, {}, x, p,
                  static_cast<float>(time), .125f, 11, 0, ProcessCode::mu2ePrimary);
    b.addEndInfo(x + CLHEP::Hep3Vector(0., 0., 1.), p, static_cast<float>(time), .25f, 11, 2,
                 ProcessCode::eBrem, 3.f, 3, .75f);
    b.addDaughter(gamma); continued->insert({cet::map_vector_key(101), b});
    SimParticle c(cet::map_vector_key(102), 2, primary, PDGCode::gamma, {}, x,
                  CLHEP::HepLorentzVector(0., 0., .5, .5), static_cast<float>(time), 0.f, 11, 0,
                  ProcessCode::eBrem, SimParticle::IonDetail(.0625f, 3));
    c.addEndInfo(x + CLHEP::Hep3Vector(0., 0., 2.), CLHEP::HepLorentzVector(0., 0., .5, .5),
                 static_cast<float>(time), 0.f, 11, 2, ProcessCode::Transportation, .5f, 1, 2.f);
    continued->insert({cet::map_vector_key(102), c});
    SimParticle d(cet::map_vector_key(103), 2, copied, PDGCode::e_minus, {}, x, p,
                  static_cast<float>(time + 2), .125f, 11, 0, ProcessCode::mu2ePrimary);
    d.addEndInfo(x, p, static_cast<float>(time + 2), .25f, 11, 2, ProcessCode::Transportation, 3.f, 1, .5f);
    continued->insert({cet::map_vector_key(103), d});
    for (unsigned i = 0; i < 2; ++i)
      source->emplace_back(inptr, 9, .0, .0, .0, time + 2 * i, .125,
                           x, x + CLHEP::Hep3Vector(0., 0., .5), p.vect(), p.vect(), .5, ProcessCode::Transportation);
    photon->emplace_back(gamma, 9, .125, .03125, .015625, time, 0., x,
                         x + CLHEP::Hep3Vector(.25, -.5, 2.), CLHEP::Hep3Vector(0., 0., .5),
                         CLHEP::Hep3Vector(0., .25, .25), 2.125, ProcessCode::compt);
    if (unavailable_) {
      art::Ptr<SimParticle> bad(cp, 999, e.productGetter(cp));
      tracker->emplace_back(bad, 123, .25, .125, .0625, time, .25, x, x, p.vect(), p.vect(), .5, ProcessCode::eIoni);
    } else {
      tracker->emplace_back(primary, 123, .25, .125, .0625, time, .25, x, x, p.vect(), p.vect(), .5, ProcessCode::eIoni);
    }
  }
  auto status = std::make_unique<StatusG4>(fail_ ? 1 : 0, continued->size() ? 3 : 0, false, 0, fail_ ? 1 : 0, .0625f, .125f);
  e.put(std::move(gen)); e.put(std::move(input), "input"); e.put(std::move(continued), "continued");
  e.put(std::move(source), "source"); e.put(std::move(photon), "photon"); e.put(std::move(tracker), "tracker");
  e.put(std::move(status));
}
}

DEFINE_ART_MODULE(mu2e::FlashReferenceFixture)
