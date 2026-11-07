// Export stored aggregated accessors without changing transport, aggregation or selection.
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Provenance.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "Offline/MCDataProducts/inc/CaloShowerStep.hh"
#include "Offline/MCDataProducts/inc/CrvStep.hh"
#include "Offline/MCDataProducts/inc/GenEventCount.hh"
#include "Offline/MCDataProducts/inc/StrawGasStep.hh"

namespace mu2e {
namespace flashboard {
using U32 = UInt_t;
using U64 = ULong64_t;
using F32 = Float_t;
using EventKey = std::tuple<U32, U32, U32>;
using SubRunKey = std::pair<U32, U32>;
using PIDValue = decltype(std::declval<art::ProductID const&>().value());
using PtrKey = decltype(std::declval<art::Ptr<SimParticle> const&>().key());
static_assert(sizeof(U32) == 4 && sizeof(U64) == 8 && sizeof(F32) == 4);
static_assert(std::is_unsigned_v<PIDValue> && sizeof(PIDValue) <= sizeof(U64));
static_assert(std::is_unsigned_v<PtrKey> && sizeof(PtrKey) <= sizeof(U64));
static_assert(sizeof(std::size_t) <= sizeof(U64));
static_assert(std::is_unsigned_v<GenEventCount::count_t> && sizeof(GenEventCount::count_t) <= sizeof(U64));
// A different backing accessor contract must fail the build, not silently round or recompute.
static_assert(std::is_same_v<decltype(std::declval<StrawGasStep const&>().ionizingEdep()), float>);
static_assert(std::is_same_v<decltype(std::declval<StrawGasStep const&>().momentum().R()), float>);
static_assert(std::is_same_v<decltype(std::declval<CaloShowerStep const&>().energyDepBirks()), float>);
static_assert(std::is_same_v<decltype(std::declval<CaloShowerStep const&>().momentumIn()), float>);
static_assert(std::is_same_v<decltype(std::declval<CrvStep const&>().visibleEDep()), float>);
static_assert(std::is_same_v<decltype(std::declval<CrvStep const&>().startMom().R()), float>);

U64 add(U64 a, U64 b) {
  if (b > std::numeric_limits<U64>::max() - a)
    throw cet::exception("FlashBoardExport") << "Export count overflow";
  return a + b;
}

struct ProvenanceRow {
  U64 product{};
  std::string tag, label, instance, process, branch;
  void bind(TTree* t) {
    t->Branch("product", &product); t->Branch("tag", &tag);
    t->Branch("label", &label); t->Branch("instance", &instance);
    t->Branch("process", &process); t->Branch("branch", &branch);
  }
  void assign(art::Provenance const* p, art::InputTag const& input) {
    if (!p) throw cet::exception("FlashBoardExport") << "Missing product provenance for " << input.encode();
    product = p->productID().value(); tag = input.encode();
    label = p->moduleLabel(); instance = p->productInstanceName();
    process = p->processName(); branch = p->branchName();
    if (!product || label.empty() || process.empty() || branch.empty())
      throw cet::exception("FlashBoardExport") << "Invalid resolved provenance for " << tag;
  }
};

struct ProductRow : ProvenanceRow {
  U32 detector{}, slot{};
  U64 size{};
  void bind(TTree* t) {
    ProvenanceRow::bind(t); t->Branch("detector", &detector);
    t->Branch("slot", &slot); t->Branch("size", &size);
  }
};

struct StepRow {
  U64 product{}, index{}, particle_product{}, particle_key{};
  F32 energy{}, momentum{};
  void bind(TTree* t, char const* energyName) {
    t->Branch("product", &product); t->Branch("index", &index);
    t->Branch("particle_product", &particle_product); t->Branch("particle_key", &particle_key);
    t->Branch(energyName, &energy); t->Branch("momentum_mev_c", &momentum);
  }
  void assign(art::Ptr<SimParticle> const& p, F32 e, F32 m) {
    // Availability alone need not detect an absent map key; force pointer resolution too.
    if (p.isNull() || !p.isAvailable() || !p.get())
      throw cet::exception("FlashBoardExport") << "Null or unavailable SimParticle pointer at product " << product << " index " << index;
    particle_product = p.id().value(); particle_key = p.key();
    if (!particle_product)
      throw cet::exception("FlashBoardExport") << "Invalid SimParticle ProductID";
    if (!std::isfinite(e) || e < 0 || !std::isfinite(m) || m < 0)
      throw cet::exception("FlashBoardExport") << "Nonfinite or negative accessor at product " << product << " index " << index;
    energy = e; momentum = m;
  }
};
}

class FlashBoardExporter : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Sequence<art::InputTag> strawSteps{fhicl::Name("strawSteps"), fhicl::Comment("Nonempty required StrawGasStep collections")};
    fhicl::Sequence<art::InputTag> caloSteps{fhicl::Name("caloSteps"), fhicl::Comment("Nonempty required CaloShowerStep collections")};
    fhicl::Sequence<art::InputTag> crvSteps{fhicl::Name("crvSteps"), fhicl::Comment("Nonempty required CrvStep collections")};
    fhicl::Atom<art::InputTag> generatedEvents{fhicl::Name("generatedEvents"), fhicl::Comment("Exact SubRun GenEventCount tag; no inferred exposure")};
    fhicl::Atom<ULong64_t> expectedEvents{fhicl::Name("expectedEvents"), fhicl::Comment("Positive exact exported event count"), 0};
    fhicl::Atom<ULong64_t> expectedSubRuns{fhicl::Name("expectedSubRuns"), fhicl::Comment("Positive exact exported SubRun count"), 0};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit FlashBoardExporter(Parameters const&);
  void beginJob() override;
  void analyze(art::Event const&) override;
  void endSubRun(art::SubRun const&) override;
  void endJob() override;
 private:
  using U32 = flashboard::U32;
  using U64 = flashboard::U64;
  void fill(TTree* t) {
    if (t->Fill() < 0) {
      failed_ = true;
      throw cet::exception("FlashBoardExport") << "TTree write failure: " << t->GetName();
    }
  }
  void bindEvent(TTree* t) {
    t->Branch("run", &run_); t->Branch("subrun", &subrun_); t->Branch("event", &event_);
  }
  template<class Collection, class Energy, class Momentum>
  void collect(art::Event const&, std::vector<art::InputTag> const&, U32,
               std::set<U64>&, std::vector<flashboard::ProductRow>&,
               std::vector<flashboard::StepRow>&, Energy, Momentum);
  std::vector<art::InputTag> strawTags_, caloTags_, crvTags_;
  art::InputTag generatedTag_;
  U64 expectedEvents_, expectedSubRuns_;
  U32 run_{}, subrun_{}, event_{}, version_{1}, complete_{}, requirementDetector_{}, requirementSlot_{};
  U64 nproducts_{}, nstraw_{}, ncalo_{}, ncrv_{}, generated_{};
  U64 eventsCount_{}, subrunsCount_{}, strawCount_{}, caloCount_{}, crvCount_{};
  bool failed_{false};
  std::string requirementTag_;
  std::set<flashboard::EventKey> eventIDs_;
  std::set<flashboard::SubRunKey> subrunIDs_;
  flashboard::ProductRow product_;
  flashboard::ProvenanceRow counter_;
  flashboard::StepRow step_;
  TTree *schema_{}, *requirements_{}, *events_{}, *products_{}, *straw_{}, *calo_{}, *crv_{}, *subruns_{}, *completion_{};
};

FlashBoardExporter::FlashBoardExporter(Parameters const& c)
    : art::EDAnalyzer(c), strawTags_(c().strawSteps()), caloTags_(c().caloSteps()),
      crvTags_(c().crvSteps()), generatedTag_(c().generatedEvents()),
      expectedEvents_(c().expectedEvents()), expectedSubRuns_(c().expectedSubRuns()) {
  if (!expectedEvents_ || !expectedSubRuns_)
    throw cet::exception("FlashBoardExport") << "expectedEvents and expectedSubRuns must be positive exact counts";
  for (auto const* tags : {&strawTags_, &caloTags_, &crvTags_}) {
    if (tags->empty() || tags->size() > std::numeric_limits<U32>::max())
      throw cet::exception("FlashBoardExport") << "Each detector requires a nonempty uint32-slot tag list";
    std::set<std::string> seen;
    for (auto const& tag : *tags) {
      if (tag.label().empty() || !seen.insert(tag.encode()).second)
        throw cet::exception("FlashBoardExport") << "Empty or duplicate detector InputTag: " << tag.encode();
    }
  }
  if (generatedTag_.label().empty())
    throw cet::exception("FlashBoardExport") << "generatedEvents requires a nonempty InputTag";
  for (auto const& tag : strawTags_) consumes<StrawGasStepCollection>(tag);
  for (auto const& tag : caloTags_) consumes<CaloShowerStepCollection>(tag);
  for (auto const& tag : crvTags_) consumes<CrvStepCollection>(tag);
  consumes<GenEventCount, art::InSubRun>(generatedTag_);
}

void FlashBoardExporter::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  schema_ = tfs->make<TTree>("schema", "Aggregated accessor schema v1");
  schema_->Branch("version", &version_); schema_->Branch("expected_events", &expectedEvents_);
  schema_->Branch("expected_subruns", &expectedSubRuns_); fill(schema_);
  requirements_ = tfs->make<TTree>("requirements", "Every configured collection is mandatory");
  requirements_->Branch("detector", &requirementDetector_); requirements_->Branch("slot", &requirementSlot_);
  requirements_->Branch("tag", &requirementTag_);
  requirementDetector_ = 0;
  for (auto const* tags : {&strawTags_, &caloTags_, &crvTags_}) {
    ++requirementDetector_; requirementSlot_ = 0;
    for (auto const& tag : *tags) {
      requirementTag_ = tag.encode(); fill(requirements_); ++requirementSlot_;
    }
  }
  events_ = tfs->make<TTree>("events", "Fully exported events including empty events"); bindEvent(events_);
  events_->Branch("nproducts", &nproducts_); events_->Branch("nstraw", &nstraw_);
  events_->Branch("ncalo", &ncalo_); events_->Branch("ncrv", &ncrv_);
  products_ = tfs->make<TTree>("products", "Resolved configured collections including empty collections");
  bindEvent(products_); product_.bind(products_);
  straw_ = tfs->make<TTree>("straw_steps", "Stored StrawGasStep accessors"); bindEvent(straw_); step_.bind(straw_, "ionizing_mev");
  calo_ = tfs->make<TTree>("calo_steps", "Stored CaloShowerStep accessors"); bindEvent(calo_); step_.bind(calo_, "birks_mev");
  crv_ = tfs->make<TTree>("crv_steps", "Stored CrvStep accessors"); bindEvent(crv_); step_.bind(crv_, "visible_mev");
  subruns_ = tfs->make<TTree>("subruns", "Exact generated exposure independent of detector acceptance");
  subruns_->Branch("run", &run_); subruns_->Branch("subrun", &subrun_);
  counter_.bind(subruns_); subruns_->Branch("generated_events", &generated_);
  completion_ = tfs->make<TTree>("completion", "Exact export counts; not a full-input or closed-file certificate");
  completion_->Branch("complete", &complete_); completion_->Branch("events", &eventsCount_);
  completion_->Branch("subruns", &subrunsCount_); completion_->Branch("straw_steps", &strawCount_);
  completion_->Branch("calo_steps", &caloCount_); completion_->Branch("crv_steps", &crvCount_);
}

template<class Collection, class Energy, class Momentum>
void FlashBoardExporter::collect(art::Event const& e, std::vector<art::InputTag> const& tags, U32 detector,
                                 std::set<U64>& seen, std::vector<flashboard::ProductRow>& products,
                                 std::vector<flashboard::StepRow>& steps, Energy energy, Momentum momentum) {
  U32 slot = 0;
  for (auto const& tag : tags) {
    auto const h = e.getValidHandle<Collection>(tag);
    flashboard::ProductRow p;
    p.assign(h.provenance(), tag); p.detector = detector; p.slot = slot++; p.size = h->size();
    if (!seen.insert(p.product).second)
      throw cet::exception("FlashBoardExport") << "Aliased configured ProductID " << p.product << " at " << tag.encode();
    products.push_back(p);
    U64 index = 0;
    for (auto const& s : *h) {
      flashboard::StepRow row;
      row.product = p.product; row.index = index++;
      row.assign(s.simParticle(), energy(s), momentum(s)); steps.push_back(row);
    }
  }
}

void FlashBoardExporter::analyze(art::Event const& e) {
  try {
    if (eventsCount_ >= expectedEvents_)
      throw cet::exception("FlashBoardExport") << "Event count exceeds expectation before " << e.id();
    run_ = e.run(); subrun_ = e.subRun(); event_ = e.event();
    if (!eventIDs_.emplace(run_, subrun_, event_).second)
      throw cet::exception("FlashBoardExport") << "Duplicate EventID: " << e.id();
    // Validate the entire event before writing payload; the event ledger is committed last.
    std::set<U64> seen;
    std::vector<flashboard::ProductRow> products;
    std::vector<flashboard::StepRow> straw, calo, crv;
    collect<StrawGasStepCollection>(e, strawTags_, 1, seen, products, straw,
        [](StrawGasStep const& s) { return s.ionizingEdep(); }, [](StrawGasStep const& s) { return s.momentum().R(); });
    collect<CaloShowerStepCollection>(e, caloTags_, 2, seen, products, calo,
        [](CaloShowerStep const& s) { return s.energyDepBirks(); }, [](CaloShowerStep const& s) { return s.momentumIn(); });
    collect<CrvStepCollection>(e, crvTags_, 3, seen, products, crv,
        [](CrvStep const& s) { return s.visibleEDep(); }, [](CrvStep const& s) { return s.startMom().R(); });
    nproducts_ = products.size(); nstraw_ = straw.size(); ncalo_ = calo.size(); ncrv_ = crv.size();
    auto const nextStraw = flashboard::add(strawCount_, nstraw_);
    auto const nextCalo = flashboard::add(caloCount_, ncalo_);
    auto const nextCrv = flashboard::add(crvCount_, ncrv_);
    for (auto const& row : products) { product_ = row; fill(products_); }
    for (auto const& row : straw) { step_ = row; fill(straw_); }
    for (auto const& row : calo) { step_ = row; fill(calo_); }
    for (auto const& row : crv) { step_ = row; fill(crv_); }
    fill(events_); ++eventsCount_;
    strawCount_ = nextStraw; caloCount_ = nextCalo; crvCount_ = nextCrv;
  } catch (...) { failed_ = true; throw; }
}

void FlashBoardExporter::endSubRun(art::SubRun const& sr) {
  try {
    if (subrunsCount_ >= expectedSubRuns_)
      throw cet::exception("FlashBoardExport") << "SubRun count exceeds expectation before " << sr.id();
    run_ = sr.run(); subrun_ = sr.subRun();
    if (!subrunIDs_.emplace(run_, subrun_).second)
      throw cet::exception("FlashBoardExport") << "Duplicate SubRun callback; fragments are not summed: " << sr.id();
    auto const h = sr.getValidHandle<GenEventCount>(generatedTag_);
    auto const count = h->count();
    if (count == 0 || count == std::numeric_limits<GenEventCount::count_t>::max())
      throw cet::exception("FlashBoardExport") << "Nonpositive or default-sentinel GenEventCount in " << sr.id();
    counter_.assign(h.provenance(), generatedTag_); generated_ = count;
    fill(subruns_); ++subrunsCount_;
  } catch (...) { failed_ = true; throw; }
}

void FlashBoardExporter::endJob() {
  if (failed_ || eventsCount_ != expectedEvents_ || subrunsCount_ != expectedSubRuns_)
    throw cet::exception("FlashBoardExport") << "Incomplete export: events " << eventsCount_ << "/" << expectedEvents_
        << ", SubRuns " << subrunsCount_ << "/" << expectedSubRuns_ << ", failed=" << failed_;
  for (auto const& id : eventIDs_)
    if (!subrunIDs_.count({std::get<0>(id), std::get<1>(id)}))
      throw cet::exception("FlashBoardExport") << "Exported event has no generated SubRun counter";
  if (complete_) throw cet::exception("FlashBoardExport") << "Duplicate endJob callback";
  complete_ = 1; fill(completion_);
}
}

DEFINE_ART_MODULE(mu2e::FlashBoardExporter)
