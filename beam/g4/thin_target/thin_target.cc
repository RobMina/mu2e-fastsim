// Thin-target interaction bank generator: single hadronic interactions of one projectile species in one material,
// per log10(kinetic energy) bin, with no geometry other than an infinite block of that material.
//
//   thin_target MODE SPECIES MATERIAL NBINS LOG10_KE_LO LOG10_KE_HI EVENTS_PER_BIN SEED OUT [PHYSICS_LIST]
//     MODE inel : only the projectile's inelastic process is active; the first step ends in an interaction. Each
//                 event writes the interaction's secondaries in the projectile frame (projectile along +z).
//     MODE el   : only hadronic elastic is active; each event writes cos(theta) of the first scatter.
//     MODE xs   : no events; writes per bin (at the log centre) the macroscopic inelastic and elastic cross
//                 sections [1/mm], the unrestricted stopping power [MeV/mm] and the radiation length [mm].
//   SPECIES: proton neutron pi+ pi- kaon+ kaon- (and mu+ mu- in xs mode); MATERIAL: a NIST name (G4_W) or one of the composites below.
//
// Secondary classes (inel): p, n, pi+-, K+-, mu+- are written (p and n only above STORE_KE_MIN_NUCLEON, the rest
// add their kinetic energy to the local deposit). Short-lived particles (0 < lifetime < 1 ns: K0S, Lambda, Sigma,
// eta, ...) are decayed at the vertex with their Geant4 decay tables, recursively. Photons, e+-, pi0 and nuclear
// fragments add their kinetic energy to the local deposit; neutrinos, K0L and antinucleons are counted as escaping.
//
// Binary output (little endian), per event: int32 bin, int32 nsec, float32 ke_projectile [MeV], float32 edep [MeV],
// float32 escaping [MeV], then nsec x (int32 pdg, float32 px, py, pz [MeV/c]). el mode: int32 bin, float32 ke,
// float32 cos_theta per event. xs mode: text, one line per bin.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4DecayProducts.hh"
#include "G4DecayTable.hh"
#include "G4DynamicParticle.hh"
#include "G4EmCalculator.hh"
#include "G4Event.hh"
#include "G4HadronicProcessStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4ProcessManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4UserEventAction.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VDecayChannel.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

namespace {

const double STORE_KE_MIN_NUCLEON = 50.0 * MeV;   // below: deposit (the transport cut is 150 MeV)
const std::set<int> STORED = {2212, 2112, 211, -211, 321, -321, 13, -13};

struct Config {
  std::string mode, species, material, out, physics = "QGSP_BERT_HP";
  int nbins = 0, events = 0;
  long seed = 1;
  double lo = 0, hi = 0;
  double edge(int i) const { return lo + (hi - lo) * i / nbins; }
};

G4Material* make_material(const std::string& name) {
  auto* nist = G4NistManager::Instance();
  if (name.rfind("G4_", 0) == 0) return nist->FindOrBuildMaterial(name);
  auto el = [&](const char* s) { return nist->FindOrBuildElement(s); };
  if (name == "Inconel718") {   // UNS N07718 nominal
    auto* m = new G4Material(name, 8.19 * g / cm3, 7);
    m->AddElement(el("Ni"), 0.525); m->AddElement(el("Cr"), 0.19); m->AddElement(el("Fe"), 0.185);
    m->AddElement(el("Nb"), 0.051); m->AddElement(el("Mo"), 0.03); m->AddElement(el("Ti"), 0.009);
    m->AddElement(el("Al"), 0.01);
    return m;
  }
  if (name == "WL10") {         // W + 1 % La2O3 (the G4beamline deck's WL10: W 0.99, La 0.0085, O 0.0015)
    auto* m = new G4Material(name, 18.75 * g / cm3, 3);
    m->AddElement(el("W"), 0.99); m->AddElement(el("La"), 0.0085); m->AddElement(el("O"), 0.0015);
    return m;
  }
  if (name == "Stainless316") {
    auto* m = new G4Material(name, 8.00 * g / cm3, 5);
    m->AddElement(el("Fe"), 0.655); m->AddElement(el("Cr"), 0.17); m->AddElement(el("Ni"), 0.12);
    m->AddElement(el("Mo"), 0.025); m->AddElement(el("Mn"), 0.03);
    return m;
  }
  std::cerr << "unknown material " << name << "\n";
  std::exit(2);
}

class World : public G4VUserDetectorConstruction {
 public:
  explicit World(G4Material* m) : mat_(m) {}
  G4VPhysicalVolume* Construct() override {
    auto* box = new G4Box("world", 1 * km, 1 * km, 1 * km);
    auto* lv = new G4LogicalVolume(box, mat_, "world");
    return new G4PVPlacement(nullptr, {}, lv, "world", nullptr, false, 0);
  }
 private:
  G4Material* mat_;
};

struct State {
  const Config* cfg = nullptr;
  int bin = 0;
  double ke = 0;
  std::FILE* out = nullptr;
  std::string target_process;
  long written = 0, no_interaction = 0;
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun(State& s, G4ParticleDefinition* def) : s_(s), gun_(1) { gun_.SetParticleDefinition(def); }
  void GeneratePrimaries(G4Event* ev) override {
    const auto& c = *s_.cfg;
    double l = c.edge(s_.bin) + G4UniformRand() * (c.edge(s_.bin + 1) - c.edge(s_.bin));
    s_.ke = std::pow(10.0, l) * MeV;
    gun_.SetParticleEnergy(s_.ke);
    gun_.SetParticlePosition({0, 0, 0});
    gun_.SetParticleMomentumDirection({0, 0, 1});
    gun_.GeneratePrimaryVertex(ev);
  }
 private:
  State& s_;
  G4ParticleGun gun_;
};

struct Final {
  std::vector<int32_t> pdg;
  std::vector<float> p;
  double edep = 0, escaping = 0;
};

void classify(const G4ParticleDefinition* d, const G4LorentzVector& p4, Final& f, int depth = 0) {
  const int pdg = d->GetPDGEncoding();
  const double m = d->GetPDGMass(), ke = p4.e() - m;
  const double tau = d->GetPDGLifeTime();
  const bool short_lived = !d->GetPDGStable() && tau > 0 && tau < 1 * ns && d->GetDecayTable() && depth < 8;
  if (STORED.count(pdg)) {
    if ((pdg == 2212 || pdg == 2112) && ke < STORE_KE_MIN_NUCLEON) { f.edep += ke; return; }
    f.pdg.push_back(pdg);
    f.p.push_back(p4.px() / MeV); f.p.push_back(p4.py() / MeV); f.p.push_back(p4.pz() / MeV);
    return;
  }
  if (short_lived && pdg != 111) {
    G4VDecayChannel* ch = d->GetDecayTable()->SelectADecayChannel(m);
    if (ch) {
      G4DecayProducts* prod = ch->DecayIt(m);
      prod->Boost(p4.e(), p4.vect().unit());
      for (G4int i = 0; i < prod->entries(); ++i) {
        const G4DynamicParticle* dp = (*prod)[i];
        classify(dp->GetDefinition(), dp->Get4Momentum(), f, depth + 1);
      }
      delete prod;
      return;
    }
  }
  const int apdg = std::abs(pdg);
  if (apdg == 12 || apdg == 14 || apdg == 16 || pdg == 130 || pdg == -2212 || pdg == -2112) { f.escaping += p4.e(); return; }
  f.edep += std::max(ke, 0.0);   // gamma, e+-, pi0, nuclear fragments, other stable hadrons
}

class Stepper : public G4UserSteppingAction {
 public:
  explicit Stepper(State& s) : s_(s) {}
  void UserSteppingAction(const G4Step* step) override {
    G4Track* t = step->GetTrack();
    if (t->GetTrackID() != 1) { t->SetTrackStatus(fStopAndKill); return; }
    const auto* proc = step->GetPostStepPoint()->GetProcessDefinedStep();
    const std::string name = proc ? proc->GetProcessName() : "";
    if (name != s_.target_process) {
      if (step->GetPostStepPoint()->GetStepStatus() == fWorldBoundary) ++s_.no_interaction;
      return;
    }
    const auto& c = *s_.cfg;
    if (c.mode == "el") {
      const float cz = static_cast<float>(step->GetPostStepPoint()->GetMomentumDirection().z());
      const float ke = static_cast<float>(s_.ke / MeV);
      std::fwrite(&s_.bin, 4, 1, s_.out); std::fwrite(&ke, 4, 1, s_.out); std::fwrite(&cz, 4, 1, s_.out);
    } else {
      Final f;
      for (const G4Track* sec : *step->GetSecondaryInCurrentStep())
        classify(sec->GetDefinition(), sec->GetDynamicParticle()->Get4Momentum(), f);
      if (t->GetTrackStatus() == fAlive || t->GetTrackStatus() == fStopButAlive)   // leading particle kept by G4
        classify(t->GetDefinition(), t->GetDynamicParticle()->Get4Momentum(), f);
      const int32_t n = static_cast<int32_t>(f.pdg.size());
      const float hdr[3] = {static_cast<float>(s_.ke / MeV), static_cast<float>(f.edep / MeV),
                            static_cast<float>(f.escaping / MeV)};
      std::fwrite(&s_.bin, 4, 1, s_.out); std::fwrite(&n, 4, 1, s_.out); std::fwrite(hdr, 4, 3, s_.out);
      for (int i = 0; i < n; ++i) { std::fwrite(&f.pdg[i], 4, 1, s_.out); std::fwrite(&f.p[3 * i], 4, 3, s_.out); }
    }
    ++s_.written;
    t->SetTrackStatus(fStopAndKill);
  }
 private:
  State& s_;
};

class Stacker : public G4UserStackingAction {   // secondaries are recorded at the vertex, never tracked
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override { return t->GetParentID() == 0 ? fUrgent : fKill; }
};

std::string inelastic_name(const std::string& s) {
  if (s == "proton") return "protonInelastic";
  if (s == "neutron") return "neutronInelastic";
  if (s == "pi+" || s == "pi-") return s + "Inelastic";
  if (s == "kaon+" || s == "kaon-") return s + "Inelastic";
  std::cerr << "unknown species " << s << "\n";
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 10) {
    std::cerr << "usage: thin_target MODE SPECIES MATERIAL NBINS LOG10_KE_LO LOG10_KE_HI EVENTS_PER_BIN SEED OUT [LIST]\n";
    return 2;
  }
  Config c;
  c.mode = argv[1]; c.species = argv[2]; c.material = argv[3]; c.nbins = std::atoi(argv[4]);
  c.lo = std::atof(argv[5]); c.hi = std::atof(argv[6]); c.events = std::atoi(argv[7]); c.seed = std::atol(argv[8]);
  c.out = argv[9];
  if (argc > 10) c.physics = argv[10];
  if (c.mode != "inel" && c.mode != "el" && c.mode != "xs") { std::cerr << "bad mode\n"; return 2; }
  G4Random::setTheSeed(c.seed);

  auto* rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  G4Material* mat = make_material(c.material);
  rm->SetUserInitialization(new World(mat));
  G4PhysListFactory factory;
  rm->SetUserInitialization(factory.GetReferencePhysList(c.physics));
  State s;
  s.cfg = &c;
  const bool muon = c.species == "mu+" || c.species == "mu-";
  if (muon && c.mode != "xs") { std::cerr << "muons: xs mode only (loss and scattering)\n"; return 2; }
  s.target_process = c.mode == "el" ? "hadElastic" : muon ? "" : inelastic_name(c.species);
  G4ParticleDefinition* def = G4ParticleTable::GetParticleTable()->FindParticle(c.species);
  if (!def) { std::cerr << "no particle " << c.species << "\n"; return 2; }
  rm->SetUserAction(new Gun(s, def));
  rm->SetUserAction(new Stepper(s));
  rm->SetUserAction(new Stacker);
  rm->Initialize();

  G4ProcessManager* pm = def->GetProcessManager();
  G4ProcessVector* pv = pm->GetProcessList();
  bool found = false;
  for (G4int i = 0; i < (G4int)pv->size(); ++i) {
    G4VProcess* p = (*pv)[i];
    const std::string n = p->GetProcessName();
    if (n == s.target_process) found = true;
    // xs mode keeps every process active: G4EmCalculator returns zero stopping power for inactivated EM processes
    if (c.mode != "xs" && n != "Transportation" && n != s.target_process) pm->SetProcessActivation(p, false);
  }
  if (!found && c.mode != "xs") { std::cerr << "process " << s.target_process << " not found\n"; return 3; }
  rm->BeamOn(0);   // builds the physics tables

  if (c.mode == "xs") {
    std::ofstream o(c.out);
    o << "# bin log10_ke_center sigma_inel[1/mm] sigma_el[1/mm] dedx[MeV/mm] x0[mm] species=" << c.species
      << " material=" << c.material << " list=" << c.physics << "\n";
    auto* store = G4HadronicProcessStore::Instance();
    G4EmCalculator calc;
    for (int b = 0; b < c.nbins; ++b) {
      const double lc = 0.5 * (c.edge(b) + c.edge(b + 1)), e = std::pow(10.0, lc) * MeV;
      const double si = muon ? 0.0 : store->GetInelasticCrossSectionPerVolume(def, e, mat) * mm;
      const double se = muon ? 0.0 : store->GetElasticCrossSectionPerVolume(def, e, mat) * mm;
      const double dedx = calc.ComputeTotalDEDX(e, def, mat) / (MeV / mm);
      char line[256];
      std::snprintf(line, sizeof line, "%d %.6f %.8e %.8e %.8e %.8e\n", b, lc, si, se, dedx, mat->GetRadlen() / mm);
      o << line;
    }
    std::cout << "xs written: " << c.out << "\n";
    delete rm;
    return 0;
  }

  s.out = std::fopen(c.out.c_str(), "wb");
  if (!s.out) { std::cerr << "cannot open " << c.out << "\n"; return 2; }
  for (s.bin = 0; s.bin < c.nbins; ++s.bin) rm->BeamOn(c.events);
  std::fclose(s.out);
  // A projectile can leave the 1 km world without the selected process (low-energy pi+ below the Coulomb barrier,
  // elastic at tiny cross sections): those events write nothing; the bank builder zeroes cells without interactions.
  std::cout << "events written " << s.written << " no_interaction " << s.no_interaction << "\n";
  delete rm;
  return s.written + s.no_interaction == (long)c.nbins * c.events ? 0 : 4;
}
