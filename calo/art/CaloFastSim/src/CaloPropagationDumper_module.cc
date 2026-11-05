#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "Offline/CaloMC/inc/CaloPhotonPropagation.hh"
#include "Offline/CalorimeterGeom/inc/Calorimeter.hh"
#include "Offline/ConfigTools/inc/ConfigFileLookupPolicy.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "CLHEP/Random/NonRandomEngine.h"
#include "TFile.h"
#include "TH2F.h"
#include "TTree.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace mu2e {

class CaloPropagationDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<std::string> file{fhicl::Name("file")};
    fhicl::Atom<std::string> histogram{fhicl::Name("histogram")};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit CaloPropagationDumper(Parameters const& config) : art::EDAnalyzer(config), conf_(config()) {}
  void analyze(art::Event const&) override;

 private:
  class SuppliedEngine : public CLHEP::NonRandomEngine {
   public:
    double flat() override { ++draws; return CLHEP::NonRandomEngine::flat(); }
    std::uint64_t draws{0};
  };
  Config conf_;
  bool completed_{false};
};

void CaloPropagationDumper::analyze(art::Event const&) {
  if (completed_) throw cet::exception("CaloPropagationDumper") << "exactly one control event required";
  completed_ = true;
  ConfigFileLookupPolicy resolve;
  std::string resolvedFile = resolve(conf_.file());
  std::string histogram = conf_.histogram();
  TFile file(resolvedFile.c_str());
  if (!file.IsOpen()) throw cet::exception("CaloPropagationDumper") << "cannot open " << resolvedFile;
  std::unique_ptr<TH2F> hist(dynamic_cast<TH2F*>(file.Get(histogram.c_str())));
  if (!hist) throw cet::exception("CaloPropagationDumper") << "TH2F required";
  hist->SetDirectory(nullptr);
  file.Close();

  int nz = hist->GetNbinsX(), nt = hist->GetNbinsY();
  float dz = hist->GetXaxis()->GetBinWidth(1);
  if (nz <= 0 || nt <= 0 || !std::isfinite(dz) || dz <= 0)
    throw cet::exception("CaloPropagationDumper") << "invalid axes";
  art::ServiceHandle<art::TFileService> tfs;
  auto metadata = tfs->make<TTree>("metadata", "resolved ROOT and geometry inputs; no stock private cache access");
  double refractiveIndex = GeomHandle<Calorimeter>()->G4Info().get<double>("refractiveIndex");
  metadata->Branch("file", &resolvedFile); metadata->Branch("histogram", &histogram);
  metadata->Branch("nz", &nz); metadata->Branch("nt", &nt); metadata->Branch("dz", &dz);
  metadata->Branch("refractive_index", &refractiveIndex); metadata->Fill();

  auto bins = tfs->make<TTree>("bins", "observed ROOT regular bins; reconstructed CDF is probe design only");
  int iz = 0, it = 0;
  float content = 0, time = 0, threshold = 0;
  double xlow = 0, xup = 0, ylow = 0, yup = 0;
  bins->Branch("iz", &iz); bins->Branch("it", &it); bins->Branch("content", &content);
  bins->Branch("time", &time); bins->Branch("probe_threshold", &threshold);
  bins->Branch("xlow", &xlow); bins->Branch("xup", &xup);
  bins->Branch("ylow", &ylow); bins->Branch("yup", &yup);
  std::vector<float> thresholds;
  thresholds.reserve(nz * nt);
  for (iz = 0; iz < nz; ++iz) {
    std::vector<float> cumulative;
    float sum = 1e-6f;
    for (it = 0; it < nt; ++it) {
      const double count = hist->GetBinContent(iz + 1, it + 1);
      if (!std::isfinite(count) || count < 0) throw cet::exception("CaloPropagationDumper") << "invalid bin";
      sum += count;
      cumulative.push_back(sum);
    }
    if (!std::isfinite(sum)) throw cet::exception("CaloPropagationDumper") << "CDF overflow";
    for (it = 0; it < nt; ++it) {
      content = hist->GetBinContent(iz + 1, it + 1);
      time = hist->GetYaxis()->GetBinCenter(it + 1);
      threshold = cumulative[it] / sum;
      xlow = hist->GetXaxis()->GetBinLowEdge(iz + 1); xup = hist->GetXaxis()->GetBinUpEdge(iz + 1);
      ylow = hist->GetYaxis()->GetBinLowEdge(it + 1); yup = hist->GetYaxis()->GetBinUpEdge(it + 1);
      thresholds.push_back(threshold); bins->Fill();
    }
  }

  SuppliedEngine engine;
  CaloPhotonPropagation stock(conf_.file(), histogram, engine);
  stock.buildTable();
  auto probes = tfs->make<TTree>("probes", "public unmodified stock buildTable/propTimeSimu with supplied engine draws");
  float depth = 0, uniform = 0, observed = 0;
  std::uint64_t before = 0, after = 0;
  probes->Branch("depth", &depth); probes->Branch("uniform", &uniform); probes->Branch("time", &observed);
  probes->Branch("draws_before", &before); probes->Branch("draws_after", &after);
  const auto sample = [&](float z, float u) {
    depth = z; uniform = u;
    const double quotient = std::trunc(static_cast<double>(depth / dz));
    if (!std::isfinite(quotient) || quotient < std::numeric_limits<int>::min() || quotient > std::numeric_limits<int>::max())
      throw cet::exception("CaloPropagationDumper") << "unsafe depth-to-int conversion";
    engine.setNextRandom(static_cast<double>(uniform));
    before = engine.draws;
    observed = stock.propTimeSimu(depth);
    after = engine.draws;
    if (after != before + 1) throw cet::exception("CaloPropagationDumper") << "expected exactly one supplied draw";
    probes->Fill();
  };
  for (iz = 0; iz < nz; ++iz) {
    const float center = (static_cast<float>(iz) + .5f) * dz;
    for (it = 0; it < nt; ++it) {
      const float u = thresholds[iz * nt + it];
      sample(center, std::nextafter(u, 0.f)); sample(center, u); sample(center, std::nextafter(u, 1.f));
    }
  }
  const float uniforms[] = {0.f, std::nextafter(0.f, 1.f), .25f, .5f, .75f, 1.f};
  for (float u : uniforms) sample(-100.f * dz, u);
  for (iz = 0; iz <= nz; ++iz) {
    const float boundary = static_cast<float>(iz) * dz;
    const float depths[] = {std::nextafter(boundary, -std::numeric_limits<float>::infinity()), boundary,
                            std::nextafter(boundary, std::numeric_limits<float>::infinity())};
    for (float z : depths) for (float u : uniforms) sample(z, u);
  }
  for (float u : uniforms) sample(static_cast<float>(nz + 100) * dz, u);
  metadata->ResetBranchAddresses(); bins->ResetBranchAddresses(); probes->ResetBranchAddresses();
}

}
DEFINE_ART_MODULE(mu2e::CaloPropagationDumper)
