// Dump the CRV response and reco chain (SiPM charges, digi MC voltages, digis, reco pulses, coincidence clusters,
// conditions) to flat ROOT TTrees: the reference for the crvfastsim.reco fast paths.
#include <bitset>
#include <stdexcept>
#include <string>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/CRVConditions/inc/CRVCalib.hh"
#include "Offline/CRVConditions/inc/CRVOrdinal.hh"
#include "Offline/CRVConditions/inc/CRVStatus.hh"
#include "Offline/DataProducts/inc/CRVId.hh"
#include "Offline/DataProducts/inc/EventWindowMarker.hh"
#include "Offline/MCDataProducts/inc/CrvDigiMC.hh"
#include "Offline/MCDataProducts/inc/CrvSiPMCharges.hh"
#include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/RecoDataProducts/inc/CrvCoincidenceCluster.hh"
#include "Offline/RecoDataProducts/inc/CrvDigi.hh"
#include "Offline/RecoDataProducts/inc/CrvRecoPulse.hh"
#include "Offline/RecoDataProducts/inc/ProtonBunchTime.hh"

namespace mu2e {

class CrvRecoTableDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Atom<art::InputTag> sipmCharges{fhicl::Name("sipmCharges"), art::InputTag("CrvSiPMCharges")};
    fhicl::Atom<art::InputTag> digiMCs{fhicl::Name("digiMCs"), art::InputTag("CrvWaveforms")};
    fhicl::Atom<art::InputTag> digis{fhicl::Name("digis"), art::InputTag("CrvDigi")};
    fhicl::Atom<art::InputTag> pulses{fhicl::Name("pulses"), art::InputTag("CrvRecoPulses")};
    fhicl::Atom<art::InputTag> clusters{fhicl::Name("clusters"), art::InputTag("CrvCoincidenceClusterFinder")};
    fhicl::Atom<art::InputTag> ewm{fhicl::Name("eventWindowMarker"), art::InputTag("EWMProducer")};
    fhicl::Atom<art::InputTag> pbt{fhicl::Name("protonBunchTime"), art::InputTag("EWMProducer")};
    fhicl::Atom<bool> mc{fhicl::Name("mc"), fhicl::Comment("dump SiPM charges and digi MC (false for data)"), true};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit CrvRecoTableDumper(const Parameters& conf);
  void beginJob() override;
  void analyze(const art::Event& event) override;

 private:
  void dumpConditions(const art::Event& event);
  Config conf_;
  ProditionsHandle<CRVCalib> calib_h_;
  ProditionsHandle<CRVStatus> status_h_;
  ProditionsHandle<CRVOrdinal> ordinal_h_;
  bool conditionsDone_{false};
  TTree *events_{nullptr}, *charges_{nullptr}, *digimc_{nullptr}, *digis_{nullptr}, *pulses_{nullptr},
      *clusters_{nullptr}, *calib_{nullptr};
  unsigned run_{0}, subrun_{0}, event_{0};
  // events
  int spill_{-1};
  double evlen_{0}, pbtime_{0};
  int ndigi_{0}, npulse_{0}, ncluster_{0};
  // per channel: bar, SiPM, row index within its collection
  int bar_{0}, sipm_{0}, idx_{0};
  // charges (one row per CrvSiPMCharges = one SiPM)
  std::vector<double> qtime_, qcharge_, qpe_;
  std::vector<int> qsim_;
  // digi MC (one row per CrvDigiMC)
  double mcstart_{0}, mctdc0_{0};
  std::vector<double> volt_;
  int mcsim_{-1};
  int nzs_{0};
  // digis (one row per CrvDigi)
  int starttdc_{0}, roc_{0}, feb_{0}, febch_{0};
  std::vector<short> adc_;
  // pulses (one row per CrvRecoPulse)
  float pe_{0}, peph_{0}, height_{0}, beta_{0}, chi2_{0}, ped_{0};
  double ptime_{0}, letime_{0};
  int seq_{0}, flags_{0}, pedfromdb_{0};
  std::vector<int> wfidx_;
  // clusters
  int sectortype_{0}, poscalc_{0};
  double cstart_{0}, cend_{0}, avgtime_{0};
  float cpe_{0}, slope_{0}, cx_{0}, cy_{0}, cz_{0};
  std::vector<int> layers_, cpulses_;
  float spe0_{0}, spe1_{0};
  double st0_{0}, st1_{0};
  int sh0_{0}, sh1_{0};
  // conditions (one row per channel, first event)
  int chan_{0}, status_{0}, mroc_{-1}, mfeb_{-1}, mfebch_{-1};
  float cped_{0}, carea_{0}, cheight_{0}, ctoff_{0};
};

CrvRecoTableDumper::CrvRecoTableDumper(const Parameters& conf) : art::EDAnalyzer(conf), conf_(conf()) {}

void CrvRecoTableDumper::beginJob() {
  art::ServiceHandle<art::TFileService> tfs;
  auto ids = [this](TTree* t) {
    t->Branch("run", &run_);
    t->Branch("subrun", &subrun_);
    t->Branch("event", &event_);
  };
  auto chan = [this](TTree* t) {
    t->Branch("idx", &idx_);
    t->Branch("bar", &bar_);
    t->Branch("sipm", &sipm_);
  };
  events_ = tfs->make<TTree>("events", "event window");
  ids(events_);
  events_->Branch("spill", &spill_);
  events_->Branch("evlen", &evlen_);
  events_->Branch("pbtime", &pbtime_);
  events_->Branch("ndigi", &ndigi_);
  events_->Branch("npulse", &npulse_);
  events_->Branch("ncluster", &ncluster_);
  if (conf_.mc()) {
    charges_ = tfs->make<TTree>("charges", "CrvSiPMCharges, one row per SiPM");
    ids(charges_);
    chan(charges_);
    charges_->Branch("time", &qtime_);
    charges_->Branch("charge", &qcharge_);
    charges_->Branch("pe", &qpe_);
    charges_->Branch("sim", &qsim_);
    digimc_ = tfs->make<TTree>("digimc", "CrvDigiMC");
    ids(digimc_);
    chan(digimc_);
    digimc_->Branch("start", &mcstart_);
    digimc_->Branch("tdc0", &mctdc0_);
    digimc_->Branch("nzs", &nzs_);
    digimc_->Branch("sim", &mcsim_);
    digimc_->Branch("volt", &volt_);
  }
  digis_ = tfs->make<TTree>("digis", "CrvDigi");
  ids(digis_);
  chan(digis_);
  digis_->Branch("starttdc", &starttdc_);
  digis_->Branch("nzs", &nzs_);
  digis_->Branch("roc", &roc_);
  digis_->Branch("feb", &feb_);
  digis_->Branch("febch", &febch_);
  digis_->Branch("adc", &adc_);
  pulses_ = tfs->make<TTree>("pulses", "CrvRecoPulse");
  ids(pulses_);
  chan(pulses_);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{
           {"pe", &pe_}, {"peph", &peph_}, {"height", &height_}, {"beta", &beta_}, {"chi2", &chi2_}, {"ped", &ped_}})
    pulses_->Branch(n, v);
  pulses_->Branch("time", &ptime_);
  pulses_->Branch("letime", &letime_);
  pulses_->Branch("seq", &seq_);
  pulses_->Branch("flags", &flags_);
  pulses_->Branch("pedfromdb", &pedfromdb_);
  pulses_->Branch("wfidx", &wfidx_);
  clusters_ = tfs->make<TTree>("clusters", "CrvCoincidenceCluster");
  ids(clusters_);
  clusters_->Branch("idx", &idx_);
  clusters_->Branch("sectortype", &sectortype_);
  clusters_->Branch("start", &cstart_);
  clusters_->Branch("end", &cend_);
  clusters_->Branch("pe", &cpe_);
  clusters_->Branch("slope", &slope_);
  clusters_->Branch("layers", &layers_);
  clusters_->Branch("pulses", &cpulses_);
  clusters_->Branch("poscalc", &poscalc_);
  clusters_->Branch("avgtime", &avgtime_);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{
           {"x", &cx_}, {"y", &cy_}, {"z", &cz_}, {"sidepe0", &spe0_}, {"sidepe1", &spe1_}})
    clusters_->Branch(n, v);
  clusters_->Branch("sidetime0", &st0_);
  clusters_->Branch("sidetime1", &st1_);
  clusters_->Branch("sidehits0", &sh0_);
  clusters_->Branch("sidehits1", &sh1_);
  calib_ = tfs->make<TTree>("calib", "CRVCalib, CRVStatus and CRVOrdinal at the first event, one row per channel");
  calib_->Branch("run", &run_);
  calib_->Branch("channel", &chan_);
  calib_->Branch("pedestal", &cped_);
  calib_->Branch("pulsearea", &carea_);
  calib_->Branch("pulseheight", &cheight_);
  calib_->Branch("timeoffset", &ctoff_);
  calib_->Branch("status", &status_);
  calib_->Branch("roc", &mroc_);
  calib_->Branch("feb", &mfeb_);
  calib_->Branch("febch", &mfebch_);
}

void CrvRecoTableDumper::dumpConditions(const art::Event& event) {
  auto const& calib = calib_h_.get(event.id());
  auto const& status = status_h_.get(event.id());
  auto const& ordinal = ordinal_h_.get(event.id());
  for (std::size_t c = 0; c < CRVId::nChannels; ++c) {
    try {
      cped_ = calib.pedestal(c);
      carea_ = calib.pulseArea(c);
      cheight_ = calib.pulseHeight(c);
      ctoff_ = calib.timeOffset(c);
    } catch (std::out_of_range const&) {
      break;  // the calibration vector ends here
    }
    chan_ = int(c);
    status_ = status.status(c);
    mroc_ = mfeb_ = mfebch_ = -1;
    if (ordinal.onlineExists(c)) {
      auto const& on = ordinal.online(c);
      mroc_ = on.ROC();
      mfeb_ = on.FEB();
      mfebch_ = on.FEBchannel();
    }
    calib_->Fill();
  }
  conditionsDone_ = true;
}

void CrvRecoTableDumper::analyze(const art::Event& event) {
  run_ = event.run();
  subrun_ = event.subRun();
  event_ = event.event();
  if (!conditionsDone_) dumpConditions(event);

  art::Handle<EventWindowMarker> ewm;
  spill_ = event.getByLabel(conf_.ewm(), ewm) ? int(ewm->spillType()) : -1;
  evlen_ = ewm.isValid() ? ewm->eventLength() : 0;
  pbtime_ = 0;
  art::Handle<ProtonBunchTime> pbt;
  if (event.getByLabel(conf_.pbt(), pbt)) pbtime_ = pbt->pbtime_;

  if (conf_.mc()) {
    auto qh = event.getValidHandle<CrvSiPMChargesCollection>(conf_.sipmCharges());
    for (std::size_t i = 0; i < qh->size(); ++i) {
      auto const& q = qh->at(i);
      idx_ = int(i);
      bar_ = int(q.GetScintillatorBarIndex().asInt());
      sipm_ = q.GetSiPMNumber();
      qtime_.clear();
      qcharge_.clear();
      qpe_.clear();
      qsim_.clear();
      for (auto const& c : q.GetCharges()) {
        qtime_.push_back(c._time);
        qcharge_.push_back(c._charge);
        qpe_.push_back(c._chargeInPEs);
        qsim_.push_back(c._step.isNonnull() && c._step->simParticle().isNonnull() ? int(c._step->simParticle().key()) : -1);
      }
      charges_->Fill();
    }
    auto mh = event.getValidHandle<CrvDigiMCCollection>(conf_.digiMCs());
    for (std::size_t i = 0; i < mh->size(); ++i) {
      auto const& m = mh->at(i);
      idx_ = int(i);
      bar_ = int(m.GetScintillatorBarIndex().asInt());
      sipm_ = m.GetSiPMNumber();
      mcstart_ = m.GetStartTime();
      mctdc0_ = m.GetTDC0Time();
      nzs_ = m.IsNZS();
      mcsim_ = m.GetSimParticle().isNonnull() ? int(m.GetSimParticle().key()) : -1;
      volt_ = m.GetVoltages();
      digimc_->Fill();
    }
  }

  auto dh = event.getValidHandle<CrvDigiCollection>(conf_.digis());
  ndigi_ = int(dh->size());
  for (std::size_t i = 0; i < dh->size(); ++i) {
    auto const& d = dh->at(i);
    idx_ = int(i);
    bar_ = int(d.GetScintillatorBarIndex().asInt());
    sipm_ = d.GetSiPMNumber();
    starttdc_ = d.GetStartTDC();
    nzs_ = d.IsNZS();
    roc_ = d.GetROC();
    feb_ = d.GetFEB();
    febch_ = d.GetFEBchannel();
    adc_.assign(d.GetADCs().begin(), d.GetADCs().end());
    digis_->Fill();
  }

  auto ph = event.getValidHandle<CrvRecoPulseCollection>(conf_.pulses());
  npulse_ = int(ph->size());
  for (std::size_t i = 0; i < ph->size(); ++i) {
    auto const& p = ph->at(i);
    idx_ = int(i);
    bar_ = int(p.GetScintillatorBarIndex().asInt());
    sipm_ = p.GetSiPMNumber();
    pe_ = p.GetPEs();
    peph_ = p.GetPEsPulseHeight();
    height_ = p.GetPulseHeight();
    beta_ = p.GetPulseBeta();
    chi2_ = p.GetPulseFitChi2();
    ped_ = p.GetPedestal();
    ptime_ = p.GetPulseTime();
    letime_ = p.GetLEtime();
    seq_ = int(p.GetSequenceIndex());
    flags_ = int(p.GetRecoPulseFlags().to_ulong());
    pedfromdb_ = p.IsPedestalFromDB();
    wfidx_.assign(p.GetWaveformIndices().begin(), p.GetWaveformIndices().end());
    pulses_->Fill();
  }

  auto ch = event.getValidHandle<CrvCoincidenceClusterCollection>(conf_.clusters());
  ncluster_ = int(ch->size());
  for (std::size_t i = 0; i < ch->size(); ++i) {
    auto const& c = ch->at(i);
    idx_ = int(i);
    sectortype_ = c.GetCrvSectorType();
    cstart_ = c.GetStartTime();
    cend_ = c.GetEndTime();
    cpe_ = c.GetPEs();
    slope_ = c.GetSlope();
    layers_ = c.GetLayers();
    cpulses_.clear();
    for (auto const& p : c.GetCrvRecoPulses()) cpulses_.push_back(int(p.key()));
    poscalc_ = c.HitPosAndTimeCalculated();
    avgtime_ = c.GetAvgHitTime();
    cx_ = c.GetAvgHitPos().x();
    cy_ = c.GetAvgHitPos().y();
    cz_ = c.GetAvgHitPos().z();
    spe0_ = c.GetSidePEs()[0];
    spe1_ = c.GetSidePEs()[1];
    st0_ = c.GetSideTimes()[0];
    st1_ = c.GetSideTimes()[1];
    sh0_ = int(c.GetSideHits()[0]);
    sh1_ = int(c.GetSideHits()[1]);
    clusters_->Fill();
  }
  events_->Fill();
}

}  // namespace mu2e

DEFINE_ART_MODULE(mu2e::CrvRecoTableDumper)
