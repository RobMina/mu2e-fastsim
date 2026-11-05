#ifndef Mu2eG4_inc_Mu2eG4BoundaryRecorder_hh
#define Mu2eG4_inc_Mu2eG4BoundaryRecorder_hh

#include "Offline/Mu2eG4/inc/Mu2eG4Config.hh"
#include "Offline/Mu2eG4/inc/SimParticleHelper.hh"
#include "canvas/Persistency/Provenance/EventID.h"
#include "Geant4/G4Event.hh"
#include "Geant4/G4Step.hh"
#include "Geant4/G4StepPoint.hh"
#include "Geant4/G4StepStatus.hh"
#include "Geant4/G4TrackStatus.hh"
#include "Geant4/G4Track.hh"
#include "Geant4/G4VTouchable.hh"
#include "Geant4/G4VPhysicalVolume.hh"
#include "Geant4/G4PhysicalVolumeStore.hh"
#include "Geant4/G4LogicalVolume.hh"
#include "Geant4/G4Material.hh"
#include "Geant4/G4VProcess.hh"
#include "Geant4/G4ParticleDefinition.hh"
#include "Geant4/G4Threading.hh"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mu2e {
  // Diagnostic candidate membership only; no geometry or causal-domain certification.
  class Mu2eG4BoundaryRecorder {
    struct TrackInfo {
      std::uint64_t visit = 0;
      bool active = false;
      bool birthUnknown = false;
      bool saved = false;
    };
    std::ofstream out_;
    std::vector<std::string> roots_;
    std::map<int, TrackInfo> tracks_;
    const SimParticleHelper* helper_ = nullptr;
    bool enabled_ = false;
    bool inEvent_ = false;
    int worker_ = -999;
    std::uint64_t sequence_ = 0, rows_ = 0, bytes_ = 0;
    std::uint64_t maxRows_ = 0, maxBytes_ = 0;
    unsigned run_ = 0, subrun_ = 0;
    unsigned unknownBirths_ = 0, missingSaved_ = 0, unknownEnds_ = 0;
    unsigned long long event_ = 0;

    [[noreturn]] static void fail(const std::string& why) {
      throw std::runtime_error("Mu2eG4BoundaryRecorder: " + why);
    }
    static std::string q(const std::string& s) {
      std::ostringstream o;
      o << '"';
      for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << char(c);
        else if (c < 32 || c >= 127) {
          o << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c) << std::dec;
        } else o << char(c);
      }
      o << '"';
      return o.str();
    }
    static std::ostringstream stream() {
      std::ostringstream o;
      o.imbue(std::locale::classic());
      o << std::setprecision(std::numeric_limits<double>::max_digits10);
      return o;
    }
    static double finite(double x) {
      if (!std::isfinite(x)) fail("nonfinite numeric state; incomplete ledger");
      return x;
    }
    static std::string vec(const G4ThreeVector& v) {
      auto o = stream();
      o << '[' << finite(v.x()) << ',' << finite(v.y()) << ',' << finite(v.z()) << ']';
      return o.str();
    }
    template<class T> static std::string identity(const art::Ptr<T>& p) {
      if (p.isNull()) return "null";
      return "{\"product\":" + std::to_string(p.id().value()) + ",\"key\":" + std::to_string(p.key()) + "}";
    }
    static std::string material(const G4Material* m) {
      if (!m) return "null";
      return "{\"name\":" + q(m->GetName()) + ",\"index\":" + std::to_string(m->GetIndex()) + "}";
    }
    static std::string path(const G4VTouchable* t, bool worldNull = false) {
      if (worldNull) return "{\"availability\":\"world_null\",\"nodes\":[]}";
      if (!t || !t->GetVolume()) return "{\"availability\":\"unavailable\",\"nodes\":[]}";
      auto o = stream();
      o << "{\"availability\":\"available\",\"nodes\":[";
      for (int d = t->GetHistoryDepth(); d >= 0; --d) {
        if (d != t->GetHistoryDepth()) o << ',';
        const auto* v = t->GetVolume(d);
        if (!v) fail("null ancestor in touchable");
        o << "{\"pv\":" << q(v->GetName()) << ",\"lv\":" << q(v->GetLogicalVolume()->GetName())
          << ",\"copy\":" << v->GetCopyNo() << ",\"replica\":" << t->GetReplicaNumber(d) << '}';
      }
      o << "]}";
      return o.str();
    }
    int member(const G4VTouchable* t, bool worldNull = false) const {
      if (worldNull) return 0;
      if (!t || !t->GetVolume()) return -1;
      for (int d = 0; d <= t->GetHistoryDepth(); ++d) {
        const auto* v = t->GetVolume(d);
        if (!v) fail("null ancestor in membership");
        if (std::find(roots_.begin(), roots_.end(), v->GetName()) != roots_.end()) return 1;
      }
      return 0;
    }
    static std::string state(const G4StepPoint& p, bool worldNull = false) {
      auto o = stream();
      o << "{\"position\":" << vec(p.GetPosition()) << ",\"momentum\":" << vec(p.GetMomentum())
        << ",\"kinetic_energy\":" << finite(p.GetKineticEnergy()) << ",\"total_energy\":" << finite(p.GetTotalEnergy())
        << ",\"global_time\":" << finite(p.GetGlobalTime()) << ",\"proper_time\":" << finite(p.GetProperTime())
        << ",\"local_time\":" << finite(p.GetLocalTime()) << ",\"step_status\":" << int(p.GetStepStatus())
        << ",\"material\":" << material(p.GetMaterial()) << ",\"touchable\":" << path(p.GetTouchable(), worldNull) << '}';
      return o.str();
    }
    static std::string state(const G4Track& t) {
      auto o = stream();
      const auto* touch = t.GetTouchable();
      const auto* v = touch ? touch->GetVolume() : nullptr;
      o << "{\"position\":" << vec(t.GetPosition()) << ",\"momentum\":" << vec(t.GetMomentum())
        << ",\"kinetic_energy\":" << finite(t.GetKineticEnergy()) << ",\"total_energy\":" << finite(t.GetTotalEnergy())
        << ",\"global_time\":" << finite(t.GetGlobalTime()) << ",\"proper_time\":" << finite(t.GetProperTime())
        << ",\"local_time\":" << finite(t.GetLocalTime()) << ",\"material_source\":\"touchable_logical_not_step_material\",\"material\":"
        << material(v ? v->GetLogicalVolume()->GetMaterial() : nullptr)
        << ",\"touchable\":" << path(touch) << '}';
      return o.str();
    }
    void validateRoots() const {
      // Validate the actual built geometry, not only configured strings, before any begin row.
      const auto* store = G4PhysicalVolumeStore::GetInstance();
      if (!store) fail("physical-volume store unavailable");
      for (const auto& root : roots_) {
        std::size_t matches = 0;
        const G4VPhysicalVolume* selected = nullptr;
        for (const auto* volume : *store) {
          if (volume && volume->GetName() == root) { ++matches; selected = volume; }
        }
        if (matches == 0) fail("configured root missing from physical-volume store: " + root);
        if (matches != 1) fail("configured root name is not unique in physical-volume store: " + root);
        if (selected->IsReplicated() || selected->IsParameterised())
          fail("replicated/parameterised root unsupported: " + root);
      }
      // A unique store object can still occur under repeated ancestors; no path uniqueness certificate.
    }
    void checkEvent() const {
      if (!inEvent_ || !helper_) fail("hook outside event");
      if (worker_ != G4Threading::G4GetThreadId()) fail("cross-worker access");
    }
    void row(const std::string& type, const std::string& fields) {
      auto o = stream();
      o << "{\"schema\":\"mu2e-boundary-diagnostic-v1\",\"type\":" << q(type)
        << ",\"run\":" << run_ << ",\"subrun\":" << subrun_ << ",\"event\":" << event_
        << ",\"sequence\":" << sequence_ << fields << "}\n";
      const auto s = o.str();
      if (rows_ >= maxRows_ || s.size() > maxBytes_ - bytes_) fail("file row/byte bound exceeded; incomplete ledger, never truncate");
      out_ << s;
      out_.flush();
      if (!out_) fail("write/flush failed; incomplete ledger");
      ++sequence_;
      ++rows_;
      bytes_ += s.size();
    }
    TrackInfo& info(const G4Track& t) {
      auto i = tracks_.find(t.GetTrackID());
      if (i == tracks_.end()) fail("track without tracking-start hook");
      return i->second;
    }
    std::string trackFields(const G4Track& t, const TrackInfo& i) const {
      return ",\"track\":" + std::to_string(t.GetTrackID()) + ",\"parent_track\":" + std::to_string(t.GetParentID())
        + ",\"particle\":" + identity(helper_->particlePtr(&t)) + ",\"sim_saved_at_start\":" + (i.saved ? "true" : "false")
        + ",\"step\":" + std::to_string(t.GetCurrentStepNumber()) + ",\"visit\":" + std::to_string(i.visit)
        + ",\"track_status\":" + std::to_string(int(t.GetTrackStatus()));
    }
  public:
    explicit Mu2eG4BoundaryRecorder(const Mu2eG4Config::Debug& c) {
      const auto prefix = c.boundaryRecorderPrefix();
      if (prefix.empty()) return;
      roots_ = c.boundaryRecorderRoots();
      maxRows_ = c.boundaryRecorderMaxRows();
      maxBytes_ = c.boundaryRecorderMaxBytes();
      if (roots_.empty() || !maxRows_ || !maxBytes_) fail("enabled recorder needs exact roots and positive bounds");
      std::set<std::string> unique;
      for (const auto& r : roots_) if (r.empty() || !unique.insert(r).second) fail("empty/duplicate root");
      if (G4Threading::IsMultithreadedApplication()) fail("diagnostic currently supports sequential G4 only");
      worker_ = G4Threading::G4GetThreadId();
      const auto dir = prefix + ".pid" + std::to_string(::getpid()) + ".worker" + std::to_string(worker_);
      if (::mkdir(dir.c_str(), 0700) != 0) fail("cannot exclusively create private output directory " + dir);
      const auto file = dir + "/boundary.jsonl";
      const int fd = ::open(file.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
      if (fd < 0) fail("cannot exclusively create " + file);
      if (::close(fd) != 0) fail("close after exclusive create failed");
      // Reopen only our empty file in a newly exclusive private directory; append never truncates.
      out_.open(file, std::ios::out | std::ios::app | std::ios::binary);
      if (!out_) fail("ofstream open failed");
      enabled_ = true;
    }
    bool enabled() const { return enabled_; }
    void begin(const art::EventID& id, const G4Event& e, const SimParticleHelper& h) {
      if (!enabled_) return;
      if (inEvent_) fail("previous event has no end marker");
      validateRoots();
      helper_ = &h;
      inEvent_ = true;
      checkEvent();
      run_ = id.run(); subrun_ = id.subRun(); event_ = id.event();
      sequence_ = 0;
      unknownBirths_ = missingSaved_ = unknownEnds_ = 0;
      tracks_.clear();
      auto o = stream();
      o << ",\"g4_event\":" << e.GetEventID() << ",\"sim_stage\":" << h.simStage()
        << ",\"candidate_only\":true,\"coordinate_frame\":\"G4_world\",\"units\":\"mm_MeV_ns\",\"product_id_encoding\":\"art_ProductID_value_uint32\",\"roots\":[";
      for (std::size_t n = 0; n < roots_.size(); ++n) { if (n) o << ','; o << q(roots_[n]); }
      o << ']';
      row("event_begin", o.str());
    }
    void start(const G4Track& t, const SimParticle* saved) {
      if (!enabled_) return;
      checkEvent();
      auto inserted = tracks_.emplace(t.GetTrackID(), TrackInfo{});
      auto& i = inserted.first->second;
      if (!inserted.second) {
        row("tracking_resume", trackFields(t, i) + ",\"state\":" + state(t));
        return;
      }
      i.saved = saved != nullptr;
      if (!i.saved) ++missingSaved_;
      const int m = member(t.GetTouchable());
      i.birthUnknown = m < 0;
      if (i.birthUnknown) ++unknownBirths_;
      if (m == 1) { ++i.visit; i.active = true; }
      const auto* creator = t.GetCreatorProcess();
      auto o = stream();
      o << trackFields(t, i) << ",\"membership\":" << m << ",\"birth_phase\":\"PreUserTracking_after_saveSimParticleStart\""
        << ",\"pdg\":" << t.GetDefinition()->GetPDGEncoding() << ",\"creator_process\":"
        << (creator ? q(creator->GetProcessName()) : "null")
        << ",\"persistent_parent\":" << (saved ? identity(saved->parent()) : "null")
        << ",\"gen_particle\":" << (saved ? identity(saved->genParticle()) : "null")
        << ",\"state\":" << state(t);
      row(m == 1 ? "born_inside" : "track_start", o.str());
    }
    void step(const G4Step& s) {
      if (!enabled_) return;
      checkEvent();
      const auto& t = *s.GetTrack();
      auto& i = info(t);
      const auto& pre = *s.GetPreStepPoint();
      const auto& post = *s.GetPostStepPoint();
      const bool worldNull = post.GetStepStatus() == fWorldBoundary && post.GetPhysicalVolume() == nullptr;
      const int a = member(pre.GetTouchable());
      const int b = member(post.GetTouchable(), worldNull);
      if (i.birthUnknown) {
        i.birthUnknown = false;
        if (a == 1) { ++i.visit; i.active = true; }
        row("birth_membership_unresolved", trackFields(t, i) + ",\"first_observed_pre\":" + state(pre));
      }
      if (a < 0 || b < 0) {
        row("membership_unavailable", trackFields(t, i) + ",\"pre\":" + state(pre) + ",\"post\":" + state(post, worldNull));
        fail("non-world step touchable unavailable; membership cannot be reconstructed safely");
      }
      if (a == b) return;
      if (b == 1) { ++i.visit; i.active = true; }
      else i.active = false;
      const auto* p = post.GetProcessDefinedStep();
      auto o = stream();
      o << trackFields(t, i) << ",\"pre_membership\":" << a << ",\"post_membership\":" << b
        << ",\"classification\":" << q(b == 1 ? (i.visit == 1 ? "first_observed_entry" : "reentry") : "exit")
        << ",\"endpoint\":\"post_step\",\"hook_phase\":\"UserSteppingAction_before_stock_cuts\""
        << ",\"step_length\":" << finite(s.GetStepLength()) << ",\"process\":" << (p ? q(p->GetProcessName()) : "null")
        << ",\"pre\":" << state(pre) << ",\"post\":" << state(post, worldNull);
      row(b == 1 ? "entry" : "exit", o.str());
    }
    void stop(const G4Track& t) {
      if (!enabled_) return;
      checkEvent();
      auto& i = info(t);
      const bool terminal = t.GetTrackStatus() == fStopAndKill || t.GetTrackStatus() == fKillTrackAndSecondaries;
      const auto* s = t.GetStep();
      const auto* post = s && t.GetCurrentStepNumber() > 0 ? s->GetPostStepPoint() : nullptr;
      const bool worldNull = post && post->GetStepStatus() == fWorldBoundary && post->GetPhysicalVolume() == nullptr;
      const int m = post ? member(post->GetTouchable(), worldNull) : member(t.GetTouchable());
      if (m < 0) ++unknownEnds_;
      const auto* process = post ? post->GetProcessDefinedStep() : nullptr;
      // The last post-step touchable wins over potentially stale current-track navigation at a boundary.
      row(terminal && m == 1 ? "terminal_inside" : "tracking_end",
          trackFields(t, i) + ",\"terminal\":" + (terminal ? "true" : "false") + ",\"membership\":" + std::to_string(m)
          + ",\"process\":" + (process ? q(process->GetProcessName()) : "null")
          + ",\"endpoint_phase\":" + q(post ? "last_post_step" : "track_no_step")
          + ",\"state\":" + (post ? state(*post, worldNull) : state(t)));
      if (terminal) i.active = false;
    }
    void end(const G4Event& e, bool accepted, unsigned nTracks, bool overflow, unsigned killedStep, unsigned killedField) {
      if (!enabled_) return;
      checkEvent();
      unsigned active = 0;
      for (const auto& x : tracks_) if (x.second.active) ++active;
      const int status = overflow ? 10 : ((killedStep || killedField) ? 1 : 0);
      auto o = stream();
      o << ",\"g4_event\":" << e.GetEventID() << ",\"g4_aborted\":" << (e.IsAborted() ? "true" : "false")
        << ",\"stock_filter_pass\":" << (accepted ? "true" : "false") << ",\"status_g4_equivalent\":" << status
        << ",\"n_g4_tracks\":" << nTracks << ",\"overflow_sim_particles\":" << (overflow ? "true" : "false")
        << ",\"killed_step_limit\":" << killedStep << ",\"killed_field\":" << killedField
        << ",\"unclosed_visits\":" << active << ",\"unknown_birth_memberships\":" << unknownBirths_
        << ",\"missing_saved_particles\":" << missingSaved_ << ",\"unknown_tracking_ends\":" << unknownEnds_
        << ",\"rows_before_end\":" << sequence_;
      row("event_end", o.str());
      inEvent_ = false;
      helper_ = nullptr;
      tracks_.clear();
    }
  };
}
#endif
