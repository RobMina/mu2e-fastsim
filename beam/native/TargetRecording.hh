#ifndef MU2E_TARGET_RECORDING_HH
#define MU2E_TARGET_RECORDING_HH

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Offline/MCDataProducts/inc/SimParticle.hh"

namespace art { class Event; class InputTag; }
namespace CLHEP { class Hep3Vector; }
class G4Track;
class G4Step;

namespace mu2e {
class SimParticleHelper;
class SimParticlePrimaryHelper;

// Unapplied, per-thread source proposal; constructing or executing requires approval.
class TargetRecording final {
public:
  TargetRecording(std::string outputDirectory,
                  std::vector<std::string> targetPhysicalVolumeNames,
                  std::string targetMaterialName,
                  std::uint64_t maxRows,
                  std::uint64_t maxVolumes,
                  std::uint64_t maxBytes,
                  CLHEP::Hep3Vector const& mu2eOrigin);
  ~TargetRecording();

  TargetRecording(TargetRecording const&) = delete;
  TargetRecording& operator=(TargetRecording const&) = delete;
  TargetRecording(TargetRecording&&) = delete;
  TargetRecording& operator=(TargetRecording&&) = delete;

  void prepareEvent(art::Event const&, art::InputTag const&, SimParticleHelper const&);
  void beginEvent(unsigned run, unsigned subrun, std::uint64_t event,
                  SimParticleHelper const&, SimParticlePrimaryHelper const&);
  void simParticleInserted(G4Track const&, SimParticle const&);
  void simParticleOmitted(G4Track const&);
  void stacking(G4Track const&, int classification, bool stackingCut, bool commonCut);
  void beginTrack(G4Track const&);
  void step(G4Step const&);
  void cuts(G4Track const&, bool steppingCut, bool commonCut);
  void endTrack(G4Track const&);
  void endEvent(int status, bool eventAborted, bool simOverflow,
                std::uint64_t stepKills, std::uint64_t fieldKills,
                SimParticleCollection const&);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}

#endif
