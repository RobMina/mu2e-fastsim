#include "EventJson.hh"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <G4Event.hh>
#include <G4HCofThisEvent.hh>
#include <G4PrimaryVertex.hh>
#include <G4VHitsCollection.hh>
#include <nlohmann/json.hpp>
#include "geocel/GeantUtils.hh"
#include "corecel/sys/Environment.hh"
#include "SensitiveHit.hh"

namespace m2fs
{
namespace
{
struct CloseFile
{
    void operator()(std::FILE* file) const { std::fclose(file); }
};
std::unique_ptr<std::FILE, CloseFile> output;

void require(bool ok, char const* message)
{
    if (!ok)
        throw std::runtime_error(message);
}
}

void initialize_event_output()
{
    char const* name = std::getenv("M2FS_EVENT_OUTPUT");
    if (!name || !*name)
        return;
    require(!celeritas::getenv_flag("CELER_KILL_OFFLOAD", false).value, "Event output forbids kill-offload mode");
    require(!celeritas::getenv_flag("CELER_NONFATAL_FLUSH", false).value, "Event output requires fatal transport failures");
    require(celeritas::get_geant_num_threads() == 1, "Event JSON output requires serial/one-worker Geant4");
    require(!output, "Event output already initialized");
    int fd = ::open(name, O_WRONLY | O_CREAT | O_EXCL, 0600);
    require(fd >= 0, "Cannot create exclusive event output file");
    auto* file = ::fdopen(fd, "w");
    if (!file)
    {
        ::close(fd);
        throw std::runtime_error("Cannot open event output stream");
    }
    output.reset(file);
}

void write_event(G4Event const& event)
{
    if (!output)
        return;
    require(!event.IsAborted(), "Refusing to score an aborted event");
    auto* collections = event.GetHCofThisEvent();
    require(collections && collections->GetNumberOfCollections() > 0, "Event has no sensitive hit collections");
    std::size_t primaries = 0;
    for (int i = 0; i < event.GetNumberOfPrimaryVertex(); ++i)
        primaries += event.GetPrimaryVertex(i)->GetNumberOfParticle();
    nlohmann::json rows = nlohmann::json::array();
    std::size_t hits = 0;
    double total = 0;
    for (int i = 0; i < collections->GetCapacity(); ++i)
    {
        auto* collection = collections->GetHC(i);
        if (!collection)
            continue;
        nlohmann::json values = nlohmann::json::array();
        double energy = 0;
        for (std::size_t j = 0; j < collection->GetSize(); ++j)
        {
            auto const* hit = dynamic_cast<celeritas::app::SensitiveHit const*>(collection->GetHit(j));
            require(hit, "Unexpected hit type in event output");
            auto const& d = hit->data();
            require(std::isfinite(d.energy_dep) && d.energy_dep >= 0 && std::isfinite(d.time), "Invalid hit energy/time");
            energy += d.energy_dep;
            values.push_back({{"volume_instance", d.volume}, {"copy_num", d.copy_num},
                              {"total_edep_mev", d.energy_dep}, {"pre_time_s", d.time}});
        }
        total += energy;
        hits += collection->GetSize();
        rows.push_back({{"sd", collection->GetSDname()}, {"collection", collection->GetName()},
                        {"total_edep_mev", energy}, {"hits", std::move(values)}});
    }
    require(std::isfinite(total), "Event energy sum overflow");
    nlohmann::json row{{"format", "m2fs-event-total-edep-v1"}, {"event_id", event.GetEventID()},
                       {"primary_count", primaries}, {"hit_count", hits}, {"total_edep_mev", total},
                       {"collections", std::move(rows)}};
    std::string text = row.dump() + "\n";
    require(std::fwrite(text.data(), 1, text.size(), output.get()) == text.size(), "Event output write failed");
    require(std::fflush(output.get()) == 0, "Event output flush failed");
}
}
