#include <array>
#include <cerrno>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <fcntl.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "corecel/Config.hh"
#include "corecel/Version.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Units.hh"
#include "celeritas/field/FieldDriverOptionsIO.json.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/vg/VecgeomParams.hh"
#include "CartMapIO.hh"
#include "NavAlgorithms.hh"

namespace m2fs::nav
{
namespace
{
using Json = nlohmann::json;
using celeritas::units::millimeter;
namespace fs = std::filesystem;

void require(bool ok, std::string const& message)
{
    if (!ok)
        throw std::runtime_error(message);
}

Json read_json(fs::path const& path)
{
    std::ifstream file(path);
    require(bool(file), "Cannot open " + path.string());
    Json result;
    file >> result;
    return result;
}

void exact_keys(Json const& obj, std::set<std::string> const& expected)
{
    require(obj.is_object(), "Expected object");
    std::set<std::string> found;
    for (auto const& item : obj.items())
        found.insert(item.key());
    require(found == expected, "Unexpected or missing input key");
}

double number(Json const& j)
{
    require(j.is_number(), "Expected numeric value");
    double result = j.get<double>();
    require(std::isfinite(result), "Nonfinite input");
    return result;
}

unsigned count(Json const& j, unsigned lo, unsigned hi)
{
    require(j.is_number_integer(), "Expected integer count");
    auto value = j.get<long long>();
    require(value >= lo && value <= hi, "Count outside bounded diagnostic range");
    return static_cast<unsigned>(value);
}

Real3 vector(Json const& j)
{
    require(j.is_array() && j.size() == 3, "Expected 3-vector");
    return {number(j[0]), number(j[1]), number(j[2])};
}

fs::path sibling(fs::path const& root, Json const& name)
{
    require(name.is_string(), "Expected input filename");
    fs::path value = name.get<std::string>();
    require(!value.empty() && value == value.filename() && value != "." && value != "..", "Input must be a sibling file");
    auto path = fs::canonical(root / value);
    require(path.parent_path() == fs::canonical(root), "Input symlink escapes fixture directory");
    return path;
}

Json vector_json(Real3 const& v, double unit)
{
    require(detail::finite(v), "Nonfinite output vector");
    return Json::array({v[0] / unit, v[1] / unit, v[2] / unit});
}

Json state_json(State const& state)
{
    return {{"position_mm", vector_json(state.position, static_cast<double>(millimeter))},
            {"direction", vector_json(state.direction, 1)}, {"volume_id", state.volume},
            {"instance_id", state.instance}, {"boundary", state.boundary}, {"outside", state.outside}};
}

Json steps_json(Batch const& batch, std::vector<std::string> const& ids)
{
    require(batch.tracks.size() == ids.size() && batch.steps.size() == ids.size() * batch.stride, "Wrong output shape");
    Json rows = Json::array();
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        auto const& track = batch.tracks[i];
        require(track.tag == completed_tag && track.count <= batch.stride && track.status <= 6, "Incomplete track record");
        Json steps = Json::array();
        for (unsigned k = 0; k < track.count; ++k)
        {
            auto const& rec = batch.steps[i * batch.stride + k];
            require(rec.tag == completed_tag && detail::finite(rec.distance) && detail::finite(rec.requested)
                        && detail::finite(rec.cumulative_path) && detail::finite(rec.time_ns)
                        && detail::finite(rec.momentum_mev_c), "Incomplete/nonfinite step record");
            steps.push_back({{"before", state_json(rec.before)}, {"arrival", state_json(rec.arrival)},
                             {"after", state_json(rec.after)}, {"requested_mm", rec.requested / millimeter},
                             {"distance_mm", rec.distance / millimeter},
                             {"cumulative_path_mm", rec.cumulative_path / millimeter},
                             {"momentum_mev_c", rec.momentum_mev_c}, {"time_ns", rec.time_ns},
                             {"field_evaluations", rec.field_evaluations}, {"substep_calls", rec.substep_calls},
                             {"boundary", rec.boundary}, {"looping", rec.looping},
                             {"geometry_failed", rec.geometry_failed}, {"completed_tag", rec.tag}});
        }
        require(detail::finite(track.path), "Nonfinite total path");
        Json margin = nullptr;
        if (track.audit.field_evaluations)
        {
            require(detail::finite(track.audit.min_margin / millimeter), "Nonfinite field margin");
            margin = track.audit.min_margin / millimeter;
        }
        rows.push_back({{"id", ids[i]}, {"status", track.status}, {"completed_tag", track.tag},
                        {"count", track.count}, {"crossings", track.crossings}, {"path_mm", track.path / millimeter},
                        {"field_evaluations", track.audit.field_evaluations}, {"substep_calls", track.audit.substep_calls},
                        {"min_margin_mm", margin}, {"final", state_json(track.final)}, {"steps", std::move(steps)}});
    }
    return rows;
}

template<class Id, class Map>
Json label_table(Map const& labels)
{
    Json result = Json::array();
    for (unsigned i = 0; i < labels.size(); ++i)
    {
        auto const& label = labels.at(Id{i});
        if (!label.name.empty())
            result.push_back({{"id", i}, {"name", label.name}, {"ext", label.ext}});
    }
    return result;
}

void save_exclusive(fs::path const& path, Json const& result)
{
    std::string text = result.dump(2) + "\n";
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    require(fd >= 0, "Cannot create exclusive output " + path.string());
    std::size_t used = 0;
    while (used < text.size())
    {
        auto n = ::write(fd, text.data() + used, text.size() - used);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
        {
            ::close(fd);
            throw std::runtime_error("Output write failed");
        }
        used += n;
    }
    int synced = ::fsync(fd);
    int closed = ::close(fd);
    require(synced == 0 && closed == 0, "Output persistence failed");
}

int run(fs::path const& request_path, fs::path const& output, std::string const& backend)
{
    static_assert(std::is_same_v<celeritas::real_type, double>);
    require(backend == "host" || backend == "device" || backend == "both", "Unknown backend");
    require(!fs::exists(output) && fs::is_directory(output.parent_path()), "Need NEW output in existing directory");
    auto request = read_json(request_path);
    exact_keys(request, {"format", "map_file", "geometry_file", "max_records", "tracks"});
    require(request.at("format") == "m2fs-navigation-request-v1", "Wrong request format");
    auto root = request_path.parent_path();
    auto field_file = sibling(root, request.at("map_file"));
    auto gdml = sibling(root, request.at("geometry_file"));
    auto input = read_cartmap(field_file.string());
    Domain domain;
    std::array<celeritas::AxisGrid<celeritas::real_type>, 3> axes{input.x, input.y, input.z};
    for (int i = 0; i < 3; ++i)
    {
        double spacing = (axes[i].max - axes[i].min) / (axes[i].num - 1);
        domain.lower[i] = axes[i].min + spacing;
        domain.upper[i] = axes[i].max - spacing;
    }
    auto const& entries = request.at("tracks");
    unsigned stride = count(request.at("max_records"), 1, 8192);
    require(entries.is_array() && !entries.empty() && entries.size() <= 128
                && entries.size() * stride <= 262144, "Excessive trajectory/record budget");
    std::vector<TrackInput> tracks;
    std::vector<std::string> ids;
    std::set<std::string> unique;
    for (auto const& row : entries)
    {
        Json normalized = row;
        if (!normalized.contains("max_substeps"))
            normalized["max_substeps"] = 1000;
        if (!normalized.contains("max_nsteps"))
            normalized["max_nsteps"] = 100;
        exact_keys(normalized, {"id", "position_mm", "direction", "momentum_mev_c", "charge_e", "length_mm", "max_step_mm",
                                "minimum_step_mm", "delta_chord_mm", "delta_intersection_mm", "epsilon_rel_max",
                                "max_substeps", "max_nsteps"});
        require(row.at("charge_e").is_number_integer(), "Expected integer elementary charge");
        require(row.at("id").is_string(), "Expected track ID");
        std::string id = row.at("id").get<std::string>();
        require(!id.empty() && unique.insert(id).second, "Missing or duplicate track ID");
        TrackInput track{};
        track.position = vector(row.at("position_mm"));
        for (double& value : track.position)
        {
            require(std::abs(value) <= 1e6, "Position exceeds diagnostic bounds");
            value *= millimeter;
        }
        track.direction = vector(row.at("direction"));
        require(std::fabs(detail::norm(track.direction) - 1) <= 1e-12, "Direction must be unit, not silently normalized");
        double charge = number(row.at("charge_e"));
        require(charge == -1 || charge == 0 || charge == 1, "Charge must be -1/0/+1");
        track.particle_id = charge == -1 ? 0 : charge == 1 ? 1 : 2;
        double momentum = number(row.at("momentum_mev_c"));
        require(momentum >= 0.1 && momentum <= 1000, "Momentum outside probe range");
        track.kinetic_mev = momentum * momentum / (std::hypot(momentum, electron_mass_mev) + electron_mass_mev);
        double length = number(row.at("length_mm"));
        double cap = number(row.at("max_step_mm"));
        require(length >= 1e-4 && length <= 10000 && cap >= 1e-4 && cap <= 1000, "Length/cap outside probe range");
        track.length = length * millimeter;
        track.cap = cap * millimeter;
        track.options.minimum_step = number(row.at("minimum_step_mm")) * millimeter;
        track.options.delta_chord = number(row.at("delta_chord_mm")) * millimeter;
        track.options.delta_intersection = number(row.at("delta_intersection_mm")) * millimeter;
        track.options.epsilon_rel_max = number(row.at("epsilon_rel_max"));
        require(track.options.epsilon_rel_max >= 1e-12 && track.options.epsilon_rel_max <= 0.01, "Epsilon outside probe range");
        track.options.max_substeps = static_cast<short>(count(normalized.at("max_substeps"), 1, 30000));
        track.options.max_nsteps = static_cast<short>(count(normalized.at("max_nsteps"), 1, 30000));
        celeritas::validate_input(track.options);
        tracks.push_back(track);
        ids.push_back(id);
    }
    require(celeritas::version_major == 0 && celeritas::version_minor == 6 && celeritas::version_patch == 3,
            "Requires verified Celeritas 0.6.3");
    require(celeritas::Device::num_devices() > 0, "A100 required, no implicit host fallback");
    celeritas::activate_device(celeritas::Device{0});
    auto const& device = celeritas::device();
    require(device.capability() == 80 && device.name().find("A100") != std::string::npos, "A100 sm80 required");
    celeritas::GeantGeoParams geant(gdml.string());
    require_world_inside(input, *geant.world());
    celeritas::VecgeomParams geometry(geant.world());
    require(!geometry.use_surface_tracking(), "This snapshot targets h5 volume navigation");
    require(geometry.max_depth() <= 8, "Geometry hierarchy exceeds bounded probe scope");
    celeritas::CartMapFieldParams field(input);
    using namespace celeritas::units;
    celeritas::ParticleParams particles(celeritas::ParticleParams::Input{
        {"electron", celeritas::PDGNumber{11}, MevMass{electron_mass_mev}, ElementaryCharge{-1}, 0},
        {"positron", celeritas::PDGNumber{-11}, MevMass{electron_mass_mev}, ElementaryCharge{1}, 0},
        {"neutral-diagnostic", celeritas::PDGNumber{9900011}, MevMass{electron_mass_mev}, ElementaryCharge{0}, 0}});
    Json output_tracks = Json::object();
    if (backend == "device" || backend == "both")
    {
        celeritas::CollectionStateStore<celeritas::VecgeomStateData, celeritas::MemSpace::device> geo_states(geometry.host_ref(), tracks.size());
        celeritas::CollectionStateStore<celeritas::ParticleStateData, celeritas::MemSpace::device> par_states(particles.host_ref(), tracks.size());
        auto batch = run_device(field.device_ref(), geometry.device_ref(), geo_states.ref(), particles.device_ref(),
                                par_states.ref(), domain, tracks, stride);
        output_tracks["device"] = steps_json(batch, ids);
    }
    if (backend == "host" || backend == "both")
    {
        celeritas::CollectionStateStore<celeritas::VecgeomStateData, celeritas::MemSpace::host> geo_states(geometry.host_ref(), tracks.size());
        celeritas::CollectionStateStore<celeritas::ParticleStateData, celeritas::MemSpace::host> par_states(particles.host_ref(), tracks.size());
        Batch batch;
        batch.stride = stride;
        batch.tracks.resize(tracks.size());
        batch.steps.resize(tracks.size() * stride);
        for (unsigned i = 0; i < tracks.size(); ++i)
            batch.tracks[i] = detail::track_one(field.host_ref(), geometry.host_ref(), geo_states.ref(),
                                               particles.host_ref(), par_states.ref(), celeritas::TrackSlotId{i},
                                               domain, tracks[i], batch.steps.data() + i * stride, stride);
        output_tracks["host"] = steps_json(batch, ids);
    }
    Json metadata{{"logical_volumes", label_table<celeritas::VolumeId>(geometry.volumes())},
                  {"volume_instances", label_table<celeritas::VolumeInstanceId>(geometry.volume_instances())},
                  {"world_lower_mm", vector_json(geometry.bbox().lower(), static_cast<double>(millimeter))},
                  {"world_upper_mm", vector_json(geometry.bbox().upper(), static_cast<double>(millimeter))},
                  {"guard_lower_mm", vector_json(domain.lower, static_cast<double>(millimeter))},
                  {"guard_upper_mm", vector_json(domain.upper, static_cast<double>(millimeter))},
                  {"electron_mass_mev", electron_mass_mev}, {"neutral_control_same_mass", true},
                  {"geometry_loader", "Geant4/G4VG"}, {"interactions_enabled", false},
                  {"surface_normal_available", false}, {"inner_exhaustion_status_available", false},
                  {"momentum_magnitude_is_transport_constraint", true}, {"native_length_mm", 1.0 / millimeter},
                  {"celeritas_version", "0.6.3"}};
    save_exclusive(output, {{"format", "m2fs-navigation-result-v1"}, {"input", request}, {"backend", backend},
                            {"units", {{"length", "mm"}, {"time", "ns"}, {"momentum", "MeV/c"}}},
                            {"metadata", metadata}, {"tracks", std::move(output_tracks)},
                            {"device", {{"name", device.name()}, {"capability", device.capability()}, {"synchronized", true}}}});
    return 0;
}
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 4)
            throw std::runtime_error("Usage: m2fs-navigation-probe REQUEST.json NEW_RESULT.json host|device|both");
        return m2fs::nav::run(std::filesystem::absolute(argv[1]), std::filesystem::absolute(argv[2]), argv[3]);
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
