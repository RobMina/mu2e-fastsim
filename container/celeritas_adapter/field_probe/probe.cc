#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <nlohmann/json.hpp>
#include "corecel/Version.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/Units.hh"
#include "celeritas/field/CartMapFieldData.hh"
#include "CartMapIO.hh"
#include "ProbeAlgorithms.hh"

namespace m2fs::probe
{
namespace
{
using Json = nlohmann::json;
using celeritas::units::millimeter;
using celeritas::units::tesla;

void require(bool ok, std::string const& message)
{
    if (!ok)
        throw std::runtime_error(message);
}

Json read_json(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    require(bool(stream), "Cannot open input: " + path.string());
    Json result;
    stream >> result;
    return result;
}

double number(Json const& value)
{
    require(value.is_number(), "Expected numeric value");
    double result = value.get<double>();
    require(std::isfinite(result), "Expected finite value");
    return result;
}

Real3 vector(Json const& value)
{
    require(value.is_array() && value.size() == 3, "Expected three components");
    return {number(value[0]), number(value[1]), number(value[2])};
}

Json vector_json(Real3 const& value, double unit)
{
    return Json::array({value[0] / unit, value[1] / unit, value[2] / unit});
}

Json audit_json(Audit const& a)
{
    require((a.field_evaluations > 0 || a.guard_violations > 0) && a.numerical_failures == 0,
            "Missing or nonfinite field evaluation");
    require(std::isfinite(a.min_margin / millimeter), "Nonfinite field-domain margin");
    return {{"field_evaluations", a.field_evaluations}, {"guard_violations", a.guard_violations},
            {"numerical_failures", a.numerical_failures}, {"min_margin_mm", a.min_margin / millimeter}};
}

Json serialize(SampleResult const& result)
{
    require(result.tag == completed_tag, "Missing sample completion marker");
    require(result.valid || result.audit.guard_violations, "Invalid unaccounted sample");
    auto j = audit_json(result.audit);
    j["field_tesla"] = vector_json(result.field, static_cast<double>(tesla));
    j["valid"] = bool(result.valid);
    j["completed_tag"] = result.tag;
    return j;
}

Json serialize(TrackResult const& result)
{
    require(result.tag == completed_tag, "Missing track completion marker");
    require(result.status == 0 || result.status == 1, "Integration failed or exhausted work limit");
    require(result.steps > 0 && detail::finite(result.end.pos) && detail::finite(result.end.mom)
                && detail::finite(result.path) && detail::finite(result.momentum_relative_drift)
                && detail::finite(result.max_step_used), "Invalid integration diagnostics");
    require(result.status != 0 || (result.path > 0 && result.audit.guard_violations == 0),
            "Successful track did not advance safely");
    require(result.status != 1 || (result.audit.guard_violations > 0 && result.audit.min_margin < 0),
            "Expected finite outside-domain rejection");
    auto j = audit_json(result.audit);
    j["end_position_mm"] = vector_json(result.end.pos, static_cast<double>(millimeter));
    j["end_momentum_mev_c"] = vector_json(result.end.mom, 1);
    j["path_mm"] = result.path / millimeter;
    j["steps"] = result.steps;
    j["momentum_relative_drift"] = result.momentum_relative_drift;
    j["max_step_mm_used"] = result.max_step_used / millimeter;
    j["status"] = result.status;
    j["completed_tag"] = result.tag;
    return j;
}

int run(std::filesystem::path const& path)
{
    static_assert(std::is_same_v<celeritas::real_type, double>);
    auto request = read_json(path);
    require(request.at("format") == "m2fs-field-probe-v1", "Wrong request format");
    require(request.at("map_file").is_string(), "Expected map filename");
    auto map_path = path.parent_path() / request.at("map_file").get<std::string>();
    auto input = read_cartmap(map_path.string());
    celeritas::validate_input(input.driver_options);
    Domain domain{};
    std::array<celeritas::AxisGrid<celeritas::real_type>, 3> axes{input.x, input.y, input.z};
    for (int i = 0; i < 3; ++i)
    {
        require(std::abs(axes[i].min / millimeter) <= 1e7 && std::abs(axes[i].max / millimeter) <= 1e7,
                "Map coordinate exceeds the bounded 10 km probe domain");
        double spacing = (axes[i].max - axes[i].min) / (axes[i].num - 1);
        domain.lower[i] = axes[i].min + spacing;
        domain.upper[i] = axes[i].max - spacing;
    }
    auto const& point_json = request.at("sample_points_mm");
    require(point_json.is_array() && !point_json.empty() && point_json.size() <= 65536,
            "Require 1..65536 sample points");
    std::vector<Real3> points;
    points.reserve(point_json.size());
    for (auto const& item : point_json)
    {
        Real3 point = vector(item);
        for (double& coordinate : point)
            coordinate *= millimeter;
        points.push_back(point);
    }
    auto const& track_json = request.at("tracks");
    require(track_json.is_array() && !track_json.empty() && track_json.size() <= 256,
            "Require 1..256 tracks");
    std::vector<TrackInput> tracks;
    std::set<std::string> ids;
    for (auto const& j : track_json)
    {
        require(j.at("id").is_string() && ids.insert(j.at("id").get<std::string>()).second,
                "Track IDs must be unique strings");
        TrackInput t{};
        t.start.pos = vector(j.at("position_mm"));
        for (double& coordinate : t.start.pos)
        {
            require(std::abs(coordinate) <= 1e7, "Initial position exceeds probe domain");
            coordinate *= millimeter;
        }
        t.start.mom = vector(j.at("momentum_mev_c"));
        double momentum = detail::norm(t.start.mom);
        require(momentum >= 0.1 && momentum <= 100000, "Momentum outside probe range");
        t.charge = number(j.at("charge_e"));
        require(t.charge == -1 || t.charge == 0 || t.charge == 1, "Require charge -1, 0 or 1");
        double length_mm = number(j.at("length_mm"));
        double step_mm = number(j.at("max_step_mm"));
        require(length_mm >= 1e-4 && length_mm <= 10000 && step_mm >= 1e-4 && step_mm <= 100,
                "Length or step outside bounded probe range");
        t.length = length_mm * millimeter;
        t.step_cap = step_mm * millimeter;
        auto method = j.at("method").get<std::string>();
        require(method == "fixed" || method == "adaptive", "Unknown integration method");
        t.adaptive = method == "adaptive";
        t.options.epsilon_rel_max = number(j.at("epsilon_rel_max"));
        t.options.delta_chord = number(j.at("delta_chord_mm")) * millimeter;
        require(t.options.epsilon_rel_max >= 1e-12 && t.options.epsilon_rel_max <= 0.1,
                "Relative tolerance outside probe range");
        celeritas::validate_input(t.options);
        tracks.push_back(t);
    }
    require(celeritas::version_major == 0 && celeritas::version_minor == 6 && celeritas::version_patch == 3,
            "Expected installed Celeritas 0.6.3");
    require(celeritas::Device::num_devices() > 0, "A CUDA device is required; no CPU fallback");
    celeritas::activate_device(celeritas::Device{0});
    auto const& device = celeritas::device();
    require(device.capability() == 80 && device.name().find("A100") != std::string::npos,
            "This probe requires an A100 sm80 device");
    celeritas::CartMapFieldParams params(input);
    auto gpu_samples = sample_device(params, domain, points);
    auto gpu_tracks = track_device(params, domain, tracks);
    Json host_samples = Json::array(), host_tracks = Json::array();
    Json device_samples = Json::array(), device_tracks = Json::array();
    for (auto const& point : points)
        host_samples.push_back(serialize(detail::sample_one(params.host_ref(), domain, point)));
    for (auto const& track : tracks)
        host_tracks.push_back(serialize(detail::track_one(params.host_ref(), domain, track)));
    for (auto const& result : gpu_samples)
        device_samples.push_back(serialize(result));
    for (auto const& result : gpu_tracks)
        device_tracks.push_back(serialize(result));
    Json output{{"format", "m2fs-field-probe-result-v1"}, {"input", request},
                {"celeritas_version", "0.6.3"},
                {"device", {{"name", device.name()}, {"capability", device.capability()}, {"synchronized", true}}},
                {"samples", {{"host", host_samples}, {"device", device_samples}}},
                {"tracks", {{"host", host_tracks}, {"device", device_tracks}}}};
    std::cout << output.dump(2) << '\n';
    require(bool(std::cout), "Cannot write complete result");
    return 0;
}
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
            throw std::runtime_error("Usage: m2fs-field-probe REQUEST.json");
        return m2fs::probe::run(std::filesystem::absolute(argv[1]));
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
