#include "CartMapIO.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <G4LogicalVolume.hh>
#include <G4RotationMatrix.hh>
#include <G4SystemOfUnits.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSolid.hh>
#include <nlohmann/json.hpp>
#include "geocel/g4/Convert.hh"
#include "celeritas/ext/GeantUnits.hh"

namespace m2fs
{
namespace
{
using Json = nlohmann::json;
void require(bool ok, char const* message)
{
    if (!ok)
        throw std::runtime_error(message);
}

double number(Json const& value)
{
    require(value.is_number(), "Map value must be numeric");
    double result = value.get<double>();
    require(std::isfinite(result), "Map value must be finite");
    return result;
}
}

celeritas::CartMapFieldInput read_cartmap(std::string const& filename)
{
    std::ifstream stream(filename);
    require(bool(stream), "Cannot open Cartesian map");
    Json j;
    stream >> j;
    require(j.at("format") == "m2fs-cartmap-v1", "Wrong Cartesian map format");
    require(j.at("length_unit") == "mm" && j.at("field_unit") == "tesla", "Wrong map units");
    require(j.at("coordinates") == "gdml-world", "Map must use GDML world coordinates");
    require(j.at("outside_policy") == "world-contained", "Unsupported outside-map policy");
    require(j.at("order") == "xyz-component", "Wrong map flattening order");
    celeritas::CartMapFieldInput result;
    std::array<celeritas::AxisGrid<celeritas::real_type>*, 3> axes{&result.x, &result.y, &result.z};
    auto const& input_axes = j.at("axes");
    require(input_axes.is_array() && input_axes.size() == 3, "Expected three axes");
    std::size_t count = 3;
    for (std::size_t i = 0; i < axes.size(); ++i)
    {
        auto const& a = input_axes.at(i);
        require(a.at("num").is_number_unsigned(), "Axis count must be an unsigned integer");
        auto n = a.at("num").get<std::size_t>();
        require(n >= 3 && n <= 1000000, "Axis count outside supported range");
        require(count <= 30000000 / n, "Map exceeds 30M scalar diagnostic limit");
        count *= n;
        double lo = number(a.at("min_mm"));
        double hi = number(a.at("max_mm"));
        require(lo < hi, "Map axis must increase");
        axes[i]->min = celeritas::convert_from_geant(lo * CLHEP::mm, celeritas::clhep_length);
        axes[i]->max = celeritas::convert_from_geant(hi * CLHEP::mm, celeritas::clhep_length);
        axes[i]->num = n;
        require(std::isfinite(axes[i]->min) && std::isfinite(axes[i]->max), "Map axis conversion overflow");
        double step = (axes[i]->max - axes[i]->min) / (n - 1);
        require(std::isfinite(step) && step > 0, "Invalid map spacing");
        float flo = static_cast<float>(axes[i]->min);
        float fhi = static_cast<float>(axes[i]->max);
        float scale = static_cast<float>(1 / step);
        require(std::isfinite(flo) && std::isfinite(fhi) && flo < fhi
                    && std::isfinite(scale) && scale > 0
                    && std::isfinite(scale * flo) && std::isfinite(scale * fhi),
                "Grid affine transform is not representable in covfie float storage");
        require(step > 8 * std::numeric_limits<float>::epsilon()
                           * std::max(std::abs(axes[i]->min), std::abs(axes[i]->max)),
                "Grid spacing is unresolved in covfie float coordinates");
    }
    auto const& field = j.at("field");
    require(field.is_array() && field.size() == count, "Map array size disagrees with axes");
    result.field.reserve(count);
    for (auto const& value : field)
    {
        double b = celeritas::convert_from_geant(number(value) * CLHEP::tesla, celeritas::clhep_field);
        require(std::isfinite(b) && std::abs(b) <= std::numeric_limits<float>::max(), "Map field conversion overflow");
        result.field.push_back(b);
    }
    require(bool(result), "Invalid Cartesian map input");
    return result;
}

void require_world_inside(celeritas::CartMapFieldInput const& map, G4VPhysicalVolume const& world)
{
    require(!world.GetRotation() || world.GetRotation()->isIdentity(), "Rotated world is unsupported");
    G4ThreeVector lo, hi;
    world.GetLogicalVolume()->GetSolid()->BoundingLimits(lo, hi);
    lo += world.GetTranslation();
    hi += world.GetTranslation();
    std::array<celeritas::AxisGrid<celeritas::real_type>, 3> axes{map.x, map.y, map.z};
    for (std::size_t i = 0; i < axes.size(); ++i)
    {
        auto const& a = axes[i];
        double step = (a.max - a.min) / (a.num - 1);
        double lower = celeritas::convert_from_geant(lo[i], celeritas::clhep_length);
        double upper = celeritas::convert_from_geant(hi[i], celeritas::clhep_length);
        require(std::isfinite(lower) && std::isfinite(upper) && lower < upper,
                "World must have finite nonempty bounds");
        require(lower >= a.min + step && upper <= a.max - step,
                "World must fit inside Cartesian map with a one-cell margin on each face");
    }
}
}
