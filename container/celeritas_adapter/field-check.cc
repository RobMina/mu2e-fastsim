#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <G4SystemOfUnits.hh>
#include <nlohmann/json.hpp>
#include "accel/CartMapMagneticField.hh"
#include "geocel/g4/Convert.hh"
#include "CartMapIO.hh"

int main(int argc, char** argv)
{
    try
    {
        if (argc != 3)
            throw std::runtime_error("Usage: m2fs-field-check MAP.json POINTS.json");
        auto input = m2fs::read_cartmap(argv[1]);
        celeritas::CartMapMagneticField field(std::make_shared<celeritas::CartMapFieldParams>(input));
        std::ifstream stream(argv[2]);
        if (!stream)
            throw std::runtime_error("Cannot read point file");
        nlohmann::json points;
        stream >> points;
        if (!points.is_array())
            throw std::runtime_error("Expected array of points in mm");
        nlohmann::json values = nlohmann::json::array();
        std::array<celeritas::AxisGrid<celeritas::real_type>, 3> axes{input.x, input.y, input.z};
        for (auto const& p : points)
        {
            if (!p.is_array() || p.size() != 3)
                throw std::runtime_error("Expected three coordinates");
            double point[4] = {0, 0, 0, 0};
            for (std::size_t i = 0; i < 3; ++i)
            {
                if (!p[i].is_number())
                    throw std::runtime_error("Point must be numeric");
                point[i] = p[i].get<double>() * CLHEP::mm;
                double native = point[i] / celeritas::clhep_length;
                double step = (axes[i].max - axes[i].min) / (axes[i].num - 1);
                if (!std::isfinite(native) || native < axes[i].min + step || native > axes[i].max - step)
                    throw std::runtime_error("Point must have a one-cell interpolation margin");
            }
            double b[3] = {0, 0, 0};
            field.GetFieldValue(point, b);
            values.push_back({b[0] / CLHEP::tesla, b[1] / CLHEP::tesla, b[2] / CLHEP::tesla});
        }
        std::cout << nlohmann::json{{"field_unit", "tesla"}, {"field", values}}.dump(2) << '\n';
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
