#pragma once

#include <string>
#include "celeritas/field/CartMapFieldInput.hh"

class G4VPhysicalVolume;

namespace m2fs
{
celeritas::CartMapFieldInput read_cartmap(std::string const& filename);
void require_world_inside(celeritas::CartMapFieldInput const&, G4VPhysicalVolume const&);
}
