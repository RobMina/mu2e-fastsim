#pragma once

class G4Event;

namespace m2fs
{
void initialize_event_output();
void write_event(G4Event const&);
}
