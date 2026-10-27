#pragma once

#include <vector>
#include "corecel/cont/Array.hh"
#include "celeritas/field/CartMapFieldParams.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/phys/ParticleData.hh"
#include "geocel/vg/VecgeomData.hh"

namespace m2fs::nav
{
using Real3 = celeritas::Array<celeritas::real_type, 3>;
inline constexpr unsigned completed_tag = 0x6e37;
inline constexpr double electron_mass_mev = 0.5109989461;

struct Domain
{
    Real3 lower;
    Real3 upper;
};

struct Audit
{
    unsigned field_evaluations{};
    unsigned substep_calls{};
    double min_margin{};
};

struct TrackInput
{
    Real3 position;
    Real3 direction;
    double kinetic_mev;
    double length;
    double cap;
    unsigned particle_id;
    celeritas::FieldDriverOptions options;
};

struct State
{
    Real3 position{};
    Real3 direction{};
    int volume{-1};
    int instance{-1};
    bool boundary{};
    bool outside{};
};

struct StepRecord
{
    State before{};
    State arrival{};
    State after{};
    double requested{};
    double distance{};
    double cumulative_path{};
    double momentum_mev_c{};
    double time_ns{};
    unsigned field_evaluations{};
    unsigned substep_calls{};
    bool boundary{};
    bool looping{};
    bool geometry_failed{};
    unsigned tag{};
};

// Status 0=length, 1=world exit, 2=loop limit, 3=record limit, 4=outside start, 5=invalid, 6=no progress.
struct TrackResult
{
    State final{};
    Audit audit{};
    double path{};
    unsigned count{};
    unsigned crossings{};
    unsigned status{};
    unsigned tag{};
};

struct Batch
{
    unsigned stride{};
    std::vector<TrackResult> tracks;
    std::vector<StepRecord> steps;
};

Batch run_device(celeritas::CartMapFieldParams::DeviceRef const&,
                 celeritas::DeviceCRef<celeritas::VecgeomParamsData> const&,
                 celeritas::DeviceRef<celeritas::VecgeomStateData> const&,
                 celeritas::DeviceCRef<celeritas::ParticleParamsData> const&,
                 celeritas::DeviceRef<celeritas::ParticleStateData> const&,
                 Domain, std::vector<TrackInput> const&, unsigned stride);
}
