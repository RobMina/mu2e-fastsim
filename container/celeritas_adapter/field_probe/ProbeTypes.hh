#pragma once

#include <vector>
#include "corecel/cont/Array.hh"
#include "celeritas/field/CartMapFieldParams.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/Types.hh"

namespace m2fs::probe
{
using Real3 = celeritas::Array<celeritas::real_type, 3>;
using State = celeritas::OdeState;
inline constexpr unsigned int completed_tag = 0x6f31;
inline constexpr unsigned int max_steps = 200000;

struct Domain
{
    Real3 lower;
    Real3 upper;
};

struct Audit
{
    unsigned int field_evaluations{};
    unsigned int guard_violations{};
    unsigned int numerical_failures{};
    double min_margin{};
};

struct SampleResult
{
    Real3 field{};
    Audit audit{};
    unsigned int valid{};
    unsigned int tag{};
};

struct TrackInput
{
    State start;
    double charge;
    double length;
    double step_cap;
    celeritas::FieldDriverOptions options;
    bool adaptive;
};

struct TrackResult
{
    State end{};
    Audit audit{};
    double path{};
    double momentum_relative_drift{};
    double max_step_used{};
    unsigned int steps{};
    unsigned int status{};
    unsigned int tag{};
};

std::vector<SampleResult> sample_device(celeritas::CartMapFieldParams const&, Domain,
                                       std::vector<Real3> const&);
std::vector<TrackResult> track_device(celeritas::CartMapFieldParams const&, Domain,
                                     std::vector<TrackInput> const&);
}
