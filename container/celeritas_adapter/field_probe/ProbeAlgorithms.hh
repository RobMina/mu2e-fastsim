#pragma once

#include <cmath>
#include "corecel/Macros.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/FieldSubstepper.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "ProbeTypes.hh"

namespace m2fs::probe
{
namespace detail
{
using celeritas::numeric_limits;

CELER_FUNCTION inline bool finite(double v)
{
    return v == v && v <= numeric_limits<double>::max() && v >= -numeric_limits<double>::max();
}

CELER_FUNCTION inline bool finite(Real3 const& v)
{
    return finite(v[0]) && finite(v[1]) && finite(v[2]);
}

CELER_FUNCTION inline double norm(Real3 const& p)
{
    return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
}

CELER_FUNCTION inline Real3 invalid_field()
{
    double n = numeric_limits<double>::quiet_NaN();
    return {n, n, n};
}

CELER_FUNCTION inline Audit new_audit()
{
    Audit result{};
    result.min_margin = numeric_limits<double>::max();
    return result;
}

class GuardedField
{
  public:
    CELER_FUNCTION GuardedField(celeritas::CartMapField::FieldParamsRef const& params,
                               Domain domain, Audit* audit)
        : field_(params), domain_(domain), audit_(audit)
    {
    }

    CELER_FUNCTION Real3 operator()(Real3 const& pos) const
    {
        if (audit_->guard_violations || audit_->numerical_failures)
            return invalid_field();
        if (!finite(pos))
        {
            ++audit_->numerical_failures;
            return invalid_field();
        }
        double margin = numeric_limits<double>::max();
        for (int i = 0; i < 3; ++i)
        {
            double a = pos[i] - domain_.lower[i];
            double b = domain_.upper[i] - pos[i];
            margin = a < margin ? a : margin;
            margin = b < margin ? b : margin;
        }
        audit_->min_margin = margin < audit_->min_margin ? margin : audit_->min_margin;
        if (margin < 0)
        {
            ++audit_->guard_violations;
            return invalid_field();
        }
        ++audit_->field_evaluations;
        auto result = field_(pos);
        if (!finite(result))
        {
            ++audit_->numerical_failures;
            return invalid_field();
        }
        return result;
    }

  private:
    celeritas::CartMapField field_;
    Domain domain_;
    Audit* audit_;
};

CELER_FUNCTION inline SampleResult sample_one(celeritas::CartMapField::FieldParamsRef const& params,
                                              Domain domain, Real3 const& point)
{
    SampleResult result{};
    result.audit = new_audit();
    GuardedField field(params, domain, &result.audit);
    result.field = field(point);
    result.valid = result.audit.guard_violations == 0 && finite(result.field);
    result.tag = completed_tag;
    return result;
}

CELER_FUNCTION inline TrackResult track_one(celeritas::CartMapField::FieldParamsRef const& params,
                                            Domain domain, TrackInput const& input)
{
    TrackResult result{};
    result.end = input.start;
    result.audit = new_audit();
    GuardedField field(params, domain, &result.audit);
    celeritas::MagFieldEquation equation(field, celeritas::units::ElementaryCharge{input.charge});
    celeritas::DormandPrinceIntegrator integrate(equation);
    celeritas::FieldSubstepper advance(input.options, integrate);
    double p0 = norm(input.start.mom);
    while (result.path < input.length && result.steps < max_steps)
    {
        double remaining = input.length - result.path;
        double request = remaining < input.step_cap ? remaining : input.step_cap;
        celeritas::Substep step;
        if (input.adaptive)
        {
            step = advance(request, result.end);
        }
        else
        {
            auto integrated = integrate(request, result.end);
            step.state = integrated.end_state;
            step.length = request;
        }
        ++result.steps;
        if (result.audit.numerical_failures)
        {
            result.status = 2;
            break;
        }
        if (result.audit.guard_violations)
        {
            result.status = 1;
            break;
        }
        double p = norm(step.state.mom);
        double next_path = result.path + step.length;
        if (!finite(step.state.pos) || !finite(step.state.mom) || !finite(p) || p <= 0
            || !finite(step.length) || step.length <= 0 || step.length > request * (1 + 1e-12)
            || next_path <= result.path
            || (step.state.pos == result.end.pos && step.state.mom == result.end.mom))
        {
            result.status = 2;
            break;
        }
        result.end = step.state;
        result.path = next_path;
        double drift = std::fabs(p / p0 - 1);
        result.momentum_relative_drift = drift > result.momentum_relative_drift ? drift : result.momentum_relative_drift;
        result.max_step_used = step.length > result.max_step_used ? step.length : result.max_step_used;
        if (input.length - result.path <= 32 * numeric_limits<double>::epsilon() * input.length)
            break;
    }
    if (!result.status && input.length - result.path > 32 * numeric_limits<double>::epsilon() * input.length)
        result.status = 3;
    result.tag = completed_tag;
    return result;
}
}
}
