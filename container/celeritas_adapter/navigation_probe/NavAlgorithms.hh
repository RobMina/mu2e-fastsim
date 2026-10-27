#pragma once

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include "corecel/Macros.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/Units.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/FieldPropagator.hh"
#include "celeritas/field/FieldSubstepper.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "geocel/vg/VecgeomTrackView.hh"
#include "NavTypes.hh"

namespace m2fs::nav::detail
{
CELER_FUNCTION inline bool finite(double v)
{
    return v == v && std::fabs(v) <= celeritas::numeric_limits<double>::max();
}

CELER_FUNCTION inline bool finite(Real3 const& v)
{
    return finite(v[0]) && finite(v[1]) && finite(v[2]);
}

CELER_FUNCTION inline double norm(Real3 const& v)
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

CELER_FUNCTION inline void fail_field(bool guard)
{
#if defined(CELER_DEVICE_COMPILE) && CELER_DEVICE_COMPILE
    printf("%s\n", guard ? "M2FS_NAV_MAP_GUARD" : "M2FS_NAV_NONFINITE_FIELD");
    asm("trap;");
#else
    throw std::runtime_error(guard ? "M2FS_NAV_MAP_GUARD" : "M2FS_NAV_NONFINITE_FIELD");
#endif
}

class GuardedField
{
  public:
    CELER_FUNCTION GuardedField(celeritas::CartMapField::FieldParamsRef const& field,
                               Domain domain, Audit* audit)
        : field_(field), domain_(domain), audit_(audit)
    {
    }

    CELER_FUNCTION Real3 operator()(Real3 const& pos) const
    {
        if (!finite(pos))
        {
            fail_field(false);
            return {};
        }
        double margin = celeritas::numeric_limits<double>::max();
        for (int i = 0; i < 3; ++i)
        {
            margin = celeritas::min(margin, pos[i] - domain_.lower[i]);
            margin = celeritas::min(margin, domain_.upper[i] - pos[i]);
        }
        audit_->min_margin = celeritas::min(audit_->min_margin, margin);
        if (margin < 0)
        {
            fail_field(true);
            return {};
        }
        ++audit_->field_evaluations;
        Real3 result = field_(pos);
        if (!finite(result))
            fail_field(false);
        return result;
    }

  private:
    celeritas::CartMapField field_;
    Domain domain_;
    Audit* audit_;
};

template<class Advance>
class AuditedSubstepper
{
  public:
    CELER_FUNCTION AuditedSubstepper(Advance&& advance, Audit* audit)
        : advance_(::celeritas::forward<Advance>(advance)), audit_(audit)
    {
    }

    CELER_FUNCTION celeritas::Substep operator()(double distance, celeritas::OdeState const& state)
    {
        ++audit_->substep_calls;
        return advance_(distance, state);
    }

    CELER_FUNCTION short max_substeps() const { return advance_.max_substeps(); }
    CELER_FUNCTION double minimum_step() const { return advance_.minimum_step(); }
    CELER_FUNCTION double delta_intersection() const { return advance_.delta_intersection(); }

  private:
    Advance advance_;
    Audit* audit_;
};

template<class A>
CELER_FUNCTION AuditedSubstepper(A&&, Audit*) -> AuditedSubstepper<A>;

CELER_FUNCTION inline State snapshot(celeritas::VecgeomTrackView const& geo)
{
    State result;
    result.position = geo.pos();
    result.direction = geo.dir();
    result.boundary = geo.is_on_boundary();
    result.outside = geo.is_outside();
    if (!result.outside)
    {
        result.volume = static_cast<int>(geo.volume_id().unchecked_get());
        result.instance = static_cast<int>(geo.volume_instance_id().unchecked_get());
    }
    return result;
}

CELER_FUNCTION inline TrackResult track_one(
    celeritas::CartMapField::FieldParamsRef const& field,
    celeritas::VecgeomTrackView::ParamsRef const& geometry,
    celeritas::VecgeomTrackView::StateRef const& geometry_state,
    celeritas::ParticleTrackView::ParamsRef const& particles,
    celeritas::ParticleTrackView::StateRef const& particle_state,
    celeritas::TrackSlotId tid, Domain domain, TrackInput const& input,
    StepRecord* records, unsigned capacity)
{
    TrackResult result;
    result.audit.min_margin = celeritas::numeric_limits<double>::max();
    celeritas::VecgeomTrackView geo(geometry, geometry_state, tid);
    geo = celeritas::GeoTrackInitializer{input.position, input.direction};
    celeritas::ParticleTrackView particle(particles, particle_state, tid);
    particle = {celeritas::ParticleId{input.particle_id}, celeritas::units::MevEnergy{input.kinetic_mev}};
    result.final = snapshot(geo);
    result.status = geo.is_outside() ? 4 : 0;
    double momentum = particle.momentum().value();
    double speed_mm_ns = particle.speed().value() * 299.792458;
    unsigned tiny_steps = 0;
    GuardedField guarded(field, domain, &result.audit);
    celeritas::MagFieldEquation equation(guarded, particle.charge());
    celeritas::DormandPrinceIntegrator integrate(equation);
    celeritas::FieldSubstepper substepper(input.options, integrate);
    AuditedSubstepper advance(substepper, &result.audit);
    auto propagate = celeritas::FieldPropagator{advance, particle, geo};
    while (result.status == 0 && result.path < input.length)
    {
        if (result.count == capacity)
        {
            result.status = 3;
            break;
        }
        StepRecord rec;
        rec.before = snapshot(geo);
        rec.requested = celeritas::min(input.cap, input.length - result.path);
        auto before_fields = result.audit.field_evaluations;
        auto before_substeps = result.audit.substep_calls;
        auto moved = propagate(rec.requested);
        rec.arrival = snapshot(geo);
        rec.after = rec.arrival;
        rec.distance = moved.distance;
        rec.boundary = moved.boundary;
        rec.looping = moved.looping;
        rec.geometry_failed = geo.failed();
        rec.momentum_mev_c = momentum;
        rec.field_evaluations = result.audit.field_evaluations - before_fields;
        rec.substep_calls = result.audit.substep_calls - before_substeps;
        double next = result.path + moved.distance;
        if (!finite(rec.arrival.position) || !finite(rec.arrival.direction) || !finite(moved.distance)
            || moved.distance <= 0 || moved.distance > rec.requested * (1 + 1e-10)
            || next <= result.path || rec.geometry_failed
            || std::fabs(norm(rec.arrival.direction) - 1) > 1e-10
            || moved.boundary != rec.arrival.boundary)
        {
            result.status = 5;
        }
        else
        {
            result.path = next;
            if (moved.boundary)
            {
                geo.cross_boundary();
                rec.after = snapshot(geo);
                ++result.crossings;
            }
            if (moved.looping)
                result.status = 2;
            else if (geo.is_outside())
                result.status = 1;
            double tiny = celeritas::max(2 * input.options.minimum_step,
                                        0.11 * input.options.delta_intersection);
            bool suspicious = moved.distance <= tiny && !moved.boundary
                              && moved.distance < rec.requested * (1 - 1e-10)
                              && rec.before.instance == rec.after.instance;
            tiny_steps = suspicious ? tiny_steps + 1 : 0;
            if (tiny_steps > 8 && result.status == 0)
                result.status = 6;
        }
        rec.cumulative_path = result.path;
        rec.time_ns = result.path / celeritas::units::millimeter / speed_mm_ns;
        rec.tag = completed_tag;
        records[result.count++] = rec;
        result.final = rec.after;
        if (input.length - result.path <= 64 * celeritas::numeric_limits<double>::epsilon() * input.length)
            break;
    }
    result.tag = completed_tag;
    return result;
}
}
