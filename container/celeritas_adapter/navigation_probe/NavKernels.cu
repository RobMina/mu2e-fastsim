#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "corecel/data/DeviceVector.hh"
#include "NavAlgorithms.hh"

namespace m2fs::nav
{
namespace
{
void check_cuda(cudaError_t code, char const* operation)
{
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

__global__ void navigation_kernel(
    celeritas::CartMapFieldParams::DeviceRef field,
    celeritas::DeviceCRef<celeritas::VecgeomParamsData> geometry,
    celeritas::DeviceRef<celeritas::VecgeomStateData> geometry_state,
    celeritas::DeviceCRef<celeritas::ParticleParamsData> particles,
    celeritas::DeviceRef<celeritas::ParticleStateData> particle_state,
    Domain domain, TrackInput const* tracks, unsigned size, unsigned stride,
    TrackResult* result, StepRecord* records)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        result[i] = detail::track_one(field, geometry, geometry_state, particles, particle_state,
                                     celeritas::TrackSlotId{i}, domain, tracks[i], records + i * stride, stride);
}
}

Batch run_device(celeritas::CartMapFieldParams::DeviceRef const& field,
                 celeritas::DeviceCRef<celeritas::VecgeomParamsData> const& geometry,
                 celeritas::DeviceRef<celeritas::VecgeomStateData> const& geometry_state,
                 celeritas::DeviceCRef<celeritas::ParticleParamsData> const& particles,
                 celeritas::DeviceRef<celeritas::ParticleStateData> const& particle_state,
                 Domain domain, std::vector<TrackInput> const& tracks, unsigned stride)
{
    auto size = static_cast<celeritas::size_type>(tracks.size());
    celeritas::DeviceVector<TrackInput> input(size);
    celeritas::DeviceVector<TrackResult> results(size);
    celeritas::DeviceVector<StepRecord> records(size * stride);
    input.copy_to_device({tracks.data(), size});
    check_cuda(cudaMemset(results.data(), 0, size * sizeof(TrackResult)), "initialize track tags");
    check_cuda(cudaMemset(records.data(), 0, size * stride * sizeof(StepRecord)), "initialize step tags");
    navigation_kernel<<<(size + 31) / 32, 32>>>(field, geometry, geometry_state, particles, particle_state,
                                              domain, input.data(), size, stride, results.data(), records.data());
    check_cuda(cudaGetLastError(), "navigation kernel launch");
    check_cuda(cudaDeviceSynchronize(), "navigation kernel completion");
    Batch result;
    result.stride = stride;
    result.tracks.resize(size);
    result.steps.resize(size * stride);
    results.copy_to_host({result.tracks.data(), size});
    records.copy_to_host({result.steps.data(), size * stride});
    return result;
}
}
