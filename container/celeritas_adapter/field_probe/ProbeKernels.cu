#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "corecel/data/DeviceVector.hh"
#include "celeritas/field/CartMapFieldData.hh"
#include "ProbeAlgorithms.hh"

namespace m2fs::probe
{
namespace
{
void check_cuda(cudaError_t code, char const* operation)
{
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

__global__ void sample_kernel(celeritas::CartMapFieldParams::DeviceRef params, Domain domain,
                              Real3 const* points, SampleResult* result, unsigned int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        result[i] = detail::sample_one(params, domain, points[i]);
}

__global__ void track_kernel(celeritas::CartMapFieldParams::DeviceRef params, Domain domain,
                             TrackInput const* tracks, TrackResult* result, unsigned int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        result[i] = detail::track_one(params, domain, tracks[i]);
}
}

std::vector<SampleResult> sample_device(celeritas::CartMapFieldParams const& field, Domain domain,
                                       std::vector<Real3> const& points)
{
    auto size = static_cast<celeritas::size_type>(points.size());
    celeritas::DeviceVector<Real3> input(size);
    celeritas::DeviceVector<SampleResult> output(size);
    input.copy_to_device({points.data(), size});
    check_cuda(cudaMemset(output.data(), 0, size * sizeof(SampleResult)), "sample sentinel initialization");
    sample_kernel<<<(size + 127) / 128, 128>>>(field.device_ref(), domain, input.data(), output.data(), size);
    check_cuda(cudaGetLastError(), "sample kernel launch");
    check_cuda(cudaDeviceSynchronize(), "sample kernel synchronization");
    std::vector<SampleResult> result(size);
    output.copy_to_host({result.data(), size});
    return result;
}

std::vector<TrackResult> track_device(celeritas::CartMapFieldParams const& field, Domain domain,
                                     std::vector<TrackInput> const& tracks)
{
    auto size = static_cast<celeritas::size_type>(tracks.size());
    celeritas::DeviceVector<TrackInput> input(size);
    celeritas::DeviceVector<TrackResult> output(size);
    input.copy_to_device({tracks.data(), size});
    check_cuda(cudaMemset(output.data(), 0, size * sizeof(TrackResult)), "track sentinel initialization");
    track_kernel<<<(size + 31) / 32, 32>>>(field.device_ref(), domain, input.data(), output.data(), size);
    check_cuda(cudaGetLastError(), "track kernel launch");
    check_cuda(cudaDeviceSynchronize(), "track kernel synchronization");
    std::vector<TrackResult> result(size);
    output.copy_to_host({result.data(), size});
    return result;
}
}
