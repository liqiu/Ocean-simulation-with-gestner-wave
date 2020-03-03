#include "mesh.h"

#include <sutil/Matrix.h>
#include <sutil/Exception.h>
#include <sutil/vec_math.h>

#include <optix_stubs.h>

#include <cuda_runtime.h>


Mesh::~Mesh()
{
    CUDA_CHECK(cudaFree((void*)mdGasOutputBuffer));
    CUDA_CHECK(cudaFree((void*)mdTempBufferGas));
}

void Mesh::generateMesh(float t)
{
    mVerticesSize = mSamplesX * mSamplesZ * sizeof(Vertex);// *vertices.size();
    mIndicesSize = 4 * (mSamplesX - 1) * (mSamplesZ - 1) * 6;// indices.size();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdVertices), mVerticesSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdIndices), mIndicesSize));

    std::shared_ptr<Wave> wave = mpWaves[0];
    cudaGenerateGridMesh(reinterpret_cast<Vertex*>(mdVertices), reinterpret_cast<unsigned int*>(mdIndices),
        wave->amplitude, wave->waveLength, wave->frequency, mSamplesX, mLength, t);
}

void Mesh::updateMesh(float t)
{
    std::shared_ptr<Wave> wave = mpWaves[0];
    cudaUpdateGridMesh(reinterpret_cast<Vertex*>(mdVertices), wave->amplitude, wave->waveLength,
        wave->frequency, mSamplesX, mLength, t);
}

void Mesh::buildAccelerationStructure(OptixDeviceContext context)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = mSamplesX * mSamplesZ;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.vertexBuffers = &mdVertices;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = 12;
    triangle_input.triangleArray.numIndexTriplets = (mSamplesX - 1) * (mSamplesZ - 1) * 2;
    triangle_input.triangleArray.indexBuffer = mdIndices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input,
        1,  // Number of build input
        &gas_buffer_sizes));
    if (!mdTempBufferGas)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdTempBufferGas), gas_buffer_sizes.tempSizeInBytes));

    if (!mdGasOutputBuffer)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdGasOutputBuffer), gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,              // CUDA stream
        &accel_options,
        &triangle_input,
        1,              // num build inputs
        mdTempBufferGas,
        gas_buffer_sizes.tempSizeInBytes,
        mdGasOutputBuffer,
        gas_buffer_sizes.outputSizeInBytes,
        &mGasHandle,
        nullptr,  // emitted property list
        0               // num emitted properties
    ));
}

void Mesh::updateAccelerationStructure(OptixDeviceContext context)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = mSamplesX * mSamplesZ;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.vertexBuffers = &mdVertices;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = 12;
    triangle_input.triangleArray.numIndexTriplets = (mSamplesX - 1) * (mSamplesZ - 1) * 2;
    triangle_input.triangleArray.indexBuffer = mdIndices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input,
        1,  // Number of build input
        &gas_buffer_sizes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,              // CUDA stream
        &accel_options,
        &triangle_input,
        1,              // num build inputs
        mdTempBufferGas,
        gas_buffer_sizes.tempUpdateSizeInBytes,
        mdGasOutputBuffer,
        gas_buffer_sizes.outputSizeInBytes,
        &mGasHandle,
        nullptr,  // emitted property list
        0             // num emitted properties
    ));
}
