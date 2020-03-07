#include "WaveMesh.h"

#include <sutil/Matrix.h>
#include <sutil/MeshGroup.h>
#include <sutil/Exception.h>
#include <sutil/vec_math.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <algorithm>


WaveMesh::WaveMesh()
{
    mTransform = sutil::Matrix4x4::identity();
}

WaveMesh::~WaveMesh()
{
    CUDA_CHECK(cudaFree((void*)mdGasOutputBuffer));
    CUDA_CHECK(cudaFree((void*)mdTempBufferGas));
    CUDA_CHECK(cudaFree((void*)mdWaves));
}

void WaveMesh::generateMesh(float t)
{
    size_t verticesByteSize = mSamplesX * mSamplesZ * sizeof(float3);
    size_t indicesByteSize = 4 * (mSamplesX - 1) * (mSamplesZ - 1) * 6;
    size_t waveByteSize = mWaves.size() * sizeof(Wave);

    float3* dVertices, *dNormals;
    uint32_t* dIndices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVertices), verticesByteSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dNormals), verticesByteSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIndices), indicesByteSize));
    mMeshBuffer.pos = dVertices;
    mMeshBuffer.normal = dNormals;
    mMeshBuffer.indices = dIndices;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdWaves), waveByteSize));
    CUDA_CHECK(cudaMemcpy((void*)mdWaves, mWaves.data(), waveByteSize, cudaMemcpyHostToDevice));

    cudaGenerateGridMesh(mMeshBuffer, reinterpret_cast<Wave*>(mdWaves), mWaves.size(),
        mSamplesX, mSamplesZ, mLength, t);

    mpMesh = std::make_shared<sutil::MeshGroup>();
    mpMesh->name = "Wave";
    mpMesh->material_idx.push_back(-1);
    mpMesh->texcoords.push_back(BufferView<float2>());
    mpMesh->transform = mTransform;

    float maxAmplitude = 0.f;
    for (auto wave : mWaves) {
        maxAmplitude = std::max((float)maxAmplitude, wave.amplitude);
    }
    mpMesh->object_aabb.m_min = make_float3(-mLength / 2, -maxAmplitude, -mLength / 2);
    mpMesh->object_aabb.m_max = make_float3(mLength / 2, maxAmplitude, mLength / 2);
    mpMesh->world_aabb = mpMesh->object_aabb;
    mpMesh->world_aabb.transform(mTransform);
   
    BufferView<float3> bvPos;
    bvPos.data = reinterpret_cast<CUdeviceptr>(mMeshBuffer.pos);
    bvPos.byte_stride = sizeof(float3);
    bvPos.count = mSamplesX * mSamplesZ;
    bvPos.elmt_byte_size = sizeof(float3);
    mpMesh->positions.push_back(bvPos);

    BufferView<float3> bvNormal;
    bvNormal.data = reinterpret_cast<CUdeviceptr>(mMeshBuffer.normal);
    bvNormal.byte_stride = sizeof(float3);
    bvNormal.count = mSamplesX * mSamplesZ;
    bvNormal.elmt_byte_size = sizeof(float3);
    mpMesh->normals.push_back(bvNormal);

    BufferView<uint32_t> bvIndices;
    bvIndices.data = reinterpret_cast<CUdeviceptr>(mMeshBuffer.indices);
    bvIndices.byte_stride = sizeof(uint32_t);
    bvIndices.count = (mSamplesX - 1) * (mSamplesZ - 1) * 6;
    bvIndices.elmt_byte_size = sizeof(uint32_t);
    mpMesh->indices.push_back(bvIndices);
}

void WaveMesh::updateMesh(float t)
{
    cudaUpdateGridMesh(mMeshBuffer, reinterpret_cast<Wave*>(mdWaves),
        mWaves.size(), mSamplesX, mSamplesZ, mLength, t);
}

void WaveMesh::buildAccelerationStructure(OptixDeviceContext context)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = mSamplesX * mSamplesZ;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&mMeshBuffer.pos);
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangle_input.triangleArray.numIndexTriplets = (mSamplesX - 1) * (mSamplesZ - 1) * 2;
    triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(mMeshBuffer.indices);
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

    mpMesh->d_gas_output = mdGasOutputBuffer;
    mpMesh->gas_handle = mGasHandle;
}

void WaveMesh::updateAccelerationStructure(OptixDeviceContext context)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = mSamplesX * mSamplesZ;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&mMeshBuffer.pos);
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangle_input.triangleArray.numIndexTriplets = (mSamplesX - 1) * (mSamplesZ - 1) * 2;
    triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(mMeshBuffer.indices);
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