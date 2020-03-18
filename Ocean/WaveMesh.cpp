#include "WaveMesh.h"
#include "ProjectedGrid.h"

#include <sutil/Matrix.h>
#include <sutil/MeshGroup.h>
#include <sutil/Exception.h>
#include <sutil/vec_math.h>

#include <glm/mat4x4.hpp>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <algorithm>


WaveMesh::WaveMesh(int windowWidth, int windowHeight, int samplesPerPixel) :
    mWindowWidth(windowWidth),
    mWindowHeight(windowHeight),
    mSamplesPerPixel(samplesPerPixel)
{
    mTransform = sutil::Matrix4x4::identity();
    mpMesh = std::make_shared<sutil::MeshGroup>();

    mpProjectedGrid = std::make_shared<ProjectedGrid>();
    mpProjectedGrid->samplesU = windowWidth * samplesPerPixel;
    mpProjectedGrid->samplesV = windowHeight * samplesPerPixel;
}

WaveMesh::~WaveMesh()
{
    CUDA_CHECK(cudaFree((void*)mdGasOutputBuffer));
    CUDA_CHECK(cudaFree((void*)mdTempBufferGas));
    CUDA_CHECK(cudaFree((void*)mdWaves));

    CUDA_CHECK(cudaFree((void*)mMeshBuffer.pos));
    CUDA_CHECK(cudaFree((void*)mMeshBuffer.normal));
    CUDA_CHECK(cudaFree((void*)mMeshBuffer.indices));
}

void WaveMesh::generateMesh(float t)
{
    mpProjectedGrid->init();
    mNumVerts = mpProjectedGrid->samplesU * mSamplesPerPixel * mpProjectedGrid->samplesV * mSamplesPerPixel;
    mNumTriangles = (mpProjectedGrid->samplesU * mSamplesPerPixel - 1) * (mpProjectedGrid->samplesV * mSamplesPerPixel - 1) * 2;
    size_t verticesByteSize = mNumVerts * sizeof(float3);
    size_t indicesByteSize = 4 * mNumTriangles * 3;
    size_t waveByteSize = mWaves.size() * sizeof(Wave);

    float3* dVertices, * dNormals;
    bool* dValidityMask;
    uint32_t* dIndices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dVertices), verticesByteSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dNormals), verticesByteSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dIndices), indicesByteSize));
    mMeshBuffer.pos = dVertices;
    mMeshBuffer.normal = dNormals;
    mMeshBuffer.indices = dIndices;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdWaves), waveByteSize));
    CUDA_CHECK(cudaMemcpy((void*)mdWaves, mWaves.data(), waveByteSize, cudaMemcpyHostToDevice));

    glm::mat4 projectorMatrix;
    bool isVisible = mpProjectedGrid->calculateRangeMatrix(projectorMatrix);

    if (isVisible) {
        mpProjectedGrid->calculateCorners(projectorMatrix);
        cudaGenerateGridMesh(mMeshBuffer, reinterpret_cast<Wave*>(mdWaves), mWaves.size(),
            *mpProjectedGrid, t);
        mpMesh->skipRendering = false;
    }
    else {
        mpMesh->skipRendering = true;
    }

    mpMesh->name = "Wave";
    mpMesh->material_idx.push_back(-1);
    mpMesh->texcoords.push_back(BufferView<float2>());
    mpMesh->transform = mTransform;

    float maxAmplitude = 0.f;
    for (auto wave : mWaves) {
        maxAmplitude = std::max((float)maxAmplitude, wave.amplitude);
    }
    mpMesh->object_aabb.m_min = make_float3(-2000, -maxAmplitude, -2000);
    mpMesh->object_aabb.m_max = make_float3(2000, maxAmplitude, 2000);
    mpMesh->world_aabb = mpMesh->object_aabb;
    mpMesh->world_aabb.transform(mTransform);
   
    BufferView<float3> bvPos;
    bvPos.data = reinterpret_cast<CUdeviceptr>(mMeshBuffer.pos);
    bvPos.byte_stride = sizeof(float3);
    bvPos.count = mNumVerts;
    bvPos.elmt_byte_size = sizeof(float3);
    mpMesh->positions.push_back(bvPos);

    BufferView<float3> bvNormal;
    bvNormal.data = reinterpret_cast<CUdeviceptr>(mMeshBuffer.normal);
    bvNormal.byte_stride = sizeof(float3);
    bvNormal.count = mNumVerts;
    bvNormal.elmt_byte_size = sizeof(float3);
    mpMesh->normals.push_back(bvNormal);

    BufferView<uint32_t> bvIndices;
    bvIndices.data = reinterpret_cast<CUdeviceptr>(mMeshBuffer.indices);
    bvIndices.byte_stride = sizeof(uint32_t);
    bvIndices.count = mNumTriangles * 3;
    bvIndices.elmt_byte_size = sizeof(uint32_t);
    mpMesh->indices.push_back(bvIndices);

}

void WaveMesh::updateMesh(float t)
{
    glm::mat4 projectorMatrix;
    bool isVisible = mpProjectedGrid->calculateRangeMatrix(projectorMatrix);

    if (isVisible) {
        mpProjectedGrid->calculateCorners(projectorMatrix);
        cudaUpdateGridMesh(mMeshBuffer, reinterpret_cast<Wave*>(mdWaves),
            mWaves.size(), *mpProjectedGrid, t);
        mpMesh->skipRendering = false;
    }
    else {
        mpMesh->skipRendering = true;
    }
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
    triangle_input.triangleArray.numVertices = mNumVerts;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&mMeshBuffer.pos);
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangle_input.triangleArray.numIndexTriplets = mNumTriangles;
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
    triangle_input.triangleArray.numVertices = mNumVerts;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&mMeshBuffer.pos);
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;
    triangle_input.triangleArray.numIndexTriplets = mNumTriangles;
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

void WaveMesh::setTransform(const sutil::Matrix4x4& transform)
{
    mTransform = transform;
    mpProjectedGrid->elevation = mTransform.getData()[7]; // Assume vertical translation
}

void WaveMesh::setCamera(sutil::Camera* pCamera)
{
    mpProjectedGrid->pRenderingCamera = pCamera;
}
