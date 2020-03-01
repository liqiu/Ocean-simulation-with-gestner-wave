#include "mesh.h"

#include <sutil/Matrix.h>
#include <sutil/Exception.h>
#include <sutil/vec_math.h>

#include <optix_stubs.h>

#include <cuda_runtime.h>


void Mesh::generateMesh(float t)
{
    /*
    mVerticesSize = mSamplesX * mSamplesZ* sizeof(Vertex);// *vertices.size();
    mIndicesSize = 4 * (mSamplesX - 1) * (mSamplesZ - 1) * 6;// indices.size();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdVertices), mVerticesSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdIndices), mIndicesSize));

	std::shared_ptr<Wave> wave = mpWaves[0];
	cudaGenerateGridMesh(reinterpret_cast<Vertex*>(mdVertices), reinterpret_cast<unsigned int*>(mdIndices), 
		wave->amplitude, wave->waveLength, wave->frequency, mSamplesX, mLength, t);*/

    mVerticesSize = 3 * sizeof(Vertex);
    mIndicesSize = 3 * 4;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdVertices), mVerticesSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdIndices), mIndicesSize));

    v[0].pos.x = 0.f;
    v[0].pos.y = 0.f;
    v[0].pos.z = 0.f;

    v[1].pos.x = 1.f;
    v[1].pos.y = 0.f;
    v[1].pos.z = 0.f;

    v[2].pos.x = 0.f;
    v[2].pos.y = 1.f;
    v[2].pos.z = 0.f;


    i[0] = 0;
    i[1] = 1;
    i[2] = 2;

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mdVertices), reinterpret_cast<void*>(v), mVerticesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mdIndices), reinterpret_cast<void*>(i), mIndicesSize, cudaMemcpyHostToDevice));

}

void Mesh::updateMesh(float t)
{/*
	std::shared_ptr<Wave> wave = mpWaves[0];
	cudaUpdateGridMesh(reinterpret_cast<Vertex*>(mdVertices), wave->amplitude, wave->waveLength,
		wave->frequency, mSamplesX, mLength, t);*/

    Vertex vv[3];
    sutil::Matrix4x4 rot = sutil::Matrix4x4::rotate(0.1f * t, make_float3(0, 0, 1));
    for (size_t i = 0; i < 3; i++)
    {
        vv[i].pos = make_float3(rot * make_float4(v[i].pos, 1));
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mdVertices), reinterpret_cast<void*>(vv), mVerticesSize, cudaMemcpyHostToDevice));
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
    triangle_input.triangleArray.numVertices = mVerticesSize;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.vertexBuffers = &mdVertices;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = 12;
    triangle_input.triangleArray.numIndexTriplets = mIndicesSize / 3;
    triangle_input.triangleArray.indexBuffer = mdIndices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input,
        1,  // Number of build input
        &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_output_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
        &d_buffer_output_gas),
        gas_buffer_sizes.outputSizeInBytes
    ));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,              // CUDA stream
        &accel_options,
        &triangle_input,
        1,              // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_output_gas,
        gas_buffer_sizes.outputSizeInBytes,
        &mGasHandle,
        nullptr,  // emitted property list
        0               // num emitted properties
    ));

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
    mdGasOutputBuffer = d_buffer_output_gas;
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
    triangle_input.triangleArray.numVertices = mVerticesSize;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.vertexBuffers = &mdVertices;
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = 12;
    triangle_input.triangleArray.numIndexTriplets = mIndicesSize / 3;
    triangle_input.triangleArray.indexBuffer = mdIndices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input,
        1,  // Number of build input
        &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempUpdateSizeInBytes));


    OPTIX_CHECK(optixAccelBuild(
        context,
        0,              // CUDA stream
        &accel_options,
        &triangle_input,
        1,              // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempUpdateSizeInBytes,
        mdGasOutputBuffer,
        gas_buffer_sizes.outputSizeInBytes,
        &mGasHandle,
        nullptr,  // emitted property list
        0             // num emitted properties
    ));

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
}
