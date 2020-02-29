#include "mesh.h"

#include <sutil/Exception.h>
#include <sutil/vec_math.h>

#include <optix_stubs.h>

#include <cuda_runtime.h>


void Mesh::generateMesh(float t)
{
    mVerticesSize = mSamplesX * mSamplesZ* sizeof(Vertex);// *vertices.size();
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

void Mesh::buidAccelerationStructure(OptixDeviceContext context)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
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
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
        &d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
    ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,              // CUDA stream
        &accel_options,
        &triangle_input,
        1,              // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &mGasHandle,
        &emitProperty,  // emitted property list
        1               // num emitted properties
    ));

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mdGasOutputBuffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(context, 0, mGasHandle, mdGasOutputBuffer, compacted_gas_size, &mGasHandle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        mdGasOutputBuffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}
