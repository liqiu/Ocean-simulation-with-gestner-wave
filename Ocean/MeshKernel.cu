#include "MeshKernel.h"

#include <sutil/vec_math.h>

#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.hpp>


//Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b)
{
	return (a + (b - 1)) / b;
}

__forceinline__ __device__ float3 calculateGerstnerWaveOffset(Wave* waves, int numWaves,
	float2 gridLocation, float t)
{
	float3 sum = make_float3(0.f);

	float L, wi, phi, rad, Qi, Ai, cosRad;
	float2 Di;
	for (int i = 0; i < numWaves; i++)
	{
		Qi = waves[i].steepness;
		Ai = waves[i].amplitude;
		L = waves[i].waveLength;
		wi = 2 / L;
		Di = make_float2(cos(waves[i].direction), sin(waves[i].direction));
		phi = waves[i].speed * 2 / L;
		rad = wi * dot(Di, gridLocation) + phi * t;
		cosRad = cos(rad);

		sum.x += Qi * Ai * Di.x * cosRad;
		sum.y += Ai * sin(rad);
		sum.z += Qi * Ai * Di.y * cosRad;
	}

	return sum;
}


__forceinline__ __device__ float3 calculateGerstnerWaveNormal(Wave* waves, int numWaves,
	float2 posPlane, float t)
{
	float3 sum = make_float3(0.f, 1.f, 0.f);

	float L, wi, phi, rad, Qi, Ai, WA, cosRad, sinRad;
	float2 Di;
	for (int i = 0; i < numWaves; i++)
	{
		Qi = waves[i].steepness;
		Ai = waves[i].amplitude;
		L = waves[i].waveLength;
		wi = 2 / L;
		WA = wi * Ai;
		Di = make_float2(cos(waves[i].direction), sin(waves[i].direction));
		phi = waves[i].speed * 2 / L;
		rad = wi * dot(Di, posPlane) + phi * t;
		cosRad = cos(rad);
		sinRad = sin(rad);

		sum.x += -Di.x * WA * cosRad;
		sum.y += -Qi * WA * sinRad;
		sum.z += -Di.y * WA * cosRad;
	}

	return sum;
}

__global__ void generateGridMesh(MeshBuffer meshBuffer, Wave* waves,
	int numWaves, int numSamplesX, int numSamplesZ, float length, float t)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int X = numSamplesX - 1;
	int Z = numSamplesZ - 1;
	if (tx > X || ty > Z) return;
	unsigned int indexVertex = tx * numSamplesZ + ty;

	float x0 = (tx - X / 2.0f) * length / X;
	float z0 = (ty - Z / 2.0f) * length / Z;
	float2 gridLocation = make_float2(x0, z0);

	float3 newPos = make_float3(gridLocation.x, 0.f, gridLocation.y) +
		calculateGerstnerWaveOffset(waves, numWaves, gridLocation, t);
	meshBuffer.pos[indexVertex] = newPos;

	meshBuffer.normal[indexVertex] = calculateGerstnerWaveNormal(waves,
		numWaves, make_float2(newPos.x, newPos.z), t);

	if (tx < X && ty < Z) {
		int indexIndices = 6 * (tx * X + ty);
		meshBuffer.indices[indexIndices] = indexVertex;
		meshBuffer.indices[indexIndices + 1] = indexVertex + numSamplesZ;
		meshBuffer.indices[indexIndices + 2] = indexVertex + numSamplesZ + 1;
		meshBuffer.indices[indexIndices + 3] = indexVertex;
		meshBuffer.indices[indexIndices + 4] = indexVertex + numSamplesZ + 1;
		meshBuffer.indices[indexIndices + 5] = indexVertex + 1;
	}
}

void cudaGenerateGridMesh(MeshBuffer meshBuffer, Wave* waves, int numWaves,
	int numSamplesX, int numSamplesZ, float length, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(numSamplesX, block.x), cuda_iDivUp(numSamplesZ, block.y), 1);

	generateGridMesh << <grid, block, 0, 0 >> > (meshBuffer, waves, numWaves,
		numSamplesX, numSamplesZ, length, t);
}

__global__ void updateGridMesh(MeshBuffer meshBuffer, Wave* waves, int numWaves,
	int numSamplesX, int numSamplesZ, float length, float t)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int X = numSamplesX - 1;
	int Z = numSamplesZ - 1;
	if (tx > X || ty > Z) return;
	unsigned int indexVertex = tx * numSamplesZ + ty;

	float x0 = (tx - X / 2.0f) * length / X;
	float z0 = (ty - Z / 2.0f) * length / Z;
	float2 gridLocation = make_float2(x0, z0);

	float3 newPos = make_float3(gridLocation.x, 0.f, gridLocation.y) +
		calculateGerstnerWaveOffset(waves, numWaves, gridLocation, t);
	meshBuffer.pos[indexVertex] = newPos;

	meshBuffer.normal[indexVertex] = calculateGerstnerWaveNormal(waves,
		numWaves, make_float2(newPos.x, newPos.z), t);
}

void cudaUpdateGridMesh(MeshBuffer meshBuffer, Wave* waves, int numWaves,
	int numSamplesX, int numSamplesZ, float length, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(numSamplesX, block.x), cuda_iDivUp(numSamplesZ, block.y), 1);

	updateGridMesh << <grid, block, 0, 0 >> > (meshBuffer, waves, numWaves,
		numSamplesX, numSamplesZ, length, t);
}
