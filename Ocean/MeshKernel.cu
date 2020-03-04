#include "MeshKernel.h"

#include <sutil/vec_math.h>

#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <corecrt_math.h>
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
		sum.y += Qi * Ai * Di.y * cosRad;
		sum.z += Ai * sin(rad);
	}

	return sum;
}

__global__ void generateGridMesh(Vertex* vertices, unsigned int* indices, Wave* waves,
	int numWaves, int numSamplesX, int numSamplesY, float length, float t)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int X = numSamplesX - 1;
	int Y = numSamplesY - 1;
	if (tx > X || ty > Y) return;
	unsigned int indexVertex = tx * numSamplesY + ty;

	float x0 = (tx - X / 2.0f) * length / X;
	float y0 = (ty - Y / 2.0f) * length / Y;
	float2 gridLocation = make_float2(x0, y0);

	vertices[indexVertex].pos = make_float3(gridLocation, 0.f) + 
		calculateGerstnerWaveOffset(waves, numWaves, gridLocation, t);

	if (tx < X && ty < Y) {
		int indexIndices = 6 * (tx * X + ty);
		indices[indexIndices] = indexVertex;
		indices[indexIndices + 1] = indexVertex + numSamplesY;
		indices[indexIndices + 2] = indexVertex + numSamplesY + 1;
		indices[indexIndices + 3] = indexVertex;
		indices[indexIndices + 4] = indexVertex + numSamplesY + 1;
		indices[indexIndices + 5] = indexVertex + 1;
	}
}

void cudaGenerateGridMesh(Vertex* vertices, unsigned int* indices, Wave* waves,
	int numWaves, int numSamplesX, int numSamplesY, float length, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(numSamplesX, block.x), cuda_iDivUp(numSamplesY, block.y), 1);

	generateGridMesh << <grid, block, 0, 0 >> > (vertices, indices, waves, numWaves,
		numSamplesX, numSamplesY, length, t);
}

__global__ void updateGridMesh(Vertex* vertices, Wave* waves, int numWaves,
	int numSamplesX, int numSamplesY, float length, float t)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int X = numSamplesX - 1;
	int Y = numSamplesY - 1;
	if (tx > X || ty > Y) return;
	unsigned int indexVertex = tx * numSamplesY + ty;

	float x0 = (tx - X / 2.0f) * length / X;
	float y0 = (ty - Y / 2.0f) * length / Y;
	float2 gridLocation = make_float2(x0, y0);

	vertices[indexVertex].pos = make_float3(gridLocation, 0.f) +
		calculateGerstnerWaveOffset(waves, numWaves, gridLocation, t);
}

void cudaUpdateGridMesh(Vertex* vertices, Wave* waves, int numWaves,
	int numSamplesX, int numSamplesY, float length, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(numSamplesX, block.x), cuda_iDivUp(numSamplesY, block.y), 1);

	updateGridMesh << <grid, block, 0, 0 >> > (vertices, waves, numWaves,
		numSamplesX, numSamplesY, length, t);
}
