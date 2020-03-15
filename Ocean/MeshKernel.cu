#include "MeshKernel.h"

#include <sutil/vec_math.h>

#include <cuda/cuda_noise.cuh>

#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.hpp>


#define NOISE_STRENGTH 0.5

//Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b)
{
	return (a + (b - 1)) / b;
}

__device__ float fBM(int numOctaves, float3 coordinate, float persistence, float scale, float low, float high, int seed)
{
	float maxAmp = 0;
	float amp = 1;
	float freq = scale;
	float noise = 0;

	// add successively smaller, higher - frequency terms
	for (int i = 0; i < numOctaves; ++i) {
		noise += cudaNoise::simplexNoise(coordinate, freq, seed) * amp;
		maxAmp += amp;
		amp *= persistence;
		freq *= 2;
	}

	// take the average value of the iterations
	noise /= maxAmp;

	// normalize the result
	noise = noise * (high - low) / 2 + (high + low) / 2;

	return noise;
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

	return normalize(sum);
}

__forceinline__ __device__ float3 calculateGerstnerWavePosition(Wave* waves, int numWaves,
	float3 samplePoint, float t)
{
	float2 gridLocation = make_float2(samplePoint.x, samplePoint.z);

	float3 newPos = make_float3(gridLocation.x, 0.f, gridLocation.y) +
		calculateGerstnerWaveOffset(waves, numWaves, gridLocation, t);

	float noise = fBM(1, newPos, 0.5, 0.03, -NOISE_STRENGTH, NOISE_STRENGTH, 9);
	newPos.y += noise;

	return newPos;
}

__global__ void generateGridMesh(MeshBuffer meshBuffer, Wave* waves, int numWaves,
	ProjectedGrid projectedGrid, float t)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int numSamplesX = projectedGrid.samplesU;
	int numSamplesY = projectedGrid.samplesV;
	int X = numSamplesX - 1;
	int Y = numSamplesY - 1;

	if (tx > X || ty > Y) return;

	float3 samplePoint;
	unsigned int indexVertex = tx * numSamplesY + ty;
/*
	if (!projectedGrid.intersectXZGrid(tx, ty, &samplePoint)) {
		meshBuffer.validityMask[indexVertex] = false;
		meshBuffer.pos[indexVertex] = samplePoint;
		return;
	}*/
	samplePoint = make_float3((tx - X / 2.0f) * 2000 / X, 0, (ty - Y / 2.0f) * 2000 / Y);

	meshBuffer.pos[indexVertex] = calculateGerstnerWavePosition(waves, numWaves, samplePoint, t);
	meshBuffer.validityMask[indexVertex] = true;

	if (tx < X && ty < Y) {
		int indexIndices = 6 * (tx * X + ty);
		meshBuffer.indices[indexIndices] = indexVertex;
		meshBuffer.indices[indexIndices + 1] = indexVertex + numSamplesY;
		meshBuffer.indices[indexIndices + 2] = indexVertex + numSamplesY + 1;
		meshBuffer.indices[indexIndices + 3] = indexVertex;
		meshBuffer.indices[indexIndices + 4] = indexVertex + numSamplesY + 1;
		meshBuffer.indices[indexIndices + 5] = indexVertex + 1;
	}
}

__global__ void calculateNormalDuDv(MeshBuffer meshBuffer, ProjectedGrid projectedGrid)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int numSamplesX = projectedGrid.samplesU;
	int numSamplesY = projectedGrid.samplesV;
	int X = numSamplesX - 1;
	int Y = numSamplesY - 1;
	
	if (tx > X || ty > Y) return;

	unsigned int indexVertex = tx * numSamplesY + ty;
	if (!meshBuffer.validityMask[indexVertex]) {
		meshBuffer.normal[indexVertex] = make_float3(0.f, 1.f, 0.f);
		return;
	}

	float2 slope= make_float2(0.f);
	float2 diff = make_float2(1.f);
	if (tx > 0 && ty > 0 && tx < X && ty < Y)
	{
		int ixp1 = (tx + 1) * numSamplesY + ty;
		int ixm1 = (tx - 1) * numSamplesY + ty;
		int iyp1 = tx * numSamplesY + ty + 1;
		int iym1 = tx * numSamplesY + ty - 1;

		float3 xp1 = meshBuffer.pos[ixp1];
		float3 xm1 = meshBuffer.pos[ixm1];
		float3 yp1 = meshBuffer.pos[iyp1];
		float3 ym1 = meshBuffer.pos[iym1];

		slope.x = xp1.y - xm1.y;
		slope.y = yp1.y - ym1.y;

		diff.x = xp1.x - xm1.x;
		diff.y = yp1.z - ym1.z;
	}

	float3 normal = normalize(cross(make_float3(0.0f, slope.y, diff.y),
		make_float3(diff.x, slope.x, 0.0f)));

	meshBuffer.normal[indexVertex] = normal;
}

void cudaGenerateGridMesh(MeshBuffer& meshBuffer, Wave* waves, int numWaves,
	ProjectedGrid projectedGrid, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(projectedGrid.samplesU, block.x),
		cuda_iDivUp(projectedGrid.samplesV, block.y), 1);

	generateGridMesh << <grid, block, 0, 0 >> > (meshBuffer, waves, numWaves,
		projectedGrid, t);

	calculateNormalDuDv << <grid, block, 0, 0 >> > (meshBuffer, projectedGrid);
}

__global__ void updateGridMesh(MeshBuffer meshBuffer, Wave* waves, int numWaves,
	ProjectedGrid projectedGrid, float t)
{
	unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int numSamplesX = projectedGrid.samplesU;
	int numSamplesY = projectedGrid.samplesV;
	int X = numSamplesX - 1;
	int Y = numSamplesY - 1;

	if (tx > X || ty > Y) return;

	float3 samplePoint;
	unsigned int indexVertex = tx * numSamplesY + ty;
	if (!projectedGrid.intersectXZGrid(tx, ty, &samplePoint)) {
		meshBuffer.validityMask[indexVertex] = false;
		meshBuffer.pos[indexVertex] = samplePoint;
		return;
	}
	//samplePoint = make_float3((tx - X / 2.0f) * 2000 / X, 0, (ty - Y / 2.0f) * 2000 / Y);
	meshBuffer.pos[indexVertex] = calculateGerstnerWavePosition(waves, numWaves, samplePoint, t);
	meshBuffer.validityMask[indexVertex] = true;
}

void cudaUpdateGridMesh(MeshBuffer& meshBuffer, Wave* waves, int numWaves,
	ProjectedGrid projectedGrid, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(projectedGrid.samplesU, block.x),
		cuda_iDivUp(projectedGrid.samplesV, block.y), 1);

	updateGridMesh << <grid, block, 0, 0 >> > (meshBuffer, waves, numWaves,
		projectedGrid, t);
	calculateNormalDuDv << <grid, block, 0, 0 >> > (meshBuffer, projectedGrid);
}
