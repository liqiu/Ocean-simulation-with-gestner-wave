#include "MeshKernel.h"

#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <corecrt_math.h>
#include <vector_types.h>


//Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b)
{
	return (a + (b - 1)) / b;
}

__global__ void generateGridMesh(Vertex* vertices, unsigned int* indices,
	float amplitude, float w, float k, float length, float t)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int N = blockDim.x * gridDim.x - 1;
	int Nplus1 = N + 1;
	unsigned int index = x * Nplus1 + y;

	float x0 = (y - N / 2.0f) * length / N;
	float y0 = 0.f;

	vertices[index].pos.x = x0 + amplitude * sinf(w * t - k * x0);
	vertices[index].pos.y = y0 + amplitude * cosf(w * t - k * x0);
	vertices[index].pos.z = (x - N / 2.0f) * length / N;

	if (x < N && y < N) {
		int indexIndices = 6 * index;
		indices[indexIndices] = index;
		indices[indexIndices + 1] = index + Nplus1;
		indices[indexIndices + 2] = index + Nplus1 + 1;
		indices[indexIndices + 3] = index;
		indices[indexIndices + 4] = index + Nplus1 + 1;
		indices[indexIndices + 5] = index + 1;
	}
}

void cudaGenerateGridMesh(Vertex* vertices, unsigned int* indices,
	float amplitude, float lamda, float frequency, int Nplus1, float length, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(Nplus1, block.x), cuda_iDivUp(Nplus1, block.y), 1);

	float angularFreq = 2 * CUDART_PI_F * frequency;
	float k = 2 * CUDART_PI_F / lamda;
	generateGridMesh << <grid, block >> > (vertices, indices, amplitude, angularFreq, k, length, t);
}

__global__ void updateGridMesh(Vertex* vertices, float amplitude,
	float w, float k, float length, float t)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int N = blockDim.x * gridDim.x - 1;
	int Nplus1 = N + 1;
	unsigned int index = x * Nplus1 + y;

	float x0 = (y - N / 2.0f) * length / N;
	float y0 = 0.f;

	vertices[index].pos.x = x0 + amplitude * sinf(w * t - k * x0);
	vertices[index].pos.y = y0 + amplitude * cosf(w * t - k * x0);
}

void cudaUpdateGridMesh(Vertex* vertices, float amplitude, float lamda,
	float frequency, int Nplus1, float length, float t)
{
	dim3 block(16, 16, 1);
	dim3 grid(cuda_iDivUp(Nplus1, block.x), cuda_iDivUp(Nplus1, block.y), 1);

	float angularFreq = 2 * CUDART_PI_F * frequency;
	float k = 2 * CUDART_PI_F / lamda;
	updateGridMesh << <grid, block >> > (vertices, amplitude, angularFreq, k, length, t);
}
