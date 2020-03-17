#pragma once


#include <sutil/vec_math.h>

#include <vector_types.h>
#include <vector_functions.h>


struct ProjectedGrid
{
#ifdef __CUDACC__
	__forceinline__ __device__ float3 cameraRayDirection(int u, int v)
	{

		const float2 d = 2.0f * make_float2(
			(static_cast<float>(u)) / static_cast<float>(samplesU),
			(static_cast<float>(v)) / static_cast<float>(samplesV)
		) - 1.0f;

		return normalize(d.x * U + d.y * V + W);
	}

	__forceinline__ __device__ bool intersectXZGrid(int u, int v, float3* point)
	{
		float3 L = cameraRayDirection(u, v);
		float3 N = make_float3(0, 1, 0);

		float denom = dot(L, N);
		float3 intersection;
		bool flag = false;
		if (abs(denom) > 1e-6) {
			float t = (waveHeight - dot(N, eye)) / denom;
			if (t > 0) {
				intersection = eye + t * L;
				flag = true;
			}
		}
		
		float3 eyeProjected = make_float3(eye.x, waveHeight, eye.z);
		float distance = length(intersection - eyeProjected);
		if (distance > infinite) {
			flag = false;
		}

		if (!flag) {
			intersection = eyeProjected + make_float3(L.x * infinite, waveHeight, L.z * infinite);
		}

		*point = intersection;

		return flag;
	}
#endif

	int samplesU;
	int samplesV;

	float infinite;
	float waveHeight;

	float3                   eye;
	float3                   U;
	float3                   V;
	float3                   W;
};