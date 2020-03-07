#pragma once


#include "Wave.h"

#include <vector_types.h>


struct Vertex {
    float3 pos;
    float3 normal;
};

struct MeshBuffer
{
    float3* pos;
    float3* normal;
    unsigned int* indices;
};

void cudaGenerateGridMesh(MeshBuffer meshBuffer, Wave* waves, int numWaves,
    int numSamplesX, int numSamplesZ, float length, float t);

void cudaUpdateGridMesh(MeshBuffer meshBuffer, Wave* waves, int numWaves,
    int numSamplesX, int numSamplesZ, float length, float t);
