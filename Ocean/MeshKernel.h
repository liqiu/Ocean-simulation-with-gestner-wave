#pragma once


#include "Wave.h"
#include <vector_types.h>


struct Vertex {
    float3 pos;
    float3 normal;
};

void cudaGenerateGridMesh(Vertex* vertices, unsigned int* indices, Wave* waves, int numWaves,
    int numSamplesX, int numSamplesY, float length, float t);

void cudaUpdateGridMesh(Vertex* vertices, Wave* waves, int numWaves,
    int numSamplesX, int numSamplesY, float length, float t);
