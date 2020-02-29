#pragma once


#include <vector_types.h>


struct Vertex {
    float3 pos;
    float3 normal;
};

void cudaGenerateGridMesh(Vertex* vertices, unsigned int* indices,
    float amplitude, float lamda, float frequency, int Nplus1, float length, float t);

void cudaUpdateGridMesh(Vertex* vertices, float amplitude, float lamda,
    float frequency, int Nplus1, float length, float t);
