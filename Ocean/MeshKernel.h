#pragma once


#include "Wave.h"
#include "ProjectedGrid.h"

#include <vector_types.h>


struct MeshBuffer
{
    float3* pos;
    float3* normal;
    unsigned int* indices;
};

void cudaGenerateGridMesh(MeshBuffer& meshBuffer, Wave* waves, int numWaves,
    ProjectedGrid projectedGrid, float t);

void cudaUpdateGridMesh(MeshBuffer& meshBuffer, Wave* waves, int numWaves,
    ProjectedGrid projectedGrid, float t);
