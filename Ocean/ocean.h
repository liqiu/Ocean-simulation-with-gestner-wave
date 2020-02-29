#pragma once 

#include <optix.h>

#include <vector_types.h>

#include <stdint.h>


struct Params
{
    uchar4* image;

    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;

    uint32_t                 subframe_index;

    OptixTraversableHandle handle;
};


struct RayGenData
{
};


struct MissData
{
    float r, g, b;
};


struct HitGroupData
{
    float radius;
};
