#pragma once


#include "Aabb.h"
#include "Matrix.h"

#include <cuda/BufferView.h>

#include <optix.h>

#include <vector>
#include <string>

namespace sutil
{
    struct MeshGroup
    {
        std::string                       name;
        Matrix4x4                         transform;

        std::vector<GenericBufferView>    indices;
        std::vector<BufferView<float3> >  positions;
        std::vector<BufferView<float3> >  normals;
        std::vector<BufferView<float2> >  texcoords;

        std::vector<int32_t>              material_idx;

        OptixTraversableHandle            gas_handle = 0;
        CUdeviceptr                       d_gas_output = 0;

        Aabb                              object_aabb;
        Aabb                              world_aabb;
    };
}