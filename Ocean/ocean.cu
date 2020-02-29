#include <optix.h>

#include "ocean.h"

#include <sutil/vec_math.h>

#include <cuda/random.h>

extern "C" {
    __constant__ Params params;
}


static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    float3* prd
)
{
    uint32_t p0, p1, p2;
    p0 = float_as_int(prd->x);
    p1 = float_as_int(prd->y);
    p2 = float_as_int(prd->z);
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset
        0,                   // SBT stride
        0,                   // missSBTIndex
        p0, p1, p2);
    prd->x = int_as_float(p0);
    prd->y = int_as_float(p1);
    prd->z = int_as_float(p2);
}


static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(float_as_int(p.x));
    optixSetPayload_1(float_as_int(p.y));
    optixSetPayload_2(float_as_int(p.z));
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
        int_as_float(optixGetPayload_0()),
        int_as_float(optixGetPayload_1()),
        int_as_float(optixGetPayload_2())
    );
}


__forceinline__ __device__ uchar4 make_color(const float3& c)
{
    return make_uchar4(
        static_cast<uint8_t>(clamp(c.x, 0.0f, 1.0f) * 255.0f),
        static_cast<uint8_t>(clamp(c.y, 0.0f, 1.0f) * 255.0f),
        static_cast<uint8_t>(clamp(c.z, 0.0f, 1.0f) * 255.0f),
        255u
    );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const int    subframe_index = params.subframe_index;
    //
    // Generate camera ray
    //
    uint32_t seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);

    const float2 subpixel_jitter = subframe_index == 0 ?
        make_float2(0.0f, 0.0f) :
        make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

    const float2 d = 2.0f * make_float2(
        (static_cast<float>(launch_idx.x) + subpixel_jitter.x) / static_cast<float>(launch_dims.x),
        (static_cast<float>(launch_idx.y) + subpixel_jitter.y) / static_cast<float>(launch_dims.y)
    ) - 1.0f;
    const float3 direction = normalize(d.x * U + d.y * V + W);
    const float3 origin = eye;

    float3       payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
    trace(params.handle,
        origin,
        direction,
        0.00f,  // tmin
        1e16f,  // tmax
        &payload_rgb);

    params.image[launch_idx.y * launch_dims.x + launch_idx.x] = make_color(payload_rgb);
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    float3    payload = getPayload();
    setPayload(make_float3(rt_data->r, rt_data->g, rt_data->b));
}


extern "C" __global__ void __closesthit__ch()
{

    //const float2 barycentrics = optixGetTriangleBarycentrics();

    setPayload(make_float3(1.f,0.f, 1.0f));
}
