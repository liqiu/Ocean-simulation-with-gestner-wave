//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/random.h>
#include <cuda/whitted.h> 
#include <cuda/sun.cuh> 
#include <sutil/vec_math.h>

#include <stdint.h>

#include <stdio.h>


extern "C"
{
__constant__ whitted::LaunchParams params;
}


__device__ inline float degree_to_radians(
    float degree)
{

    return degree * M_PIf / 180.0f;

}


__device__ inline float3 degree_to_cartesian(
    float azimuth,
    float elevation)
{

    float az = clamp(azimuth, .0f, 360.0f);
    float el = clamp(elevation, -90.0f, 90.0f);

    az = degree_to_radians(az);
    el = degree_to_radians(90.0f - el);

    float x = sinf(el) * cosf(az);
    float y = cosf(el);
    float z = sinf(el) * sinf(az);

    return normalize(make_float3(x, y, z));
}

__device__ inline float3 sunPosition()
{
    float3 sun = params.sunDistance * degree_to_cartesian(params.sunAzimuth, params.sunElevation);
    return sun;
}

//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
// TODO: move into header so can be shared by path tracer and bespoke renderers
//
//------------------------------------------------------------------------------

__device__  bool refract(float3& r, const float3& i, const float3& n, const float ior)
{
    float3 nn = n;
    float negNdotV = dot(i, nn);
    float eta;

    if (negNdotV > 0.0f)
    {
        eta = ior;
        nn = -n;
        negNdotV = -negNdotV;
    }
    else
    {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

    if (k < 0.0f) {
        // Initialize this value, so that r always leaves this function initialized.
        r = make_float3(0.f);
        return false;
    }
    else {
        r = normalize(eta * i - (eta * negNdotV + sqrtf(k)) * nn);
        return true;
    }
}

__device__ float3 schlick( const float3 spec_color, const float V_dot_H )
{
    return spec_color + ( make_float3( 1.0f ) - spec_color ) * powf( 1.0f - V_dot_H, 5.0f );
}


__device__ float vis( const float N_dot_L, const float N_dot_V, const float alpha )
{
    const float alpha_sq = alpha*alpha;

    const float ggx0 = N_dot_L * sqrtf( N_dot_V*N_dot_V * ( 1.0f - alpha_sq ) + alpha_sq );
    const float ggx1 = N_dot_V * sqrtf( N_dot_L*N_dot_L * ( 1.0f - alpha_sq ) + alpha_sq );

    return 2.0f * N_dot_L * N_dot_V / (ggx0+ggx1);
}


__device__ float ggxNormal( const float N_dot_H, const float alpha )
{
    const float alpha_sq   = alpha*alpha;
    const float N_dot_H_sq = N_dot_H*N_dot_H;
    const float x          = N_dot_H_sq*( alpha_sq - 1.0f ) + 1.0f;
    return alpha_sq/( M_PIf*x*x );
}


__device__ float3 linearize( float3 c )
{
    return make_float3(
            powf( c.x, 2.2f ),
            powf( c.y, 2.2f ),
            powf( c.z, 2.2f )
            );
}

#define M_PI       3.14159265358979323846


__forceinline__ __device__ float3 evaluateEnv(cudaTextureObject_t tex, float3 rayDirection)
{
    float3 skyUp = make_float3(0.f, 1.f, 0.f);
    const float directionDotUp = dot(rayDirection, skyUp);
    float3 queryDir = rayDirection;
    if (directionDotUp < 0.0f)
    {
        float3 clampedDir = normalize(cross(rayDirection, skyUp));
        clampedDir = normalize(cross(skyUp, clampedDir));
        queryDir = clampedDir;
    }

    const float4 texval = tex2D<float4>(
        tex,
        atan2f(queryDir.z, queryDir.x) * (float)(0.5 / M_PI) + 0.5f,
        acosf(fmaxf(fminf(queryDir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI));

    return make_float3(texval);
}
//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        whitted::PayloadRadiance*   payload
        )
{
    uint32_t u0=0, u1=0, u2=0, u3=0;
    optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            whitted::RAY_TYPE_RADIANCE,        // SBT offset
            whitted::RAY_TYPE_COUNT,           // SBT stride
            whitted::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1, u2, u3 );

     payload->result.x = __int_as_float( u0 );
     payload->result.y = __int_as_float( u1 );
     payload->result.z = __int_as_float( u2 );
     payload->depth    = u3;
}

static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    uint32_t occluded = 0u;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        whitted::RAY_TYPE_OCCLUSION,      // SBT offset
        whitted::RAY_TYPE_COUNT,          // SBT stride
        whitted::RAY_TYPE_OCCLUSION,      // missSBTIndex
        occluded);
    return occluded;
}


static __forceinline__ __device__ float3 traceRefraction(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    uint32_t u0 = 0, u1 = 0, u2 = 0;
    float3 result;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            whitted::RAY_TYPE_REFRACTION,      // SBT offset
            whitted::RAY_TYPE_COUNT,          // SBT stride
            whitted::RAY_TYPE_REFRACTION,      // missSBTIndex
            u0, u1, u2 );
    
    result.x = __int_as_float(u0);
    result.y = __int_as_float(u1);
    result.z = __int_as_float(u2);

    return result;
}

__forceinline__ __device__ void setPayloadResult( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}


__forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<uint32_t>( occluded ) );
}


__forceinline__ __device__ uchar4 make_color( const float3&  c )
{
    const float gamma = 2.2f;
    return make_uchar4(
            static_cast<uint8_t>( powf( clamp( c.x, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<uint8_t>( powf( clamp( c.y, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<uint8_t>( powf( clamp( c.z, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            255u
            );
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx      = optixGetLaunchIndex();
    const uint3  launch_dims     = optixGetLaunchDimensions();
    const float3 eye             = params.eye;
    const float3 U               = params.U;
    const float3 V               = params.V;
    const float3 W               = params.W;
    const int    subframe_index  = params.subframe_index;

    //
    // Generate camera ray
    //
    uint32_t seed = tea<4>( launch_idx.y*launch_dims.x + launch_idx.x, subframe_index );

    const float2 subpixel_jitter = subframe_index == 0 ?
        make_float2( 0.0f, 0.0f ) :
        make_float2( rnd( seed )-0.5f, rnd( seed )-0.5f );

    const float2 d = 2.0f * make_float2(
            ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
            ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y )
            ) - 1.0f;
    const float3 ray_direction = normalize(d.x*U + d.y*V + W);
    const float3 ray_origin    = eye;

    //
    // Trace camera ray
    //
    whitted::PayloadRadiance payload;
    payload.result = make_float3( 0.0f );
    payload.importance = 1.0f;
    payload.depth = 0.0f;

    traceRadiance(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload );

    //
    // Update results
    // TODO: timview mode
    //
    const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
    float3         accum_color  = payload.result;

    /*
    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }*/
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}


extern "C" __global__ void __miss__constant_radiance()
{
    setPayloadResult( evaluateEnv(params.environmentTexture, optixGetWorldRayDirection()) );
}

extern "C" __global__ void __miss__constant_refraction()
{
    setPayloadResult(params.waterColor);
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);
    float refraction_index = hit_group_data->material_data.pbr.indexOfRefraction;
    if (refraction_index > 1.f)
    {
        float3 I = optixGetWorldRayDirection();
        float3 T;
        float3 result = make_float3(0.f);
        float3 refraction_color = make_float3(0.f);

        // Refraction
        if (refract(T, I, geom.N, refraction_index))
        {
            const float tmin = 0.001f;
            const float tmax = 100.f;
            refraction_color = traceRefraction(params.handle, geom.P, T, tmin, tmax);
        }

        // Reflection
        float3 R = reflect(I, geom.N);
        float3 env_color = evaluateEnv(params.environmentTexture, R);

        float3 sunDir = normalize(sunPosition() - geom.P);

        float3 reflectionSun = params.sunColor * pow(max(dot(R, sunDir), 0.0), 64);
        float3 refectionColor = 0.5 * env_color + 1.5 * reflectionSun;

        //float3 reflectionSun = params.sunColor * pow(max(dot(R, sunDir), 0.0), 32);
        //float3 refectionColor = env_color;// +1.5 * reflectionSun;

        const float3 F = schlick(make_float3(0.02f), dot(R,  geom.N));
        result = refectionColor * F + (1.f -F) * refraction_color;
        setPayloadResult(result);
    }
    else 
    {
        float3 base_color = make_float3(hit_group_data->material_data.pbr.base_color);
        if (hit_group_data->material_data.pbr.base_color_tex)
            base_color *= linearize(make_float3(
                tex2D<float4>(hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y)
            ));

        float metallic = hit_group_data->material_data.pbr.metallic;
        float roughness = hit_group_data->material_data.pbr.roughness;
        float4 mr_tex = make_float4(1.0f);
        if (hit_group_data->material_data.pbr.metallic_roughness_tex)
            // MR tex is (occlusion, roughness, metallic )
            mr_tex = tex2D<float4>(hit_group_data->material_data.pbr.metallic_roughness_tex, geom.UV.x, geom.UV.y);
        roughness *= mr_tex.y;
        metallic *= mr_tex.z;

        const float  F0 = 0.04f;
        const float3 diff_color = base_color * (1.0f - F0) * (1.0f - metallic);
        const float3 spec_color = lerp(make_float3(F0), base_color, metallic);
        const float  alpha = roughness * roughness;

        float3 N = geom.N;
        /*if (hit_group_data->material_data.pbr.normal_tex)
        {
            const float4 NN = 2.0f * tex2D<float4>(hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y) - make_float4(1.0f);
            N = normalize(NN.x * normalize(geom.dpdu) + NN.y * normalize(geom.dpdv) + NN.z * geom.N);
        }*/

        float3 result = make_float3(0.0f);

        for (int i = 0; i < params.lights.count; ++i)
        {
            Light::Point light = params.lights[i];

            // TODO: optimize
            const float  L_dist = length(light.position - geom.P);
            const float3 L = (light.position - geom.P) / L_dist;
            const float3 V = -normalize(optixGetWorldRayDirection());
            const float3 H = normalize(L + V);
            const float  N_dot_L = dot(N, L);
            const float  N_dot_V = dot(N, V);
            const float  N_dot_H = dot(N, H);
            const float  V_dot_H = dot(V, H);

            if (N_dot_L > 0.0f && N_dot_V > 0.0f)
            {
                const float tmin = 0.001f;          // TODO
                const float tmax = L_dist - 0.001f; // TODO
                const bool  occluded = traceOcclusion(params.handle, geom.P, L, tmin, tmax);
                if (!occluded)
                {
                    const float3 F = schlick(spec_color, V_dot_H);
                    const float  G_vis = vis(N_dot_L, N_dot_V, alpha);
                    const float  D = ggxNormal(N_dot_H, alpha);

                    const float3 diff = (1.0f - F) * diff_color / M_PIf;
                    const float3 spec = F * G_vis * D;

                    result += light.color * light.intensity * N_dot_L * (diff + spec);
                }
            }
        }

        setPayloadResult(result);
    }
}

extern "C" __global__ void __closesthit__refraction()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry          geom = getLocalGeometry(hit_group_data->geometry_data);
    if (hit_group_data->material_data.pbr.indexOfRefraction > 1.f)
    {
        setPayloadResult(make_float3(0.95f, 0.95f, 0.95f));
    }
    else
    {
        float3 base_color = make_float3(hit_group_data->material_data.pbr.base_color);
        if (hit_group_data->material_data.pbr.base_color_tex)
            base_color *= linearize(make_float3(
                tex2D<float4>(hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y)
            ));

        float metallic = hit_group_data->material_data.pbr.metallic;
        float roughness = hit_group_data->material_data.pbr.roughness;
        float4 mr_tex = make_float4(1.0f);
        if (hit_group_data->material_data.pbr.metallic_roughness_tex)
            // MR tex is (occlusion, roughness, metallic )
            mr_tex = tex2D<float4>(hit_group_data->material_data.pbr.metallic_roughness_tex, geom.UV.x, geom.UV.y);
        roughness *= mr_tex.y;
        metallic *= mr_tex.z;


        const float  F0 = 0.04f;
        const float3 diff_color = base_color * (1.0f - F0) * (1.0f - metallic);
        const float3 spec_color = lerp(make_float3(F0), base_color, metallic);
        const float  alpha = roughness * roughness;

        float3 N = geom.N;
        //if (hit_group_data->material_data.pbr.normal_tex)
        //{
        //    const float4 NN = 2.0f * tex2D<float4>(hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y) - make_float4(1.0f);
        //    N = normalize(NN.x * normalize(geom.dpdu) + NN.y * normalize(geom.dpdv) + NN.z * geom.N);
        //}

        float3 result = make_float3(0.0f);

        for (int i = 0; i < params.lights.count; ++i)
        {
            Light::Point light = params.lights[i];

            // TODO: optimize
            const float  L_dist = length(light.position - geom.P);
            const float3 L = (light.position - geom.P) / L_dist;
            const float3 V = -normalize(optixGetWorldRayDirection());
            const float3 H = normalize(L + V);
            const float  N_dot_L = dot(N, L);
            const float  N_dot_V = dot(N, V);
            const float  N_dot_H = dot(N, H);
            const float  V_dot_H = dot(V, H);

            if (N_dot_L > 0.0f && N_dot_V > 0.0f)
            {
                const float tmin = 0.001f;          // TODO
                const float tmax = L_dist - 0.001f; // TODO
                const bool  occluded = false;// traceOcclusion(params.handle, geom.P, L, tmin, tmax);
                if (!occluded)
                {
                    const float3 F = schlick(spec_color, V_dot_H);
                    const float  G_vis = vis(N_dot_L, N_dot_V, alpha);
                    const float  D = ggxNormal(N_dot_H, alpha);

                    const float3 diff = (1.0f - F) * diff_color / M_PIf;
                    const float3 spec = F * G_vis * D;

                    result += light.color * light.intensity * N_dot_L * (diff + spec);
                }
            }
        }
        setPayloadResult(result);
    }
}