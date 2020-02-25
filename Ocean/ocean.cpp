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
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "ocean.h"

#include <vector>
#include <iomanip>
#include <iostream>
#include <string>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/GLDisplay.h>

#include <GLFW/glfw3.h>


bool              resize_dirty = false;
bool              camera_changed = true;
sutil::Camera cam;
sutil::Trackball  trackball;
int32_t           mouse_button = -1;
int32_t                 width = 768;
int32_t                 height = 768;

CUdeviceptr d_param;
Params   params = {};
OptixShaderBindingTable sbt = {};
OptixTraversableHandle gas_handle;
OptixPipeline pipeline = nullptr;

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

const size_t raygen_record_size = sizeof(RayGenSbtRecord);
CUdeviceptr  raygen_record;
RayGenSbtRecord rg_sbt;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
        camera_changed = true;
    }
}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    width = res_x;
    height = res_y;
    camera_changed = true;
    resize_dirty = true;
}


static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q ||
            key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }
}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


void configureCamera(sutil::Camera& cam, const uint32_t width, const uint32_t height)
{
    cam.setEye({ 0.0f, 0.0f, 3.0f });
    cam.setLookat({ 0.0f, 0.0f, 0.0f });
    cam.setUp({ 0.0f, 1.0f, 0.0f });
    cam.setFovY(45.0f);
    cam.setAspectRatio((float)width / (float)height);

    trackball.setCamera(&cam);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}


void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit(1);
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

void initLaunchParams() {
    params.image_width = width;
    params.image_height = height;
    params.origin_x = width / 2;
    params.origin_y = height / 2;
    params.handle = gas_handle;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
}

void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    // Update params on device
  //  if (camera_changed || resize_dirty)
    //    params.subframe_index = 0;

    //handleCameraUpdate(params);
    rg_sbt.data.cam_eye = cam.eye();
    cam.UVWFrame(rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w);
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));
}

void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.image = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_param),
        &params,
        sizeof(Params),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    OPTIX_CHECK(optixLaunch(
        pipeline,
        0,             // stream
        d_param,
        sizeof(Params),
        &sbt,
        width,  // launch width
        height, // launch height
        1       // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

void displaySubframe(
    sutil::CUDAOutputBuffer<uchar4>& output_buffer,
    sutil::GLDisplay& gl_display,
    GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}


int main(int argc, char* argv[])
{
    std::string outfile;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i < argc - 1)
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            sutil::parseDimensions(dims_arg.c_str(), width, height);
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        char log[2048]; // For error reporting from OptiX creation functions

        int N = 100;
        int Nplus1 = N + 1;

        struct Vertex {
            float3 pos;
            float3 normal;
        };

        std::vector<Vertex> vertices(Nplus1 * Nplus1);
        std::vector<uint32_t> indices;

        int index;
        int length = 1;
        {
            float t = 0.f, f = 10.f, amplitude = 0.1, lamda = 0.5f, x0, y0 = 0.f;
            float w = 2 * M_PIf * f, k = 2 * M_PIf / lamda;
            for (int m_prime = 0; m_prime < Nplus1; m_prime++) {
                for (int n_prime = 0; n_prime < Nplus1; n_prime++) {
                    index = m_prime * Nplus1 + n_prime;

                    x0 = (n_prime - N / 2.0f) * length / N;

                    vertices[index].pos.x = x0 + amplitude * sin(w * t - k * x0);
                    vertices[index].pos.y = y0 + amplitude * cos(w * t - k * x0);
                    vertices[index].pos.z = (m_prime - N / 2.0f) * length / N;

                    vertices[index].normal.x = 0.0f;
                    vertices[index].normal.y = 1.0f;
                    vertices[index].normal.z = 0.0f;
                }
            }

            for (int m_prime = 0; m_prime < N; m_prime++) {
                for (int n_prime = 0; n_prime < N; n_prime++) {
                    index = m_prime * Nplus1 + n_prime;

                    indices.push_back(index);               // two triangles
                    indices.push_back(index + Nplus1);
                    indices.push_back(index + Nplus1 + 1);
                    indices.push_back(index);
                    indices.push_back(index + Nplus1 + 1);
                    indices.push_back(index + 1);
                }
            }
        }
        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(0));

            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        }


        //
        // accel handling
        //
        //OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

            const size_t vertices_size = sizeof(Vertex) * vertices.size();
            CUdeviceptr d_vertices = 0;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_vertices),
                vertices.data(),
                vertices_size,
                cudaMemcpyHostToDevice
            ));

            const size_t indices_size = 4 * indices.size();
            CUdeviceptr d_indices = 0;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_indices),
                indices.data(),
                indices_size,
                cudaMemcpyHostToDevice
            ));

            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            OptixBuildInput triangle_input = {};
            triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
            triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
            triangle_input.triangleArray.vertexBuffers = &d_vertices;
            triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes = 12;
            triangle_input.triangleArray.numIndexTriplets = indices.size() / 3;
            triangle_input.triangleArray.indexBuffer = d_indices;
            triangle_input.triangleArray.flags = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input,
                1,  // Number of build input
                &gas_buffer_sizes));
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

            // non-compacted output
            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
                &d_buffer_temp_output_gas_and_compacted_size),
                compactedSizeOffset + 8
            ));

            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

            OPTIX_CHECK(optixAccelBuild(
                context,
                0,              // CUDA stream
                &accel_options,
                &triangle_input,
                1,              // num build inputs
                d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_compacted_size,
                gas_buffer_sizes.outputSizeInBytes,
                &gas_handle,
                &emitProperty,  // emitted property list
                1               // num emitted properties
            ));

            CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
            CUDA_CHECK(cudaFree((void*)d_vertices));

            size_t compacted_gas_size;
            CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

            if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
            {
                CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

                // use handle as input and output
                OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

                CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
            }
            else
            {
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues = 3;
            pipeline_compile_options.numAttributeValues = 3;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            const std::string ptx = sutil::getPtxString(NULL, "ocean");
            size_t sizeof_log = sizeof(log);

            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                context,
                &module_compile_options,
                &pipeline_compile_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &sizeof_log,
                &module
            ));
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc = {}; //
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &raygen_prog_group
            ));

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &miss_prog_group
            ));

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                context,
                &hitgroup_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                log,
                &sizeof_log,
                &hitgroup_prog_group
            ));
        }

        //
        // Link pipeline
        //
        //OptixPipeline pipeline = nullptr;
        {
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = 5;
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            pipeline_link_options.overrideUsesMotionBlur = false;
            size_t sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                log,
                &sizeof_log,
                &pipeline
            ));
        }

        //
        // Set up shader binding table
        //
        {
            //CUdeviceptr  raygen_record;
            //const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        
            configureCamera(cam, width, height);
            //RayGenSbtRecord rg_sbt;
            rg_sbt.data = {};
            rg_sbt.data.cam_eye = cam.eye();
            cam.UVWFrame(rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w);
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(miss_record),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
            HitGroupSbtRecord hg_sbt;
            hg_sbt.data = { 1.5f };
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(hitgroup_record),
                &hg_sbt,
                hitgroup_record_size,
                cudaMemcpyHostToDevice
            ));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
            sbt.hitgroupRecordBase = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
            sbt.hitgroupRecordCount = 1;
        }

        initLaunchParams();

        GLFWwindow* window = sutil::initUI("ocean", width, height);
        glfwSwapBuffers(window);
        sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);

        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetWindowSizeCallback(window, windowSizeCallback);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetWindowUserPointer(window, &params);

        {
            sutil::GLDisplay gl_display;

            std::chrono::duration<double> state_update_time(0.0);
            std::chrono::duration<double> render_time(0.0);
            std::chrono::duration<double> display_time(0.0);

            do
            {
                auto t0 = std::chrono::steady_clock::now();
                glfwPollEvents();

                updateState(output_buffer, params);
                auto t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;

                launchSubframe(output_buffer);
                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe(output_buffer, gl_display, window);
                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

                sutil::displayStats(state_update_time, render_time, display_time);

                glfwSwapBuffers(window);

            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
        }

        sutil::cleanupUI(window);
        //
        // Cleanup
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));

            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }
    }
    catch (std::exception & e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
