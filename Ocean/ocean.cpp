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
#include "Wave.h"
#include "WaveMesh.h"
#include "hdr_loader.h"

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/whitted.h>
#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Scene.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <imgui/imgui.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

bool              resize_dirty = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

int32_t           samples_per_launch = 16;

whitted::LaunchParams* d_params = nullptr;
whitted::LaunchParams   params = {};
int32_t                 width = 768;
int32_t                 height = 768;

cudaArray_t environmentTextureData = 0;


#define check_success(expr) \
    do { \
        if(!(expr)) { \
            fprintf(stderr, "Error in file %s, line %u: \"%s\".\n", __FILE__, __LINE__, #expr); \
            exit(EXIT_FAILURE); \
        } \
    } while(false)

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    bool imgui_hover = ImGui::IsAnyWindowHovered();

    if (action == GLFW_PRESS && !imgui_hover)
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


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --model <model.gltf>        Specify model to render (required)\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}


void initLaunchParams(const sutil::Scene& scene) {
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        width* height * sizeof(float4)
    ));
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.subframe_index = 0u;

    const float loffset = scene.aabb().maxExtent();

    // TODO: add light support to sutil::Scene
    std::vector<Light::Point> lights(1);
    lights[0].color = { 1.0f, 1.0f, 0.8f };
    lights[0].intensity = 5.0f;
    lights[0].position = scene.aabb().center() + make_float3(loffset);
    lights[0].falloff = Light::Falloff::QUADRATIC;

    params.lights.count = static_cast<uint32_t>(lights.size());
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.lights.data),
        lights.size() * sizeof(Light::Point)
    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.lights.data),
        lights.data(),
        lights.size() * sizeof(Light::Point),
        cudaMemcpyHostToDevice
    ));

    params.miss_color = make_float3(0.1f);

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(whitted::LaunchParams)));

    params.handle = scene.traversableHandle();
}


void handleCameraUpdate(whitted::LaunchParams& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
    
    std::cerr
        << "Updating camera:\n"
        << "\tU: " << params.U.x << ", " << params.U.y << ", " << params.U.z << std::endl
        << "\tV: " << params.V.x << ", " << params.V.y << ", " << params.V.z << std::endl
        << "\tW: " << params.W.x << ", " << params.W.y << ", " << params.W.z << std::endl;
        
    std::cerr
        << "Eye:" << (int)params.eye.x << "\t" << (int)params.eye.y << "\t" << (int)params.eye.z
        << "\tLookAt:" << (int)camera.lookat().x << "\t" << (int)camera.lookat().y << "\t" << (int)camera.lookat().z << std::endl;
}


void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(width, height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        width* height * sizeof(float4)
    ));
}


void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, whitted::LaunchParams& params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.subframe_index = 0;

    handleCameraUpdate(params);
    handleResize(output_buffer);
}


void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, const sutil::Scene& scene)
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(whitted::LaunchParams),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    OPTIX_CHECK(optixLaunch(
        scene.pipeline(),
        0,             // stream
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(whitted::LaunchParams),
        scene.sbt(),
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


void initCameraState(const sutil::Scene& scene)
{
    camera = scene.camera();
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}


void cleanup()
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.lights.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}

bool createEnvironmentTexture(std::string& fileName)
{
    unsigned int rx, ry;
    float* pixels;
    if (!load_hdr_float4(&pixels, &rx, &ry, fileName.c_str())) {
        fprintf(stderr, "error loading environment map file %s\n", fileName.c_str());
        return false;
    }

    const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    check_success(cudaMallocArray(&environmentTextureData, &channel_desc, rx, ry) == cudaSuccess);

    check_success(cudaMemcpyToArray(
        environmentTextureData, 0, 0, pixels,
        rx * ry * sizeof(float4), cudaMemcpyHostToDevice) == cudaSuccess);

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = environmentTextureData;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    check_success(cudaCreateTextureObject(&params.environmentTexture, &res_desc, &tex_desc, NULL) == cudaSuccess);

    return true;
}

void renderImgui(std::chrono::duration<double>& state_update_time,
    std::chrono::duration<double>& render_time,
    std::chrono::duration<double>& display_time)
{
    sutil::beginFrameImGui();
    sutil::displayStats(state_update_time, render_time, display_time);
    // Atmosphere Parameters GUI
    ImGui::Begin("Atmosphere Parameters");
    ImGui::ColorEdit3("Sun Color", (float*)&params.sunColor);
    ImGui::SliderFloat("Sun Distance", (float*)&params.sunDistance, 100.f, 10000.f);
    ImGui::SliderFloat("Sun Azimuth", (float*)&params.sunAzimuth, 0.f, 360);
    ImGui::SliderFloat("Sun Elevation", (float*)&params.sunElevation, 0.f, 90);
    ImGui::End();
    
    ImGui::Begin("Water Parameters");
    ImGui::ColorEdit3("Water Color", (float*)&params.waterColor);
    ImGui::End();

    sutil::endFrameImGui();
}

float random()
{
    return ((float)rand() / (RAND_MAX));
}
//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;

    //
    // Parse command line options
    //
    std::string outfile;
    std::string envFile;
    //std::string infile = sutil::sampleDataFilePath("Island/island.gltf");
    std::string infile = sutil::sampleDataFilePath("map_gta5/scene.gltf");
    //std::string infile = sutil::sampleDataFilePath("WaterBottle/WaterBottle.gltf");

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--model")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            infile = argv[++i];
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else if (arg == "--env")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            envFile = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    if (infile.empty())
    {
        std::cerr << "--model argument required" << std::endl;
        printUsageAndExit(argv[0]);
    }


    try
    {
        sutil::Scene scene;
        sutil::loadScene(infile.c_str(), scene);
        scene.createContext();
        scene.buildMeshAccels();

        createEnvironmentTexture(envFile);
        
        std::shared_ptr<Wave> pWave = std::make_shared<Wave>();
        pWave->amplitude = 2.f;
        pWave->direction = 1.57f;
        pWave->waveLength = 5.f;
        pWave->speed = 5 * sqrt(pWave->waveLength);
        pWave->steepness = 0.9f;

        std::shared_ptr<Wave> pWave2 = std::make_shared<Wave>();
        pWave2->amplitude = 1.6f;
        pWave2->direction = 110.f;
        pWave2->waveLength = 30.f;
        pWave2->speed = 4 * sqrt(pWave2->waveLength);
        pWave2->steepness = 0.9f;

        std::shared_ptr<Wave> pWave3 = std::make_shared<Wave>();
        pWave3->amplitude = 3.f;
        pWave3->direction = 210.f;
        pWave3->waveLength = 50.f;
        pWave3->speed = 6 * sqrt(pWave3->waveLength);
        pWave3->steepness = 0.8f;

        std::shared_ptr<Wave> pWave4 = std::make_shared<Wave>();
        pWave4->amplitude = 2.f;
        pWave4->direction = 270.f;
        pWave4->waveLength = 25.f;
        pWave4->speed = 3.8 * sqrt(pWave4->waveLength);
        pWave4->steepness = 0.9f;

        OPTIX_CHECK(optixInit());

        std::shared_ptr<WaveMesh> pMesh = std::make_shared<WaveMesh>(1280, 1280);
        /*pMesh->addWave(pWave);
        pMesh->addWave(pWave2);
        pMesh->addWave(pWave3);
        pMesh->addWave(pWave4);*/

        std::srand(std::time(nullptr));
        int numWaves = 20;
        for (int i = 0; i < numWaves; i++)
        {
            std::shared_ptr<Wave> pWave = std::make_shared<Wave>();
            
            pWave->direction = random() * M_PI * 2;
            pWave->amplitude = 0.2f + powf(2.0f, random() * 2.0f) * 0.65f;
            pWave->waveLength = 1.0f + powf(2.f, 1.f + random()) * 15.f;
            pWave->speed = 7 * sqrt(pWave->waveLength);
            pWave->steepness = 0.8 + random() / 5;
            pMesh->addWave(pWave);
        }
         
        sutil::Matrix4x4 transform = sutil::Matrix4x4::translate(make_float3(0, -685, 0));
        pMesh->setTransform(transform);


        scene.addMesh(pMesh->getMesh());
        scene.calculateAABB();
        initCameraState(scene);

        pMesh->setCamera(&camera);
        pMesh->generateMesh(0.f);
        pMesh->buildAccelerationStructure(scene.context());

        scene.buildInstanceAccel();
        scene.createPTXModule();
        scene.createProgramGroups();
        scene.createPipeline();
        scene.createSBT();


        
        initLaunchParams(scene);


        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("Ocean", width, height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &params);

            ImGui::SetCurrentContext(sutil::getContextImGui());
            ImGuiIO& io = ImGui::GetIO(); (void)io;
            io.ConfigWindowsMoveFromTitleBarOnly = true;
            ImGui::StyleColorsDark();
            //
            // Render loop
            //
            auto t00 = std::chrono::steady_clock::now();
            float t;
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, width, height);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                float3 U, V, W;
                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    t = std::chrono::duration<float>(t0 - t00).count();
                    pMesh->updateMesh(t);
                    pMesh->updateAccelerationStructure(scene.context());
                    scene.updateInstanceAccel();

                    updateState(output_buffer, params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, scene);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    renderImgui(state_update_time, render_time, display_time);
                    
                    glfwSwapBuffers(window);

                    ++params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        {
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW(); // For GL context
                sutil::initGL();
            }

            sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, width, height);
            handleCameraUpdate(params);
            handleResize(output_buffer);
            launchSubframe(output_buffer, scene);

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::displayBufferFile(outfile.c_str(), buffer, false);

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        cleanup();

    }
    catch (std::exception & e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
