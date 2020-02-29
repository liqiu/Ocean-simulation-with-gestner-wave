#include "Ocean.h"
#include "Wave.h"
#include "Mesh.h"
#include "RTStateObject.h"

#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
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
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>


//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

bool              resize_dirty = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;

// Mouse state
int32_t           mouse_button = -1;

int32_t           samples_per_launch = 16;

Params* d_params = nullptr;
Params   params = {};
int32_t                 width = 768;
int32_t                 height = 768;

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


void initLaunchParams(std::shared_ptr<Mesh> mesh) {

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));

    params.subframe_index = 0u;
    params.handle = mesh->getTraversableHandle();
}


void handleCameraUpdate(Params& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
    /*
    std::cerr
        << "Updating camera:\n"
        << "\tU: " << params.U.x << ", " << params.U.y << ", " << params.U.z << std::endl
        << "\tV: " << params.V.x << ", " << params.V.y << ", " << params.V.z << std::endl
        << "\tW: " << params.W.x << ", " << params.W.y << ", " << params.W.z << std::endl;
        */

}


void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(width, height);
}


void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{

    handleCameraUpdate(params);
    handleResize(output_buffer);
}


void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, std::shared_ptr<RTStateObject> rtStateObject)
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.image = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(Params),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    OPTIX_CHECK(optixLaunch(
        rtStateObject->getPipeline(),
        0,             // stream
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(Params),
        rtStateObject->getSbt(),
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


void initCameraState()
{
    camera.setEye({ 0.0f, 0.0f, 3.0f });
    camera.setLookat({ 0.0f, 0.0f, 0.0f });
    camera.setUp({ 0.0f, 1.0f, 0.0f });
    camera.setFovY(45.0f);
    camera.setAspectRatio((float)width / (float)height);

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}


void cleanup()
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;

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
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }


    try
    {
        std::shared_ptr<RTStateObject> pRTStateObject = std::make_shared<RTStateObject>();
        pRTStateObject->initialize();

        std::shared_ptr<Wave> pWave = std::make_shared<Wave>();
        pWave->amplitude = 0.1f;
        pWave->frequency = 1.f;
        pWave->waveLength = 0.5f;
        pWave->phase = 0.f;

        std::shared_ptr<Mesh> pMesh = std::make_shared<Mesh>();
        pMesh->addWave(pWave);
        pMesh->generateMesh(0.f);
        pMesh->buidAccelerationStructure(pRTStateObject->getContext());

        initCameraState();
        initLaunchParams(pMesh);

        GLFWwindow* window = sutil::initUI("optixMeshViewer", width, height);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetWindowSizeCallback(window, windowSizeCallback);
        glfwSetKeyCallback(window, keyCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetWindowUserPointer(window, &params);

        //
        // Render loop
        //
        {
            sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, width, height);
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

                launchSubframe(output_buffer, pRTStateObject);
                t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                displaySubframe(output_buffer, gl_display, window);
                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

                sutil::displayStats(state_update_time, render_time, display_time);

                glfwSwapBuffers(window);

                ++params.subframe_index;

            } while (!glfwWindowShouldClose(window));
            CUDA_SYNC_CHECK();
        }

        sutil::cleanupUI(window);
        

        cleanup();

    }
    catch (std::exception & e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
