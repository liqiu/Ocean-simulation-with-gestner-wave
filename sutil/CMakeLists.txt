set(sources
    ${CMAKE_CURRENT_BINARY_DIR}/../sampleConfig.h
    ${SAMPLES_CUDA_DIR}/BufferView.h
    ${SAMPLES_CUDA_DIR}/GeometryData.h
    ${SAMPLES_CUDA_DIR}/Light.h
    ${SAMPLES_CUDA_DIR}/LocalGeometry.h
    ${SAMPLES_CUDA_DIR}/MaterialData.h
    ${SAMPLES_CUDA_DIR}/util.h
    Aabb.h
    Camera.cpp
    Camera.h
    CUDAOutputBuffer.h
    Exception.h
    GLDisplay.cpp
    GLDisplay.h
    Matrix.h
	MeshGroup.h
    Preprocessor.h
    Quaternion.h
    Record.h
    Scene.cpp
    Scene.h
    sutilapi.h
    sutil.cpp
    sutil.h
    Trackball.cpp
    Trackball.h
    vec_math.h
    )

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/..
	${CMAKE_CURRENT_BINARY_DIR}/..
	"${OptiX_INCLUDE}"
  ${CMAKE_CURRENT_SOURCE_DIR}/../Externals/glm
	 )


# Make the library.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -arch sm_75 -g -G -lineinfo --keep-device-functions")
set(sutil_target "sutil_7_sdk")
add_library(${sutil_target}  SHARED ${sources})
if( WIN32 )
  target_compile_definitions( ${sutil_target} PUBLIC GLAD_GLAPI_EXPORT )
endif()


target_link_libraries(${sutil_target}
  glad
  imgui
  cudart
  )
	
if(WIN32)
  target_link_libraries(${sutil_target} winmm.lib)
endif()

set(optix_target "sutil_optix_module")
add_library(${optix_target} OBJECT 
  ${SAMPLES_CUDA_DIR}/whitted.cu
  ${SAMPLES_CUDA_DIR}/cuda_noise.cuh
  )
set_property(TARGET ${optix_target} PROPERTY CUDA_PTX_COMPILATION ON)

install(TARGETS ${optix_target} OBJECTS DESTINATION ${PROJECT_BINARY_DIR}/bin/ptx )
add_dependencies(${sutil_target} ${optix_target})