cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(Ocean LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/lib CACHE
    PATH "Directory where all the .lib files are dumped." FORCE)
	
SET(LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/lib CACHE
    PATH "Directory where all the .lib files are dumped." FORCE)
	
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE
    PATH "Directory where .exe and .dll files are dumped." FORCE)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )
include (CompilerInfo)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
set(OptiX_INSTALL_DIR "${PROJECT_SOURCE_DIR}/Externals/optix" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX REQUIRED)

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/bin/ptx/objects-Debug/sutil_optix_module\", \\
	\"${CMAKE_BINARY_DIR}/bin/ptx/objects-Debug/ocean_optix_module")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \".\", ")

if (WIN32)
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
else (WIN32)
  if ( USING_GNU_C AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif (WIN32)
  
set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")


configure_file(sampleConfig.h.in sampleConfig.h @ONLY)

add_subdirectory(Externals)
add_subdirectory(sutil)
add_subdirectory(support)
add_subdirectory(Ocean)