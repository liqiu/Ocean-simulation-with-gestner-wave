set(target_name "Ocean")

file(GLOB_RECURSE  SOURCE_FILES
    "*.h"
    "*.cpp"
    "*.vert"
    "*.frag"
    "*.cu"
)
list(FILTER SOURCE_FILES EXCLUDE REGEX "Ocean.cu$")

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})
					   
add_executable(
    ${target_name}
    ${SOURCE_FILES}
    )
	
set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ..
  ../support
  "${OptiX_INCLUDE}"
  ${CMAKE_CURRENT_BINARY_DIR}/..
  )
  
target_link_libraries( ${target_name}
  sutil_7_sdk
  
  )
  
# Optix Module
file(GLOB_RECURSE  CUDA_FILES "*ocean.cu")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -arch sm_50 -lineinfo --keep-device-functions")
set(optix_target "ocean_optix_module")
add_library(${optix_target} OBJECT ${CUDA_FILES})
set_property(TARGET ${optix_target} PROPERTY CUDA_PTX_COMPILATION ON)
install(TARGETS ${optix_target} OBJECTS DESTINATION ${PROJECT_BINARY_DIR}/bin/ptx )
add_dependencies(${target_name} ${optix_target})