set(target_name "Ocean")

file(GLOB_RECURSE  SOURCE_FILES
    "*.h"
    "*.cpp"
    "*.vert"
    "*.frag"
    "*.cu"
    "*.cuh"
)

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
  ../Externals/glm
  "${OptiX_INCLUDE}"
  ${CMAKE_CURRENT_BINARY_DIR}/..
  )
  
target_link_libraries( ${target_name}
  sutil_7_sdk
  
  )