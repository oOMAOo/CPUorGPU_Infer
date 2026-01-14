
# 检测CUDA设备
enable_language(CUDA)
project(trt_plugins LANGUAGES CXX CUDA)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 设置库文件输出目录  
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})  # 静态库

add_library(${PROJECT_NAME} STATIC)

if(NOT DEFINED CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "CUDA_INCLUDE_DIR not set by parent project")
endif()

if(NOT DEFINED TensorRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT_INCLUDE_DIR not set by parent project")
endif()

function(add_plugin_source)
    target_sources(trt_plugins PRIVATE ${ARGN})
endfunction()
#插件列表
set(TRT_PLUGIN_NAMES
    addgridsamplePlugin
)

target_include_directories(trt_plugins PRIVATE 
    common 
    ${CUDA_INCLUDE_DIR} 
    ${TensorRT_INCLUDE_DIR}
)
target_link_directories(trt_plugins PRIVATE 
    ${CUDA_LIB_DIR} 
    ${TensorRT_LIB_DIR}
)
foreach(PLUGIN_NAME IN LISTS TRT_PLUGIN_NAMES)
    add_subdirectory(${PLUGIN_NAME})
endforeach()

target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer_10 cudart)
