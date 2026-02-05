project(openvino_plugins LANGUAGES CXX)

# 设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 设置库文件输出目录  
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(${PROJECT_NAME} STATIC)

if(NOT DEFINED OpenVINO_INCLUDE_DIR)
    message(FATAL_ERROR "OpenVINO_INCLUDE_DIR not set by parent project")
endif()

function(add_plugin_source)
    target_sources(${PROJECT_NAME} PRIVATE ${ARGN})
endfunction()
#插件列表
set(OPENVINO_PLUGIN_NAMES
    addgridsamplePlugin
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    common
    ${OpenVINO_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${Eigen_ROOT}
    
)
target_link_directories(${PROJECT_NAME} PRIVATE 
    ${OpenVINO_LIB_DIR} 
    ${OpenCV_LIB_DIR}
)
foreach(PLUGIN_NAME IN LISTS OPENVINO_PLUGIN_NAMES)
    add_subdirectory(${PLUGIN_NAME})
endforeach()

target_link_libraries(${PROJECT_NAME} PRIVATE openvino opencv_world4120)
