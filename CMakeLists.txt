cmake_minimum_required(VERSION 3.25)

project(GPUorCPU_Infer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)



# 编译器设置 // 后续可以测试补充更多架构
set(CMAKE_CUDA_ARCHITECTURES "89")

# 检测CUDA设备
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(Use Cuda: ${CMAKE_CUDA_COMPILER})
    set(USE_CUDA ON)
    enable_language(CUDA)
else()
    #失败退出
    message(FATAL_ERROR Can't\ Find\ Cuda ...)
    set(USE_CUDA OFF)
endif() 


if(MSVC)
    # 通用编译选项
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>      # 正确的 __cplusplus 宏
        $<$<COMPILE_LANGUAGE:CXX>:/std:c++20>      # 正确的 __cplusplus 宏
        $<$<COMPILE_LANGUAGE:CXX>:/W3>                  # 警告级别4
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:inline>           # 移除未使用的函数
        $<$<COMPILE_LANGUAGE:CXX>:/MP>                  # 多处理器编译
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>               # 源代码和可执行文件使用UTF-8
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>                # C++异常处理
        $<$<COMPILE_LANGUAGE:CXX>:/GS>                  # 启用安全检查
        $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:column>  # 更好的错误信息（列信息）
    )
endif()


# 准备文件集合 （当前所有库仅有Windows 64位的）
set(IncludeDirs "")
set(LibDirs "")
set(BinDirs "")

# ============================= OpenCV =============================
set(OpenCV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dep/opencv)
set(OpenCV_INCLUDE_DIR ${OpenCV_ROOT}/include)
set(OpenCV_LIB_DIR ${OpenCV_ROOT}/lib)
set(OpenCV_BIN_DIR ${OpenCV_ROOT}/bin)
list(APPEND IncludeDirs ${OpenCV_INCLUDE_DIR})
# Only Windows 64
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" AND WIN32)
    list(APPEND LibDirs ${OpenCV_LIB_DIR})
    list(APPEND BinDirs ${OpenCV_BIN_DIR})
endif()



# =============== CUDA （需根据安装自行替换路径） ===============
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
set(CUDA_INCLUDE_DIR ${CUDA_ROOT}/include)
set(CUDA_LIB_DIR ${CUDA_ROOT}/lib/x64)
set(CUDA_BIN_DIR ${CUDA_ROOT}/bin/x64)
list(APPEND IncludeDirs ${CUDA_INCLUDE_DIR})
# Only Windows 64
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" AND WIN32)
    list(APPEND LibDirs ${CUDA_LIB_DIR})
    list(APPEND BinDirs ${CUDA_BIN_DIR})
endif()
# ========================== TensorRT ==========================
set(TensorRT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dep/tensorrt)
set(TensorRT_INCLUDE_DIR ${TensorRT_ROOT}/include)
set(TensorRT_LIB_DIR ${TensorRT_ROOT}/lib)
set(TensorRT_BIN_DIR ${TensorRT_ROOT}/bin)
list(APPEND IncludeDirs ${TensorRT_INCLUDE_DIR})
# Only Windows 64
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" AND WIN32)
    list(APPEND LibDirs ${TensorRT_LIB_DIR})
    list(APPEND BinDirs ${TensorRT_BIN_DIR})
endif()
 
# ============================= OpenVINO =============================
set(OpenVINO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dep/openvino)
set(OpenVINO_INCLUDE_DIR ${OpenVINO_ROOT}/include)
list(APPEND IncludeDirs ${OpenVINO_INCLUDE_DIR})
# Only Windows 64
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" AND WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OpenVINO_LIB_DIR ${OpenVINO_ROOT}/lib/intel64/Debug)
        set(OpenVINO_BIN_DIR ${OpenVINO_ROOT}/bin/intel64/Debug)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(OpenVINO_LIB_DIR ${OpenVINO_ROOT}/lib/intel64/Release)
        set(OpenVINO_BIN_DIR ${OpenVINO_ROOT}/bin/intel64/Release)
    else()
        message(Error: please choose Debug/Release for 'CMAKE_BUILD_TYPE')
    endif()
    list(APPEND LibDirs ${OpenVINO_LIB_DIR})
    list(APPEND BinDirs ${OpenVINO_BIN_DIR})
else()
    message(Error: support AMD64 only, current processor: ${CMAKE_SYSTEM_PROCESSOR})
endif()

# ============================= OnnxRuntime =============================
set(OnnxRuntime_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/dep/onnxruntime_gpu)
set(OnnxRuntime_INCLUDE_DIR ${OnnxRuntime_ROOT}/include)
set(OnnxRuntime_LIB_DIR ${OnnxRuntime_ROOT}/lib)
set(OnnxRuntime_BIN_DIR ${OnnxRuntime_ROOT}/bin)
list(APPEND IncludeDirs ${OnnxRuntime_INCLUDE_DIR})
# Only Windows 64
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" AND WIN32)
    list(APPEND LibDirs ${OnnxRuntime_LIB_DIR})
    list(APPEND BinDirs ${OnnxRuntime_BIN_DIR})
else()
    message(Error: support AMD64 only, current processor: ${CMAKE_SYSTEM_PROCESSOR})
endif()

# 查找文件
file(GLOB CORE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
file(GLOB KERNELS_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu")

message(STATUS "Found Core files: ${CORE_FILES}")
message(STATUS "Found KERNELS_FILES files: ${KERNELS_FILES}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IncludeDirs}
)
message("\n----------- INCLUDE -------------\n" ${IncludeDirs})
message("\n----------- LIBRARY ----------\n" ${LibDirs})
# 添加lib目录
link_directories(
    ${LibDirs}
)
# 添加可执行文件
add_executable(${PROJECT_NAME}
    src/main.cpp
    ${KERNELS_FILES}
    ${CORE_FILES}
    
)
foreach(libdir ${LibDirs})
    message(Add lib Dir:${libdir})
    file(GLOB libs "${libdir}/*.lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${libs})
endforeach()

foreach(bindir ${BinDirs})
    message(Add Bin Dir:${bindir})
    file(GLOB dlls "${bindir}/*")
    install(FILES ${dlls} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
endforeach()





