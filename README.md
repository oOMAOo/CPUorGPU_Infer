# CPUorGPU_Infer
作用：使用TensorRt、OpenVino、OnnxRuntime进行模型推理
# 如何使用
## 1、依赖链接
visual studio 2022  环境: MSVC 14.44.35207

### CUDA V12.8
下载链接：
```
https://developer.nvidia.com/cuda-12-8-0-download-archive
```
### CUDNN 9.17 
下载链接：
```
https://developer.nvidia.com/cudnn-9-17-0-download-archive
```
### TensorRT 10.14
下载链接：
```
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.14.1/zip/TensorRT-10.14.1.48.Windows.win10.cuda-13.0.zip
```
### OpenVino 2025.3
下载链接：
```
https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.3/windows/openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64.zip
```
### OpenCV 4.12
下载链接：
```
https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip
```
### OnnxRuntime 1.23.2+
下载链接：
```
git clone https://github.com/microsoft/onnxruntime.git
cd .\onnxruntime\
# 路径需根据配置路径自行替换👇
.\build.bat --use_cuda --cudnn_home "C:/Program Files/NVIDIA/CUDNN/v9.17" --cuda_home "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" --use_tensorrt --tensorrt_home "D:/TensorRT_10.14" --config Release --use_openvino "CPU" --build_shared_lib --build_wheel --cmake_generator "Visual Studio 17 2022" 
```
## 2、Clone Demo
```
git clone https://github.com/oOMAOo/CPUorGPU_Infer.git
cd CPUorGPU_Infer
vim CMakeLists.txt
```
修改CMakeLists中的环境路径
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cmake --install . --config Release
.\Release\GPUorCPU_Infer.exe
```
## 示例
<img width="1212" height="699" alt="image" src="https://github.com/user-attachments/assets/d701d615-b20b-47f1-983e-5360b8510f34" />

