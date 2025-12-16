cmake_minimum_required(VERSION 3.20.0)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# 设置CUDA路径
if(NOT DEFINED ENV{CUDA_PATH})
    set(CUDA_PATH "/usr/local/cuda")
else()
    set(CUDA_PATH $ENV{CUDA_PATH})
endif()

# 包含CUDA头文件
include_directories(${CUDA_PATH}/include)
project(cuda_learning VERSION 0.1.0 LANGUAGES CUDA CXX C)
set(CMAKE_CUDA_ARCHITECTURES 75)
# 启用CUDA支持
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(CUDAToolkit)
add_subdirectory(1_cuda_reduce_study)
# add_subdirectory(2_cuda_sgemm_study)
# add_subdirectory(3_kernel_profiling_guide)
# add_subdirectory(4_tensor_core_wmma)
# add_subdirectory(5_mma_and_swizzle)
# add_subdirectory(6_cutlass_study)