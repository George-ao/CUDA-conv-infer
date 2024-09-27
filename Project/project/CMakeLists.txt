cmake_minimum_required (VERSION 3.5)

# set(CMAKE_BUILD_TYPE Debug) # Do not set if profiling using NSight
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_STANDARD 11)

project (ece408project)
include(FindCUDA)
list( APPEND CUDA_NVCC_FLAGS "; -arch=sm_86; -std=c++11; -lineinfo")

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_library(ece408net ece408net.cc ece408net.h)

cuda_add_executable(m1_cpu m1_cpu.cc)
target_link_libraries(m1_cpu ece408net MiniDNNLib)
# Set compiler flags for debug builds
target_compile_options(m1_cpu PRIVATE "$<$<CONFIG:DEBUG>:-gdwarf-5>")

cuda_add_executable(m1_gpu m1_gpu.cc)
target_link_libraries(m1_gpu ece408net MiniDNNLib)

