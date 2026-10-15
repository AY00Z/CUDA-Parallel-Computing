cmake_minimum_required(VERSION 3.20)
project(CudaMatrixSum LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Archi GPU (ajuste si besoin : 61 75 86 89 ...)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-O3")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

set(SRC_CPP src/chronoCPU.cpp)
set(SRC_CUDA src/main.cu src/chronoGPU.cu src/student.cu)

add_executable(matsum ${SRC_CPP} ${SRC_CUDA})
target_include_directories(matsum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(matsum PRIVATE CUDA::cudart)
set_target_properties(matsum PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
