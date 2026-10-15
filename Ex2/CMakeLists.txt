cmake_minimum_required(VERSION 3.20)  # 3.20+ suffit pour CUDA proprement
project(CudaAddVector LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Archis GPU (ajuste si besoin)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)   # ex: 61 75 86 89
endif()

# Type de build par défaut
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Flags SANS generator expressions ----
# C++ (hôte)
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
else()
  # Pas de -pedantic pour éviter qu'il se propage vers NVCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas")
endif()

# CUDA (device + host via NVCC)
# Important : pas de -lineinfo par défaut pour éviter le warning "style of line directive..."
# Tu peux l’activer en Debug/RelWithDebInfo plus tard si tu veux.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-O3")

# Linking device multi-fichiers
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Toolkit CUDA
find_package(CUDAToolkit REQUIRED)

# Sources
set(SRC_CPP
  src/chronoCPU.cpp
)
set(SRC_CUDA
  src/main.cu
  src/chronoGPU.cu
  src/student.cu
)

add_executable(addVector ${SRC_CPP} ${SRC_CUDA})

target_include_directories(addVector PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(addVector PRIVATE
  CUDA::cudart
)

set_target_properties(addVector PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Optionnel mais pratique pour IDE/clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
