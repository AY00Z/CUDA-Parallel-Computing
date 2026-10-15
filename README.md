# GPGPU – CUDA Practice 

Hands-on projects in **parallel programming on NVIDIA GPUs** with CUDA C++.

Course: GPU Architecture & High-Performance Computing – Master 2, Université Gustave Eiffel.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop |
| Compute Capability | 8.6 |
| Global memory | 6 GB (6143 MB) |
| Shared memory | 48 KB per block |
| OS | Windows |

## Skills

- CUDA C++ / C++17
- GPU parallel programming (GPGPU)
- Kernels `__global__`, threads, blocks, grids (1D & 2D)
- Grid-stride loops
- Device memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
- Memory coalescing
- Performance measurement (`cudaEvent`, CPU vs GPU)
- Error handling (`cudaGetLastError`)
- Image processing on GPU
- CMake, nvcc

## Projects

| Folder | Project |
|---|---|
| `Ex1` | GPU device properties |
| `Ex2` | Vector addition |
| `Ex3` | Sepia image filter |
| `Ex4` | Matrix addition |

## Build

```bash
cd Ex4
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

## Credits

Course material by Maxime Maria. Image I/O: [LodePNG](https://lodev.org/lodepng/).
