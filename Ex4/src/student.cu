/*
 * TP 1 - Premiers pas en CUDA
 * --------------------------
 * Ex 4: Somme de deux matrices
 *
 * File: student.cu
 */

#include "student.hpp"
#include "chronoGPU.hpp"

#include <algorithm>
#include <cmath>

namespace IMAC
{
    // Chaque thread traite (x, y) puis saute de (stride_x, stride_y) :
    // le kernel reste correct meme si la grille est plus petite que la matrice.
    // x (colonne) est la dimension contigue en memoire -> boucle interne sur x,
    // donc les threads d'un warp (threadIdx.x consecutifs) lisent/ecrivent des
    // adresses consecutives (acces coalesces).
    __global__ void sumMatricesKernel(const uint width, const uint height,
                                      const float *const dev_a,
                                      const float *const dev_b,
                                      float *const dev_out)
    {
        const uint x0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y0 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint stride_x = gridDim.x * blockDim.x;
        const uint stride_y = gridDim.y * blockDim.y;

        for (uint y = y0; y < height; y += stride_y)
        {
            for (uint x = x0; x < width; x += stride_x)
            {
                const size_t idx = static_cast<size_t>(y) * width + x;
                dev_out[idx] = dev_a[idx] + dev_b[idx];
            }
        }
    }

    GPUTimings studentJob(const std::vector<float> &a, const std::vector<float> &b,
                          const uint width, const uint height,
                          std::vector<float> &out,
                          const LaunchConfig &cfg)
    {
        GPUTimings t;
        ChronoGPU chr;

        const size_t nbElem = static_cast<size_t>(width) * height;
        const size_t bytes  = nbElem * sizeof(float);

        float *dev_a = NULL;
        float *dev_b = NULL;
        float *dev_out = NULL;

        // 1. Allocation
        chr.start();
        HANDLE_ERROR(cudaMalloc((void **)&dev_a, bytes));
        HANDLE_ERROR(cudaMalloc((void **)&dev_b, bytes));
        HANDLE_ERROR(cudaMalloc((void **)&dev_out, bytes));
        chr.stop();
        t.alloc = chr.elapsedTime();

        // 2. Transfert Host -> Device
        chr.start();
        HANDLE_ERROR(cudaMemcpy(dev_a, a.data(), bytes, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_b, b.data(), bytes, cudaMemcpyHostToDevice));
        chr.stop();
        t.h2d = chr.elapsedTime();

        // 3. Configuration de la grille 2D
        const dim3 dimBlock(cfg.threadsX, cfg.threadsY);
        uint gx = (width  + dimBlock.x - 1) / dimBlock.x;
        uint gy = (height + dimBlock.y - 1) / dimBlock.y;

        if (cfg.maxBlocks > 0)
        {
            // On limite le nombre total de blocks (~ maxBlocks) : grille carree
            const uint side = std::max<uint>(1, static_cast<uint>(std::sqrt(static_cast<double>(cfg.maxBlocks))));
            gx = std::min(gx, side);
            gy = std::min(gy, side);
        }
        gy = std::min<uint>(gy, 65535u); // limite de gridDim.y (les boucles compensent)
        const dim3 dimGrid(gx, gy);
        t.gridX = gx;
        t.gridY = gy;

        // 4. Kernel
        chr.start();
        sumMatricesKernel<<<dimGrid, dimBlock>>>(width, height, dev_a, dev_b, dev_out);
        HANDLE_ERROR(cudaGetLastError());
        chr.stop(); // stop() synchronise sur l'event de fin
        t.kernel = chr.elapsedTime();

        // 5. Transfert Device -> Host
        chr.start();
        HANDLE_ERROR(cudaMemcpy(out.data(), dev_out, bytes, cudaMemcpyDeviceToHost));
        chr.stop();
        t.d2h = chr.elapsedTime();

        // 6. Liberation
        chr.start();
        HANDLE_ERROR(cudaFree(dev_a));
        HANDLE_ERROR(cudaFree(dev_b));
        HANDLE_ERROR(cudaFree(dev_out));
        chr.stop();
        t.release = chr.elapsedTime();

        return t;
    }
}
