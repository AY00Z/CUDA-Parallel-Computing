/*
 * TP 1 - Premiers pas en CUDA
 * --------------------------
 * Ex 3: Filtre d'images sepia
 *
 * File: student.cu
 * Author: Maxime MARIA
 */

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
    __global__ void sepiaKernel(const uint width, const uint height,
                                const uchar *const dev_input,
                                uchar *const dev_output)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

    
        // Question 1 et 2 
     
        /*
        if (x >= width || y >= height)
            return;

        // 3 canaux RGB
        const size_t idx = (y * width + x) * 3;

        const float r = dev_input[idx + 0];
        const float g = dev_input[idx + 1];
        const float b = dev_input[idx + 2];

        dev_output[idx + 0] = (uchar)fminf(255.0f, r * 0.393f + g * 0.769f + b * 0.189f);
        dev_output[idx + 1] = (uchar)fminf(255.0f, r * 0.349f + g * 0.686f + b * 0.168f);
        dev_output[idx + 2] = (uchar)fminf(255.0f, r * 0.272f + g * 0.534f + b * 0.131f);
        */

        


        // Nouveau code Question 4 
    
        const int stride_x = gridDim.x * blockDim.x;
        const int stride_y = gridDim.y * blockDim.y;
 
		for (int xx = x; xx < width; xx += stride_x)
        
        {
            for (int yy = y; yy < height; yy += stride_y)
            {
               
                const size_t idx = (yy * width + xx) * 3;

                const float r = dev_input[idx + 0];
                const float g = dev_input[idx + 1];
                const float b = dev_input[idx + 2];

                dev_output[idx + 0] = (uchar)fminf(255.0f, r * 0.393f + g * 0.769f + b * 0.189f);
                dev_output[idx + 1] = (uchar)fminf(255.0f, r * 0.349f + g * 0.686f + b * 0.168f);
                dev_output[idx + 2] = (uchar)fminf(255.0f, r * 0.272f + g * 0.534f + b * 0.131f);
            }
        }
    }

    void studentJob(const std::vector<uchar> &input,
                    const uint width, const uint height,
                    std::vector<uchar> &output)
    {
        ChronoGPU chrGPU;
        chrGPU.start();

        uchar *dev_input = NULL;
        uchar *dev_output = NULL;

        // 3 canaux RGB
        const size_t size = width * height * 3 * sizeof(uchar);

        cudaMalloc(&dev_input, size);
        cudaMalloc(&dev_output, size);

        cudaMemcpy(dev_input, input.data(), size, cudaMemcpyHostToDevice);

        
        //Question 1 et 2 
        /*
        const dim3 dimBlock(16, 16);
        const dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                           (height + dimBlock.y - 1) / dimBlock.y);
        */

       



        // Noubeau code question 4 
    
        const dim3 dimBlock(16, 16);
        const dim3 dimGrid(8, 8);   // on se limite à 64 blocks 

        sepiaKernel<<<dimGrid, dimBlock>>>(width, height, dev_input, dev_output);
        cudaDeviceSynchronize();

        cudaMemcpy(output.data(), dev_output, size, cudaMemcpyDeviceToHost);

        cudaFree(dev_input);
        cudaFree(dev_output);

        chrGPU.stop();
        std::cout << " -> Done GPU : " << chrGPU.elapsedTime() << " ms" << std::endl
                  << std::endl;
    }
}