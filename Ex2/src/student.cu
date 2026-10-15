
/*
 * TP 1 - Premiers pas en CUDA
 * --------------------------
 * Exercice 2 : Addition de vecteurs
 *
 * Fichier : student.cu
 * Auteur original : Maxime MARIA
 */

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
    __global__ void sumArraysCUDA(
        const int n,
        const int *const dev_a,
        const int *const dev_b,
        int *const dev_res)
    {
        // Questions 2, 3 et 4 : chaque thread calcule un seul élément.
        /*
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < n)
        {
            dev_res[i] = dev_a[i] + dev_b[i];
        }
        */

        // Question 5 : chaque thread peut calculer plusieurs éléments.

        // Calcul de l'indice global du thread.
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Nombre total de threads lancés dans la grille.
        int stride = blockDim.x * gridDim.x;

        // Chaque thread parcourt plusieurs éléments en avançant
        // de stride positions jusqu'à la fin du vecteur.
        for (; i < n; i += stride)
        {
            dev_res[i] = dev_a[i] + dev_b[i];
        }
    }

    void studentJob(
        const int size,
        const int *const a,
        const int *const b,
        int *const res)
    {
        ChronoGPU chrGPU;

        // Question 1 : déclaration des tableaux sur le GPU
        int *dev_a = NULL;
        int *dev_b = NULL;
        int *dev_res = NULL;

        // Taille en octets d'un tableau.
        const size_t bytes = size * sizeof(int);

        std::cout << "Allocating input (3 arrays): "
                  << ((3 * bytes) >> 20)
                  << " MB on Device" << std::endl;

        // Question 1 : allocation de la mémoire GPU
        chrGPU.start();

        cudaMalloc((void **)&dev_a, bytes);
        cudaMalloc((void **)&dev_b, bytes);
        cudaMalloc((void **)&dev_res, bytes);

        chrGPU.stop();

        std::cout << "-> Done : "
                  << chrGPU.elapsedTime()
                  << " ms" << std::endl
                  << std::endl;

        // Question 1 : transfert des tableaux du CPU vers le GPU
        chrGPU.start();

        cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

        chrGPU.stop();

        std::cout << "Host to Device: "
                  << chrGPU.elapsedTime()
                  << " ms" << std::endl;

        // Lancement du kernel CUDA
        chrGPU.start();

        // Question 1 : lancement avec un seul bloc de 256 threads.
        /*
        sumArraysCUDA<<<1, 256>>>(
            size, dev_a, dev_b, dev_res
        );
        */

        // Question 3 : lancement avec un seul bloc de 512 threads.
        /*
        sumArraysCUDA<<<1, 512>>>(
            size, dev_a, dev_b, dev_res
        );
        */

        // Question 4 : calcul automatique du nombre de blocs.
        /*
        const int threadsPerBlock = 256;

        const int blocksPerGrid =
            (size + threadsPerBlock - 1) / threadsPerBlock;

        sumArraysCUDA<<<blocksPerGrid, threadsPerBlock>>>(
            size, dev_a, dev_b, dev_res
        );
        */

        // Question 5 : limitation du nombre de blocs à 1024

        // Nombre de threads dans chaque bloc.
        const int threadsPerBlock = 256;

        // Calcul du nombre de blocs nécessaires.
        const int blocksNeeded =
            (size + threadsPerBlock - 1) / threadsPerBlock;

        // Limitation à 1024 blocs maximum
        const int blocksPerGrid =
            (blocksNeeded < 1024) ? blocksNeeded : 1024;

        // Lancement du kernel avec la nouvelle configuration
        sumArraysCUDA<<<blocksPerGrid, threadsPerBlock>>>(
            size, dev_a, dev_b, dev_res
        );

        // Attendre la fin des calculs sur le GPU
        cudaDeviceSynchronize();

        chrGPU.stop();

        std::cout << "Kernel execution: "
                  << chrGPU.elapsedTime()
                  << " ms" << std::endl;

        // Question 1 : transfert du résultat du GPU vers le CPU.
        chrGPU.start();

        cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);

        chrGPU.stop();

        std::cout << "Device to Host: "
                  << chrGPU.elapsedTime()
                  << " ms" << std::endl;

        // Question 1 : libération de la mémoire GPU
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_res);
    }
}
