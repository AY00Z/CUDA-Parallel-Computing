/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 4: Somme de deux matrices
*
* File: student.hpp
*/

#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"

namespace IMAC
{
	// Configuration de lancement du kernel
	struct LaunchConfig
	{
		uint threadsX   = 16;  // threads par block en x
		uint threadsY   = 16;  // threads par block en y
		uint maxBlocks  = 0;   // 0 = grille complete (1 thread / element)
		                       // N>0 = on limite la grille a ~N blocks (boucles grid-stride)
	};

	// Temps mesures (ms) avec ChronoGPU
	struct GPUTimings
	{
		float alloc   = 0.f;
		float h2d     = 0.f;
		float kernel  = 0.f;
		float d2h     = 0.f;
		float release = 0.f;
		uint  gridX   = 0;     // grille effectivement lancee
		uint  gridY   = 0;
		float total() const { return alloc + h2d + kernel + d2h + release; }
	};

	// Kernel: out[y*width+x] = a[y*width+x] + b[y*width+x]
	__global__ void sumMatricesKernel(const uint width, const uint height,
	                                  const float *const dev_a,
	                                  const float *const dev_b,
	                                  float *const dev_out);

	// - a, b: matrices d'entree (width*height, row-major)
	// - out : matrice resultat (deja dimensionnee)
	GPUTimings studentJob(const std::vector<float> &a, const std::vector<float> &b,
	                      const uint width, const uint height,
	                      std::vector<float> &out,
	                      const LaunchConfig &cfg);
}

#endif
