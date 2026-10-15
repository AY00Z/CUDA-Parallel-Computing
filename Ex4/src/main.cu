/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 4: Somme de deux matrices
*
* File: main.cu
*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <exception>

#include "student.hpp"
#include "chronoCPU.hpp"

namespace IMAC
{
	void printUsageAndExit(const char *prg)
	{
		std::cerr << "Usage: " << prg << " -l <L> -h <H> [options]" << std::endl
		          << "  -l  <L>    largeur de la matrice " << std::endl
		          << "  -h  <H>    hauteur de la matrice " << std::endl
		          << "  -tx <n>    threads par block en x (defaut 16)" << std::endl
		          << "  -ty <n>    threads par block en y (defaut 16)" << std::endl
		          << "  -mb <n>    limite le nombre de blocks (defaut 0)" << std::endl
		          << "  -bench     teste automatiquement plusieurs configurations" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Somme sequentielle sur CPU
	void sumMatricesCPU(const std::vector<float> &a, const std::vector<float> &b,
	                    const uint width, const uint height, std::vector<float> &out)
	{
		ChronoCPU chr;
		chr.start();
		for (uint y = 0; y < height; ++y)
		{
			for (uint x = 0; x < width; ++x)
			{
				const size_t id = static_cast<size_t>(y) * width + x;
				out[id] = a[id] + b[id];
			}
		}
		chr.stop();
		std::cout << " -> Done CPU : " << chr.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Compare deux matrices
	bool compare(const std::vector<float> &a, const std::vector<float> &b)
	{
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			return false;
		}
		for (size_t i = 0; i < a.size(); ++i)
		{
			if (std::fabs(a[i] - b[i]) > 1e-6f)
			{
				std::cout << "Error at index " << i << ": CPU = " << a[i] << " - GPU = " << b[i] << std::endl;
				return false;
			}
		}
		return true;
	}

	void printTimings(const GPUTimings &t)
	{
		std::cout << std::fixed << std::setprecision(3)
		          << "    grille lancee  : " << t.gridX << " x " << t.gridY << std::endl
		          << "    allocation     : " << t.alloc   << " ms" << std::endl
		          << "    transfert H->D : " << t.h2d     << " ms" << std::endl
		          << "    kernel         : " << t.kernel  << " ms" << std::endl
		          << "    transfert D->H : " << t.d2h     << " ms" << std::endl
		          << "    liberation     : " << t.release << " ms" << std::endl
		          << "    TOTAL GPU      : " << t.total() << " ms" << std::endl;
	}

	// Teste plusieurs configurations (threads/block, nb max de blocks)
	void runBench(const std::vector<float> &a, const std::vector<float> &b,
	              const uint width, const uint height,
	              const std::vector<float> &ref, std::vector<float> &outGPU,
	              const uint maxThreadsPerBlock)
	{
		struct Cfg { uint tx, ty, mb; };
		const Cfg cfgs[] = {
			{ 1,   1,  0 }, { 8,   8,  0 }, { 16, 16, 0 }, { 32, 32, 0 },
			{ 32,  8,  0 }, { 8,  32,  0 }, { 64, 16, 0 }, { 128, 1, 0 },
			{ 16, 16, 1 }, { 16, 16, 16 }, { 16, 16, 64 }, { 16, 16, 1024 },
			{ 32, 32, 64 }
		};

		std::cout << std::endl << "=== BENCH " << width << " x " << height << " ===" << std::endl
		          << std::setw(9) << "threads" << std::setw(8) << "maxBlk" << std::setw(12) << "grille"
		          << std::setw(13) << "kernel(ms)" << std::setw(12) << "total(ms)" << std::setw(6) << "OK?" << std::endl;

		for (const Cfg &c : cfgs)
		{
			if (c.tx * c.ty > maxThreadsPerBlock) continue;

			LaunchConfig cfg;
			cfg.threadsX = c.tx; cfg.threadsY = c.ty; cfg.maxBlocks = c.mb;
			std::fill(outGPU.begin(), outGPU.end(), 0.f);
			const GPUTimings t = studentJob(a, b, width, height, outGPU, cfg);
			const bool ok = (outGPU == ref);

			std::ostringstream th, gr;
			th << c.tx << "x" << c.ty;
			gr << t.gridX << "x" << t.gridY;
			std::cout << std::setw(9) << th.str() << std::setw(8) << c.mb << std::setw(12) << gr.str()
			          << std::fixed << std::setprecision(3)
			          << std::setw(13) << t.kernel << std::setw(12) << t.total()
			          << std::setw(6) << (ok ? "yes" : "NO") << std::endl;
		}
	}

	void main(int argc, char **argv)
	{
		uint width = 0, height = 0;
		bool bench = false;
		LaunchConfig cfg;

		if (argc == 1) printUsageAndExit(argv[0]);

		for (int i = 1; i < argc; ++i)
		{
			if      (!strcmp(argv[i], "-l")  && i + 1 < argc) width         = static_cast<uint>(atoi(argv[++i]));
			else if (!strcmp(argv[i], "-h")  && i + 1 < argc) height        = static_cast<uint>(atoi(argv[++i]));
			else if (!strcmp(argv[i], "-tx") && i + 1 < argc) cfg.threadsX  = static_cast<uint>(atoi(argv[++i]));
			else if (!strcmp(argv[i], "-ty") && i + 1 < argc) cfg.threadsY  = static_cast<uint>(atoi(argv[++i]));
			else if (!strcmp(argv[i], "-mb") && i + 1 < argc) cfg.maxBlocks = static_cast<uint>(atoi(argv[++i]));
			else if (!strcmp(argv[i], "-bench"))              bench = true;
			else printUsageAndExit(argv[0]);
		}

		if (width == 0 || height == 0 || cfg.threadsX == 0 || cfg.threadsY == 0)
			printUsageAndExit(argv[0]);

		// Verification des limites du device
		cudaDeviceProp prop;
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
		if (cfg.threadsX * cfg.threadsY > static_cast<uint>(prop.maxThreadsPerBlock))
		{
			std::cerr << "Erreur: " << cfg.threadsX << "x" << cfg.threadsY << " = " << cfg.threadsX * cfg.threadsY
			          << " threads/block > maxThreadsPerBlock (" << prop.maxThreadsPerBlock << ")" << std::endl;
			exit(EXIT_FAILURE);
		}

		std::cout << "Device : " << prop.name << std::endl
		          << "Matrices " << width << " x " << height << " (" << static_cast<size_t>(width) * height << " elements)" << std::endl;

		// Creation + initialisation aleatoire (graine fixe -> resultats reproductibles)
		const size_t n = static_cast<size_t>(width) * height;
		std::vector<float> a(n), b(n), outCPU(n), outGPU(n);

		std::mt19937 gen(42);
		std::uniform_real_distribution<float> dist(0.f, 100.f);
		for (size_t i = 0; i < n; ++i) { a[i] = dist(gen); b[i] = dist(gen); }

		// Calcul CPU
		std::cout << "Process on CPU (sequential)" << std::endl;
		sumMatricesCPU(a, b, width, height, outCPU);

		// Warm-up : initialise le contexte CUDA pour ne pas fausser la 1ere mesure
		HANDLE_ERROR(cudaFree(0));

		if (bench)
		{
			runBench(a, b, width, height, outCPU, outGPU, static_cast<uint>(prop.maxThreadsPerBlock));
			return;
		}

		std::cout << "============================================" << std::endl
		          << "              STUDENT'S JOB !               " << std::endl
		          << "============================================" << std::endl
		          << " threads/block : " << cfg.threadsX << " x " << cfg.threadsY
		          << ", maxBlocks : " << cfg.maxBlocks << std::endl;

		const GPUTimings t = studentJob(a, b, width, height, outGPU, cfg);
		printTimings(t);
		std::cout << "============================================" << std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(outCPU, outGPU))
			std::cout << " -> Well done!" << std::endl;
		else
			std::cout << " -> You failed, retry!" << std::endl;
	}
}

int main(int argc, char **argv)
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
