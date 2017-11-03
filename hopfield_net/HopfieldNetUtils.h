#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <stdlib.h>
#include <map>
#include <string>
#include <iostream>
#include <fstream>

#define SIGN(x) (x < 0 ? LO : HI)
#define SIGNTH(x) (x < 0 ? negState : posState)
#define SIGNTHLAMBDA(x) (x < 0 ? nS : pS)

#define CHECK(call) \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
	    {                                                                      \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
	    }                                                                      \
}

inline void cudaCheck( cudaError_t cuSts ) {
	if (cuSts != cudaSuccess) {
		std::cout << "CUDA ERROR in file " << __FILE__ << " at line " << __LINE__ << std::endl;
		 abort();
	 }
}

#define cudaCheck2(cuSts,file,line) {\
	if (cuSts != cudaSuccess) {\
		std::cout << "Cuda error in file " << file << " at line " << line << std::endl;\
		std::cout << "CUDA Report: " << cudaGetErrorString( cuSts ) << std::endl;\
		abort();\
	}\
}

// draw a random float in [a, b)
__inline float randf(float a, float b) {
	float r = (float) ((float) rand() / (float) RAND_MAX);
	return r * (b - a) + a;
}

void saveGeneNames( std::map<int, std::string> * mappaGeniInv, int n, std::string filename );
