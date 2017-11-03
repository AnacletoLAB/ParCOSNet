#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif

#include "GPUStream.h"

GPUStream::GPUStream( int n ) : numThreads( n ) {

    streams = new cudaStream_t[numThreads];

    for (int i = 0; i < numThreads; i++)
        cudaStreamCreate(&streams[i]);
		//cudaStreamCreateWithFlags( &streams[i], cudaStreamNonBlocking	);
}

GPUStream::~GPUStream() {

    for (int i = 0; i < numThreads; i++)
        cudaStreamDestroy(streams[i]);

    delete[] streams;
}
