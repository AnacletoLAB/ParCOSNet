#include <iostream>
#include "GPUCudaCleaner.h"

CudaCleaner::CudaCleaner() {}

CudaCleaner::~CudaCleaner() {
	std::cout << "calling cudaDeviceReset()..." << std::endl;
	cudaDeviceReset();
	std::cout << "Done." << std::endl;
}
