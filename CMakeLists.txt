CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ParCOSNet)
include_directories(.)
include_directories(COSNet)
include_directories(GPUutils)
include_directories(graph)
include_directories(graph_coloring)
include_directories(hopfield_net)
include_directories(utils)

INCLUDE(FindCUDA)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_60,code=sm_60;)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_52,code=sm_52;)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_35,code=sm_35;)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -O3; -Xcompiler -fPIC;)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -default-stream per-thread;)
## Debug!
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -O0; -G; -Xcompiler -rdynamic; -keep; -Xcompiler -fPIC; -Xcompiler -fopenmp;)

SET(CUDA_SEPARABLE_COMPILATION TRUE) # -rdc true
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NON_PROPAGATED_HOST_FLAGS ON)

FIND_PACKAGE(CUDA REQUIRED QUIET)



SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
## Debug!
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -g")

SET(SOURCESCPP      "COSNet/COSNet.cpp"
					"COSNet/COSNetCommon.cpp"
					"graph/graphCPU.cpp"
					"graph_coloring/colorer.cpp"
					"hopfield_net/HopfieldNetUtils.cpp"
					"hopfield_net/HopfieldNet.cpp"
					"utils/ArgHandle.cpp"
					"utils/fileImporter.cpp"
					"utils/fileExporter.cpp"
					"utils/timer.cpp")
SET(SOURCESCUDA     "GPUutils/GPUCudaCleaner.cu"
                	"GPUutils/GPUStream.cu"
                	"GPUutils/GPURandomizer.cu"
					"graph/graphGPU.cu"
                	"graph_coloring/coloringLuby.cu"
					"graph_coloring/coloringLubyFast.cu"
					"hopfield_net/HopfieldNetGPU.cu"
					"ParCOSNet.cu")
SET(SOURCEGEN		"datasetGenerator.cpp")

CUDA_ADD_EXECUTABLE(ParCOSNet ${SOURCESCUDA} ${SOURCESCPP})
add_executable(datasetGen ${SOURCEGEN})

target_link_libraries(ParCOSNet    ${CUDA_LIBRARIES} "/usr/local/cuda/lib64/libcudadevrt.a" ${CUDA_curand_LIBRARY})
