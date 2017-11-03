#include <stdio.h>
#include <type_traits>
#include "graph.h"
#include "../GPUutils/GPUutils.h"

using namespace std;

namespace Graph_k {
template<typename nodeW, typename edgeW> __global__ void print_d(GraphStruct<nodeW,edgeW>*,bool);
};


/**
 * Set the CUDA Unified Memory for nodes and edges
 * @param memType node or edge memory type
 */

template<typename nodeW, typename edgeW>
void Graph<nodeW,edgeW>::setMemGPU(node_sz nn, int mode) {

	cudaError cuSts;
	if (mode == GPUINIT_NODES) {
		//std::cout << "Alloc degs" << std::endl;
		cuSts = cudaMallocManaged(&str, sizeof(GraphStruct<nodeW,edgeW>)); cudaCheck( cuSts, __FILE__, __LINE__ );
		cuSts = cudaMallocManaged(&(str->cumulDegs), (nn+1)*sizeof(node_sz)); cudaCheck( cuSts, __FILE__, __LINE__ );
		//GPUMemTracker::graphStructSize = sizeof(GraphStruct<nodeW,edgeW>);
		//GPUMemTracker::graphDegsSize   = (nn+1)*sizeof(node_sz);
	}
	else if (mode == GPUINIT_EDGES) {
		//std::cout << "Alloc neighs" << std::endl;
		cuSts = cudaMallocManaged(&(str->neighs), str->nEdges*sizeof(node)); cudaCheck( cuSts, __FILE__, __LINE__ );
		//GPUMemTracker::graphNeighsSize = str->nEdges*sizeof(node);
	}
	else if (mode == GPUINIT_NODEW) {
		cuSts = cudaMallocManaged(&(str->nodeWeights), str->nEdges*sizeof(nodeW)); cudaCheck( cuSts, __FILE__, __LINE__ );
		//GPUMemTracker::graphNodeWSize = str->nEdges*sizeof(nodeW);
	}
	else if (mode == GPUINIT_EDGEW) {
		cuSts = cudaMallocManaged(&(str->edgeWeights), str->nEdges*sizeof(edgeW)); cudaCheck( cuSts, __FILE__, __LINE__ );
		//GPUMemTracker::graphEdgeWSize = str->nEdges*sizeof(edgeW);
	}
	else if (mode == GPUINIT_NODET) {
		cuSts = cudaMallocManaged(&(str->nodeThresholds), str->nNodes*sizeof(nodeW)); cudaCheck( cuSts, __FILE__, __LINE__ );
		//GPUMemTracker::graphNodeTSize = str->nNodes*sizeof(nodeW);
	}
}

/**
 * Invoke the kernel to print the graph on device
 * @param verbose print details
 */
template<typename nodeW, typename edgeW> void Graph<nodeW, edgeW>::print_d(bool verbose) {

	Graph_k::print_d<<<1,1>>>(str,verbose);
	cudaDeviceSynchronize();
}

/**
 * Print the graph on device (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
template<typename nodeW, typename edgeW>
__global__ void Graph_k::print_d(GraphStruct<nodeW,edgeW>* str, bool verbose) {
	printf("** Graph (num node: %d, num edges: %d)\n", str->nNodes,str->nEdges);

	if (verbose) {
		for (int i = 0; i < str->nNodes; i++) {
			printf("  node(%d)[%d]-> ",i,str->cumulDegs[i+1]-str->cumulDegs[i]);
			for (int j = 0; j < str->cumulDegs[i+1] - str->cumulDegs[i]; j++) {
				printf("%d ", str->neighs[str->cumulDegs[i]+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}


template<typename nodeW, typename edgeW>
void Graph<nodeW, edgeW>::deleteMemGPU() {
	if (str->neighs != nullptr) {
		cudaFree( str->neighs );
		str->neighs = nullptr;
	}
	if (str->cumulDegs != nullptr) {
		cudaFree( str->cumulDegs );
		str->cumulDegs = nullptr;
	}
	if (str->nodeWeights != nullptr) {
		cudaFree( str->nodeWeights );
		str->nodeWeights = nullptr;
	}
	if (str->edgeWeights != nullptr) {
		cudaFree( str->edgeWeights );
		str->edgeWeights = nullptr;
	}
	if (str->nodeThresholds != nullptr) {
		cudaFree( str->nodeThresholds );
		str->nodeThresholds = nullptr;
	}
	if (str != nullptr) {
		cudaFree( str );
		str = nullptr;
	}
}
