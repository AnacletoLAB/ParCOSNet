// COSnet - Hopfield Net GPU class
// Alessandro Petrini, 2017
#ifdef WIN32
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#endif

#include <iostream>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <math.h>
#include "hopfield_net/HopfieldNet.h"
#include "hopfield_net/HopfieldNetUtils.h"
#include "graph/graph.h"

#define ITERATION_LIMIT 5000

template<typename nodeW, typename edgeW>
HopfieldNetGPU<nodeW, edgeW>::HopfieldNetGPU( const Graph<nodeW, edgeW> * const inGraph_d, const Coloring * const inCol_d,
		float inPosState, float inNegState, float inRegulWeight ):
		HopfieldNet<nodeW, edgeW>( inGraph_d, inCol_d, inPosState, inNegState, inRegulWeight ),
		graph_d( inGraph_d ),
		col_d( inCol_d ) {

	cudaError_t cuSts;
	this->hState.state = new float[this->hState.size];
	this->hState.score = new float[this->hState.size];

	hState_d.size = this->hState.size;
	cuSts = cudaMalloc( (void**)&(hState_d.state), hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&(hState_d.score), hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );

	numThreads = 32;
	threadsPerBlock = dim3( numThreads, 1, 1 );
}


template<typename nodeW, typename edgeW>
HopfieldNetGPU<nodeW, edgeW>::~HopfieldNetGPU() {
	cudaError_t cuSts;
	cuSts = cudaFree( hState_d.score ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( hState_d.state ); cudaCheck( cuSts, __FILE__, __LINE__ );
	delete[] this->hState.score;
	delete[] this->hState.state;
}



///////////////////////////////////

template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::run_nodewise() {
	cudaError_t cuSts;

#ifdef PRINTHOPFIELDTITLE
	std::cout << "\033[32;1m** Hopfiled Net GPU alternative runner **\033[0m" << std::endl;
#endif

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	this->numIter = 0;
	bool modified = true;
	bool *modified_d;
	cuSts = cudaMalloc( (void**) &modified_d, sizeof(bool) ); cudaCheck(cuSts,__FILE__,__LINE__);

	std::unique_ptr<uint32_t[]> ISsize_h( new uint32_t[col_d->nCol + 1] );
	cuSts = cudaMemcpy( ISsize_h.get(), col_d->cumulSize, (col_d->nCol + 1) * sizeof( uint32_t ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );

#ifdef VERBOSEHOPFIELD
	printf( "Numero colori: %d\n", col->nCol );
	for ( int i = 0; i < col->nCol; i++)
		printf( "colore %d: %d\n", i, ISsize_h[i] );
#endif

	cudaEventRecord( start );
	while (modified) {

		this->numIter++;
		cuSts = cudaMemset( modified_d, false, sizeof(bool) ); cudaCheck( cuSts, __FILE__, __LINE__ );

		for (uint32_t ISidx = 0; ISidx < col_d->nCol; ISidx++) {

			uint32_t numberOfNodes = ISsize_h[ISidx + 1] - ISsize_h[ISidx];
			blocksPerGrid = dim3( (numberOfNodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1 );

			// launch the Hopfield kernel
			HopfieldNetGPU_k::updateIS_nodewise<nodeW, edgeW> <<<blocksPerGrid, threadsPerBlock >>> (
					hState_d.state,
					hState_d.score,
					graph_d->getStruct()->cumulDegs, graph_d->getStruct()->edgeWeights, graph_d->getStruct()->neighs, graph_d->getStruct()->nodeThresholds,
					graph_d->getStruct()->nNodes,
					col_d->nCol, col_d->colClass, col_d->cumulSize,
					ISidx,
					modified_d,
					this->posState,
					this->negState,
					this->regulWeight
					);

			cudaDeviceSynchronize();
			cuSts = cudaGetLastError(); cudaCheck( cuSts, __FILE__, __LINE__ );

		}

		cuSts = cudaMemcpy(&modified, modified_d, sizeof(bool), cudaMemcpyDeviceToHost); cudaCheck( cuSts, __FILE__, __LINE__ );
		if (this->numIter > ITERATION_LIMIT) {
			std::cout << "Massimo numero di iterazioni raggiunto!!! Uscita forzata" << std::endl;
			break;
		}
	}

	cuSts = cudaEventRecord(stop); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaEventSynchronize(stop); cudaCheck( cuSts, __FILE__, __LINE__ );
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
#ifdef VERBOSEHOPFIELD
	std::cout << "Stabilita' raggiunta in " << numIter << " iterazioni" << std::endl;
#endif

	// final state & log
	cuSts = cudaMemcpy( this->hState.state, this->hState_d.state, this->hState.size * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
	//HL->GPUrunTime = milliseconds / 1000;
	//HL->GPUnumIter = num_iter;
	//HL->speedup = HL->runTime / HL->GPUrunTime;
	cuSts = cudaFree(modified_d); cudaCheck( cuSts, __FILE__, __LINE__ );

	cuSts = cudaEventDestroy( stop ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaEventDestroy( start ); cudaCheck( cuSts, __FILE__, __LINE__ );
}

template<typename nodeW, typename edgeW>
__global__ void HopfieldNetGPU_k::updateIS_nodewise
			( float * const state, float * const score,
			node_sz * cumulDegs, edgeW * edgeWeights, node * neighs_, nodeW * nodeThresholds,
			const node_sz nNodes,
			const uint32_t nCol, const uint32_t	* const colClass, const uint32_t * const cumulSize,
			const int colorIdx,
			bool * const modified_d,
			const float posState, const float negState, const float regulWeight ) {

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= (cumulSize[colorIdx + 1] - cumulSize[colorIdx]))
		return;

	float newScore = 0;

	const int 			offsetCol = cumulSize[colorIdx];						// offset per il coloring
	const int 			node   = colClass[offsetCol + tid];
	const int			offset = cumulDegs[node];
	const int 			degree = cumulDegs[node + 1] - offset;
	const edgeW * const weights = &(edgeWeights[offset]);
	const uint32_t	* const neighs  = &(neighs_[offset]);
	unitVal oldState = state[node];

	for (int i = 0; i < degree; i++) {
		newScore += (weights[i] - regulWeight) * state[neighs[i]];
	}
	__syncthreads(); // Non dovrebbe servire

	// modifica per regolarizzazione
	uint32_t nodoreg;
	for (uint32_t i = 0; i < nCol; i++) {
		if (i == colorIdx)
			continue;
		else {
			uint32_t IS_size = cumulSize[i + 1] - cumulSize[i];
			for (uint32_t k = 0; k < IS_size; k++) {
				nodoreg = colClass[cumulSize[i] + k];
				newScore -= state[nodoreg] * regulWeight;
			}
		}
	}
	__syncthreads();

	score[node] = newScore - nodeThresholds[node];
	state[node] = SIGNTH( (newScore - nodeThresholds[node]) );

	if (state[node] != oldState) {
		*modified_d = true;
	}
}


////////////////////////////////////////

template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::run_edgewise() {
	cudaError_t cuSts;

#ifdef PRINTHOPFIELDTITLE
	std::cout << "\033[32;1m** Hopfiled Net GPU runner **\033[0m" << std::endl;
#endif

	//timer cudaEvent per Benchmark
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//conto iterazioni e criterio di arresto
	this->numIter = 0;
	bool modified = true;
	bool *modified_d;
	cuSts = cudaMalloc( (void**) &modified_d, sizeof(bool) ); cudaCheck( cuSts, __FILE__, __LINE__ );

	//alloco e copio cumulSize
	std::unique_ptr<uint32_t[]> cumulSize_h( new uint32_t[ (col_d->nCol+1) ] );
	cuSts = cudaMemcpy( cumulSize_h.get(), col_d->cumulSize, (col_d->nCol+1) * sizeof( uint32_t ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );

#ifdef VERBOSEHOPFIELD
	printf( "Numero colori: %d\n", col->nCol );
	for ( int i = 0; i < col->nCol; i++)
		printf( "colore %d: %d\n", i, ISsize_h[i] );
#endif

	// **  run net on device: loop on ISs  **
	CHECK( cudaEventRecord( start ) );
	while ( modified ) {

		this->numIter++;
		cuSts = cudaMemset( modified_d, false, sizeof(bool) ); cudaCheck( cuSts, __FILE__, __LINE__ );

		// update all ISs
		for (uint32_t ISidx = 0; ISidx < col_d->nCol; ISidx++) {
			// col.meanUnitDeg non implementato nel colorer.
			/*
			int numThreads = pow(2, floor(log(col.meanUnitDeg[ISidx]) / log(2)));
			if (numThreads < 32)
				numThreads = 32;
			if (numThreads > 1024)
				numThreads = 1024;
			*/
			//int numThreads = 32;
			//dim3 blocksize(numThreads);			// num threads = average deg IS nodes crop to (32,1024)
			// col.ISsize[] inaccessibile da host.
			//dim3 gridsize(col.ISsize[ISidx]);	// num blocks = IS size
			//dim3 gridsize( ISsize_h[ISidx] );

			uint32_t colorSize = cumulSize_h[ISidx + 1] - cumulSize_h[ISidx];
			blocksPerGrid = dim3( colorSize, 1, 1 );

			// launch the Hopfield kernel
			HopfieldNetGPU_k::updateIS_edgewise<<<blocksPerGrid, threadsPerBlock, numThreads * sizeof(float)>>>(
					hState_d.state,
					hState_d.score,
					graph_d->getStruct()->cumulDegs, graph_d->getStruct()->edgeWeights, graph_d->getStruct()->neighs, graph_d->getStruct()->nodeThresholds,
					graph_d->getStruct()->nNodes,
					col_d->nCol, col_d->colClass, col_d->cumulSize,
					ISidx,
					modified_d,
					this->posState,
					this->negState,
					this->regulWeight
				);

			cudaDeviceSynchronize();
			cuSts = cudaGetLastError(); cudaCheck( cuSts, __FILE__, __LINE__ );

		}

		cuSts = cudaMemcpy(&modified, modified_d, sizeof(bool), cudaMemcpyDeviceToHost); cudaCheck( cuSts, __FILE__, __LINE__ );
		if (this->numIter > ITERATION_LIMIT) {
			std::cout << "Massimo numero di iterazioni raggiunto!!! Uscita forzata" << std::endl;
			break;
		}
	}

	cuSts = cudaEventRecord(stop); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaEventSynchronize(stop); cudaCheck( cuSts, __FILE__, __LINE__ );
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
#ifdef VERBOSEHOPFIELD
	std::cout << "Stabilita' raggiunta in " << numIter << " iterazioni" << std::endl;
#endif

	// final state & log
	cuSts = cudaMemcpy(this->hState.state, hState_d.state, this->hState.size * sizeof( unitVal ), cudaMemcpyDeviceToHost); cudaCheck( cuSts, __FILE__, __LINE__ );
	//HL->GPUrunTime = milliseconds / 1000;
	//HL->GPUnumIter = num_iter;
	//HL->speedup = HL->runTime / HL->GPUrunTime;
	cuSts = cudaFree(modified_d); cudaCheck( cuSts, __FILE__, __LINE__ );

	cuSts = cudaEventDestroy( stop ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaEventDestroy( start ); cudaCheck( cuSts, __FILE__, __LINE__ );
}


template<typename nodeW, typename edgeW>
__global__ void HopfieldNetGPU_k::updateIS_edgewise( float * const state, float * const score,
		node_sz * cumulDegs, edgeW * edgeWeights, node * neighs_, nodeW * nodeThresholds,
		const node_sz nNodes,
		const uint32_t nCol, const uint32_t	* const colClass, const uint32_t * const cumulSize,
		const int colorIdx,
		bool * const modified_d,
		const float posState, const float negState, const float regulWeight  ) {

	// ID del thread all'interno del nodo, serve per la parallel reduction sum
	unsigned int tid = threadIdx.x;
	// ID del blocco nella griglia, serve per il calcolo dello score edgewise
	unsigned int bid = blockIdx.x;
	// DIM del blocco 1D, serve per il calcolo dello score edgewise
	// serve per segmentare il calcolo/quanti edge associo ad un thread
	// nel caso il numero di vicini del nodo corrente superi blockDim
	unsigned int dim = blockDim.x;

	// color idx supera numero colori?
	if (colorIdx >= nCol)
		return;

	const int 		offsetCol = cumulSize[colorIdx];

	// il kernel non deve prendere nodi nodeIdx al di fuori del colore attuale
	if( (offsetCol + bid) >= cumulSize[colorIdx + 1] )
		return;

	const int 		nodeIdx = colClass[offsetCol + bid];

	extern __shared__ float smem[];

	const int 		offsetDeg	= cumulDegs[nodeIdx];
	const int 		degree = cumulDegs[nodeIdx+1] - offsetDeg;

	// Indica il numero di vicini che ogni thread deve cuccarsi
	// es. se numero vicini = 146 e numero thread per blocco = 32 =>
	// 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 4 4 4 4 4 4 4 4 4 4 4 4 4 4
	// es. se numero vicini = 6 e numero thread per blocco = 32 =>
	// 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	// verifica con:
	// 		int nn = 146; int tPerBlk = 32;
    //		for (int tid = 0; tid < tPerBlk; tid++)
    //		    std::cout << nn / tPerBlk + ((nn % tPerBlk) > tid) << " ";}
	int neighPerThread = degree / blockDim.x + ((degree % blockDim.x) > tid);

	smem[tid] = 0;
	__syncthreads();

	for (int i = 0; i < neighPerThread; i++) {
		int indx = neighs_[offsetDeg + i * dim + tid];
		smem[tid] += edgeWeights[offsetDeg + i * dim + tid] * state[indx];

#ifdef DEBUGPRINTK_IS
		if (bid == 0)
			printf("node: %d tid: %d nperthisThread: %d i: %d idx: %d smem[tid]: %f\n", nodeIdx, tid, neighPerThread, i, indx, smem[tid]);
#endif

	}
	__syncthreads();

	// sum cache by parallel reduction
	for (unsigned int i = dim / 2; i > 32; i >>= 1) {
		if (tid < i)
			smem[tid] += smem[tid + i];
		__syncthreads();
	}
	// last warp
	if (tid < 16) {
		smem[tid] += smem[tid + 16];
		__syncthreads();
		smem[tid] += smem[tid + 8];
		__syncthreads();
		smem[tid] += smem[tid + 4];
		__syncthreads();
		smem[tid] += smem[tid + 2];
		__syncthreads();
		smem[tid] += smem[tid + 1];
		__syncthreads();
	}
	//__syncthreads();

#ifdef DEBUGPRINTK_IS
	// Naive reduction usata in fase di test...
	if (tid == 0) {
		//for( int i = 1; i < dim; i++) {
		//	smem[0] += smem[i];
		//}
		printf("node: %d tid: %d smem[tid]: %f\n", nodeIdx, tid, smem[tid]);
	}
#endif

	// update state
	if (tid == 0) {
		unitVal oldState = state[nodeIdx];

		// modifica per regolarizzazione
		uint32_t nodoreg;
		for (uint32_t i = 0; i < nCol; i++) {
			if (i == colorIdx)
				continue;
			else {
				uint32_t IS_size = cumulSize[i + 1] - cumulSize[i];
				for (uint32_t k = 0; k < IS_size; k++) {
					nodoreg = colClass[cumulSize[i] + k];
					smem[0] -= state[nodoreg] * regulWeight;
				}
			}
		}

		// aggiorno state e score
		score[nodeIdx] = smem[0] - nodeThresholds[nodeIdx];
		state[nodeIdx] = SIGNTH((smem[0] - nodeThresholds[nodeIdx]));

		//controllo se lo stato è stato modificato
		if (state[nodeIdx] != oldState) {
			*modified_d = true;
		}
	}
}





//////////////////////////////////////////////////////////

// va lasciato cudaMemset a 0 per hScore?
template<typename nodeW, typename edgeW>
	void HopfieldNetGPU<nodeW, edgeW>::setInitState( const unitVal * const inState, const  unitVal* const inScore ) {
		cudaError_t cuSts;
		for (int i = 0; i < this->hState.size; i++){
			this->hState.state[i] = static_cast<float>(inState[i]);
			this->hState.score[i] = 0;
		}
		cuSts = cudaMemcpy( hState_d.state, this->hState.state, hState_d.size * sizeof( unitVal ), cudaMemcpyHostToDevice ); cudaCheck( cuSts, __FILE__, __LINE__ );
		cuSts = cudaMemset( hState_d.score, 0, hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	}

// setta tuti gli initial state = inValue
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::setInitState( const unitVal inValue ) {
	cudaError_t cuSts;
	cuSts = cudaMemset( hState_d.state, inValue, hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset( hState_d.score, 0, hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
}

// setta a 0 state e score su memoria device
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::clearInitState() {
	cudaError_t cuSts;
	cuSts = cudaMemset(hState_d.state, 0, hState_d.size * sizeof( unitVal )); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemset(hState_d.score, 0, hState_d.size * sizeof( unitVal )); cudaCheck( cuSts, __FILE__, __LINE__ );
}

// GPURandomizer riempie casualmente state e score su memoria device
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::setRandomInitState( GPURand * const randomizer ) {
	//randomizer->fillRandom( hState_d.state, hState_d.size );
}

template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::setInitStateProb( Prob p, char type ) {
	cudaError_t cuSts;
	if ( type == 'z') {
		std::fill( this->hState.state, this->hState.state + this->hState.size, (-0.5 < 0 ? this->negState : this->posState) );
	} else if ( type == 'o') {
		std::fill( this->hState.state, this->hState.state + this->hState.size, (0.5 < 0 ? this->negState : this->posState) );
	} else if ( type == 'r') {
		unitVal pS = this->posState;
		unitVal nS = this->negState;
		std::generate( this->hState.state, this->hState.state + this->hState.size, [p, pS, nS](){return SIGNTHLAMBDA(p-randf(0, 1));} );
	}
	cuSts = cudaMemcpy(hState_d.state, this->hState.state, hState_d.size * sizeof( unitVal ), cudaMemcpyHostToDevice ); cudaCheck( cuSts, __FILE__, __LINE__ );
}

// ritorna i valori di state e score
// serve perchè hState e hState_d sono campi protected
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::returnVal( float * const inState, float * const inScore ) {
	cudaError_t cuSts;
	cuSts = cudaMemcpy(this->hState.state, hState_d.state, hState_d.size * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemcpy(this->hState.score, hState_d.score, hState_d.size * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
	for (int i = 0; i < hState_d.size; i++) {
		inState[i] = this->hState.state[i];
		inScore[i] = this->hState.score[i];
	}
}

// Funzione di test per valutare correttezza del kernel "accumulateScores"
// template<typename nodeW, typename edgeW>
// void HopfieldNetGPU<nodeW, edgeW>::normalizeScore( const GraphStruct<nodeW, edgeW> * const bigGraph, const uint32_t *const reduxToFull, const edgeW * const sumOfWghs_h ) {
// 	cudaError_t cuSts;
// 	uint32_t n = graph_d->getStruct()->nNodes;
// 	uint32_t nOrig = bigGraph->nNodes;
// 	dim3 threadPerBlk( TPB_ACCUMUL, 1, 1 );
// 	uint32_t bPg = (n + 2 * threadPerBlk.x - 1) / (2 * threadPerBlk.x);
// 	dim3 blocksPerGrd( bPg, 1, 1 );
//
// 	unitVal		*	input = new unitVal[n];
// 	unitVal		*	input_d;
// 	unitVal		*	accumulatedScores_h = new unitVal[bPg];
// 	unitVal		*	accumulatedScores_d;
//
// 	cuSts = cudaMalloc( (void**)&input_d, n * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
// 	cuSts = cudaMalloc( (void**)&accumulatedScores_d, bPg * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
// 	std::fill( input, input + n, 1.0 );
// 	cuSts = cudaMemcpy( input_d, input, n * sizeof( unitVal ), cudaMemcpyHostToDevice );
// 	HopfieldNetGPU_k::accumulateScores <<<blocksPerGrd, threadPerBlk>>> ( n, input_d, accumulatedScores_d );
// 	cudaDeviceSynchronize();
// 	cuSts = cudaGetLastError(); cudaCheck( cuSts, __FILE__, __LINE__ );
// 	cuSts = cudaMemcpy( accumulatedScores_h, accumulatedScores_d, bPg * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
// 	unitVal totScore_d = std::accumulate( accumulatedScores_h, accumulatedScores_h + bPg, 0.0 );
// 	unitVal totScore_h = std::accumulate( input, input + n, 0.0 );
//
//
// 	uint32_t bPg2 = (n + threadPerBlk.x - 1) / (threadPerBlk.x);
// 	unitVal		*	accumulatedScores2_h = new unitVal[bPg2];
// 	unitVal		*	accumulatedScores2_d;
// 	cuSts = cudaMalloc( (void**)&accumulatedScores2_d, bPg2 * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
// 	dim3 blocksPerGrd2( bPg2, 1, 1 );
// 	HopfieldNetGPU_k::accumulateScores2 <<<blocksPerGrd2, threadPerBlk>>> ( n, input_d, accumulatedScores2_d );
// 	cudaDeviceSynchronize();
// 	cuSts = cudaGetLastError(); cudaCheck( cuSts, __FILE__, __LINE__ );
// 	cuSts = cudaMemcpy( accumulatedScores2_h, accumulatedScores2_d, bPg2 * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
// 	unitVal totScore2_d = std::accumulate( accumulatedScores2_h, accumulatedScores2_h + bPg2, 0.0 );
// 	std::cout << "n: " << n << " totScore_d: " << totScore_d << " totScore2_d: " << totScore2_d << " - totScore_h: " << totScore_h << std::endl;
// 	// if (totScore_d != totScore_h) {
// 	// 	std::cout << "totScore_d: " << totScore_d << " - totScore_h: " << totScore_h << std::endl;
// 	// 	abort();
// 	// }
// }



template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::normalizeScore( const GraphStruct<nodeW, edgeW> * const bigGraph, const uint32_t *const reduxToFull, const edgeW * const sumOfWghs_h ) {
	cudaError_t	cuSts;
	uint32_t	n = graph_d->getStruct()->nNodes;
	uint32_t	nOrig = bigGraph->nNodes;
	dim3		threadPerBlk( TPB_ACCUMUL, 1, 1 );
	uint32_t	bPg = (n + 2 * threadPerBlk.x - 1) / (2 * threadPerBlk.x);
	dim3		blocksPerGrd( bPg, 1, 1 );

	unitVal		*	accumulatedScores;
	uint32_t	*	reduxToFull_d;
	unitVal		*	sumOfWghs_d;
	cuSts = cudaMalloc( (void**)&accumulatedScores,	bPg * sizeof( unitVal ) );		cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&reduxToFull_d,		n * sizeof( uint32_t ) );		cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMalloc( (void**)&sumOfWghs_d,		nOrig * sizeof( unitVal ) );	cudaCheck( cuSts, __FILE__, __LINE__ );
	std::unique_ptr<unitVal[]> 	accumulatedScores_h( new unitVal[bPg] );

	// Calcolo somma degli scores
	HopfieldNetGPU_k::accumulateScores <<<blocksPerGrd, threadPerBlk>>> ( n, hState_d.score, accumulatedScores );
	cuSts = cudaGetLastError(); cudaCheck( cuSts, __FILE__, __LINE__ );

		// uint32_t bPg2 = (n + threadPerBlk.x - 1) / (threadPerBlk.x);
		// unitVal		*	accumulatedScores2_h = new unitVal[bPg2];
		// unitVal		*	accumulatedScores2_d;
		// cuSts = cudaMalloc( (void**)&accumulatedScores2_d, bPg2 * sizeof( unitVal ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
		// dim3 blocksPerGrd2( bPg2, 1, 1 );
		// HopfieldNetGPU_k::accumulateScores2 <<<blocksPerGrd2, threadPerBlk>>> ( n, hState_d.score, accumulatedScores2_d );
		// cudaDeviceSynchronize();
		// cuSts = cudaGetLastError(); cudaCheck( cuSts, __FILE__, __LINE__ );
		// cuSts = cudaMemcpy( accumulatedScores2_h, accumulatedScores2_d, bPg2 * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
		// unitVal totScore2_d = std::accumulate( accumulatedScores2_h, accumulatedScores2_h + bPg2, 0.0 );



	// accumulazione della somma dei pesi dei nodi unlabelled
	float accumulatedWDeg = 0.0;
	for (uint32_t j = 0; j < n; j++) {
		accumulatedWDeg += sumOfWghs_h[reduxToFull[j]];
	}
	cudaDeviceSynchronize();

	cuSts = cudaMemcpy( accumulatedScores_h.get(), accumulatedScores, bPg * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemcpy( sumOfWghs_d, sumOfWghs_h, nOrig * sizeof( unitVal ), cudaMemcpyHostToDevice ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaMemcpy( reduxToFull_d, reduxToFull, n * sizeof( uint32_t ), cudaMemcpyHostToDevice ); cudaCheck( cuSts, __FILE__, __LINE__ );

	// finisco l'accumulazione degli score su CPU
	unitVal totScore = std::accumulate( accumulatedScores_h.get(), accumulatedScores_h.get() + bPg, 0.0 );

			// unitVal * temphStateScores = new unitVal[n];
			// cuSts = cudaMemcpy( temphStateScores, hState_d.score, n * sizeof( unitVal ), cudaMemcpyDeviceToHost );
			// unitVal tempAccScores = 0.0f;
			// std::for_each( temphStateScores, temphStateScores + n, [&tempAccScores]( unitVal nn ) {tempAccScores += fabs( nn ); } );
			// //if (totScore != tempAccScores) {
			// //	std::cout << "errore nella reduction sugli score. GPU = " << totScore << " - CPU: " << tempAccScores << std::endl;
			// //	abort();
			// //}
			// //std::cout << std::setprecision(6) << "n: " << n << " - Host - k1: " << tempAccScores - totScore << " - Host - k2: " << tempAccScores - totScore2_d << std::endl;
			// //std::cout << "n: " << n << " totScore: " << totScore << " totScore2_d: " << totScore2_d << " - host: " << tempAccScores << std::endl;
			// delete[] temphStateScores;
			// totScore = tempAccScores;

	bPg = (n + threadPerBlk.x - 1) / threadPerBlk.x;
	blocksPerGrd = dim3( bPg, 1, 1 );
	HopfieldNetGPU_k::normalizeScores <<<blocksPerGrd, threadPerBlk>>> ( n, accumulatedWDeg, totScore, sumOfWghs_d, reduxToFull_d, hState_d.score );
	cudaDeviceSynchronize();
	cuSts = cudaGetLastError(); cudaCheck( cuSts, __FILE__, __LINE__ );

	cuSts = cudaFree( sumOfWghs_d ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( reduxToFull_d ); cudaCheck( cuSts, __FILE__, __LINE__ );
	cuSts = cudaFree( accumulatedScores ); cudaCheck( cuSts, __FILE__, __LINE__ );
}


// __global__ void HopfieldNetGPU_k::accumulateScores2( const uint32_t unlab, const unitVal * const scores, unitVal * const accumScores ) {
// 	uint32_t baseBlock = blockDim.x * blockIdx.x;
//
// 	accumScores[blockIdx.x] = 0.0f;
// 	if (threadIdx.x==0) {
// 		for(uint32_t i = 0; i < blockDim.x; i++) {
// 			if (baseBlock + i < unlab)
// 				accumScores[blockIdx.x] += fabsf( scores[baseBlock + i] );
// 		}
// 	}
// }

__global__ void HopfieldNetGPU_k::accumulateScores( const uint32_t unlab, const unitVal * const scores, unitVal * const accumScores ) {

	__shared__ float tempScores[TPB_ACCUMUL];

	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= unlab / 2)
		return;
	uint32_t baseBlock = 2 * blockDim.x * blockIdx.x;
	tempScores[threadIdx.x] = fabsf(scores[baseBlock + threadIdx.x]);
	__syncthreads();

	uint32_t incremento = ((baseBlock + threadIdx.x + blockDim.x) < unlab) ? blockDim.x : (unlab % blockDim.x) / 2;
	tempScores[threadIdx.x] += fabsf(scores[baseBlock + threadIdx.x + incremento]);
	__syncthreads();

#pragma unroll
	for (uint32_t i = blockDim.x / 2; i > 0; i >>= 1) {
		if ((threadIdx.x < i) & (tid + i < unlab / 2)){
			tempScores[threadIdx.x] += tempScores[threadIdx.x + i];
		}
		__syncthreads();
	}

	// Mi prendo cura degli ultimi elementi dei vettori score e deg nel caso in cui
	// i nodi siano dispari
	if ((unlab % 2) && (tid == 0)) {
		tempScores[0] += fabsf(scores[unlab - 1]);
	}
	__syncthreads();

	accumScores[blockIdx.x] = tempScores[0];

	return;
}


__global__ void HopfieldNetGPU_k::normalizeScores( const uint32_t unlab, const float accumWDeg, const unitVal accumScores,
	const unitVal * const sumOfWeights, const uint32_t * const indexes, unitVal * const scores ) {

	uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= unlab)
		return;
	scores[tid] = sumOfWeights[indexes[tid]] / accumWDeg + scores[tid] / accumScores;

	return;
}



//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
//template class HopfieldNetGPU<col, col>;
template class HopfieldNetGPU<float, float>;
