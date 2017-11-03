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
#include <math.h>
#include "hopfield_net/HopfieldNet.h"
#include "hopfield_net/HopfieldNetUtils.h"
#include "graph/graph.h"

#define ITERATION_LIMIT 5000

/*
#	#	#	#	#	#	#	#	#	#	#	#	#

				HopfieldNetGPU CLASS

#	#	#	#	#	#	#	#	#	#	#	#	#
*/

/*
 *  CLASS HOPFIELDNETGPU: Hopfield Net run by GPU
 */
// Attenzione: si sta dando per scontato che il coloring passato al costruttore di
// HopfieldNetGPU sia un coloring_d già presente sulla GPU

 //COME FA A RIEMPIRE hState.size nel costruttore HopfieldNet se il grafo è in device?
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
			cuSts = cudaMalloc( (void**)&(hState_d.state), hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts );
			cuSts = cudaMalloc( (void**)&(hState_d.score), hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts );

			numThreads = 32;
			threadsPerBlock = dim3( numThreads, 1, 1 );
		};

/*
 *  Virttual destructor
 */
template<typename nodeW, typename edgeW>
HopfieldNetGPU<nodeW, edgeW>::~HopfieldNetGPU() {
	cudaError_t cuSts;
	cuSts = cudaFree( hState_d.score ); cudaCheck( cuSts );
	cuSts = cudaFree( hState_d.state ); cudaCheck( cuSts );
	delete[] this->hState.score;
	delete[] this->hState.state;
}



/*
#	#	#	#	#	#	#	#	#	#	#	#	#

		HopfieldNetGPU run & updateIS NODEWISE

#	#	#	#	#	#	#	#	#	#	#	#	#
*/

//		MAIN MEMBER
//		SINGLE THREAD

/*
	NB ricorda, usa un coloring già in memoria device

	RUN FLOW:
		1 variabili per benchmark, iterazioni e criterio di arresto
		2 copio cumulsize del coloring
		3 CICLO WHILE (finchè gli stati della rete vengono modificati o ITERATION_LIMIT)
			4 CICLO FOR (sui colori/indipendent set)
				lancio dei kernel per ogni colore

*/

template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::run_nodewise() {
	cudaError_t cuSts;

#ifdef PRINTHOPFIELDTITLE
	std::cout << "\033[32;1m** Hopfiled Net GPU alternative runner **\033[0m" << std::endl;
#endif

	//timer cudaEvent per Benchmark
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//conto iterazioni e criterio di arresto
	this->numIter = 0;
	bool modified = true;
	bool *modified_d;
	cuSts = cudaMalloc( (void**) &modified_d, sizeof(bool) ); cudaCheck( cuSts );
	cuSts = cudaMemset( modified_d, true, sizeof( bool ) ); cudaCheck( cuSts ); //forse non serve ma andiamo sul sicuro

	//alloco e copio cumulSize
	std::unique_ptr<uint32_t[]> cumulSize_h( new uint32_t[ (col_d->nCol+1) ] );
	cuSts = cudaMemcpy( cumulSize_h.get(), col_d->cumulSize, (col_d->nCol+1) * sizeof( uint32_t ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts );

#ifdef VERBOSEHOPFIELD
	printf( "Numero colori: %d\n", col_d->nCol );
	for ( int i = 0; i < col_d->nCol; i++)
		printf( "colore %d: %d\n", i, cumulSize_h[i] );
#endif

	cudaEventRecord( start );

	//ciclo finchè non si è stabilizzato numItert=ITERATION_LIMIT
	while (modified) {
		(this->numIter)++;
		cuSts = cudaMemset( modified_d, false, sizeof(bool) ); cudaCheck( cuSts );

		//ciclo sui colori
		for (uint32_t ISidx = 0; ISidx < col_d->nCol; ISidx++) {
			uint32_t numberOfNodes = cumulSize_h[ISidx + 1] - cumulSize_h[ISidx];

			blocksPerGrid = dim3( (numberOfNodes + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1 );

			// launch the Hopfield kernel
			HopfieldNetGPU_k::updateIS_nodewise<<<blocksPerGrid, threadsPerBlock, numberOfNodes * sizeof(float)>>>(
					hState_d.state,			// net state
					hState_d.score,			// net score
					graph_d->getStruct(),	// graph structure (neighs, weighs, thresholds)
					col_d,					// graph coloring
					ISidx,					// indipendent set/color index
					modified_d,				// stop cond
					this->posState,
					this->negState
					);

			cudaDeviceSynchronize();
			cuSts = cudaGetLastError();
			cudaCheck2(cuSts,__FILE__,__LINE__);
			/*if (cudaGetLastError() != cudaSuccess) {
				std::cout << "HopfieldNetGPU_k::updateIS_nodewise, iterazione n: " << this->numIter << std::endl;
				std::cout << cudaGetErrorString( cudaGetLastError() ) << std::endl; abort();
			}// DEBUG*/
		}

		cuSts = cudaMemcpy(&modified, modified_d, sizeof(bool), cudaMemcpyDeviceToHost); cudaCheck( cuSts );
		if (this->numIter > ITERATION_LIMIT) {
			std::cout << "Massimo numero di iterazioni raggiunto!!! Uscita forzata" << std::endl;
			break;
		}
	}

	cuSts = cudaEventRecord(stop); cudaCheck( cuSts );
	cuSts = cudaEventSynchronize(stop); cudaCheck( cuSts );
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
#ifdef VERBOSEHOPFIELD
	std::cout << "Stabilita' raggiunta in " << numIter << " iterazioni" << std::endl;
#endif

	// final state & log
	cuSts = cudaMemcpy( this->hState.state, hState_d.state, this->hState.size * sizeof( int ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts );
	//HL->GPUrunTime = milliseconds / 1000;
	//HL->GPUnumIter = num_iter;
	//HL->speedup = HL->runTime / HL->GPUrunTime;
	cuSts = cudaFree(modified_d); cudaCheck( cuSts );
}



// Kernel alternativo di esecuzione della dinamica della rete
// Ogni nodo e' assegnato ad un thread e calcola sequenzialemente il prodotto
// pesoArco * statoDelVicino accumulandolo in shared memory

/*
	KERNEL FLOW
		1 controllo tid < graphStruct_d->nNodes
		2 controllo colorIdx < col_d->nCol
			se tutto ok creo offsetCol
		3 controllo se (offsetCol + tid) < col_d->cumulSize[colorIdx + 1],
			(se vado a prendere un nodo del colore sbagliato)
		4 shared memory, offset e grado per il nodeIdx
		5 ciclo algoritmo della rete di Hopfield
		6 aggiorno state e score di nodeIdx
		7 controllo se lo stato di nodeIdx è stato modificato
			se sì *modified_d=true
*/

template<typename nodeW, typename edgeW>
__global__ void HopfieldNetGPU_k::updateIS_nodewise( float * const state, float * const score,				// Out values
		const GraphStruct<nodeW, edgeW> * const graphStruct_d,									// graph stuff
		const Coloring * const col_d, const int colorIdx,											// coloring stuff
		bool * const modified_d,																	// stop condition
		const float posState, const float negState ) {												// float const

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// thread idx supera numero nodi?
	if (tid >= graphStruct_d->nNodes)
		return;

	// color idx supera numero colori?
	if (colorIdx >= col_d->nCol)
		return;

	const int 		offsetCol = col_d->cumulSize[colorIdx];						// offset per il coloring
	//const int 		colSize = col_d->cumulSize[colorIdx + 1] - offsetCol;		// dim del colore attuale

	// ricorda la struttura di colClass e cumulSize nella classe coloring
	// il kernel non deve prendere nodi nodeIdx al di fuori del colore attuale
	if( (offsetCol + tid) >= col_d->cumulSize[colorIdx + 1] )
		return;

	// NOTA per la natura di update_to_standard_notation in ColoringLuby,
	// i nodi di ogni colore sono in ordine crescente di indice
	const int 		nodeIdx = col_d->colClass[offsetCol + tid];					// indice del nodo in cui lavoreremo

	// non alloco prima dei 3 controlli (?)
	extern __shared__ float smem[];
	smem[tid] = 0;

	const int 		offsetDeg	= graphStruct_d->cumulDegs[nodeIdx];		// offset per neighs di nodeIdx
	const int 		degree = graphStruct_d->cumulDegs[nodeIdx+1] - offsetDeg;	// per il ciclo
	unitVal oldState = state[nodeIdx];
	int neighIdx;

	for (int i = 0; i < degree; i++) {
		neighIdx = graphStruct_d->neighs[offsetDeg + i];
		//smem[tid] += (graphStruct_d->nodeWeights[neighIdx] - regulWeight) * state[neighIdx];
		smem[tid] += graphStruct_d->edgeWeights[offsetDeg + i] * state[neighIdx];
	}

	__syncthreads();	// Forse inutile, ma con la shared mem meglio andarci cauti!
/*
	// modifica per la regolarizzazione
	// COMMENTARE
	// la regolazione non viene fatta nell'Hopfiled liscio
	int nodoreg;
	int regOffset;
	int regSize;
	for (int i = 0; i < col_d->nCol; i++) {
		if (i == colorIdx)
			continue;
		else{
			regOffset = col_d->cumulSize[i];
			regSize = col_d->cumulSize[i+1] - regOffset;
			for (int k = 0; k < regSize; k++){
				nodoreg=col_d->colClass[regOffset+k];
				smem[tid] -= state[nodoreg] * regulWeight;
			}
		}
	}

	__syncthreads();*/

	// aggiorno state e score
	score[nodeIdx] = smem[tid] - graphStruct_d->nodeThresholds[nodeIdx];
	state[nodeIdx] = SIGNTH((smem[tid] - graphStruct_d->nodeThresholds[nodeIdx]));

	//controllo se lo stato è stato modificato
	if (state[nodeIdx] != oldState) {
		*modified_d = true;
	}
}



/*
#	#	#	#	#	#	#	#	#	#	#	#	#

		HopfieldNetGPU run & updateIS EDGEWISE

#	#	#	#	#	#	#	#	#	#	#	#	#
*/

//		MAIN MEMBER
//		SINGLE THREAD

/*
	NB ricorda, usa un coloring già in memoria device

	RUN FLOW:
		1 variabili per benchmark, iterazioni e criterio di arresto
		2 copio cumulsize del coloring
		3 CICLO WHILE (finchè gli stati della rete vengono modificati o ITERATION_LIMIT)
			4 CICLO FOR (sui colori/indipendent set)
				lancio dei kernel edgewise per ogni colore

*/

template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::run_edgewise() {
	cudaError_t cuSts;

#ifdef PRINTHOPFIELDTITLE
	std::cout << "\033[32;1m** Hopfiled Net GPU runner **\033[0m" << std::endl;
#endif

	//IMPLEMENTATO(?)
	/*
	if (!comprGraph->connected) {
		std::cout << "Warning: graph non connected... EXIT!\n";
		return;
	}*/

	//timer cudaEvent per Benchmark
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//conto iterazioni e criterio di arresto
	unsigned N = graph_d->getStruct()->nNodes;
	this->numIter = 0;
	bool modified = true;
	bool *modified_d;
	cuSts = cudaMalloc( (void**) &modified_d, sizeof(bool) ); cudaCheck( cuSts );
	cuSts = cudaMemset( modified_d, true, sizeof( bool ) ); cudaCheck( cuSts ); //forse non serve ma andiamo sul sicuro

	//alloco e copio cumulSize
	std::unique_ptr<uint32_t[]> cumulSize_h( new uint32_t[ (col_d->nCol+1) ] );
	cuSts = cudaMemcpy( cumulSize_h.get(), col_d->cumulSize, (col_d->nCol+1) * sizeof( uint32_t ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts );

#ifdef VERBOSEHOPFIELD
	printf( "Numero colori: %d\n", col->nCol );
	for ( int i = 0; i < col->nCol; i++)
		printf( "colore %d: %d\n", i, ISsize_h[i] );
#endif

	// **  run net on device: loop on ISs  **
	CHECK( cudaEventRecord( start ) );
	while ( modified ) {
		/*numIter++;
		CHECK(cudaMemset(CUDASTOP_d, 0, sizeof(int)));*/
		this->numIter++;
		cuSts = cudaMemset( modified_d, false, sizeof(bool) ); cudaCheck( cuSts );

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

			uint32_t colorSize = col_d->cumulSize[ISidx + 1] - col_d->cumulSize[ISidx];
			blocksPerGrid = dim3( colorSize, 1, 1 );

			// launch the Hopfield kernel
			HopfieldNetGPU_k::updateIS_edgewise<<<blocksPerGrid, threadsPerBlock, numThreads * sizeof(float)>>>(
					hState_d.state,			// net state
					hState_d.score,
					graph_d->getStruct(),	// graph structure (neighs, weighs, thresholds)
					col_d,					// graph coloring
					ISidx,					// IS ID
					modified_d,				// stop cond
					this->posState,
					this->negState
					);

			cudaDeviceSynchronize();
			if (cudaGetLastError() != cudaSuccess) { std::cout << "CUDA ERROR: HopfieldNetGPU_k::updateIS_edgewise" << std::endl; abort(); }			// DEBUG

		}

		cuSts = cudaMemcpy(&modified, modified_d, sizeof(bool), cudaMemcpyDeviceToHost); cudaCheck( cuSts );
		if (this->numIter > ITERATION_LIMIT) {
			std::cout << "Massimo numero di iterazioni raggiunto!!! Uscita forzata" << std::endl;
			break;
		}
	}

	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
#ifdef VERBOSEHOPFIELD
	std::cout << "Stabilita' raggiunta in " << numIter << " iterazioni" << std::endl;
#endif

	// final state & log
	CHECK(cudaMemcpy(this->hState.state, hState_d.state, N * sizeof(int), cudaMemcpyDeviceToHost));
	//HL->GPUrunTime = milliseconds / 1000;
	//HL->GPUnumIter = num_iter;
	//HL->speedup = HL->runTime / HL->GPUrunTime;
	cuSts = cudaFree(modified_d); cudaCheck( cuSts );
}



// Questo kernel calcola la dinamica della rete di Hopfield.
// Un nodo e' assegnato ad un blocco di thread, fino al massimo numero di blocchi previsti dalla
// configurazione di lancio del kernel.
// Ad ogni thread di un blocco viene assegnato un numero di edge del nodo corrente compatibile
// con il numero massimi di thread prevsti dalla configurazione di lancio, e in esso
// calcola il prodotto pesoArco * statoDelVicino, accumulandolo in shared memory.
// Dopo la fase di calcolo, una reduction additiva calcola il nuovo score e da esso se ne deriva
// il nuovo stato

/*
	KERNEL FLOW
		1 estraggo threadIdx, blockIdx e blockDim
		2 controllo colorIdx < col_d->nCol
			se tutto ok creo offsetCol
		3 controllo se (offsetCol + blockIdx) < col_d->cumulSize[colorIdx + 1],
			(se vado a prendere un nodo del colore sbagliato)
		4 NOTA:
			nodeIdx dipende da blockIdx
			ogni blocco si occupa di un nodo
			ogni thread si occupa di uno o più vicini di nodeIdx
		5 shared memory, offset e grado per il nodeIdx
		6 calcolo interno al thread di quanti edge/vicini deve occuparsi
		7 ciclo algoritmo della rete di Hopfield
			(nella shared mem ogni thread memorizza la sua somma parziale)
		8 somma in parallelo con riduzione, ultimi cicli unrolled
		9 aggiorno state e score di nodeIdx
		10 controllo se lo stato di nodeIdx è stato modificato
			se sì *modified_d=true
			(9 e 10 sono eseguiti dal thread con threadIdx=0 di ogni blocco)
*/

template<typename nodeW, typename edgeW>
__global__ void HopfieldNetGPU_k::updateIS_edgewise( float * const state, float * const score,				// Out values
		const GraphStruct<nodeW, edgeW> * const graphStruct_d,									// graph stuff
		const Coloring * const col_d, const int colorIdx,											// coloring stuff
		bool * const modified_d,																	// stop condition
		const float posState, const float negState ) {

	// ID del thread all'interno del nodo, serve per la parallel reduction sum
	unsigned int tid = threadIdx.x;
	// ID del blocco nella griglia, serve per il calcolo dello score edgewise
	unsigned int bid = blockIdx.x;
	// DIM del blocco 1D, serve per il calcolo dello score edgewise
	// serve per segmentare il calcolo/quanti edge associo ad un thread
	// nel caso il numero di vicini del nodo corrente superi blockDim
	unsigned int dim = blockDim.x;		 // "larghezza" thread:

	// color idx supera numero colori?
	if (colorIdx >= col_d->nCol)
		return;

	const int 		offsetCol = col_d->cumulSize[colorIdx];						// offset per il coloring

	// il kernel non deve prendere nodi nodeIdx al di fuori del colore attuale
	if( (offsetCol + bid) >= col_d->cumulSize[colorIdx + 1] )
		return;

	const int 		nodeIdx = col_d->colClass[offsetCol + bid];					// indice del nodo in cui lavoreremo

	//unsigned int uid = IS[ISidx][bid];   // unit ID
	extern __shared__ float smem[];		 // fissata (per ora) al lancio del kernel come 32 * sizeof(float)

	const int 		offsetDeg	= graphStruct_d->cumulDegs[nodeIdx];				// offset per neighs di nodeIdx
	const int 		degree = graphStruct_d->cumulDegs[nodeIdx+1] - offsetDeg;		// per il ciclo

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
		int indx = graphStruct_d->neighs[offsetDeg + i * dim + tid];
		smem[tid] += graphStruct_d->edgeWeights[offsetDeg + i * dim + tid] * state[indx];

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
		// aggiorno state e score
		score[nodeIdx] = smem[0] - graphStruct_d->nodeThresholds[nodeIdx];
		state[nodeIdx] = SIGNTH((smem[0] - graphStruct_d->nodeThresholds[nodeIdx]));

		//controllo se lo stato è stato modificato
		if (state[nodeIdx] != oldState) {
			*modified_d = true;
		}
	}
}





/*
#	#	#	#	#	#	#	#	#	#	#	#	#

			HopfieldNetGPU methods

#	#	#	#	#	#	#	#	#	#	#	#	#
*/

// va lasciato cudaMemset a 0 per hScore?
template<typename nodeW, typename edgeW>
	void HopfieldNetGPU<nodeW, edgeW>::setInitState( const unitVal * const inState, const  unitVal* const inScore ) {
		cudaError_t cuSts;
		for (int i = 0; i < this->hState.size; i++){
			this->hState.state[i] = static_cast<float>(inState[i]);
			//hState.score[i] = static_cast<float>( inScore[i] );
			this->hState.score[i] = 0;
		}
		cuSts = cudaMemcpy( hState_d.state, this->hState.state, hState_d.size * sizeof( unitVal ), cudaMemcpyHostToDevice ); cudaCheck( cuSts );
		//cuSts = cudaMemcpy(hState_d.score, hState.score, hState_d.size * sizeof( unitVal ), cudaMemcpyHostToDevice ); cudaCheck( cuSts );
		cuSts = cudaMemset( hState_d.score, 0, hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts );
	}

// setta tuti gli initial state = inValue
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::setInitState( const unitVal inValue ) {
	cudaError_t cuSts;
	cuSts = cudaMemset( hState_d.state, inValue, hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts );
	cuSts = cudaMemset( hState_d.score, 0, hState_d.size * sizeof( unitVal ) ); cudaCheck( cuSts );
}

// setta a 0 state e score su memoria device
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::clearInitState() {
	cudaError_t cuSts;
	cuSts = cudaMemset(hState_d.state, 0, hState_d.size * sizeof( unitVal )); cudaCheck( cuSts );
	cuSts = cudaMemset(hState_d.score, 0, hState_d.size * sizeof( unitVal )); cudaCheck( cuSts );
}

// GPURandomizer riempie casualmente state e score su memoria device
// serve ancora?
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::setRandomInitState( GPURand * const randomizer ) {
	//randomizer->fillRandom( hState_d.state, hState_d.size );
}

// vecchia versione?
// serve ancora?
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
	cuSts = cudaMemcpy(hState_d.state, this->hState.state, hState_d.size * sizeof( unitVal ), cudaMemcpyHostToDevice ); cudaCheck( cuSts );
}

// ritorna i valori di state e score
// serve perchè hState e hState_d sono campi protected
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::returnVal( double * const inState, double * const inScore ) {
	cudaError_t cuSts;
	cuSts = cudaMemcpy(this->hState.state, hState_d.state, hState_d.size * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts );
	cuSts = cudaMemcpy(this->hState.score, hState_d.score, hState_d.size * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts );
	for (int i = 0; i < hState_d.size; i++) {
		inState[i] = static_cast<double>( this->hState.state[i] );
		inScore[i] = static_cast<double>( this->hState.score[i] );
	}
}



/*
#	#	#	#	#	#	#	#	#	#	#	#	#

			HopfieldNetGPU normalize score

#	#	#	#	#	#	#	#	#	#	#	#	#
*/
/*
template<typename nodeW, typename edgeW>
void HopfieldNetGPU<nodeW, edgeW>::normalizeScore( const GraphStruct<nodeW, edgeW> * const bigGraph, const uint32_t *const reduxToFull ) {
	cudaError_t cuSts;
	int n = graph_d->getStruct()->nNodes;
	int nOrig = bigGraph->nNodes;
	dim3 threadPerBlk( TPB_ACCUMUL, 1, 1 );
	int bPg = (n + 2 * threadPerBlk.x - 1) / (2 * threadPerBlk.x);
	dim3 blocksPerGrd( bPg, 1, 1 );

	unitVal	*	accumulatedScores;
	int		*	indexes;
	unitVal	*	sumOfWghs;
	cuSts = cudaMalloc( (void**)&accumulatedScores, bPg * sizeof( unitVal ) ); cudaCheck( cuSts );
	cuSts = cudaMalloc( (void**)&indexes, n * sizeof( int ) ); cudaCheck( cuSts );
	cuSts = cudaMalloc( (void**)&sumOfWghs, nOrig * sizeof( unitVal ) ); cudaCheck( cuSts );
	std::unique_ptr<unitVal[]> accumulatedScores_h( new unitVal[bPg] );
	std::unique_ptr<int[]>   indexes_h( new int[n] );
	std::unique_ptr<unitVal[]> sumOfWghs_h( new unitVal[nOrig] );

	//HopfieldNetGPU_k::accumulateDegAndScores <<<blocksPerGrd, threadPerBlk>>> ( n, graph->deg, hState_d.score, accumulatedDeg, accumulatedScores );
	// Calcolo somma degli scores
	HopfieldNetGPUCompr_k::accumulateScores <<<blocksPerGrd, threadPerBlk>>> ( n, hState_d.score, accumulatedScores );
	if (cudaGetLastError() != cudaSuccess) { std::cout << "CUDA ERROR: HopfieldNetGPUCompr_k::accumulateScores" << std::endl; abort(); }			// DEBUG

	// Calcolo somma dei pesi per ogni nodo del grafo originale
	for (int j = 0; j < nOrig; j++) {
		sumOfWghs_h[j] = std::accumulate(bigGraph->weight[j], bigGraph->weight[j] + bigGraph->deg[j], 0.0);
	}

	int i = 0;
	for_each( indexes_h.get(), indexes_h.get() + n, [reduxToFull, &i]( int &val ) {val = reduxToFull->at( i++ ); });
	// ora indexes_h contiene la mappa reduxToFull per ogni nodo del grafo unlabelled

	// accumulazione della somma dei pesi dei nodi unlabelled
	float accumulatedWDeg = 0.0;
	for (int j = 0; j < n; j++) {
		accumulatedWDeg += sumOfWghs_h[indexes_h[j]];
	}
	cudaDeviceSynchronize();

	cuSts = cudaMemcpy( accumulatedScores_h.get(), accumulatedScores, bPg * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts );
	cuSts = cudaMemcpy( sumOfWghs, sumOfWghs_h.get(), nOrig * sizeof( unitVal ), cudaMemcpyHostToDevice ); cudaCheck( cuSts );
	cuSts = cudaMemcpy( indexes, indexes_h.get(), n * sizeof( int ), cudaMemcpyHostToDevice ); cudaCheck( cuSts );

	// finisco l'accumulazione di deg e score su CPU
	unitVal totScore = std::accumulate( accumulatedScores_h.get(), accumulatedScores_h.get() + bPg, 0.0 );

	bPg = (n + threadPerBlk.x - 1) / threadPerBlk.x;
	blocksPerGrd = dim3( bPg, 1, 1 );
	HopfieldNetGPUCompr_k::normalizeScores <<<blocksPerGrd, threadPerBlk>>> ( n, accumulatedWDeg, totScore, sumOfWghs, indexes, hState_d.score );
	if (cudaGetLastError() != cudaSuccess) { std::cout << "CUDA ERROR: HopfieldNetGPUCompr_k::normalizeScores" << std::endl; abort(); }			// DEBUG

	cuSts = cudaMemcpy( sumOfWghs_h.get(), sumOfWghs, nOrig * sizeof( unitVal ), cudaMemcpyDeviceToHost ); cudaCheck( cuSts );
	cudaDeviceSynchronize();

	cuSts = cudaFree( sumOfWghs ); cudaCheck( cuSts );
	cuSts = cudaFree( indexes ); cudaCheck( cuSts );
	cuSts = cudaFree( accumulatedScores ); cudaCheck( cuSts );
}
*/

/*
__global__ void HopfieldNetGPU_k::accumulateScores( const int unlab, const unitVal * const scores,
	unitVal * const accumScores ) {

	__shared__ float tempScores[TPB_ACCUMUL];

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= unlab / 2)
		return;
	int baseBlock = 2 * blockDim.x * blockIdx.x;
	tempScores[threadIdx.x] = fabsf(scores[baseBlock + threadIdx.x]);
	__syncthreads();

	int incremento = ((baseBlock + threadIdx.x + blockDim.x) < unlab) ? blockDim.x : (unlab % blockDim.x) / 2;
	tempScores[threadIdx.x] += fabsf(scores[baseBlock + threadIdx.x + incremento]);
	__syncthreads();

	// ahah! Non capisco un cazzo!
#pragma unroll
	for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1) {
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
}*/

/*
__global__ void HopfieldNetGPU_k::normalizeScores( const int unlab, const float accumWDeg, const unitVal accumScores,
	const unitVal * const sumOfWeights, const int * const indexes, unitVal * const scores ) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= unlab)
		return;

	scores[tid] = sumOfWeights[indexes[tid]] / accumWDeg + scores[tid] / accumScores;

	return;
}
*/


//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class HopfieldNetGPU<col, col>;
template class HopfieldNetGPU<float, float>;
