// COSnet - Cuda Randomizer class
// Alessandro Petrini - Giuliano Grossi, 2017
#include "GPUutils.h"
#include "GPURandomizer.h"

#define OFFSET 0

__global__ void GPURand_k::initCurand(curandState* states, unsigned int seed, unsigned int nElem ) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < nElem) {
		curand_init( seed, tid, 0, &states[tid] );
	}
}

/**
 * Samples a discrete distribution
 * @param states curand state
 * @param x the element sampled in the range [0,n-1]
 * @param dist the probability mass function (not normalized)
 * @param n the distribution size
 */
__device__ int GPURand_k::discreteSampling(curandState* states, discreteDistribution dist) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int n = dist->length;
	unsigned int l = 0, r = n-1;
	float u = curand_uniform(&states[tid]);
	float bin = dist->prob[n-1]*u;
	while (l < r) {
		unsigned int m = floorf((l+r)/2);
		if (bin < dist->prob[m])
			r = m;
		else
			l = m+1;
	}
	return r;
}

/**
 * Create a discrete exponential distribution
 * @param dist probabilities
 * @param lambda parameter
 * @param n number of mass points
 */
void CPURand::createExpDistribution(expDiscreteDistribution dist, float lambda, unsigned int nCol) {
	dist->CDF.prob = new float[nCol];
	dist->CDF.length = nCol;
	dist->CDF.normFactor = 0;
	for (unsigned int i = 1; i <= nCol; i++) {
		dist->CDF.prob[i-1] = expf(-lambda*i);
		dist->CDF.normFactor += dist->CDF.prob[i];
	}
}

void CPURand::discreteSampling(discreteDistribution dist, unsigned int* C, unsigned int n, unsigned int seed) {
	unsigned int nCol = dist->length;
	std::default_random_engine eng {seed};
	std::uniform_real_distribution<> randU(0.0, 1.0);

	float* P = new float[nCol];
	P[0] = dist->prob[0];   // cumSum
	for (unsigned i = 1; i < nCol; i++)
		P[i] = P[i-1]+dist->prob[i];

	for (unsigned int i = 0; i < n; i++) {
		unsigned int l = 0;  // , r = nCol-1;
		float u = static_cast<float>( randU(eng) );
		float bin = P[nCol-1]*u;

		while (P[l] < bin) {l++;}


		//		while (l < r) {
//			unsigned int m = floorf((l+r)/2);
//			if (bin < P[m])
//				r = m;
//			else
//				l = m+1;
//		}
		C[i] = l+1;
//		std::cout << C[i] << std::endl;
	}
	delete[] P;
}

GPURand::GPURand( int n, long seed ) : num( n ), seed( seed ) {

    // configuro la griglia e i blocchi
	dim3 threadsPerBlock( 32, 1, 1 );
	dim3 blocksPerGrid( (n + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1 );

    // alloco abbastanza generatori di numeri casuali e li inizializzo
	cuSts = cudaMalloc( (void **)&randStates, n * sizeof( curandState ) ); cudaCheck( cuSts, __FILE__, __LINE__ );
	//GPURand_k::initCurand <<< blocksPerGrid, threadsPerBlock >>> (randStates, 11, n);
    //GPURand_k::initCurand <<< blocksPerGrid, threadsPerBlock >>> (randStates, time(NULL), n);
	GPURand_k::initCurand << < blocksPerGrid, threadsPerBlock >> > (randStates, seed, n);
	cudaDeviceSynchronize();

	// init generatore per chiamate host
	/*curandSts = curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT ); curandCheck( curandSts, __FILE__, __LINE__ );
	curandSts = curandSetPseudoRandomGeneratorSeed( gen, seed ); curandCheck( curandSts, __FILE__, __LINE__ );*/
}

GPURand::~GPURand() {
	cuSts = cudaFree( randStates ); cudaCheck( cuSts, __FILE__, __LINE__ );
	//curandSts = curandDestroyGenerator( gen ); curandCheck( curandSts, __FILE__, __LINE__ );
}
