#pragma once

#include "graph/graph.h"
#include "graph_coloring/coloring.h"
#include "GPUutils/GPUutils.h"
#include "GPUutils/GPURandomizer.h"

using namespace std;

#define HI 1
#define LO 0

//ifdef di debug
//#define DEBUGPRINTK_IS
//#define PRINTHOPFIELDTITLE
//#define VERBOSEHOPFIELD

#define TPB_ACCUMUL 128

typedef int unitIdx;      	// net unit index
typedef float unitVal;      // net unit value

struct HopfState {
	int				size;
	unitVal		*	state;
	unitVal		*	score;
};


template<typename nodeW, typename edgeW> class HopfieldNet {
public:
	HopfieldNet( const Graph<nodeW,edgeW> * const inGraph, const Coloring * const inCol, float inPosState, float inNegState, float inRegulWeight );
	virtual ~HopfieldNet() /*= 0*/;
	//virtual void run() = 0;

	int				getNumIter() const;
/*
	HopfState	*	getState() const;
	void			permRunOrder();
	void			permRunColorOrder();
	float			getEnergy();
	void			checkStableState();
	string			isStableState() const;
	void			display();*/

protected:
	const Graph<nodeW, edgeW> 			* const graph;
	const Coloring						* const col;
	HopfState							hState{};
	float								posState{0};
	float								negState{0};

	// modifica per regolarizzazione
	float 								regulWeight{0};

	string								stableState{};
	int							*		unitUpdateOrder{};
	int							*		colUpdateOrder{};		// Uh oh... questo dove viene allocato?
	int									numIter{0};
};


template<typename nodeW, typename edgeW>
class HopfieldNetCPU : public HopfieldNet<nodeW, edgeW> {
public:
	HopfieldNetCPU( const Graph<nodeW,edgeW> * const inGraph, const Coloring * const inCol, float inPosState, float inNegState, float inRegulWeight );
	void run();
	~HopfieldNetCPU();
};

template<typename nodeW, typename edgeW>
class HopfieldNetGPU : public HopfieldNet<nodeW, edgeW> {
public:
	HopfieldNetGPU( const Graph<nodeW, edgeW> * const inGraph_d, const Coloring * const inCol_d, float inPosState, float inNegState, float inRegulWeight );

	~HopfieldNetGPU();

	void 			run_nodewise();
	void 			run_edgewise();

	void 			setInitState( const unitVal * const inState, const unitVal * const inScore );
	void			setInitState( const unitVal inValue );
	void 			clearInitState();
	void 			setInitStateProb( Prob p, char type);
	void 			setRandomInitState( GPURand * const randomizer );
	void 			returnVal( double * const inState, double * const inScore );
	//void			normalizeScore( const GraphStruct<nodeW, edgeW> * const bigGraph, const uint32_t *const reduxToFull );

	const Graph<nodeW, edgeW> 		* const graph_d;
	const Coloring					* const col_d;

protected:
	HopfState		hState_d;

	int				threadId;
	GPUStream   *   GPUstreams;

	int				numThreads;
	dim3			threadsPerBlock;
	dim3			blocksPerGrid;
};

namespace HopfieldNetGPU_k {
	template<typename nodeW, typename edgeW>
	__global__ void updateIS_nodewise( float * const state, float * const score,
		node_sz * cumulDegs, edgeW * edgeWeights, node * neighs_, nodeW * nodeTresholds,
		const node_sz nNodes,
		const uint32_t nCol, const uint32_t	* const colClass, const uint32_t * const cumulSize,
		const int colorIdx,
		bool * const modified_d,
		const float posState, const float negState, const float regulWeight );

	template<typename nodeW, typename edgeW>
	__global__ void updateIS_edgewise( float * const state, float * const score,
		node_sz * cumulDegs, edgeW * edgeWeights, node * neighs_, nodeW * nodeTresholds,
		const node_sz nNodes,
		const uint32_t nCol, const uint32_t	* const colClass, const uint32_t * const cumulSize,
		const int colorIdx,
		bool * const modified_d,
		const float posState, const float negState );
	/*
	__global__ void accumulateScores( const int unlab, const unitVal * const scores,
			unitVal * const accumScores );
	__global__ void normalizeScores( const int unlab, const float accumWDeg, const unitVal accumScores,
			const unitVal * const sumOfWeights, const int * const indexes, unitVal * const scores );*/
}
