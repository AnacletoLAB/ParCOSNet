#pragma once

#include "graph/graph.h"
#include "graph_coloring/coloring.h"
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


//Struttura dove vengono memorizzati state e score della rete

struct HopfState {
	int				size;
	unitVal		*	state;
	unitVal		*	score;
};



/*
#	#	#	#	#	#	#	#	#	#	#	#	#

				HopfieldNet PARAMETERS

#	#	#	#	#	#	#	#	#	#	#	#	#
*/

//tipo generico del grafo impostato a unitVal? va bene?
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


/*
#	#	#	#	#	#	#	#	#	#	#	#	#

				HNetLog PARAMETERS

#	#	#	#	#	#	#	#	#	#	#	#	#
*/
template<typename nodeW, typename edgeW>
class HNetLog {

public:
	void setNetRunTimeCPU(double);
	void setNetRunTimeGPU(double);
	void setColRunTimeCPU(double);
	void setColRunTimeGPU(double);
	//void graphLog(const NetGraph * const);
	void coloringLog( const Colorer<nodeW, edgeW>& );
	void netRunLogCPU( const HopfieldNet<nodeW, edgeW>& );
	void netRunLogGPU( const HopfieldNet<nodeW, edgeW>& );
	void display();

private:
	bool verbose;          // print all (weights & IS cover)

	// graph
	int nUnit;             // num of units (neurons)
	float netDensity;      // link density
	float meanDegree;      // average node degree
	float stdDegree;       // std node degree

	// coloring
	int numColorsCPU;      // num of colors
	int numColorsGPU;      // size of the IS cover by GPU
	float speedup;         // speedup CPU time/GPU time
	float efficiency;      // measure of coloring efficiency

	// H net
	string stableState;    // tell if the final state is stable
	float energyCPU;       // CPU final energy
	float energyGPU;       // GPU final energy

	// GPU memory
	float totalGlobalMem;  // total amount device global memory
	float reqGlobalMem;    // total amount device global memory req.

	// iteration number
	int numIterCPU;           // num of net updating - CPU
	int numIterGPU;           // num of net updating - GPU

	// running times
	float netRunTimeCPU;
	float netRunTimeGPU;
	float colRunTimeCPU;
	float colRunTimeGPU;
};



/*
#	#	#	#	#	#	#	#	#	#	#	#	#

				HopfieldNetGPU PARAMETERS

#	#	#	#	#	#	#	#	#	#	#	#	#
*/
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
	__global__ void updateIS_nodewise( float * const state, float * const score,				// Out values
		const GraphStruct<nodeW, edgeW> * const graphStruct_d,
		//uint32_t * cumulDegs, float * edgeWeights, uint32_t * neighs_, float * nodeTresholds,//
		//const node_sz nNodes,
		const uint32_t nCol, const uint32_t	* const colClass, const uint32_t * const cumulSize,
		const int colorIdx,											// coloring stuff
		bool * const modified_d,																	// stop condition
		const float posState, const float negState, const float regulWeight );												// float const

	template<typename nodeW, typename edgeW>
	__global__ void updateIS_edgewise( float * const state, float * const score,					// Out values
		const GraphStruct<nodeW, edgeW> * const graphStruct_d,										// graph stuff
		const Coloring * const col_d, const int colorIdx,											// coloring stuff
		bool * const modified_d,																	// stop condition
		const float posState, const float negState ) ;												// float const
	/*
	__global__ void accumulateScores( const int unlab, const unitVal * const scores,
			unitVal * const accumScores );
	__global__ void normalizeScores( const int unlab, const float accumWDeg, const unitVal accumScores,
			const unitVal * const sumOfWeights, const int * const indexes, unitVal * const scores );*/
}
