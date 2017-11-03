#include <iostream>
#include <algorithm>
#include <math.h>
#include "HopfieldNet.h"
#include "HopfieldNetUtils.h"
#include "graph/graph.h"
#include "graph_coloring/coloring.h"

using namespace std;

template<typename nodeW, typename edgeW>
HopfieldNet<nodeW, edgeW>::HopfieldNet( const Graph<nodeW, edgeW> * const inGraph, const Coloring * const inCol, float inPosState, float inNegState, float inRegulWeight ):
		graph(inGraph),
		col(inCol),
		posState((float)inPosState),
		negState((float)inNegState),
		regulWeight((float)inRegulWeight) {

	hState.size = inGraph->getStruct()->nNodes;

	//set ordine di aggiornamento (non serve nella versione GPU)
	//riscritto per sicurezza
	/*unitUpdateOrder = new int[inGraph->getStruct()->nNodes];
	for (uint32_t i = 0; i < inGraph->getStruct()->nNodes; i++)
		unitUpdateOrder[i] = i;*/
}

template<typename nodeW, typename edgeW>
HopfieldNet<nodeW, edgeW>::~HopfieldNet() {
	//delete[]	unitUpdateOrder;
};


template<typename nodeW, typename edgeW>
int HopfieldNet<nodeW, edgeW>::getNumIter() const {
	return numIter;
}

template<typename nodeW, typename edgeW>
HopfieldNetCPU<nodeW, edgeW>::HopfieldNetCPU( const Graph<nodeW, edgeW> * const inGraph, const Coloring * const inCol,
	 float inPosState, float inNegState, float inRegulWeight ) :
	 	HopfieldNet<nodeW, edgeW>( inGraph, inCol, inPosState, inNegState, inRegulWeight ) {}

template<typename nodeW, typename edgeW>
HopfieldNetCPU<nodeW, edgeW>::~HopfieldNetCPU() {}

template<typename nodeW, typename edgeW>
void HopfieldNetCPU<nodeW, edgeW>::run() {
	std::cout << " run " << std::endl;
}


//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class HopfieldNet<col, col>;
template class HopfieldNet<float, float>;
template class HopfieldNetCPU<col, col>;
template class HopfieldNetCPU<float, float>;
