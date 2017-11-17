#include <iostream>
#include <algorithm>
#include <math.h>
#include "HopfieldNet.h"
#include "HopfieldNetUtils.h"
#include "graph/graph.h"
#include "graph_coloring/coloring.h"

#define ITERATION_LIMIT 5000

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
	unitUpdateOrder = new int[inGraph->getStruct()->nNodes];
	for (uint32_t i = 0; i < inGraph->getStruct()->nNodes; i++)
		unitUpdateOrder[i] = i;
}

template<typename nodeW, typename edgeW>
HopfieldNet<nodeW, edgeW>::~HopfieldNet() {
	delete[]	unitUpdateOrder;
};


template<typename nodeW, typename edgeW>
int HopfieldNet<nodeW, edgeW>::getNumIter() const {
	return numIter;
}

template<typename nodeW, typename edgeW>
float HopfieldNet<nodeW, edgeW>::getEnergy() {
	float E=0;
	uint32_t offset, degree;

	for (int i = 0; i < graph->getStruct()->nNodes; i++){
		E += graph->getStruct()->nodeThresholds[i] * hState.state[i];

		offset = graph->getStruct()->cumulDegs[i];
		degree = graph->getStruct()->cumulDegs[i+1] - offset;

		for (int j = 0; j < degree; j++){
			if ((hState.state[i] > 0) && (hState.state[graph->getStruct()->neighs[offset + j]] > 0)){
				E -= 0.5 * graph->getStruct()->edgeWeights[offset +j];
			}
		}

	}

	return E;

/*
		float E = 0;
	for (int i = 0; i < graph->nUnit; i++) {
		E += graph->threshold[i] * hState.state[i];
		for (int j = 0; j < graph->deg[i]; j++)
			if (hState.state[i] && hState.state[graph->neigh[i][j]])
				E -= 0.5*graph->weight[i][j];
	}
	return E;*/
}


/*
#	#	#	#	#	#	#	#	#	#	#	#	#

			HopfieldNetCPU CLASS

#	#	#	#	#	#	#	#	#	#	#	#	#
*/

// Costruttore per la rete di Hopfield sequenziale su CPU
template<typename nodeW, typename edgeW>
HopfieldNetCPU<nodeW, edgeW>::HopfieldNetCPU( const Graph<nodeW, edgeW> * const inGraph,
	 float inPosState, float inNegState, float inRegulWeight ) :
	 	HopfieldNet<nodeW, edgeW>( inGraph, nullptr, inPosState, inNegState, inRegulWeight ) {
	this->hState.state = new float[this->hState.size];
	this->hState.score = new float[this->hState.size];
}

template<typename nodeW, typename edgeW>
HopfieldNetCPU<nodeW, edgeW>::~HopfieldNetCPU() {
	delete[] this->hState.score;
	delete[] this->hState.state;
}

template<typename nodeW, typename edgeW>
void HopfieldNetCPU<nodeW, edgeW>::run() {
	//std::cout << " run " << std::endl;

	/*if (!graph->connected) {
		cout << "Warning: graph non connected... EXIT!\n";
		return;
	}*/

	this->numIter = 0;
	bool modified = true;

	// run net until equilibrium
	while (modified) {
		this->numIter++;
		modified = false;

		// ciclo sui nodi
		for (int i = 0; i < this->graph->getStruct()->nNodes; i++) {

			// non ciclo sull'indice ma sull'updateOrder
			int 			uid = this->unitUpdateOrder[i];
			unitVal 		old = this->hState.state[uid];
			float 			sum = - this->graph->getStruct()->nodeThresholds[uid];
			const int		offset = this->graph->getStruct()->cumulDegs[uid];
			const int 		degree = this->graph->getStruct()->cumulDegs[uid + 1] - offset;

			// semplificato: sum += (edgeWeights[offset+j] - regulWeight) * state[neighs]
			for (int j = 0; j < degree; j++){
				sum += (this->graph->getStruct()->edgeWeights[offset+j] - this->regulWeight ) * this->hState.state[ this->graph->getStruct()->neighs[offset+j] ];
			}
			this->hState.state[uid] = (sum < 0 ? this->negState : this->posState);

			// controllo se state[uid] è stato modificato
			if (this->hState.state[uid] != old)
				modified = true;
		}

		// condizione di arresto, limite di iterazioni
		if (this->numIter > ITERATION_LIMIT) {
			std::cout << "Massimo numero di iterazioni raggiunto!!! Uscita forzata" << std::endl;
			break;
		}
	}

	//stableState = "-?-";
}

template<typename nodeW, typename edgeW>
void HopfieldNetCPU<nodeW, edgeW>::clearInitState() {
	std::fill(this->hState.state, this->hState.state + this->hState.size, 0);
	std::fill(this->hState.score, this->hState.score + this->hState.size, 0);
}

template<typename nodeW, typename edgeW>
void HopfieldNetCPU<nodeW, edgeW>::returnVal( float * const inState, float * const inScore ) {
	for (int i = 0; i < this->hState.size; i++) {
		inState[i] = this->hState.state[i];
		inScore[i] = this->hState.score[i];
	}
}


//// Questo serve per mantenere le dechiarazioni e definizioni in classi separate
//// E' necessario aggiungere ogni nuova dichiarazione per ogni nuova classe tipizzata usata nel main
template class HopfieldNet<col, col>;
template class HopfieldNet<float, float>;
template class HopfieldNetCPU<col, col>;
template class HopfieldNetCPU<float, float>;
