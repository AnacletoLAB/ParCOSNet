#include <COSNet/COSNetCommon.h>

template<typename nodeW, typename edgeW>
COSNetCommon<nodeW,edgeW>::COSNetCommon( uint32_t nNodes, GraphStruct<nodeW, edgeW> * graph ) :
		nNodes{ nNodes }, str{ graph } {

	weightedDegree = new double[nNodes];
}

template<typename nodeW, typename edgeW>
COSNetCommon<nodeW,edgeW>::~COSNetCommon( ) {
	delete[] weightedDegree;
}

template<typename nodeW, typename edgeW>
void COSNetCommon<nodeW, edgeW>::calWeightedDegree() {
	for (int j = 0; j < nNodes; j++) {
		weightedDegree[j] = std::accumulate( str->nodeWeights[j], str->nodeWeights[j] + str->deg(j), 0.0 );
	}
	sumOfDegree = std::accumulate( weightedDegree, weightedDegree + nNodes, 0.0 );
}
