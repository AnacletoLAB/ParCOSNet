// COSnet - COSNet utilities class
// Alessandro Petrini - Giuliano Grossi - Marco Frasca, 2017
#pragma once
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cfloat>

#include "graph/graph.h"


// TODO: rifare per supportare grafi non memorizzati in Unified Memory
template<typename nodeW, typename edgeW>
class COSNetCommon {
public:
    COSNetCommon(  uint32_t nNodes, GraphStruct<nodeW, edgeW> * graph );
    ~COSNetCommon();
	void calWeightedDegree();

	uint32_t						nNodes;
	float							sumOfDegree;
	GraphStruct<nodeW, edgeW>	* 	str;
	float						*	weightedDegree;

};
