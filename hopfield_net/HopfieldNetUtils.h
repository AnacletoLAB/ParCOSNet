#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define SIGN(x) (x < 0 ? LO : HI)
#define SIGNTH(x) (x < 0 ? negState : posState)
#define SIGNTHLAMBDA(x) (x < 0 ? nS : pS)

// draw a random float in [a, b)
__inline float randf(float a, float b) {
	float r = (float) ((float) rand() / (float) RAND_MAX);
	return r * (b - a) + a;
}

/*
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
*/
