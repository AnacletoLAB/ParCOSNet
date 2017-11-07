#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <algorithm>
#include <memory>
#include <vector>
#include <cfloat>
#include <cstring>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "graph/graph.h"
#include "graph_coloring/coloring.h"
#include "hopfield_net/HopfieldNet.h"
#include "GPUutils/GPURandomizer.h"

template<typename nodeW, typename edgeW>
class COSNet {
public:
	COSNet( uint32_t nNodes, GraphStruct<nodeW, edgeW> * graph, GPURand * GPURandGen );
    ~COSNet();
	void setLabels( std::vector<int32_t> &labelsFromFile );
	void prepare();
	void train( uint32_t currentFold );
	void run();
	void deallocateLabs();

	uint32_t						nNodes;
	float						*	scores;
	float						*	states;

	uint32_t						numberOfFolds;
	uint32_t					*	folds;
	uint32_t					*	foldIndex;
	uint32_t					*	numPositiveInFolds;
	uint32_t					*	fullToRedux;
	uint32_t					*	reduxToFull;

	int32_t						*	labels;
	int32_t						*	labelsPurged;
	float						*	newLabels;
	uint32_t 						labelledSize, unlabelledSize;
	uint32_t					*	labelledPositions;   // Allocato da labUnlab
	uint32_t					*	unlabelledPositions; // Allocato da labUnlab

	float						*	threshold;			// Allocato da COSNet::train()

	float						*	pos_neigh;			// Allocato da netP// TODO: rifare per supportare grafi non memorizzati in Unified Memoryrojection
	float						*	neg_neigh;			// Allocato da netProjection

	GraphStruct<nodeW, edgeW>	*	str;

	GPURand						*	GPURandGen;

	float 							eta;
	float							beta;
	float							alpha;
	float							regulWeight;

private:
    // Da COSNet_utils
	void compute_best_c(int size, int tp, int fp, int tn, int fn, int32_t *labels, float *c_values,
			int *order_c_values, float *max_F, float *c_best, float theta_best);
	void error_minimization(float *pos_vect, float *neg_vect, int32_t *labels, uint32_t *n,
        	float *theta, float *c, float *opt_hmean);
	void check_halfplane( const float opt_hmean_over, const float opt_hmean_under, const float theta_best_over,
			const float theta_best_under, int *pos_halfplane, float *max_F, float *theta_best );
	void check_update( float tmp_hmean, float *opt_hmean, float *theta_best, const float angle,
    		int *opt_tp, int *opt_fp, int *opt_fn, const int tp, const int fp, const int fn );
	void quicksort( float a[], int indices[], int l, int r);
	int partition( float a[], int indices[], int l, int r);
	void compute_c(const float * const pos_vect, const float * const neg_vect, int * const order_c_values,
    		float * const c_values, const float theta_best, const int size);
	float compute_F(int tp, int fn, int fp);
	void compute_angles( const float * const pos_vect, const float * const neg_vect, int * const order_thetas,
        	float * const thetas, const int size );
	void update_i_under( const float * const W, const float theta_k, const float pos_state,
    		const float neg_state, const int k_, float * const state, float * const scores, const int N );

	void labUnlab( const int32_t * const labels, const uint32_t size, uint32_t ** labelledPositions,
			uint32_t * labelledSize, uint32_t ** unlabelledPositions, uint32_t * unlabelledSize );
	void netProjection( const node_sz * const cumulDegs, const node * const neigh,
			const nodeW * const wgh, const int32_t * const labels, const uint32_t labelledSize,
			const uint32_t * const labelledPosition, float ** pos_neigh, float ** neg_neigh );
	void extractFolds( const int32_t * const labels, const uint32_t n, const uint32_t numberOfFolds,
			uint32_t * const folds, uint32_t * const foldsIndex, uint32_t * const foldPosNum );

};
