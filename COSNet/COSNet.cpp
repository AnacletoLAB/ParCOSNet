#include "COSNet/COSNet.h"


template<typename nodeW, typename edgeW>
COSNet<nodeW, edgeW>::COSNet( uint32_t nNodes, GraphStruct<nodeW, edgeW> * graph, GPURand * GPURandGen ) :
		nNodes{ nNodes }, str{ graph }, labelledPositions{ nullptr }, unlabelledPositions{ nullptr },
		pos_neigh{ nullptr }, neg_neigh{ nullptr }, threshold{ nullptr }, GPURandGen{ GPURandGen } {
	scores				= new double[nNodes];
	states				= new double[nNodes];
	labels				= new int32_t[nNodes];

	eta					= 3.5e-4f;
	beta				= eta;

	numberOfFolds		= 5;
	folds				= new uint32_t[nNodes];
	foldIndex			= new uint32_t[numberOfFolds + 1];
	labelsPurged		= new int32_t[nNodes];
	numPositiveInFolds	= new uint32_t[numberOfFolds];
	newLabels			= new float[nNodes];
	fullToRedux			= new uint32_t[nNodes];
	reduxToFull			= new uint32_t[nNodes];
}

template<typename nodeW, typename edgeW>
COSNet<nodeW,edgeW>::~COSNet( ) {
	if (threshold != nullptr)
		delete[] threshold;
	if (neg_neigh != nullptr)
		delete[] neg_neigh;
	if (pos_neigh != nullptr)
		delete[] pos_neigh;
	if (unlabelledPositions != nullptr)
		delete[] unlabelledPositions;
	if (labelledPositions != nullptr)
		delete[] labelledPositions;
	delete[] reduxToFull;
	delete[] fullToRedux;
	delete[] newLabels;
	delete[] numPositiveInFolds;
	delete[] labelsPurged;
	delete[] foldIndex;
	delete[] folds;
	delete[] labels;
	delete[] states;
	delete[] scores;
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW,edgeW>::update_i_under( const float * const W, const float theta_k, const float pos_state,
            const float neg_state, const int k_, float * const state, float * const scores, const int N ) {
	float int_act = 0.0f;
	for (int j = 0; j < N; j++)
		int_act = int_act + (W[k_*N + j] * state[j]);
	scores[k_] = int_act - theta_k;
    state[k_] = (scores[k_] > 0) ? pos_state : neg_state;
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW,edgeW>::compute_angles( const float * const pos_vect, const float * const neg_vect, int * const order_thetas, float * const thetas, const int size ) {
	float m, x1, y1;
	for (int j = 0; j < size; j++) {
		order_thetas[j] = j;
		x1 = pos_vect[j];	y1 = neg_vect[j];
		// excluding lanes parallel to y axes
		if (x1 != 0) {
			m = (float)y1/x1;
			thetas[j] = atan(m);
		}
		else
			//thetas[j] = (float)M_PI/2 - DBL_MIN;
			//thetas[j] = 1.57;
			thetas[j] = (float)1.57 - DBL_MIN;
	}
}

template<typename nodeW, typename edgeW>
inline float COSNet<nodeW,edgeW>::compute_F( const int tp, const int fn, const int fp ) {
	float prec, recall, tmp_F;
    prec = (tp + fp != 0) ? (float)tp / ( tp + fp ) : 0;
    recall = (tp + fn != 0) ? (float)tp / ( tp + fn ) : 0;
    tmp_F = (prec + recall != 0) ? ( 2 * prec * recall ) / ( prec + recall ) : 0;
	return tmp_F;
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW,edgeW>::compute_c( const float * const pos_vect, const float * const neg_vect, int * const order_c_values,
            float * const c_values, const float theta_best, const int size ) {
	float x1, y1;
	for (int j = 0; j < size; j++) {
		order_c_values[j] = j;
		x1 = pos_vect[j];	y1 = neg_vect[j];
		c_values[j] = y1 - tan(theta_best) * x1;
	}
}

// AP: Non possiamo usare la funzione std::sort() ???
template<typename nodeW, typename edgeW>
int COSNet<nodeW,edgeW>::partition( float a[], int indices[], int l, int r ) {
   int i, j;
   float pivot;
   pivot = a[l];
   i = l; j = r+1;
   while(1) {
       do ++i; while( i <= r && a[i] <= pivot );
       do --j; while( a[j] > pivot );
       if( i >= j ) break;
       std::swap(a[i], a[j]);
       std::swap(indices[i], indices[j]);
   }
   std::swap(a[l], a[j]);
   std::swap(indices[l], indices[j]);
   return j;
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW,edgeW>::quicksort( float a[], int indices[], int l, int r ) {
   int j;
   if( l < r ) {
       j = partition( a, indices, l, r);
       quicksort( a, indices, l, j-1);
       quicksort( a, indices, j+1, r);
   }
}

template<typename nodeW, typename edgeW>
inline void COSNet<nodeW,edgeW>::check_update( float tmp_hmean, float *opt_hmean, float *theta_best, const float angle,
	       int *opt_tp, int *opt_fp, int *opt_fn, const int tp, const int fp, const int fn ) {
	if(tmp_hmean > *opt_hmean){
		*opt_hmean = tmp_hmean;
		*theta_best = angle;
		*opt_tp = tp;
		*opt_fn = fn;
		*opt_fp = fp;
	}
}

template<typename nodeW, typename edgeW>
inline void COSNet<nodeW,edgeW>::check_halfplane( const float opt_hmean_over, const float opt_hmean_under, const float theta_best_over,
            const float theta_best_under, int *pos_halfplane, float *max_F, float *theta_best ) {
	if (opt_hmean_over > opt_hmean_under) {
		*pos_halfplane = 1;
		*max_F = opt_hmean_over;
		*theta_best = theta_best_over;
	} else {
		*pos_halfplane = -1;
		*max_F = opt_hmean_under;
		*theta_best = theta_best_under;
	}
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW,edgeW>::compute_best_c( int size, int tp, int fp, int tn, int fn, int32_t *labels, float *c_values,
			int *order_c_values, float *max_F, float *c_best, float theta_best ) {
	int cnt = 0, pos_labels = 0, neg_labels = 0;
	float tmp_hmean_under;
	*max_F = compute_F(tp, fn, fp);
	for (int i = 0; i < size; i++) {
		cnt = 0; pos_labels = 0; neg_labels = 0;
		// counting the number of collinear points
		while (((i + cnt + 1) < size) && (c_values[i] == c_values[i + cnt + 1])) cnt++;
		for (int h = 0; h <= cnt; h++) {
			if(labels[ order_c_values[i + h] ] > 0)
				pos_labels++;
			else
				neg_labels++;
 		}
 		// updating fscore
		tp += pos_labels; fn -= pos_labels; fp += neg_labels; tn -= neg_labels;
 		// compute the F-score relative to the current line when the positive half-plane is that under the line
		tmp_hmean_under = compute_F(tp, fn, fp);
		// check whether current hmean is greater than actual maximum Fscore
		if (tmp_hmean_under > *max_F) {
			*max_F = tmp_hmean_under;
			*c_best = c_values[i];
		}
		i = i + cnt;
	}
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW,edgeW>::error_minimization( float *pos_vect, float *neg_vect, int32_t *labels, uint32_t *n,
           float *theta, float *c, float *opt_hmean ) {
	int N_pos = 0, N_neg, tp_u, fn_u, fp_u, tn_u,
		cnt, pos_labels, neg_labels, opt_fn_u = 0, opt_fp_u = 0, opt_tp_u = 0;
	const int n_ = (const int) (*n);
	float max_F = 0.0f, c_best = 0.0f, theta_best = 0.0f, tmp_hmean_under, opt_hmean_under = 0.0f;

	std::unique_ptr<int[]> order_thetas( new int[n_] );
	std::unique_ptr<int[]> order_c_values( new int[n_] );
	std::unique_ptr<float[]> thetas( new float[n_] );
	std::unique_ptr<float[]> c_values( new float[n_] );

	// finding the number of positive labels
	N_pos = std::count_if( labels, labels + n_, []( int nn ){return nn > 0;} );
	N_neg = n_ - N_pos;

	// initial errors when positive halfplane 'under' the line
	tp_u = 0; fp_u = 0; tn_u = N_neg; fn_u = N_pos; tmp_hmean_under = 0.0f;
	// computing the angles of each line passing through the origin and a point of the training set
	compute_angles(pos_vect, neg_vect, order_thetas.get(), thetas.get(), n_);
	// sorting angles and their indices
	quicksort(thetas.get(), order_thetas.get(), 0, (n_)-1);
	// scanning ordered angles to find the optimum line

    for (int i = 0; i < n_; i++) {
		//if(thetas[i] > 1.5)break;// checking possible out range
		cnt = 0; pos_labels = 0; neg_labels = 0;
		// counting the number of collinear points
		while ((cnt+i < (n_ - 1) && (thetas[i] == thetas[i + cnt + 1]))) cnt++;
		if (i != (n_-1)){
			for (int h = 0; h <= cnt; h++) {
				if(labels[ order_thetas[i + h] ] > 0)
					pos_labels++;
				else
					neg_labels++;
 			}
 		}
 		// updating actual errors
		tp_u += pos_labels; fn_u -= pos_labels;	fp_u += neg_labels; tn_u -= neg_labels;
		tmp_hmean_under = compute_F(tp_u, fn_u, fp_u);
		// check whether current F-scores is greater than actual maximum Fscores
		check_update(tmp_hmean_under, &opt_hmean_under, &theta_best, thetas[i],
						&opt_tp_u, &opt_fp_u, &opt_fn_u, tp_u, fp_u, fn_u);
 		// increment in order to avoid to consider again collinear points
 		i = i + cnt;
	}
    //aggiunta per ovviare a mancanza di regolarizzazione
    if(theta_best > 1.55f)
		theta_best = 1.55f;

// ------- Step 2: computing best intercept---------------
	compute_c(pos_vect, neg_vect, order_c_values.get(), c_values.get(), theta_best, n_);
	// sorting intercepts and their indices
	quicksort(c_values.get(), order_c_values.get(), 0, n_-1);
	tp_u = 0; fp_u = 0; tn_u = N_neg; fn_u = N_pos;
	compute_best_c(*n, tp_u, fp_u, tn_u, fn_u, labels, c_values.get(), order_c_values.get(), &max_F, &c_best, theta_best);

	theta_best = theta_best + FLT_MIN;
	*opt_hmean = max_F;
	*theta = theta_best;

	*c = -c_best * cos(*theta);
}

//////////////
template<typename nodeW, typename edgeW>
void COSNet<nodeW, edgeW>::labUnlab( const int32_t * const labels, const uint32_t size, uint32_t ** labelledPositions,
		uint32_t * labelledSize, uint32_t ** unlabelledPositions, uint32_t * unlabelledSize ) {

	*unlabelledSize = std::count( labels, labels + size, 0 );
	*labelledSize = size - *unlabelledSize;

	*labelledPositions = new uint32_t[*labelledSize];
	*unlabelledPositions = new uint32_t[*unlabelledSize];

	uint32_t labIdx = 0;
	uint32_t unlIdx = 0;
	for (uint32_t i = 0; i < size; i++) {
		if (labels[i] == 0)
			(*unlabelledPositions)[unlIdx++] = i;
		else
			(*labelledPositions)[labIdx++] = i;
	}
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW, edgeW>::netProjection( const node_sz * const cumulDegs, const node * const neigh,
		const nodeW * const wgh, const int32_t * const labels, const uint32_t labelledSize,
		const uint32_t * const labelledPosition, float ** pos_neigh, float ** neg_neigh ) {

	*pos_neigh = new float[labelledSize];
	*neg_neigh = new float[labelledSize];

	for (uint32_t i = 0; i < labelledSize; i++) {
		node_sz nodeIdx = cumulDegs[labelledPosition[i]];
		node_sz grado = cumulDegs[labelledPosition[i] + 1] - nodeIdx;
		const node   * vicinato = &(neigh[nodeIdx]);
		const nodeW  * pesiDeiVicini = &(wgh[nodeIdx]);
		(*pos_neigh)[i] = 0;
		(*neg_neigh)[i] = 0;
		for (uint32_t j = 0; j < grado; j++) {
			if (labels[vicinato[j]] == 1)
				(*pos_neigh)[i] += pesiDeiVicini[j];
			else if (labels[vicinato[j]] == -1)
				(*neg_neigh)[i] += pesiDeiVicini[j];
		}
	}

}

template<typename nodeW, typename edgeW>
void COSNet<nodeW, edgeW>::extractFolds( const int32_t * const labels, const uint32_t n, const uint32_t numberOfFolds,
		uint32_t * const folds, uint32_t * const foldsIndex, uint32_t * const foldPosNum ) {

	for (uint32_t i = 0; i < numberOfFolds; i++)
		foldPosNum[i] = 0;

	// Conta dei label == 1 dentro array labels
	uint32_t positiveLabels = std::count_if( labels, labels + n, []( int32_t nn ){return nn == 1;} );

	if (positiveLabels >= numberOfFolds) {		// Stratificato
		// Conto quanti positivi e quanti negativi
		uint32_t pos = 0, neg = 0;
		std::for_each( labels, labels + n, [&pos, &neg](int32_t e){if (e > 0) pos++; else neg++; } );
		std::unique_ptr<uint32_t[]> posIdx( new uint32_t[pos] );
		std::unique_ptr<uint32_t[]> negIdx( new uint32_t[neg] );
		// Riempio posIdx e negIdx (questo si può fare con un for_each...)
		pos = neg = 0;
		for (uint32_t i = 0; i < n; i++)
			(labels[i] > 0) ? posIdx[pos++] = i : negIdx[neg++] = i;

		// Mescolo gli indici...
		std::random_shuffle( posIdx.get(), posIdx.get() + pos );
		std::random_shuffle( negIdx.get(), negIdx.get() + neg );
		// Stabilisco le dimensioni delle partizioni e i rispettivi indici di partenza nell'array foldsIndex
		// Le dimensioni di tutti i fold sono il più possibile bilanciate
		uint32_t partitionSize = n / numberOfFolds;
		uint32_t partitionRemn = n % numberOfFolds;

		foldsIndex[0] = 0;
		for (uint32_t i = 0; i < numberOfFolds; i++) {
			foldsIndex[i + 1] = foldsIndex[i] + partitionSize;
			if (partitionRemn != 0) {
				foldsIndex[i + 1]++;
				partitionRemn--;
			}
		}

		// Riempio l'array dei fold, cominciando dai positivi
		uint32_t index = 0;
		for (uint32_t i = 0; i < pos; i++) {
			index = foldsIndex[i % numberOfFolds] + i / numberOfFolds;
			folds[index] = posIdx[i];
			foldPosNum[i % numberOfFolds]++;
		}
		// E poi gli butto dentro i negativi
		for (uint32_t i = pos; i < pos + neg; i++) {
			index = foldsIndex[i % numberOfFolds] + i / numberOfFolds;
			folds[index] = negIdx[i - pos];
		}
		// Non perdo tempo ad ordinarli.
		// I primi elementi di ogni fold sono i nodi positivi.
	}
	else {		// Non stratificato
		// Riempio l'array folds con numeri da 0 a n - 1, poi mescolo
		uint32_t ii = 0;
		std::generate( folds, folds + n, [&ii](){return ii++;} );
		std::random_shuffle( folds, folds + n );

		uint32_t partitionSize = n / numberOfFolds;
		uint32_t partitionRemn = n % numberOfFolds;
		foldsIndex[0] = 0;
		for (uint32_t i = 0; i < numberOfFolds; i++) {
			foldsIndex[i + 1] = foldsIndex[i] + partitionSize;
			if (partitionRemn != 0) {
				foldsIndex[i + 1]++;
				partitionRemn--;
			}
		}

		for ( uint32_t i = 0; i < numberOfFolds; i++ ) {
			for ( uint32_t k = foldsIndex[i]; k < foldsIndex[i + 1]; k++) {
				if (labels[k] > 0)
					foldPosNum[i]++;
			}
		}
		// Non serve altro.
		// Anche qui non riordino.
	}
}

///////////
template<typename nodeW, typename edgeW>
void COSNet<nodeW, edgeW>::prepare() {
	extractFolds( labels, nNodes, numberOfFolds, folds, foldIndex, numPositiveInFolds );
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW, edgeW>::train( uint32_t currentFold ) {
	const uint32_t foldSize = foldIndex[currentFold + 1] - foldIndex[currentFold];
	const uint32_t * const foldLabels = &(folds[foldIndex[currentFold]]);

	// Ma questo a che cacchio serviva?!?!?!
	// Ah, ora ricordo: i nodi negativi hanno etichetta "-1", quelli positivi "+1"
	// labelsPurged è un array delle etichette "di lavoro" in cui le etichette del
	// fold corrente vengono messe a "0"; perciò, prima copio l'array labels e poi
	// imposto a 0 le etichette corrispondenti ai nodi del fold
	std::memcpy( labelsPurged, labels, nNodes * sizeof(int32_t) );
	std::for_each( foldLabels, foldLabels + foldSize, [&]( int labId ){ labelsPurged[labId] = 0; } );
	// Di solito alle lambda non vanno passate tutte le variabili per referenza, ma quest volta è necessario
	// affinché implicitamente venga passato "this" per accedere ad un membro della classe

	// Creazione e fill fullToRedux e reduxToFull.
	// Nota: si può fare in modo più furbo, usando l'array dei fold...
	uint32_t j = 0;
	std::fill( fullToRedux, fullToRedux + nNodes, 0 );
	std::fill( reduxToFull, reduxToFull + nNodes, 0 );
	for (uint32_t i = 0; i < nNodes; i++) {
		if (labelsPurged[i] == 0) {
			fullToRedux[i] = j;
			reduxToFull[j] = i;
			j++;
		}
	}

	// Occhio: queste funzioni allocano labelledPositions, unlabelledPositions, pos_neigh e neg_neigh
	labUnlab( labelsPurged, nNodes, &labelledPositions, &labelledSize, &unlabelledPositions, &unlabelledSize );
	netProjection( str->cumulDegs, str->neighs, str->edgeWeights, labelsPurged, labelledSize, labelledPositions, &pos_neigh, &neg_neigh );

	std::unique_ptr<int32_t[]> tempLabs( new int32_t[labelledSize] );
	for (uint32_t i = 0; i < labelledSize; i++)
		tempLabs[i] = labelsPurged[labelledPositions[i]];

	// Qui inizia il training vero e proprio
	alpha = 0.0;
	float c = 0.0;
	float Fscore = -1.0;
	error_minimization( pos_neigh, neg_neigh, tempLabs.get(), &labelledSize, &alpha, &c, &Fscore );

	for (uint32_t i = 0; i < nNodes; i++) {
		newLabels[i] = (labelsPurged[i] == 1) ? sin( alpha ) : ((labelsPurged[i] == -1) ? -cos( alpha ) : 0);
	}

	// modifica per regolarizzazione
	float trainPosProp = numPositiveInFolds[currentFold] / (float)(nNodes - foldSize);
	const float posState = (float)  sin( alpha );
	const float negState = (float) -cos( alpha );
	const float a = 1.0f / (posState - negState);
	const float b = negState / (negState - posState);
	eta = beta * abs( tanf( (alpha - M_PI/4.0f) * 2.0f ) );
	const int h = foldSize;
	regulWeight = 2.0f * eta * a * a;

	threshold = new float[unlabelledSize];
	for (uint32_t i = 0; i < unlabelledSize; i++) {

		node_sz nodeIdx = str->cumulDegs[unlabelledPositions[i]];
		node_sz grado = str->cumulDegs[unlabelledPositions[i] + 1] - nodeIdx;
		const node   * vicinato = &(str->neighs[nodeIdx]);
		const nodeW  * pesiDeiVicini = &(str->edgeWeights[nodeIdx]);
		threshold[i] = c;
		for (uint32_t j = 0; j < grado; j++) {
			threshold[i] -= pesiDeiVicini[j] * newLabels[vicinato[j]];
		}
		threshold[i] += eta * a * ( 2.0f * b * (h - 1.0f) + (1.0f - 2.0f * trainPosProp * h) );

		// e copia dentro al grafo
		//str->nodeThreshold[unlabelledPositions[i]] = threshold[i];
		// NOOOO!!!! Questo deve rimanere locale al thread!
		// Non bisogna modificare il grafo, perché condiviso tra tutti i thread!!!!!
		// Modificare l'array delle soglie, in modo che sia grande nNodes e vengano
		// riempite le posizioni corrispondenti ai nodi non etichettati!
		// La versione precedente di COSNet era corretta, poiché ogni thread manteneva
		// una copia locale del grafo.
	}
}

template<typename nodeW, typename edgeW>
void COSNet<nodeW, edgeW>::run() {

	// Costruttore grafo ridotto SU GPU!
	Graph<nodeW, edgeW> grafoRedux( unlabelledPositions, unlabelledSize, labelsPurged, str, fullToRedux, reduxToFull, threshold, true );

	ColoringLuby<nodeW, edgeW> colLuby( &grafoRedux, GPURandGen->randStates );
	colLuby.run();

	std::unique_ptr<double[]> stateRedux( new double[grafoRedux.getStruct()->nNodes] );
	std::unique_ptr<double[]> scoreRedux( new double[grafoRedux.getStruct()->nNodes] );

	HopfieldNetGPU<nodeW, edgeW> HN_d( &grafoRedux, colLuby.getColoringGPU(), sin( alpha ), -cos( alpha ), regulWeight );
	HN_d.clearInitState();
	HN_d.run_edgewise();
	//HN_d.normalizeScore( str, &reduxToFull );

	//HN_d.returnVal( stateRedux.get(), scoreRedux.get() );
	for (uint32_t i = 0; i < grafoRedux.getStruct()->nNodes; i++) {
		states[reduxToFull[i]] = static_cast<double>(stateRedux[i]);
		scores[reduxToFull[i]] = static_cast<double>(scoreRedux[i]);
	}

}

template<typename nodeW, typename edgeW>
void COSNet<nodeW, edgeW>::setLabels( std::vector<int32_t> &labelsFromFile ) {
	std::memcpy( labels, labelsFromFile.data(), labelsFromFile.size() * sizeof(int32_t) );
	return;
}



//template class COSNet<uint32_t, uint32_t>;
template class COSNet<float, float>;
