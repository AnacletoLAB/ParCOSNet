#include <iostream>
#include <cstdio>
#include <ctime>
#include "utils/ArgHandle.h"
#include "utils/fileImporter.h"
#include "graph/graph.h"
#include "graph/graphCPU.cpp"
#include "graph/graphGPU.cu"
#include "graph_coloring/coloring.h"
#include "COSNet/COSNet.h"
#include "COSNet/COSNetCommon.h"
#include "GPUutils/GPURandomizer.h"
#include "GPUutils/GPUCudaCleaner.h"
#include <omp.h>

// --data C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\string.yeast.v10.5.net.n1.tsv --label C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\yeast.go.ann.CC.6.june.17.stringID.atl5.tsv
// --data C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\test.tsv --label C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\testLab.tsv


int main(int argc, char *argv[]) {

	CudaCleaner CC;

	ArgHandle commandLine( argc, argv );
	commandLine.processCommandLine();

	uint32_t			N				= commandLine.n;
	//uint32_t			M				= commandLine.m;
	//float				prob			= (float) commandLine.prob;
	uint32_t			seed			= commandLine.seed;
	uint32_t			nThrd			= commandLine.nThreads;
	std::string			graphFileName	= commandLine.dataFilename;
	std::string			labelsFileName	= commandLine.labelFilename;

	bool GPUEnabled = 1;

	fileImporter fImport( graphFileName, labelsFileName );
	fImport.getNumberOfClasses();
	Graph<float, float> test( &fImport, !GPUEnabled );
	std::cout << "Nodi: " << test.getStruct()->nNodes << " - Archi: " << test.getStruct()->nEdges << std::endl;

	N = test.getStruct()->nNodes;

	omp_set_num_threads( nThrd );
	omp_lock_t labelLock;				// omp lock for the accumulation phase
	omp_init_lock( &labelLock );

	GPURand * curandGen = new GPURand[nThrd];
	for (uint32_t i = 0; i < omp_get_max_threads(); i++)
		curandGen[i] = GPURand( N, seed );

	std::cout << "classe: " << std::flush;
	// Ciclo sulle classi contenute nel file...
		#pragma omp parallel for default(none) shared(labelLock, fImport, test, std::cout, N, seed, curandGen)
		for (uint32_t cl = 0; cl < fImport.nOfClasses; cl++) {

			std::cout << cl << " " << std::flush;
			uint32_t thrnum = omp_get_thread_num();

			//GPURand curandGen( N, seed );	// Questo potrei portarlo fuori e creare un array di oggetti...

			COSNet<float, float> CN( N, test.getStruct(), &curandGen[thrnum] );

			// In multithread, questa parte dovrebbe essere messa in sezione critica
			omp_set_lock( &labelLock );
				fImport.getNextLabelling();
				CN.setLabels( fImport.labelsFromFile );
			omp_unset_lock( &labelLock );

			// estrae i fold
			CN.prepare();

			// Ciclo sui fold
			for (uint32_t currentFold = 0; currentFold < CN.numberOfFolds; currentFold++) {
				CN.train( currentFold );
				CN.run();
			}

			// Eventuale salvataggio dei risultati; deve essere messo in sezione critica
			// se volessi salvare tutto su un file
			// {
			//		fExport.save()
			// }

		}



	omp_destroy_lock( &labelLock );

	return 0;
}
