#include <iostream>
#include <cstdio>
#include <ctime>
#include <thread>
#include <mutex>
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

/*
#ifdef __linux__
extern "C" {
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
}
#endif
*/

// --data C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\string.yeast.v10.5.net.n1.tsv --label C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\yeast.go.ann.CC.6.june.17.stringID.atl5.tsv
// --data C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\test.tsv --label C:\Users\User\Documents\Ricerca\datasets\COSNet\tsv_compressedLabels\testLab.tsv

std::mutex g_labelLock;

template<typename nodeW, typename edgeW>
void doMT(uint32_t thrdNum, uint32_t totThreads, uint32_t N, uint32_t seed, GraphStruct<nodeW,edgeW> * test, fileImporter * fImport );

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
	int device;
	struct cudaDeviceProp properties;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&properties,device);

	fileImporter fImport( graphFileName, labelsFileName );
	fImport.getNumberOfClasses();

	Graph<float, float> test( &fImport, !GPUEnabled );	// Grafo completo DEVE rimanere su CPU
	std::cout << "Nodi: " << test.getStruct()->nNodes << " - Archi: " << test.getStruct()->nEdges << std::endl;

	N = test.getStruct()->nNodes;

	std::cout << "Classe: " << std::flush;

	std::thread *tt = new std::thread[nThrd];
	for (uint32_t i = 0; i < nThrd; ++i) {
		tt[i] = std::thread( doMT<float,float>, i, nThrd, N, seed, test.getStruct(), &fImport );
	}

	std::this_thread::yield();

	for (int i = 0; i < nThrd; ++i)
		tt[i].join();
	delete[] tt;

	return 0;
}

template<typename nodeW, typename edgeW>
void doMT(uint32_t thrdNum, uint32_t totThreads, uint32_t N, uint32_t seed, GraphStruct<nodeW,edgeW> * test, fileImporter * fImport ) {
	for (uint32_t cl = thrdNum; cl < fImport->nOfClasses; cl += totThreads) {

		std::cout << cl << " " << std::flush;

		GPURand curandGen( N, seed + thrdNum );

		// la graphStruct che passiamo a COSNet e' su CPU
		COSNet<float, float> CN( N, test, &curandGen );

		g_labelLock.lock();
			fImport->getNextLabelling();
			CN.setLabels( fImport->labelsFromFile );
		g_labelLock.unlock();

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
}
