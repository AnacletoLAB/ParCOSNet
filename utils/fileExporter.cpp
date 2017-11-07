#include "utils/fileExporter.h"

fileExporter::fileExporter( std::string outFileName, std::string geneFileName, std::map<int, std::string> * inverseGeneMap, uint32_t nNodes ) :
		outFile{ std::ofstream( outFileName.c_str(), std::ios::out ) },
		geneFileName( geneFileName ), outFileName( outFileName ),
		inverseGeneMap { inverseGeneMap }, nNodes{ nNodes } {
}

fileExporter::~fileExporter() {
	outFile.close();
	if (geneFile.is_open())
		geneFile.close();
}

void fileExporter::saveGeneNames() {
	geneFile = std::ofstream( geneFileName.c_str()/*, std::ios::out*/ );

	if (geneFile.is_open()) {
		//std::cout << "Salvo i nomi su un file" << std::endl;

		geneFile << nNodes << std::endl;

		for (uint32_t i = 0; i < nNodes; i++) {
			geneFile << inverseGeneMap->at(i) << std::endl;
		}

		geneFile.close();
	}
}

template <typename nodeW, typename edgeW>
void fileExporter::saveClass( const std::string & currentClassName, COSNet<nodeW, edgeW> * CN ) {
	if (outFile.is_open()) {
		outFile << currentClassName << "\t";

		uint32_t N = CN->nNodes;
		for (uint32_t i = 0; i < N; i++) {
			outFile << CN->states[i] << "\t";
		}

		outFile << std::endl;
	}
}

template void fileExporter::saveClass<float,float>( const std::string & currentClassName, COSNet<float, float> * CN );
