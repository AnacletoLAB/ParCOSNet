#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <set>
#include <cstring>

std::vector<std::string> generateRandomName( const int n );

int main( int argc, char ** argv ) {
	if (argc < 5) {
		std::cout << "  *** ParCOSNet random dataset generator ***" << std::endl;
		std::cout << "by Alessandro Petrini, 2017 - Università degli Studi di Milano - Dept. Computer Science" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "This utility generates a random dataset for evaluating ParCOSNet." << std::endl;
		std::cout << "The network is created as a Erdos graph, given the number of nodes and" << std::endl;
		std::cout << " the probability p of having an edge (i,j), for every i and j," << std::endl;
		std::cout << " 0 <= i < nNodes, 0 <= j < nNodes. For every edge, also the corrisponding" << std::endl;
		std::cout << " weight is generated as a random number [0,1] uniformly distributed." << std::endl;
		std::cout << " The resulting net has no isolated nodes." << std::endl;
		std::cout << "A label file is also created, highly unbalanced towards the majority class." << std::endl;
		std::cout << "" << std::endl;
		std::cout << "Usage: ./datasetGen nNodes prob netFile.tsv labFile.tsv" << std::endl;
		std::cout << "		nNodes:      number of nodes to be generated (positive, integer)" << std::endl;
		std::cout << "		prob:        probability for each edge (0 <= prob < 1, float)" << std::endl;
		std::cout << "      netFile.tsv: outfile name for the generated net" << std::endl;
		std::cout << "      labFile.tsv: outfile name for the generated labels" << std::endl;
		return -1;
	}

	// argc
	// 1: nNodes
	// 2: densita'
	// 3: nome rete
	// 4: nome label
	uint32_t		nNodes = atoi( argv[1] );
	uint32_t		nClasses = 1;
	float			probLabels	= 0.01f;
	float			probDensity	= atof( argv[2] );
	std::string		labelFileName( argv[4] );
	std::string		netFileName( argv[3] );

	std::cout << "nNodes: " << nNodes << " - probDensity: " << probDensity << " - label: "
		<< labelFileName << " - net: " << netFileName << std::endl;

	std::default_random_engine eng( time( NULL ) );
	std::uniform_real_distribution<> randR(0.0, 1.0);
	std::normal_distribution<> randNorm(0, 0.1);

	std::ofstream labelFile( labelFileName.c_str(), std::ios::out );
	std::ofstream netFile( netFileName.c_str(), std::ios::out );

	if (!labelFile.is_open()) {
		std::cerr << "error opening labels file" << std::endl;
		abort();
	}

	if (!netFile.is_open()) {
		std::cerr << "errore opening net file" << std::endl;
		abort();
	}

	// Richiama generateRandomNames per generare il vettore dei nomi dei nodi
	std::vector<std::string> nodeNames = generateRandomName( nNodes );

	std::string classBaseName("GO::00");
	// Ciclo for da 0 a nClasses per generare le etichettature
	// Ogni iterazione genera una nuova string formata da classBaseName + numIterazione
	// esempio: "GO::001", poi "GO::002", ecc...
	// Nel file devono essere salvati solo i nomi dei nodi positivi
	for (uint32_t i = 0; i < nClasses; i++) {
		std::string currentClassName( classBaseName + std::to_string( i ) );
		for (uint32_t j = 0; j < nNodes; j++ ) {
			if (randR(eng) < probLabels) {
				labelFile << nodeNames[j] << "\t" << currentClassName << std::endl;
			}
		}
	}

	labelFile.close();

	// Ciclo per generazione della rete
	uint64_t 	*	cumulDegs = new uint64_t[nNodes + 1];
	uint64_t	*	neighs;
	float		*	weights;
	uint64_t		nEdges = 0;
	uint64_t		nodiIsolatiCorretti = 0;

	std::fill(cumulDegs, cumulDegs + nNodes + 1, 0);
	std::cout << "|--------------------|" << std::endl << "\033[F|";

	std::vector<uint64_t>	* edges = new std::vector<uint64_t>[nNodes];
	for (uint32_t i = 0; i < nNodes - 1; i++) {
		if (i % (nNodes / 20) == 0)
			std::cout << "#" << std::flush;
		bool haiAlmenoUnArco = false;
		for (uint32_t j = i + 1; j < nNodes; j++)
			if (randR(eng) < probDensity) {
				edges[i].push_back(j);
				edges[j].push_back(i);
				cumulDegs[i + 1]++;
				cumulDegs[j + 1]++;
				nEdges += 2;
				haiAlmenoUnArco = true;
			}
		if (!haiAlmenoUnArco) {
			//std::cout << "Nodo isolato!" << std::endl;
			uint32_t aa = (rand() % (nNodes - i)) + 1;
			edges[i].push_back(aa);
			edges[aa].push_back(i);
			cumulDegs[i + 1]++;
			cumulDegs[aa + 1]++;
			nEdges += 2;
			nodiIsolatiCorretti++;
		}
	}
	cumulDegs[0] = 0;
	for (uint32_t i = 0; i < nNodes; i++)
		cumulDegs[i + 1] += cumulDegs[i];

	std::cout << std::endl << "nEdges: " << nEdges <<std::endl;

	neighs = new uint64_t[nEdges];
	for (uint32_t i = 0; i < nNodes; i++)
		memcpy((neighs + cumulDegs[i]), edges[i].data(), sizeof(uint64_t) * edges[i].size());

	std::cout << "Saving..." << std::endl;
	std::cout << "|--------------------|" << std::endl << "\033[F|";

	for (uint32_t i = 0; i < nNodes; i++) {
		if (i % (nNodes / 20) == 0)
			std::cout << "#" << std::flush;
		for (uint64_t j = cumulDegs[i]; j < cumulDegs[i+1]; j++) {
			netFile << nodeNames[i] << "\t" << nodeNames[neighs[j]] << "\t" << randR( eng ) << std::endl;
		}
	}

	std::cout << std::endl;
	std::cout << "Number of correct isolated nodes: " << nodiIsolatiCorretti << std::endl;

	netFile.close();
	return 0;

}


std::vector<std::string> generateRandomName( const int n ) {
	const char alphanum[] =
	        "0123456789"
	        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	        "abcdefghijklmnopqrstuvwxyz";
	std::vector<std::string> out;
	std::set<std::string> tempSet;
	const int slen = 24;
	char stringa[slen + 1];
	stringa[slen] = 0;

	while (tempSet.size() < n) {
		std::for_each( stringa, stringa + slen, [alphanum](char &c){c = alphanum[rand() % (sizeof(alphanum) - 1)];} );
		tempSet.emplace( stringa );
	}

	for ( auto it = tempSet.begin(); it != tempSet.end(); it++ ) {
		out.push_back( std::string( *it ) );
	}

	return out;
}
