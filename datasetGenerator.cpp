#include <iostream>
#include <fstream>
#include <cinttypes>
#include <memory>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include "graph/graph.h"	// in realta' non serve!

std::vector<std::string> generateRandomName( const int n );

int main( int argc, char ** argv ) {

	uint32_t		nNodes = 500 /* = inserire */;
	uint32_t		nClasses = 10 /* = inserire */;
	std::string labelFileName( "generatedLabels.txt" );
	std::string netFileName( "generatedNet.txt" );
	// OPZIONALE: leggere i parametri precedenti da riga di comando

	float probLabels = 0.05f;
	float probDensity = 0.1f;
	std::default_random_engine eng{ 5 };
	std::uniform_real_distribution<> randR(0.0, 1.0);
	std::normal_distribution<> randNorm(0, 0.1);

	std::ofstream labelFile( labelFileName.c_str(), std::ios::out );
	std::ofstream netFile( netFileName.c_str(), std::ios::out );

	if (!labelFile.is_open()) {
		std::cerr << "errore aperture file etichette" << std::endl;
		abort();
	}

	if (!netFile.is_open()) {
		std::cerr << "errore aperture file rete" << std::endl;
		abort();
	}

	// Richiama generateRandomNames per generare il vettore dei nomi dei nodi
	std::vector<std::string> nodeNames = generateRandomName( nNodes );
	// OPZIONALE: i nomi dei nodi sono stringhe casuali di 24 caratteri
	// quindi la pribabilita' che vengano generati due nodi uguali e' molto bassa.
	// Tuttavia, ad ora non avviene nessun controllo...
	// E' possibile implementare questo check in due modi:
	// - usare la funzione std::find() e scorrere la lista
	// - cambiare la funzione generateRandomName in modo che restituisca un std::set<std::string>
	//   che non puo' contenere elementi uguali

	std::string classBaseName("GO::00");
	// Ciclo for da 0 a nClasses per generare le etichettature
	// Ogni iterazione genera una nuova string formata da classBaseName + numIterazione
	// esempio: "GO::001", poi "GO::002", ecc...
	// Nel file devono essere salvati solo i nomi dei nodi positivi
	for (uint32_t i = 0; i < nClasses; i++) {
		std::string currentClassName( classBaseName + std::to_string( i ) );
		for (uint32_t j = 0; j < nNodes; j++ ) {
			// estrazione di un numero random
			// se estrazione ha successo
			if (randR(eng) < probLabels) {
				labelFile << nodeNames[j] << "\t" << currentClassName << std::endl;
			}
		}
	}

	labelFile.close();

	// Ciclo per generazione della rete
	// Usare stessa strategia di generazione del generatore di grafi di Erdos
	// gia' implementata.
	// Per la scrittura dell'arco generato, usare l'istruzione:
	// netfile << nodeNames[i] << " " << nodeNames[j] << " " << randomWeight << std::endl;
	uint64_t 	*	cumulDegs = new uint64_t[nNodes + 1];
	uint64_t	*	neighs;
	float		*	weights;
	uint64_t		nEdges = 0;

	std::fill(cumulDegs, cumulDegs + nNodes + 1, 0);

	std::vector<uint64_t>	* edges = new std::vector<uint64_t>[nNodes];
	for (uint32_t i = 0; i < nNodes - 1; i++) {
		for (uint32_t j = i + 1; j < nNodes; j++)
			if (randR(eng) < probDensity) {
				edges[i].push_back(j);
				edges[j].push_back(i);
				cumulDegs[i + 1]++;
				cumulDegs[j + 1]++;
				nEdges += 2;
			}
	}
	cumulDegs[0] = 0;
	for (uint32_t i = 0; i < nNodes; i++)
		cumulDegs[i + 1] += cumulDegs[i];

	neighs = new uint64_t[nEdges];
	for (uint32_t i = 0; i < nNodes; i++)
		memcpy((neighs + cumulDegs[i]), edges[i].data(), sizeof(uint64_t) * edges[i].size());

	for (uint32_t i = 0; i < nNodes; i++) {
		for (uint64_t j = cumulDegs[i]; j < cumulDegs[i+1]; j++) {
			netFile << nodeNames[i] << "\t" << nodeNames[neighs[j]] << "\t" << fabs( randNorm( eng ) ) << std::endl;
			//netFile << i << "\t" << neighs[j] << "\t" <<  fabs( randNorm( eng ) ) << std::endl;
		}
	}






	netFile.close();
	return 0;

}


std::vector<std::string> generateRandomName( const int n ) {
	const char alphanum[] =
	        "0123456789"
	        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	        "abcdefghijklmnopqrstuvwxyz";
	std::vector<std::string> out;
	const int slen = 24;	// <- dovrebbe bastare
	char stringa[slen + 1];
	stringa[slen] = 0;

	for (int i = 0; i < n; i++) {
		std::for_each( stringa, stringa + slen, [alphanum](char &c){c = alphanum[rand() % (sizeof(alphanum) - 1)];} );
		out.push_back( std::string( stringa ) );
	}

	return out;
}
