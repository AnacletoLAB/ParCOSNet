#include <iostream>
#include <fstream>
#include "HopfieldNetUtils.h"

void saveGeneNames( std::map<int, std::string> * mappaGeniInv, int n, std::string filename ) {
	std::ofstream fout( filename.c_str() );

	if (fout.is_open()) {
		std::cout << "Salvo i nomi su un file" << std::endl;

		fout << n << std::endl;

		for (int i = 0; i < n; i++) {
			fout << mappaGeniInv->at(i) << std::endl;
		}

		fout.close();
	}
}
