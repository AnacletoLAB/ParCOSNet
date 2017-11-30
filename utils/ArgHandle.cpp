// COSnet - Commandline Argument Handler
// Alessandro Petrini, 2017
#include "ArgHandle.h"
#include <getopt.h>

ArgHandle::ArgHandle( int argc, char **argv ) :
		dataFilename( "" ), foldFilename( "" ), labelFilename( "" ), outFilename( "" ), geneOutFilename( "" ),
		statesOutFilename( "" ), foldsOutFilename( "" ), timeOutFilename( "" ),
		m( 0 ), n( 0 ), prob( 0.0 ),
		nFolds( 0 ), seed( 0 ), verboseLevel(0),
		nThreads( 0 ),
		generateRandomFold( false ), simulate( false ), argc( argc ), argv( argv ) {
}

ArgHandle::~ArgHandle() {}

void ArgHandle::processCommandLine() {

	char const *short_options = "d:l:f:m:n:N:o:g:u:j:S:q:t:v:h";
	const struct option long_options[] = {

		{ "data",			required_argument, 0, 'd' },
		{ "labels",			required_argument, 0, 'l' },
		{ "folds",			required_argument, 0, 'f' },
		{ "features",		required_argument, 0, 'm' },
		{ "variables",		required_argument, 0, 'n' },
		{ "nFolds",			required_argument, 0, 'N' },
		{ "out",			required_argument, 0, 'o' },
		{ "geneOut",		required_argument, 0, 'g' },
		{ "foldsOut",		required_argument, 0, 'u' },
		{ "statesOut",		required_argument, 0, 'j' },
		{ "seed",			required_argument, 0, 'S' },
		{ "nThrd",			required_argument, 0, 'q' },
		{ "tttt",			required_argument, 0, 't' },
		{ "verbose-level",	required_argument, 0, 'v' },
		{ "help",			no_argument,	   0, 'h' },
		{ 0, 0, 0, 0 }
	};

	while (1) {
		int option_index = 0;
		int c = getopt_long( argc, argv, short_options, long_options, &option_index );

		if (c == -1) {
			break;
		}

		switch (c) {
		case 's':
			simulate = true;
			try {
				double temp = std::stod( optarg );
				if ((temp < 0) | (temp > 1)) {
					std::cout << "\033[31;1mSimulation: probabilty of positive class must be 0 < prob < 1.\033[0m" << std::endl;
					abort();
				}
				else {
					prob = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mArgument missing: specify the probabilty for positive class.\033[0m" << std::endl;
				abort();
			}
			break;
		case 'd':
			dataFilename = std::string( optarg );
			break;

		case 'l':
			labelFilename = std::string( optarg );
			break;

		case 'f':
			foldFilename = std::string( optarg );
			break;

		case 'o':
			outFilename = std::string( optarg );
			break;

		case 'g':
			geneOutFilename = std::string( optarg );
			break;

		case 'u':
			foldsOutFilename = std::string( optarg );
			break;

		case 'j':
			statesOutFilename = std::string( optarg );
			break;

		case 't':
			timeOutFilename = std::string( optarg );
			break;

		case 'n':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mn must be a positive integer.\033[0m" << std::endl;
					abort();
				}
				else {
					n = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mn must be a positive integer.\033[0m" << std::endl;
				abort();
			}
			break;

		case 'm':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mm must be a positive integer.\033[0m" << std::endl;
					abort();
				}
				else {
					m = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mm must be a positive integer.\033[0m" << std::endl;
				abort();
			}
			break;
		case 'N':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					std::cout << "\033[31;1mnFold argument must be a positive integer.\033[0m" << std::endl;
					abort();
				}
				else {
					nFolds = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mnFold argument must be a positive integer.\033[0m" << std::endl;
				abort();
			}
			break;

		case 'S':
			try {
				int temp = std::stoi( optarg );
				seed = temp;
			}
			catch (...) {
				std::cout << "\033[31;1mseed argument must be integer.\033[0m" << std::endl;
				abort();
			}
			break;

		case 'q':
			try {
				int temp = std::stoi( optarg );
				if (temp < 1) {
					temp = 0;
				}
				else {
					nThreads = temp;
				}
			}
			catch (...) {
				std::cout << "\033[31;1mensThrd argument must be integer.\033[0m" << std::endl;
				abort();
			}
			break;

		case 'v':
			try {
				int temp = std::stoi( optarg );
				verboseLevel = temp;
			}
			catch (...) {
				std::cout << "\033[31;1mverbose-level argument must be integer.\033[0m" << std::endl;
				abort();
			}
			break;

		case 'h':
			displayHelp();
			exit( 0 );
			break;

		default:
			break;
		}
	}

	if (outFilename.empty()) {
		std::cout << "\033[33;1mNo output file name defined. Default used (--out).\033[0m" << std::endl;
		outFilename = std::string( "output.txt" );
	}

	if (simulate && (n == 0)) {
		std::cout << "\033[31;1mSimualtion enabled: specify n (-n).\033[0m" << std::endl;
		abort();
	}

	if (simulate && ((prob < 0) | (prob > 1))) {
		std::cout << "\033[31;1mSimulation: probabilty of positive class must be 0 < prob < 1.\033[0m" << std::endl;
		abort();
	}

	if (!simulate) {
		if (dataFilename.empty()) {
			std::cout << "\033[31;1mMatrix file undefined (--data).\033[0m" << std::endl;
			abort();
		}

		if (labelFilename.empty()) {
			std::cout << "\033[31;1mLabel file undefined (--label).\033[0m" << std::endl;
			abort();
		}

		if (foldFilename.empty()) {
			std::cout << "\033[33;1mNo fold file name defined. Random generation of folds enabled (--fold).\033[0m";
			generateRandomFold = true;
			if (nFolds == 0) {
				std::cout << "\033[33;1m [nFold = 3 as default (--nFolds)]\033[0m";
				nFolds = 3;
			}
			std::cout << std::endl;
		}

		if (!foldFilename.empty() && (nFolds != 0)) {
			std::cout << "\033[33;1mnFolds option ignored (mumble, mumble...).\033[0m" << std::endl;
		}

		if (geneOutFilename.empty()) {
			std::cout << "\033[33;1mNo output gene names file name defined (--gene).\033[0m" << std::endl;
			abort();
		}
	}

	if (simulate & (nFolds == 0)) {
		std::cout << "\033[33;1mNo number of folds specified. Using default setting: 5 (--nFolds).\033[0m" << std::endl;
		nFolds = 3;
	}

	if (seed == 0) {
		seed = (uint32_t) time( NULL );
		std::cout << "\033[33;1mNo seed specified. Generating a random seed: " << seed << " (--seed).\033[0m" << std::endl;
		srand( seed );
	}

	if (nThreads <= 0) {
		std::cout << "\033[33;1mNo threads specified. Executing in single thread mode (--nThrd).\033[0m" << std::endl;
		nThreads = 1;
	}

	if (verboseLevel > 3) {
		std::cout << "\033[33;1mNverbose-level higher than 3.\033[0m" << std::endl;
		verboseLevel = 3;
	}

	if (verboseLevel < 0) {
		std::cout << "\033[33;1mverbose-level lower than 0.\033[0m" << std::endl;
		verboseLevel = 0;
	}
}


void ArgHandle::displayHelp() {
	std::cout << " **** ParCosNET ****" << std::endl;
	std::cout << "A sparse and GPU parallel implementation of COSNet for solving the AFP (automated function prediction) problem."<< std::endl;
	std::cout << std::endl;
	std::cout << "by Alessandro Petrini (1), Marco Notaro (1), Jessica Gliozzo (2), Paolo Perlasca (1), Marco Mesiti (1)," << std::endl;
	std::cout << "Giorgio Valentini (1), Giuliano Grossi (1) and Marco Frasca (1)" << std::endl;
	std::cout << "1: Università degli Studi di Milano - Dept. Computer Science" << std::endl;
	std::cout << "2: Fondazione IRCCS Ca’ Granda - Ospedale Maggiore Policlinico, Università degli Studi di Milano";
	std::cout << "Milano - 2017" << std::endl;
	std::cout << std::endl;

	std::cout << "Usage: " << std::endl;
	std::cout << "    " << argv[0] << " [options]" << std::endl;
	std::cout << std::endl;

	std::cout << "Options:" << std::endl;
	std::cout << "    " << "--help                Print this help." << std::endl;
	std::cout << "    " << "--data file.txt       Network input file." << std::endl;
	std::cout << "    " << "                      Each row in the file is a triplet 'sourceNode destinatioNode edgeWeight' representing" << std::endl;
	std::cout << "    " << "                      an edge of the graph" << std::endl;
	std::cout << "    " << "--label file.txt      Labelling input file in compressed format." << std::endl;
	std::cout << "    " << "                      Each row in the file represents a candidate of the minority class, divided by class," << std::endl;
	std::cout << "    " << "                      i.e. nodeA GO::00012" << std::endl;
	std::cout << "    " << "--out file.txt        Output file." << std::endl;
	std::cout << "    " << "--geneOut file.txt    Gene names output file." << std::endl;
	std::cout << "    " << "--foldsOut file.txt   Optional fold output file." << std::endl;
	std::cout << "    " << "--fstatesOut file.txt Optional node states output file." << std::endl;
	// std::cout << "    " << "--simulate P          Enable simulation of a random data / label / fold set. Positive examples are" << std::endl;
	// std::cout << "    " << "                      generated with probability P (0 < P < 1). -m and -n parameters are required." << std::endl;
	// std::cout << "    " << "-m M                  Number of features to be generated. Enabled only if --simulate option is specified." << std::endl;
	// std::cout << "    " << "-n N                  Number of samples to be generated. Enabled only if --simulate option is specified." << std::endl;
	std::cout << "    " << "--nFolds N            Number of folds for cross validation [default = 5]." << std::endl;
	std::cout << "    " << "--nThrd               Number of CPU threads" << std::endl;
	std::cout << "    " << "--seed N              Seed for the random number generator." << std::endl;
	// std::cout << "    " << "--verbose-level N     Level of verbosity: 0 = silent" << std::endl;
	// std::cout << "    " << "                                         1 = only progress" << std::endl;
	// std::cout << "    " << "                                         2 = rf and progress" << std::endl;
	// std::cout << "    " << "                                         3 = complete" << std::endl;
	std::cout << "    " << "--tttt fileout.txt    Optional file for collecting computation stats" << std::endl;
	std::cout << std::endl << std::endl;
}
