#include <string>
#include <iostream>
#include <chrono>
#include "npy.hpp"
#include "parallelParser.h"
#include "serialParser.h"
#include "lib/parseMoves.h"
#include "lib/validate.h"
#include "profiling/profiler.h"
#include "utils/utils.h"
#include "lib/decompress.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"

ABSL_FLAG(std::string, zst, "", ".zst archive to decompress and parse");
ABSL_FLAG(std::string, name, "", "human-readable name for archive");
ABSL_FLAG(std::string, outdir, ".", "output directory to store npy output files");
ABSL_FLAG(bool, serial, false, "Disable parallel processing");
ABSL_FLAG(int, printFreq, 60, "Print status every printFreq seconds");
ABSL_FLAG(int, nReaders, std::thread::hardware_concurrency()-1, "Number of zst/pgn readers for parallel processing");
ABSL_FLAG(int, nMoveProcessors, 1, "Number of game parsers for parallel processing");
ABSL_FLAG(int, minSec, 300, "Minimum time control for game in seconds");
ABSL_FLAG(int, maxSec, 10800, "Maximum time control for game in seconds");
ABSL_FLAG(int, maxInc, 60, "Maximum increment for game in seconds");
ABSL_FLAG(bool, allowNoClock, false, "Allow games with no clock time data to be included");

void writeNpy(std::string outdir, std::shared_ptr<ParserOutput> res) {

	std::ofstream output_file(outdir + "/moves.data");
    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    std::copy(res->mvs.begin(), res->mvs.end(), output_iterator);

	npy::npy_data_ptr<int16_t> welo_ptr;
	npy::npy_data_ptr<int16_t> belo_ptr;
	npy::npy_data_ptr<int64_t> gs_ptr; 
	npy::npy_data_ptr<int16_t> timeCtl_ptr;
	npy::npy_data_ptr<int16_t> inc_ptr;
	
	welo_ptr.data_ptr = res->welos.data(); 	
	welo_ptr.shape = { res->welos.size() };
	belo_ptr.data_ptr = res->belos.data();
	belo_ptr.shape = { res->belos.size() };
	gs_ptr.data_ptr = res->gamestarts.data();
	gs_ptr.shape = { res->gamestarts.size() };
	timeCtl_ptr.data_ptr = res->timeCtl.data();
	timeCtl_ptr.shape = { res->timeCtl.size() };
	inc_ptr.data_ptr = res->increment.data();
	inc_ptr.shape = { res->increment.size() };

	npy::write_npy(outdir + "/welos.npy", welo_ptr);
	npy::write_npy(outdir + "/belos.npy", belo_ptr);
	npy::write_npy(outdir + "/gamestarts.npy", gs_ptr);
	npy::write_npy(outdir + "/timeCtl.npy", timeCtl_ptr);
	npy::write_npy(outdir + "/inc.npy", inc_ptr);
}

int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("Decompress and parse lichess .zst game archives into npy files for use with training mimicChess network");
	absl::ParseCommandLine(argc, argv);
	auto start = std::chrono::high_resolution_clock::now();
	std::shared_ptr<ParserOutput> res;
	std::string name = absl::GetFlag(FLAGS_name);



	if (absl::GetFlag(FLAGS_serial)) {
		res = processSerial(absl::GetFlag(FLAGS_zst));
	} else {
		ParallelParser parser(
				absl::GetFlag(FLAGS_nReaders),
			   	absl::GetFlag(FLAGS_nMoveProcessors),
				absl::GetFlag(FLAGS_minSec),
				absl::GetFlag(FLAGS_maxSec),
				absl::GetFlag(FLAGS_maxInc)
				);
		res = parser.parse(absl::GetFlag(FLAGS_zst), 
				name,
			   	!absl::GetFlag(FLAGS_allowNoClock),
			   	absl::GetFlag(FLAGS_printFreq)
				);
	}
	writeNpy(absl::GetFlag(FLAGS_outdir), res);
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << name << " finished parsing in " << getEllapsedStr(start, stop) << std::endl;
	profiler.report();
	return 0;
}
